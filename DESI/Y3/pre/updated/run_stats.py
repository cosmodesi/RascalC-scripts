"""
Script to run a large batch of clustering measurements on an interactive GPU node.
To run on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python run_stats.py
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import setup_logging

from clustering_statistics import tools

setup_logging()

def run_stats(tracer='LRG', project='', version='holi-v3-altmtl', onthefly=None, stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['NGC','SGC'], ibatch=None, postprocess=None, zranges=None, do_jackknife=False, **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options, postprocess_stats_from_options
    setup_logging()

    cache = {}
    if zranges is None:
        raise ValueError('Please provide zranges.')
    for region in regions:
        mesh2_spectrum = {'cut': True if 'shape' in analysis else None, 
                            'auw': True if 'altmtl' in version and onthefly is None and 'shape' in analysis else None}
        window_mesh2_spectrum = {'cut': True if 'shape' in analysis else None}
        
        options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight), 
                        mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum,
                        particle2_correlation={'jackknife': {'nsplits': 60}} if do_jackknife else {},
                        window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
        options = fill_fiducial_options(options, analysis=analysis)
        
        for itracer in options['catalog']:
            options['catalog'][itracer]['zranges'] = zranges # override fiducial zranges 
            options['catalog'][itracer]['expand']  = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=itracer, nran=options['catalog'][itracer]['nran']), 'from_data': ['Z', 'WEIGHT_SYS', 'FRAC_TLOBS_TILES']}
            if onthefly == 'complete':
                options['catalog'][itracer]['complete'] = {}
            elif onthefly == 'reshuffle':
                options['catalog'][itracer]['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', **(options['catalog'][itracer] | dict(region='ALL')))}                
        
        get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project, extra=onthefly if onthefly else '')
        compute_stats_from_options(stats, analysis=analysis, get_stats_fn=get_stats_fn, cache=cache, **options)

    # postprocess
    if postprocess:
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight), 
                                   combine_regions={'stats': stats}, mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum)
        postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **postprocess_options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='holi-v3-altmtl', onthefly=None, stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', postprocess=['combine_regions'], zranges=None, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight), combine_regions={'stats': stats}, mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
    stats_dir_kws = dict(stats_dir=stats_dir, project=project)
    if onthefly == 'complete':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
    elif onthefly == 'reshuffle':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
    else:
        get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)

    postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **options)



if __name__ == '__main__':

    stats, postprocess = [], []
    version  = 'data-dr2-v2'
    check_for_existing_measurements = False
    
    # stats_dir  = tools.base_stats_dir
    stats_dir = Path(".")

    # run fiducial full_shape
    stats = ['particle2_correlation']
    do_jackknife = True
    postprocess = ['combine_regions']
    analysis = 'protected'
    project  = f'{analysis}/base'
    weight   = 'default-FKP'
    regions  = ['NGC','SGC']
    tracers  = ['LRG', 'ELG_LOPnotqso', 'QSO']

    # onthefly = 'reshuffle'
    # onthefly = 'complete'
    onthefly = None
    
    for tracer in tracers:
        if 'png' in analysis:
            # do not compute measurements for overlapping redshifts
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
        else:
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
            
        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, onthefly=onthefly, zranges=zranges, regions=regions, weight=weight, do_jackknife=do_jackknife, postprocess=postprocess)
        run_stats(**run_stats_kws)