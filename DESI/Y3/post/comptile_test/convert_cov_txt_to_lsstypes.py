"Export RascalC text covariance matrices to lsstypes & desi-clustering format. Adapted from desi-clustering:clustering_statistics/tools/convert_rascalc_txt.py originally written by Qinxun Li."

from pathlib import Path

import numpy as np
from lsstypes import CovarianceMatrix, Count2CorrelationPole, Count2CorrelationPoles, ObservableTree

from clustering_statistics import tools

version = 'data-dr2-v2-comptile'
txt_dir = Path(f'cov_txt/{version}/recon_sm15_IFFT_recsym')

# Binning of the text files: nbin = 45, rmax = 180, skip_r_bins = 5 => 40 bins of 4 Mpc/h from 20.
srange, sstep = (20., 180.), 4.
ells = (0, 2, 4)


def read_covariance_txt(txt_fn, srange=srange, sstep=sstep, ells=ells):
    """Read a RascalC Legendre covariance text file into a :class:`CovarianceMatrix`."""
    value = np.loadtxt(txt_fn)
    edges = np.arange(srange[0], srange[1] + sstep / 2., sstep)
    nbins = len(edges) - 1
    if value.shape != (nbins * len(ells),) * 2:
        raise ValueError(f'{txt_fn}: expected a {nbins * len(ells)}^2 matrix for {nbins} s-bins '
                         f'and ells {ells}, got {value.shape}')
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{txt_fn}: covariance contains non-finite entries')
    if not np.allclose(value, value.T, rtol=1e-10, atol=0.):
        raise ValueError(f'{txt_fn}: covariance is not symmetric')
    s_edges = np.column_stack([edges[:-1], edges[1:]])
    s = np.mean(s_edges, axis=-1)
    # Only the binning matters for matching the covariance to the data vector; the correlation
    # function values are unknown here, and are NaN in the DR2 products as well.
    poles = Count2CorrelationPoles([Count2CorrelationPole(s=s, s_edges=s_edges, value=np.full(nbins, np.nan), ell=ell)
                                    for ell in ells])
    return CovarianceMatrix(value=value, observable=poles)


def convert_covariance(txt_fn, dr_fn, tracer):
    """Convert one RascalC text covariance to the lsstypes file expected by the fits."""
    covariance = read_covariance_txt(txt_fn)
    covariance = covariance.clone(observable=ObservableTree([covariance.observable], observables=['correlation2'],
                                                           tracers=[(tools.get_simple_tracer(tracer),) * 2]))
    Path(dr_fn).parent.mkdir(parents=True, exist_ok=True)
    covariance.write(dr_fn)
    return dr_fn


if __name__ == '__main__':

    list_zrange = [('BGS_BRIGHT-21.35', (0.1, 0.4))]
    for tracer, zrange in list_zrange:
        for region in ['GCcomb', 'NGC', 'SGC']:
            txt_fn = txt_dir / (f'xi{"".join(map(str, ells))}_{tracer}_{region}_z{zrange[0]}-{zrange[1]}'
                                f'_default_FKP_lin{sstep:.0f}_s{srange[0]:.0f}-{srange[1]:.0f}_cov_RascalC.txt')
            kw = dict(stats_dir='.', project='bao/rascalc', weight='default-FKP', version=version,
                      tracer=tracer, region=region, zrange=zrange)
            covariance_fn = tools.get_stats_fn(kind='covariance_recon_particle2_correlation', **kw)
            convert_covariance(txt_fn, covariance_fn, tracer=tracer)
            print(f'{txt_fn} -> {covariance_fn}')
