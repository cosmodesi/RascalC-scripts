import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from desilike import setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import DESI
import matplotlib.pyplot as plt
import os
import argparse
import logging
from mpi4py import MPI
comm = MPI.COMM_WORLD

EFFECTIVE_REDSHIFTS = {'BGS_ANY-21.35': {(0.1, 0.4): 0.295}, 'BGS_BRIGHT-21.35': {(0.1, 0.4): 0.295}} # outer dictionary is indexed by tracer names, the inner dictionary is indexed by redshift ranges. The effective redshift values are external at the moment
# SMOOTHINGS = {'BGS_ANY-21.35': 15} # only post-recon

b0 = 1.34
sigmapar, sigmaper = 10., 6.5
# if post: sigmapar, sigmaper = 8., 3.
sigmas = {'sigmas': (2., 2.), 'sigmapar': (sigmapar, 2.), 'sigmaper': (sigmaper, 1.)}


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger("fit_bao")

    parser = argparse.ArgumentParser()
    parser.add_argument("--verspec", type=str, default='loa-v1')
    parser.add_argument("--version", type=str, default='v2')
    parser.add_argument("--confs", type=str, nargs='*', default=['PIP', 'nonKP'])
    parser.add_argument("--regions", type=str, nargs='*', default=['GCcomb', 'NGC', 'SGC'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rerun", action='store_true')
    parser.add_argument("--smin", type=float, default=60)
    parser.add_argument("--smax", type=float, default=150)
    parser.add_argument("--ds", type=float, default=4)
    parser.add_argument("--max_l", type=int, default=2)
    parser.add_argument("--broadband", type=str, default='pcs2')
    parser.add_argument("--cov", choices=['', '_Gaussian'], default='')
    

    args = parser.parse_args()
    
    ells = np.arange(0, args.max_l + 1, 2)
    smin, smax, ds = args.smin, args.smax, args.ds
    fiducial = DESI()
    for conf in args.confs:
        cov_dir = f"cov_txt/{args.verspec}/{args.version}/{conf}"
        xi_dir = f"{args.verspec}/LSScats/{args.version}/{conf}/xi/smu"
        output_basedir = f"{args.verspec}/LSScats/{args.version}/{conf}/fits"
        for tracer, redshifts in EFFECTIVE_REDSHIFTS.items():
            for (zmin, zmax), zeff in redshifts.items():
                for region in args.regions:
                    output_dir = f"{output_basedir}/desilike_bao_{tracer}_{region}_z{zmin:.1f}-{zmax:.1f}_xi_{smin}-{smax}_lin{ds}_max_l{args.max_l}_{args.broadband}{args.cov}/"
                    output_fn = f"{output_dir}/minuit_prof"
                    if os.path.isfile(output_fn + ".npy") and not args.rerun:
                        logger.info(f"Skipping fit for {conf} {tracer} z={zmin}-{zmax} {region} since {output_fn}.npy already exists (use --rerun to force)")
                        continue
                    logger.info(f"Starting fit for {conf} {tracer} z={zmin}-{zmax} {region}")

                    xi_fn = f"{xi_dir}/allcounts_{tracer}_{region}_{zmin}_{zmax}_default_FKP_lin_njack60_nran1_split20.npy"
                    cov_fn = f"{cov_dir}/xi024_{tracer}_{region}_z{zmin}-{zmax}_default_FKP_lin{ds}_s20-200_cov_RascalC{args.cov}.txt"
                    
                    logger.info(f"Using covariance saved in {cov_fn}")
                    if not os.path.isfile(cov_fn):
                        logger.warning(f"{cov_fn} not found, skipping fit for {conf} {tracer} z={zmin}-{zmax} {region}")
                        continue
                    cov_rc = np.loadtxt(cov_fn)
                    # cut the covariance
                    cov_ells = (0, 2, 4)
                    cov_s = np.arange(20 + ds/2, 200, ds)
                    cov_rc_l = np.repeat(cov_ells, len(cov_s))
                    cov_rc_s = np.tile(cov_s, len(cov_ells))
                    cov_rc_cut_1d = np.logical_and(cov_rc_l <= args.max_l, np.logical_and(cov_rc_s >= args.smin, cov_rc_s <= args.smax))
                    cov_rc_cut_2d = np.ix_(cov_rc_cut_1d, cov_rc_cut_1d)
                    covariance = cov_rc[cov_rc_cut_2d]
                        
                    template = BAOPowerSpectrumTemplate(z = zeff, fiducial = 'DESI', apmode = 'qiso' + 'qap' * (args.max_l > 0))
                    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, ells=list(ells), broadband=args.broadband) # pre-recon, for post-recon need mode="recsym" or "reciso"

                    # update the parameters to match the fiducial settings for DESI DR2 BAO from desi-y3-kp
                    for param in theory.init.params.select(basename=list(sigmas)):
                        value = sigmas[param.basename]
                        if value is None:
                            kw = {'prior': {'limits': [0., 20.]}, 'fixed': False}
                        elif isinstance(value, (tuple, list)):
                            loc, scale = value
                            kw = {'value': loc, 'prior': {'dist': 'norm', 'loc': loc, 'scale': scale, 'limits': [0., 20.]}, 'fixed': False}
                        else:
                            # print(param.basename, value)
                            kw = {'value': value, 'prior': None, 'fixed': True}
                        param.update(**kw)

                    b1 = b0 / fiducial.growth_factor(zeff)

                    if 'b1p' in theory.init.params:  # physical
                        b1p = b1 * fiducial.sigma8_z(zeff)
                        theory.init.params['b1p'].update(value=b1p, ref=dict(dist='norm', loc=b1p, scale=0.1))
                    else:
                        theory.init.params['b1'].update(value=b1, ref=dict(dist='norm', loc=b1, scale=0.1))
                    
                    observable = TracerCorrelationFunctionMultipolesObservable(data = TwoPointCorrelationFunction.load(xi_fn)[::ds], covariance=covariance, slim={ell: [smin, smax, ds] for ell in ells}, theory=theory, wmatrix={'resolution': 1})
                    likelihood = ObservablesGaussianLikelihood(observables=[observable])
                    for param in likelihood.all_params.select(basename=['alpha*', 'sn*']):
                        param.update(derived='.auto')
                        
                    
                    if os.path.isfile(output_fn + ".npy"): os.remove(output_fn + ".npy")
                    os.makedirs(output_dir, exist_ok = True)
                    profiler = MinuitProfiler(likelihood, save_fn=output_fn + ".npy", seed=args.seed, mpicomm=comm)
                    profiles = profiler.maximize(niterations=50)
                    # To print relevant information
                    print(profiles.to_stats(tablefmt='pretty', fn=output_fn + ".txt"))
                    profiles.save(output_fn+".npy")
                    # additional formats
                    profiles.to_stats(tablefmt='latex_raw', fn=output_fn + ".tex")
                    profiles.to_stats(tablefmt='tsv', fn=output_fn + ".tsv")

                    likelihood(**profiler.profiles.bestfit.choice(input=True))
                    observable.plot()
                    plt.gcf()
                    plt.title(f"{tracer} {region} z={zmin:.1f}-{zmax:.1f} {args.broadband}" + (args.cov.lstrip('_') + " cov") * bool(args.cov))
                    plt.savefig(f"{output_dir}/data_bestfit.png")
            