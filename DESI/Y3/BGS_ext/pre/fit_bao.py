import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from desilike import setup_logging
from pycorr import TwoPointCorrelationFunction
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD

EFFECTIVE_REDSHIFTS = {'BGS_ANY-21.35': {(0.1, 0.4): 0.295}} # outer dictionary is indexed by tracer names, the inner dictionary is indexed by redshift ranges. The effective redshift values are external at the moment
# SMOOTHINGS = {'BGS_ANY-21.35': 15} # only post-recon

CAPS = ["GCcomb", "NGC", "SGC"]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verspec", type = str, default = 'loa-v1')
    parser.add_argument("--version", type = str, default = 'v2')
    parser.add_argument("--conf", type = str, default = 'PIP')
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--rerun", action = 'store_true')
    parser.add_argument("--smin", type = float, default = 48)
    parser.add_argument("--smax", type = float, default = 152)
    parser.add_argument("--ds", type = float, default = 4)
    parser.add_argument("--max_l", type = int, default = 2)
    parser.add_argument("--broadband", type = str, default = 'pcs2')
    

    args = parser.parse_args()
    
    ells = np.arange(0, args.max_l + 1, 2)
    smin, smax, ds = args.smin, args.smax, args.ds
    cov_dir = f"cov_txt/{args.verspec}/{args.version}/{args.conf}"
    xi_dir = f"{args.verspec}/LSScats/{args.version}/{args.conf}/xi/smu"
    output_basedir = f"{args.verspec}/LSScats/{args.version}/{args.conf}/fits"
    for tracer, redshifts in EFFECTIVE_REDSHIFTS.items():
        for (zmin, zmax), zeff in redshifts.items():
            for cap in CAPS:
                print(f"Starting fit for {tracer} z={zmin}-{zmax} {cap}", flush = True)

                xi_fn = f"{xi_dir}/allcounts_{tracer}_{cap}_{zmin}_{zmax}_default_FKP_lin_njack60_nran1_split20.npy"
                cov_fn = f"{cov_dir}/xi024_{tracer}_{cap}_z{zmin}-{zmax}_default_FKP_lin{ds}_s20-200_cov_RascalC.txt"
                
                print(f"Using covariance saved in {cov_fn}", flush = True)
                if not os.path.isfile(cov_fn):
                    print(f"{cov_fn} not found!")
                    continue
                cov_rc = np.loadtxt(cov_fn)
                # cut the covariance
                cov_rc_l = np.concatenate([np.tile(ell, 45) for ell in (0, 2, 4)])
                cov_rc_s = np.concatenate([np.arange(20, 200, 4) + 2 for ell in (0, 2, 4)])
                cov_rc_cut_1d = np.logical_and(cov_rc_l <= args.max_l, np.logical_and(cov_rc_s >= args.smin, cov_rc_s <= args.smax))
                cov_rc_cut_2d = np.ix_(cov_rc_cut_1d, cov_rc_cut_1d)
                covariance = cov_rc[cov_rc_cut_2d]
                
                
                
                output_dir = f"{output_basedir}/desilike_bao_{tracer}_{cap}_z{zmin:.1f}-{zmax:.1f}_xi_{smin}-{smax}_lin{ds}_max_l{args.max_l}_{args.broadband}/"
                os.makedirs(output_dir, exist_ok = True)
                    
                setup_logging()
                    
                template = BAOPowerSpectrumTemplate(z = zeff, fiducial = 'DESI', apmode = 'qiso' + 'qap' * (args.max_l > 0))
                theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, ells=list(ells), broadband=args.broadband) # pre-recon, for post-recon need mode="recsym" or "reciso"
                
                output_fn = f"{output_dir}/minuit_prof"
                if os.path.isfile(output_fn + ".npy") and not args.rerun: continue
                observable = TracerCorrelationFunctionMultipolesObservable(data = TwoPointCorrelationFunction.load(xi_fn)[smin:smax:ds],  
                                                                    covariance = covariance,
                                                                    slim = {ell: [smin, smax, ds] for ell in ells},
                                                                    theory = theory,
                                                                    wmatrix = {'resolution': 1})
                likelihood = ObservablesGaussianLikelihood(observables=[observable])
                for param in likelihood.all_params.select(basename=['alpha*', 'sn*']):
                    param.update(derived='.auto')
                    
                
                if os.path.isfile(output_fn + ".npy"): os.remove(output_fn + ".npy")
                profiler = MinuitProfiler(likelihood, save_fn = output_fn + ".npy", seed = args.seed, mpicomm = comm)
                profiles = profiler.maximize(niterations=50)
                # To print relevant information
                print(profiles.to_stats(tablefmt='pretty', fn = output_fn + ".txt"))
                profiles.save(output_fn + ".npy")

                likelihood(**profiler.profiles.bestfit.choice(input=True))
                observable.plot()
                plt.gcf()
                plt.savefig(f"{output_dir}/data_bestfit.png")
            