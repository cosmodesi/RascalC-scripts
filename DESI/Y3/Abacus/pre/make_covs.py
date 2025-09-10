# This script generates all covs

import os
from datetime import datetime
import pickle
import hashlib
from typing import Callable
import traceback
import desi_y3_files.file_manager as desi_y3_file_manager
from RascalC.raw_covariance_matrices import cat_raw_covariance_matrices, collect_raw_covariance_matrices
from RascalC.post_process.legendre import post_process_legendre
from RascalC.post_process.legendre_mix_jackknife import post_process_legendre_mix_jackknife
from RascalC.pycorr_utils.sample_cov_multipoles import sample_cov_multipoles_from_pycorr_files
from RascalC.post_process.legendre_mocks import post_process_legendre_mocks
from RascalC.convergence_check_extra import convergence_check_extra
from RascalC.cov_utils import export_cov_legendre
from RascalC.comb.combine_covs_legendre import combine_covs_legendre

max_l = 4
nbin = 50 # radial bins for output cov
rmax = 200 # maximum output cov radius in Mpc/h

jackknife = 1
njack = 60 if jackknife else 0
if jackknife: mbin = 100

skip_r_bins = 5
skip_l = 0

r_step = rmax // nbin
rmin_real = r_step * skip_r_bins

xilabel = "".join([str(i) for i in range(0, max_l+1, 2)])

make_mock_cov = 1
mock_post_processing = 1 # this is about mock post-processing, i.e. fitting RascalC cov to the mock sample cov and not jackknife

# Settings for filenames
fa = "altmtl" # fiber assignment method
mock_id = 0 # mock number, starting from 0
version_main = "v4_1"
version_alt = "v2" # for BGS, further versions not available (yet)

# Set DESI CFS before creating the file manager
os.environ["DESICFS"] = "/dvs_ro/cfs/cdirs/desi" # read-only path

fm = desi_y3_file_manager.get_abacus_file_manager()

regs = ('SGC', 'NGC') # regions for filenames
reg_comb = "GCcomb"

tracers = ['LRG'] * 3 + ['ELG_LOPnotqso'] * 2 + ['LRG+ELG_LOPnotqso', 'BGS_BRIGHT-21.5'] + ['BGS_BRIGHT-21.35'] * 2 + ['BGS_BRIGHT-20.2'] * 2 + ['QSO', 'BGS_ANY-02']
zs = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1), (0.8, 1.1), (1.1, 1.6), (0.8, 1.1), (0.1, 0.4), (0.1, 0.4), (0.25, 0.4), (0.1, 0.25), (0.1, 0.4), (0.8, 2.1), (0.1, 0.4)]

hash_dict_file = "make_covs.hash_dict.pkl"
if os.path.isfile(hash_dict_file):
    # Load hash dictionary from file
    with open(hash_dict_file, "rb") as f:
        hash_dict = pickle.load(f)
else:
    # Initialize hash dictionary as empty
    hash_dict = {}
# Hash dict keys are goal filenames, the elements are also dictionaries with dependencies/sources filenames as keys

# Set up logging
logfile = "make_covs.log.txt"

def print_and_log(s: object = "", max_length: int = 1000) -> None:
    if max_length:
        s_length = len(str(s))
        if s_length > max_length:
            s = str(s)[:max_length] + f" ... ({s_length - max_length} additional characters not shown)"
    print(s)
    print_log(s)
print_log = lambda l: os.system(f"echo \"{l}\" >> {logfile}")

print_and_log(datetime.now())
print_and_log(f"Executing {__file__}")

def my_make(goal: str, deps: list[str], recipe: Callable, force: bool = False, verbose: bool = False) -> None:
    need_make, current_dep_hashes = hash_check(goal, deps, force=force, verbose=verbose)
    if need_make:
        print_and_log(f"Making {goal} from {deps}")
        try:
            # make sure the directory exists
            goal_dir = os.path.dirname(goal)
            if goal_dir: os.makedirs(goal_dir, exist_ok = True) # creating empty directory throws an error
            # now can actually run
            recipe()
        except Exception as e:
            traceback.print_exc()
            print_and_log(f"{goal} not built: {e}")
            return
        hash_dict[goal] = current_dep_hashes # update the dependency hashes only if the make was successfully performed
        print_and_log()

def hash_check(goal: str, srcs: list[str], force: bool = False, verbose: bool = False) -> tuple[bool, dict]:
    # First output indicates whether we need to/should execute the recipe to make goal from srcs
    # Also returns the src hashes in the dictionary current_src_hashes
    current_src_hashes = {}
    for src in srcs:
        if not os.path.exists(src):
            if verbose: print_and_log(f"Can not make {goal} from {srcs}: {src} missing\n") # and next operations can be omitted
            return False, current_src_hashes
        current_src_hashes[src] = sha256sum(src)
    if not os.path.exists(goal) or force: return True, current_src_hashes # need to make if goal is missing or we are forcing, but hashes needed to be collected beforehand, also ensuring the existence of sources
    try:
        if set(current_src_hashes.values()) == set(hash_dict[goal].values()): # comparing to hashes of sources used to build the goal last, regardless of order and names. Collisions seem unlikely
            if verbose: print_and_log(f"{goal} uses the same {srcs} as previously, no need to make\n")
            return False, current_src_hashes
    except KeyError: pass # if hash dict is empty need to make, just proceed
    return True, current_src_hashes

def sha256sum(filename: str, buffer_size: int = 128*1024) -> str: # from https://stackoverflow.com/a/44873382
    h = hashlib.sha256()
    b = bytearray(buffer_size)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

# Make steps for making covs
for tracer, z_range in zip(tracers, zs):
    tlabels = [tracer]
    z_min, z_max = z_range
    version = version_alt if tracer.startswith("BGS") else version_main # different version for BGS
    reg_results = []
    # get options automatically
    xi_setup = desi_y3_file_manager.get_baseline_2pt_setup(tlabels[0], z_range)
    xi_setup.update({"version": version, "fa": fa, "tracer": tracer, "zrange": z_range, "cut": None, "njack": 0}) # specify regions, version, z range and no cut; no need for jackknives
    if jackknife: reg_results_jack = []
    if mock_post_processing: reg_results_mocks = []
    cov_dir = f"cov_txt/{version}/{fa}"
    for reg in regs:
        if make_mock_cov or mock_post_processing:
            # set the mock covariance matrix filename
            mock_cov_name = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_{z_min}_{z_max}_default_FKP_lin{r_step}_cov_sample.txt"
        
        if make_mock_cov:
            # Make the mock sample covariance matrix
            this_reg_pycorr_filenames = [f.filepath for f in fm.select(id = 'correlation_abacus_y3', region = reg, **xi_setup)]
            if len(this_reg_pycorr_filenames) > 0: # only if any files found
                my_make(mock_cov_name, [], lambda: sample_cov_multipoles_from_pycorr_files([this_reg_pycorr_filenames], mock_cov_name, max_l = max_l, r_step = r_step, r_max = rmax)) # empty dependencies should result in making this only if the destination file is missing; checking hashes of 1000 mock pycorr files has been taking long

        outdir = os.path.join(f"outdirs", version, fa, "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}") # output file directory
        if not os.path.isdir(outdir): # try to find the dirs with suffixes and concatenate samples from them
            outdirs_w_suffixes = [outdir + "_" + str(i) for i in range(11)] # append suffixes
            outdirs_w_suffixes = [dirname for dirname in outdirs_w_suffixes if os.path.isdir(dirname)] # filter only existing dirs
            if len(outdirs_w_suffixes) == 0: continue # can't really do anything else
            cat_raw_covariance_matrices(nbin, f"l{max_l}", outdirs_w_suffixes, [None] * len(outdirs_w_suffixes), outdir, print_function = print_and_log) # concatenate subsamples
        
        raw_name = os.path.join(outdir, f"Raw_Covariance_Matrices_n{nbin}_l{max_l}.npz")

        # detect the per-file dirs if any
        outdirs_perfile = [int(name) for name in os.listdir(outdir) if name.isdigit()] # per-file dir names are pure integers
        if len(outdirs_perfile) > 0: # if such dirs found, need to cat the raw covariance matrices
            outdirs_perfile = [os.path.join(outdir, str(index)) for index in sorted(outdirs_perfile)] # sort integers, transform back to strings and prepend the parent directory
            cat_raw_covariance_matrices(nbin, f"l{max_l}", outdirs_perfile, [None] * len(outdirs_perfile), outdir, print_function = print_and_log) # concatenate the subsamples
        elif not os.path.isfile(raw_name): # run the raw matrix collection, which creates this file. Non-existing file breaks the logic in my_make()
            collect_raw_covariance_matrices(outdir, print_function = print_and_log)

        # Gaussian covariances

        results_name = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Legendre_n%d_l%d.npz' % (nbin, max_l))
        reg_results.append(results_name)

        cov_name = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt"

        def make_gaussian_cov():
            results = post_process_legendre(outdir, nbin, max_l, outdir, skip_r_bins = skip_r_bins, skip_l = skip_l, print_function = print_and_log)
            convergence_check_extra(results, print_function = print_and_log)

        # RascalC results depend on full output (most straightforwardly)
        my_make(results_name, [raw_name], make_gaussian_cov)
        # Recipe: run post-processing
        # Also perform convergence check (optional but nice)

        # Individual cov file depends on RascalC results
        my_make(cov_name, [results_name], lambda: export_cov_legendre(results_name, max_l, cov_name))
        # Recipe: export cov

        # Jackknife post-processing
        if jackknife:
            results_name_jack = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Legendre_Jackknife_n%d_l%d_j%d.npz' % (nbin, max_l, njack))
            reg_results_jack.append(results_name_jack)
            xi_jack_name = os.path.join(outdir, f"xi_jack/xi_jack_n{nbin}_m{mbin}_j{njack}_11.dat")

            def make_rescaled_cov():
                results = post_process_legendre_mix_jackknife(xi_jack_name, os.path.join(outdir, 'weights'), outdir, mbin, max_l, outdir, skip_r_bins = skip_r_bins, skip_l = skip_l, print_function = print_and_log)
                convergence_check_extra(results, print_function = print_and_log)

            # RascalC results depend on full output (most straightforwardly)
            my_make(results_name_jack, [raw_name], make_rescaled_cov)
            # Recipe: run post-processing
            # Also perform convergence check (optional but nice)

            cov_name_jack = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC.txt"
            # Individual cov file depends on RascalC results
            my_make(cov_name_jack, [results_name_jack], lambda: export_cov_legendre(results_name_jack, max_l, cov_name_jack))
            # Recipe: run convert cov
        
        # Mock post-processing
        if mock_post_processing:
            cov_name_mocks = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_mocks.txt"
            results_name_mocks = os.path.join(outdir, 'Rescaled_Covariance_Matrices_Legendre_Mocks_n%d_l%d.npz' % (nbin, max_l))
            reg_results_mocks.append(results_name_mocks)

            def make_rescaled_cov():
                results = post_process_legendre_mocks(mock_cov_name, outdir, nbin, max_l, outdir, skip_r_bins = skip_r_bins, skip_l = skip_l, print_function = print_and_log)
                convergence_check_extra(results, print_function = print_and_log)

            # RascalC results depend on full output (most straightforwardly)
            my_make(results_name_mocks, [raw_name], make_rescaled_cov)
            # Recipe: run post-processing
            # Also perform convergence check (optional but nice)

            # Individual cov file depends on RascalC results
            my_make(cov_name_mocks, [results_name_mocks], lambda: export_cov_legendre(results_name_mocks, max_l, cov_name_mocks))
            # Recipe: run convert cov

    reg_pycorr_names = [f.filepath for f in fm.select(id = 'correlation_abacus_y3', imock = mock_id, region = regs, **xi_setup)]

    if len(reg_pycorr_names) == len(regs): # if we have pycorr files for all regions
        if len(reg_results) == len(regs): # if we have RascalC results for all regions
            # Combined Gaussian cov

            cov_name = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name, reg_results, lambda: combine_covs_legendre(*reg_results, *reg_pycorr_names, cov_name, max_l, r_step = r_step, skip_r_bins = skip_r_bins, print_function = print_and_log))
            # Recipe: run combine covs

        if jackknife and len(reg_results_jack) == len(regs): # if jackknife and we have RascalC jack results for all regions
            # Combined rescaled cov
            cov_name_jack = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name_jack, reg_results_jack, lambda: combine_covs_legendre(*reg_results_jack, *reg_pycorr_names, cov_name_jack, max_l, r_step = r_step, skip_r_bins = skip_r_bins, print_function = print_and_log))
            # Recipe: run combine covs

        if mock_post_processing and len(reg_results_mocks) == len(regs): # if mock post-processing and we have RascalC mocks results for all regions
            # Combined rescaled cov
            cov_name_mocks = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_mocks.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name_mocks, reg_results_mocks, lambda: combine_covs_legendre(*reg_results_mocks, *reg_pycorr_names, cov_name_mocks, max_l, r_step = r_step, skip_r_bins = skip_r_bins, print_function = print_and_log))
            # Recipe: run combine covs

    if make_mock_cov:
        # Make the mock sample covariance matrix for reg_comb
        mock_cov_name = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_cov_sample.txt"
        this_reg_pycorr_filenames = [f.filepath for f in fm.select(id = 'correlation_abacus_y3', region = reg_comb, **xi_setup)]
        if len(this_reg_pycorr_filenames) > 0: # only if any files found
            my_make(mock_cov_name, [], lambda: sample_cov_multipoles_from_pycorr_files([this_reg_pycorr_filenames], mock_cov_name, max_l = max_l, r_step = r_step, r_max = rmax)) # empty dependencies should result in making this only if the destination file is missing; checking hashes of 1000 mock pycorr files has been taking long

# Save the updated hash dictionary
with open(hash_dict_file, "wb") as f:
    pickle.dump(hash_dict, f)

print_and_log(datetime.now())
print_and_log("Finished execution.")
