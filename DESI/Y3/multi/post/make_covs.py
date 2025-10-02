# This script generates all covs

import os
from datetime import datetime
import pickle
import hashlib
from typing import Callable
import traceback
import desi_y3_files.file_manager as desi_y3_file_manager
from RascalC.raw_covariance_matrices import cat_raw_covariance_matrices, collect_raw_covariance_matrices
from RascalC import post_process_auto
from RascalC.utils import blank_function
from RascalC.cov_utils import export_cov_legendre_multi, export_cov_legendre_cross, convert_txt_cov_multi_to_cross
from RascalC.comb.combine_covs_legendre_multi import combine_covs_legendre_multi

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

# Settings for filenames
verspec = 'loa-v1'
version = "v1.1"
conf = "BAO/unblinded"
conf_alt = "unblinded"

# Set DESI CFS before creating the file manager
os.environ["DESICFS"] = "/dvs_ro/cfs/cdirs/desi" # read-only path

fm = desi_y3_file_manager.get_data_file_manager(conf, verspec)

regs = ('SGC', 'NGC') # regions for filenames
reg_comb = "GCcomb"

tracers = [['ELG_LOPnotqso', 'QSO']]
zs = [(1.1, 1.6)]

# for custom cross-counts
nrandoms = 5
split_above = 20

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

def print_and_log(s: object = "") -> None:
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
for tlabels, z_range in zip(tracers, zs):
    corlabels = [tlabels[0], "_".join(tlabels), tlabels[1]]
    tracers_label_full = "&".join(tlabels)
    z_min, z_max = z_range
    reg_results = []
    # get options automatically
    xi_setup = desi_y3_file_manager.get_baseline_2pt_setup(tlabels[0], z_range, recon = True)
    recon_spec = 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}'.format_map(xi_setup) # recon specifier string
    recon_spec += '' if (zr := xi_setup['recon_zrange']) is None else '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange = zr)
    recon_spec += '' if (w := xi_setup['recon_weighting']) == 'default' else '_{}'.format(w)
    if jackknife: reg_results_jack = []
    for reg in regs:
        outdir = os.path.join("outdirs", verspec, version, conf, recon_spec, "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}") # output file directory
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

        results_name = post_process_auto(outdir, jackknife=False, two_tracers=True, skip_s_bins=skip_r_bins, skip_l=skip_l, print_function=blank_function, dry_run=True)["path"]
        reg_results.append(results_name)

        cov_dir = f"cov_txt/{verspec}/{version}/{conf}/{recon_spec}"
        cov_name = f"{cov_dir}/xi" + xilabel + "_" + "_".join([tracers_label_full, reg]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt"

        # RascalC results depend on full output (most straightforwardly)
        my_make(results_name, [raw_name],
                lambda: post_process_auto(outdir, jackknife=False, two_tracers=True, skip_s_bins=skip_r_bins, skip_l=skip_l, print_function=print_and_log))
        # Recipe: run post-processing; extra convergence check included by default

        # Individual cov file depends on RascalC results
        my_make(cov_name, [results_name], lambda: export_cov_legendre_multi(results_name, max_l, cov_name))
        # Recipe: export cov

        # Export cross-only covariance to a filename with "_" instead of "&" between the tracers
        cov_name = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt"
        my_make(cov_name, [results_name], lambda: export_cov_legendre_cross(results_name, max_l, cov_name))

        # Jackknife post-processing
        if jackknife:
            results_name_jack = post_process_auto(outdir, jackknife=True, two_tracers=True, skip_s_bins=skip_r_bins, skip_l=skip_l, print_function=blank_function, dry_run=True)["path"]
            reg_results_jack.append(results_name_jack)

            # RascalC results depend on full output (most straightforwardly)
            my_make(results_name_jack, [raw_name],
                    lambda: post_process_auto(outdir, jackknife=True, two_tracers=True, skip_s_bins=skip_r_bins, skip_l=skip_l, print_function=print_and_log))
            # Recipe: run post-processing; extra convergence check included by default

            cov_name_rescaled = f"{cov_dir}/xi" + xilabel + "_" + "_".join([tracers_label_full, reg]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC.txt"
            # Individual cov file depends on RascalC results
            my_make(cov_name_rescaled, [results_name_jack], lambda: export_cov_legendre_multi(results_name_jack, max_l, cov_name_rescaled))
            # Recipe: export cov

            # Export cross-only covariance to a filename with "_" instead of "&" between the tracers
            cov_name_rescaled = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC.txt"
            my_make(cov_name_rescaled, [results_name_jack], lambda: export_cov_legendre_cross(results_name_jack, max_l, cov_name_rescaled))

    # get the pycorr filenames from the file manager
    reg_pycorr_names = []
    for tracer in tlabels[:1]:
        xi_setup = desi_y3_file_manager.get_baseline_2pt_setup(tracer, z_range, recon=True)
        xi_setup.update({"version": version, "tracer": tracer, "region": regs, "zrange": z_range, "cut": None, "njack": 0}) # specify regions, version, z range and no cut; no need for jackknives
        reg_pycorr_names += [[f.filepath for f in fm.select(id = 'correlation_recon_y3', **xi_setup)]]
    # add the customized path to the cross-correlation and second auto-correlation
    reg_pycorr_names += [[os.environ["DESICFS"] + f"/users/sandersn/DA2/{verspec}/{version}/{conf_alt}/{recon_spec}/xi/smu/allcounts_{corlabel}_{reg}_{z_min}_{z_max}_default_FKP_lin_njack{njack}_nran{nrandoms}_split{split_above}.npy" for reg in regs] for corlabel in corlabels[1:]]

    if len(reg_pycorr_names) == len(regs): # if we have pycorr files for all regions
        if len(reg_results) == len(regs): # if we have RascalC results for all regions
            # Combined Gaussian cov
            cov_name = f"{cov_dir}/xi" + xilabel + "_" + "_".join([tracers_label_full, reg_comb]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC_Gaussian.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name, reg_results, lambda: combine_covs_legendre_multi(*reg_results, *reg_pycorr_names, cov_name, max_l, r_step = r_step, skip_r_bins = skip_r_bins, print_function = print_and_log))
            # Recipe: run combine covs

        if jackknife and len(reg_results_jack) == len(regs): # if jackknife and we have RascalC jack results for all regions
            # Combined rescaled cov
            cov_name_rescaled = f"{cov_dir}/xi" + xilabel + "_" + "_".join([tracers_label_full, reg_comb]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC.txt" # combined cov name

            # Comb cov depends on the region RascalC results
            my_make(cov_name_rescaled, reg_results_jack, lambda: combine_covs_legendre_multi(*reg_results_jack, *reg_pycorr_names, cov_name_rescaled, max_l, r_step = r_step, skip_r_bins = skip_r_bins, print_function = print_and_log))
            # Recipe: run combine covs

            # Export cross-only covariance to a filename with "_" instead of "&" between the tracers
            cov_name_cross = f"{cov_dir}/xi" + xilabel + "_" + "_".join(tlabels + [reg_comb]) + f"_z{z_min}-{z_max}_default_FKP_lin{r_step}_s{rmin_real}-{rmax}_cov_RascalC.txt" # combined cov name
            my_make(cov_name_cross, [cov_name_rescaled], lambda: convert_txt_cov_multi_to_cross(cov_name_rescaled, cov_name_cross))

# Save the updated hash dictionary
with open(hash_dict_file, "wb") as f:
    pickle.dump(hash_dict, f)

print_and_log(datetime.now())
print_and_log("Finished execution.")
