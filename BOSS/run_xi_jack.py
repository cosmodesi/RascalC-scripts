# This is the custom script to compute the 2PCF with jackknives

import numpy as np
from pycorr import TwoPointCorrelationFunction
from tqdm import trange

from pre_process import prepare_galaxy_random_catalogs, get_rdd_positions, n_jack, z_min, z_max

galaxies, randoms = prepare_galaxy_random_catalogs()

# decide number of splits so that random parts are about the data size
n_splits = len(randoms) // len(galaxies)

# get the desired weight ratio
sum_w_randoms = sum(randoms["WEIGHT"])
sum_w_goal = sum_w_randoms / n_splits

print("Shuffling randoms")
random_indices = np.arange(len(randoms))
np.random.shuffle(random_indices) # in place, by first axis
print("Splitting randoms")
all_randoms = [randoms[random_indices[i::n_splits]] for i in range(n_splits)]

# reweigh each part so that the ratio of randoms to data is the same, just in case
for i_random in trange(n_splits, desc="Reweighting random part"):
    sum_w_random_part = sum(all_randoms[i_random]["WEIGHT"])
    w_ratio = sum_w_goal / sum_w_random_part
    all_randoms[i_random]["WEIGHT"] *= w_ratio

n_mu_bins = 200 # between -1 and 1
s_max = 200
split_above = 20
# tuple of edges, so that non-split randoms are below split_above and above they are split
all_edges = ((s_edges, np.linspace(-1, 1, n_mu_bins+1)) for s_edges in (np.arange(split_above+1), np.arange(split_above, s_max+1)))

results = []
# compute
for i_split_randoms, edges in enumerate(all_edges):
    result = 0
    D1D2 = None
    for i_random in trange(n_splits if i_split_randoms else 1, desc = "Computing xi with random part"):
        these_randoms = all_randoms[i_random] if i_split_randoms else randoms
        tmp = TwoPointCorrelationFunction(mode = 'smu', edges = edges,
                                          data_positions1 = get_rdd_positions(galaxies), data_weights1 = galaxies["WEIGHT"], data_samples1 = galaxies["JACK"],
                                          randoms_positions1 = get_rdd_positions(these_randoms), randoms_weights1 = these_randoms["WEIGHT"], randoms_samples1 = these_randoms["JACK"],
                                          position_type = 'rdd', engine = 'corrfunc', D1D2 = D1D2, gpu = True, nthreads = 4)
        # position_type='pos' corresponds to (N, 3) x,y,z positions shape like we have here
        D1D2 = tmp.D1D2
        result += tmp
    results.append(result)
print("Finished xi computations")
corr = results[0].concatenate_x(*results)
corr.D1D2.attrs['nsplits'] = n_splits

print("Saving the result")
corr.save(f"allcounts_BOSS_CMASS_N_{z_min}_{z_max}_lin_njack{n_jack}_nran{n_splits}_split{split_above}.npy")
print("Finished")