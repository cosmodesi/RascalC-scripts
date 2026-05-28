import asdf

hash_dict_file = "make_covs.hash_dict.asdf"
# Load hash dictionary from file
with asdf.open(hash_dict_file) as af:
    hash_dict = af["goal_deps_hashes"]

replace_from = 'z0.0-' # old incorrect format for z_min=0, incompatible with desi-y3-kp
replace_to = 'z0-' # new corrected format for z_min=0, compatible with desi-y3-kp

# rename the top-level keys in the hash dictionary, which are the goal/destination file names
for fname in list(hash_dict.keys()): # need list() to avoid "dictionary changed size during iteration" error when renaming keys
    if replace_from in fname:
        new_fname = fname.replace(replace_from, replace_to)
        hash_dict[new_fname] = hash_dict.pop(fname) # rename the key by popping the old one and assigning to the new one. will replace the key if it already exists, but that should only correspond to old files

# rename the keys in the nested dictionaries, which are the dependency file names
for goal, deps_hashes in hash_dict.items():
    for dep, hash_val in list(deps_hashes.items()): # need list() to avoid "dictionary changed size during iteration" error when renaming keys
        if replace_from in dep:
            new_dep = dep.replace(replace_from, replace_to)
            deps_hashes[new_dep] = deps_hashes.pop(dep) # rename the key by popping the old one and assigning to the new one

# Save the updated hash dictionary
af = asdf.AsdfFile(dict(goal_deps_hashes=hash_dict))
af.write_to(hash_dict_file)