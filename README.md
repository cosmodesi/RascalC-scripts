# RascalC scripts
[RascalC](https://rascalc.readthedocs.io/en/latest/) is a code for fast estimation of semi-analytical covariance matrices for galaxy 2-point correlation functions (not only, but other uses have been less common).
This is a cleaned collection of scripts utilizing its Python library (installed from <https://github.com/misharash/RascalC>).

## Technical recommendations (for DESI members at NERSC)

Load the `cosmodesi` environment:
```
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
```
(this is done in the batch scripts.)

Install the `RascalC` library from the common directory in development mode (after loading the `cosmodesi` environment):
```
pip install -e /global/common/software/desi/users/mrash/RascalC
```

You may need to additionally install the LSS package (<https://github.com/desihub/LSS>) for cutsky computations (fiducial DESI cosmology to convert redshift to distance).

DESI Y1 data runs rely on [cosmodesi/desi-y1-kp](https://github.com/cosmodesi/desi-y1-kp) installed to retrieve the filenames.
DESI Y3 data runs similarly rely on [cosmodesi/desi-y3-kp](https://github.com/cosmodesi/desi-y3-kp).
These repositories are private and only accessible to [cosmodesi](https://github.com/cosmodesi) team members; they should become public after the data release.

## Usage remarks

If anything is not clear or not working, please feel free to [open an issue](https://github.com/misharash/RascalC-scripts/issues) or contact Michael 'Misha' Rashkovetskyi (<mrashkovetskyi@cfa.harvard.edu>) personally.

It is okay if you do not feel confident to actually run the code.
Setting up the filenames in `run_cov.py` for your setup would be a big help already, since it takes surprisingly much time to figure out an alternative naming scheme.

## Common files

- `run_covs.py` – computation-heavy (needs a compute node) Python script running the code, typically expects a command-line argument to determine which of the similar tasks to perform.
IMPORTANT: do not launch this script multi-threaded (e.g. at NERSC, do not set `OMP_*` and other `*_THREADS` environment variables), otherwise the main computation may run effectively single-threaded.
The Python function `run_cov` for calling `RascalC` should set up OpenMP threads by itself based on `nthread` argument.
If you need to run something besides `RascalC` on multiple threads, run it separately (one of the options is to create a child process).
- `run_covs.sh` – SLURM batch script submitting an array of jobs doing the whole list of similar tasks.
- `make_covs.py` – computation-light (usually ok to run on a login node) Python script to format the covariance matrices into text files. Only run when all the submitted jobs have finished. The script tracks the dependencies: does not run the recipe if sources are missing or have not been changed since the previous run (the knowledge about the latter is only gathered after the first execution without critical failures).
