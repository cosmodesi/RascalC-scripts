# Script that summarizes the output/log files
import sys
import numpy as np
from datetime import datetime

print_if_words = ["warn", "error"] # lowercase

for filename in sys.argv[1:]:
    print(f"\nSummary of {filename}:")
    start_time, time = None, None
    R_inv = []
    for line in open(filename):
        # print if warning or error
        if any(word in line.lower() for word in print_if_words):
            print(line.strip())
        # get R_inv
        if line.startswith("RMS eigenvalues"):
            line_parts = line.split()
            R_inv += [float(line_parts[i]) for i in (-3, -1)]
        # try to get timestamps
        try:
            time = datetime.fromisoformat(line.strip())
            if not start_time: start_time = time
        except ValueError: pass
    # report time if possible
    if time != start_time: print(f"Elapsed time: {time - start_time}")
    # report full R_inv if possible
    if len(R_inv) >= 4:
        R_inv_full = np.around(100 * np.sort(R_inv[:4])[::-1], 1)
        print("R_inv:", *R_inv_full, "%")
        if len(R_inv) >= 8:
            R_inv_jack = np.around(100 * np.sort(R_inv[4:8])[::-1], 1)
            print("R_inv jack:", *R_inv_jack, "%")

print()