"Usage: python sum_elapsed_for_job_array_id.py <job_array_id> [<job_array_id2> ...]"
import sys
import subprocess

total_seconds = 0

for job_array_id in sys.argv[1:]:
    query = subprocess.check_output(["sacct", "-j", job_array_id, "--format=Elapsed", "--noheader", "-X"]).decode() # -X suppresses the duplicate entries for each job ID, leaving only the single total for each

    for l in query.splitlines():
        h, m, s = map(int, l.split(':'))
        total_seconds += h * 3600 + m * 60 + s

print(f"Total elapsed time: {total_seconds // 3600}:{total_seconds % 3600 // 60}:{total_seconds % 60}, or {total_seconds / 3600:.2f} hours, or {total_seconds} seconds")