import subprocess
import os

grid_sizes = [128, 256]
time_grid_size = 512
steps = 20
T = 1.0
Ls = [1.0, 3.141592653589793]
threads = [1, 2, 4, 8]
processes = [2, 4, 8]

for grid_size in grid_sizes:
    for L in Ls:
        for num_processes in processes:
            for num_threads in threads:
                modified_env = os.environ.copy()
                modified_env["OMP_NUM_THREADS"] = str(num_threads)
                command = f"mpiexec -n {num_processes} ./main_mpi {L} {grid_size} {T} {time_grid_size} {steps}"
                print(command)
                subprocess.run(command, env=modified_env, shell=True)