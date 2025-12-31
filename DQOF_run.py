from mpi4py import MPI
from DQOF import DQOF
import numpy as np
import random, time, logging
import utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# main
problem_size = 40
sub_HUBO_size = 4
num_sub_HUBOs = problem_size
num_parallel = 10
num_DQAOA_iters = 50

if rank == 0:
    tic = time.time()
    
Q = utils.load_QUBO(problem_size)
k_third = utils.load_k_third(problem_size)
dqof = DQOF(Q, k_third, sub_HUBO_size, num_sub_HUBOs, num_parallel, num_DQAOA_iters)
solution = dqof.run()

if rank == 0:
    print('======= DQOF result =======')
    print(f"problem size: {problem_size}, sub_HUBO_size: {sub_HUBO_size}, num_sub_HUBOs: {num_sub_HUBOs}, num_parallel: {num_parallel}, num_DQAOA_iters: {num_DQAOA_iters}")
    print(f"solution x: {solution}")
    print(f"solution energy: {utils.cal_HUBO_energy(solution, Q, k_third)}")
    print(f"elapsed time: {time.time() - tic}")
