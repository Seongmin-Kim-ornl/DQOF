import numpy as np
import pandas as pd
import random, time, logging, os
import utils, ML, sim
from mpi4py import MPI
from DQOF import DQOF

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# hyperparameter setting
problem_size = 12
sub_HOBO_size = 4
num_sub_HOBOs = problem_size
num_parallel = 10      
num_DQAOA_iters = 30  
AL_num_iters = 10
AL_iters = 0  #

# dataset import for active learning
if rank == 0:
    dataset = pd.read_csv('1_dataset/1_trainingset_'+str(problem_size)+'bits.txt', sep=' ', header=None) # dataset import
    dataset = dataset.values.astype(np.float32)

    time_file_path = '1_dataset/2_time_dataset_'+str(problem_size)+'bits.txt' # time dataset import if exist
    if os.path.exists(time_file_path):
        time_save = pd.read_csv(time_file_path, sep=' ', header=None)
        time_save = time_save.values.astype(np.float32)
        repeat_count = time_save[-1][-1]
    else:
        time_save = np.zeros((1,5))
        repeat_count = 0

while AL_iters < AL_num_iters:
    if rank == 0:
        # 3rd FM start
        FM_tic = time.time()
        QUBO, k_third = ML.FM_3rd(dataset)
        FM_time = time.time() - FM_tic
        
        for bcast in range(1,size):
            comm.send(QUBO, dest=bcast)
            comm.send(k_third, dest=bcast)

    if rank != 0:
        QUBO = comm.recv(source=0)
        k_third = comm.recv(source=0)
        
    comm.Barrier()

    # DQOF start
    if rank == 0: 
        DQOF_tic = time.time()

    dqof = DQOF(QUBO, k_third, sub_HOBO_size, num_sub_HOBOs, num_parallel, num_DQAOA_iters)
    solution = dqof.run()
        
    if rank == 0:
        DQOF_time = time.time() - DQOF_tic
        qv_ii = solution
        rep_flag = 0
        # repeat check
        for rep_ct in range(0,dataset.shape[0]):
            if np.array_equal(dataset[rep_ct][1:], qv_ii):  # if there is a same structure
                rep_flag = rep_flag + 10     # increase repeat flag
                print('Repeat structure found!')
                repeat_count = repeat_count + 1
                print(f"original qv_ii: {dataset[rep_ct][1:]} at: {rep_ct}")
            else:
                rep_flag = rep_flag + 0
                    
        while (rep_flag > 5):
            rep_flag = 0
            for ini_data in range(0,np.size(qv_ii)):
                qv_ii[ini_data] = random.randrange(0,2)  # element = 0 or 1        

            for rep_ct in range(0,dataset.shape[0]):
                if np.array_equal(dataset[rep_ct][1:], qv_ii):  # if there is a same structure
                    rep_flag = rep_flag + 10     # increase repeat flag
                else:
                    rep_flag = rep_flag + 0

        # FOM calculation start
        FOM_tic = time.time()
        FOM = sim.TMM_cal(qv_ii)
        FOM_time = time.time() - FOM_tic

        # one cycle is done, data save
        new_data = np.concatenate((np.array(FOM[0]), np.array(qv_ii)), axis=0)
        time_data = np.concatenate((np.array(time_save.shape[0]).reshape(1,1), np.array(FM_time).reshape(1,1), 
                                    np.array(DQOF_time).reshape(1,1), np.array(FOM_time).reshape(1,1), np.array(repeat_count).reshape(1,1)), axis=1)  
        dataset = np.vstack((dataset, new_data))
        time_save = np.vstack((time_save, time_data))
        np.savetxt('1_dataset/1_trainingset_'+str(problem_size)+'bits.txt', dataset, fmt='%.4f')
        np.savetxt('1_dataset/2_time_dataset_'+str(problem_size)+'bits.txt', time_save, fmt='%.4f')
        
        print('='*77)
        print(f"Iter {dataset.shape[0]}, FOM: {FOM}, min FOM: {min(dataset[:,0]):.6f}, elapsed time: {round((FM_time+DQOF_time+FOM_time), 6)}")        
        
    AL_iters += 1

# active learning done
if rank == 0:
    print('*'*77)
    print('Active learning done')   
