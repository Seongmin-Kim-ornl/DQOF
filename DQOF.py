from mpi4py import MPI
import numpy as np
import random, time, logging
import utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class DQOF:
    def __init__(self, Q, k_third, sub_HUBO_size, num_sub_HUBOs, num_parallel, num_DQAOA_iters):
        self.Q = Q
        self.problem_size = len(Q)
        self.k_third = k_third
        self.HUBO = utils.formulate_HUBO(self.problem_size, self.Q, self.k_third)
        self.sub_HUBO_size = sub_HUBO_size
        self.num_sub_HUBOs = num_sub_HUBOs
        self.num_parallel = num_parallel
        self.num_DQAOA_iters = num_DQAOA_iters
        self.idx_list = list(range(self.problem_size))

    def initialize(self):
        np.random.seed(12345)
        self.x_optimal = np.random.choice([0, 1], size=self.problem_size)

    def decompose(self):
        random.seed(time.time())
        choicelist = random.sample(self.idx_list, self.sub_HUBO_size)
        choicelist.sort()
        rand_subHUBO = np.zeros((self.sub_HUBO_size, self.sub_HUBO_size, self.sub_HUBO_size))
        for sub_i in range(self.sub_HUBO_size):
            for sub_j in range(sub_i, self.sub_HUBO_size):
                for sub_k in range(sub_j, self.sub_HUBO_size):
                    rand_subHUBO[sub_i,sub_j,sub_k] = self.HUBO[choicelist[sub_i], choicelist[sub_j], choicelist[sub_k]]
                    
        return rand_subHUBO, choicelist

    def solve(self, H):
        return utils.solve_H(H)

    def aggregate(self, x_sub, choicelist):
        for idx in range(self.sub_HUBO_size):
            x_new_subQUBO = np.array(self.x_optimal)
            x_new_subQUBO[choicelist[idx]] = x_sub[idx]
            
            QUBO_original = utils.cal_HUBO_energy(self.x_optimal, self.Q, self.k_third)
            QUBO_new = utils.cal_HUBO_energy(x_new_subQUBO, self.Q, self.k_third)
            if QUBO_original > QUBO_new:
                self.x_optimal = x_new_subQUBO

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        MPI_size = comm.Get_size()

        if MPI_size < self.num_parallel*self.num_sub_HUBOs + self.num_parallel + 1:
            if rank ==0:
                raise Exception(f"More than {self.num_parallel * self.num_sub_HUBOs + self.num_parallel + 1} cores should be used!")
        
        else:
            self.initialize()
            for cycle in range(self.num_DQAOA_iters):
                if rank >= 1 and rank <= self.num_parallel:
                    # Generate subproblems and pass them to workers
                    for dist_num in range(0, self.num_sub_HUBOs):
                        subHUBO, choices = self.decompose()
                        comm.send(subHUBO, dest=(rank-1)*self.num_sub_HUBOs + self.num_parallel + dist_num + 1)
                        comm.send(choices, dest=(rank-1)*self.num_sub_HUBOs + self.num_parallel + dist_num + 1)
    
                for parallel_ct in range(1, self.num_parallel+1):  
                    if rank >= self.num_parallel + (parallel_ct - 1)*self.num_sub_HUBOs + 1 and rank <= self.num_parallel + parallel_ct*self.num_sub_HUBOs:
                        recv_data_subHUBO = comm.recv(source=parallel_ct)
                        recv_data_choicelist = comm.recv(source=parallel_ct)
                        x_subHUBO = self.solve(recv_data_subHUBO)
                        comm.send(x_subHUBO, dest=parallel_ct)
                        comm.send(recv_data_choicelist, dest=parallel_ct)
                        break
                        
                if rank >= 1 and rank <= self.num_parallel:
                    for res_rec_rank in range(0,self.num_sub_HUBOs):
                        x_subHUBO_rank = comm.recv(source = (rank -1)*self.num_sub_HUBOs + self.num_parallel + res_rec_rank + 1)
                        choicelist = comm.recv(source = (rank -1)*self.num_sub_HUBOs + self.num_parallel + res_rec_rank + 1)
                        self.aggregate(x_subHUBO_rank, choicelist)
                    
            if rank >= 1 and rank <= self.num_parallel:        
                enegy_each = utils.cal_HUBO_energy(self.x_optimal, self.Q, self.k_third)
                comm.send(self.x_optimal, dest=0)
                comm.send(enegy_each, dest=0)
    
            if rank == 0:
                x_optimal_p = np.zeros((self.num_parallel, self.problem_size))
                energy_p = np.zeros(self.num_parallel)
                for res_p in range(1, self.num_parallel+1):
                    x_optimal_p[res_p-1] = comm.recv(source=res_p)
                    energy_p[res_p-1] = comm.recv(source=res_p)

                
                global_idx = np.where(np.min(energy_p) == energy_p)[0][0]
                energy_global_optimal = energy_p[global_idx]
                x_global_optimal = x_optimal_p[global_idx,:]                
                
                '''print(f"x_candidates: {x_optimal_p}")                  
                print(f"energy_candidates: {energy_p}")                      
                print('*'*77)
                print(f"best energy: {energy_global_optimal:.6f}, best candidate: {x_global_optimal}")   '''
    
                return x_global_optimal       
