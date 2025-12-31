import numpy as np
import os

def load_QUBO(HUBO_size):
    QUBO_file = open(str(HUBO_size)+'_QUBO.txt', 'r') # import Q matrix
    A = QUBO_file.read()
    A = np.asmatrix(A)
    m = int(np.size(A)**(1/2))  # number of binary variables
    A = np.reshape(A, (m,m))
    Q = A
    Q = np.array(Q)
    return Q

def load_k_third(HUBO_size):
    # import 3rd interactions
    file = open(str(HUBO_size)+'_k_third.txt', 'r') 
    A = file.read()
    A = np.asmatrix(A)
    A = np.reshape(A, (HUBO_size,8))
    k_third = A
    k_third = np.array(k_third)
    return k_third

def formulate_HUBO(HUBO_size, QUBO_2nd, k_third):
    HUBO = np.zeros((HUBO_size,HUBO_size,HUBO_size))
    # 1st order (linear)
    for i in range(HUBO_size):
        HUBO[i,i,i] = QUBO_2nd[i,i]
    
    # 2nd order (quadratic)
    for i in range(HUBO_size):
        for j in range(i+1,HUBO_size):
            HUBO[i,i,j] = QUBO_2nd[i,j]
    
    # 3rd order (3rd interaction)
    for i in range(HUBO_size):
        for j in range(i+1,HUBO_size):
            for k in range(j+1,HUBO_size):
                for f in range(8):
                    HUBO[i,j,k]+=k_third[i,f]*k_third[j,f]*k_third[k,f]

    return HUBO