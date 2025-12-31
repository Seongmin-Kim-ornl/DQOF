import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def cal_3_interaction(x, third_interactions):
    # x = binary bit. np.array
    # Third-order terms "Lemma3, Polynomial Networks and Factorization Machines" 
    #homogenius polynomial kernel H3(K_second, x)
    # input x와 third_interaction을 torch의 tensor로 바꿔주기.
    x_2_dimension = x.reshape(1, -1)
    x_2_dimension = x_2_dimension.astype(np.float32)
    f_array = third_interactions.astype(np.float32)
    tensor_x = torch.from_numpy(x_2_dimension)
    tensor_k = torch.from_numpy(f_array)
    H_3_px = torch.matmul(tensor_x, tensor_k).pow(3).sum(1, keepdim=True) 
    #D3(K_third, x)
    D_3_px = torch.matmul(tensor_x.pow(3), tensor_k.pow(3)).sum(1, keepdim=True)        
    #D2,1(K_third, x)
    D_21_px_D2 = torch.matmul(tensor_x.pow(2), tensor_k.pow(2))
    D_21_px_D1 = torch.matmul(tensor_x.pow(1), tensor_k.pow(1))
    D_21_px=torch.mul(D_21_px_D2,D_21_px_D1).sum(1, keepdim=True) 
    # (1/6)*(H3-3D21+2D3)
    Anova_3rd = (1/6)*(H_3_px - 3*D_21_px + 2*D_3_px) 
    return Anova_3rd

# energy state
def cal_HUBO_energy(x, Q, k_third):
    x = np.array(x)
    HUBO_energy = x@Q@np.transpose(x) + np.array(cal_3_interaction(x, k_third))
    energy = HUBO_energy.item()
    return energy