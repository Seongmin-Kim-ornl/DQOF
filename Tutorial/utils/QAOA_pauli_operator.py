# General imports
import numpy as np
from qiskit.quantum_info import SparsePauliOp, random_hermitian, Pauli

# Puali operator
def get_operator(weight_matrix: np.ndarray) -> tuple[SparsePauliOp, float]:
    num_nodes = len(weight_matrix)
    pauli_list = []
    coeffs = []
    shift = 0
    w = weight_matrix
    for i in range(num_nodes):
        for j in range(num_nodes):     
            for k in range(num_nodes):   
                if weight_matrix[i, j, k] != 0:
                    # 3rd order
                    
                    if (i != j and i != k and j != k): 
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[i] = True
                        z_p[j] = True
                        z_p[k] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(-0.125*w[i,j,k])
                        
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[i] = True
                        z_p[j] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(0.125*w[i,j,k])
    
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[i] = True
                        z_p[k] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(0.125*w[i,j,k])
    
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[j] = True
                        z_p[k] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(0.125*w[i,j,k])
                        
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[i] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(-0.125*w[i,j,k])
                        
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[j] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(-0.125*w[i,j,k])
                        
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[k] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(-0.125*w[i,j,k])
                        
                        shift += 0.125*w[i,j,k]
                        
                    # 2nd order
                    if (i == j and i != k):   # i & k
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[i] = True
                        z_p[k] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(0.25*w[i,j,k])
                        
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[i] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(-0.25*w[i,j,k])
                        
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[k] = True
                        pauli_list.append(Pauli((z_p, x_p)))
                        coeffs.append(-0.25*w[i,j,k])
                        
                        shift += 0.25*w[i,j,k]
                    
                    # 1st order
                    if (i == j and i == k and j == k):
                         x_p = np.zeros(num_nodes, dtype=bool)
                         z_p = np.zeros(num_nodes, dtype=bool)
                         z_p[i] = True
                         #z_p[j] = True
                         #z_p[k] = True
                         pauli_list.append(Pauli((z_p, x_p)))
                         coeffs.append(-0.5*w[i,j,k])
                         
                         shift += 0.5*w[i,j,k]
                     
    return SparsePauliOp(pauli_list, coeffs=coeffs), shift