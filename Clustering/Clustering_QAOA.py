from qiskit.result import QuasiDistribution
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IQP, QAOAAnsatz
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp
from qiskit.quantum_info import SparsePauliOp, random_hermitian, Pauli
from rustworkx.visualization import mpl_draw as draw_graph
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
import time, warnings, timeit, torch
warnings.simplefilter("ignore")
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import utils

backend = AerSimulator(method='automatic') 
#backend = service.backend(name="ibm_kingston")
print('backend_har',backend)
backend_sim = AerSimulator(method='automatic')
print('backend_sim',backend_sim)

print('='*77)

problem_size = 4
num_layers = 2
transpile_level = 2
n_shots = int(1e4)

for num_parallel in range (1,6):
    H = []
    Q = []
    k_thirds = []
    grounds = []
    
    def int_to_padded_float64_binary_array(num, num_bits):
        if num < 0:
            raise ValueError("We cannot process negative numbers.")
        
        binary_string = bin(num)[2:]  
        binary_string = binary_string.zfill(num_bits)
        binary_array = np.array(list(binary_string), dtype=np.float32)
        return binary_array
    
    def cal_3_interaction(x, third_interactions):
        x_2_dimension = x.reshape(1, -1)
        x_2_dimension = x_2_dimension.astype(np.float32)
        f_array = third_interactions.astype(np.float32)
        tensor_x = torch.from_numpy(x_2_dimension)
        tensor_k = torch.from_numpy(f_array)
        H_3_px = torch.matmul(tensor_x, tensor_k).pow(3).sum(1, keepdim=True) 
        D_3_px = torch.matmul(tensor_x.pow(3), tensor_k.pow(3)).sum(1, keepdim=True)        
        D_21_px_D2 = torch.matmul(tensor_x.pow(2), tensor_k.pow(2))
        D_21_px_D1 = torch.matmul(tensor_x.pow(1), tensor_k.pow(1))
        D_21_px=torch.mul(D_21_px_D2,D_21_px_D1).sum(1, keepdim=True) 
        Anova_3rd = (1/6)*(H_3_px - 3*D_21_px + 2*D_3_px) 
        return Anova_3rd
    
    for i in range(num_parallel):
        QUBO_file = open('QUBO'+str(i+1)+'_4.txt', 'r') # import Q matrix
        A = QUBO_file.read()
        A = np.asmatrix(A)
        m = int(np.size(A)**(1/2))
        A = np.reshape(A, (m,m))
        QUBO = A
        QUBO = np.array(QUBO)
        Q.append(QUBO)
        
        file = open('k_third'+str(i+1)+'_4.txt', 'r') 
        A = file.read()
        A = np.asmatrix(A)
        A = np.reshape(A, (problem_size,8))
        k_third = A
        k_third = np.array(k_third)
        k_thirds.append(k_third)
        
        HUBO = utils.formulate_HUBO(problem_size, QUBO, k_third)
        H.append(HUBO)
        
        # Calculate ground truth
        ground = 0
        for i in range(0, 2**problem_size):
            x = np.array(int_to_padded_float64_binary_array(i, problem_size), dtype='i')
            energy = x@QUBO@np.transpose(x) + np.array(cal_3_interaction(x, k_third))
            if (energy) < np.float32(ground):
                x_optimal = x
                ground = energy[0][0]
        grounds.append(ground)
    
        print(f"Ground truth: {ground}")
     
    cost_hamiltonian = []
    
    for i in range(num_parallel):
        cost_ham, offset = utils.get_operator(H[i])
        cost_hamiltonian.append(cost_ham)
    
    
    qaoa_circuits = []
    
    for i in range(num_parallel):
        # Define parameter vectors for this circuit
        gamma_vec = ParameterVector(f'γ{i}', num_layers)
        beta_vec = ParameterVector(f'β{i}', num_layers)
        
        # Build the circuit (you can also customize cost_operator per circuit)
        circuit = QAOAAnsatz(cost_operator=cost_hamiltonian[i], reps=num_layers)
        
        # Get the parameters in order: first all gammas, then all betas
        circuit_params = list(circuit.parameters)
        
        # Create mapping
        param_map = {circuit_params[j]: gamma_vec[j] for j in range(num_layers)}
        param_map.update({circuit_params[j + num_layers]: beta_vec[j] for j in range(num_layers)})
        
        # Bind and store
        bound_circuit = circuit.assign_parameters(param_map)
        qaoa_circuits.append(bound_circuit)
    
    
    qr = QuantumRegister(problem_size*num_parallel)
    combined_circuit = QuantumCircuit(qr)
    
    # Combine each circuit on its own qubit subset
    for idx, key in enumerate(qaoa_circuits):
        circuit = qaoa_circuits[idx]
        start = idx * problem_size
        end = start + problem_size
        combined_circuit.compose(circuit, qubits=[qr[i] for i in range(start, end)], inplace=True)
    
    # Add measurements to all qubits
    combined_circuit.measure_all()
    
    # Transpile with preset pass manager
    pm = generate_preset_pass_manager(optimization_level=transpile_level, backend=backend_sim)
    candidate_circuit = pm.run(combined_circuit)
    

    

    
    def shift_sparse_pauli_op(pauli_op: SparsePauliOp, shift: int, total_qubits: int) -> SparsePauliOp:
        shifted_paulis = []
        for p, coeff in zip(pauli_op.paulis, pauli_op.coeffs):
            new_z = np.zeros(total_qubits, dtype=bool)
            new_x = np.zeros(total_qubits, dtype=bool)
            new_z[shift:shift+len(p.z)] = p.z
            new_x[shift:shift+len(p.x)] = p.x
            shifted_paulis.append(Pauli((new_z, new_x)))
        return SparsePauliOp(shifted_paulis, coeffs=pauli_op.coeffs)
        
    def identity_sparse_pauli_op(n_qubits: int) -> SparsePauliOp:
        z = np.zeros(n_qubits, dtype=bool)
        x = np.zeros(n_qubits, dtype=bool)
        return SparsePauliOp([Pauli((z, x))], coeffs=[1.0])
    
    
    # Combine cost_hamiltonian[0] to cost_hamiltonian[num_parallel - 1]
    def combine_all_hamiltonians(cost_hamiltonians, problem_size, num_parallel):
        total_qubits = problem_size * num_parallel
        
        # Start with the first Hamiltonian
        combined_hamiltonian = shift_sparse_pauli_op(cost_hamiltonians[0], shift=0, total_qubits=total_qubits)
        
        # Add the shifted versions of the other Hamiltonians
        for i in range(1, num_parallel):
            shifted = shift_sparse_pauli_op(cost_hamiltonians[i], shift=i * problem_size, total_qubits=total_qubits)
            combined_hamiltonian += shifted
        
        return combined_hamiltonian
    combined_hamiltonian = combine_all_hamiltonians(cost_hamiltonian, problem_size, num_parallel)
    
  
    
    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        global iter_count, qpu_usage, queue_time, caltime, best_cost
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    
        pub = (ansatz, isa_hamiltonian, params)
    
        queue_start = time.time()
        
        job = estimator.run([pub])    
        results = job.result()[0]
        cost = results.data.evs
        objective_func_vals.append(cost)
    
        qpu_usage += 0
        queue_elapsed = 0
        queue_time += 0
        caltime = time.time() - tic
        
        iter_count += 1
    
        if cost < best_cost:
            best_cost = cost
    
        return cost  
    
    initial_gamma = np.pi / 4
    initial_beta = np.pi / 8
    init_params = [initial_gamma, initial_beta]*num_layers*num_parallel
    objective_func_vals = [] 
    
    best_cost = 0
    iter_count = 0
    qpu_usage = 0
    queue_time = 0
    
    estimator = Estimator(mode=backend_sim)
    estimator.options.default_shots = n_shots
    
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"
    
    tic = time.time()
    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, combined_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-4,
        options={'maxiter': 200}
    )

    pm = generate_preset_pass_manager(optimization_level=transpile_level, backend=backend)
    candidate_circuit = pm.run(combined_circuit)
    optimized_circuit = candidate_circuit.assign_parameters(result.x)
    
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = n_shots
    
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

    queue_start = time.time()
    pub= (optimized_circuit, )
    job = sampler.run([pub], shots=int(n_shots))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    max_bitstring = max(counts_bin, key=counts_bin.get)
    most_likely_bitstring = [int(bit) for bit in max_bitstring]
    print("Most probable bitstring:", max_bitstring)
    print("As list:", most_likely_bitstring)
    
    qpu_usage += job.metrics()['usage']['quantum_seconds']
    queue_elapsed = time.time() - queue_start - job.metrics()['usage']['quantum_seconds']
    queue_time += queue_elapsed
    caltime = time.time() - tic
    
    most_likely_bitstring.reverse()
    
    print("Result bitstring:", most_likely_bitstring)
    
    most_likely_bitstring = np.array(most_likely_bitstring)
    bitstrings = [most_likely_bitstring[i*problem_size : (i+1)*problem_size] for i in range(num_parallel)]
    
    # Assume H1, H2, ..., Hnum_parallel are the corresponding H matrices for each sub-bitstring
    bitstring_save = []
    predicted_E = []
    exact_E = []
    AR = []
    costs_save = []
    qpu_usage_time_save = []
    queue_time_save = []
    caltime_save = []
    
    # Loop over each sub-bitstring
    for i in range(num_parallel):
        x = bitstrings[i]  # Current sub-bitstring
        Qmatrix = H[i]  # Dynamically access the correct H matrix (H1, H2, ..., Hnum_parallel)
    
        # Compute predicted energy
        predicted_E_i = x@Q[i]@np.transpose(x) + np.array(cal_3_interaction(x, k_thirds[i]))
        predicted_E.append(predicted_E_i)
        
        # Exact energy (ground truth)
        exact_E_i = grounds[i]
        exact_E.append(exact_E_i)
    
        # Calculate the approximation ratio
        AR_i = predicted_E_i / exact_E_i
        AR.append(AR_i)
    
        print(f"({i+1}) Approximation ratio: {AR_i}, predicted E: {predicted_E_i}, exact E: {exact_E_i}")
    
    print(f"elapsed time: {caltime}")
    print('-'*77)

        
