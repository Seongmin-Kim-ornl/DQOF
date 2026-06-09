# Qiskit imports
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.result import QuasiDistribution
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.quantum_info import Pauli, SparsePauliOp, random_hermitian
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec, FakeBrisbane
from qiskit.circuit import ParameterVector
from qiskit import transpile
from qiskit.circuit.library import IQP
from qiskit.circuit.library import QAOAAnsatz

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
#from qiskit.primitives import Sampler

from scipy.optimize import minimize
import numpy as np
import time, warnings
warnings.simplefilter("ignore")

from .QAOA_pauli_operator import get_operator

num_layers = 2
transpile_level = 2
qpu_usage = 0
queue_elapsed = 0
queue_time = 0
iter_count = 0

service = QiskitRuntimeService(channel = "<...>",
                               instance='<...>',
                               token = "<...>",
                              )

# retrive a bitstring using quantum computer
#backend = service.least_busy()
#backend = service.backend(name="ibm_strasbourg")
#backend = service.backend(name="ibm_marrakesh")
#backend = service.backend(name="ibm_torino")
backend = AerSimulator(method='automatic') #service.backend(name="ibm_brisbane")
#backend = service.backend(name="ibm_fez")
#backend = service.backend(name="ibm_brussels")

backend_sim = AerSimulator(method='automatic')

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
        
def solve_H2(H, num_parallel):
    problem_size = len(H[0])
    qpu_usage = 0
    queue_elapsed = 0
    queue_time = 0
    iter_count = 0
    cost_hamiltonian = []
    
    for i in range(num_parallel):
        cost_ham, offset = get_operator(H[i])
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
    
    # Optional: transpile with preset pass manager
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
    # Usage
    # Assume you already have cost_hamiltonian = [cost_hamiltonian0, cost_hamiltonian1, ..., cost_hamiltonian[num_parallel-1]]
    combined_hamiltonian = combine_all_hamiltonians(cost_hamiltonian, problem_size, num_parallel)

    
    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        nonlocal qpu_usage, queue_time, caltime
        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
        pub = (ansatz, isa_hamiltonian, params)
        queue_start = time.time()
        job = estimator.run([pub])
        results = job.result()[0]
        cost = results.data.evs
        objective_func_vals.append(cost)
    
        qpu_usage += 0#job.metrics()['usage']['quantum_seconds']
        queue_elapsed = 0#time.time() - queue_start - job.metrics()['usage']['quantum_seconds']
        queue_time += 0#queue_elapsed
        caltime = time.time() - tic
        return cost
             
    initial_gamma = np.pi / 4
    initial_beta = np.pi / 8
    init_params = [initial_gamma, initial_beta]*num_layers*num_parallel
    objective_func_vals = [] # Global variable
    estimator = Estimator(mode=backend_sim)
    estimator.options.default_shots = 1e4

    tic = time.time()
    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, combined_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-4,
        options={'maxiter': 5000}
    )

    pm = generate_preset_pass_manager(optimization_level=transpile_level, backend=backend)
    candidate_circuit = pm.run(combined_circuit)
    optimized_circuit = candidate_circuit.assign_parameters(result.x)
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 1e4

    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

    queue_start = time.time()
    pub= (optimized_circuit, )
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    
    qpu_usage += job.metrics()['usage']['quantum_seconds']
    queue_elapsed = time.time() - queue_start - job.metrics()['usage']['quantum_seconds']
    queue_time += queue_elapsed
    caltime = time.time() - tic
    
    max_bitstring = max(counts_bin, key=counts_bin.get)
    #print("Most probable bitstring:", max_bitstring)
    most_likely_bitstring = [int(bit) for bit in max_bitstring]
    most_likely_bitstring.reverse()
    most_likely_bitstring = np.array(most_likely_bitstring)
    #print(f"solver2: most_likely_bitstring: {most_likely_bitstring}, qpu usage: {qpu_usage}, queue: {queue_time}, elapsed time: {caltime}")
    return most_likely_bitstring, qpu_usage, queue_time, caltime
