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

# retrive a bitstring using quantum computer
backend = AerSimulator(method='automatic') 
#backend = service.backend(name="ibm_fez")
backend_sim = AerSimulator(method='automatic')

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
        
def solve_H(HUBO):
    problem_size = len(HUBO[0])
    qpu_usage = 0
    queue_elapsed = 0
    queue_time = 0
    iter_count = 0
    cost_hamiltonian = []

    cost_hamiltonian, offset = get_operator(HUBO)
    
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=num_layers)
    circuit.measure_all()
    
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
    
        qpu_usage += job.metrics()['usage']['quantum_seconds']
        queue_elapsed = time.time() - queue_start - job.metrics()['usage']['quantum_seconds']
        queue_time += queue_elapsed
        caltime = time.time() - tic
        return cost
    
    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(optimization_level=2,
                                        backend=backend)
    
    candidate_circuit = pm.run(circuit)
    
    #run QAOA
    initial_gamma = np.pi / 4
    initial_beta = np.pi / 8
    init_params = [initial_gamma, initial_beta] * num_layers
    objective_func_vals = [] # Global variable
    
    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 1e4
    
    # Set simple error suppression/mitigation options
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"
    
    tic = time.time()
    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-4,
        options={'maxiter': 200}
    )      
    optimized_circuit = candidate_circuit.assign_parameters(result.x) 

    # retrive a bitstring
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    
    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
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
    shots = sum(counts_int.values())
    final_distribution_int = {key: val/shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
    
    qpu_usage += job.metrics()['usage']['quantum_seconds']
    queue_elapsed = time.time() - queue_start - job.metrics()['usage']['quantum_seconds']
    queue_time += queue_elapsed
    caltime = time.time() - tic

    max_bitstring = max(counts_bin, key=counts_bin.get)
    most_likely_bitstring = [int(bit) for bit in max_bitstring]
    most_likely_bitstring.reverse()
    most_likely_bitstring = np.array(most_likely_bitstring)
    return most_likely_bitstring