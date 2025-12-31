from .solve_HUBO import solve_H
from .solve_HOBO1 import solve_H1 # for AL_DQOF
from .solve_HOBO2 import solve_H2 # for AL_DQOF
from .load_HUBO import load_QUBO, load_k_third, formulate_HUBO
from .QAOA_pauli_operator import get_operator
from .HUBO_energy import cal_HUBO_energy