from mdp.algorithms.lrtdp_simulator import enumerative_lrtdp_with_simulator
from seir_modelling.simulator import Simulator

gamma = 0.9
max_depth = 1000
epsilon = 0.0001

simulator = Simulator(
    initial_state=(0.999, 0.0, 0.001, 0.0),
    r0_values=[1.8, 1.6, 1.0, 0.8],
    days_per_action=7
)

policy, value_function, statistics = enumerative_lrtdp_with_simulator(simulator, gamma, max_depth, epsilon)
print("aha!")
print(statistics)
print(policy)