import numpy as np

##############################################################################
# Script start
##############################################################################

from sir_modelling.enumerative_model import create_representation, simulate_policy
from mdp.algorithms.value_iteration import enumerative_finite_horizon_value_iteration

import time

approximation_threshold = 0.01

# beta (infection rate)
# assumption
# beta < 1.0 - social distancing
# beta >= 1.0 - no social distancing
# betas = [0.5, 1.0, 2.5, 4]
betas = [0.5, 1.0, 2.5, 4]

# recovery rate (1 person each 4 days)
gamma = 1.0 / 4.0

steps_per_transition = 7
initial_state = "s_99_i_01_r_00"

print("Building SIR enumerative representation")
print()

start_time = time.perf_counter()
reward_function = lambda susceptibles, infective, recovered: 10 * susceptibles + 5 * recovered - 15 * infective
mdp = create_representation(approximation_threshold, gamma, betas, steps_per_transition, reward_function)
elapsed_time = time.perf_counter() - start_time

print(f"states: {len(mdp.states)}, {mdp.states[0:5]}")
print(f"actions: {mdp.actions}")

first_action = mdp.actions[0]

print(f"reward: {mdp.reward_matrix(first_action)[0, 0]}")

print(f"transition: {mdp.transition_matrix(first_action)[0, 0]}")
print(f"transition: {mdp.transition_matrix(first_action)[0, 1]}")
print(f"Elapsed time: {elapsed_time:0.4f} seconds")

print()
print("Running VI for representation")
print()

start_time = time.perf_counter()
policy, value_function, statistics = enumerative_finite_horizon_value_iteration(mdp, 0.9, horizon=30)
elapsed_time = time.perf_counter() - start_time

print(f"statistics: {statistics}")
print(f"max value: {max(value_function)}")
print(f"policy for first state: {policy[mdp.states[0]]}")
print(f"policy for initial state: {policy[initial_state]}")
print(f"min beta in policy: {min(policy.values())}")
print(f"max beta in policy: {max(policy.values())}")
print(f"Elapsed time: {elapsed_time:0.4f} seconds")

print()
print("Simulate policy")
print()

start_time = time.perf_counter()
chosen_betas, S, I, R = simulate_policy(policy, initial_state=initial_state, mdp=mdp, horizon=30, approximation_threshold=approximation_threshold)
elapsed_time = time.perf_counter() - start_time

print(chosen_betas)
print(S)
print(I)
print(R)

print(f"Elapsed time: {elapsed_time:0.4f} seconds")
