from pyddlib.add import ADD
import numpy as np

def value_to_variable_string(prefix, value, approximation_threshold):
    int_value = int(value / approximation_threshold)
    return f"#{prefix}_#{int_value:02d}"

def state_to_string(state, approximation_threshold, prime=False):
    susceptibles, infective, recovered = state

    prime_suffix = ""

    if prime:
        prime_suffix = "_prime"

    susceptibles_var = value_to_variable_string(f"s{prime_suffix}", susceptibles, approximation_threshold)
    infective_var = value_to_variable_string(f"i{prime_suffix}", infective, approximation_threshold)

    return susceptibles_var, infective_var

def create_var_to_id_hash(approximation_threshold):
    # should include 0 and 1 in values (this is why we have division + 1 values)
    divisions = int((1.0 / approximation_threshold) + 1.0)
    variable_values = np.linspace(0.0, 1.0, num=divisions)

    var_to_id = {}
    index = 0

    # ommiting "r" because it can be deduced from "s" and "i"
    for prefix in [ "s", "i", "s_prime", "i_prime" ]:
        for value in variable_values:
            variable_name = value_to_variable_string(prefix, value, approximation_threshold)
            var_to_id[variable_name] = index
            index = index + 1

    return var_to_id

def create_adds_for_transition(transition, var_to_id, approximation_threshold):
    from_state, to_state, probability = transition

    susceptibles_var, infective_var = state_to_string(from_state, approximation_threshold)
    susceptibles_prime_var, infective_prime_var = state_to_string(to_state, approximation_threshold)

    susceptibles_var_index, infective_var_index = var_to_id[susceptibles_var], var_to_id[infective_var]
    susceptibles_prime_var_index, infective_prime_var_index = var_to_id[susceptibles_prime_var], var_to_id[infective_prime_var]

    susceptible_add = ADD.variable(susceptibles_var_index) * ADD.variable(infective_var_index) * ADD.variable(susceptibles_prime_var_index) * ADD.constant(probability)
    infective_add = ADD.variable(susceptibles_var_index) * ADD.variable(infective_var_index) * ADD.variable(infective_prime_var_index) * ADD.constant(probability)

    return {
        susceptibles_prime_var: susceptible_add,
        infective_prime_var: infective_add
    }

def factored_transitions(transitions, var_to_id, approximation_threshold):
    # para cada transição
    #  cria os adds parciais de transição
    #  atualiza os adds das variáveis
    pass

def factored_reward(rewards, var_to_id, approximation_threshold):
    reward_add = ADD.constant(0.0)

    for state, reward in rewards:
        susceptibles_var, infective_var = state_to_string(state, approximation_threshold)
        susceptibles_var_index, infective_var_index = var_to_id[susceptibles_var], var_to_id[infective_var]

        partial_add = ADD.variable(susceptibles_var_index) * ADD.variable(infective_var_index) * ADD.constant(reward)
        reward_add = reward_add + partial_add

    return reward_add

# TODO:
# como resolver a questão dos intervalos?
# será que precisamos integrar / fazer soma acumulada nos intervalos?

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
betas = [0.5, 1.0, 2.5, 4]

# recovery rate (1 person each 4 days)
gamma = 1.0 / 4.0

print("Building SIR enumerative representation")
print()

start_time = time.perf_counter()
mdp = create_representation(approximation_threshold, gamma, betas)
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

initial_state = "s_99_i_01_r_00"

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
S, I, R = simulate_policy(policy, initial_state=initial_state, mdp=mdp, horizon=30, approximation_threshold=approximation_threshold)
elapsed_time = time.perf_counter() - start_time

print(S)
print(I)
print(R)

print(f"Elapsed time: {elapsed_time:0.4f} seconds")
