from collections import namedtuple
from sir_modelling.base_model import create_representation as create_base_representation

import numpy as np

MDP = namedtuple("MarkovDecisionProcess", ["states", "actions", "transition_matrix", "reward_matrix"])

def get_variable_string(prefix, value, approximation_threshold):
    precision = 1.0 / approximation_threshold
    int_value = int(np.around(value * precision))

    format_digits = int(abs(np.log10(approximation_threshold)))
    format_string = f"0{format_digits}d"

    formatted_number = ("{:" + format_string + "}").format(int_value)

    return f"{prefix}_{formatted_number}"

def get_single_human_readable_state(state, approximation_threshold):
    susceptibles, infective, recovered = state

    susceptibles_var = get_variable_string("s", susceptibles, approximation_threshold)
    infective_var = get_variable_string("i", infective, approximation_threshold)
    recovered_var = get_variable_string("r", recovered, approximation_threshold)

    return f"{susceptibles_var}_{infective_var}_{recovered_var}"

def get_state_numeric_values(human_readable_state, approximation_threshold):
    values_string = human_readable_state.replace("s_", "").replace("i_", "").replace("r_", "")
    values = list(map(lambda value: int(value) * approximation_threshold, values_string.split("_")))
    return values[0], values[1], values[2]

def get_human_readable_states(states, approximation_threshold):
    human_readable_states = []

    for state in states:
        human_readable_states.append(get_single_human_readable_state(state, approximation_threshold))

    return sorted(human_readable_states)

def get_reward_function(rewards_per_beta, indexed_states, approximation_threshold):
    reward_function = {}

    for beta in rewards_per_beta.keys():
        reward_list = np.zeros(( len(indexed_states), 1 ))

        for state, reward in rewards_per_beta[beta]:
            human_readable_state = get_single_human_readable_state(state, approximation_threshold)
            state_index = indexed_states.get(human_readable_state)
            reward_list[state_index] = reward

        reward_function[beta] = reward_list

    return reward_function

def get_transition_matrices(approximation_threshold, indexed_states, transitions_per_beta):
    transition_matrix_per_beta = {}

    for beta in transitions_per_beta.keys():
        number_of_states = len(indexed_states)
        transition_matrix = np.zeros((number_of_states, number_of_states))

        for from_state, to_state, probability in transitions_per_beta[beta]:
            human_readable_from_state = get_single_human_readable_state(from_state, approximation_threshold)
            human_readable_to_state = get_single_human_readable_state(to_state, approximation_threshold)

            from_state_index = indexed_states.get(human_readable_from_state)
            to_state_index = indexed_states.get(human_readable_to_state)

            transition_matrix[from_state_index][to_state_index] = probability

        transition_matrix_per_beta[beta] = transition_matrix

    return transition_matrix_per_beta

def get_indexed_states(states):
    indexed_states = {}

    for index, state in enumerate(states):
        indexed_states[state] = index

    return indexed_states

def create_representation(approximation_threshold, gamma, betas, steps_per_transition = 1, reward_function = None):
    states, transitions_per_beta, rewards_per_beta = create_base_representation(approximation_threshold, gamma, betas, steps_per_transition, reward_function)

    human_readable_states = get_human_readable_states(states, approximation_threshold)

    human_readable_indexed_states = get_indexed_states(human_readable_states)

    transition_matrix_per_beta = get_transition_matrices(approximation_threshold, human_readable_indexed_states, transitions_per_beta)
    reward_matrix_per_beta = get_reward_function(rewards_per_beta, human_readable_indexed_states, approximation_threshold)

    return MDP(
        states = human_readable_states,
        actions = betas,
        transition_matrix = lambda action: transition_matrix_per_beta[action],
        reward_matrix = lambda action: reward_matrix_per_beta[action]
    )
