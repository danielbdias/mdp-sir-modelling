import numpy as np

from sir_modelling.enumerative_model import get_state_numeric_values

def sample_state(mdp, state, action):
    sampled_probability = np.random.random_sample()
    cummulative_probability = 0.0

    state_index = mdp.states.index(state)
    transition_matrix = mdp.transition_matrix(action)

    indexes = np.nonzero(transition_matrix[state_index,:] > 0)

    for next_state_index in indexes:
        index = next_state_index[0]

        probability = transition_matrix[state_index, index]
        cummulative_probability = cummulative_probability + probability
        if sampled_probability < cummulative_probability:
            return mdp.states[index]

    return mdp.states[indexes[-1]]

def simulate_policy_with_mdp_model(policy, initial_state, mdp, horizon, approximation_threshold):
    state_name = initial_state
    visited_states = [ get_state_numeric_values(initial_state, approximation_threshold) ]
    chosen_betas = []

    for i in range(horizon):
        beta = policy[state_name]
        state_name = sample_state(mdp, state_name, beta)

        chosen_betas.append(beta)
        visited_states.append(get_state_numeric_values(state_name, approximation_threshold))

    S = list(map(lambda state: state[0], visited_states))
    I = list(map(lambda state: state[1], visited_states))
    R = list(map(lambda state: state[2], visited_states))

    return chosen_betas, S, I, R
