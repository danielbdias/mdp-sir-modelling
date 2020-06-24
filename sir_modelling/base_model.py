from sir_modelling.simulation import compartments_derivative

import numpy as np

def simulate_transition(state, beta, gamma):
  St, It, Rt = state
  dSdt, dIdt, dRdt = compartments_derivative(state, beta, gamma)

  return ( St + dSdt, It + dIdt, Rt + dRdt )

def approximate_state(state, states, approximation_threshold = 0.01):
    St, It, Rt = state

    # use integers to avoid floating point problems
    approximated_St_percentile = int(np.trunc(St / approximation_threshold))
    approximated_It_percentile = int(np.trunc(It / approximation_threshold))
    approximated_Rt_percentile = int(np.trunc(Rt / approximation_threshold))

    if approximated_St_percentile < 0:
        # suscetible derivative can return a negative value sometimes
        # in these cases assumes 0 for St and add it on infected compartment
        approximated_It_percentile += (-1 * approximated_St_percentile)
        approximated_St_percentile = 0

    residual = 100 - (approximated_St_percentile + approximated_It_percentile + approximated_Rt_percentile)

    if residual != 0:
        # sometimes we can have a residual due the truncation procedure
        # in that case assume the residual on chain start
        if approximated_St_percentile == 0:
            approximated_It_percentile += residual
        else:
            approximated_St_percentile += residual

    return (
        approximated_St_percentile * approximation_threshold,
        approximated_It_percentile * approximation_threshold,
        approximated_Rt_percentile * approximation_threshold
    )

def enumerate_states(approximation_threshold):
    # should include 0 and 1 in values (this is why we have division + 1 values)
    divisions = int((1.0 / approximation_threshold) + 1.0)

    states = []

    for s_percentile in range(0, divisions):
        for i_percentile in range(0, divisions):
            for r_percentile in range(0, divisions):
                # make sum as integer to avoid floating point comparison
                if s_percentile + i_percentile + r_percentile == 100:
                    susceptibles = s_percentile * approximation_threshold
                    infective = i_percentile * approximation_threshold
                    recovered = r_percentile * approximation_threshold

                    states.append( (susceptibles, infective, recovered) )

    return states

def enumerate_transitions(approximation_threshold, gamma, beta, states):
    transitions_for_beta = []

    for state in states:
        next_state = simulate_transition(state, beta, gamma)
        approximated_next_state = approximate_state(next_state, states, approximation_threshold)
        transitions_for_beta.append( (state, approximated_next_state, 1.0) )

    return transitions_for_beta

def enumerate_reward(approximation_threshold, beta, states, reward_function = None):
    if reward_function is None:
        reward_function = lambda susceptibles, infective, recovered: 10 * susceptibles + 5 * recovered - 15 * infective

    reward_per_state = []

    for state in states:
        susceptibles, infective, recovered = state

        reward = reward_function(susceptibles, infective, recovered) / approximation_threshold
        reward_per_state.append( (state, reward) )

    return reward_per_state

def create_representation(approximation_threshold, gamma, betas, reward_function = None):
    states = enumerate_states(approximation_threshold)

    transitions_per_beta = {}
    rewards_per_beta = {}

    # for each beta discover deterministic transitions
    for beta in betas:
        transitions_per_beta[beta] = enumerate_transitions(approximation_threshold, gamma, beta, states)
        rewards_per_beta[beta] = enumerate_reward(approximation_threshold, beta, states, reward_function)

    return states, transitions_per_beta, rewards_per_beta
