from sir_modelling.simulation import simulate_sir_epidemics

import numpy as np

def approximate_state(state, approximation_threshold = 0.01):
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

    precision = int(1.0 / approximation_threshold)
    residual = precision - (approximated_St_percentile + approximated_It_percentile + approximated_Rt_percentile)

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

def simulate_transition(state, beta, gamma, approximation_threshold, steps_per_transition = 1):
    t, S, I, R = simulate_sir_epidemics(
        infected_people_per_day = beta,
        infection_duration = 1.0 / gamma,
        days_of_simulation = (steps_per_transition + 1),
        initial_state = state
    )

    state = ( S[-1], I[-1], R[-1] )

    return approximate_state(state, approximation_threshold)

def enumerate_states(approximation_threshold):
    # should include 0 and 1 in values (this is why we have division + 1 values)
    precision = int(1.0 / approximation_threshold)
    divisions = precision + 1

    states = []

    for s_percentile in range(0, divisions):
        for i_percentile in range(0, divisions):
            r_percentile = precision - (s_percentile + i_percentile)
            if r_percentile < 0: continue

            susceptibles = s_percentile * approximation_threshold
            infective = i_percentile * approximation_threshold
            recovered = r_percentile * approximation_threshold

            states.append( (susceptibles, infective, recovered) )

    return states

def enumerate_transitions(approximation_threshold, beta, gamma, states, steps_per_transition):
    transitions_for_beta = []

    for state in states:
        next_state = simulate_transition(state, beta, gamma, approximation_threshold, steps_per_transition)
        transitions_for_beta.append( (state, next_state, 1.0) )

    return transitions_for_beta

def enumerate_reward(approximation_threshold, beta, states, reward_function):
    reward_per_state = []

    for state in states:
        susceptibles, infective, recovered = state

        reward = reward_function(susceptibles, infective, recovered) / approximation_threshold
        reward_per_state.append( (state, reward) )

    return reward_per_state

def create_representation(approximation_threshold, gamma, betas, steps_per_transition = 1, reward_function = None):
    if reward_function is None:
        reward_function = lambda susceptibles, infective, recovered: 10 * susceptibles + 5 * recovered - 15 * infective

    states = enumerate_states(approximation_threshold)

    transitions_per_beta = {}
    rewards_per_beta = {}

    # for each beta discover deterministic transitions
    for beta in betas:
        transitions_per_beta[beta] = enumerate_transitions(approximation_threshold, beta, gamma, states, steps_per_transition)
        rewards_per_beta[beta] = enumerate_reward(approximation_threshold, beta, states, reward_function)

    return states, transitions_per_beta, rewards_per_beta
