# TODO: convert into a horizon oriented algorithn
# TODO: test algorithm

import numpy as np

def compute_maximum_residual(mdp, first_value_function, second_value_function):
    state_residual = lambda state: abs(first_value_function[state] - second_value_function[state])
    residuals = map(state_residual, mdp.states)
    return max(residuals)

def compute_quality(state_index, action, mdp, gamma, value_function):
    transition_matrix = mdp.transition_matrix(action)
    reward_matrix = mdp.reward_matrix(action)

    pondered_sum = transition_matrix[state_index].dot(value_function)

    return reward_matrix[state_index, 0] + gamma * pondered_sum

def compute_bellman_backup(state_index, mdp, gamma, value_function):
    qualities = []

    for action in mdp.actions:
        qualities.append(
            compute_quality(state_index, action, mdp, gamma, value_function)
        )

    return max(qualities)

def compute_greedy_action(state_index, mdp, gamma, value_function):
    policy = {}

    best_action = None
    max_value_for_best_action = float("-inf")

    for action in mdp.actions:
        quality = compute_quality(state_index, action, mdp, gamma, value_function)

        if quality > max_value_for_best_action:
            max_value_for_best_action = quality
            best_action = action

    return best_action

def compute_policy(mdp, gamma, value_function):
    policy = {}

    for state_index, state_name in enumerate(mdp.states):
        policy[state_name] = compute_greedy_action(state_index, mdp, gamma, value_function)

    return policy

def residual(state_index, mdp, gamma, value_function):
    action = compute_greedy_action(state_index, mdp, gamma, value_function)
    quality = compute_quality(state_index, action, mdp, gamma, value_function)

    return abs(value_function[state_index] - quality)

def reachable_states(mdp, state_index, action):
    state_indexes = []

    transition_matrix = mdp.transition_matrix(action)

    indexes = np.nonzero(transition_matrix[state_index,:] > 0)

    for next_state_index in indexes:
        state_indexes.append(next_state_index[0])

    return state_indexes

def sample_state(mdp, state_index, action):
    sampled_probability = np.random.random_sample()
    cummulative_probability = 0.0

    transition_matrix = mdp.transition_matrix(action)

    indexes = np.nonzero(transition_matrix[state_index,:] > 0)

    for next_state_index in indexes:
        index = next_state_index[0]

        probability = transition_matrix[state_index, index]
        cummulative_probability = cummulative_probability + probability
        if sampled_probability < cummulative_probability:
            return index

    return indexes[-1]

def check_solved(root_state_index, epsilon, solved_states, mdp, gamma, value_function):
    solved = True
    open_states = []
    closed_states = []
    bellman_backups_done = 0

    if root_state_index not in solved_states:
        open_states.append(root_state_index)

    while len(open_states) != 0:
        state_index = open_states.pop()
        closed_states.append(state_index)

        if residual(state_index, mdp, gamma, value_function) > epsilon:
            solved = False
            continue

        action = compute_greedy_action(state_index, mdp, gamma, value_function)

        for next_state_index in reachable_states(mdp, state_index, action):
            is_solved = next_state_index in solved_states
            is_open = next_state_index in open_states
            is_closed = next_state_index in closed_states

            if not (is_solved or is_closed or is_open):
                open_states.append(next_state_index)

    if solved:
        for closed_state_index in closed_states:
            solved_states.append(closed_state_index)
    else:
        while len(closed_states) != 0:
            closed_state_index = closed_states.pop()

            value_function[closed_state_index] = compute_bellman_backup(closed_state_index, mdp, gamma, value_function)
            bellman_backups_done = bellman_backups_done + 1

    return solved, bellman_backups_done

def enumerative_lrtdp(mdp, gamma, max_depth, epsilon, initial_state, goal_states, seed = None):
    """Executes the Labeled Real Time Dynamic Programming algorithm.
    Parameters:
    mdp (EnumerativeMDP): enumerative Markov Decison Problem to be solved
    gamma (float): discount factor applied to solve this MDP (assumes infinite on indefinite horizon)
    max_depth (int): max depth to search (used to avoid infinite loops on deadends)
    epsilon (float): maximum residual allowed between V_k and V_{k+1}
    initial_state (string): MDP initial state
    goal_states (list of string): MDP goal states
    seed (int): optional seed used to initialize random number generator
    Returns:
    policy (dict): resulting policy computed for a mdp, represented as a dict that maps a state to an action
    value_function (list): value function found by this algorithm, represented as list with values w.r.t mdp.states
    statistics (dict): dictionary containing some statistics about the algorithm execution. We have three statistics here:
                      "iterations" that is equal to the horizon parameter, "bellman_backups_done" that is the overall
                      number of Bellman backups executed and "maximum_residuals" that is the maximum residual found in
                      each iteration.
    """
    if seed is not None:
        np.random.seed(seed)

    value_function = np.zeros(( len(mdp.states), 1 ))

    goal_state_indexes = list(map(lambda goal_state: mdp.states.index(goal_state), goal_states))
    initial_state_index = mdp.states.index(initial_state)

    solved_states = []
    bellman_backups_done = 0
    trials = 0
    maximum_residuals = []

    while (initial_state_index not in solved_states):
        trials = trials + 1
        visited_states = []

        state_index = initial_state_index

        while (state_index not in solved_states):
            visited_states.append(state_index)

            if state_index in goal_state_indexes:
                break

            value_function[state] = compute_bellman_backup(state_index, mdp, gamma, value_function)
            bellman_backups_done = bellman_backups_done + 1

            next_action = compute_greedy_action(state_index, mdp, gamma, value_function)
            state = sample_state(mdp, state_index, next_action)

            if len(visited_states) > max_depth:
                break

        while len(visited_states) != 0:
            state_index = visited_states.pop()

            solved, bellman_backups = check_solved(state_index, epsilon, solved_states, mdp, gamma, value_function)
            bellman_backups_done = bellman_backups_done + bellman_backups

            if not solved:
                break

        # keep initial state residual
        maximum_residuals.append(max(residual(initial_state_index, mdp, gamma, value_function)))

    # compute policy
    policy = compute_policy(mdp, gamma, value_function)

    statistics = {
        "iterations": trials,
        "bellman_backups_done": bellman_backups_done,
        "maximum_residuals": maximum_residuals
    }

    return policy, value_function, statistics
