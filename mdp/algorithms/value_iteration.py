import numpy as np

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

def compute_policy(mdp, gamma, value_function):
    policy = {}

    for state_index, state_name in enumerate(mdp.states):
        max_value = float("-inf")
        best_action = None

        for action in mdp.actions:
            quality = compute_quality(state_index, action, mdp, gamma, value_function)

            if quality > max_value:
                max_value = quality
                best_action = action

        policy[state_name] = best_action

    return policy

def enumerative_finite_horizon_value_iteration(mdp, gamma, horizon):
    """Executes the Value Iteration algorithm for finite horizon MDPs.
    Parameters:
    mdp (EnumerativeMDP): enumerative Markov Decison Problem to be solved
    gamma (float): discount factor applied to solve this MDP
    horizon (int): number of steps that can be done in this MDP
    Returns:
    policy (dict): resulting policy computed for a mdp, represented as a dict that maps a state to an action
    value_function (list): value function found by this algorithm, represented as list with values w.r.t mdp.states
    statistics (dict): dictionary containing some statistics about the algorithm execution. We have two statistics here:
                      "iterations" that is equal to the horizon parameter and "bellman_backups_done" that is the overall
                      number of Bellman backups executed.
    """
    last_horizon_value_function = np.zeros(( len(mdp.states), 1 ))

    bellman_backups_done = 0

    for n in range(horizon - 1, -1, -1): # range from H - 1 to 0
        current_horizon_value_function = np.zeros(( len(mdp.states), 1 ))

        # do bellman update
        for state_index, state_name in enumerate(mdp.states):
            current_horizon_value_function[state_index] = compute_bellman_backup(state_index, mdp, gamma, last_horizon_value_function)
            bellman_backups_done = bellman_backups_done + 1

        last_horizon_value_function = current_horizon_value_function

    # compute policy
    policy = compute_policy(mdp, gamma, last_horizon_value_function)

    statistics = {
        "iterations": horizon,
        "bellman_backups_done": bellman_backups_done
    }

    return policy, last_horizon_value_function, statistics
