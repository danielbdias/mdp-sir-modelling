import numpy as np

def discretize_state(state, truncate_digits = 4):
    values = []
    for state_value in state:
        truncated_state_value = int(np.trunc(state_value * (10 ** truncate_digits)))
        values.append(str(truncated_state_value))
    return "_".join(values)

def recreate_state(discretized_state, truncate_digits = 4):
    splitted_state = discretized_state.split("_")
    return list(map(lambda item: int(item) / (10 ** truncate_digits), splitted_state))

def is_goal(discretized_state, simulator):
    state = recreate_state(discretized_state)
    return simulator.is_goal(state)

def compute_quality(state, action, simulator, gamma, value_function):
    reward = simulator.reward(recreate_state(state))
    next_state = discretize_state(simulator.simulate_action(action))

    return reward + gamma * value_function.get(next_state, 0.0)

def compute_bellman_backup(state, simulator, gamma, value_function):
    qualities = []

    for action in simulator.actions:
        qualities.append(
            compute_quality(state, action, simulator, gamma, value_function)
        )

    return max(qualities)

def compute_greedy_action(state, simulator, gamma, value_function):
    policy = {}

    best_action = None
    max_value_for_best_action = float("-inf")

    for action in simulator.actions:
        quality = compute_quality(state, action, simulator, gamma, value_function)

        if quality > max_value_for_best_action:
            max_value_for_best_action = quality
            best_action = action

    return best_action

def compute_policy(simulator, gamma, value_function):
    policy = {}

    for state in value_function.keys():
        policy[state] = compute_greedy_action(state, simulator, gamma, value_function)

    return policy

def residual(state, simulator, gamma, value_function):
    action = compute_greedy_action(state, simulator, gamma, value_function)
    quality = compute_quality(state, action, simulator, gamma, value_function)

    return abs(value_function.get(state, 0.0) - quality)

def check_solved(root_state, epsilon, solved_states, simulator, gamma, value_function):
    solved = True
    open_states = []
    closed_states = []
    bellman_backups_done = 0

    if root_state not in solved_states:
        open_states.append(root_state)

    while len(open_states) != 0:
        state = open_states.pop()
        closed_states.append(state)

        if residual(state, simulator, gamma, value_function) > epsilon:
            solved = False
            continue

        action = compute_greedy_action(state, simulator, gamma, value_function)
        next_state = discretize_state(simulator.simulate_action(action))

        is_solved = next_state in solved_states
        is_open = next_state in open_states
        is_closed = next_state in closed_states

        if not (is_solved or is_closed or is_open):
            open_states.append(next_state)

    if solved:
        for closed_state in closed_states:
            solved_states.append(closed_state)
    else:
        while len(closed_states) != 0:
            closed_state = closed_states.pop()

            value_function[closed_state] = compute_bellman_backup(closed_state, simulator, gamma, value_function)
            bellman_backups_done = bellman_backups_done + 1

    return solved, bellman_backups_done

def enumerative_lrtdp_with_simulator(simulator, gamma, max_depth, epsilon):
    value_function = {}

    solved_states = []
    bellman_backups_done = 0
    trials = 0
    maximum_residuals = []

    initial_state = discretize_state(simulator.start())

    while (initial_state not in solved_states):
        trials = trials + 1
        visited_states = []

        state = initial_state

        while (state not in solved_states):
            visited_states.append(state)

            if is_goal(state, simulator):
                break

            value_function[state] = compute_bellman_backup(state, simulator, gamma, value_function)
            bellman_backups_done = bellman_backups_done + 1

            next_action = compute_greedy_action(state, simulator, gamma, value_function)
            state = discretize_state(simulator.execute_action(next_action))

            if len(visited_states) > max_depth:
                break

        while len(visited_states) != 0:
            state = visited_states.pop()

            solved, bellman_backups = check_solved(state, epsilon, solved_states, simulator, gamma, value_function)
            bellman_backups_done = bellman_backups_done + bellman_backups

            if not solved:
                break

        # keep initial state residual
        #maximum_residuals.append(max(residual(initial_state, simulator, gamma, value_function)))

    # compute policy
    policy = compute_policy(simulator, gamma, value_function)

    statistics = {
        "iterations": trials,
        "bellman_backups_done": bellman_backups_done,
        "maximum_residuals": maximum_residuals
    }

    return policy, value_function, statistics
