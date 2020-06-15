from pyddlib.add import ADD
from sir_modelling.sir_model import compartments_derivative

import numpy as np

def simulate_transition(state, beta, gamma):
  St, It, Rt = state
  dSdt, dIdt, dRdt = compartments_derivative(state, beta, gamma)

  return ( St + dSdt, It + dIdt, Rt + dRdt )

def approximate_value(value, approximation_threshold):
    # TODO: melhorar essa lógica para casos em que o threshold não é multiplo de 10
    digits = int(np.log10(approximation_threshold))
    return np.around(value, digits)

def approximate_state(state, approximation_threshold = 0.01):
    St, It, Rt = state

    approximated_St = approximate_value(St, approximation_threshold)
    approximated_It = approximate_value(It, approximation_threshold)
    approximated_Rt = approximate_value(Rt, approximation_threshold)

    return (approximated_St, approximated_It, approximated_Rt)

def enumerate_states(approximation_threshold):
    # should include 0 and 1 in values (this is why we have division + 1 values)
    divisions = int((1.0 / approximation_threshold) + 1.0)
    variable_values = np.linspace(0.0, 1.0, num=divisions)

    states = []

    for s_index in range(0, divisions):
        for i_index in range(0, divisions):
            for r_index in range(0, divisions):
                susceptibles = variable_values[s_index]
                infective = variable_values[i_index]
                recovered = variable_values[r_index]

                if (susceptibles + infective + recovered) == 1.0:
                    states.append( (susceptibles, infective, recovered) )

    return states

def enumerate_transitions(approximation_threshold, gamma, beta, states):
    transitions_for_beta = []

    for state in states:
        next_state = simulate_transition(state, beta, gamma)
        approximated_next_state = approximate_state(next_state, approximation_threshold)
        transitions_for_beta.append( (state, approximated_next_state, 1.0) )

    return transitions_for_beta

def enumerate_reward(approximation_threshold, states):
    reward_per_state = []

    for state in states:
        susceptibles, infective, recovered = state

        reward = (10 * susceptibles + 5 * recovered - 15 * infective) / approximation_threshold
        reward_per_state.append( (state, reward) )

    return reward_per_state

def create_enumerative_representation(approximation_threshold, gamma, betas):
    states = enumerate_states(approximation_threshold)

    transitions_per_beta = {}

    # for each beta discover deterministic transitions
    for beta in betas:
        transitions_per_beta[beta] = enumerate_transitions(approximation_threshold, gamma, beta, states)

    rewards = enumerate_reward(approximation_threshold, states)

    return states, transitions_per_beta, rewards

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

approximation_threshold = 0.01

# beta (infection rate)
# assumption
# beta < 1.0 - social distancing
# beta >= 1.0 - no social distancing
betas = [0.5, 1.0]

# recovery rate (1 person each 4 days)
gamma = 1.0 / 4.0

states, transitions_per_beta, rewards = create_enumerative_representation(approximation_threshold, gamma, betas)

# Results

print(f"states: {len(states)}")

for beta in betas:
    print(f"action {beta}: {len(transitions_per_beta[beta])}")

print(f"rewards: {len(rewards)}")
