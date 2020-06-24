from scipy.integrate import odeint

import matplotlib.pyplot as plt
import numpy as np

def suscetibles_derivative(compartments, beta, gamma):
    S, I, R = compartments
    return -beta * S * I

def infected_derivative(compartments, beta, gamma):
    S, I, R = compartments
    return beta * S * I - gamma * I

def recovered_derivative(compartments, beta, gamma):
    S, I, R = compartments
    return gamma * I

def compartments_derivative(compartments, beta, gamma):
    dSdt = suscetibles_derivative(compartments, beta, gamma)
    dIdt = infected_derivative(compartments, beta, gamma)
    dRdt = recovered_derivative(compartments, beta, gamma)
    return dSdt, dIdt, dRdt

def simulate_sir_epidemics(population_size, infected_people_per_day, infection_duration, days_of_simulation, initial_state = None):
    beta = infected_people_per_day
    gamma = 1.0 / infection_duration

    if initial_state is None:
        S0 = (population_size - 1.0) / population_size
        I0 = 1.0 / population_size
        R0 = 0.0
        initial_state = S0, I0, R0

    t = np.linspace(0, days_of_simulation - 1, days_of_simulation)

    # solve ODEs
    derivative = lambda compartments, t, beta, gamma: compartments_derivative(compartments, beta, gamma)

    ret = odeint(derivative, initial_state, t, args=(beta, gamma))
    S, I, R = ret.T # arrays with compartment variation through days

    return t, S, I, R

def plot_sir(t, S, I, R):
  f, ax = plt.subplots(1,1,figsize=(10,4))
  ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
  ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
  ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')

  ax.set_xlabel('Time (days)', labelpad=10)
  ax.set_ylabel('Population (%)', labelpad=10)

  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)

  ax.grid(b=True, which='major', c='w', lw=2, ls='-')

  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)

  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)

  plt.show()
