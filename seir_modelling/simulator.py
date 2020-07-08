from seir_modelling.seir import seir

class Simulator:
    def __init__(self, initial_state, r0_values, days_per_action):
        self.initial_state = initial_state
        self.actions = r0_values
        self.days_per_action = days_per_action

        self.inner_simulator = seir(days_per_action)

        self.start()

    def start(self):
        self.current_state = self.initial_state.copy()
        return self.current_state

    def simulate_action(self, action):
        self.inner_simulator.R0 = action

        S, E, I, R = self.inner_simulator.run(self.current_state)
        return (S[-1], E[-1], I[-1], R[-1])

    def execute_action(self, action):
        self.current_state = self.simulate_action(action)
        return self.current_state

    def reward(self, state):
        S, E, I, R = state
        return 1.0 - I
