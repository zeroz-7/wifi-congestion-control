import numpy as np

class QLearningAgent:
    """
    Discrete Q-learning agent.
    State: encoded integer id
    Action: index into an action list (here, delta CWmin actions)
    """

    def __init__(self, n_states: int, n_actions: int,
                 lr: float = 0.1, gamma: float = 0.95,
                 eps_start: float = 1.0, eps_end: float = 0.1, eps_decay: float = 0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def select_action(self, s: int) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s2: int, done: bool):
        best_next = 0.0 if done else np.max(self.Q[s2])
        td_target = r + self.gamma * best_next
        self.Q[s, a] += self.lr * (td_target - self.Q[s, a])

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
