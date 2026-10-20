"""Tabular Q-learning implemented directly with NumPy."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    alpha: float = 0.5
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.999
    episodes: int = 12000


class QLearningAgent:
    def __init__(self, states, actions, config=Config(), seed=0):
        if not (0 < config.alpha <= 1 and 0 <= config.gamma <= 1):
            raise ValueError('Invalid learning rate or discount')
        if not (0 <= config.epsilon_min <= config.epsilon_start <= 1
                and 0 < config.epsilon_decay <= 1 and config.episodes > 0):
            raise ValueError('Invalid exploration schedule or episode count')
        self.config = config
        self.q = np.zeros((states, actions), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def action(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        # Random tie-breaking during learning avoids directional bias.
        values = self.q[state]
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def update(self, state, action, reward, next_state, terminated):
        target = reward + (0.0 if terminated else self.config.gamma * self.q[next_state].max())
        self.q[state, action] += self.config.alpha * (target - self.q[state, action])

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        table = np.load(path, allow_pickle=False)
        if table.shape != self.q.shape or not np.isfinite(table).all():
            raise ValueError('Invalid Q-table')
        self.q[:] = table
