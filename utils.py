"""Training, independent policy evaluation, plots and GIF rendering."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from agent import Config, QLearningAgent


def evaluate(q=None, seeds=range(30000, 30500), random_seed=123):
    """Greedy deterministic evaluation; fixed initial states for fair comparisons."""
    env = gym.make('Taxi-v3')
    rng = np.random.default_rng(random_seed)
    rows = []
    try:
        for seed in seeds:
            state, _ = env.reset(seed=int(seed))
            total = illegal = 0
            for step in range(1, 201):
                action = int(rng.integers(6)) if q is None else int(np.argmax(q[state]))
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                illegal += reward == -10
                if terminated or truncated:
                    break
            rows.append((total, step, int(terminated), illegal))
    finally:
        env.close()
    data = np.asarray(rows, dtype=float)
    return dict(mean_return=float(data[:, 0].mean()),
                std_return=float(data[:, 0].std()),
                mean_steps=float(data[:, 1].mean()),
                success_rate=float(data[:, 2].mean()),
                mean_illegal=float(data[:, 3].mean()), episodes=len(rows))


def train(config=Config(), seed=0, checkpoint_every=1000):
    env = gym.make('Taxi-v3')
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, config, seed)
    history, checkpoints = [], []
    try:
        for episode in range(config.episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            epsilon = max(config.epsilon_min, config.epsilon_start * config.epsilon_decay ** episode)
            total = 0
            for step in range(1, 201):
                action = agent.action(state, epsilon)
                nxt, reward, terminated, truncated, _ = env.step(action)
                # A time limit ends the rollout, but does not erase the bootstrap.
                agent.update(state, action, reward, nxt, terminated)
                state = nxt
                total += reward
                if terminated or truncated:
                    break
            history.append((episode + 1, total, step, int(terminated), epsilon))
            if (episode + 1) % checkpoint_every == 0 or episode + 1 == config.episodes:
                metrics = evaluate(agent.q, range(20000, 20100))
                checkpoints.append((episode + 1, metrics['mean_return'], metrics['success_rate']))
    finally:
        env.close()
    return agent, np.asarray(history), np.asarray(checkpoints)


def plot_results(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha, entries in runs.items():
        curves = np.asarray([np.convolve(e['history'][:, 1], np.ones(200)/200, mode='valid') for e in entries])
        x = np.arange(200, curves.shape[1] + 200)
        mean, sd = curves.mean(0), curves.std(0)
        line, = axes[0].plot(x, mean, label=f'alpha={alpha}')
        axes[0].fill_between(x, mean-sd, mean+sd, color=line.get_color(), alpha=.15)
        c = np.asarray([e['checkpoints'] for e in entries])
        axes[1].plot(c[0, :, 0], c[:, :, 2].mean(0)*100, label=f'alpha={alpha}')
    axes[0].set(xlabel='Training episode', ylabel='Return (200-episode moving average)', title='Training: mean +/- 1 SD across seeds')
    axes[1].set(xlabel='Training episode', ylabel='Greedy validation success (%)', title='100 held-out validation resets', ylim=(-2, 102))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def render_gif(q, path, seed=40000):
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    frames = []
    try:
        state, _ = env.reset(seed=seed)
        frames.append(Image.fromarray(env.render()))
        total = 0
        for step in range(1, 201):
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
            total += reward
            frames.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break
        if not terminated:
            raise RuntimeError('Demonstration policy failed to deliver passenger')
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=350, loop=0)
        return dict(seed=seed, steps=step, total_return=total, success=True)
    finally:
        env.close()
