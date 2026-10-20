"""Reproduce all experiments and submission artifacts from the project root."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from agent import Config, QLearningAgent
from utils import train, evaluate, plot_results, render_gif


def experiment(output=Path('artifacts'), episodes=12000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = {}
    summary = {'environment': 'Taxi-v3', 'seeds': [0, 1, 2], 'configs': {}, 'selection': {}, 'test': {}}
    for alpha in (0.1, 0.5, 0.9):
        config = replace(Config(), alpha=alpha, episodes=episodes)
        entries = []
        for seed in summary['seeds']:
            agent, history, checkpoints = train(config, seed)
            validation = evaluate(agent.q, range(20000, 20100))
            entries.append(dict(agent=agent, history=history, checkpoints=checkpoints, validation=validation))
            prefix = output / f'alpha_{alpha}_seed_{seed}'
            np.save(str(prefix)+'_q.npy', agent.q)
            np.savetxt(str(prefix)+'_training.csv', history, delimiter=',', header='episode,return,steps,success,epsilon', comments='')
            np.savetxt(str(prefix)+'_validation.csv', checkpoints, delimiter=',', header='episode,mean_return,success_rate', comments='')
            print(f'alpha={alpha} seed={seed}: validation={validation}', flush=True)
        runs[alpha] = entries
        summary['configs'][str(alpha)] = dict(parameters=asdict(config), validation=[e['validation'] for e in entries])
    # Selection uses validation only; fixed seed 0 avoids cherry-picking training seeds.
    chosen = max(runs, key=lambda a: (np.mean([e['validation']['mean_return'] for e in runs[a]]), -a))
    agent = runs[chosen][0]['agent']
    agent.save(output/'q_table.npy')
    summary['selection'] = dict(alpha=chosen, seed=0, criterion='Highest mean validation return across three seeds; ties choose lower alpha. Deployment uses predeclared seed 0.')
    summary['test']['random'] = evaluate()
    for alpha, entries in runs.items():
        summary['test'][str(alpha)] = [evaluate(e['agent'].q) for e in entries]
    summary['test']['selected'] = evaluate(agent.q)
    summary['demo'] = render_gif(agent.q, output/'taxi_policy.gif')
    plot_results(runs, output/'learning_curves.png')
    (output/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    from report import build_report
    build_report(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'evaluate', 'render'], nargs='?', default='all')
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--episodes', type=int, default=12000)
    args = parser.parse_args()
    if args.command == 'all':
        experiment(args.output, args.episodes)
    else:
        agent = QLearningAgent(500, 6)
        agent.load(args.output/'q_table.npy')
        print(evaluate(agent.q) if args.command == 'evaluate' else render_gif(agent.q, args.output/'taxi_policy.gif'))


if __name__ == '__main__':
    main()
