"""Create and execute a clean notebook using this virtual environment's kernel."""
from pathlib import Path
import os
import sys
import nbformat as nbf
from nbclient import NotebookClient


def build():
    os.environ['IPYTHONDIR'] = str(Path('.venv/ipython').resolve())
    os.environ['JUPYTER_RUNTIME_DIR'] = str(Path('.venv/jupyter_runtime').resolve())
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md('# From-scratch Q-learning on Taxi-v3\nTopics in AI, Assignment I. See `FA26-TAI-AI.pdf` for requirements. This notebook executes a fresh training run, shows its progress, then inspects the full three-seed learning-rate study. Run `python main.py all` first to regenerate that study.'),
        code('import sys, json\nfrom pathlib import Path\nimport numpy as np\nfrom IPython.display import display, Image\nfrom agent import Config, QLearningAgent\nfrom utils import train, evaluate\nprint("Python:", sys.executable)\nassert sys.prefix != sys.base_prefix, "Run this notebook inside the project virtual environment"'),
        md('## Environment and algorithm\n500 encoded states and six actions. Q-learning uses $Q(s,a) \\leftarrow Q(s,a)+\\alpha[r+\\gamma(1-\\mathrm{terminated})\\max_b Q(s_{next},b)-Q(s,a)]$. Truncation ends a rollout but retains bootstrapping. All actions are available; no action mask is used.'),
        code('config = Config(alpha=0.5, episodes=12000)\nagent, history, checkpoints = train(config, seed=0)\nprint("Episode | greedy validation return | success")\nfor episode, mean_return, success in checkpoints:\n    print(f"{episode:7.0f} | {mean_return:24.2f} | {success:7.1%}")'),
        code('import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(9, 3))\nax.plot(np.arange(200, len(history)+1), np.convolve(history[:,1], np.ones(200)/200, "valid"))\nax.set(xlabel="Episode", ylabel="200-episode mean return", title="Fresh seed-0 training run")\ndisplay(fig)\nplt.close(fig)'),
        md('## Independent evaluation\nThe test reset seeds differ from training and validation seeds. The random baseline receives the same test reset seeds and the same time limit.'),
        code('print("Fresh trained policy:", evaluate(agent.q))\nprint("Uniform random:", evaluate())'),
        md('## Learning-rate sensitivity across three training seeds\nThe complete experiment compares alpha 0.1, 0.5, and 0.9 with seeds 0, 1, and 2. Curves show mean and one standard deviation of smoothed training returns. Validation selects the learning rate; test results are reported after selection.'),
        code('results = json.loads(Path("artifacts/results.json").read_text())\nprint("Selected configuration:", results["selection"])\nfor alpha in ("0.1", "0.5", "0.9"):\n    scores = results["test"][alpha]\n    print(alpha, "test mean return:", np.mean([s["mean_return"] for s in scores]), "success:", np.mean([s["success_rate"] for s in scores]))\ndisplay(Image(filename="artifacts/learning_curves.png"))'),
        md('## Saved model and successful demonstration'),
        code('saved = QLearningAgent(500, 6)\nsaved.load("artifacts/q_table.npy")\nprint("Saved table:", saved.q.shape)\nprint("Selected policy test:", evaluate(saved.q))\nprint("Demonstration:", results["demo"])\ndisplay(Image(filename="artifacts/taxi_policy.gif"))'),
        md('## Discussion\nCompare early validation checkpoints as well as final returns: settings can reach the same final performance at different speeds. Training returns include exploration penalties, while greedy evaluation measures the learned policy. Three seeds and 500 sampled test resets provide empirical evidence, not a convergence proof or exhaustive guarantee. The report discusses failure modes and scope. All measured results and episode logs are in `artifacts/`.')]
    nb.metadata['kernelspec'] = dict(display_name='Python (project .venv)', language='python', name='python3')
    client = NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}})
    # Explicit kernel command prevents accidental use of a global Python kernel.
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec
    km = KernelManager(kernel_name='python3')
    km._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], display_name='Project .venv', language='python')
    client.km = km
    client.execute()
    nbf.write(nb, 'assignment.ipynb')
    print('Executed assignment.ipynb')


if __name__ == '__main__':
    build()
