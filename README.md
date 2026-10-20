# Taxi-v3: Q-learning from scratch

COMSATS University, Attock — Topics in AI, Assignment I. Requirements are in [FA26-TAI-AI.pdf](FA26-TAI-AI.pdf).

## Setup (PowerShell, Python 3.11+)

All installations belong in the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The committed `requirements-lock.txt` records the exact environment used for the submitted run (Windows, Python 3.13). For that platform, install it instead of `requirements.txt` for exact dependency versions. Other platforms should use `requirements.txt`.

## Reproduce

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe main.py all
.\.venv\Scripts\python.exe build_notebook.py
.\.venv\Scripts\python.exe build_word_report.py
.\.venv\Scripts\python.exe main.py evaluate
```

`main.py all` trains nine policies (12,000 episodes each), evaluates a random baseline and every trained policy on 500 test resets, saves the selected model and GIF, plots learning curves, and creates the four-page PDF. `build_notebook.py` executes another 12,000-episode run and embeds progress, plots, metrics, and animation in the notebook. No global kernel registration is needed. On macOS/Linux use `.venv/bin/python`.

## Deliverables

| File | Purpose |
|---|---|
| `main.py`, `agent.py`, `utils.py` | Experiment runner, custom NumPy Q-learning, training/evaluation/rendering |
| `assignment.ipynb` | Executed notebook with training progress and visualizations |
| `artifacts/q_table.npy` | Selected 500 x 6 trained Q-table |
| `artifacts/taxi_policy.gif` | Successful delivery using the learned policy |
| `artifacts/report.pdf` | Four-page submission report |
| `artifacts/technical_report.docx` | Editable four-page Word report with output tables, plots and GIF frames |
| `artifacts/learning_curves.png` | Convergence and sensitivity plots |
| `artifacts/results.json` | Hyperparameters, seed information, raw aggregate metrics |
| `artifacts/*_training.csv`, `*_validation.csv`, `*_q.npy` | Per-run logs, checkpoint metrics and models |

## Experiment design

Learning rates 0.1, 0.5, and 0.9 are compared across training seeds 0, 1, and 2. Discount is 0.99; epsilon decays from 1.0 by 0.999 per episode to 0.01. No action masking or route planning is used. Terminal transitions suppress bootstrapping; time-limit truncations retain it. Greedy validation uses reset seeds 20000–20099; independent test resets use 30000–30499. Hyperparameters are selected by mean final validation return, with lower alpha breaking ties; deployment always uses seed 0. Test returns do not select the model.

A nonzero exploration floor makes training return lower than greedy policy return. Reported seed variation is descriptive, not a confidence interval. The benchmark is small and deterministic; results do not establish generalization to real traffic.

## Measured results

The selected model uses alpha 0.1 and training seed 0. On 500 held-out test resets:

| Policy | Delivery success | Mean return | Mean steps | Illegal actions per episode |
|---|---:|---:|---:|---:|
| Selected Q-learning | 100% | 8.030 | 12.970 | 0 |
| Uniform random | 4.2% | -773.024 | 197.618 | 64.032 |

All nine trained policies achieved 100% test delivery success, with small differences in route length. The GIF shows a separate successful 10-step episode. See the PDF for the complete comparison and checkpoint milestones.

## Submission

Review the executed notebook and PDF, add your name/student ID if required, and submit https://github.com/MahaIshaq04/taxi-q-learning-assignment together with `artifacts/report.pdf`. The editable report is `artifacts/technical_report.docx`. Exclude `.venv` (already in `.gitignore`). Running `main.py all` regenerates the experiment PDF as `artifacts/technical_report.pdf`; the final submission PDF is `artifacts/report.pdf`.

## References

- [Farama Taxi documentation](https://gymnasium.farama.org/environments/toy_text/taxi/) (online docs may describe a newer environment version; this project pins Gymnasium 1.2.1 and Taxi-v3).
- [Farama: handling time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition, Chapter 6.
