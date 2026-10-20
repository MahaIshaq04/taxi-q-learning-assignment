"""Build a four-page Word report from the saved experiment outputs."""
import base64
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path('artifacts')


def build():
    r = json.loads((ROOT/'results.json').read_text())
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.top_margin = sec.bottom_margin = Inches(.58)
    sec.left_margin = sec.right_margin = Inches(.65)
    normal = doc.styles['Normal']
    normal.font.name, normal.font.size = 'Calibri', Pt(10)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.0
    for name, size in [('Title', 19), ('Heading 1', 13), ('Heading 2', 11)]:
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].font.color.rgb = RGBColor.from_string('17365D')
        doc.styles[name].paragraph_format.space_before = Pt(7)
        doc.styles[name].paragraph_format.space_after = Pt(4)
    footer = sec.footer.paragraphs[0]
    footer.alignment = 2
    footer.add_run('Topics in AI | Assignment I | Page ').font.size = Pt(8)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    def p(text):
        return doc.add_paragraph(text)
    def h(text):
        doc.add_heading(text, level=1)
    def table(headers, rows, size=9):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = 'Light Shading Accent 1'
        for cell, val in zip(t.rows[0].cells, headers):
            cell.text = str(val)
        for row in rows:
            for cell, val in zip(t.add_row().cells, row):
                cell.text = str(val)
        for row in t.rows:
            pr = row._tr.get_or_add_trPr()
            pr.append(OxmlElement('w:cantSplit'))
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(2)
                    para.paragraph_format.space_before = Pt(1)
                    for run in para.runs:
                        run.font.size = Pt(size)
        p('') .paragraph_format.space_after = Pt(0)
    def picture(path, caption, width=6.8):
        doc.add_picture(str(path), width=Inches(width))
        para = p(caption)
        para.paragraph_format.space_after = Pt(4)
        for run in para.runs:
            run.font.size = Pt(8)
            run.italic = True

    doc.add_heading('Custom Reinforcement Learning Agent\nusing Farama Gymnasium', 0)
    p('COMSATS University, Attock Campus | Topics in AI | Assignment I')
    p('Student name: ____________________    Registration no.: ____________________')
    p('Repository: https://github.com/MahaIshaq04/taxi-q-learning-assignment')
    h('1. Problem description')
    p('Taxi-v3 models passenger pickup and delivery on a 5 × 5 grid with internal walls and four landmarks. The agent must collect a passenger and deliver them to the requested destination. The environment is initialized with gymnasium.make("Taxi-v3") under Gymnasium 1.2.1, using its default deterministic dynamics.')
    p('There are 500 encoded states: 25 taxi positions × 5 passenger locations (four landmarks or inside the taxi) × 4 destinations. The encoding is s = ((row × 5 + column) × 5 + passenger) × 4 + destination. Only a subset is reachable, with 300 possible initial states. Six actions are available: south, north, east, west, pickup, and dropoff.')
    table(['Event', 'Reward / episode behavior'], [['Ordinary move or wall collision', '−1'], ['Illegal pickup or dropoff', '−10'], ['Correct delivery', '+20; episode terminates'], ['200-step time limit', 'Episode truncates']])
    h('2. Methodology')
    p('Tabular Q-learning is implemented from scratch with NumPy. Each Q(s,a) estimates the discounted return for taking action a in state s and then following a greedy policy. The 500 × 6 float64 table starts at zero. After each transition, the agent applies:')
    eq = p('Q(s,a) ← Q(s,a) + α [r + γ(1 − d) maxᵦ Q(s′,b) − Q(s,a)]')
    eq.runs[0].bold = True
    p('Here α is the learning rate, γ is the discount, and d indicates true termination. Successful delivery suppresses bootstrapping. A time-limit truncation ends the rollout but retains bootstrapping because the limit is external to the underlying task.')
    p('Training uses epsilon-greedy exploration: a uniformly random action with probability ε, otherwise a maximal-Q action with random tie-breaking. εₑ = max(0.01, 1.0 × 0.999ᵉ), where e starts at zero. Greedy evaluation uses deterministic argmax. No action masking, predefined routes, planning model, or RL framework is used.')
    h('Implementation architecture')
    p('agent.py owns Q-values, updates, action selection, and persistence. utils.py handles training, evaluation, plots, and GIF rendering. main.py orchestrates experiments. assignment.ipynb displays executed training and evaluation outputs. All dependencies are installed inside the local .venv.')

    doc.add_page_break()
    h('3. Experimental setup and hyperparameters')
    cfg = r['configs'][str(r['selection']['alpha'])]['parameters']
    table(['Parameter', 'Final selected value'], [[k,v] for k,v in cfg.items()] + [['Training seeds', '0, 1, 2'], ['Maximum steps / episode', 200], ['Selected model seed', 0]])
    p('The sensitivity study compares α ∈ {0.1, 0.5, 0.9}, keeping all other settings fixed: nine runs of 12,000 episodes (108,000 episodes in total). Validation every 1,000 episodes uses reset seeds 20000–20099. The highest mean final validation return across three training seeds selects α; ties select the lower α. Deployment always uses the predeclared seed 0.')
    p('After selection, every trained policy and a uniform-random baseline are evaluated on the same 500 separate test resets (30000–30499). Test results do not select hyperparameters. Validation does not update the Q-table or consume the training action RNG. The demonstration uses another reset seed, 40000.')
    h('Executed training-progress output')
    nb = json.loads(Path('assignment.ipynb').read_text(encoding='utf-8'))
    progress = []
    for cell in nb['cells']:
        for output in cell.get('outputs', []):
            txt = ''.join(output.get('text', []))
            if 'Episode | greedy validation return' in txt:
                for line in txt.splitlines()[1:]:
                    if '|' in line:
                        progress.append([x.strip() for x in line.split('|')])
    table(['Episode', 'Greedy validation return', 'Success (100 resets)'], progress, size=8)
    p('This is the actual checkpoint output from the notebook’s fresh α=0.5, seed-0 run. It is a demonstration run, separate from the validation-selected α=0.1 saved model. Training logs preserve every episode for all nine study runs.')
    h('Reproducibility outputs')
    p('Saved model: artifacts/q_table.npy; shape: (500, 6); 3,000 float64 action values. Raw outputs: nine *_training.csv logs, nine *_validation.csv logs, nine per-run *_q.npy models, and results.json. The notebook contains six successfully executed code cells. Full numerical logs remain in the repository; this report presents their measured summaries.')

    doc.add_page_break()
    h('4. Results and comparative analysis')
    table(['Policy', 'Success', 'Mean return', 'Mean steps', 'Illegal / ep.'], [
        ['Selected Q-learning', '100% (500/500)', '8.030', '12.970', '0'],
        ['Uniform random', '4.2% (21/500)', '−773.024', '197.618', '64.032']], size=9)
    p('The selected policy (α=0.1, seed 0) completes every sampled test episode without illegal actions. Random behavior frequently incurs pickup/dropoff penalties and reaches the time limit. The selected policy’s return standard deviation across test episodes is 2.553, reflecting differences among starting tasks.')
    h('All nine trained-policy test outputs')
    rows = []
    for a in ('0.1','0.5','0.9'):
        for seed, metrics in zip(r['seeds'], r['test'][a]):
            rows.append([a, seed, f'{metrics["mean_return"]:.3f}', f'{metrics["mean_steps"]:.3f}', f'{100*metrics["success_rate"]:.0f}%', f'{metrics["mean_illegal"]:.0f}'])
    table(['α', 'Seed', 'Mean return', 'Mean steps', 'Success', 'Illegal / ep.'], rows, size=8)
    picture(ROOT/'learning_curves.png', 'Figure 1. Left: 200-episode moving-average training returns, mean ± one SD over three training seeds. Right: greedy validation success at 1,000-episode checkpoints.')
    p('Final test returns (mean ± population SD of seed-level means) are 8.030 ± 0.000 for α=0.1, 7.990 ± 0.018 for α=0.5, and 8.019 ± 0.010 for α=0.9. These descriptive spreads are not confidence intervals. All nine policies reach 100% test success, with small differences in route length.')
    p('Higher learning rates improve early learning speed. First observed 100% validation success occurs at episodes 4000/3000/4000 for α=0.1, 2000/2000/1000 for α=0.5, and 2000/1000/1000 for α=0.9 (seeds 0/1/2). The validation-selected α=0.1 learns more slowly but has the best final average validation return. These are observed milestones, not a convergence proof.')

    doc.add_page_break()
    h('5. Visual outputs and learned-policy demonstration')
    # Copy the actual fresh-run plot embedded in the executed notebook.
    fresh = ROOT/'notebook_training_output.png'
    found = False
    for cell in nb['cells']:
        for output in cell.get('outputs', []):
            png = output.get('data', {}).get('image/png')
            if png and not found:
                fresh.write_bytes(base64.b64decode(''.join(png)))
                found = True
    if found:
        picture(fresh, 'Figure 2. Actual fresh seed-0, α=0.5 training plot from the executed notebook.', width=5.8)
    with Image.open(ROOT/'taxi_policy.gif') as gif:
        indices = [0, gif.n_frames//2, gif.n_frames-1]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
        for ax, idx, title in zip(axes, indices, ['Initial state', 'Mid-episode', 'Successful delivery']):
            gif.seek(idx)
            ax.imshow(gif.convert('RGB'))
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        fig.tight_layout()
        frames = ROOT/'policy_frames.png'
        fig.savefig(frames, dpi=160, bbox_inches='tight')
        plt.close(fig)
    picture(frames, 'Figure 3. Frames from taxi_policy.gif. Reset seed 40000; successful delivery in 10 steps; total return 11. The GIF and executed notebook contain the full animation.', width=6.6)
    h('6. Discussion and failure modes')
    p('Early exploration produces illegal pickup/dropoff attempts, wall collisions, and unnecessary movements. Before enough learning, greedy ties or inaccurate values can cause loops until truncation. The ε=0.01 exploration floor continues to introduce occasional training penalties even when the greedy policy is reliable; training returns therefore differ from evaluation returns.')
    p('Limitations: only three training seeds and 500 sampled test resets were used, rather than exhaustive verification. Only learning rate was varied. Constant learning rates and a finite budget do not prove convergence. Performance on a small deterministic grid does not establish generalization to changing traffic, stochastic maps, or continuous control.')
    h('7. Conclusion')
    p('From-scratch Q-learning solves the sampled Taxi-v3 test tasks with 100% delivery success, mean return 8.030, and mean length 12.970 steps. Independent validation/test resets, a random baseline, multiple seeds, saved models, executed plots, and a successful animation support these empirical findings.')
    h('References')
    para = p('Farama Foundation, Taxi: https://gymnasium.farama.org/environments/toy_text/taxi/\nFarama Foundation, Handling Time Limits: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/\nSutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., Chapter 6.\nEnvironment facts were checked against installed Gymnasium 1.2.1; current web documentation may describe a newer version.')
    for run in para.runs:
        run.font.size = Pt(8)
    # Declare intended page count; three explicit breaks provide four page layouts.
    props = doc.core_properties
    props.title = 'Custom Reinforcement Learning Agent using Farama Gymnasium'
    props.subject = 'Topics in AI Assignment I — Taxi-v3 Q-learning'
    doc.save(ROOT/'technical_report.docx')
    print('Created artifacts/technical_report.docx (four explicit page layouts)')


if __name__ == '__main__':
    build()
