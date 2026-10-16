# LeHWM

An attempt to improve [LeWM](https://github.com/lucas-maes/le-wm) by adding a hierarchical level on top of it.
The goal is to beat the state of the art in this area.

## Training (H-JEPA setup, cube)

Training uses the data split and frozen LeWM of [H-JEPA](https://arxiv.org/abs/2610.06805) (Zhang et al., 2026),
so that none of the models we compare against has seen the evaluation episodes:

| | |
|---|---|
| Level 0 (frozen) | H-JEPA's own LeWM checkpoints `cube_lewm/seed{42,43,44}` (converted bit-exactly by `convert_lewm.py`) |
| Data | first 9,800 episodes train, last 200 held out (their `split_cube.py`); their action normalizer |
| Level 1 training | 8 epochs (H-JEPA's cube epoch budget), seed *s* on top of LeWM seed *s* |

## Benchmark: held-out episodes, increasing goal offsets

We do not reuse the paper's evaluation (50 segments of 20 steps centred on a grasp). There, a goal-conditioned
policy alone already scores high, so the score says little about planning. Instead, every model, ours and the
paper's, is run by us on the same tasks:

| | |
|---|---|
| Tasks | drawn from the 200 held-out episodes (`make_val_tasks.py`): uniform random start, goal = the frame *K* env steps later, *K* = 25, 50, 100, 200; 50 tasks per *K*; tasks where the cube moves ≤ 4 cm are dropped |
| Success | environment criterion: cube within 4 cm of its position in the goal frame |
| Budget | 1.5 × *K* env steps (as in the paper: 30 steps for a 20-step goal) |
| Eval loop | H-JEPA's code (`third_party/H-JEPA`): replan every 10 env steps, decaying horizon, per-task reseeding |
| Reporting | success % per *K* and averaged over *K*, mean ± SE over paired model/planner seeds 42, 43, 44 |
| References | the paper's released Cube weights, each with its published planner: LeWM (flat), H-JEPA 2/3/4 levels, HWM 2/3/4 levels |

Planner compute is not part of the comparison.

```bash
./setup_hjepa.sh                                   # H-JEPA code, checkpoints, split, tasks, conversion
python benchmark.py --baselines                    # the paper's 7 models x 3 seeds -> results/leaderboard.json (long)
for s in 42 43 44; do python train.py exp=my_exp seed=$s; done
python benchmark.py my_exp                         # hwm / hwm-hybrid / hwm-noplan x 3 seeds
```

Our methods in `benchmark.py`: `hwm` plans with level 1 only (CEM on the skill `m`, actions from `pi`);
`hwm-hybrid` runs LeWM's GradientSolver with candidate 0 initialised by `pi(z0, m*)`; `hwm-noplan`
(diagnostic) skips planning: `m` is the posterior mean, actions are `pi(z0, m)`.

Every complete benchmark (all offsets, 50 tasks each) adds its per-method summary to `results/leaderboard.json`.
`benchmark.png` plots the current experiment against earlier ones and the paper's models.


python benchmark.py 1_hwm_base --plot-only