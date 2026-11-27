# Selective MPC for multi-UAV semantic image offloading under rate uncertainty

Code, stored results and checkpoints for the paper
"Predictive mode selection under wireless rate uncertainty for multi-UAV semantic image offloading"
(submitted to ICT Express).

## Layout

| Path | Content |
|---|---|
| `swarmbridge/experiments/uav_semcom/` | Simulator (link-adapted uplink, JA scheduler, 14 image modes), controllers (heuristics, PPO / QR-DQN / discrete SAC, selective MPC in `a9_mpc.py`), evaluation (`ax2_eval.py`, `fd_*.py`), paper numbers / figures (`paperA_fd_*.py`), tests (`test_*.py`) |
| `runs/uav_semcom/paperA_fd/` | Stored results of the full-data protocol fd_v1: manifest, codec / detector profiles, Phase 1-4, ablation, post-hoc analyses, per-episode delivery logs (27,600 evaluation episodes) |
| `runs/uav_semcom/paperA_ix/` | Independent evaluation (pre-registered, 10,200 test episodes): manifest `manifest_ix_v1.json`, dataset audits, controller states, H\* tuning, RL training records with action histograms, checkpoint selection, L2 lock, per-arm results (`eval/`), registered analysis (`analysis/`), publication values and supplementary CSVs (`paper/`), launch scripts and logs |
| `runs/uav_semcom/{a5,a6l1,ax9,paperA_rl}/` | Small inputs from earlier development (A5 tables, heuristic tuning, checkpoint lists) |
| `checkpoints/` | Detector (YOLO11s fine-tuned on VisDrone2019-DET) and RL checkpoints used in the paper; `SHA256SUMS` |
| `paper/paperA/` | Generated tables and figures. Independent evaluation: `figs/fig_ix_ablation.pdf`, `figs/fig_ix_tradeoff.pdf`, `table_ix_results.tex`. Earlier VisDrone evaluation: `figs/fig_fd_*.pdf`, `table_results.tex` |
| `docs/protocol/` | Data protocol, pre-registrations and result notes of each evaluation block |

The package directory keeps its original name (`swarmbridge`) so that all stored paths and artifact hashes stay valid.

## Environment

Python 3.12 (Linux / WSL), see `requirements.txt`. The GPU is needed only for codec / detector profiling
(`fd_profile.py`) and generative modes; simulation, MPC and RL evaluation run on CPU.

## Reproducing the paper numbers from stored results (no simulation)

```bash
cd swarmbridge/experiments/uav_semcom
python fd_posthoc.py verify          # fail-fast coverage / integrity check of all 27,600 episodes
python paperA_fd_numbers.py          # paper_numbers.json, table_fd.tex, table_results.tex
python paperA_fd_numbers.py tables   # re-render the tables only
python paperA_fd_figures.py ../../../paper/paperA/figs
python -m pytest -q test_*.py
```

## Independent evaluation (paperA_ix)

The test pools of the earlier VisDrone evaluation had been used during development. The method was therefore
re-evaluated on data never used for development, under a pre-registered protocol (`docs/protocol/paperA-ix-prereg.md`;
results in `docs/protocol/paperA-ix-result.md`):

| Block | Data | Artifacts |
|---|---|---|
| X1 (principal) | VisDrone2019-DET test-challenge, 52 sequences / 631 frames, label-free (return only) | Frozen from the earlier evaluation |
| X2-frozen | UAVDT official test split, 20 sequences / 600 frames | Frozen from the earlier evaluation |
| X2-retrained | Same UAVDT test split | Re-fitted on UAVDT train (26 sequences): controller state, CPU estimate, heuristic $H^*$, 9 RL runs, checkpoint selection on UAVDT val |

**Hash lock:** every fitted component is fixed by `runs/uav_semcom/paperA_ix/L2_lock.json` before any test image is
profiled. This covers the data manifest, the stage-A profile, the CPU estimate, both controller states, $H^*$, the
selection file, the 9 checkpoint hashes and 22 code files. The evaluation drivers refuse to run unless the lock
verifies.

**Reproducing the figures, table and supplementary CSVs from stored results (no simulation):**

```bash
cd swarmbridge/experiments/uav_semcom
python paperA_ix_figures.py ../../../paper/paperA/figs   # Fig. 2/3, table_ix_results.tex, runs/.../paperA_ix/paper/*
python -m pytest -q test_ix_analysis.py test_ix_recall.py test_ix_actlog.py
```

**Re-running the experiments:**
- The pipeline is driven by `ix_profile.py`, `ix_cpu.py`, `ix_stage.py` (fit / tune / train / select / lock / verify),
  `ix_eval.py` and `ix_analysis.py`.
- `runs/uav_semcom/paperA_ix/scripts/` contains the launch scripts.
- It requires the images (UAVDT, and VisDrone test-challenge for X1), the YOLO11s detector checkpoint and a GPU for
  profiling.
- Per-image profiles and delivery logs of the new datasets are not distributed (dataset terms). Per-arm, per-seed
  results and all analysis outputs are.
- The selected checkpoints are in `checkpoints/paperA_ix/rl/`.

## Re-running experiments

The evaluation drivers are `fd_phase2.py` (reference block), `fd_next.py` (ablation, network settings, learned
incumbents) and `fd_phase3_queue.py` (RL training). Image data are not included: download VisDrone2019-DET
(train / val / test-dev) and set the paths in `fd_manifest.py` (`SRC`). Checkpoint paths in the stored JSON files
refer to the original machine (`/home/uservutl/work/<sub>`). `checkpoints/<sub>` mirrors that layout, so replace the
prefix with the location of `checkpoints/`.

## Notes

- The detector checkpoint was trained with Ultralytics YOLO11 (AGPL-3.0).
- VisDrone images are not redistributed. The stored profiles contain only per-image metrics and generated captions.
- UAVDT and VisDrone test-challenge: no images, reconstructions, per-image profiles or delivery logs are included. Only
  file lists and hashes (manifest, audits), aggregate per-episode results and analysis outputs are. Obtain the datasets
  from their official sources under their own terms.
