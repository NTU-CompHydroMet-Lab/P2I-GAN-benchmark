# Experiments

## Structure
- `config.py`: Central configuration (paths, mode, crop size, experiment switches).
- `main.py`: Entry point; dispatches exp1/exp2/exp3 and writes results.
- `io.py`: Shared I/O helpers (zarr loading, masks, cropping, saving).
- `exp1.py`: Numerical metrics (MAE/RMSE/PSS/SSIM/DTSSIM/NSE + categorical POD/FAR/CSI/HSS).
- `exp2.py`: Visualization + GIF output; optional paper-style PDF panel.
- `exp3.py`: NSE scatter plot and residual plot.
- `results/`: Output root; each run writes to `results/<experiment_name>/`.

## How to run
```bash
python -m experiments.main
```

## Configuration
Edit `experiments/config.py`:
- `mode`: `radar` or `gauge`.
- `run_exp1`, `run_exp2`, `run_exp3`: enable or disable experiments.
- `crop_size`: center crop size shared across experiments.
- `data.*`: input paths (zarr, masks, method outputs).

### exp2 paper-style PDF
Set in `experiments/config.py`:
- `exp2_paper_enabled = True`
- `exp2_paper_folders`: map method names to PNG folders (expects `rain{event_id}` subfolders).
- `exp2_paper_events`: list of `{event_id, select_idx, title}`.

Output will be written to:
- `experiments/results/<experiment_name>/exp2/<output_pdf>`

