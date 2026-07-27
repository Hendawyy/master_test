# Continuing the Neuro-DT Pipeline From Home — No Lab Required

## When you can leave the lab for good

As soon as **Cell 14** (ablation training) finishes — and ideally Cells 15–16 too —
every remaining cell in `NeuroDT_GPU_Lab.ipynb` is evaluation/analysis on
*already-trained* models, not training. None of it needs a GPU, and none of
it needs an Azure ML workspace. You can do all of it from any laptop,
anywhere, once the right files are on that laptop.

---

## Step 1 — Before you leave the lab, confirm training is actually done

- [ ] Cell 8 — all 5 folds complete (`checkpoints\best_model_fold1.pth` … `fold5.pth`)
- [ ] Cell 8B — all 5 folds complete (`*_aucsel.pth`) — kept for the methodology
      discussion section only, not used downstream
- [ ] Cell 14 — `checkpoints\ablation\ablation_results.json` has all 7 variants
      (A0–A6)
- [ ] Cell 15 — classical ML baselines (B1–B4) run
- [ ] Cell 16 — `checkpoints\ablation\ablation_results_table.csv` generated

If Cell 15/16 aren't done yet, don't worry — they need **no GPU and no
tensor_cache**, so they're just as easy to run from home (see Step 4).

---

## Step 2 — Decide which files you actually need on the laptop

| Folder | Needed for | Size |
|---|---|---|
| `checkpoints\` (incl. `checkpoints\ablation\`) | Everything — this is always required | Small (checkpoints + JSON/CSV, low GB) |
| `tensor_cache\` | Only Cells 18, 20, 24, 26–33 — anything that loads an actual MRI scan (evaluation, Grad-CAM, Digital Twin inference, what-if simulations) | ~13 GB |

Cells 15, 16, and 22 (classical ML, results table, Markov chain) only touch
`master_manifest.csv` / `df` and the results JSON files — **no tensor_cache
needed at all** for those three.

So: if you only care about the ablation table and the Markov/prognostic
side, you're done after copying `checkpoints\`. If you want the evaluation
metrics, Grad-CAM figures, or the Digital Twin/what-if simulations
(Cells 18, 20, 24, 26–33) — including Cell 33, the one `dashboard.py` will
eventually call — copy `tensor_cache\` too.

---

## Step 3 — Move the files to your laptop

Pick whichever is easiest:

- **Direct copy** — USB drive, external SSD, or a cloud-sync folder
  (OneDrive/Google Drive), straight from the lab PC to the laptop. Simplest
  for a one-time transfer, no extra scripting.
- **Azure relay** — reuse the blob storage you already have credentials
  for: upload from the lab PC, then run the existing `download_files.py` on
  the laptop pointed at the same blob paths. Ask me to write the small
  upload counterpart if you want this route — it's a five-minute script,
  same retry/atomic-write logic as `download_files.py`.

---

## Step 4 — Point the notebook at the laptop's paths

In **Cell 7**, update:
```python
BEST_MODEL_DIR = Path(r"<laptop's checkpoints folder>")
CACHE_DIR      = Path(r"<laptop's tensor_cache folder>")   # only if you copied it
```

A few other cells (recovery cells, Cell 9, etc.) also have the lab PC's
hardcoded path (`C:\Users\seif\neuro_dt\...`) — search the notebook for that
string and update every occurrence to match your laptop.

⚠️ If the laptop isn't Windows, or has a different Windows username, these
paths need to change accordingly — use a plain `Path("...")` with
forward slashes on Mac/Linux instead of the raw Windows string.

---

## Step 5 — Run order on a fresh laptop kernel

1. **1 → 2 → 3 → 4 → 5 → 6 → 7** — same setup cells as always (imports,
   warnings, Golden DataFrame, dataset/model class defs, training config).
   No GPU required — `IS_GPU` will just be `False`, which only affects
   `BATCH_SIZE`/`AMP`, not correctness.
2. **9** — Kernel Recovery. Loads the finished checkpoints, rebuilds
   `fold_results`/`best_model`/`df_val_best`. No retraining happens.
3. **11** — CPU vs GPU comparison (if you haven't captured this already).
4. **12 → 13** — re-run just to redefine `ABLATION_VARIANTS`/
   `run_ablation_variant` and `ABLATION_DIR` (needed by later cells) — this
   does **not** retrain anything by itself, it only defines things. Then:
   - **15** (if not already run) — classical ML baselines, no GPU needed.
   - **16** — compiles the full results table + charts.
5. **18 → 20 → 22 → 24 → 26 → 27 → 28 → 29 → 30 → 31 → 32 → 33** —
   evaluation, Grad-CAM, Markov chain, Digital Twin assembly, and every
   what-if/simulation cell, ending with single-patient inference. All of
   this is inference/analysis on the already-trained model — CPU-only is
   fine, just slower per cell (seconds to low minutes, not hours).

---

## No Azure ML workspace needed for any of this

This whole guide deliberately avoids Azure ML compute — see the "Azure ML
workspace vs. local run" question below for why that's a deliberate choice,
not just a simplification.
