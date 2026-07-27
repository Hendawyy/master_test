# Neuro-DT Project Handover — GPU Lab Training Session

---

# Goal (what we are trying to build)

Complete the Brain Digital Twin (Neuro-DT) master's thesis at the Arab Academy for
Science, Technology and Maritime Transport.

Student: Seif Hendawy
Supervisors: Prof. Fahima Maghraby · Assoc. Prof. Ahmed Salem

The Azure CPU compute (`Azure/` in this repo) could only complete 1/5 folds of
cross-validated training before early-stopping instability, and couldn't run the
DL ablation variants (A2–A6) at all. This session's goal was to get the university
GPU lab machine (RTX 5070 Ti) fully set up, transfer all data from Azure, get
`NeuroDT_GPU_Lab.ipynb` actually running end-to-end, and produce a real 5-fold
cross-validated AUC mean ± std plus the complete ablation study — the numbers the
CPU run couldn't produce. If GPU results are better, `dashboard.py` (Streamlit
digital twin app) will be pointed at the GPU-trained checkpoints for new-patient
inference.

---

## Current State (where the work stands right now)

**The entire notebook (Cells 1–33, plus 8B and 14B) has now run end to end on
the GPU lab machine.** Main training, ablation study, and the full
post-training/Digital-Twin pipeline are all complete. What's left is not
more training — it's (1) investigating the two flagged issues below (age
sensitivity direction, Cell 33's zero-tensor fallback) before anything goes
into the thesis or `dashboard.py`, and (2) getting the files off the lab PC
(see `Continue_From_Home_Guide.md`).

### GPU Lab Machine — training complete, no longer needed for this project
- Machine: University lab PC, Windows, `bdt-env` conda env, Python 3.10
- GPU: NVIDIA RTX 5070 Ti — 17.1 GB VRAM, PyTorch `2.11.0+cu128`, CUDA confirmed working
- All dependencies installed and verified in `bdt-env`

### Data — fully downloaded and verified
- Tensor cache: 1,549/1,549 `.pt` files, all `torch.load()`-verified (see
  `verify_downloads.py`), at `C:\Users\seif\neuro_dt\tensor_cache`
- Checkpoints: 9 CPU-baseline `.pth` files + `auc_results.json` + `markov_matrices.pkl`
  + PNGs, at `C:\Users\seif\neuro_dt\checkpoints` — these are being **overwritten** by
  the current GPU run's own checkpoints (safe — Cell 11/COMPARE only reads
  `auc_results.json`, not the `.pth` weight files)
- `master_manifest.csv` — fetched separately from Azure ML's own `workspaceblobstore`
  datastore (was never part of the original `gpu_transfer` blob transfer) and routed
  through the existing `adni-data/gpu_transfer/checkpoints/` path so `download_files.py`
  picked it up with no code changes
- `bad_scans.txt` — confirmed genuinely absent from the ingestion job output (not
  a bug); Cell 4 handles its absence gracefully
- `ablation_results.json` — **still not found** in any download, on two independent
  machines/runs. Needed before Cell 14 so the CPU's completed ablation results
  (A0, A1, B1–B4) aren't silently redone. Unresolved — see Next Steps.

### Notebook (`Local_GPU/NeuroDT_GPU_Lab.ipynb`) — fixed, Cell 8 (loss-selected) run COMPLETE
All 33 cells renumbered sequentially (1→33, no gaps, no letter-suffix labels),
every Azure-only dependency removed or shimmed for lab-PC mode, and a long list of
real runtime bugs fixed (full detail in **Changed**, below). A new optional
**Cell 8B** (AUC-based checkpoint selection, see below) was also added.

**Cell 8, loss-selected 5-fold run — all 5 folds complete:**

| Fold | Status | Best (loss-selected) val AUC | Notes |
|---|---|---|---|
| 1 | Done, early-stopped ep.13 | 0.8821 | Peak AUC seen was 0.9180 @ ep.13 (worse loss, not selected) |
| 2 | Done, early-stopped ep.13 | 0.8591 | Epoch 9 had a transient instability spike (vl_loss 1.95), self-corrected by ep.10 |
| 3 | Done, early-stopped ep.7 | 0.8125 | Weakest fold — plateaued fast, likely genuine fold-to-fold variance |
| 4 | Done, ran full 20 epochs (no early stop) | 0.9511 | Strongest fold by far — smooth monotonic improvement, no instability spikes |
| 5 | Done, early-stopped ep.10 | 0.8481 | — |

**Final 5-fold CV mean AUC (loss-selected): 0.8706 ± 0.0461** — this is the
real, complete GPU number. Still need to run Cell 11 (COMPARE) for the
formal CPU-vs-GPU writeup.

Checkpoint selection is by **minimum validation loss**, not maximum AUC — these
disagreed in 3 of 5 folds (Folds 1, 2, 3; only Fold 4 agreed exactly, being the
one clean non-early-stopped fold). Peak-AUC-if-selected-differently: Fold 1
0.9180, Fold 2 0.9035, Fold 3 0.8249 — a consistent, non-trivial gap. Given
that pattern, decided to run a **second, separate** full training pass with
AUC-based checkpoint selection for a clean two-run comparison (not a 3-way
comparison — see chat history for the reasoning). That second pass is
**Cell 8B** — see **Changed**, below.

**Cell 8B, AUC-selected 5-fold run — COMPLETE:**

| Fold | Loss-selected AUC | AUC-selected AUC | Δ |
|---|---|---|---|
| 1 | 0.8821 | 0.9439 | +0.0618 |
| 2 | 0.8591 | 0.9398 | +0.0808 |
| 3 | 0.8125 | 0.9265 | +0.1140 |
| 4 | 0.9511 | 0.9451 | −0.0059 |
| 5 | 0.8481 | 0.9342 | +0.0861 |
| **Mean** | **0.8706 ± 0.0461** | **0.9379 ± 0.0069** | — |

At face value this looks like a large, consistent win for AUC-based selection.
It almost certainly is not a genuine generalization improvement, for two
compounding reasons visible directly in the per-epoch logs:

1. **All 5 AUC-selected folds ran the full 20 epochs with no early stop**,
   because AUC on a noisy ~310-sample validation fold rarely gets *worse* for
   5 consecutive epochs the way loss does — so patience never triggers. By
   epoch 18–20 every fold shows `tr_loss≈0.05–0.09, tr_acc≈0.97–0.99` — deep
   train-set memorization. The loss-selected run, by contrast, early-stopped
   4 of 5 folds specifically to avoid training this far into that regime.
2. **The selected checkpoint is the argmax of a noisy per-epoch metric on a
   small (~310-sample) validation set**, evaluated using the *same* metric
   being reported. Val AUC visibly bounces ±0.01–0.03 epoch-to-epoch late in
   every fold (e.g. Fold 2: 0.9398→0.9398→0.9392 across ep.18–20; Fold 5:
   0.9146→0.9192→0.9342→0.9315→0.9308 across ep.16–20). Picking the single
   epoch that happened to land highest is a biased, optimistic estimator of
   true generalization — structurally similar to reporting the best of 20
   lottery draws as the typical draw. This also explains the suspiciously
   *tight* std (0.0069 vs. the loss-selected run's 0.0461): cherry-picking a
   per-epoch max compresses apparent fold-to-fold variance rather than
   reflecting genuinely more consistent generalization.

Net read: the model **is** learning real signal in both runs (val AUC is well
above chance — 0.5 — in every fold of both runs). But the AUC-selected
number is inflated on top of that real signal by training deep into
memorization *and* then selecting on the same noisy metric being reported.
**The loss-selected result (0.8706 ± 0.0461) is the one to report as the
primary, defensible 5-fold CV estimate.** The AUC-selected run is kept as an
explicit methodological discussion point (selection-on-validation-metric
bias + why loss-based early stopping is the more conservative choice) —
this is a good thesis discussion point, not a discarded result.

### Repo / git state
- Working on `master` branch directly (the original feature branch's PR #1 was
  merged, then further work moved to `master` per explicit instruction)
- Everything below is committed and pushed
- ⚠️ **Azure client secret is still sitting in plaintext** in this repo's git
  history (`Handoff.md`, `download_files.py`) across many commits, on GitHub,
  unrotated since the start of this session. See Next Steps.

---

## Files in Flight (Active Files Being Modified)

- `Local_GPU/NeuroDT_GPU_Lab.ipynb` — the main notebook; extensively modified this
  session, currently open and running live on the lab PC
- `Local_GPU/download_files.py` — heavily hardened (retries, atomic writes, resume,
  manifest support); stable now, not actively changing
- `Local_GPU/verify_downloads.py` — new this session; validates cache + checkpoints
  by actually loading them, not just checking existence
- `Local_GPU/Handoff.md` — this file
- `Local_GPU/cell4_local_replacement.py`, `fetch_and_upload_manifest.py`,
  `cell8_resumable_standalone.py`, `cell8b_auc_selection_standalone.py` — working
  reference snippets created mid-session; their content is now fully embedded in
  the notebook itself, kept in the repo for reference/reuse but not required for
  the notebook to run
- `Local_GPU/already_downloaded.txt` — manifest support file for cross-machine
  download deduplication (currently empty; only useful if you deliberately populate
  it with filenames already present on another machine)

---

## Changed (what has been touched)

### `download_files.py`
- Added elapsed-time/ETA logging; upfront already-have/to-fetch counts; more
  frequent progress updates (every 10 files or 30s) instead of long silent stretches
- Added skip-if-already-downloaded to the checkpoints loop (previously only
  tensor_cache had it)
- Made downloads atomic: writes to a `.part` temp file, renames only on full
  success — an interrupted download (Ctrl+C, network drop, closed terminal) can
  no longer leave a corrupt file that a later run mistakes for complete
- Added auto-cleanup of stale `.part`/zero-byte files at the start of each run
- Added retry-with-exponential-backoff for individual blob downloads **and** for
  blob *listing* calls (`list_blobs()` is a lazy pager — it can drop mid-page on
  a flaky connection just like a download can)
- Added support for an optional `already_downloaded.txt` manifest so a second
  machine (e.g. a home laptop) skips files already fetched elsewhere

### `verify_downloads.py` (new)
- Actually `torch.load()`s every cached tensor and checkpoint file (with
  `weights_only=False`) rather than just checking file existence — catches
  corruption that finishes writing but is still bad
- Must be run in `bdt-env` (needs `torch`/`monai`/`sklearn`); running it in the
  lightweight download-only `.venv` produces false "CORRUPT" reports (wrong
  environment, not real corruption — happened twice, confirmed both false alarms)

### `NeuroDT_GPU_Lab.ipynb`
- **Cell 2 (was "Cell 3")**: replaced the real Azure ML workspace connection
  (`MLClient.from_config()`, `Workspace.from_config()`) with `ml_client=None; ws=None`
  for lab-PC mode
- **Cell 4**: replaced the Azure-Datastore-based `master_manifest.csv` fetch with
  a local-file read; fixed a `visit_date` extraction bug (`extract_visit_date`
  required an exact 10-character `YYYY-MM-DD` path segment, but real `scan_dir`
  segments are like `2007-06-05_12_04_39.0_I59318` — date+time+ID concatenated —
  so it silently returned `NaT` for all 1,549 rows every time). This same bug
  exists in the original CPU-era notebook too — any CPU-era Markov
  chain/Digital Twin output may have been built on all-`NaT` dates.
- **Removed** Cell 5 (raw-DICOM cross-tenant validation) and Cell 6 (3D volume
  validation) — irrelevant once training reads only from the local tensor_cache
- **Removed** the old "Cell 9b" (One-Time Preprocessing Cache) — rebuilds the
  tensor cache from raw DICOM; already done, and its hardcoded Linux AML-compute
  paths didn't match anything on the lab PC (would have silently rebuilt the cache
  in the wrong place for hours if run by accident)
- **Removed** the old "Cell 10c" (Monitor Running Job) — polls an async Azure ML
  compute job via `ws.get_mlflow_tracking_uri()`, which crashes with `ws=None`;
  not needed for a synchronous foreground run
- **Renumbered all 33 remaining cells sequentially 1→33**, fixing every
  cross-reference throughout (docstrings, print statements, "Prerequisites:
  Cells..." lists, dash-ranges, the run-order table in the intro cell). Resolved a
  pre-existing ambiguity where two unrelated recovery cells were both informally
  labeled "15b" — now distinct Cells 27 and 28.
- **Cell 8 (training loop) made resumable**: checkpoints now also store
  optimizer/scheduler/AMP-scaler state and a `fold_complete` flag. Re-running the
  cell skips fully-completed folds and resumes an interrupted fold from its last
  saved epoch instead of restarting from scratch. Added a compatibility check
  (`training_config.scheduler == 'OneCycleLR'`) so it correctly recognizes leftover
  CPU-baseline checkpoints as *not* resumable and overwrites them fresh, instead
  of trying to resume from a different run's model/hyperparameters.
- **Cell 9 (Kernel Recovery)**: fixed a hardcoded AML-compute checkpoint path
  (`/mnt/batch/tasks/...`) to the lab PC path
- Fixed a malformed Windows `file://` MLflow tracking URI
  (`f"file://{path}/mlruns"` isn't valid on Windows — missing slash before the
  drive letter) — switched to `Path.as_uri()`
- Fixed a `RecursionError`: Cell 3's `print` monkey-patch captured whatever
  `print` currently was, so re-running the cell in the same kernel wrapped the
  filter around itself repeatedly until it blew the recursion limit; added an
  idempotency guard attribute
- Fixed `weights_only=True` (PyTorch 2.6+'s new default) rejecting `torch.load()`
  of every checkpoint (embeds a `StandardScaler`) and every cached tensor
  (MONAI-wrapped objects) — added `weights_only=False` to every `torch.load()`
  call in the notebook
- Fixed a `ValueError` in `roc_auc_score` ("scores need to sum to 1.0") caused by
  computing `softmax` on float16 logits straight out of AMP `autocast` — cast to
  float32 before softmax in Cell 8 and Cell 13 (ablation training function, which
  also uses autocast and would have hit the same bug later)
- Suppressed a noisy but harmless `mlflow`/gitpython warning
  (`GIT_PYTHON_REFRESH=quiet`) — no git executable on the lab PC
- Set `NUM_WORKERS=0` — `num_workers>0` PyTorch DataLoaders inside a Jupyter
  kernel on Windows use spawn-based multiprocessing, which was hanging
  indefinitely on worker startup; data is pre-cached `.pt` tensors, so parallel
  workers weren't worth the risk
- Fixed a pre-existing `SyntaxError` in Cell 1 (Install Libraries): a second
  `pip install ...` line was missing its `#` comment marker
- **Added Cell 8B** (new, optional, inserted right after Cell 8): identical
  model/data/OneCycleLR/AMP/patience-5 setup to Cell 8, but checkpoint
  selection and early-stopping patience are driven by **maximum validation
  AUC** instead of minimum validation loss. Writes to separate
  `best_model_fold{N}_aucsel.pth` files so Cell 8's loss-selected checkpoints
  are never overwritten; resumable the same way Cell 8 is. Prints a
  loss-selected-vs-AUC-selected comparison table at the end if Cell 8's
  `fold_results` is still in the kernel. **Run to completion** — see the
  Cell 8B results table under Current State above, plus the selection-bias
  analysis of why the loss-selected run remains the primary result.
- **Cell 13 hardened against NaN**: `A3_CNN_MLP_Fusion` crashed twice with
  `ValueError: Input contains NaN` (fp16 overflow under AMP autocast,
  producing non-finite logits that failed `roc_auc_score`'s strict
  finite-value check). Added: skip any training batch whose loss is
  non-finite instead of letting it corrupt the model's weights; gradient
  clipping (`max_norm=1.0`) after unscaling AMP gradients; treat a
  non-finite validation-probability epoch as non-improving (counts against
  patience) instead of raising. Third attempt (with both guards active)
  completed cleanly (AUC=0.9389), so the fp32 fallback below wasn't needed
  for this run, but is kept as a safety net.
- **Cell 14 made resilient to a single variant failing**: wrapped the
  `run_ablation_variant(...)` call in `try/except` — a variant that still
  fails is logged and skipped (not added to `completed`, so a later rerun
  retries just that one) instead of crashing the whole loop and taking down
  every variant after it.
- **Added Cell 14B** (new, inserted right after Cell 14, same
  non-renumbering pattern as Cell 8B): retries any variant still missing
  from `completed` with `USE_AMP` forced to `False` for that retry only,
  removing the fp16-overflow path entirely. Ended up unneeded this run
  (A3 succeeded on retry under Cell 13's guards) but stays in the notebook
  as a one-click fallback if a future variant fails outright.
- **Cell 24 (`NeuroDT` class)**: `_get_diagnosis_probs` now returns
  `(probs, image_available)` instead of just `probs`; `predict_patient()`
  and `simulate_intervention()` propagate `image_available` in their
  result dicts, so every caller (Cell 33, and `dashboard.py` in the
  future) can tell a real image-based prediction apart from a
  tabular-only zero-tensor fallback instead of it failing silently.
- **Cell 30 (age sensitivity)**: rewritten to check 5 unique patients
  instead of 1, to distinguish a real bug from a weak/noisy tabular
  feature. Also fixed a duplicate-`patient_id` bug in this new check
  itself (dedup on `patient_id` before sampling).
- **Cell 33 (single-patient inference)**: surfaces `image_available` as
  an unmissable warning at every point it matters — console banner, the
  predicted-class line, the saved figure's title (rendered in red), and
  the clinical summary header.

### `dashboard.py` (`Azure/dashboard.py`)
- Fixed `load_assets()` missing `weights_only=False` — would crash loading
  *any* checkpoint (CPU or GPU) on PyTorch 2.6+, since these checkpoints
  embed a `StandardScaler`. Same bug as the notebook's `torch.load()` fix,
  never applied here.
- Fixed the identical silent zero-tensor fallback in `run_inference()` —
  tracks `image_available`, surfaces a warning immediately after inference
  and as a persistent `st.error()` banner on every rerun (session-state
  backed, matching this file's existing state-persistence pattern).
- Replaced every hardcoded CPU-era metric (AUC 0.912, per-class
  0.957/0.936/0.844, 79% accuracy) with the real GPU numbers — Fold 4 AUC
  0.951 (per-class CN 0.982 / Dementia 0.958 / MCI 0.913), 85% accuracy,
  0.85 macro F1 — and added the honest 5-fold CV estimate (0.871 ± 0.046)
  alongside Fold 4's number everywhere it's shown (header, sidebar, About
  tab, both PDF reports), so a single best fold isn't presented as the
  model's general expected performance.
- **Not yet done**: the checkpoint files themselves haven't been copied
  into `dashboard.py`'s `CHECKPOINT_DIR` — see Next Steps.

---

## Failed attempts (things you tried but didn't work and why)

| What was tried | Why it failed | Fix |
|---|---|---|
| Running `verify_downloads.py` in the lightweight `.venv` | Only has `azure-storage-blob`/`azure-identity`; missing `monai`/`sklearn`/`numpy` needed to unpickle the cached objects | Run it in `bdt-env` instead |
| Running Cell 8 as originally written | `torch.load()` defaults to `weights_only=True` on PyTorch 2.6+, rejects the embedded `StandardScaler`/MONAI objects | Added `weights_only=False` everywhere |
| Running Cell 8 after that fix | Malformed `file://{path}/mlruns` URI on Windows crashed `mlflow.set_experiment()` | Switched to `Path.as_uri()` |
| Re-running Cell 3 after a kernel restart | Print monkey-patch wrapped itself repeatedly across runs → `RecursionError` | Added idempotency guard |
| Running Cell 8 with `NUM_WORKERS=4` | DataLoader worker-spawn hangs indefinitely — known Windows+Jupyter+multiprocessing issue | Set `NUM_WORKERS=0` |
| Running Cell 8 after the hang fix | `torch.softmax()` on float16 AMP logits lost enough precision that row-sums failed sklearn's strict `roc_auc_score` check | Cast logits to float32 before softmax |
| First resume-check design in Cell 8 | Didn't distinguish a genuine in-progress GPU checkpoint from a leftover CPU-baseline checkpoint with the same filename — would have "resumed" fold 1 from the wrong model entirely | Added `training_config.scheduler == 'OneCycleLR'` compatibility check |
| Suspected a 4,207-item folder found on the laptop might be (or substitute for) the tensor cache | It's ADNI raw metadata (XML sidecars + empty subject dirs) — 9.8 MB total, no actual imaging data | Confirmed harmless if left in place (no filename/extension collision with the real cache); not a substitute for the real download |
| Several `download_files.py` runs on university wifi | `ConnectionResetError` / blob listing failures from an unreliable network | Retry-with-backoff on both downloads and listing; not a dead end, just needed resilience |

---

## Next Steps (things to try next)

### Immediate
0. **Re-run Cell 30 once** (fast, no retraining) — picks up the dedup fix so
   its multi-patient summary count is correct (was "1/4", should read
   "N/5" after the fix). Then **copy `checkpoints\` (final, complete
   version) and `tensor_cache\` to the flash drive** — nothing else is
   writing to either folder at this point. See
   `Continue_From_Home_Guide.md` for what to do with them next.
0b. Copy `best_model_fold4.pth` + `markov_matrices.pkl` into
   `dashboard.py`'s `CHECKPOINT_DIR` when ready to switch the dashboard to
   the GPU model — no other dashboard changes needed, the code-side fixes
   are already done (see Changed, above).
1. ~~Let Cell 8 finish Fold 5~~ — **done.** Loss-selected 5-fold CV mean AUC:
   **0.8706 ± 0.0461** (Folds: 0.8821, 0.8591, 0.8125, 0.9511, 0.8481).
2. Run **Cell 11 (COMPARE)** for the formal writeup of the GPU 5-fold mean AUC
   ± std vs. the CPU's single-fold 0.9120 — this is the actual fair comparison,
   not any individual fold along the way. Use the **loss-selected** number
   (0.8706 ± 0.0461) as the headline GPU result, not the AUC-selected one.
2b. ~~Run Cell 8B (AUC-based checkpoint selection)~~ — **done.** Mean AUC
   0.9379 ± 0.0069 — see the comparison table and selection-bias analysis
   under **Current State**. Conclusion: keep this as a methodology discussion
   point (validation-metric selection bias, why loss-based early stopping is
   more defensible), not as a replacement headline number.
3. ~~Track down `ablation_results.json`~~ — moot. Cell 14's new path
   (`checkpoints/ablation/ablation_results.json`) is separate from whatever
   CPU-era file was expected at the top-level `checkpoints/` path, so all 7
   DL variants were trained fresh on GPU regardless. See the completed
   ablation study results below.

### Ablation study — COMPLETE (all 7 DL variants + 4 classical baselines)

All trained/evaluated on the identical Fold 4 split (n=310 validation
scans), matching the main run's best fold:

| Model | AUC (Macro) | Accuracy | Macro F1 | AUC CN | AUC MCI | AUC Dementia |
|---|---|---|---|---|---|---|
| A5_Transformer_1Layer | 0.9563 | 0.8516 | 0.8552 | 0.9877 | 0.9228 | 0.9586 |
| A4_CNN_Linear_Fusion | 0.9487 | 0.8419 | 0.8454 | 0.9803 | 0.9132 | 0.9526 |
| **A0_NeuroDT_Full** | **0.9486** | 0.8516 | 0.8550 | 0.9815 | 0.9069 | 0.9575 |
| A2_CNN_Only | 0.9474 | 0.8452 | 0.8497 | 0.9816 | 0.9163 | 0.9442 |
| B3_Random_Forest | 0.9428 | 0.8258 | 0.8298 | 0.9825 | 0.9002 | 0.9458 |
| B4_Gradient_Boosting | 0.9407 | 0.8226 | 0.8253 | 0.9801 | 0.8977 | 0.9442 |
| A3_CNN_MLP_Fusion | 0.9389 | 0.8387 | 0.8436 | 0.9802 | 0.8970 | 0.9396 |
| A6_No_Class_Weights | 0.9294 | 0.7839 | 0.7906 | 0.9745 | 0.8666 | 0.9472 |
| A1_Tabular_Only | 0.8716 | 0.6839 | 0.6750 | 0.9193 | 0.7717 | 0.9237 |
| B2_SVM_RBF | 0.8711 | 0.7355 | 0.7348 | 0.9337 | 0.7686 | 0.9109 |
| B1_Logistic_Regression | 0.8649 | 0.7097 | 0.7073 | 0.9105 | 0.7589 | 0.9255 |

Saved to `checkpoints/ablation/ablation_results_table.csv`, plus
`fig_ablation_auc_comparison.png` and `fig_ablation_perclass_auc.png`.

**Read with caution — this is single-fold (n=310), not 5-fold CV.** The top
5 models (A0, A2, A4, A5, B3) all sit within ~0.01 AUC of each other —
that's normal fold-level noise on 310 samples, not a real ranking. Don't
report "A5 beats the full model" as a finding; the F1 delta (+0.0002) is
essentially zero.

Two findings worth real discussion-section space:

1. **CNN-only (A2) ≈ full multimodal (A0)** — 0.9474 vs 0.9486. Dropping the
   tabular branch costs almost nothing, i.e. the imaging branch alone
   carries nearly all the signal in this fusion design.
2. **Random Forest/Gradient Boosting on the same 4 tabular features
   (0.9428/0.9407) beat the deep Tabular-only MLP (A1, 0.8716) by ~7 points
   of AUC** — a capacity/architecture mismatch (a large MLP is likely
   poorly suited to a 4-dimensional input; tree ensembles are naturally
   strong on small tabular sets). Separately: one of those 4 features is
   **MMSE**, which is itself part of ADNI's clinical criteria for
   diagnosing MCI/Dementia — so ~0.94 AUC from 4 features including a
   near-diagnostic-criterion score is a legitimate label-leakage caveat to
   name explicitly, not evidence that tabular data is unexpectedly
   powerful on its own.

### Post-training pipeline — ALL CELLS (18–33) COMPLETE
4. ~~Cell 9 → 18 → 20 → 22 → 24 → 26–32 → 33~~ — **done, full notebook run
   end to end.** Two items below are flagged for follow-up before anything
   here goes into the thesis or `dashboard.py` as-is — see items 4c and 4g.

   - **Cell 18 (evaluation)**: macro AUC by class — CN 0.982, Dementia 0.958,
     MCI 0.913 (averages to ≈0.951, consistent with Fold 4's loaded
     checkpoint, val_AUC=0.9511 — good cross-check). Confusion matrix and
     SHAP summary saved. **Check**: SHAP plot only shows 3 of 4 tabular
     features (no APOE4 bar) — could be genuinely near-zero contribution
     once imaging + other features are in, or the plotting code could be
     dropping it silently. Worth a quick look, not urgent.
   - **Cell 20 (Grad-CAM)**: one example (patient 057_S_1373, Dementia,
     correctly predicted) shows a fairly **diffuse, whole-hemisphere**
     activation pattern rather than a focal hippocampal/medial-temporal-lobe
     region (the classic AD-relevant atrophy site). One example proves
     nothing either way — run it on several more patients (correct and
     incorrect, all 3 classes) before drawing conclusions. If the pattern
     stays diffuse rather than converging on temporal-lobe/hippocampal
     regions across many patients, that's a real caveat for the thesis
     discussion on learned-vs-shortcut features, tying back to the earlier
     "is it learning or cheating" question from this session.
   - **Cell 22 (Markov chain)**: good clinical face-validity. Dementia is a
     correctly absorbing state (0.997–1.00 self-transition). **APOE4-positive
     patients show a higher MCI→Dementia transition rate (0.184) than
     APOE4-negative (0.133)** — the correct clinical direction (APOE4 is a
     known progression risk factor), reproduced without being told to.
     Worth citing as external/face validation in the write-up.
   - **Cell 26 (what-if: APOE4 + drug intervention)**: Lecanemab/Donanemab
     bring 5-year Dementia probability (patient 016_S_1149) to 12.1%/11.9%,
     *below* even the no-APOE4 baseline (14.5%). Worth double-checking the
     assumed effect-size parameters (30%/35%) against real trial data, and
     stating explicitly in the thesis that these are simulated/assumed
     effect sizes, not calibrated outcomes — otherwise it reads as an
     overclaim.
   - **Cell 29 (cognitive reserve)**: lower education → higher risk
     (16.9% vs 15.2% at year 5) — correct direction, good face validity.
   - **Cell 30 (age sensitivity) — RESOLVED, not a bug.** Rewrote the cell to
     sweep AGE across 5 sample MCI patients instead of 1. Result: only 1 of 4
     unique patients showed risk falling with age; the rest were mixed
     (one patient increased, one decreased, one had inconsistent direction
     across two of their own visits). **Direction varies by patient — this
     is a weak/noisy AGE signal in the tabular branch, not a systematic
     code bug**, consistent with the ablation finding that the tabular
     branch barely matters once imaging is present (A2_CNN_Only ≈ A0_Full).
     Conclusion for the thesis: do not present any single patient's
     age-sweep as a general finding; report this instability explicitly as
     a limitation of the tabular AGE feature's weak/inconsistent learned
     effect, distinct from the imaging branch's real signal.
     — Found and fixed a **counting bug in this diagnostic itself**: the
     first 5 rows of `df_val_best`'s MCI patients included the same
     `patient_id` twice (two different visits/scans), which silently
     collapsed to 4 entries in the results dict and made the summary read
     "1/4" — logically correct but based on 4 unique patients, not 5.
     Fixed by deduplicating on `patient_id` before sampling. **Re-run Cell
     30 once more** (fast, no retraining) to get the corrected 5-unique-
     patient count before citing any number from it.
   - **Cell 31 (subgroup comparison)**: CN (26.3%) < MCI (64.4%) < Dementia
     (95.1%) at year 5 — correctly ordered, good face validity.
   - **Cell 32 (early vs. late intervention)**: earlier treatment → lower
     risk (95.7% vs 96.5% vs 97.3% untreated, patient 128_S_0167) — correct
     direction, though this patient is already at ~97% baseline risk,
     leaving little room to show a bigger effect. Consider picking a less
     severe example patient for the actual thesis figure.
   - **Cell 33 (single-patient inference) — FIXED and confirmed working.**
     Root cause lived in Cell 24's `NeuroDT._get_diagnosis_probs`, which
     silently substituted a zero image tensor with no signal to the
     caller. Fixed: `_get_diagnosis_probs` now returns
     `(probs, image_available)`; `predict_patient()` and
     `simulate_intervention()` propagate `image_available` in their result
     dicts; Cell 33 checks it after every call (Step 1 classifier, and
     separately for the 4 drug/age/education scenario calls, since those
     go through a different internal cache lookup). Re-ran with the same
     no-scan demo patient — confirmed working: prints an unmissable
     warning banner before the diagnosis, appends `[TABULAR-ONLY -- NO
     SCAN]` to the predicted-class line, the saved figure's title renders
     in red with the same suffix, and the clinical summary leads with an
     explicit "TABULAR-ONLY" warning. `dashboard.py` got the identical fix
     (see Changed, below) since it has the exact same fallback pattern in
     its own `run_inference()`.
     — Noted but not fixed (non-blocking): the Monte Carlo simulation
     (`_monte_carlo`, n=1,000 draws) has no fixed random seed, so re-running
     Cell 33 on the same patient gives slightly different numbers each time
     (e.g. baseline 5yr risk 85.7% → 84.9% between two runs). Not a
     correctness issue, but worth a `np.random.seed(...)` before citing an
     exact figure in the thesis, so the number in the write-up matches
     whatever's in the saved PNG.

### Worth deciding on
6. ~~Whether to run a second, AUC-based-checkpoint-selection training run~~ —
   **done** (Cell 8B). Decision: report loss-selected (0.8706 ± 0.0461) as the
   primary thesis number; write up the AUC-selected run (0.9379 ± 0.0069) as a
   discussion section on validation-metric selection bias — every AUC-selected
   fold trained the full 20 epochs into visible train-set memorization
   (tr_acc≈0.97–0.99, tr_loss≈0.05–0.09 by ep.18–20) with no early stop,
   then had its checkpoint picked by argmax of a noisy metric on a ~310-sample
   validation fold, which both inflates the mean and artificially compresses
   the reported std.
7. Whether to regenerate the CPU-era Markov chain/Digital Twin figures
   (`markov_heatmap.png`, `subgroup_trajectories.png`, `markov_matrices.pkl`) —
   they may have been built on the same broken all-`NaT` `visit_date` bug found
   in Cell 4.
8. Mention Fold 3's weak performance (AUC 0.8125, early-stopped at epoch 7) in
   the thesis's results/discussion as an example of genuine fold-to-fold variance.

### Outstanding hygiene
9. **Rotate the Azure client secret** — still sitting in plaintext in this repo's
   git history, unrotated since flagged at the start of this session.
10. Delete the merged feature branch `claude/gpu-accuracy-testing-local-7ys9vo`
    from GitHub if desired — a prior attempt hit a 403 through this session's git
    proxy; needs to be done from the GitHub UI directly.
