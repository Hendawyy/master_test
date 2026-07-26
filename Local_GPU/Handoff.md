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

### GPU Lab Machine — READY, actively training
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
**Cell 8B** — see **Changed**, below. Not yet run; the lab PC has the code
in hand and will run it once free.

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
  `fold_results` is still in the kernel. Not yet run.

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
1. ~~Let Cell 8 finish Fold 5~~ — **done.** Loss-selected 5-fold CV mean AUC:
   **0.8706 ± 0.0461** (Folds: 0.8821, 0.8591, 0.8125, 0.9511, 0.8481).
2. Run **Cell 11 (COMPARE)** for the formal writeup of the GPU 5-fold mean AUC
   ± std vs. the CPU's single-fold 0.9120 — this is the actual fair comparison,
   not any individual fold along the way.
2b. Run the new **Cell 8B** (AUC-based checkpoint selection) — a second,
   separate 5-fold run to check whether the AUC-based selection closes the gap
   seen in Folds 1–3 (see **Current State**, above). Writes to distinct
   `*_aucsel.pth` files, so it's safe to run without disturbing the completed
   loss-selected results. Prints a side-by-side comparison table at the end.
3. Track down `ablation_results.json` — confirmed missing from the Azure blob
   transfer on two independent download attempts. Check the Azure Portal/Storage
   Explorer for `adni-data/gpu_transfer/checkpoints/` directly before running
   Cell 14, so the completed CPU ablation results (A0, A1, B1–B4) aren't redone.

### Post-training pipeline (run in order)
4. Cell 9 (recovery, only if the kernel died) → **18** (evaluation) → **20**
   (Grad-CAM) → **22** (Markov chain — now safe, given the `visit_date` fix) →
   **24** (Digital Twin assembly) → **26–32** (simulations) → **33** (single
   patient inference — the cell relevant to `dashboard.py`)
5. Ablation study: **13** → **14** (DL variants A2–A6, ~3–4 hrs) → **15**
   (classical ML baselines) → **16** (compile results table) — this is the work
   the CPU run couldn't do at all.

### Worth deciding on
6. Whether to run a **second**, separate AUC-based-checkpoint-selection training
   run for comparison — only worth the extra ~1 hour if the loss-based 5-fold
   mean disappoints. Not a 3-way comparison; a clean two-run comparison if done.
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
