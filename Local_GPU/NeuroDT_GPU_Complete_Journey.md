# Neuro-DT: Complete GPU Migration Journey
## From CPU Bottleneck to a Full Cross-Validated Result — Every Step, Every Error
### Seif Hendawy — Arab Academy for Science, Technology and Maritime Transport
### Supervisors: Prof. Fahima Maghraby · Assoc. Prof. Ahmed Salem

---

## PHASE 1: Why GPU Migration Was Necessary

The Azure CPU pipeline (`Standard_E4ds_v4`, 4 cores, 32 GB RAM, CPU only) completed
only **1 of 5** cross-validation folds (Fold 4, AUC 0.9120) before hitting a
training-instability failure mode in the other 4 folds (`CosineAnnealingLR` with
no warmup spiking validation loss in epoch 1–3, triggering early stopping before
any useful learning happened). The full deep-learning ablation study (A2–A6) never
ran at all — infeasible on CPU (projected 69–129 hours for that study alone).

**Decision:** move the same codebase to a university lab PC with a GPU, and
complete: (1) a genuine 5-fold cross-validated result, (2) the full ablation
study, (3) the complete Digital Twin pipeline end to end.

---

## PHASE 2: GPU Lab Machine Setup

### Step 1 — Machine specs
- University lab PC, Windows
- GPU: NVIDIA RTX 5070 Ti — 17.1 GB VRAM
- Conda environment: `bdt-env`, Python 3.10
- PyTorch: `2.11.0+cu128`, CUDA confirmed working
- Constraint that shaped the whole session: **the lab closes at 2pm daily** —
  every long-running cell had to be interruption-safe.

### Step 2 — Transfer data from Azure to the lab PC
`download_files.py` (originally a simple blob-download script) was hardened
iteratively against unreliable university wifi:
- Atomic writes: download to `dest.name + ".part"`, rename only on success —
  an interrupted download can never leave a file a later run mistakes for
  complete.
- Retry-with-exponential-backoff (`MAX_RETRIES=4`, `2**attempt` seconds) on
  both individual blob downloads **and** blob *listing* (`list_blobs()` is a
  lazy pager — it can drop mid-page on a flaky connection too).
- Auto-cleanup of stale `.part`/zero-byte files at the start of each run.
- Manifest support (`already_downloaded.txt`) so a second machine could skip
  files already fetched elsewhere, for cross-machine deduplication.
- `BlobServiceClient(..., retry_total=8, retry_connect=8)` at the SDK level.

**Problem encountered:** `master_manifest.csv` was never part of the original
`gpu_transfer` blob transfer — it lived only in Azure ML's own
`workspaceblobstore` datastore, reachable only from an AML compute session,
not from the lab PC directly.
**Fix:** wrote `fetch_and_upload_manifest.py`, run once from the Azure ML
DevBox: pulled `master_manifest.csv` + `bad_scans.txt` from the AML
datastore, then re-uploaded them into the same `adni-data/gpu_transfer/
checkpoints/` blob path `download_files.py` already scanned — the lab PC
picked them up with zero script changes on its next run.

### Step 3 — Verify the download
Wrote `verify_downloads.py`: actually `torch.load()`s every cached tensor and
checkpoint (not just checking file existence) to catch corruption that
finishes writing but is still bad.

**Problem encountered:** ran it twice in a lightweight `.venv` (missing
`monai`/`sklearn`/`numpy`) instead of `bdt-env` — both runs reported false
"CORRUPT" results.
**Fix:** re-ran in `bdt-env` — confirmed genuinely clean: **1,549/1,549**
tensor cache files and all checkpoints loaded successfully.

---

## PHASE 3: Adapting the Notebook for Lab-PC Mode

The original notebook was built entirely around a live Azure ML Workspace
connection. For the lab PC (no Azure ML access), every workspace/datastore
dependency had to be removed or shimmed, and the notebook renumbered after
cell removals.

### Removed entirely (irrelevant once training reads only from local files)
- **Data Access Validation** — cross-tenant blob check
- **3D Volume Validation** — redundant with the tensor cache already built
- **One-Time Preprocessing Cache** — cache already complete; its hardcoded
  Linux AML-compute paths didn't match anything on the lab PC and would have
  silently rebuilt the cache in the wrong place for hours if run by accident
- **Monitor Running Job** — polls an async Azure ML job via
  `ws.get_mlflow_tracking_uri()`, which crashes with `ws=None`; not needed
  for a synchronous foreground run

### Replaced
- Real `Workspace.from_config()`/`MLClient.from_config()` → `ws=None;
  ml_client=None`, with an explicit "lab PC mode" print.
- Azure-Datastore-based `master_manifest.csv` fetch → plain local file read.

### Bug found while adapting Cell 4 (Golden DataFrame)
```python
# BROKEN (original):
def extract_visit_date(scan_dir):
    for part in str(scan_dir).split("/"):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return pd.to_datetime(part, errors="coerce")
    return pd.NaT
```
Real `scan_dir` segments look like `2007-06-05_12_04_39.0_I59318` (date +
time + image ID concatenated, 14+ characters) — the `len(part) == 10` check
**never matched**, silently returning `NaT` for all 1,549 rows.
**Fix:**
```python
def extract_visit_date(scan_dir):
    for part in str(scan_dir).split("/"):
        if len(part) >= 10 and part[4] == "-" and part[7] == "-":
            return pd.to_datetime(part[:10], errors="coerce")
    return pd.NaT
```
Result after fix: **1,549/1,549** dates extracted. This bug predates the GPU
work — it would have silently broken any Markov chain/Digital Twin output
built on the original CPU-side code too.

### Renumbering
75 → 67 cells after 4 cell-pair removals; all 33 markdown section headers
renumbered sequentially 1→33 (later becoming 1→33 plus non-renumbered `8B`
and `14B` insertions — see Phase 5), every "Prerequisites: Cells..." list and
cross-reference updated to match.

---

## PHASE 4: Main Training — Five Sequential Bugs, Then a Resumable Design

### The resumability requirement
Because the lab closes at 2pm, the training loop (Cell 8) was rebuilt to
survive interruption: checkpoints store optimizer/scheduler/AMP-scaler
state and a `fold_complete` flag; re-running the cell skips completed folds
and resumes an interrupted fold from its last saved epoch. A compatibility
guard (`training_config.scheduler == 'OneCycleLR'`) rejects leftover
CPU-baseline checkpoints with the same filenames instead of trying to
"resume" from an incompatible model/hyperparameter set.

### Five crashes hit and fixed in sequence, first real run
1. **`MlflowException: file://...\mlruns is not a valid remote uri`** —
   malformed Windows file URI (`f"file://{path}/mlruns"`, missing slash
   before the drive letter). **Fix:** `Path.as_uri()`.
2. **`RecursionError`** in the print-filtering monkeypatch after a kernel
   restart — it captured whatever `print` currently was, so re-running the
   cell wrapped the filter around itself repeatedly. **Fix:** idempotency
   guard attribute.
3. **`UnpicklingError: ... GLOBAL sklearn.preprocessing._data.StandardScaler`**
   — PyTorch 2.6+'s new `weights_only=True` default rejects checkpoints that
   embed a fitted `StandardScaler`. **Fix:** `weights_only=False` added to
   every `torch.load()` call across the entire notebook.
4. **DataLoader hang** — stuck at `0/78` for 4+ minutes, kernel "Busy," zero
   progress. Root cause: `num_workers>0` PyTorch DataLoaders inside a
   Jupyter kernel on Windows use spawn-based multiprocessing, which hangs
   indefinitely on worker startup. **Fix:** `NUM_WORKERS=0` — data is
   pre-cached `.pt` tensors, so parallel workers weren't worth the risk.
5. **`ValueError: Target scores need to be probabilities ... sum up to 1.0`**
   in `roc_auc_score` — float16 softmax precision loss straight out of AMP
   `autocast`. **Fix:** cast logits to float32 before softmax.

### Final result — Cell 8, loss-selected, all 5 folds complete

| Fold | Status | Best (loss-selected) val AUC | Notes |
|---|---|---|---|
| 1 | Early-stopped ep.13 | 0.8821 | Peak AUC seen was 0.9180 @ ep.13 (worse loss, not selected) |
| 2 | Early-stopped ep.13 | 0.8591 | Epoch 9 had a transient instability spike, self-corrected by ep.10 |
| 3 | Early-stopped ep.7 | 0.8125 | Weakest fold — plateaued fast, likely genuine fold-to-fold variance |
| 4 | Ran full 20 epochs, no early stop | 0.9511 | Strongest fold — smooth monotonic improvement |
| 5 | Early-stopped ep.10 | 0.8481 | — |

**Final 5-fold CV mean AUC: 0.8706 ± 0.0461** — the first genuinely
cross-validated estimate this project produced.

**Direct comparison to CPU** (Cell 11, same `StratifiedKFold(random_state=42)`
split, so Fold 4 is literally the same patients on both machines):

| | CPU | GPU |
|---|---|---|
| Fold 4 AUC | 0.9120 | **0.9511** |
| Folds completed | 1 of 5 | 5 of 5 |
| Reported metric | Single fold, not cross-validated | 5-fold CV mean: **0.8706 ± 0.0461** |

---

## PHASE 5: A Second Training Run to Test the Methodology Itself

**Question raised:** should checkpoint selection be based on minimum
validation loss (Cell 8) or maximum validation AUC? Rather than guess,
**Cell 8B** was built: an identical training setup, differing only in using
max validation AUC as both the checkpoint-selection and early-stopping
criterion, writing to separate `_aucsel.pth` files so Cell 8's results were
never disturbed.

### Result — AUC-selected, all 5 folds

| Fold | Loss-selected AUC | AUC-selected AUC | Δ |
|---|---|---|---|
| 1 | 0.8821 | 0.9439 | +0.0618 |
| 2 | 0.8591 | 0.9398 | +0.0808 |
| 3 | 0.8125 | 0.9265 | +0.1140 |
| 4 | 0.9511 | 0.9451 | −0.0059 |
| 5 | 0.8481 | 0.9342 | +0.0861 |
| **Mean** | **0.8706 ± 0.0461** | **0.9379 ± 0.0069** | — |

**Why the AUC-selected number is inflated, not better:** all 5 AUC-selected
folds ran the full 20 epochs with **no early stop** (AUC rarely gets worse
for 5 consecutive epochs on a small validation set the way loss does), so
every fold trained deep into train-set memorization (`tr_acc≈0.97–0.99,
tr_loss≈0.05–0.09` by epoch 18–20) before the checkpoint was selected. The
selected checkpoint is also the argmax of a noisy metric on a ~310-sample
validation fold — selecting on the same metric being reported is a biased,
optimistic estimator. This also explains the AUC-selected run's suspiciously
*tighter* std (0.0069 vs. 0.0461) — cherry-picking a per-epoch max
compresses apparent variance rather than reflecting real consistency.

**Conclusion, adopted as the project's methodology:** report the
loss-selected result (0.8706 ± 0.0461) as the primary, defensible number.
Keep the AUC-selected run as an explicit thesis discussion point on
validation-metric selection bias.

---

## PHASE 6: Ablation Study — The NaN Crash Saga

**Structure:** 7 deep-learning variants (A0 full model, A1 tabular-only, A2
CNN-only, A3 CNN+MLP fusion, A4 CNN+linear fusion, A5 1-layer Transformer,
A6 no class weights), each trained on the single representative Fold 4 split
(not full 5-fold CV, to keep total runtime manageable), plus 4 classical ML
baselines (Logistic Regression, SVM, Random Forest, Gradient Boosting) on
tabular features only.

### Crash 1
A3_CNN_MLP_Fusion reached epoch 17 (val AUC still climbing, 0.9325) then:
```
ValueError: Input contains NaN.
```
fp16 overflow under AMP autocast in that fusion architecture, producing
non-finite logits that failed `roc_auc_score`'s strict finite-value check —
and crashed the **entire Cell 14 loop**, taking A4/A5/A6 down with it even
though they never got a chance to run.

**Fix, round 1 (Cell 13 hardened):**
- Skip any training batch whose loss is non-finite instead of letting it
  corrupt the model's weights.
- Gradient clipping (`clip_grad_norm_`, `max_norm=1.0`) after unscaling AMP
  gradients.
- Treat a non-finite validation-probability epoch as non-improving (counts
  against patience) instead of raising.

**Fix, round 2 (Cell 14 made resilient):** wrapped the
`run_ablation_variant(...)` call in `try/except` — a variant that still
fails is logged and skipped (not marked complete, so a rerun retries just
that one) instead of stopping the whole loop.

**Fix, round 3 (Cell 14B added):** a follow-up cell that retries any
variant still missing from `completed` with `USE_AMP` forced to `False`,
removing the fp16-overflow path entirely — kept as a one-click fallback.

### Crash 2 — same variant, different epoch
A3 restarted from epoch 1 (per-variant checkpointing only, no per-epoch
resume within a variant) and hit the same NaN at epoch 17→18, this time with
the Cell 13 guards active. Cell 14's try/except caught it correctly this
time, logged the failure, and moved straight on to A4 — which succeeded.

### Crash 3 — third attempt, success
A3 restarted a second time and completed all 20 epochs cleanly
(AUC=0.9389) — the batch-skip + gradient-clipping guards held.

### Final ablation results (all 7 DL variants + 4 classical baselines)

| Model | AUC (Macro) | Accuracy | Macro F1 |
|---|---|---|---|
| A5_Transformer_1Layer | 0.9563 | 0.8516 | 0.8552 |
| A4_CNN_Linear_Fusion | 0.9487 | 0.8419 | 0.8454 |
| **A0_NeuroDT_Full** | **0.9486** | 0.8516 | 0.8550 |
| A2_CNN_Only | 0.9474 | 0.8452 | 0.8497 |
| B3_Random_Forest | 0.9428 | 0.8258 | 0.8298 |
| B4_Gradient_Boosting | 0.9407 | 0.8226 | 0.8253 |
| A3_CNN_MLP_Fusion | 0.9389 | 0.8387 | 0.8436 |
| A6_No_Class_Weights | 0.9294 | 0.7839 | 0.7906 |
| A1_Tabular_Only | 0.8716 | 0.6839 | 0.6750 |
| B2_SVM_RBF | 0.8711 | 0.7355 | 0.7348 |
| B1_Logistic_Regression | 0.8649 | 0.7097 | 0.7073 |

Total wall-clock for all 7 DL variants, this run: **29.9 minutes** — versus
the CPU-side ablation study, which never ran at all for the DL variants.

**Read with caution:** single-fold (n=310), not 5-fold CV — the top 5
models sit within ~0.01 AUC of each other, normal fold-level noise, not a
meaningful ranking.

**Finding 1:** CNN-only (0.9474) ≈ full model (0.9486) — the tabular branch
barely matters once imaging is present.

**Finding 2:** Random Forest/Gradient Boosting (~0.94) beat the deep
tabular-only MLP (0.8716) on the *identical* 4 features — an
architecture-capacity mismatch, reported with an explicit caveat that one of
those features (MMSE) is itself part of ADNI's clinical diagnostic criteria,
so this isn't simply "tabular data is unexpectedly powerful."

---

## PHASE 7: Full Post-Training & Digital Twin Pipeline

All of Cells 18–33 ran successfully end to end — evaluation, Grad-CAM,
Markov chain, Digital Twin assembly, and every what-if simulation. None of
this was ever reachable on CPU, since training never finished there.

### Cell 18 — Evaluation
Per-class AUC on Fold 4's held-out 310 scans: CN 0.9819, Dementia 0.9583,
MCI 0.9131 (macro ≈0.951, consistent with the checkpoint's own val_AUC of
0.9511 — a good internal cross-check). Accuracy 85%, macro F1 0.85.

### Cell 20 — Grad-CAM
One example examined (patient 057_S_1373, Dementia, correctly predicted)
showed a fairly diffuse, whole-hemisphere activation pattern rather than a
focal hippocampal/medial-temporal-lobe region. Flagged for further review
across more patients before drawing conclusions about learned-vs-shortcut
features — not yet resolved.

### Cell 22 — Markov chain
Dementia correctly modeled as an absorbing state (0.997–1.00
self-transition). **APOE4-positive patients show a higher MCI→Dementia
transition rate (0.184) than APOE4-negative (0.133)** — the correct clinical
direction, reproduced without being told to. A genuine external-validity
result worth citing.

### Cell 26 — What-if drug simulation
Lecanemab/Donanemab brought 5-year Dementia probability below even the
no-APOE4 baseline in the example patient — flagged as needing the assumed
effect-size parameters (30%/35%) stated explicitly as simulated, not
calibrated to real trial data, to avoid overclaiming.

### Cell 30 — Age sensitivity investigation
Initial single-patient sweep showed dementia risk *decreasing* with
simulated age (65→69.8%, 85→16.5%) — clinically backwards. Rewrote the cell
to check 5 unique patients:

```
016_S_1149: decreases with age
099_S_0880: decreases with age
057_S_1007: INCREASES with age
109_S_1114: INCREASES with age
128_S_0167: decreases with age (pinned near 99–100% at every age)
```

**2/5 patients show the clinically expected increasing direction.**
Direction genuinely varies by patient — a weak/noisy AGE signal in the
tabular branch, not a systematic bug, consistent with Phase 6's Finding 1.

**A bug in the diagnostic itself, found and fixed along the way:** the
first version sampled 5 rows without deduplicating on `patient_id` — one
patient's two visits collided in the results dict, silently collapsing to 4
entries and misreporting "1/4" instead of "1/5." Fixed with
`.drop_duplicates(subset='patient_id')`; re-run confirmed correct.

### Cell 33 — Single-patient inference, a real deployment-risk bug
Root cause traced to `NeuroDT._get_diagnosis_probs` (Cell 24): when a
patient's scan wasn't in the tensor cache, it silently substituted a zero
image tensor and returned a normal-looking prediction with **zero
indication** anything was wrong.

**Fix:** `_get_diagnosis_probs` now returns `(probs, image_available)`;
`predict_patient()` and `simulate_intervention()` propagate this flag.
Cell 33 checks it at every point that matters — an unmissable console
warning, `[TABULAR-ONLY -- NO SCAN]` appended to the predicted-class line,
the saved figure's title rendered in red with the same suffix, and the
clinical summary leading with an explicit warning. Confirmed working
end-to-end on re-run with the same no-scan demo patient.

This matters directly for the live dashboard: a genuinely new patient (by
definition, not in `tensor_cache`) hits exactly this code path in
production.

---

## PHASE 8: Getting the Data Off the Lab PC

### Verification method
`tree` alone only shows folder structure, not file contents — insufficient
to trust. Used file-count + total-byte-size comparison instead:
```powershell
Get-ChildItem -Path <dir> -Recurse -File | Measure-Object -Property Length -Sum
```
Local: 1,757 files, 16,702,047,323 bytes. Flash drive: 1,765 files,
16,713,490,559 bytes. The 8-file/~10.9 MB difference traced to a pre-existing
`Local UNI` folder on the flash drive not present on the source machine —
unrelated to this transfer. `tensor_cache`: **1,549 = 1,549** exactly on
both sides — confirmed complete and byte-for-byte.

### Notebook state verification, three rounds
Compared the user's live, saved `.ipynb` against the pushed repo copy after
every major fix:
- Round 1: Cell 13's NaN-guard fix was missing from the saved file (paste
  hadn't been re-saved) — flagged, re-applied.
- Round 2: Cell 13 confirmed fixed, but Cell 30's dedup fix had regressed
  (reverted somehow between saves) — flagged, re-applied.
- Round 3: both confirmed correct, zero saved errors anywhere in the
  notebook, 71 cells, valid JSON. Cleared to copy and leave.

---

## PHASE 9: Dashboard Fixes (Discovered While Preparing to Switch to GPU Checkpoints)

### `weights_only` bug
`load_assets()`'s `torch.load(ckpt_path, map_location='cpu')` was missing
`weights_only=False` — the exact bug from Phase 4, never applied here. Would
crash loading *any* checkpoint (CPU or GPU) on PyTorch 2.6+.

### Missing `model.py`
`dashboard.py` does `from model import MultimodalTransformer`, but
`model.py` did not exist anywhere in the repository — an immediate
`ModuleNotFoundError` on load, regardless of which checkpoint it pointed at.
**Fix:** extracted the exact class from the notebook's Cell 6 and created
`Azure/model.py`, so the checkpoint's `state_dict` keys match precisely.

### Silent zero-tensor fallback (same bug as Phase 7's Cell 33 finding)
`run_inference()` had the identical pattern: a patient with no cached scan
got a normal prediction from a blank image with no warning. **Fix:** tracks
`image_available`, surfaces an `st.warning` immediately after inference and
a persistent `st.error` banner on every rerun (`session_state`-backed).

### Hardcoded CPU-era metrics
Replaced AUC 0.912, per-class 0.957/0.936/0.844, and 79% accuracy throughout
the UI and both PDF reports with the real GPU numbers — Fold 4 AUC 0.951
(per-class CN 0.982/Dementia 0.958/MCI 0.913), 85% accuracy, 0.85 macro F1 —
and added the honest 5-fold CV estimate (0.871 ± 0.046) alongside Fold 4's
number everywhere it's shown, so a single best fold isn't presented as the
model's general expected performance.

---

## PHASE 10: Azure Migration (In Progress)

### Discovery: the live dashboard is not what's in the tracked repo
| Setting | Value |
|---|---|
| Web App | `neuro-dt-dashboard`, resource group `hendawy-thesis`, West Europe |
| App Service Plan | `neuro-dt-plan`, Basic B2 |
| Container image | `neurodtregistry.azurecr.io/neuro-dt:v17` |
| Startup command | `streamlit run app.py --server.port 8000 ...` |

The startup command runs **`app.py`**, not `dashboard.py`. Searched the
entire tracked repo for a `Dockerfile` or `app.py` — neither exists there.
The real deployment source was confirmed to live locally at
`C:\Users\ALGOcas\Documents\ADNI_DATA\Thesis\neuro_dt_dashboard`, not yet
shared into the working session.

### Basic B2's disk limits shape the design going forward
13 GB of `tensor_cache` cannot live on a B2 instance — this points toward
on-demand Azure Blob download for scan lookups (a pattern partially already
present in `dashboard.py`) rather than bundling the cache into the
container image.

### Not yet started
Azure ML Workspace compute instance for running the remaining notebook
cells (9, 18, 20, 22, 24, 26–33) purely on Azure infrastructure, per the
user's explicit preference to do no further local-machine work.

---

## Final Results Summary

### Main model
- **5-fold CV: AUC = 0.8706 ± 0.0461** (primary, defensible headline result)
- Fold 4 head-to-head vs. CPU: **0.9511 vs. 0.9120**
- Per-class (Fold 4 eval): CN 0.9819, Dementia 0.9583, MCI 0.9131
- Accuracy 85%, macro F1 0.85

### Ablation study
11 models evaluated (7 DL variants + 4 classical baselines) — zero on CPU.
Best: A5_Transformer_1Layer 0.9563; full model A0 0.9486.

### Digital Twin pipeline
Fully executed end to end — one confirmed external-validity finding
(APOE4/Markov chain), two investigated limitations (Grad-CAM diffuseness,
AGE tabular-feature noise), one deployment-risk bug found and fixed
(silent zero-tensor fallback).

---

## Key Files Reference

| File | Location | Purpose |
|---|---|---|
| `NeuroDT_GPU_Lab.ipynb` | `Local_GPU/` | Primary GPU notebook — training, ablation, full pipeline |
| `Handoff.md` | `Local_GPU/` | Full running record of the GPU lab phase |
| `download_files.py` / `upload_files.py` | `Local_GPU/` | Hardened Blob Storage relay, bidirectional |
| `verify_downloads.py` | `Local_GPU/` | Load-verifies (not just checks existence of) cached tensors/checkpoints |
| `Continue_From_Home_Guide.md` | `Local_GPU/` | Step-by-step for finishing the pipeline without the lab/GPU |
| `dashboard.py` | `Azure/` | Streamlit app — fixed this phase, not yet confirmed identical to production `app.py` |
| `model.py` | `Azure/` | `MultimodalTransformer`, newly created — was missing entirely |
| `Handoff.md` | `Azure/` | Running record of the Azure migration & dashboard phase |
| `best_model_fold4.pth` | `checkpoints/` (copied off lab PC) | Primary GPU-trained model |
| `markov_matrices.pkl` | `checkpoints/` | Markov transition matrices, APOE4-stratified |
| `ablation_results.json` | `checkpoints/ablation/` | All 7 DL variants' results |
| `ablation_results_table.csv` | `checkpoints/ablation/` | Compiled ablation + classical baseline table |
| `tensor_cache/*.pt` | copied off lab PC | 1,549 tensors, ~13 GB, verified byte-for-byte |

---

## Questions / Details Needed
1. **Real deployment source** — `Dockerfile`/`app.py` from
   `neuro_dt_dashboard` still needs to be shared into the working
   repo/session before the Azure App Service can be safely updated.
2. **App Service scan-storage decision** — on-demand Blob download vs. any
   other design, once the real Dockerfile is visible (raised once, not yet
   answered).
3. **Azure ML compute instance** — SKU choice and exact workspace name to
   use for the remaining non-training pipeline work.
4. **Grad-CAM multi-patient check** — not yet run; needed before drawing
   any conclusion about learned-vs-shortcut features.
5. **Azure client secret rotation** — still plaintext in git history and in
   multiple files (`download_files.py`, `upload_files.py`,
   `Azure/dashboard.py`) — flagged since the start of the GPU phase, still
   unrotated.
