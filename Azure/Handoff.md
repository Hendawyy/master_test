# Neuro-DT Project Handover — Azure Migration & Dashboard Update

---

# Goal (what we are trying to build)

The GPU lab training phase is complete (see `Local_GPU/Handoff.md` for that
full history — 5-fold CV, ablation study, full Digital Twin pipeline, all
done). This phase's goal: get everything that came out of that GPU run
**onto Azure** and load-bearing there, with **no further local-machine
work** —

1. An **Azure ML Workspace compute instance** to run whatever's left of the
   notebook pipeline (evaluation, Grad-CAM, Markov chain, Digital Twin,
   what-if simulations — nothing that needs a GPU or retraining).
2. The live **Azure App Service dashboard** (`neuro-dt-dashboard`) switched
   over to serve predictions from the GPU-trained model instead of the old
   CPU-era one.

---

## Current State (where the work stands right now)

### GPU training phase — fully closed out
All training, the ablation study, and the full Digital Twin pipeline
finished successfully on the university GPU lab machine. Files were copied
off that machine (flash drive, verified byte-for-byte) and are now on the
user's own Windows machine at `C:\Users\ALGOcas\Documents\ADNI_DATA\neuro_dt\`
(`checkpoints\` + `tensor_cache\`). Full detail, all results, and all
methodology caveats: `Local_GPU/Handoff.md`.

### User's explicit constraint for this phase
**No local work.** Both the remaining notebook pipeline and the dashboard
update should happen entirely through Azure (ML Workspace compute instance
+ App Service/Container Registry), not on any local machine.

### Found and fixed: `Azure/model.py` was completely missing
`dashboard.py` does `from model import MultimodalTransformer`, but no
`model.py` existed anywhere in this repo. This would fail with
`ModuleNotFoundError` immediately on load, regardless of which checkpoint
(CPU or GPU) it points at — a pre-existing bug, not something introduced
this session. Fixed: extracted the exact `MultimodalTransformer` class from
`Local_GPU/NeuroDT_GPU_Lab.ipynb` Cell 6 and created `Azure/model.py`, so
the checkpoint's `state_dict` keys match. Committed and pushed.

### Discovered: the real production app is not what's in this repo
The live dashboard is a **Linux container Web App**, not a plain
`streamlit run dashboard.py`:

| Setting | Value |
|---|---|
| Web App name | `neuro-dt-dashboard` |
| Resource group | `hendawy-thesis` |
| Region | West Europe |
| App Service Plan | `neuro-dt-plan`, **Basic B2** |
| Container image | `neurodtregistry.azurecr.io/neuro-dt:v17` |
| Startup command | `streamlit run app.py --server.port 8000 --server.headless true --server.address 0.0.0.0` |

The startup command runs **`app.py`**, not `dashboard.py`. Searched this
entire repo (`hendawyy/master_test`) for a `Dockerfile` or `app.py` — found
**neither**. Whatever actually builds `neuro-dt:v17` is not in this repo.

User confirmed the real source lives locally at
`C:\Users\ALGOcas\Documents\ADNI_DATA\Thesis\neuro_dt_dashboard` — Claude
has no access to that path (sandboxed to this repo + chat uploads only).
**Not yet shared into this session.**

### Two Container Registries exist
- `AMLworkspaceACR1` — East US, resource group `Hendawy-Thesis`
- `neurodtregistry` — West US, resource group `hendawy-thesis` ← the one
  the app actually pulls from

### Basic B2 plan constrains the scan-storage design
B2 does not have room for the 13 GB `tensor_cache`. This all but forces the
dashboard's MRI-scan lookups to rely on **on-demand Azure Blob download**
(check local cache → if missing, download + preprocess the DICOM from Blob
on the fly) rather than bundling the full cache onto the instance.
`dashboard.py`'s sidebar "MRI Scan" input already has a partial version of
this fallback built in. This was raised to the user as an explicit decision
point (via a clarifying question) but **dismissed without an answer** — the
constraint (B2's disk limits) points strongly at Blob download regardless
of preference, but it's still an open decision to confirm once the real
`app.py`/Dockerfile are visible.

### Azure ML Workspace compute instance — not started
The plan (compute instance → Blob relay → run non-training cells) was laid
out in detail in chat but **no compute instance has been created yet**, and
no data has moved to Azure Blob Storage from the new local path.

---

## Files in Flight (Active Files Being Modified)

- `Azure/dashboard.py` — fixed this session (see Changed, below), but
  **not confirmed to be the same file as the deployed `app.py`**. Treat as
  a reference/staging copy until the real deployment source is shared and
  reconciled against it.
- `Azure/model.py` — new this session, required by `dashboard.py`.
- `Local_GPU/download_files.py` / `Local_GPU/upload_files.py` — existing
  Blob Storage relay scripts; reusable as-is (with path edits) to move
  files between the new local Windows machine, Blob Storage, and an Azure
  ML compute instance.
- **Missing from this repo, needed next**: the actual `Dockerfile` + `app.py`
  that build `neurodtregistry.azurecr.io/neuro-dt:v17`, currently sitting
  only at `C:\Users\ALGOcas\Documents\ADNI_DATA\Thesis\neuro_dt_dashboard`
  on the user's machine.

---

## Changed (what has been touched)

### `Azure/model.py` (new file)
- `MultimodalTransformer` class, extracted verbatim from the GPU notebook's
  Cell 6 (DenseNet121 backbone → concat with 4 tabular features → Linear
  projection + LayerNorm + GELU → 2-layer Transformer Encoder → classifier
  head). Needed so `dashboard.py`'s `from model import MultimodalTransformer`
  resolves and `load_state_dict()` keys match the GPU checkpoint exactly.

### `Azure/dashboard.py`
- `load_assets()`: added `weights_only=False` to `torch.load()` — was
  missing entirely, would crash loading *any* checkpoint (CPU or GPU) on
  PyTorch 2.6+, since these checkpoints embed a `StandardScaler`.
- `run_inference()`: fixed the same silent zero-tensor fallback bug found
  and fixed in the notebook's `NeuroDT` class — now tracks
  `image_available` and surfaces an unmissable warning (`st.warning`
  immediately after inference, persistent `st.error` banner on every
  rerun via `session_state`) instead of quietly predicting from a blank
  image with no signal to the user.
- Replaced every hardcoded CPU-era metric (AUC 0.912, per-class
  0.957/0.936/0.844, 79% accuracy) with the real GPU numbers throughout the
  UI and both PDF reports: Fold 4 AUC 0.951 (per-class CN 0.982 / Dementia
  0.958 / MCI 0.913), 85% accuracy, 0.85 macro F1 — plus the honest 5-fold
  CV estimate (0.871 ± 0.046) alongside Fold 4's number everywhere it's
  shown, so a single best fold isn't presented as the model's general
  expected performance.

---

## Failed attempts (things you tried but didn't work and why)

| What was tried | Why it failed | Status |
|---|---|---|
| Searching this repo for a `Dockerfile`/`app.py` to understand the deployed container | Neither file exists anywhere in `hendawyy/master_test` | User says they're at `C:\Users\ALGOcas\Documents\ADNI_DATA\Thesis\neuro_dt_dashboard` locally — not yet shared into this session (needs zip upload via chat, or a `git push` from that folder into this repo) |
| Asked (via a clarifying question) how the App Service is deployed and whether to use on-demand Blob download vs. storing `tensor_cache` locally on the instance | User dismissed both questions without answering | Re-ask or infer once the real Dockerfile is visible — B2's disk limits make on-demand Blob download the only realistic option regardless of preference |
| Gave instructions for running the dashboard locally (`streamlit run dashboard.py` with `CHECKPOINT_DIR`/`CACHE_DIR` env vars) and for a local-machine-adjacent Azure ML setup | User explicitly does not want to do anything locally at this stage | Pivoted entirely to Azure ML compute instance + Container Registry/App Service as the only paths forward |

---

## Next Steps (things to try next)

1. **Get the real deployment source into this session.** User needs to
   either zip-upload the `neuro_dt_dashboard` folder through the chat, or
   `git push` it into this repo from their local machine (e.g. as a new
   `Dashboard_Deploy/` folder). Nothing else in this phase can be verified
   against production until this happens.
2. Once the real `app.py`/`Dockerfile` are visible: check whether `app.py`
   is `dashboard.py` renamed at build time or a genuinely different/older
   file, and apply the same fixes (weights_only, image_available warning,
   real GPU metrics, the `model.py` dependency) to whichever file is
   actually authoritative.
3. Confirm the scan-storage decision (on-demand Blob download vs. local
   cache) once the real Dockerfile shows what's actually being baked into
   the image today.
4. Push `best_model_fold4.pth` + `markov_matrices.pkl` to Azure Blob
   Storage (via `upload_files.py`, pointed at
   `C:\Users\ALGOcas\Documents\ADNI_DATA\neuro_dt\checkpoints`), and add a
   "download from Blob on startup if missing" step to `load_assets()` so
   the container image doesn't need the model files baked in at build time.
5. Rebuild the container image with the updated code, push a new tag
   (e.g. `v18`) to `neurodtregistry.azurecr.io/neuro-dt`, and update the
   Web App's container settings (or use the existing "Continuous
   deployment" webhook, if enabled) to deploy it.
6. Set up an Azure ML compute instance (CPU-only SKU — nothing left needs
   a GPU) under the existing ML workspace, move checkpoints/tensor_cache
   there via the same Blob relay, and run the remaining notebook cells
   (9, 18, 20, 22, 24, 26–33) there.
7. Outstanding hygiene carried over from the GPU lab phase (see
   `Local_GPU/Handoff.md` for full detail): rotate the Azure client secret
   (still plaintext in git history and hardcoded in multiple files
   including `Azure/dashboard.py`); delete the merged feature branch.
