```markdown
# Neuro-DT Project Handover — May 2026 (GPU Lab Session)

---

## Goal
Complete the Brain Digital Twin (Neuro-DT) master's thesis at the Arab Academy for
Science, Technology and Maritime Transport.

Student Seif Hendawy
Supervisors Prof. Fahima Maghraby · Assoc. Prof. Ahmed Salem

The immediate goal of this session is to
1. Set up the university GPU lab machine (RTX 5070 Ti, 16 GB VRAM, CUDA 13.1) for training
2. Transfer the tensor cache (1,549 .pt files, ~13 GB) and checkpoints from Azure to the lab PC
3. Run the full 5-fold cross-validation training using `NeuroDT_GPU_Lab.ipynb`
4. Run the complete DL ablation study (variants A2–A6) which could not run on CPU

---

## Current State

### GPU Lab Machine — READY
- Machine University lab PC, Windows, `CUsersseif`
- GPU NVIDIA RTX 5070 Ti — 17.1 GB VRAM, CUDA 13.1 ✓
- Environment `bdt-env` conda env, Python 3.10
- PyTorch `2.11.0+cu128` — CUDA confirmed working (`CUDA True`) ✓
- Remaining pip installs Still need to run
  ```cmd
  pip install azure-ai-ml azureml-core azure-storage-blob azure-identity
  ```
  And the full deps if not done yet
  ```cmd
  pip install numpy2.0 pandas scikit-learn tqdm matplotlib monai[all] pydicom nibabel mlflow shap hmmlearn azure-storage-blob azure-identity jupyter ipykernel ipywidgets reportlab google-generativeai
  ```

### File Transfer — IN PROGRESS
- Upload from Azure compute ✅ COMPLETE
  - All 1,549 tensor cache `.pt` files uploaded to
    `adnihendawy  adni-data  gpu_transfertensor_cache`
  - All checkpoint files uploaded to
    `adnihendawy  adni-data  gpu_transfercheckpoints`
  - Includes `best_model_fold4.pth`, all fold checkpoints, `markov_matrices.pkl`,
    `auc_results.json`, all PNG figures
- Download to lab PC ⏳ IN PROGRESS (or not started yet)
  - Script `CUsersseifdownload_files.py`
  - Destination `CUsersseifneuro_dtcheckpoints` and `CUsersseifneuro_dttensor_cache`
  - Verify when done
    ```cmd
    python -c from pathlib import Path; c=list(Path(r'CUsersseifneuro_dttensor_cache').glob('.pt')); print(f'Tensor cache {len(c)}1549 files'); print('Checkpoint', Path(r'CUsersseifneuro_dtcheckpointsbest_model_fold4.pth').exists())
    ```

### Notebook — NOT YET RUN ON GPU MACHINE
- File `NeuroDT_GPU_Lab.ipynb`
- Two edits MUST be made before running

  Edit 1 — Cell 3 Replace Azure workspace connection block with
  ```python
  ml_client = None
  ws        = None
  print(⚠ Azure workspace skipped — lab PC mode. Using local files only.)
  ```

  Edit 2 — Cell 9_GPU Update paths
  ```python
  BEST_MODEL_DIR = Path(rCUsersseifneuro_dtcheckpoints)
  CACHE_DIR      = Path(rCUsersseifneuro_dttensor_cache)
  LOCAL_MOUNT_PATH = None
  ```

### Ablation Study — CPU Results Completed (Azure)
Results saved in `checkpointsablationablation_results.json`
 Variant  AUC  F1  Notes 
-------------------------
 A0 NeuroDT Full  0.9120  0.796  Loaded from existing checkpoint 
 A1 Tabular-Only MLP  0.8766  0.692  No MRI branch 
 B1 Logistic Regression  0.8649  0.707  sklearn tabular 
 B2 SVM RBF  0.8711  0.735  sklearn tabular 
 B3 Random Forest  0.9428  0.830  sklearn tabular — beats NeuroDT on AUC 
 B4 Gradient Boosting  0.9408  0.825  sklearn tabular — beats NeuroDT on AUC 
 A2–A6  ⏳ pending  —  Require GPU — run Cell A2 

⚠ CRITICAL CONTEXT on RFGB beating NeuroDT
Root cause is MMSE dominance — ADNI labels are partly derived from MMSE scores so
ensemble methods learn the threshold trivially. Thesis framing
- NeuroDT has zero CN↔Dementia misclassifications (RFGB will make these)
- NeuroDT works without MMSE at inference time
- NeuroDT provides spatial Grad-CAM explainability
- NeuroDT enables Digital Twin simulation — a scalar RF score cannot

---

## Files in Flight

 File  Location  Status 
------------------------
 `NeuroDT_GPU_Lab.ipynb`  Lab PC (needs path edits)  ⏳ Needs Cell 3 + Cell 9_GPU edits before running 
 `NeuroDT_CPU_Workhorse.ipynb`  Azure compute  ✅ Complete, all CPU ablation done 
 `download_files.py`  `CUsersseifdownload_files.py`  ⏳ Run to download tensor_cache + checkpoints 
 `tensor_cache.pt`  Azure blob `gpu_transfertensor_cache`  ✅ Uploaded, ⏳ downloading to lab PC 
 `checkpoints`  Azure blob `gpu_transfercheckpoints`  ✅ Uploaded, ⏳ downloading to lab PC 
 `ablation_results.json`  Azure blob inside checkpoints upload  ✅ Has A0, A1, B1–B4 results 

### Azure credentials
```
TENANT_ID     = '70c07c26-601e-415b-9a91-c351a5ad357b'
CLIENT_ID     = 'c638dc4d-96ec-4457-8797-23902283156b'  # ← exact, do not change
CLIENT_SECRET = 'NVp8Q~jeqNiNtwKkbCILt.p4CSNumnl1hz__Hc_E'  # ⚠ ROTATE THIS
STORAGE       = 'adnihendawy'
CONTAINER     = 'adni-data'
```

### Windows environment variable syntax (CMD — not PowerShell, not export)
```cmd
set AZURE_CLIENT_SECRET=NVp8Q~jeqNiNtwKkbCILt.p4CSNumnl1hz__Hc_E
```
Must be set in the same CMD window that starts Jupyter. Lasts only for that session.

---

## Changed

### This session
- PyTorch reinstalled Uninstalled CPU-only `torch 2.11.0`, reinstalled
  `torch 2.11.0+cu128` from `httpsdownload.pytorch.orgwhlcu128` — CUDA now working
- Azure compute upload All 1,549 tensor cache files + all checkpoint files uploaded
  to `adnihendawyadni-datagpu_transfer` via Python blob SDK script
- Cell 3 fix identified `MLClient.from_config()` fails on lab PC (no `config.json`).
  Must replace with `ml_client = None; ws = None` before running

### Previous sessions (already in code)
- `app.py` v7 — all dashboard bugs fixed, deployed to Azure App Service
- `NeuroDT_CPU_Workhorse.ipynb` — Cell A2 has graceful guard when
  `run_ablation_variant` not defined (was throwing NameError)
- `NeuroDT_GPU_Lab.ipynb` — Cell 10_GPU uses OneCycleLR (fixes early stopping),
  batch=16, AMP enabled, Cell COMPARE for CPU vs GPU results
- PDF guide — all table cells use Paragraph objects (fixed text overflow)

---

## Failed Attempts

 What was tried  Why it failed  Fix 
------------------------------------
 `pip install torch==2.1.0+cu121`  RTX 5070 Ti is Blackwell architecture, needs CUDA 12.8+. The `+cu121` build doesn't exist on PyPI (only on pytorch.orgwhl). Missing `--extra-index-url` flag.  Uninstall, reinstall with `--extra-index-url httpsdownload.pytorch.orgwhlcu128` 
 `export AZURE_CLIENT_SECRET=...`  LinuxMac syntax — does not work on Windows CMD or PowerShell  Use `set VAR=value` on CMD, `$envVAR=value` on PowerShell 
 `wget httpsaka.msdownloadazcopy-v10-linux` on lab PC PowerShell  Wrong OS (Linux URL on Windows), SSL blocked by university network  Download azcopy Windows zip from browser instead; or use Python blob SDK 
 azcopy download on lab PC  Blocked by university admin — executable wouldn't run  Use `download_files.py` Python script which needs no admin rights 
 `MLClient.from_config()` in Cell 3 on lab PC  No `config.json` exists outside Azure compute environment  Replace with `ml_client = None; ws = None` 
 `torch.cuda.get_device_name(0)` after first pip install  torch was CPU-only build from PyPI — `CUDA False`  Reinstall from pytorch.org whl with cu128 flag 
 Backslash `` line continuation in pip install command on Windows CMD  Windows CMD does not support `` for multi-line commands  Write as one single long line 

---

## Next Steps

### Immediate (in order)
1. Verify download complete on lab PC
   ```cmd
   python -c from pathlib import Path; c=list(Path(r'CUsersseifneuro_dttensor_cache').glob('.pt')); print(f'{len(c)}1549 files')
   ```

2. Finish pip installs if not done
   ```cmd
   pip install numpy2.0 pandas scikit-learn tqdm matplotlib monai[all] pydicom nibabel mlflow shap hmmlearn azure-storage-blob azure-identity azure-ai-ml azureml-core jupyter ipykernel ipywidgets reportlab google-generativeai
   python -m ipykernel install --user --name bdt-env --display-name Python (BDT)
   ```

3. Set secret and start Jupyter (same CMD window)
   ```cmd
   set AZURE_CLIENT_SECRET=NVp8Q~jeqNiNtwKkbCILt.p4CSNumnl1hz__Hc_E
   jupyter notebook
   ```

4. Edit notebook before running
   - Cell 3 replace workspace block with `ml_client = None; ws = None`
   - Cell 9_GPU set `BEST_MODEL_DIR` and `CACHE_DIR` to `CUsersseifneuro_dt...`

5. Session 1 run order
   ```
   Cell 3 → 3b → 4 → 7 → 8 → 8b → Cell 9_GPU → Cell 10_GPU
   ```
   Cell 10_GPU will take ~6–8 hours on RTX 5070 Ti for full 5-fold training.

6. Session 2 run order (ablation)
   ```
   Cell 3 → 3b → 4 → 7 → 8 → 8b → Cell 9_GPU → Cell 10b → A1 → A2 → A3 → A4
   ```
   Cell A2 will run DL variants A2–A6, ~2–3 hours total on RTX 5070 Ti.

7. Copy ablation_results.json from Azure to lab PC before running Cell A2,
   so completed results (A0, A1, B1–B4) are not lost and Cell A2 skips them
   - It will be inside `CUsersseifneuro_dtcheckpointsablationablation_results.json`
   - if the download_files.py script downloaded the ablation subfolder

### After GPU training completes
- Run Cell COMPARE to document CPU vs GPU AUC difference
- Re-run Cell A4 with full results for the final ablation table and figures
- Update thesis results section with GPU cross-validated AUC mean ± std
- Thesis framing for RFGB finding MMSE dominance, not NeuroDT failure

### GPU training config (Cell 10_GPU key settings)
```python
FAST_PROTO = False   # ← MUST be False for thesis results
BATCH_SIZE = 16      # fits in 17.1 GB VRAM
USE_AMP    = True    # mixed precision — enabled automatically on GPU
# Scheduler OneCycleLR with 10% warmup (fixes the CPU early stopping issue)
```
```