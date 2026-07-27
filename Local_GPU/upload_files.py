"""
Upload local checkpoints / ablation results / tensor_cache to the same Azure
Blob container download_files.py pulls from — the other half of the relay.

Run this from wherever the files currently live (e.g. your laptop, after
copying them off the lab PC via flash drive). Then run download_files.py
from wherever you want them next (e.g. an Azure ML compute instance) to
pull them back down — Azure-to-Azure, fast and reliable, no dependency on
flaky wifi at either end.

Update LOCAL_* below to match where you copied the flash drive's contents.
"""
import time
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.identity import ClientSecretCredential

cred = ClientSecretCredential(
    tenant_id='70c07c26-601e-415b-9a91-c351a5ad357b',
    client_id='c638dc4d-96ec-4457-8797-23902283156b',
    client_secret='NVp8Q~jeqNiNtwKkbCILt.p4CSNumnl1hz__Hc_E')

cc = BlobServiceClient(
    "https://adnihendawy.blob.core.windows.net",
    credential=cred,
    retry_total=8,
    retry_connect=8,
).get_container_client("adni-data")

MAX_RETRIES = 4

# ── UPDATE THESE to wherever you copied the flash drive's contents ─────────
LOCAL_CKPT_DIR  = Path(r"C:\Users\seif\neuro_dt\checkpoints")
LOCAL_CACHE_DIR = Path(r"C:\Users\seif\neuro_dt\tensor_cache")


def existing_blob_sizes(prefix):
    """Map of filename -> size (bytes) for blobs already at this prefix, so
    we can skip re-uploading files that are already there and unchanged."""
    return {Path(b.name).name: b.size for b in cc.list_blobs(name_starts_with=prefix)}


def upload_with_retry(local_path, blob_name):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(local_path, "rb") as f:
                cc.upload_blob(blob_name, f, overwrite=True)
            return True
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            if attempt == MAX_RETRIES:
                print(f"  ✗ FAILED after {MAX_RETRIES} attempts: {local_path.name} ({e})")
                return False
            wait = 2 ** attempt
            print(f"  ⚠ {local_path.name}: {e} — retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)


def upload_dir(local_dir, blob_prefix, pattern="*", skip_names=()):
    if not local_dir.exists():
        print(f"⚠ {local_dir} not found — skipping.")
        return
    files = [p for p in sorted(local_dir.glob(pattern))
             if p.is_file() and p.name not in skip_names]
    remote_sizes = existing_blob_sizes(blob_prefix)

    to_upload = [p for p in files if remote_sizes.get(p.name) != p.stat().st_size]
    already = len(files) - len(to_upload)
    print(f"\n{blob_prefix}: {len(files)} local file(s), {already} already uploaded "
          f"(same size), {len(to_upload)} to upload...")

    start = time.time()
    done = 0
    failed = []
    for p in to_upload:
        if upload_with_retry(p, blob_prefix + p.name):
            done += 1
            if done % 10 == 0 or done == len(to_upload):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = len(to_upload) - done
                eta = remaining / rate if rate > 0 else 0
                print(f"  {done}/{len(to_upload)} uploaded | elapsed {elapsed/60:.1f} min "
                      f"| ~{eta/60:.1f} min remaining")
        else:
            failed.append(p.name)
    print(f"  ✓ {blob_prefix} done ({done} uploaded, {already} already present)")
    return failed


all_failed = []

# ── Checkpoints (top-level .pth + manifest/bad_scans, NOT the ablation subfolder) ──
all_failed += upload_dir(
    LOCAL_CKPT_DIR, "gpu_transfer/checkpoints/",
    pattern="*", skip_names=set()) or []
# glob("*") on a directory also matches subdirectories (e.g. "ablation") —
# upload_dir already filters to p.is_file(), so the "ablation" folder itself
# is skipped automatically; its contents are handled by the next call.

# ── Ablation results (checkpoints/ablation/) ───────────────────────────────
all_failed += upload_dir(
    LOCAL_CKPT_DIR / "ablation", "gpu_transfer/checkpoints_ablation/") or []

# ── Tensor cache ────────────────────────────────────────────────────────────
all_failed += upload_dir(
    LOCAL_CACHE_DIR, "gpu_transfer/tensor_cache/", pattern="*.pt") or []

print(f"\n{'='*60}")
if all_failed:
    print(f"⚠ {len(all_failed)} file(s) failed after {MAX_RETRIES} retries — rerun this script to retry them:")
    for f in all_failed:
        print(f"    {f}")
else:
    print("✓ All files uploaded successfully.")
