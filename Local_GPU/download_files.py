import os
import time
from azure.storage.blob import BlobServiceClient
from azure.identity import ClientSecretCredential
from pathlib import Path

cred = ClientSecretCredential(
    tenant_id='70c07c26-601e-415b-9a91-c351a5ad357b',
    client_id='c638dc4d-96ec-4457-8797-23902283156b',
    client_secret='NVp8Q~jeqNiNtwKkbCILt.p4CSNumnl1hz__Hc_E')

cc = BlobServiceClient(
    "https://adnihendawy.blob.core.windows.net",
    credential=cred).get_container_client("adni-data")

MAX_RETRIES = 4

def download_with_retry(blob_name, dest):
    """Download a blob to dest. Removes partial files on failure so a later
    run doesn't mistake a truncated download for a completed one. Returns
    True on success, False if it failed after all retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(dest, "wb") as f:
                cc.get_blob_client(blob_name).download_blob().readinto(f)
            return True
        except Exception as e:
            if dest.exists():
                dest.unlink()
            if attempt == MAX_RETRIES:
                print(f"  ✗ FAILED after {MAX_RETRIES} attempts: {Path(blob_name).name} ({e})")
                return False
            wait = 2 ** attempt
            print(f"  ⚠ {Path(blob_name).name}: {e} — retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)

# ── Download checkpoints ──────────────────────────────────────────
ckpt_dir = Path(r"C:\Users\seif\neuro_dt\checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

failed = []
blobs = [b for b in cc.list_blobs(name_starts_with="gpu_transfer/checkpoints/")]
for blob in blobs:
    fname = Path(blob.name).name
    dest  = ckpt_dir / fname
    if dest.exists():
        print(f"  = {fname} (already downloaded)")
        continue
    print(f"Downloading {fname}...")
    if download_with_retry(blob.name, dest):
        print(f"  ✓ {fname}")
    else:
        failed.append(fname)

# ── Download tensor_cache ─────────────────────────────────────────
cache_dir = Path(r"C:\Users\seif\neuro_dt\tensor_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

blobs = list(cc.list_blobs(name_starts_with="gpu_transfer/tensor_cache/"))
print(f"\nDownloading {len(blobs)} tensor cache files (~13 GB)...")
start = time.time()
done_this_run = 0
skipped = 0
for i, blob in enumerate(blobs):
    fname = Path(blob.name).name
    dest  = cache_dir / fname
    if dest.exists():
        skipped += 1
        continue   # skip already downloaded
    if not download_with_retry(blob.name, dest):
        failed.append(fname)
        continue
    done_this_run += 1
    if done_this_run % 50 == 0:
        elapsed = time.time() - start
        rate = done_this_run / elapsed  # files/sec, this run only
        remaining = len(blobs) - skipped - done_this_run
        eta_sec = remaining / rate if rate > 0 else 0
        print(f"  {i+1}/{len(blobs)} processed | {done_this_run} downloaded this run, "
              f"{skipped} already had | elapsed {elapsed/60:.1f} min | "
              f"~{eta_sec/60:.1f} min remaining")

print(f"\n✓ Done. ({done_this_run} downloaded this run, {skipped} were already present)")
print(f"  Total time this run: {(time.time() - start)/60:.1f} min")
print(f"  Checkpoints: {ckpt_dir}")
print(f"  Tensor cache: {cache_dir}")
if failed:
    print(f"\n⚠ {len(failed)} file(s) failed after {MAX_RETRIES} retries — rerun the script to retry them:")
    for f in failed:
        print(f"    {f}")
