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
    credential=cred,
    retry_total=8,          # SDK-level retries for transient/connection errors
    retry_connect=8,
).get_container_client("adni-data")

MAX_RETRIES = 4

def list_blobs_with_retry(prefix):
    """list_blobs() is a lazy pager — network calls happen while iterating,
    so listing can drop mid-page on a flaky connection just like a download."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return list(cc.list_blobs(name_starts_with=prefix))
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"  ⚠ listing '{prefix}' failed: {e} — retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)

def download_with_retry(blob_name, dest):
    """Download a blob to a temp file and only rename it to the final name on
    full success — so an interrupted write (network drop, Ctrl+C, closed
    terminal) can never leave a file at `dest` that later runs would
    mistake for a completed download. Returns True on success, False if it
    failed after all retries."""
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(tmp, "wb") as f:
                cc.get_blob_client(blob_name).download_blob().readinto(f)
            tmp.replace(dest)
            return True
        except BaseException as e:
            if tmp.exists():
                tmp.unlink()
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            if attempt == MAX_RETRIES:
                print(f"  ✗ FAILED after {MAX_RETRIES} attempts: {Path(blob_name).name} ({e})")
                return False
            wait = 2 ** attempt
            print(f"  ⚠ {Path(blob_name).name}: {e} — retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)

def cleanup_stale_files(directory):
    """Remove leftover .part temp files and zero-byte files (e.g. from a run
    that got interrupted before this script had the atomic rename fix)."""
    for p in directory.glob("*"):
        if p.is_file() and (p.suffix == ".part" or p.stat().st_size == 0):
            print(f"  🧹 removing stale/incomplete file: {p.name}")
            p.unlink()

# ── Download checkpoints ──────────────────────────────────────────
ckpt_dir = Path(r"C:\Users\seif\neuro_dt\checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)
cleanup_stale_files(ckpt_dir)

failed = []
blobs = list_blobs_with_retry("gpu_transfer/checkpoints/")
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
cleanup_stale_files(cache_dir)

blobs = list_blobs_with_retry("gpu_transfer/tensor_cache/")
already_have = sum(1 for b in blobs if (cache_dir / Path(b.name).name).exists())
to_fetch = len(blobs) - already_have
print(f"\nTensor cache: {already_have}/{len(blobs)} already on disk, {to_fetch} to download (~13 GB total)...")
start = time.time()
done_this_run = 0
skipped = 0
last_print = start
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
    now = time.time()
    if done_this_run % 10 == 0 or now - last_print >= 30:
        last_print = now
        elapsed = now - start
        rate = done_this_run / elapsed  # files/sec, this run only
        remaining = to_fetch - done_this_run
        eta_sec = remaining / rate if rate > 0 else 0
        print(f"  {done_this_run}/{to_fetch} downloaded this run | "
              f"elapsed {elapsed/60:.1f} min | ~{eta_sec/60:.1f} min remaining")

print(f"\n✓ Done. ({done_this_run} downloaded this run, {skipped} were already present)")
print(f"  Total time this run: {(time.time() - start)/60:.1f} min")
print(f"  Checkpoints: {ckpt_dir}")
print(f"  Tensor cache: {cache_dir}")
if failed:
    print(f"\n⚠ {len(failed)} file(s) failed after {MAX_RETRIES} retries — rerun the script to retry them:")
    for f in failed:
        print(f"    {f}")
