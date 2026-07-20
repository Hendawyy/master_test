import os
import time
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.identity import ClientSecretCredential

cred = ClientSecretCredential(
    tenant_id=os.environ["AZ_TENANT_ID"],
    client_id=os.environ["AZ_CLIENT_ID"],
    client_secret=os.environ["AZ_CLIENT_SECRET"],
)

cc = BlobServiceClient(
    "https://adnihendawy.blob.core.windows.net",
    credential=cred
).get_container_client("adni-data")


def download_with_retry(blob_client, dest: Path, max_retries=10):
    props = blob_client.get_blob_properties()
    total_size = props.size

    downloaded = dest.stat().st_size if dest.exists() else 0
    if downloaded >= total_size:
        return  # already complete

    for attempt in range(max_retries):
        try:
            mode = "ab" if downloaded else "wb"
            with open(dest, mode) as f:
                stream = blob_client.download_blob(
                    offset=downloaded,
                    length=total_size - downloaded,
                    connection_timeout=300,
                    max_concurrency=1,
                    retry_total=5,
                )
                for chunk in stream.chunks():
                    f.write(chunk)
                    downloaded += len(chunk)
            return  # success
        except Exception as e:
            print(f"    retry {attempt+1}/{max_retries} at {downloaded} bytes ({e})")
            downloaded = dest.stat().st_size if dest.exists() else 0
            time.sleep(5)
    raise RuntimeError(f"Failed to download {dest.name} after {max_retries} retries")


# ── Download checkpoints ──────────────────────────────────────────
ckpt_dir = Path(r"C:\Users\seif\neuro_dt\checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

blobs = [b for b in cc.list_blobs(name_starts_with="gpu_transfer/checkpoints/")]
for blob in blobs:
    fname = Path(blob.name).name
    dest = ckpt_dir / fname
    print(f"Downloading {fname}...")
    download_with_retry(cc.get_blob_client(blob.name), dest)
    print(f"  ✓ {fname}")

# ── Download tensor_cache ─────────────────────────────────────────
cache_dir = Path(r"C:\Users\seif\neuro_dt\tensor_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

blobs = list(cc.list_blobs(name_starts_with="gpu_transfer/tensor_cache/"))
print(f"\nDownloading {len(blobs)} tensor cache files (~13 GB)...")
for i, blob in enumerate(blobs):
    fname = Path(blob.name).name
    dest = cache_dir / fname
    if dest.exists():
        blob_client = cc.get_blob_client(blob.name)
        if dest.stat().st_size == blob_client.get_blob_properties().size:
            continue  # already fully downloaded
    download_with_retry(cc.get_blob_client(blob.name), dest)
    if i % 100 == 0:
        print(f"  {i}/{len(blobs)} downloaded...")

print("\n✓ All files downloaded.")
print(f"  Checkpoints: {ckpt_dir}")
print(f"  Tensor cache: {cache_dir}")
