import os
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

# ── Download checkpoints ──────────────────────────────────────────
ckpt_dir = Path(r"C:\Users\seif\neuro_dt\checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

blobs = [b for b in cc.list_blobs(name_starts_with="gpu_transfer/checkpoints/")]
for blob in blobs:
    fname = Path(blob.name).name
    dest  = ckpt_dir / fname
    print(f"Downloading {fname}...")
    with open(dest, "wb") as f:
        cc.get_blob_client(blob.name).download_blob().readinto(f)
    print(f"  ✓ {fname}")

# ── Download tensor_cache ─────────────────────────────────────────
cache_dir = Path(r"C:\Users\seif\neuro_dt\tensor_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

blobs = list(cc.list_blobs(name_starts_with="gpu_transfer/tensor_cache/"))
print(f"\nDownloading {len(blobs)} tensor cache files (~13 GB)...")
for i, blob in enumerate(blobs):
    fname = Path(blob.name).name
    dest  = cache_dir / fname
    if dest.exists():
        continue   # skip already downloaded
    with open(dest, "wb") as f:
        cc.get_blob_client(blob.name).download_blob().readinto(f)
    if i % 100 == 0:
        print(f"  {i}/{len(blobs)} downloaded...")

print("\n✓ All files downloaded.")
print(f"  Checkpoints: {ckpt_dir}")
print(f"  Tensor cache: {cache_dir}")