# Run this on the Azure ML compute instance (JupyterLab or terminal),
# NOT on the lab PC. It needs a real AML workspace connection.
#
# Fetches master_manifest.csv + bad_scans.txt from the AML workspace's own
# datastore, then uploads them into adni-data/gpu_transfer/checkpoints/ —
# the same blob path download_files.py already scans — so the lab PC picks
# them up on its next run with zero script changes.
from pathlib import Path

from azureml.core import Workspace, Datastore
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient

INGESTION_JOB_NAME = "silver_farm_9tm8zbnm8b"

# ── 1. Fetch from AML's own datastore ──────────────────────────────────────
ws = Workspace.from_config()
datastore = Datastore.get(ws, "workspaceblobstore")

local_dir = Path("manifest_fetch")
local_dir.mkdir(exist_ok=True)

datastore.download(
    target_path=str(local_dir),
    prefix=f"azureml/{INGESTION_JOB_NAME}/output_dir/master_manifest.csv",
    overwrite=True,
)
datastore.download(
    target_path=str(local_dir),
    prefix=f"azureml/{INGESTION_JOB_NAME}/output_dir/bad_scans.txt",
    overwrite=True,
)

manifest_matches = list(local_dir.rglob("master_manifest.csv"))
bad_scans_matches = list(local_dir.rglob("bad_scans.txt"))

if not manifest_matches:
    raise FileNotFoundError("master_manifest.csv not found under the ingestion job output — check INGESTION_JOB_NAME.")

# ── 2. Upload to adni-data/gpu_transfer/checkpoints/ ───────────────────────
cred = ClientSecretCredential(
    tenant_id='70c07c26-601e-415b-9a91-c351a5ad357b',
    client_id='c638dc4d-96ec-4457-8797-23902283156b',
    client_secret='NVp8Q~jeqNiNtwKkbCILt.p4CSNumnl1hz__Hc_E')
cc = BlobServiceClient(
    "https://adnihendawy.blob.core.windows.net",
    credential=cred).get_container_client("adni-data")

with open(manifest_matches[0], "rb") as f:
    cc.upload_blob("gpu_transfer/checkpoints/master_manifest.csv", f, overwrite=True)
print("Uploaded master_manifest.csv")

if bad_scans_matches:
    with open(bad_scans_matches[0], "rb") as f:
        cc.upload_blob("gpu_transfer/checkpoints/bad_scans.txt", f, overwrite=True)
    print("Uploaded bad_scans.txt")
else:
    print("bad_scans.txt not found in job output — skipping (Cell 4 handles this being absent).")
