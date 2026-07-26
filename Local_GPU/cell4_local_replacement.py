# ===================================================================
# Cell 4: Load & Prepare the Golden DataFrame (lab PC — local file version)
# ===================================================================
MANIFEST_PATH = Path(r"C:\Users\seif\neuro_dt\checkpoints\master_manifest.csv")
BAD_SCANS_PATH = Path(r"C:\Users\seif\neuro_dt\checkpoints\bad_scans.txt")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"{MANIFEST_PATH} not found. Copy master_manifest.csv from Azure ML's "
        "workspaceblobstore (azureml/silver_farm_9tm8zbnm8b/output_dir/master_manifest.csv) "
        "to this path before running this cell."
    )

df_raw = pd.read_csv(MANIFEST_PATH)
print(f"Raw manifest loaded  |  Shape: {df_raw.shape}")
print(f"Columns: {df_raw.columns.tolist()}")

# ── Inline cleaning (replicates Data_Cleaning.ipynb) ──────────────────────
df = df_raw.dropna(subset=["diagnosis"]).copy()
print(f"After dropping missing diagnosis: {df.shape}")

if BAD_SCANS_PATH.exists():
    with open(BAD_SCANS_PATH) as bf:
        bad_scans = [line.strip() for line in bf if line.strip()]
    before = len(df)
    df = df[~df["scan_dir"].isin(bad_scans)].reset_index(drop=True)
    print(f"Removed {before - len(df)} bad scans listed in bad_scans.txt")
else:
    print("bad_scans.txt not found locally — skipping.")

before = len(df)
df = df.drop_duplicates(subset=["patient_id", "scan_dir"]).reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows  |  Shape now: {df.shape}")

diag_map = {
    "CN": "CN", "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI",
    "SMC": "CN", "AD": "Dementia", "Dementia": "Dementia",
}
df["diagnosis"] = df["diagnosis"].map(diag_map)
df = df.dropna(subset=["diagnosis"]).reset_index(drop=True)
print(f"After label standardisation: {df.shape}")

le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["diagnosis"])
label_map = dict(zip(le.transform(le.classes_), le.classes_))
print(f"\nLabel map: {label_map}")

TABULAR_FEATURES = ["AGE", "PTEDUCAT", "MMSE", "APOE4"]
missing_cols = [c for c in TABULAR_FEATURES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in manifest: {missing_cols}")
df[TABULAR_FEATURES] = df[TABULAR_FEATURES].fillna(df[TABULAR_FEATURES].median())

def extract_visit_date(scan_dir):
    for part in str(scan_dir).split("/"):
        if len(part) >= 10 and part[4] == "-" and part[7] == "-":
            return pd.to_datetime(part[:10], errors="coerce")
    return pd.NaT

df["visit_date"] = df["scan_dir"].apply(extract_visit_date)
date_ok = df["visit_date"].notna().sum()
print(f"visit_date extracted: {date_ok}/{len(df)} rows")

print(f"\nGolden DataFrame ready  |  Shape: {df.shape}")
print(f"\nClass distribution:")
print(df["diagnosis"].value_counts().to_string())
display(df[["patient_id", "diagnosis", "label_encoded", "scan_dir", "visit_date"] + TABULAR_FEATURES].head(5))
