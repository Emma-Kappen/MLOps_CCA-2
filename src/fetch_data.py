import pandas as pd, re, os, json
from typing import Any

TEXT_COLUMNS = ['AccidentArea', 'Make', 'VehicleCategory', 'VehiclePrice', 'Fault', 'PolicyType', 'BasePolicy',
                'Sex', 'MaritalStatus', 'AgeOfPolicyHolder', 'AgeOfVehicle', 'PastNumberOfClaims',
                'AddressChange_Claim', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments',
                'NumberOfCars', 'Deductible', 'DriverRating', 'Days_Policy_Accident', 'Days_Policy_Claim',
                'Year', 'Month', 'DayOfWeek', 'MonthClaimed']

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9_\s]', '', str(text).lower())  # underscores kept: they glue field+value into one token
    return re.sub(r'\s+', ' ', text).strip()

def row_to_text(row):
    """One field-tagged token per column (e.g. fault_policyholder, policereportfiled_no) so the same value
    ('no', '1 year', ...) in different fields is never confused. Missing columns are skipped."""
    tokens = []
    for col in TEXT_COLUMNS:
        val = row.get(col)
        if pd.notna(val):
            tokens.append(f"{col.lower()}_{re.sub(r'[^a-z0-9]+', '_', str(val).lower()).strip('_')}")
    age = pd.to_numeric(row.get('Age'), errors='coerce')
    if pd.notna(age):
        tokens.append(f"age_{int(age) // 10 * 10}s")
    return " ".join(tokens)

def stratified_split(df, frac, seed):
    """Split df into (kept, split_out), split_out being `frac` of each class."""
    out_parts, keep_parts = [], []
    for _, group in df.groupby('label'):
        n = max(1, round(len(group) * frac)) if len(group) > 1 else 0
        out = group.sample(n=min(n, len(group)), random_state=seed)
        out_parts.append(out)
        keep_parts.append(group.drop(out.index))
    split_out = pd.concat(out_parts).reset_index(drop=True) if out_parts else df.iloc[0:0]
    kept = pd.concat(keep_parts).reset_index(drop=True) if keep_parts else df.iloc[0:0]
    return kept, split_out

SOURCE_URLS = [
    "https://huggingface.co/datasets/FeatEng/Data/resolve/main/shivamb__vehicle-claim-fraud-detection/train-00000-of-00001.parquet",
    "https://huggingface.co/datasets/FeatEng/Data/resolve/main/shivamb__vehicle-claim-fraud-detection/test-00000-of-00001.parquet",
]
FRAUD_POOL_PATH = "data/_fraud_pool.csv"
NONFRAUD_POOL_PATH = "data/_nonfraud_pool.csv"
PROGRESS_PATH = "data/_ingest_progress.json"
CLAIMS_PATH = "data/claims.csv"                 # train-eligible pool only — never contains reserved eval rows
EVAL_POOL_PATH = "data/_eval_pool.csv"           # permanently reserved — NO model ever trains on this
NEW_EVAL_BATCH_PATH = "data/_new_eval_batch.csv" # just this run's newly reserved eval rows
DTYPES: dict[Any, Any] = {'claim_text': str, 'label': 'int64', 'claim_id': 'int64'}

BATCH_SIZE = 1000
MIN_FRAUD_PER_BATCH = 15
EVAL_RESERVE_FRAC = 0.20  # this run's new data set aside for eval, per your "20% of new data" spec

os.makedirs("data", exist_ok=True)

if not os.path.exists(FRAUD_POOL_PATH):
    frames = [pd.read_parquet(u) for u in SOURCE_URLS]
    raw = pd.concat(frames, ignore_index=True)
    raw['claim_text'] = raw.apply(row_to_text, axis=1).apply(clean_text)
    raw['label'] = raw['__FeatEng_target__']
    df = raw[['claim_text', 'label']].dropna().reset_index(drop=True)
    df['claim_id'] = df.index

    fraud = df[df['label'] == 1].sample(frac=1, random_state=7).reset_index(drop=True)
    nonfraud = df[df['label'] == 0].sample(frac=1, random_state=7).reset_index(drop=True)
    fraud.to_csv(FRAUD_POOL_PATH, index=False)
    nonfraud.to_csv(NONFRAUD_POOL_PATH, index=False)
    print(f"Cached pools: {len(fraud)} fraud, {len(nonfraud)} non-fraud")

fraud_pool = pd.read_csv(FRAUD_POOL_PATH, dtype=DTYPES)
nonfraud_pool = pd.read_csv(NONFRAUD_POOL_PATH, dtype=DTYPES)

overall_fraud_rate = len(fraud_pool) / (len(fraud_pool) + len(nonfraud_pool))
fraud_per_batch = max(MIN_FRAUD_PER_BATCH, round(BATCH_SIZE * overall_fraud_rate))
nonfraud_per_batch = BATCH_SIZE - fraud_per_batch

progress = json.load(open(PROGRESS_PATH)) if os.path.exists(PROGRESS_PATH) else {"fraud_ingested": 0, "nonfraud_ingested": 0}
f_start, f_end = progress["fraud_ingested"], min(progress["fraud_ingested"] + fraud_per_batch, len(fraud_pool))
n_start, n_end = progress["nonfraud_ingested"], min(progress["nonfraud_ingested"] + nonfraud_per_batch, len(nonfraud_pool))

if f_start >= len(fraud_pool) and n_start >= len(nonfraud_pool):
    print("No new claims left in either pool — no new batch this run")
    pd.DataFrame(columns=['claim_text', 'label', 'claim_id']).to_csv(NEW_EVAL_BATCH_PATH, index=False)
else:
    raw_batch = pd.concat([fraud_pool.iloc[f_start:f_end], nonfraud_pool.iloc[n_start:n_end]]).sample(frac=1, random_state=42)

    # Split THIS batch: 80% goes to the trainable pool, 20% goes to the permanent eval reserve.
    # The eval slice is never written to claims.csv, so no model — champion or challenger — ever trains on it.
    train_slice, eval_slice = stratified_split(raw_batch, EVAL_RESERVE_FRAC, seed=99)

    train_slice.to_csv(CLAIMS_PATH, mode='a' if os.path.exists(CLAIMS_PATH) else 'w',
                        header=not os.path.exists(CLAIMS_PATH), index=False)
    eval_slice.to_csv(EVAL_POOL_PATH, mode='a' if os.path.exists(EVAL_POOL_PATH) else 'w',
                       header=not os.path.exists(EVAL_POOL_PATH), index=False)
    eval_slice.to_csv(NEW_EVAL_BATCH_PATH, index=False)  # this run's new eval rows only

    progress["fraud_ingested"], progress["nonfraud_ingested"] = f_end, n_end
    json.dump(progress, open(PROGRESS_PATH, 'w'))
    print(f"Ingested {len(train_slice)} trainable + {len(eval_slice)} reserved-for-eval-only claims")

total = pd.read_csv(CLAIMS_PATH, dtype=DTYPES) if os.path.exists(CLAIMS_PATH) else pd.DataFrame()
print(f"data/claims.csv (trainable) now has {len(total)} records")
if len(total): print(total["label"].value_counts())