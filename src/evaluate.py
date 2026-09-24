import model_def  # noqa: F401  (needed to unpickle the pipeline)
import pandas as pd, joblib, json, os, datetime
from typing import Any
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

TOLERANCE = 0.005  # ROC-AUC drop we ignore as noise

def stratified_sample(df, frac, seed):
    if len(df) == 0:
        return df
    parts = []
    for _, group in df.groupby('label'):
        n = len(group) if len(group) <= 1 else max(1, round(len(group) * frac))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True) if parts else df

DTYPES: dict[Any, Any] = {'claim_text': str, 'label': 'int64', 'claim_id': 'int64'}
latest = json.load(open('artifacts/latest_version.json'))
version = latest['version']
flag_path = 'artifacts/eval_passed.flag'
if os.path.exists(flag_path):
    os.remove(flag_path)  # never let a stale flag from an earlier pass leak into this run

train_pool = pd.read_csv('data/claims.csv', dtype=DTYPES)
new_eval_batch = pd.read_csv('data/_new_eval_batch.csv', dtype=DTYPES)
eval_pool = pd.read_csv('data/_eval_pool.csv', dtype=DTYPES) if os.path.exists('data/_eval_pool.csv') else new_eval_batch
older_eval = eval_pool[~eval_pool['claim_id'].isin(new_eval_batch['claim_id'])]

# 20% of new data (already reserved at ingest) + 10% of previously reserved data. None of it is ever trained on.
test_new = stratified_sample(new_eval_batch, 1.0, seed=1)
test_prev = stratified_sample(older_eval, 0.10, seed=1)
test_df = pd.concat([test_new, test_prev]).drop_duplicates(subset='claim_id')
assert not set(test_df['claim_id']) & set(train_pool['claim_id']), "LEAKAGE: eval rows found in training pool"

if len(test_df) == 0:
    raise SystemExit("No evaluation data available — halting pipeline")
print(f"Test set: {len(test_new)} new + {len(test_prev)} previous = {len(test_df)} (never trained on)")

def score(model, df):
    proba = model.predict_proba(df['claim_text'])[:, list(model.classes_).index(1)]
    pred = model.predict(df['claim_text'])
    return {'pr_auc': average_precision_score(df['label'], proba),
            'roc_auc': roc_auc_score(df['label'], proba) if df['label'].nunique() > 1 else float('nan'),
            'fraud_f1': f1_score(df['label'], pred, pos_label=1, average='binary', zero_division=0),
            'weighted_f1': f1_score(df['label'], pred, average='weighted')}

# Evaluate the exact artifact produced by the Model Retraining stage (no second, silent retrain).
challenger = joblib.load(f'artifacts/model_{version}.pkl')
new = score(challenger, test_df)

prod_path = 'artifacts/production_metrics.json'
if os.path.exists(prod_path):
    champion_info = json.load(open(prod_path))
    champion = joblib.load(f"artifacts/model_{champion_info['version']}.pkl")
    old = score(champion, test_df)
else:
    old, champion_info = {'pr_auc': 0.0, 'roc_auc': 0.0, 'fraud_f1': 0.0, 'weighted_f1': 0.0}, {'version': 'none'}

print(f"Fraud PR-AUC  -> Challenger: {new['pr_auc']:.4f} | Champion (v{champion_info['version']}): {old['pr_auc']:.4f}")
base = (test_df['label'] == 1).mean()
print(f"ROC-AUC       -> Challenger: {new['roc_auc']:.4f} | Champion: {old['roc_auc']:.4f}  (random = 0.5)")
print(f"Lift over random (PR-AUC / fraud rate {base:.3f}) -> Challenger: {new['pr_auc']/base:.2f}x | Champion: {old['pr_auc']/base:.2f}x")
print(f"Fraud F1      -> Challenger: {new['fraud_f1']:.4f} | Champion: {old['fraud_f1']:.4f}")
print(f"Weighted F1   -> Challenger: {new['weighted_f1']:.4f} | Champion: {old['weighted_f1']:.4f}")

passed = new['roc_auc'] >= old['roc_auc'] - TOLERANCE
row = pd.DataFrame([{'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                     'version': version, 'train_records': len(train_pool), 'test_size': len(test_df),
                     'test_fraud': int((test_df['label'] == 1).sum()),
                     'challenger_pr_auc': round(float(new['pr_auc']), 4), 'champion_pr_auc': round(float(old['pr_auc']), 4),
                     'challenger_roc_auc': round(float(new['roc_auc']), 4), 'champion_roc_auc': round(float(old['roc_auc']), 4),
                     'challenger_fraud_f1': round(float(new['fraud_f1']), 4), 'champion_fraud_f1': round(float(old['fraud_f1']), 4),
                     'challenger_weighted_f1': round(float(new['weighted_f1']), 4), 'champion_weighted_f1': round(float(old['weighted_f1']), 4),
                     'passed': passed}])
history_path = 'artifacts/metrics_history.csv'
hist = pd.concat([pd.read_csv(history_path), row], ignore_index=True) if os.path.exists(history_path) else row
hist.to_csv(history_path, index=False)

if passed:
    json.dump({'roc_auc': float(new['roc_auc']), 'pr_auc': float(new['pr_auc']), 'fraud_f1': float(new['fraud_f1']), 'version': version}, open(prod_path, 'w'))
    open(flag_path, 'w').close()
    print("PASSED — promoting model")
else:
    os.remove(f'artifacts/model_{version}.pkl')  # don't leave rejected models lying around
    raise SystemExit("FAILED — new model's ROC-AUC is below the champion's, halting pipeline")