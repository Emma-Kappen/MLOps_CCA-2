import pandas as pd, joblib, json, datetime, os
from model_def import build_pipeline

df = pd.read_csv('data/claims.csv')

pipeline = build_pipeline()
pipeline.fit(df['claim_text'], df['label'])

version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('artifacts', exist_ok=True)
joblib.dump(pipeline, f'artifacts/model_{version}.pkl')
json.dump({'version': version}, open('artifacts/latest_version.json', 'w'))
print(f"Trained model version {version} on {len(df)} records")