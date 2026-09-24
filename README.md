# Automated Insurance Claim Verification — Jenkins Retraining Pipeline

**Case Study 11 · Group 11 · MLOps CCA-2**

| Name | USN |
|---|---|
| Bhoomika S Gowda | 1BY23AI029 |
| Devathi Saranya | 1BY23AI041 |
| Emma Kappen | 1BY23AI049 |
| Honey Hemant | 1BY24AI405 |

## Problem

Manual claim verification is slow and error-prone. This project automates it with an NLP model that flags
likely-fraudulent vehicle insurance claims, and uses a **Jenkins pipeline** to retrain the model as new claims
arrive. A newly trained model is only deployed if it does not regress against the model already in production.

## How it works

Each pipeline run ingests a new batch of claims, retrains a model, evaluates it against the current production
model (the *champion*) on data no model has ever trained on, and promotes it only if it passes.

```
Setup → Data Pull & Prep → Model Retraining → Evaluation ──pass──► Deployment
                                                   │
                                                   └──fail──► build fails, production unchanged
                                   (always) plot metrics + archive chart and history
```

| Stage | Script | What it does |
|---|---|---|
| Setup | — | `pip install -r requirements.txt` |
| Data Pull & Prep | `src/fetch_data.py` | Ingests the next batch of 1,000 claims (stratified so the fraud rate is preserved). 80% goes to the trainable pool `data/claims.csv`; 20% goes to a permanent evaluation reserve that is never trained on. |
| Model Retraining | `src/train.py` | Trains a fresh model on the whole trainable pool and saves it as a versioned `.pkl`. |
| Evaluation | `src/evaluate.py` | Scores the new model (challenger) and the production model (champion) on the same held-out data, and applies the promotion gate. |
| Deployment | `src/deploy.py` | Copies the promoted model to `production/model.pkl`, keeps the previous one for rollback, and runs a smoke-test prediction. |
| Post (always) | `src/plot_metrics.py` | Draws the challenger-vs-champion chart; Jenkins archives it with the metrics history. |

## The model (NLP)

- **Text:** each claim row is converted to a document of field-tagged tokens, e.g.
  `fault_policy_holder policereportfiled_no basepolicy_collision age_30s`. Tagging each value with its column name
  stops identical words in different fields (such as `no`) from being confused. About 26 columns are used;
  columns missing from the source are skipped.
- **Model:** TF-IDF features → `HistGradientBoostingClassifier` with `class_weight='balanced'`, early stopping and
  small trees (see `src/model_def.py`, the single definition shared by training, evaluation and deployment).
- **Data:** the public *Vehicle Claim Fraud Detection* dataset (about 15,400 claims, roughly 6% fraud), pulled from
  Hugging Face on first run and cached under `data/`.

## Evaluation and the promotion gate

- **Test set = 20% of the new batch + 10% of all previously reserved evaluation data.** All of it comes from the
  reserved pool, so no champion or challenger has ever seen it. `evaluate.py` asserts that no test claim appears in
  the training pool.
- **Gate metric: ROC-AUC** (fraud as the positive class). The challenger passes if
  `challenger ROC-AUC >= champion ROC-AUC - 0.005`. The tolerance (`TOLERANCE` in `evaluate.py`) ignores
  noise-level differences. The very first run always passes because there is no champion yet.
- **Also logged for context:** PR-AUC, fraud-class F1, weighted F1, and the fraud rate of the test slice (the
  random-guess baseline for PR-AUC). Accuracy is deliberately not used: predicting "not fraud" for everything is
  about 94% accurate.
- **On failure** the rejected model file is deleted, `evaluate.py` exits with an error, Jenkins marks the build red,
  and the Deployment stage is skipped, so production keeps the previous model.

## Project structure

```
MLOps_CCA-2/
├── Jenkinsfile              # pipeline definition
├── requirements.txt
├── src/
│   ├── fetch_data.py        # batch ingestion + train/eval reserve split
│   ├── model_def.py         # TF-IDF + gradient boosting pipeline
│   ├── train.py
│   ├── evaluate.py          # champion vs challenger gate
│   ├── deploy.py
│   └── plot_metrics.py
├── data/                    # generated: claim pools, eval reserve, ingest progress
├── artifacts/               # generated: models, metrics_history.csv, metric_trend.png
└── production/              # generated: model.pkl, model_previous.pkl, model_info.json
```

## Setup

Requires **Python 3.12 or older** (the pinned scikit-learn/pyarrow versions have no wheels for newer releases)
and internet access on the first run to download the dataset.

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run locally (one retraining cycle)

```powershell
venv\Scripts\python.exe src\fetch_data.py
venv\Scripts\python.exe src\train.py
venv\Scripts\python.exe src\evaluate.py     # exits with an error if the challenger is rejected
venv\Scripts\python.exe src\deploy.py       # only after a pass
venv\Scripts\python.exe src\plot_metrics.py
```

### Run in Jenkins

1. Create a Pipeline job that uses this `Jenkinsfile` (job name used during development: `claim-model-retrain`).
2. Edit `PROJECT_DIR` at the top of the `Jenkinsfile` to the absolute path of this project on the Jenkins machine.
3. Build with **Build Now**. A nightly build is also scheduled (`cron('H 2 * * *')`). Concurrent builds are disabled.
4. After each build, open the archived `metric_trend.png` and `metrics_history.csv` from the build page.

## Outputs

`artifacts/metrics_history.csv` has one row per run: timestamp, model version, training size, test size, number of
fraud rows in the test slice, challenger and champion ROC-AUC / PR-AUC / fraud F1 / weighted F1, and `passed`.
`artifacts/metric_trend.png` plots ROC-AUC (gate metric) and PR-AUC, with the random baselines and passed/failed
runs marked.

## Example result

Over 10 consecutive runs (1,000 claims per run), 6 passed and 4 were rejected. Challenger ROC-AUC ranged from
0.72 to 0.85, and PR-AUC was typically about three times the random baseline of roughly 0.06. Read this with care:
each test slice has only about 12–23 fraud claims, so run-to-run differences of a few hundredths are within noise,
and rejections such as a 0.853 model losing to an 0.869 champion reflect that noise rather than a bad model.
The pipeline's value here is the safeguard, not a steadily rising score.

## Resetting the demo

Delete `artifacts/`, `data/claims.csv`, `data/_eval_pool.csv`, `data/_new_eval_batch.csv` and
`data/_ingest_progress.json`. Keep `data/_fraud_pool.csv` and `data/_nonfraud_pool.csv` to avoid re-downloading.
At 1,000 claims per run the data lasts roughly 15 runs; after that no new batch arrives and every run passes trivially.
If you change the text features in `fetch_data.py`, also delete the two pool files so they are rebuilt.

## Troubleshooting

- **`DLL load failed ... Application Control policy has blocked this file`** — Windows (Smart App Control / WDAC)
  is blocking a compiled scikit-learn file. Check that `import sklearn.ensemble` works under the account Jenkins
  runs as, try another scikit-learn wheel version, or adjust the policy.
- **NumPy / scikit-learn import errors** — keep `numpy<2` with the pinned scikit-learn version.
- **Loading `production/model.pkl` elsewhere fails** — the pickle references `model_def.py`; make sure it is importable.

## Limitations

- The claim "text" is generated from tabular fields, not free-text claim narratives.
- Fraud is rare and the reserved evaluation slices are small, so metrics are noisy and improvement over runs is not
  guaranteed to be visible.
- Model files are stored locally; there is no model registry or serving endpoint beyond `production/model.pkl`.