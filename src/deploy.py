import model_def  # noqa: F401  (needed to unpickle the pipeline)
import json, shutil, os, joblib

info = json.load(open('artifacts/production_metrics.json'))
src = f"artifacts/model_{info['version']}.pkl"
dst = 'production/model.pkl'
os.makedirs('production', exist_ok=True)
if os.path.exists(dst):
    shutil.copy(dst, 'production/model_previous.pkl')  # one-step rollback copy
shutil.copy(src, dst)
json.dump(info, open('production/model_info.json', 'w'))

# Smoke test: load the deployed file and score one claim
model = joblib.load(dst)
sample = "accidentarea_urban vehiclecategory_sport fault_policyholder policytype_sport_collision basepolicy_collision pastnumberofclaims_2_to_4 policereportfiled_no witnesspresent_no age_30s"
print(f"Deployed v{info['version']} (ROC-AUC {info['roc_auc']:.4f}, fraud PR-AUC {info['pr_auc']:.4f}). Smoke test fraud probability: {model.predict_proba([sample])[0][1]:.3f}")