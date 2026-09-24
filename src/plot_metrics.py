import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if not os.path.exists('artifacts/metrics_history.csv'):
    raise SystemExit("No metrics history yet")
df = pd.read_csv('artifacts/metrics_history.csv')
df['run'] = range(1, len(df) + 1)
df['baseline'] = df['test_fraud'] / df['test_size']
ok, bad = df[df['passed']], df[~df['passed']]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
panels = [('roc_auc', 'ROC-AUC (gate metric)', pd.Series(0.5, index=df.index), 'Random baseline (0.5)'),
          ('pr_auc', 'Fraud PR-AUC (context)', df['baseline'], 'Random baseline (fraud rate)')]
for ax, (m, title, base, base_label) in zip(axes, panels):
    if f'challenger_{m}' not in df:
        continue
    ax.plot(df['run'], df[f'challenger_{m}'], color='tab:blue', label='Challenger (new model)')
    ax.plot(df['run'], df[f'champion_{m}'], color='tab:orange', linestyle='--', label='Champion (production)')
    ax.plot(df['run'], base, color='gray', linestyle=':', label=base_label)
    ax.scatter(ok['run'], ok[f'challenger_{m}'], color='tab:green', zorder=3, label='Passed - promoted')
    ax.scatter(bad['run'], bad[f'challenger_{m}'], color='tab:red', marker='X', s=90, zorder=3, label='Failed - rejected')
    ax.set_title(title); ax.set_xlabel('Retraining Run #'); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('artifacts/metric_trend.png')
print("Saved trend chart -> artifacts/metric_trend.png")
print(df[['run', 'timestamp', 'train_records', 'test_fraud',
          'challenger_roc_auc', 'champion_roc_auc',
          'challenger_pr_auc', 'champion_pr_auc', 'passed']].to_string(index=False))
