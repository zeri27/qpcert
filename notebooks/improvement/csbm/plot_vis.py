import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

files = sorted(glob.glob("gcn-*.csv"))
all_data = []

for f in files:
    delta = float(f.split("-")[1].replace(".csv", ""))
    df = pd.read_csv(f)
    df['delta'] = delta

    def get_robustness_label(row):
        sample = row['samplewise_robust']
        coll = row['collective_robust']
        if sample and coll == 1.0:
            return 'Both'
        elif sample:
            return 'Samplewise only'
        elif coll == 1.0:
            return 'Collective only'
        else:
            return 'None'

    df['robustness_type'] = df.apply(get_robustness_label, axis=1)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=combined_df,
    x='feature_norm',
    y='delta',
    hue='robustness_type',
    style='robustness_type',
    alpha=0.7,
    palette='Set2'
)
plt.xlabel("Feature Norm")
plt.ylabel("Label Poisoning Level (delta)")
plt.title("Feature Norm vs Poisoning Level Colored by Robustness Type")
plt.grid(True)
plt.legend(title='Robustness')
plt.savefig("cbsm_gcn_scatterplot.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
sns.histplot(data=combined_df, x='feature_norm', hue='robustness_type', multiple='stack', bins=30)
plt.title("Feature Norm Distribution by Robustness Type")
plt.xlabel("Feature Norm")
plt.ylabel("Count")
plt.savefig("csbm_gcn_histogram.png", dpi=300, bbox_inches='tight')

