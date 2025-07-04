import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

csv_files = glob.glob("gcn-*.csv")
all_dfs = []

for file in csv_files:
    perturbation = float(os.path.splitext(os.path.basename(file))[0].split('-')[1])
    df = pd.read_csv(file)
    df['perturbation'] = perturbation
    all_dfs.append(df)

combined_df = pd.concat(all_dfs)


def classify_robustness(group):
    samplewise = group['samplewise_robust'].all()
    collective = (group['collective_robust'] == 1.0).all()
    if samplewise and collective:
        rtype = 'both'
    elif samplewise:
        rtype = 'samplewise'
    elif collective:
        rtype = 'collective'
    else:
        rtype = 'none'
    return pd.Series({
        'robust_type': rtype,
        'mean_feature_norm': group['feature_norm'].mean()
    })


summary_df = combined_df.groupby('node').apply(classify_robustness).reset_index()

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
scatter = sns.scatterplot(
    data=summary_df,
    x='node',
    y='mean_feature_norm',
    hue='robust_type',
    palette={
        'both': 'green',
        'samplewise': 'blue',
        'collective': 'orange',
        'none': 'red'
    },
    alpha=0.8,
    s=70
)

plt.title("Node Feature Norms by Robustness Type", fontsize=14)
plt.xlabel("Node ID", fontsize=12)
plt.ylabel("Mean Feature Norm", fontsize=12)
plt.legend(title="Robustness Type")
plt.tight_layout()

plt.savefig("cba_gcn_featurenorm_plot.png", dpi=300, bbox_inches='tight')
