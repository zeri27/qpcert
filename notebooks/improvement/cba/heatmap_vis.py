import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from glob import glob
from matplotlib.colors import ListedColormap

files = sorted(glob("gcn-*.csv"))
heatmap_data = {}

for file in files:
    delta_match = re.search(r"gcn-([\d\.]+)\.csv", file)
    if not delta_match:
        continue
    delta = float(delta_match.group(1))
    df = pd.read_csv(file)

    def get_category(row):
        samplewise = row['samplewise_robust']
        collective = row['collective_robust'] == 1.0

        if samplewise and collective:
            return 2  # Both
        elif samplewise:
            return 1  # Samplewise only
        elif collective:
            return -1  # Collective only
        else:
            return 0  # None

    df['robust_category'] = df.apply(get_category, axis=1)
    heatmap_data[delta] = df.set_index('node')['robust_category']


heatmap_df = pd.DataFrame(heatmap_data).sort_index(axis=1)

cmap = ListedColormap(["red", "orange", "blue", "green"])
value_map = {0: 0, -1: 1, 1: 2, 2: 3}
plot_data = heatmap_df.replace(value_map)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(plot_data, cmap=cmap, cbar_kws={
    'ticks': [0.4, 1.1, 1.9, 2.6]
})

colorbar = ax.collections[0].colorbar
colorbar.set_ticklabels(['None', 'Collective Only', 'Samplewise Only', 'Both'])
colorbar.set_label("Robustness Category")
plt.xticks(rotation=45)
plt.xlabel("Delta")
plt.ylabel("Test Node ID")
plt.title("Node Robustness Heatmap across Deltas")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cba_gcn_heatmap.png", dpi=300, bbox_inches='tight')