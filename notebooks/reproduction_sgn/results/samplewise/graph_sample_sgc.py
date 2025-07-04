import os
import pandas as pd
import re


def extract_csv_data(directory):
    pattern = re.compile(r'^(csbm|cba)-(\d+\.\d+)\.csv$')
    all_data = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            dataset, delta = match.groups()
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            df['dataset'] = dataset
            df['delta'] = float(delta)
            all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def summarize_data(df):
    metrics = ['accuracy_test', 'accuracy_cert_pois_robust', 'runtime']
    summary_mean = df.groupby(['dataset', 'delta'])[metrics].mean().reset_index()
    summary_std = df.groupby(['dataset', 'delta'])[metrics].std().reset_index()
    summary = pd.merge(summary_mean, summary_std, on=['dataset', 'delta'], suffixes=('_mean', '_std'))

    def fmt(mean, std):
        return f"{mean:.2f} ± {std:.2f}"

    summary['Test Accuracy'] = summary.apply(lambda row: fmt(row['accuracy_test_mean'], row['accuracy_test_std']),
                                             axis=1)
    summary['Certified Robustness'] = summary.apply(
        lambda row: fmt(row['accuracy_cert_pois_robust_mean'], row['accuracy_cert_pois_robust_std']), axis=1)
    summary['Runtime'] = summary.apply(lambda row: fmt(row['runtime_mean'], row['runtime_std']), axis=1)
    summary = summary[['dataset', 'delta', 'Test Accuracy', 'Certified Robustness', 'Runtime']]
    return summary


if __name__ == "__main__":
    data_dir = r"C:\Users\zerya\OneDrive\Desktop\TU Delft\000 MASTERS\Q4\Scalable ML Research\qpcert\notebooks\reproduction_sgn\results\samplewise"
    df = extract_csv_data(data_dir)
    summary = summarize_data(df)
    summary.to_csv("summary_stats.csv", index=False)
    print(summary)
