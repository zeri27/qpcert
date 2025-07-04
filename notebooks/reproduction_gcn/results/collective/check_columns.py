import pandas as pd

# Check SGC data
try:
    sgc_data = pd.read_csv("logs/final_sgc_on_csbm_collective_data.csv")
    print("SGC columns:", list(sgc_data.columns))
    print(f"SGC shape: {sgc_data.shape}")
    print("\nSGC first few rows:")
    print(sgc_data.head())
except FileNotFoundError:
    print("SGC file not found")

print("\n" + "="*50 + "\n")

# Check GCN data  
try:
    gcn_data = pd.read_csv("logs/final_gcn_on_csbm_collective_data.csv")
    print("GCN columns:", list(gcn_data.columns))
    print(f"GCN shape: {gcn_data.shape}")
    print("\nGCN first few rows:")
    print(gcn_data.head())
except FileNotFoundError:
    print("GCN file not found")
