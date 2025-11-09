import pandas as pd
import numpy as np

data = pd.read_csv(r"lats1_ligands_descriptors.csv")
data.fillna(0, inplace=True)

target_col = 'pIC50'
reserved_cols = ['name', 'SMILES', target_col]
reserved_cols = [col for col in reserved_cols if col in data.columns]

feature_df = data.drop(columns=[c for c in reserved_cols if c in data.columns])

value_ratios = feature_df.apply(lambda col: col.value_counts(normalize=True).iloc[0] if not col.value_counts().empty else 0)
const_cols = value_ratios[value_ratios >= 0.8].index.tolist()

if const_cols:
    feature_df.drop(columns=const_cols, inplace=True)
    print(f"Removed {len(const_cols)} near-constant columns: {const_cols[:10]}{'...' if len(const_cols)>10 else ''}")
else:
    print("No near-constant columns found.")

print("Computing correlation matrix...")
corr_matrix = feature_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = set()

for col in upper_tri.columns:
    high_corr = upper_tri[col][upper_tri[col] > 0.8].index.tolist()
    for hc in high_corr:
        corr_col = abs(np.corrcoef(feature_df[col], data[target_col])[0, 1])
        corr_hc = abs(np.corrcoef(feature_df[hc], data[target_col])[0, 1])
        if corr_col < corr_hc:
            to_drop.add(col)
        else:
            to_drop.add(hc)

feature_df.drop(columns=list(to_drop), inplace=True)
print(f"Removed {len(to_drop)} highly correlated columns (>0.8).")

final_df = pd.concat([data[reserved_cols], feature_df], axis=1)

output_path = r"lats1_qsar_dataset.csv"
final_df.to_csv(output_path, index=False)

print(f"✅ Final dataset saved to: {output_path}")
print(f"Total features remaining: {feature_df.shape[1]}")
