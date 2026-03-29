import pandas as pd
import numpy as np

df_amox = pd.read_csv("data/raw/Amoxicillin.csv")
df_atorv = pd.read_csv("data/raw/Atorvastatin.csv")
df_paracetamol = pd.read_csv("data/raw/Paracetamol.csv")

def clean(df, drug_name):
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"items": "demand", "actual_cost": "cost"})
    df["drug"] = drug_name
    df = df.drop('row_name', axis=1)
    return df[["date", "drug", "demand", "cost"]]

df_amox = clean(df_amox, "Amoxicillin")
df_atorv = clean(df_atorv, "Atorvastatin")
df_paracetamol = clean(df_paracetamol, "Paracetamol")
df = pd.concat([df_amox, df_paracetamol, df_atorv], ignore_index=True)

print(df.shape)

# Expand across pharmacies
N_PHARMACIES = 100

pharmacies = pd.DataFrame({
    "pharmacy_id": range(1, N_PHARMACIES + 1)
})

# CROSS JOIN

df_expanded = df.merge(pharmacies, how="cross")
print(df_expanded.shape)

# Added Variation
np.random.seed(42)

variation = np.random.uniform(0.5, 1.5, len(df_expanded))

df_expanded["demand"] = (df_expanded["demand"] * variation).astype(int)

# Added City Tiers

cities = ["Tier1", "Tier2", "Tier3"]

df_expanded["city"] = np.random.choice(
    cities,
    len(df_expanded),
    p=[0.4, 0.4, 0.2]
)

# Add Supply Chain Features

df_expanded["stock"] = df_expanded["demand"] + np.random.randint(-50, 100, len(df_expanded))
df_expanded["stock"] = df_expanded["stock"].clip(lower=0)

# Lead Time
df_expanded["lead_time_days"] = np.random.randint(1, 7, len(df_expanded))

# Save Final Datset
df_expanded.to_parquet("data/processed/final_pharmalink_dataset.parquet", index=False)