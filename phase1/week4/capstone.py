import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# ==========================================
# STEP 0: GENERATE RAW, MESSY TELEMETRY DATA
# ==========================================
def generate_raw_data(num_samples=200):
    print("⏳ Simulating raw database dump (cluster_telemetry.csv)...")

    active_containers = np.random.randint(5, 50, size=num_samples).astype(float)
    api_latency_ms = np.random.exponential(scale=50, size=num_samples) ** 2 + 10
    ram_allocated_gb = active_containers * np.random.uniform(2, 4, size=num_samples)
    ram_allocated_mb = ram_allocated_gb * 1024 + np.random.normal(0, 5, size=num_samples)

    df = pd.DataFrame({
        'Containers_Active': active_containers,
        'API_Latency_ms': api_latency_ms,
        'RAM_Allocated_GB': ram_allocated_gb,
        'RAM_Allocated_MB': ram_allocated_mb,
    })

    # Inject dirty data: 5% random NaNs
    df.loc[df.sample(frac=0.05).index, 'Containers_Active'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'API_Latency_ms'] = np.nan

    return df

raw_df = generate_raw_data()

# ==========================================
# STEP 1: STRUCTURE AUDIT & MISSING DATA FIX
# ==========================================
print("\n--- STAGE 1: Structure Audit ---")
print(f"📊 Raw Dataset Dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
print("\n🔍 Missing Value Footprint:")
print(raw_df.isna().sum())

df_clean = raw_df.copy()
for col in ['Containers_Active', 'API_Latency_ms']:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)
print("\n✅ Missing data fixed via Median Imputation.")

# ==========================================
# STEP 2: DISTRIBUTION FIX (LOG TRANSFORM)
# ==========================================
print("\n--- STAGE 2: Distribution Audit ---")
print(f"📈 Raw Latency Skewness Score: {df_clean['API_Latency_ms'].skew():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_clean['API_Latency_ms'], kde=True, ax=axes[0], color='crimson')
axes[0].set_title("⚠️ Raw Latency (Heavy Skew / Outliers)")

df_clean['Log_API_Latency'] = np.log1p(df_clean['API_Latency_ms'])
df_clean = df_clean.drop(columns=['API_Latency_ms'])

sns.histplot(df_clean['Log_API_Latency'], kde=True, ax=axes[1], color='teal')
axes[1].set_title("✨ Log Transformed Latency (Clean Bell Curve)")
plt.tight_layout()
plt.show(block=False)

print(f"📉 Transformed Latency Skewness Score: {df_clean['Log_API_Latency'].skew():.2f}")
print("✅ High skew resolved using log1p transform.")

# ==========================================
# STEP 3: REDUNDANCY PRUNING (CORRELATION)
# ==========================================
print("\n--- STAGE 3: Redundancy Audit (Multicollinearity) ---")
corr_matrix = df_clean.corr()
print("\n🔢 Correlation Coefficients Matrix:")
print(corr_matrix.round(2))

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title("Feature Correlation Matrix Heatmap")
plt.tight_layout()
plt.show(block=False)

df_clean = df_clean.drop(columns=['RAM_Allocated_MB'])
print("\n✅ Pruned redundant feature 'RAM_Allocated_MB' to resolve Multicollinearity.")

# ==========================================
# STEP 4: FEATURE SCALING (STANDARDIZATION)
# ==========================================
print("\n--- STAGE 4: Feature Scaling (Z-Score) ---")
print("📋 Features prior to scaling (Notice raw unit imbalances):")
print(df_clean.head(3))

final_features = df_clean.copy()
for col in final_features.columns:
    mean = final_features[col].mean()
    std = final_features[col].std()
    final_features[col] = (final_features[col] - mean) / std

print("\n🚀 Final Cleaned, Standardized Features (Ready for ML Models!):")
print(final_features.head(3))
print(f"\nFinal Matrix Dimensions: {final_features.shape}")

print("\n💡 Review the generated plots on your screen. Close them to finish.")
plt.show()
