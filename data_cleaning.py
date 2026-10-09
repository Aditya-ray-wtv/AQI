"""
=============================================================================
AQI INDEX INDIA — Data Cleaning Script
=============================================================================
Topic: Air Quality Index (AQI) Analysis — India
Dataset: archive/ folder containing city-wise AQI data (10 Indian cities)
Reference: PandaToReadFile.ipynb (same cleaning concepts applied)

Steps followed (from PandaToReadFile.ipynb):
    1. Defining problem statement
    2. Finding the proper data source
    3. Data collection and inspection of data
    4. Data cleaning
    5. Data analysis
    6. Decision making
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
# Load all city CSV files from the archive folder and combine them
print("=" * 60)
print("STEP 1: DATA COLLECTION AND INSPECTION")
print("=" * 60)

archive_path = 'archive'
city_files = [f for f in os.listdir(archive_path) if f.endswith('.csv')]

# Read each city file and add a 'City' column
all_city_dfs = []
for file in city_files:
    city_name = file.replace('_combined.csv', '').title()
    temp_df = pd.read_csv(os.path.join(archive_path, file))
    temp_df['City'] = city_name
    all_city_dfs.append(temp_df)
    print(f"  Loaded {file}: {temp_df.shape[0]} rows × {temp_df.shape[1]} columns")

# Combine all city data into one DataFrame
df = pd.concat(all_city_dfs, ignore_index=True)
print(f"\n📊 Combined Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Cities covered: {df['City'].nunique()} — {', '.join(df['City'].unique())}")

# ─── 2. QUICK OVERVIEW ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: QUICK OVERVIEW")
print("=" * 60)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ─── 3. MISSING VALUE ANALYSIS ────────────────────────────────────────────────
# (Same approach as PandaToReadFile.ipynb — Cell 4, 5)
print("\n" + "=" * 60)
print("STEP 3: MISSING VALUE ANALYSIS")
print("=" * 60)

# Check for nulls in each column
print("\n--- Null Count per Column ---")
print(df.isnull().sum())

# Null percentage (same as notebook Cell 5)
print("\n--- Null Percentage per Column ---")
null_pct = df.isnull().sum() / len(df) * 100
print(null_pct)

# Check for 'NA' string values (Bengaluru data has 'NA' strings)
print("\n--- Checking for 'NA' string values ---")
for col in ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3']:
    na_count = (df[col] == 'NA').sum() if df[col].dtype == 'object' else 0
    if na_count > 0:
        print(f"  {col}: {na_count} 'NA' string values found")

# Replace 'NA' strings with NaN
df.replace('NA', np.nan, inplace=True)

# Check duplicates (same as notebook Cell 8, 9)
print(f"\n--- Duplicates ---")
print(f"  Duplicates found: {df[df.duplicated()].shape[0]}")

# Check for unique values in low-cardinality columns (same as notebook Cell 12)
print("\n--- Value Counts for Low-Cardinality Columns ---")
for col in df.columns:
    if df[col].nunique() < 20:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print('-' * 30)

# ─── 4. HANDLING MISSING VALUES ───────────────────────────────────────────────
# Following the same approach as PandaToReadFile.ipynb:
#   - Delete rows/columns when >50% missing
#   - Imputation: median for numerical, mode for categorical
#   - Forward fill for date/time columns
print("\n" + "=" * 60)
print("STEP 4: HANDLING MISSING VALUES")
print("=" * 60)

# Convert pollutant columns to numeric (they may have string 'NA' values)
pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3']
for col in pollutant_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\n  Before handling missing values:")
print(f"  Shape: {df.shape}")
print(f"  Total nulls: {df.isnull().sum().sum()}")

# Drop rows where ALL pollutant values are missing (completely empty readings)
df.dropna(subset=pollutant_cols, how='all', inplace=True)
print(f"\n  After dropping rows with all pollutants missing:")
print(f"  Shape: {df.shape}")

# For numeric columns: fill with median (same as notebook Cell 28, 29)
# Using median because pollutant data can have extreme values/outliers
for col in pollutant_cols:
    col_median = df[col].median()
    df.fillna({col: col_median}, inplace=True)
    print(f"  Filled {col} missing values with median: {col_median:.2f}")

# For categorical columns: fill with mode (same as notebook Cell 34)
for col in ['Location', 'City']:
    df.fillna({col: df[col].mode()[0]}, inplace=True)

# For Timestamp: forward fill (same as notebook Cell 36)
df['Timestamp'] = df['Timestamp'].ffill()

print(f"\n  After imputation:")
print(df.isnull().sum())

# ─── 5. FIXING INCONSISTENT FORMATS ──────────────────────────────────────────
# (Same as notebook Cells 38-44)
print("\n" + "=" * 60)
print("STEP 5: FIXING INCONSISTENT FORMATS")
print("=" * 60)

# Standardize City names — strip whitespace, title case
print("\n--- Before City cleanup ---")
print(f"  Unique cities: {df['City'].unique()}")
df['City'] = df['City'].str.strip().str.title()
print(f"  After cleanup: {df['City'].unique()}")

# Standardize Location names
df['Location'] = df['Location'].str.strip()

# Parse Timestamp to datetime
print("\n--- Converting Timestamp to datetime ---")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y', errors='coerce')
print(f"  Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# Extract useful time features
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Month_Name'] = df['Timestamp'].dt.month_name()
df['Day'] = df['Timestamp'].dt.day
df['Day_of_Week'] = df['Timestamp'].dt.day_name()

# ─── 6. REMOVE DUPLICATES ────────────────────────────────────────────────────
# (Same as notebook Cells 46-48)
print("\n" + "=" * 60)
print("STEP 6: HANDLING DUPLICATES")
print("=" * 60)

print(f"  Duplicates before: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"  Duplicates after:  {df.duplicated().sum()}")
print(f"  Shape after dedup: {df.shape}")

# ─── 7. CORRECTING DATA TYPES ────────────────────────────────────────────────
# (Same as notebook Cells 50, 53, 54)
print("\n" + "=" * 60)
print("STEP 7: CORRECTING DATA TYPES")
print("=" * 60)

# Ensure pollutant columns are float64
for col in pollutant_cols:
    df[col] = df[col].astype('float64')

# Year, Month, Day as int
for col in ['Year', 'Month', 'Day']:
    df[col] = df[col].astype('int64')

print("\n--- Updated Data Types ---")
print(df.dtypes)

# ─── 8. CALCULATE AQI ─────────────────────────────────────────────────────────
# AQI calculation based on Indian AQI standards (CPCB breakpoints)
print("\n" + "=" * 60)
print("STEP 8: CALCULATING AQI")
print("=" * 60)


def calculate_sub_index(value, breakpoints):
    """Calculate sub-index for a pollutant using linear interpolation."""
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= value <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (value - bp_lo) + i_lo
    return np.nan


# Indian AQI breakpoints (CPCB standards) for 24-hour average
PM25_BREAKPOINTS = [
    (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
    (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)
]
PM10_BREAKPOINTS = [
    (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
    (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)
]
NO2_BREAKPOINTS = [
    (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
    (181, 280, 201, 300), (281, 400, 301, 400), (401, 600, 401, 500)
]
SO2_BREAKPOINTS = [
    (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
    (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2400, 401, 500)
]
CO_BREAKPOINTS = [
    (0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
    (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)
]
O3_BREAKPOINTS = [
    (0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
    (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)
]
NH3_BREAKPOINTS = [
    (0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
    (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2400, 401, 500)
]

# Calculate sub-indices for each pollutant
df['PM2.5_SI'] = df['PM2.5'].apply(lambda x: calculate_sub_index(x, PM25_BREAKPOINTS))
df['PM10_SI'] = df['PM10'].apply(lambda x: calculate_sub_index(x, PM10_BREAKPOINTS))
df['NO2_SI'] = df['NO2'].apply(lambda x: calculate_sub_index(x, NO2_BREAKPOINTS))
df['SO2_SI'] = df['SO2'].apply(lambda x: calculate_sub_index(x, SO2_BREAKPOINTS))
df['CO_SI'] = df['CO'].apply(lambda x: calculate_sub_index(x, CO_BREAKPOINTS))
df['O3_SI'] = df['O3'].apply(lambda x: calculate_sub_index(x, O3_BREAKPOINTS))
df['NH3_SI'] = df['NH3'].apply(lambda x: calculate_sub_index(x, NH3_BREAKPOINTS))

# AQI = Maximum of all sub-indices
sub_index_cols = ['PM2.5_SI', 'PM10_SI', 'NO2_SI', 'SO2_SI', 'CO_SI', 'O3_SI', 'NH3_SI']
df['AQI'] = df[sub_index_cols].max(axis=1)

# AQI Category classification
def classify_aqi(aqi):
    """Classify AQI into categories based on Indian standards."""
    if pd.isna(aqi):
        return 'Unknown'
    elif aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'


df['AQI_Category'] = df['AQI'].apply(classify_aqi)

print(f"\n--- AQI Category Distribution ---")
print(df['AQI_Category'].value_counts())
print(f"\n  AQI range: {df['AQI'].min():.1f} to {df['AQI'].max():.1f}")
print(f"  Mean AQI: {df['AQI'].mean():.1f}")

# ─── 9. OUTLIER DETECTION & HANDLING ─────────────────────────────────────────
# (Same approach as notebook Cells 52-59 — using Z-scores)
print("\n" + "=" * 60)
print("STEP 9: HANDLING OUTLIERS (Z-Score Method)")
print("=" * 60)

# Boxplot before outlier removal (same as notebook Cell 56)
cols_for_outlier = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3']
plt.figure(figsize=(18, 5))
for i, col in enumerate(cols_for_outlier, 1):
    plt.subplot(1, 7, i)
    sns.boxplot(y=df[col], color='skyblue', width=0.3)
    plt.title(f'Boxplot of {col}', fontsize=10)
    plt.ylabel('')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.suptitle('Before Outlier Removal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/boxplot_before_outlier_removal.png', dpi=150)
plt.close()
print("  Saved: reports/boxplot_before_outlier_removal.png")

# Z-score based outlier detection (same as notebook Cell 57, 58)
z_scores = np.abs(stats.zscore(df[cols_for_outlier]))
print(f"\n  Shape before outlier removal: {df.shape}")
df_clean = df[~(z_scores > 3).any(axis=1)]
print(f"  Shape after outlier removal:  {df_clean.shape}")
print(f"  Rows removed: {df.shape[0] - df_clean.shape[0]}")

# Boxplot after outlier removal (same as notebook Cell 61)
plt.figure(figsize=(18, 5))
for i, col in enumerate(cols_for_outlier, 1):
    plt.subplot(1, 7, i)
    sns.boxplot(y=df_clean[col], color='lightgreen', width=0.3)
    plt.title(f'Boxplot of {col}', fontsize=10)
    plt.ylabel('')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.suptitle('After Outlier Removal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/boxplot_after_outlier_removal.png', dpi=150)
plt.close()
print("  Saved: reports/boxplot_after_outlier_removal.png")

# ─── 10. CORRELATION HEATMAP ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: CORRELATION ANALYSIS")
print("=" * 60)

numeric_cols = df_clean[pollutant_cols + ['AQI']]
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap — AQI & Pollutants', fontsize=14)
plt.tight_layout()
plt.savefig('reports/correlation_heatmap.png', dpi=150)
plt.close()
print("  Saved: reports/correlation_heatmap.png")

# ─── 11. ADDITIONAL EDA VISUALIZATIONS ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: GENERATING EDA VISUALIZATIONS")
print("=" * 60)

# City-wise Average AQI — Bar Chart
city_aqi = df_clean.groupby('City')['AQI'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
colors = sns.color_palette('RdYlGn_r', len(city_aqi))
bars = plt.bar(city_aqi.index, city_aqi.values, color=colors)
plt.title('City-wise Average AQI (Higher = Worse)', fontsize=14, fontweight='bold')
plt.xlabel('City')
plt.ylabel('Average AQI')
plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, city_aqi.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             f'{val:.0f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('reports/city_avg_aqi_bar.png', dpi=150)
plt.close()
print("  Saved: reports/city_avg_aqi_bar.png")

# Monthly AQI Trend — Line Chart
monthly_aqi = df_clean.groupby(['Year', 'Month'])['AQI'].mean().reset_index()
monthly_aqi['Date'] = pd.to_datetime(monthly_aqi[['Year', 'Month']].assign(Day=1))
plt.figure(figsize=(14, 5))
plt.plot(monthly_aqi['Date'], monthly_aqi['AQI'], color='#e74c3c', linewidth=2, marker='o', markersize=4)
plt.fill_between(monthly_aqi['Date'], monthly_aqi['AQI'], alpha=0.15, color='#e74c3c')
plt.title('Monthly Average AQI Trend — All Cities Combined', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Average AQI')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('reports/monthly_aqi_trend.png', dpi=150)
plt.close()
print("  Saved: reports/monthly_aqi_trend.png")

# AQI Category Distribution — Pie Chart
cat_counts = df_clean['AQI_Category'].value_counts()
cat_colors = {
    'Good': '#00b050', 'Satisfactory': '#92d050', 'Moderate': '#ffff00',
    'Poor': '#ff9900', 'Very Poor': '#ff0000', 'Severe': '#c00000', 'Unknown': '#808080'
}
plt.figure(figsize=(8, 8))
plt.pie(cat_counts.values,
        labels=cat_counts.index,
        colors=[cat_colors.get(c, '#808080') for c in cat_counts.index],
        autopct='%1.1f%%', startangle=140, textprops={'fontsize': 11})
plt.title('AQI Category Distribution — India', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/aqi_category_pie.png', dpi=150)
plt.close()
print("  Saved: reports/aqi_category_pie.png")

# PM2.5 vs PM10 Scatter Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_clean['PM2.5'], df_clean['PM10'],
                      c=df_clean['AQI'], cmap='RdYlGn_r', alpha=0.5, s=10)
plt.colorbar(scatter, label='AQI')
plt.xlabel('PM2.5 (µg/m³)')
plt.ylabel('PM10 (µg/m³)')
plt.title('PM2.5 vs PM10 Scatter Plot (colored by AQI)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/pm25_vs_pm10_scatter.png', dpi=150)
plt.close()
print("  Saved: reports/pm25_vs_pm10_scatter.png")

# ─── 12. SAVE CLEANED DATASET ────────────────────────────────────────────────
# (Same as notebook Cell 62)
print("\n" + "=" * 60)
print("STEP 12: SAVING CLEANED DATASET")
print("=" * 60)

# Drop sub-index columns before saving (keep it clean)
sub_index_cols_to_drop = ['PM2.5_SI', 'PM10_SI', 'NO2_SI', 'SO2_SI', 'CO_SI', 'O3_SI', 'NH3_SI']
df_clean = df_clean.drop(columns=sub_index_cols_to_drop, errors='ignore')

df_clean.to_csv('data/cleaned/aqi_india_cleaned.csv', index=False)
print(f"  Saved: data/cleaned/aqi_india_cleaned.csv")
print(f"  Final shape: {df_clean.shape}")

# Final info (same as notebook Cell 63)
print("\n--- Final Cleaned Dataset Info ---")
print(df_clean.info())
print(f"\n  Columns: {list(df_clean.columns)}")

print("\n" + "=" * 60)
print("✅ DATA CLEANING COMPLETE!")
print("=" * 60)
print(f"  Original rows: {sum(len(d) for d in all_city_dfs)}")
print(f"  Cleaned rows:  {df_clean.shape[0]}")
print(f"  Columns:       {df_clean.shape[1]}")
print(f"  Cities:        {df_clean['City'].nunique()}")
print(f"  Date range:    {df_clean['Timestamp'].min().date()} to {df_clean['Timestamp'].max().date()}")
print(f"\n  Files created:")
print(f"    • data/cleaned/aqi_india_cleaned.csv")
print(f"    • reports/boxplot_before_outlier_removal.png")
print(f"    • reports/boxplot_after_outlier_removal.png")
print(f"    • reports/correlation_heatmap.png")
print(f"    • reports/city_avg_aqi_bar.png")
print(f"    • reports/monthly_aqi_trend.png")
print(f"    • reports/aqi_category_pie.png")
print(f"    • reports/pm25_vs_pm10_scatter.png")
