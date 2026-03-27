import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf


BASE_DIR = Path(__file__).parent
sns.set(style='whitegrid', palette='muted', font_scale=1.0)


cols = ['id', 'timestamp', 'machine_id', 'cpu_usage', 'memory_usage']
df = pd.read_csv(BASE_DIR / "alibaba_cluster_data.csv", names=cols)


df = df.drop_duplicates()

# Fill missing values
df = df.interpolate(method='linear').bfill().ffill()

# Remove outliers using IQR
cols_to_use = ['cpu_usage', 'memory_usage']
for col in cols_to_use:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Normalize
scaler = MinMaxScaler()
df[cols_to_use] = scaler.fit_transform(df[cols_to_use])


# per timestamp 

data = df.groupby('timestamp')[cols_to_use].mean().reset_index(drop=True)
cpu = 'cpu_usage'
ram = 'memory_usage'

# Optional: smoothed series for visualization
data['cpu_smooth'] = data[cpu].rolling(window=3).mean()
data['ram_smooth'] = data[ram].rolling(window=3).mean()


fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# CPU
axes[0].plot(data[cpu], color='blue', alpha=0.5, linewidth=0.8, label='Raw')
axes[0].plot(data['cpu_smooth'], color='blue', linewidth=1.5, label='Smoothed')
axes[0].set_title('CPU Usage Over Time', fontsize=12, fontweight='bold')
axes[0].set_ylabel('CPU ')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Memory
axes[1].plot(data[ram], color='orange', alpha=0.5, linewidth=0.8, label='Raw')
axes[1].plot(data['ram_smooth'], color='orange', linewidth=1.5, label='Smoothed')
axes[1].set_title('Memory Usage Over Time', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Memory ')
axes[1].set_xlabel('Time Index')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(BASE_DIR / "01_timeseries_cluster.png", dpi=100)
plt.show()

# Decomposition 

period = 24  

fig, axes = plt.subplots(4, 2, figsize=(14, 12)) 

for idx, col in enumerate([cpu, ram]):
    decomp = seasonal_decompose(data[col], model='additive', period=period)
    
    #  remove NaNs
    trend = decomp.trend.dropna()
    resid = decomp.resid.dropna()
    seasonal = decomp.seasonal[trend.index]  # align with trend
    observed = decomp.observed[trend.index]
    
    # Observed
    axes[0, idx].plot(observed, color='black', linewidth=0.8)
    axes[0, idx].set_title(f'{col} - ORIGINAL', fontsize=10, fontweight='bold')
    axes[0, idx].set_ylabel('Value')
    axes[0, idx].grid(True, alpha=0.3)
    
    # Trend
    axes[1, idx].plot(trend, color='red', linewidth=1.5)
    axes[1, idx].set_title(f'{col} - TREND', fontsize=10, fontweight='bold')
    axes[1, idx].set_ylabel('Trend')
    axes[1, idx].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2, idx].plot(seasonal, color='green', linewidth=1.2)
    axes[2, idx].set_title(f'{col} - SEASONAL', fontsize=10, fontweight='bold')
    axes[2, idx].set_ylabel('Seasonal')
    axes[2, idx].grid(True, alpha=0.3)
    
    # Residual
    axes[3, idx].plot(resid, color='purple', linewidth=0.8)
    axes[3, idx].set_title(f'{col} - RESIDUAL (noise)', fontsize=10, fontweight='bold')
    axes[3, idx].set_ylabel('Residual')
    axes[3, idx].set_xlabel('Time Index')
    axes[3, idx].grid(True, alpha=0.3)

fig.suptitle(f'Decomposition  ( Period={period})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE_DIR / "02_decomposition_cluster_seasonal.png", dpi=100, bbox_inches='tight')
plt.show()

# Autocorrelation (ACF) 
from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
nlags = 30  # how many steps back to check

for idx, col in enumerate([cpu, ram]):
    # Decompo
    decomp = seasonal_decompose(data[col], model='additive', period=period)
    seasonal = decomp.seasonal.dropna()
    
    # Plot ACF 
    plot_acf(seasonal, lags=nlags, ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'{col}  ACF ', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Lag (timesteps)')
    axes[idx].set_ylabel('Correlation')
    axes[idx].grid(True, alpha=0.3)

fig.suptitle(' Autocorrelation ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE_DIR / "04_acf.png", dpi=100, bbox_inches='tight')
plt.show()

# Statistics
print("STATISTICS")
for col in [cpu, ram]:
    print(f"\n{col}:")
    print(f"  Average: {data[col].mean():.3f}")
    print(f"  Min:     {data[col].min():.3f}")
    print(f"  Max:     {data[col].max():.3f}")
    print(f"  Std dev: {data[col].std():.3f}")