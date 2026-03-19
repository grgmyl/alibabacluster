import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path



BASE_DIR = Path(__file__).parent  
cols = ['id', 'timestamp', 'machine_id', 'cpu_usage', 'memory_usage']
df = pd.read_csv(BASE_DIR / "alibaba_cluster_data.csv", names=cols)

df = df.drop_duplicates()
print("Null values:\n", df.isnull().sum())

df = df.interpolate(method='linear').bfill().ffill()


def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df


cols_to_scale = ['cpu_usage', 'memory_usage']
df = remove_outliers(df, cols_to_scale)

scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
df.to_csv(BASE_DIR / "cleaned_data.csv", index=False)



data = df[cols_to_scale].reset_index(drop=True)
columns = cols_to_scale
cpu, ram = columns[0], columns[1]



print("\nSTATISTICS")
for col in columns:
    print(f"\nColumn: {col}")
    print(f"  Mean:   {data[col].mean():.4f}")
    print(f"  Std:    {data[col].std():.4f}")
    print(f"  Min:    {data[col].min():.4f}")
    print(f"  Max:    {data[col].max():.4f}")
    print(f"  Median: {data[col].median():.4f}")


print("\nCALCULATING TREND")
window = 50
for col in columns:
    data[col + "_trend"] = data[col].rolling(window).mean()
    print(f"Trend created for {col}")



fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(data[cpu], label=cpu)
axes[0].plot(data[ram], label=ram)
axes[0].set_title("CPU and RAM Usage")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(data[cpu], label="CPU original")
axes[1].plot(data[cpu + "_trend"], label="CPU trend")
axes[1].set_title("CPU Trend")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(data[ram], label="RAM original")
axes[2].plot(data[ram + "_trend"], label="RAM trend")
axes[2].set_title("RAM Trend")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(BASE_DIR / "analysis.png")
plt.show()


print("\nTREND ANALYSIS")
for col in columns:
    trend = data[col + "_trend"].dropna()
    start = trend.iloc[0]
    end = trend.iloc[-1]
    change = end - start
    direction = "Increasing" if change > 0 else "Decreasing" if change < 0 else "Flat"
    print(f"\nColumn: {col}")
    print(f"  Start:     {start:.4f}")
    print(f"  End:       {end:.4f}")
    print(f"  Change:    {change:.4f}")
    print(f"  Direction: {direction}")

