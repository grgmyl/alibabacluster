import pandas as pd
import matplotlib.pyplot as plt

#  STATISTICS
print("STATISTICS")
for col in columns:
    print("\nColumn:", col)
    print("Mean:", data[col].mean())
    print("Std:", data[col].std())
    print("Min:", data[col].min())
    print("Max:", data[col].max())
    print("Median:", data[col].median())


# MOVING AVERAGE TREND
print("CALCULATING TREND")

window = 50

for col in columns:
    data[col + "_trend"] = data[col].rolling(window).mean()
    print("Trend created for", col)


# PLOTS

print("\nCreating plots...")

cpu = columns[0]
ram = columns[1]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Original data
axes[0].plot(data[cpu], label=cpu)
axes[0].plot(data[ram], label=ram)
axes[0].set_title("CPU and RAM Usage")
axes[0].legend()
axes[0].grid(True)

# Plot 2: CPU trend
axes[1].plot(data[cpu], label="CPU original")
axes[1].plot(data[cpu + "_trend"], label="CPU trend")
axes[1].set_title("CPU Trend")
axes[1].legend()
axes[1].grid(True)

# Plot 3: RAM trend
axes[2].plot(data[ram], label="RAM original")
axes[2].plot(data[ram + "_trend"], label="RAM trend")
axes[2].set_title("RAM Trend")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("analysis.png")
plt.show()


# TREND ANALYSIS

print("TREND ANALYSIS")


for col in columns:

    trend = data[col + "_trend"].dropna()

    start = trend.iloc[0]
    end = trend.iloc[-1]

    change = end - start

    print("\nColumn:", col)
    print("Start:", start)
    print("End:", end)
    print("Change:", change)

    if change > 0:
        print("Direction: Increasing")
    elif change < 0:
        print("Direction: Decreasing")
    else:
        print("Direction: Flat")

print("\nDONE")