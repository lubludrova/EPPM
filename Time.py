import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Dataset": ["HelpDesk", "BPI_Challenge_2012_A", "BPI_Challenge_2012_O", "Env-Permit"],
    "GNNExplainer": [127, 85.72, 76.42, 231.47],
    "PGExplainer": [62, 82.11, 71.2, 224.85],
    "Att-top-k": [57, 64.62, 40.98, 70.62],
    "SubGraphX": [5777, 3010.83, 2299.85, 9551.75],
    "TAPGExplainer": [229, 109.72, 123.83, 455.86]
}

df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
for method in df.columns[1:]:
    # if method != 'SubGraphX':
    plt.plot(df["Dataset"], df[method], marker='o', label=method)

plt.title("The time for each explainability method to train")
plt.xlabel("Dataset")
plt.ylabel("Training Time (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("Time.png")