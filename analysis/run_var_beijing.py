import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = np.load(
    "beijing_panel_segments.npz",
    allow_pickle=True,
)

segments = list(data["X"])

num_variates = segments[0].shape[1]

results = []

for var_idx in range(num_variates):

    X_all = []
    y_all = []

    for seg in segments:

        if len(seg) <= 96:
            continue

        series = seg[:, var_idx]

        for t in range(96, len(series)):

            X_all.append(
                series[t - 96:t]
            )

            y_all.append(
                series[t]
            )

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    model = LinearRegression()

    model.fit(
        X_all,
        y_all,
    )

    pred = model.predict(X_all)

    r2 = r2_score(
        y_all,
        pred,
    )

    results.append(r2)

results = np.array(results)

print("\n=== R² SUMMARY ===")
print("Median:", np.median(results))
print("Mean:  ", np.mean(results))
print("Min:   ", np.min(results))
print("Max:   ", np.max(results))

print("\n=== DESCRIBE ===")
print(pd.Series(results).describe())
