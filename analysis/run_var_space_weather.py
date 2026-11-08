import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("space_weather_clean.csv")

# remove date column
X = df.iloc[:, 1:].values

num_variates = X.shape[1]

results = []

for var_idx in range(num_variates):

    X_all = []
    y_all = []

    series = X[:, var_idx]

    for t in range(96, len(series)):

        X_all.append(
            series[t-96:t]
        )

        y_all.append(
            series[t]
        )

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    model = LinearRegression()

    model.fit(X_all, y_all)

    pred = model.predict(X_all)

    results.append(
        r2_score(y_all, pred)
    )

results = np.array(results)

print("\n=== R² SUMMARY ===")
print("Median:", np.median(results))
print("Mean:  ", np.mean(results))
print("Min:   ", np.min(results))
print("Max:   ", np.max(results))

print("\n=== DESCRIBE ===")
print(pd.Series(results).describe())
