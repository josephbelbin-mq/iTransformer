import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib

COLOURS = {
    "Linear":       "#d62728",
    "DLinear":      "#ff7f0e",
    "PatchTST":     "#2ca02c",
    "iTransformer": "#1f77b4",
}

def load_summary(csv_file, model_name):
    df = pd.read_csv(csv_file)

    df["model"] = model_name

    df["lambda"] = df["input_file"].str.extract(
        r"lambda([0-9]+(?:\.[0-9]+)?)"
    ).astype(float)

    df["gamma"] = df["input_file"].str.extract(
        r"gamma([0-9]+(?:\.[0-9]+)?)"
    ).astype(float)

    df["persistence"] = df["input_file"].str.extract(
        r"persistence([0-9]+(?:\.[0-9]+)?)"
    ).astype(float)

    return df

dfs = [
    load_summary("../sweep_summary_Linear.csv", "Linear"),
    load_summary("../../PatchTST/PatchTST_supervised/sweep_summary_DLinear.csv", "DLinear"),
    load_summary("../../PatchTST_official/PatchTST/PatchTST_supervised/sweep_summary_PatchTST_96seq.csv", "PatchTST"),
    #load_summary("../../PatchTST_official/PatchTST/PatchTST_supervised/sweep_summary_PatchTST.csv", "PatchTST"),
    load_summary("../sweep_summary_iTransformer.csv", "iTransformer"),
]
df = pd.concat(dfs, ignore_index=True)
df = df[df["persistence"] == 1]

error_cube = df.pivot_table(
    index=["lambda", "gamma"],
    columns="model",
    values="mse",
    aggfunc="mean",
)

winner = error_cube.idxmin(axis=1)

best = error_cube.min(axis=1)

second = (
    error_cube.apply(
        lambda row: row.nsmallest(2).iloc[-1],
        axis=1,
    )
)
print("Error cube")
print(error_cube)

margin = (second - best) / second

result = pd.DataFrame({
    "winner": winner,
    "margin": margin,
}).reset_index()

print(
    winner
    .value_counts()
    .sort_values(ascending=False)
)

print(result.groupby("winner")["margin"].agg(
    ["count", "mean", "median", "max"]
))

winner_df = result.pivot(
    index="gamma",
    columns="lambda",
    values="winner"
)

margin_df = result.pivot(
    index="gamma",
    columns="lambda",
    values="margin"
)

ranks = error_cube.rank(axis=1, method="average")

print(
    ranks.mean()
    .sort_values()
)
ranks = error_cube.rank(axis=1, method="average")
print(ranks)
rgb = np.zeros((*winner_df.shape, 3))

for model, colour in COLOURS.items():
    mask = winner_df.values == model
    rgb[mask] = to_rgb(colour)

# Scale brightness by win margin
m = margin_df.values

# log scaling helps when margins vary wildly
m = np.log1p(m * 100)

if m.max() > 0:
    m = m / m.max()

# keep some colour even for tiny wins
rgb_base = rgb.copy()

strength = m[..., None]
strength = np.clip(strength, 0.1, 1.0)

rgb = (
    (1 - strength) * 1.0      # white
    + strength * rgb_base     # model colour
)
fig, ax = plt.subplots(figsize=(8, 6))

ax.imshow(
    rgb,
    origin="lower",
    aspect="auto",
)

ax.set_xticks(range(len(winner_df.columns)))
ax.set_xticklabels(
    [f"{x:.2f}" for x in winner_df.columns],
    rotation=45
)

ax.set_yticks(range(len(winner_df.index)))
ax.set_yticklabels(
    [f"{y:.2f}" for y in winner_df.index]
)

ax.set_xlabel("Lambda")
ax.set_ylabel("Gamma")
ax.set_title("Winning Model Across Parameter Space")

# legend
handles = [
    plt.Line2D(
        [0], [0],
        marker="s",
        markersize=12,
        linestyle="",
        color=colour,
        label=model,
    )
    for model, colour in COLOURS.items()
]

ax.legend(
    handles=handles,
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig("analysis_heatmap.png", dpi=300, bbox_inches="tight")

df = pd.concat(dfs, ignore_index=True)
df = df[(df["lambda"] == 1.0) & (df["gamma"] == 1.0)]

error_cube = df.pivot_table(
    index=["lambda", "gamma", "persistence"],
    columns="model",
    values="mse",
    aggfunc="mean",
)


print("Error cube")
print(error_cube)
