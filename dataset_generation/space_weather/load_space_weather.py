from datasets import load_dataset
import pandas as pd

df = load_dataset("juliensimon/space-weather-indices")["train"].to_pandas()

print(df.columns.tolist())
print(df.shape)
print(df.dtypes)

df = df[df["data_type"] == "observed"].copy()
df = df.drop(
columns=[
"f107_data_type",
"data_type",
"is_storm",
"storm_level",
]
)

df = df.drop(
columns= [
"kp_sum",
"ap_avg",
"f107_adj",
"f107_obs_center81",
"f107_obs_last81",
"f107_adj_center81",
"f107_adj_last81",
])

df = df.sort_values("date")
df.to_csv("space_weather_clean.csv", index=False)
