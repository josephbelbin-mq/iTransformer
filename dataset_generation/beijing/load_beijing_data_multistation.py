from pathlib import Path
import re

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]

MIN_SEGMENT_LENGTH = 500
FILL_LIMIT = 24   # only fill 1 day


def load_raw_beijing(ts_file):

    samples = []
    starts = []
    ends = []

    with open(ts_file, "r", encoding="utf-8") as f:

        data_started = False

        for line in f:

            line = line.strip()

            if not data_started:
                if line.lower() == "@data":
                    data_started = True
                continue

            if not line:
                continue

            matches = re.findall(
                r"\(([^,]+),([^)]+)\)",
                line,
            )

            timestamps = [
                pd.Timestamp(t)
                for t, _ in matches
            ]

            values = [
                np.nan if v.strip() == "?"
                else float(v)
                for _, v in matches
            ]

            starts.append(min(timestamps))
            ends.append(max(timestamps))

            series = (
                np.array(values)
                .reshape(9, 24)
                .T
            )

            samples.append(series)

    X = np.array(samples)

    meta = pd.DataFrame(
        {
            "start": starts,
            "end": ends,
        }
    )

    meta["next_start"] = meta["start"].shift(-1)

    meta["gap_hours"] = (
        meta["next_start"] - meta["end"]
    ).dt.total_seconds() / 3600

    return X, meta


def split_into_stations(X, meta):

    station_boundaries = (
        meta.index[
            meta["gap_hours"] < 0
        ].tolist()
    )

    stations = []

    start_idx = 0

    for boundary in station_boundaries:

        stations.append(
            (
                X[start_idx:boundary + 1],
                meta.iloc[start_idx:boundary + 1].copy(),
            )
        )

        start_idx = boundary + 1

    stations.append(
        (
            X[start_idx:],
            meta.iloc[start_idx:].copy(),
        )
    )

    print(
        f"Detected {len(stations)} stations"
    )

    return stations


def build_station_frames(stations):

    station_frames = []

    for station_id, (station_X, station_meta) in enumerate(stations):

        station_meta = (
            station_meta
            .reset_index(drop=True)
        )

        timestamps = []
        rows = []

        for day_idx in range(len(station_X)):

            day_start = station_meta.iloc[day_idx]["start"]

            day_times = pd.date_range(
                start=day_start,
                periods=24,
                freq="h",
            )

            timestamps.extend(day_times)

            rows.append(
                station_X[day_idx]
            )

        station_matrix = np.vstack(rows)

        station_df = pd.DataFrame(
            station_matrix,
            columns=[
                f"S{station_id+1}_{f}"
                for f in FEATURE_NAMES
            ],
        )

        station_df.insert(
            0,
            "date",
            timestamps,
        )

        print(
            f"Station {station_id+1}: "
            f"{len(station_df):,} rows"
        )

        station_frames.append(station_df)

    return station_frames


def build_panel(station_frames):

    panel = station_frames[0]

    for station_df in station_frames[1:]:

        panel = panel.merge(
            station_df,
            on="date",
            how="outer",
        )

    panel = (
        panel
        .sort_values("date")
        .reset_index(drop=True)
    )

    print()
    print("Panel shape before fill:")
    print(panel.shape)

    print(
        "Missing values:",
        panel.isna().sum().sum()
    )

    #
    # Only fill short outages
    #
    panel = panel.ffill(limit=FILL_LIMIT)
    panel = panel.bfill(limit=FILL_LIMIT)

    print()
    print("After limited fill:")
    print(
        panel.isna().sum().sum(),
        "NaNs remaining"
    )

    return panel


def extract_segments(panel):

    feature_cols = panel.columns.drop("date")

    complete_rows = (
        ~panel[feature_cols]
        .isna()
        .any(axis=1)
    )

    segments = []

    start = None

    for idx, valid in enumerate(complete_rows):

        if valid and start is None:
            start = idx

        elif not valid and start is not None:

            seg = panel.iloc[start:idx]

            if len(seg) >= MIN_SEGMENT_LENGTH:

                segments.append(
                    seg[feature_cols]
                    .to_numpy(dtype=np.float32)
                )

            start = None

    if start is not None:

        seg = panel.iloc[start:]

        if len(seg) >= MIN_SEGMENT_LENGTH:

            segments.append(
                seg[feature_cols]
                .to_numpy(dtype=np.float32)
            )

    return segments


def main():

    file_path = Path(
        "~/BeijingPM25Quality_TRAIN.ts"
    ).expanduser()

    X, meta = load_raw_beijing(
        file_path
    )

    stations = split_into_stations(
        X,
        meta,
    )

    station_frames = build_station_frames(
        stations
    )

    panel = build_panel(
        station_frames
    )

    segments = extract_segments(
        panel
    )

    lengths = [
        len(seg)
        for seg in segments
    ]

    print()
    print("Segments kept:", len(segments))
    print(
        pd.Series(lengths).describe()
    )

    np.savez_compressed(
        "beijing_panel_segments.npz",
        X=np.array(
            segments,
            dtype=object,
        ),
    )

    print()
    print(
        "Saved: beijing_panel_segments.npz"
    )


if __name__ == "__main__":
    main()