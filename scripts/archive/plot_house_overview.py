#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create one Plotly HTML overview per participant/house from a split CSV or predictions CSV."
    )
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument("--out-dir", required=True, help="Directory where per-house HTML plots will be written.")
    parser.add_argument(
        "--participants",
        default="",
        help="Optional comma-separated participant filter. If omitted, plots all participants in the CSV.",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional text to prepend to plot titles.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row limit for quick/debug runs. 0 means all rows.",
    )
    parser.add_argument(
        "--downsample-step",
        type=int,
        default=1,
        help="Keep every Nth row after participant filtering. 1 means no downsampling.",
    )
    return parser.parse_args()


def normalize_participants(text: str) -> Optional[List[str]]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    return values or None


def detect_participant_col(df: pd.DataFrame) -> str:
    for col in ("participant", "participant_id"):
        if col in df.columns:
            return col
    raise ValueError("Missing participant column. Expected one of: participant, participant_id")


def detect_mains_col(df: pd.DataFrame) -> str:
    for col in ("mains", "energy_mains"):
        if col in df.columns:
            return col
    raise ValueError("Missing mains column. Expected one of: mains, energy_mains")


def detect_prediction_mode(df: pd.DataFrame) -> bool:
    return any(col.endswith("_pred") for col in df.columns) or any(col.endswith("_true") for col in df.columns)


def detect_devices(df: pd.DataFrame, prediction_mode: bool, mains_col: str) -> List[str]:
    devices = set()
    if prediction_mode:
        for col in df.columns:
            if col.endswith("_true"):
                devices.add(col[: -len("_true")])
            elif col.endswith("_pred"):
                devices.add(col[: -len("_pred")])
    else:
        for col in df.columns:
            if col.startswith("energy_") and col != mains_col:
                devices.add(col)
    return sorted(devices)


def sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def build_figure(df: pd.DataFrame, participant: str, participant_col: str, mains_col: str, devices: List[str], title: str):
    prediction_mode = detect_prediction_mode(df)
    secondary_rows = prediction_mode
    specs = [[{"secondary_y": False}]]
    if secondary_rows:
        specs.extend([[{"secondary_y": True}] for _ in devices])
    else:
        specs.extend([[{"secondary_y": False}] for _ in devices])

    row_heights = [0.18] + [0.82 / max(len(devices), 1)] * len(devices)
    fig = make_subplots(
        rows=1 + len(devices),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=["mains"] + devices,
    )

    part_df = df[df[participant_col].astype(str) == str(participant)].copy()
    if part_df.empty:
        raise ValueError(f"No rows found for participant {participant}")

    fig.add_trace(
        go.Scattergl(
            x=part_df["timestamp"],
            y=part_df[mains_col],
            mode="lines",
            name="mains",
            line=dict(color="#1f77b4", width=1.4),
        ),
        row=1,
        col=1,
    )

    for idx, device in enumerate(devices, start=2):
        if prediction_mode:
            true_col = f"{device}_true"
            pred_col = f"{device}_pred"
            known_col = f"{device}_known"

            if true_col in part_df.columns:
                fig.add_trace(
                    go.Scattergl(
                        x=part_df["timestamp"],
                        y=part_df[true_col],
                        mode="lines",
                        name=f"{device} true",
                        line=dict(color="#2ca02c", width=1.2),
                    ),
                    row=idx,
                    col=1,
                )
            if pred_col in part_df.columns:
                fig.add_trace(
                    go.Scattergl(
                        x=part_df["timestamp"],
                        y=part_df[pred_col],
                        mode="lines",
                        name=f"{device} pred",
                        line=dict(color="#d62728", width=1.2),
                    ),
                    row=idx,
                    col=1,
                )
            if known_col in part_df.columns:
                fig.add_trace(
                    go.Scattergl(
                        x=part_df["timestamp"],
                        y=part_df[known_col],
                        mode="lines",
                        name=f"{device} known",
                        line=dict(color="#7f7f7f", width=1.0, dash="dot"),
                        opacity=0.6,
                    ),
                    row=idx,
                    col=1,
                    secondary_y=True,
                )
                fig.update_yaxes(range=[-0.05, 1.05], row=idx, col=1, secondary_y=True, title_text="known")
        else:
            fig.add_trace(
                go.Scattergl(
                    x=part_df["timestamp"],
                    y=part_df[device],
                    mode="lines",
                    name=device,
                    line=dict(width=1.2),
                ),
                row=idx,
                col=1,
            )

        fig.update_yaxes(title_text=device, row=idx, col=1)

    fig.update_yaxes(title_text=mains_col, row=1, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(900, 220 * (1 + len(devices))),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=90, b=50),
    )
    fig.update_xaxes(rangeslider_visible=(1 + len(devices) <= 4), row=1 + len(devices), col=1)
    return fig


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    nrows = None if args.max_rows <= 0 else int(args.max_rows)
    df = pd.read_csv(args.csv, nrows=nrows)
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must contain a timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    participant_col = detect_participant_col(df)
    mains_col = detect_mains_col(df)
    prediction_mode = detect_prediction_mode(df)
    devices = detect_devices(df, prediction_mode=prediction_mode, mains_col=mains_col)

    participants = normalize_participants(args.participants)
    if participants is None:
        participants = sorted(df[participant_col].astype(str).unique().tolist())

    written_files = []
    for participant in participants:
        title_prefix = f"{args.title_prefix} - " if args.title_prefix else ""
        title = f"{title_prefix}{participant} ({'predictions' if prediction_mode else 'data'})"
        part_df = df[df[participant_col].astype(str) == str(participant)].copy()
        if args.downsample_step > 1:
            part_df = part_df.iloc[:: int(args.downsample_step)].reset_index(drop=True)
        fig = build_figure(
            df=part_df,
            participant=participant,
            participant_col=participant_col,
            mains_col=mains_col,
            devices=devices,
            title=title,
        )
        out_path = os.path.join(args.out_dir, f"{sanitize_name(participant)}_overview.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        written_files.append((str(participant), os.path.basename(out_path)))
        print(f"Wrote {out_path}")

    index_path = os.path.join(args.out_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        f.write(f"<h1>{args.title_prefix or 'House overview plots'}</h1>\n")
        f.write("<ul>\n")
        for participant, filename in written_files:
            f.write(f'  <li><a href="{filename}">{participant}</a></li>\n')
        f.write("</ul>\n")
        f.write("</body></html>\n")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
