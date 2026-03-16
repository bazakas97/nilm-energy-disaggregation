import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

##############################################################################
# Dictionary of device thresholds
##############################################################################
device_thresholds = {
    "energy_dish_washer": {"min_duration": 3, "min_energy_value": 100},
    "energy_dryer": {"min_duration": 3, "min_energy_value": 100},
    "energy_fridge_freezer": {"min_duration": 0, "min_energy_value": 50},
    "energy_pv": {"min_duration": 3, "min_energy_value": 100},
    "energy_washing_machine": {"min_duration": 3, "min_energy_value": 100},
    "energy_oven": {"min_duration": 3, "min_energy_value": 100},
    "energy_ac": {"min_duration": 3, "min_energy_value": 300},
    "energy_ev": {"min_duration": 3, "min_energy_value": 500},
    "energy_induction_hob": {"min_duration": 3, "min_energy_value": 100},
    "energy_ewh": {"min_duration": 3, "min_energy_value": 100},
}


def normalize_participant_filter(participant_filter):
    if participant_filter is None:
        return None
    if isinstance(participant_filter, str):
        return [participant_filter]
    if isinstance(participant_filter, (list, tuple)):
        values = [str(v) for v in participant_filter if str(v).strip()]
        return values or None
    return None


def resolve_energy_thresholds(device_list, thresholds_cfg=None):
    thresholds_cfg = thresholds_cfg or {}
    resolved = {}
    for dev in device_list:
        val = thresholds_cfg.get(dev)
        if isinstance(val, dict):
            resolved[dev] = float(val.get("min_energy_value", 100.0))
        elif val is not None:
            resolved[dev] = float(val)
        elif dev in device_thresholds:
            resolved[dev] = float(device_thresholds[dev]["min_energy_value"])
        else:
            resolved[dev] = 100.0
    return resolved


def limit_rows(df, max_rows=None, strategy="head"):
    if max_rows is None:
        return df
    max_rows = int(max_rows)
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    mode = str(strategy or "head").strip().lower()
    if mode == "head":
        return df.iloc[:max_rows].reset_index(drop=True)

    if mode == "random":
        return df.sample(n=max_rows, random_state=42).sort_index().reset_index(drop=True)

    if mode in {"balanced_participants", "balanced"} and "participant" in df.columns:
        parts = list(df["participant"].astype(str).unique())
        grouped = {
            p: df[df["participant"].astype(str) == p]
            for p in parts
        }
        allocations = {p: 0 for p in parts}
        remaining = max_rows

        # Round-robin allocation ensures each participant gets represented.
        while remaining > 0:
            progressed = False
            for p in parts:
                if allocations[p] < len(grouped[p]) and remaining > 0:
                    allocations[p] += 1
                    remaining -= 1
                    progressed = True
                if remaining == 0:
                    break
            if not progressed:
                break

        selected = []
        for p in parts:
            take = allocations[p]
            if take > 0:
                selected.append(grouped[p].iloc[:take])

        if not selected:
            return df.iloc[:max_rows].reset_index(drop=True)
        return pd.concat(selected).sort_index().reset_index(drop=True)

    return df.iloc[:max_rows].reset_index(drop=True)


def filter_participants_by_data_quality(df, device_list, threshold_map, filter_cfg=None):
    cfg = filter_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return df, {}
    if "participant" not in df.columns:
        return df, {"skipped": "participant column missing"}

    min_rows = max(0, int(cfg.get("min_rows", 0)))
    min_known_points = max(0, int(cfg.get("min_known_points", 0)))
    min_active_points = max(0, int(cfg.get("min_active_points", 0)))
    min_devices_with_activity = max(0, int(cfg.get("min_devices_with_activity", 0)))
    min_active_points_per_device = max(1, int(cfg.get("min_active_points_per_device", 1)))
    min_mains_nonzero_ratio = max(0.0, float(cfg.get("min_mains_nonzero_ratio", 0.0)))

    if (
        min_rows == 0
        and min_known_points == 0
        and min_active_points == 0
        and min_devices_with_activity == 0
        and min_mains_nonzero_ratio <= 0.0
    ):
        return df, {}

    target_df = df[device_list].apply(pd.to_numeric, errors="coerce")
    known_mask = (~target_df.isna()).to_numpy(np.float32)
    targets = target_df.fillna(0.0).to_numpy(np.float32)

    threshold_vec = np.array([float(threshold_map[d]) for d in device_list], dtype=np.float32)
    active_mask = ((targets > threshold_vec.reshape(1, -1)) & (known_mask > 0.5)).astype(np.float32)

    participants = df["participant"].astype(str).to_numpy()
    mains = pd.to_numeric(df["energy_mains"], errors="coerce").fillna(0.0).to_numpy(np.float32) if "energy_mains" in df.columns else None
    unique_participants = sorted(set(participants.tolist()))

    keep_parts = []
    removed = []
    for part in unique_participants:
        part_rows = participants == part
        rows_n = int(part_rows.sum())
        known_points = int(known_mask[part_rows].sum())
        active_points = int(active_mask[part_rows].sum())
        active_by_device = active_mask[part_rows].sum(axis=0)
        devices_with_activity = int((active_by_device >= float(min_active_points_per_device)).sum())
        if mains is not None:
            mains_nonzero_ratio = float((mains[part_rows] > 1e-6).mean())
        else:
            mains_nonzero_ratio = 1.0

        failed = []
        if min_rows > 0 and rows_n < min_rows:
            failed.append(f"rows<{min_rows}")
        if min_known_points > 0 and known_points < min_known_points:
            failed.append(f"known<{min_known_points}")
        if min_active_points > 0 and active_points < min_active_points:
            failed.append(f"active<{min_active_points}")
        if min_devices_with_activity > 0 and devices_with_activity < min_devices_with_activity:
            failed.append(f"active_devices<{min_devices_with_activity}")
        if min_mains_nonzero_ratio > 0.0 and mains_nonzero_ratio < min_mains_nonzero_ratio:
            failed.append(f"mains_nonzero_ratio<{min_mains_nonzero_ratio}")

        if failed:
            removed.append(
                {
                    "participant": part,
                    "rows": rows_n,
                    "known_points": known_points,
                    "active_points": active_points,
                    "active_devices": devices_with_activity,
                    "mains_nonzero_ratio": mains_nonzero_ratio,
                    "failed": failed,
                }
            )
        else:
            keep_parts.append(part)

    keep_mask = np.isin(participants, np.asarray(keep_parts, dtype=object))
    filtered = df.loc[keep_mask].reset_index(drop=True)

    stats = {
        "criteria": {
            "min_rows": min_rows,
            "min_known_points": min_known_points,
            "min_active_points": min_active_points,
            "min_devices_with_activity": min_devices_with_activity,
            "min_active_points_per_device": min_active_points_per_device,
            "min_mains_nonzero_ratio": min_mains_nonzero_ratio,
        },
        "kept_participants": keep_parts,
        "removed_participants": removed,
        "rows_before": int(len(df)),
        "rows_after": int(len(filtered)),
    }
    return filtered, stats


def _contiguous_participant_segments(participants, n_rows):
    if participants is None or len(participants) != n_rows:
        return [(0, n_rows)]
    if n_rows == 0:
        return []

    segments = []
    start = 0
    prev = str(participants[0])
    for i in range(1, n_rows):
        cur = str(participants[i])
        if cur != prev:
            segments.append((start, i))
            start = i
            prev = cur
    segments.append((start, n_rows))
    return segments


def _fill_short_off_gaps_1d(signal, known_mask, mains, cfg):
    n = len(signal)
    if n < 3:
        return signal, 0, 0

    on_power_min = float(cfg.get("on_power_min", 1000.0))
    off_power_max = float(cfg.get("off_power_max", 0.0))
    max_gap_points = max(0, int(cfg.get("max_gap_points", 1)))
    if max_gap_points <= 0:
        return signal, 0, 0

    fill_method = str(cfg.get("fill_method", "linear")).strip().lower()
    require_mains_support = bool(cfg.get("require_mains_support", False))
    mains_min = float(cfg.get("mains_min", max(0.0, 0.6 * on_power_min)))
    mains_edge_tolerance = float(cfg.get("mains_edge_tolerance", 800.0))

    out = signal.copy()
    repaired_points = 0
    repaired_gaps = 0

    i = 1
    while i < (n - 1):
        if (
            known_mask[i - 1] > 0.5
            and known_mask[i] > 0.5
            and out[i - 1] >= on_power_min
            and out[i] <= off_power_max
        ):
            j = i
            while j < n and known_mask[j] > 0.5 and out[j] <= off_power_max:
                j += 1
            gap_len = j - i

            if (
                gap_len <= max_gap_points
                and j < n
                and known_mask[j] > 0.5
                and out[j] >= on_power_min
            ):
                if require_mains_support:
                    gap_mains = mains[i:j]
                    edge_mean = 0.5 * (float(mains[i - 1]) + float(mains[j]))
                    gap_min = float(np.min(gap_mains)) if gap_mains.size else float("inf")
                    if gap_min < mains_min or gap_min < (edge_mean - mains_edge_tolerance):
                        i = j
                        continue

                left_val = float(out[i - 1])
                right_val = float(out[j])

                if fill_method in {"previous", "left"}:
                    fill_vals = np.full(gap_len, left_val, dtype=np.float32)
                elif fill_method in {"min", "min_edge"}:
                    fill_vals = np.full(gap_len, min(left_val, right_val), dtype=np.float32)
                else:
                    fill_vals = np.linspace(left_val, right_val, num=gap_len + 2, dtype=np.float32)[1:-1]

                out[i:j] = fill_vals
                repaired_points += gap_len
                repaired_gaps += 1

            i = j
        else:
            i += 1

    return out, repaired_points, repaired_gaps


def apply_label_gap_fill(mains, targets, label_mask, device_list, participants, gap_fill_cfg=None):
    cfg = gap_fill_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return targets, {}

    device_cfgs = cfg.get("devices", {})
    if not isinstance(device_cfgs, dict) or not device_cfgs:
        return targets, {}

    repaired = targets.copy()
    n_rows = repaired.shape[0]
    segments = _contiguous_participant_segments(participants, n_rows)
    stats = {}
    total_points = 0
    total_gaps = 0

    device_to_idx = {d: i for i, d in enumerate(device_list)}
    for dev, dev_cfg in device_cfgs.items():
        if dev not in device_to_idx:
            continue
        if not bool((dev_cfg or {}).get("enabled", True)):
            continue

        idx = device_to_idx[dev]
        dev_points = 0
        dev_gaps = 0
        for start, end in segments:
            seg_signal = repaired[start:end, idx]
            seg_mask = label_mask[start:end, idx]
            seg_mains = mains[start:end]
            seg_repaired, seg_points, seg_gaps = _fill_short_off_gaps_1d(
                signal=seg_signal,
                known_mask=seg_mask,
                mains=seg_mains,
                cfg=dev_cfg or {},
            )
            if seg_points > 0:
                repaired[start:end, idx] = seg_repaired
                dev_points += int(seg_points)
                dev_gaps += int(seg_gaps)

        if dev_points > 0:
            stats[dev] = {"repaired_points": dev_points, "repaired_gaps": dev_gaps}
            total_points += dev_points
            total_gaps += dev_gaps

    if total_points > 0:
        stats["_total"] = {"repaired_points": total_points, "repaired_gaps": total_gaps}
    return repaired, stats


def apply_unattributed_mains_mask(
    mains,
    targets,
    label_mask,
    participants=None,
    preprocessing_cfg=None,
):
    row_mask, stats = compute_unattributed_mains_row_mask(
        mains=mains,
        targets=targets,
        label_mask=label_mask,
        participants=participants,
        preprocessing_cfg=preprocessing_cfg,
    )

    masked = np.asarray(label_mask, dtype=np.float32).copy()
    if not row_mask.any():
        return masked, {}

    masked[row_mask, :] = 0.0
    return masked, stats


def compute_unattributed_mains_row_mask(
    mains,
    targets,
    label_mask,
    participants=None,
    preprocessing_cfg=None,
):
    cfg = (preprocessing_cfg or {}).get("unattributed_mains_mask", {}) or {}
    if not bool(cfg.get("enabled", False)):
        row_mask = np.zeros(len(np.asarray(mains)), dtype=bool)
        return row_mask, {}

    mains_min = float(cfg.get("mains_min", 1000.0))
    target_sum_max = float(cfg.get("target_sum_max", 0.0))

    masked = np.asarray(label_mask, dtype=np.float32)
    mains_arr = np.asarray(mains, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)

    # Candidate ambiguous points:
    # high mains consumption while all selected device targets are near zero.
    high_mains = mains_arr >= mains_min
    near_zero_targets = targets_arr.sum(axis=1) <= target_sum_max
    has_any_label = masked.sum(axis=1) > 0
    row_mask = high_mains & near_zero_targets & has_any_label

    if not row_mask.any():
        return row_mask, {}

    stats = {
        "masked_rows": int(row_mask.sum()),
        "mains_min": float(mains_min),
        "target_sum_max": float(target_sum_max),
    }

    if participants is not None and len(participants) == len(row_mask):
        part_arr = np.asarray(participants, dtype=object).astype(str)
        unique_parts, counts = np.unique(part_arr[row_mask], return_counts=True)
        stats["by_participant"] = {
            str(p): int(c)
            for p, c in zip(unique_parts.tolist(), counts.tolist())
        }

    return row_mask, stats


class MaskedStandardScaler:
    """Column-wise scaler that ignores masked values during fitting."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, values, mask=None):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("values must be 2D")
        if mask is None:
            mask = np.ones_like(values, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape != values.shape:
            raise ValueError("mask shape must match values shape")

        n_features = values.shape[1]
        means = np.zeros(n_features, dtype=np.float32)
        scales = np.ones(n_features, dtype=np.float32)

        for i in range(n_features):
            valid = mask[:, i] > 0
            if valid.any():
                col = values[valid, i]
                mean = float(np.mean(col))
                std = float(np.std(col))
                means[i] = mean
                scales[i] = std if std > 1e-6 else 1.0

        self.mean_ = means
        self.scale_ = scales
        self.n_features_in_ = n_features
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler is not fitted")
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler is not fitted")
        return values * self.scale_ + self.mean_


def apply_preprocessing(
    mains,
    targets,
    device_list,
    preprocessing_cfg=None,
    label_mask=None,
    participants=None,
):
    cfg = preprocessing_cfg or {}
    mains = mains.copy()
    targets = targets.copy()

    if cfg.get("correct_mains_vs_devices", False):
        sum_devices = targets.sum(axis=1)
        overflow = (sum_devices > mains) & (mains > 0)
        if overflow.any():
            scale = mains[overflow] / sum_devices[overflow]
            targets[overflow] *= scale[:, None]

    sigma = float(cfg.get("gaussian_sigma", 0))
    if sigma > 0:
        exclude = set(cfg.get("gaussian_exclude", []))
        for i, dev in enumerate(device_list):
            if dev in exclude:
                continue
            targets[:, i] = gaussian_filter1d(targets[:, i], sigma=sigma)

    gap_fill_cfg = cfg.get("label_gap_fill", {})
    if label_mask is not None and bool(gap_fill_cfg.get("enabled", False)):
        targets, gap_stats = apply_label_gap_fill(
            mains=mains,
            targets=targets,
            label_mask=label_mask,
            device_list=device_list,
            participants=participants,
            gap_fill_cfg=gap_fill_cfg,
        )
        if bool(gap_stats) and bool(gap_fill_cfg.get("log", True)):
            total = gap_stats.get("_total", {})
            print(
                "[Preprocess] Label gap fill repaired "
                f"{total.get('repaired_points', 0)} points in {total.get('repaired_gaps', 0)} gaps."
            )
            for dev, s in gap_stats.items():
                if dev == "_total":
                    continue
                print(
                    f"  - {dev}: repaired_points={s['repaired_points']}, "
                    f"repaired_gaps={s['repaired_gaps']}"
                )

    # IMPORTANT: this causes label leakage for NILM, so keep it disabled by default.
    if cfg.get("align_mains_with_devices", False):
        mains = targets.sum(axis=1)

    return mains, targets


class NILMDataset(Dataset):
    def __init__(
        self,
        data_path,
        window_size,
        device_list,
        device_thresholds=None,
        input_scaler=None,
        output_scaler=None,
        stride=1,
        is_training=True,
        participant_filter=None,
        drop_unlabeled_centers=True,
        preprocessing_cfg=None,
        timestamp_col=None,
        start_timestamp=None,
        end_timestamp=None,
        max_rows=None,
        max_rows_strategy="head",
        augmentation_cfg=None,
        target_mode="point",
    ):
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.device_list = list(device_list)
        self.is_training = bool(is_training)
        self.drop_unlabeled_centers = bool(drop_unlabeled_centers)
        self.target_mode = str(target_mode).lower()
        if self.target_mode not in {"point", "sequence"}:
            raise ValueError("target_mode must be either 'point' or 'sequence'")
        augmentation_cfg = augmentation_cfg or {}

        read_nrows = None
        if max_rows is not None and str(max_rows_strategy).lower() in {"head"}:
            read_nrows = int(max_rows)
        self.data = pd.read_csv(data_path, nrows=read_nrows)
        participants = normalize_participant_filter(participant_filter)
        if participants is not None:
            if "participant" not in self.data.columns:
                raise ValueError("participant_filter was set but 'participant' column is missing")
            self.data = self.data[self.data["participant"].isin(participants)].reset_index(drop=True)
            if self.data.empty:
                raise ValueError("No rows after participant filtering")

        if timestamp_col:
            if timestamp_col not in self.data.columns:
                raise ValueError(
                    f"timestamp_col='{timestamp_col}' was provided but column is missing from {data_path}"
                )
            self.timestamp_col = timestamp_col
        elif "datetime" in self.data.columns:
            self.timestamp_col = "datetime"
        elif "timestamp" in self.data.columns:
            self.timestamp_col = "timestamp"
        else:
            self.timestamp_col = None

        start_timestamp = str(start_timestamp).strip() if start_timestamp is not None else None
        end_timestamp = str(end_timestamp).strip() if end_timestamp is not None else None
        start_timestamp = start_timestamp if start_timestamp else None
        end_timestamp = end_timestamp if end_timestamp else None

        if start_timestamp is not None or end_timestamp is not None:
            if self.timestamp_col is None:
                raise ValueError(
                    "Date filtering was requested (start/end timestamp) "
                    "but no timestamp column was found."
                )

            ts = pd.to_datetime(self.data[self.timestamp_col], errors="coerce")
            keep_mask = pd.Series(True, index=self.data.index)

            if start_timestamp is not None:
                start_dt = pd.to_datetime(start_timestamp)
                keep_mask &= ts >= start_dt
            if end_timestamp is not None:
                end_dt = pd.to_datetime(end_timestamp)
                keep_mask &= ts <= end_dt

            before_rows = len(self.data)
            self.data = self.data[keep_mask.fillna(False)].reset_index(drop=True)
            kept_rows = len(self.data)

            print(
                "[Preprocess] Timestamp filter:",
                {
                    "column": self.timestamp_col,
                    "start": start_timestamp,
                    "end": end_timestamp,
                    "kept_rows": kept_rows,
                    "dropped_rows": int(before_rows - kept_rows),
                },
            )
            if self.data.empty:
                raise ValueError("No rows after timestamp filtering")

        self.data = limit_rows(
            self.data,
            max_rows=max_rows,
            strategy=max_rows_strategy,
        )

        if "energy_mains" not in self.data.columns:
            raise ValueError("Dataset must contain 'energy_mains' column")

        for dev in self.device_list:
            if dev not in self.data.columns:
                self.data[dev] = np.nan

        threshold_map_local = resolve_energy_thresholds(self.device_list, device_thresholds)
        participant_filter_cfg = (preprocessing_cfg or {}).get("participant_data_filter", {}) or {}
        if bool(participant_filter_cfg.get("enabled", False)):
            self.data, participant_filter_stats = filter_participants_by_data_quality(
                df=self.data,
                device_list=self.device_list,
                threshold_map=threshold_map_local,
                filter_cfg=participant_filter_cfg,
            )
            if self.data.empty:
                raise ValueError("No rows after participant_data_filter")
            if bool(participant_filter_stats) and bool(participant_filter_cfg.get("log", True)):
                removed = participant_filter_stats.get("removed_participants", []) or []
                kept = participant_filter_stats.get("kept_participants", []) or []
                print(
                    "[Preprocess] Participant data filter:",
                    {
                        "rows_before": participant_filter_stats.get("rows_before"),
                        "rows_after": participant_filter_stats.get("rows_after"),
                        "kept_count": len(kept),
                        "removed_count": len(removed),
                        "criteria": participant_filter_stats.get("criteria"),
                    },
                )
                if kept:
                    print("  - kept participants:", kept)
                for item in removed:
                    print(
                        f"  - removed {item['participant']}: "
                        f"rows={item['rows']}, known={item['known_points']}, "
                        f"active={item['active_points']}, active_devices={item['active_devices']}, "
                        f"mains_nonzero_ratio={item.get('mains_nonzero_ratio')}, "
                        f"failed={item['failed']}"
                    )

        if self.timestamp_col is not None:
            self.timestamps = self.data[self.timestamp_col].astype(str).to_numpy()
        else:
            self.timestamps = np.arange(len(self.data)).astype(str)
        if "participant" in self.data.columns:
            self.participants = self.data["participant"].astype(str).to_numpy()
        else:
            self.participants = np.array(["__unknown__"] * len(self.data), dtype=object)

        mains = pd.to_numeric(self.data["energy_mains"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        mains = np.clip(mains, 0.0, None)

        mains_nonzero_ratio = float((mains > 1e-6).mean()) if len(mains) else 0.0
        if mains_nonzero_ratio <= 1e-6:
            raise ValueError(
                "energy_mains is all zeros after filtering. "
                "Choose different participants/split or fix the source data."
            )

        zero_mains_participants = []
        if "participant" in self.data.columns:
            for part in np.unique(self.participants):
                part_mask = self.participants == part
                part_mains = mains[part_mask]
                if part_mains.size == 0:
                    continue
                part_nonzero_ratio = float((part_mains > 1e-6).mean())
                if part_nonzero_ratio <= 1e-6:
                    zero_mains_participants.append((str(part), int(part_mains.size)))
        if zero_mains_participants:
            preview = ", ".join(f"{p} (rows={n})" for p, n in zero_mains_participants[:10])
            if len(zero_mains_participants) > 10:
                preview += ", ..."
            print(
                "[WARN] Participants with all-zero mains detected: "
                f"{preview}. Consider excluding them via participants_train/val/test."
            )

        target_df = self.data[self.device_list].apply(pd.to_numeric, errors="coerce")
        label_mask = (~target_df.isna()).to_numpy(np.float32)
        raw_targets = target_df.fillna(0.0).to_numpy(np.float32)
        raw_targets = np.clip(raw_targets, 0.0, None)

        mains, raw_targets = apply_preprocessing(
            mains=mains,
            targets=raw_targets,
            device_list=self.device_list,
            preprocessing_cfg=preprocessing_cfg,
            label_mask=label_mask,
            participants=self.participants,
        )

        unattributed_cfg = (preprocessing_cfg or {}).get("unattributed_mains_mask", {}) or {}
        if bool(unattributed_cfg.get("drop_rows", False)):
            unattributed_row_mask, unattributed_row_stats = compute_unattributed_mains_row_mask(
                mains=mains,
                targets=raw_targets,
                label_mask=label_mask,
                participants=self.participants,
                preprocessing_cfg=preprocessing_cfg,
            )
            if unattributed_row_mask.any():
                keep = ~unattributed_row_mask
                self.data = self.data.loc[keep].reset_index(drop=True)
                self.timestamps = self.timestamps[keep]
                self.participants = self.participants[keep]
                mains = mains[keep]
                raw_targets = raw_targets[keep]
                label_mask = label_mask[keep]

                if len(self.data) == 0:
                    raise ValueError("No rows left after unattributed mains row-drop filtering")

                if bool(unattributed_cfg.get("log", True)):
                    print(
                        "[Preprocess] Unattributed mains row-drop removed "
                        f"{unattributed_row_stats.get('masked_rows', 0)} rows "
                        f"(mains_min={unattributed_row_stats.get('mains_min')}, "
                        f"target_sum_max={unattributed_row_stats.get('target_sum_max')})."
                    )
                    for part, cnt in (unattributed_row_stats.get("by_participant", {}) or {}).items():
                        print(f"  - {part}: dropped_rows={cnt}")

        label_mask, unattributed_stats = apply_unattributed_mains_mask(
            mains=mains,
            targets=raw_targets,
            label_mask=label_mask,
            participants=self.participants,
            preprocessing_cfg=preprocessing_cfg,
        )
        if bool(unattributed_stats) and bool(unattributed_cfg.get("log", True)):
            print(
                "[Preprocess] Unattributed mains mask removed labels from "
                f"{unattributed_stats.get('masked_rows', 0)} rows "
                f"(mains_min={unattributed_stats.get('mains_min')}, "
                f"target_sum_max={unattributed_stats.get('target_sum_max')})."
            )
            for part, cnt in (unattributed_stats.get("by_participant", {}) or {}).items():
                print(f"  - {part}: masked_rows={cnt}")

        self.raw_targets = raw_targets
        self.label_mask = label_mask

        self.input_scaler = input_scaler if input_scaler is not None else StandardScaler()
        self.output_scaler = output_scaler if output_scaler is not None else MaskedStandardScaler()

        if self.is_training:
            self.input_scaler.fit(mains.reshape(-1, 1))
            self.output_scaler.fit(self.raw_targets, mask=self.label_mask)

        self.mains_scaled = self.input_scaler.transform(mains.reshape(-1, 1)).astype(np.float32).flatten()
        self.targets_scaled = self.output_scaler.transform(self.raw_targets).astype(np.float32)
        self.targets_scaled[self.label_mask == 0] = 0.0

        self.threshold_map = threshold_map_local
        threshold_vec = np.array([self.threshold_map[d] for d in self.device_list], dtype=np.float32)
        self.active_mask = ((self.raw_targets > threshold_vec) & (self.label_mask > 0)).astype(np.float32)

        self.left_context = self.window_size // 2
        self.right_context = self.window_size - self.left_context
        centers = np.arange(
            self.left_context,
            len(self.data) - self.right_context + 1,
            self.stride,
            dtype=np.int64,
        )

        if self.drop_unlabeled_centers:
            known_counts = self.label_mask[centers].sum(axis=1)
            centers = centers[known_counts > 0]

        self.valid_centers = centers.tolist()
        if not self.valid_centers:
            raise ValueError("No valid windows found for dataset configuration")

        aug_enabled = bool(augmentation_cfg.get("enabled", False))
        self.aug_enabled = bool(self.is_training and aug_enabled)
        self.aug_active_only = bool(augmentation_cfg.get("active_only", True))

        time_shift_cfg = augmentation_cfg.get("time_shift", {})
        self.aug_time_shift_enabled = bool(time_shift_cfg.get("enabled", True))
        self.aug_time_shift_prob = float(np.clip(float(time_shift_cfg.get("probability", 0.5)), 0.0, 1.0))
        self.aug_time_shift_max_steps = max(0, int(time_shift_cfg.get("max_steps", 2)))

        jitter_cfg = augmentation_cfg.get("jitter", {})
        self.aug_jitter_enabled = bool(jitter_cfg.get("enabled", True))
        self.aug_jitter_prob = float(np.clip(float(jitter_cfg.get("probability", 0.4)), 0.0, 1.0))
        self.aug_jitter_std = max(0.0, float(jitter_cfg.get("std", 0.02)))

    def compute_device_activity_rates(self):
        centers = np.asarray(self.valid_centers, dtype=np.int64)
        known = self.label_mask[centers].sum(axis=0)
        active = self.active_mask[centers].sum(axis=0)
        return active / np.clip(known, 1e-8, None)

    def build_participant_availability(self, min_active_points=1, min_known_points=1):
        centers = np.asarray(self.valid_centers, dtype=np.int64)
        center_parts = self.participants[centers]

        availability = {}
        for part in np.unique(center_parts):
            part_mask = center_parts == part
            part_centers = centers[part_mask]
            if len(part_centers) == 0:
                continue

            known_counts = self.label_mask[part_centers].sum(axis=0)
            active_counts = self.active_mask[part_centers].sum(axis=0)

            availability[str(part)] = {}
            for i, dev in enumerate(self.device_list):
                availability[str(part)][dev] = bool(
                    known_counts[i] >= float(min_known_points)
                    and active_counts[i] >= float(min_active_points)
                )

        return availability

    def get_sample_weights(self, active_boost=4.0, device_boost=None):
        centers = np.asarray(self.valid_centers, dtype=np.int64)
        center_active = self.active_mask[centers]
        active_rows = center_active.sum(axis=1) > 0
        weights = np.ones(len(centers), dtype=np.float32)

        if device_boost is not None:
            device_boost = np.asarray(device_boost, dtype=np.float32).reshape(1, -1)
            if device_boost.shape[1] != center_active.shape[1]:
                raise ValueError(
                    f"device_boost size mismatch: got {device_boost.shape[1]}, "
                    f"expected {center_active.shape[1]}"
                )
            # For each window, use the strongest boost among active devices.
            per_row_boost = (center_active * device_boost).max(axis=1)
            weights = np.maximum(weights, per_row_boost)

        if float(active_boost) > 1.0:
            weights[active_rows] = np.maximum(weights[active_rows], float(active_boost))
        return weights

    def __len__(self):
        return len(self.valid_centers)

    def _is_valid_aug_center(self, candidate_center, original_center):
        if candidate_center < self.left_context:
            return False
        if candidate_center > (len(self.data) - self.right_context):
            return False
        if self.participants[candidate_center] != self.participants[original_center]:
            return False
        if self.drop_unlabeled_centers and self.label_mask[candidate_center].sum() <= 0:
            return False
        return True

    def _sample_time_shifted_center(self, center_idx):
        if not self.aug_enabled:
            return center_idx
        if not self.aug_time_shift_enabled:
            return center_idx
        if self.aug_time_shift_max_steps <= 0:
            return center_idx
        if torch.rand(1).item() >= self.aug_time_shift_prob:
            return center_idx

        shift = int(
            torch.randint(
                low=-self.aug_time_shift_max_steps,
                high=self.aug_time_shift_max_steps + 1,
                size=(1,),
            ).item()
        )
        if shift == 0:
            return center_idx

        candidate_center = center_idx + shift
        if self._is_valid_aug_center(candidate_center, center_idx):
            return candidate_center
        return center_idx

    def __getitem__(self, idx):
        center_idx = self.valid_centers[idx]

        if self.aug_enabled:
            active_at_center = self.active_mask[center_idx].sum() > 0
            if (not self.aug_active_only) or active_at_center:
                center_idx = self._sample_time_shifted_center(center_idx)

        start_idx = center_idx - self.left_context
        end_idx = center_idx + self.right_context

        x = self.mains_scaled[start_idx:end_idx].reshape(-1, 1).astype(np.float32)
        if self.target_mode == "sequence":
            y = self.targets_scaled[start_idx:end_idx].astype(np.float32)
            y_mask = self.label_mask[start_idx:end_idx].astype(np.float32)
            y_active = self.active_mask[start_idx:end_idx].astype(np.float32)
        else:
            y = self.targets_scaled[center_idx].astype(np.float32)
            y_mask = self.label_mask[center_idx].astype(np.float32)
            y_active = self.active_mask[center_idx].astype(np.float32)
        ts = self.timestamps[center_idx]
        participant = self.participants[center_idx]

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        y_mask_tensor = torch.from_numpy(y_mask)
        y_active_tensor = torch.from_numpy(y_active)

        if self.aug_enabled and self.aug_jitter_enabled and self.aug_jitter_std > 0:
            active_for_jitter = bool(y_active.sum() > 0)
            if (not self.aug_active_only) or active_for_jitter:
                if torch.rand(1).item() < self.aug_jitter_prob:
                    x_tensor = x_tensor + torch.randn_like(x_tensor) * self.aug_jitter_std

        return (
            x_tensor,
            y_tensor,
            y_mask_tensor,
            y_active_tensor,
            ts,
            participant,
        )
