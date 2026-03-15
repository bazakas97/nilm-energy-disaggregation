import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from plotly.offline import plot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from data_preprocessing import NILMDataset, device_thresholds, normalize_participant_filter
from models import build_model
from postprocessing import (
    advanced_postprocess_predictions,
    apply_onoff_probability_gating,
    enforce_mains_power_budget,
)


def teca(actual, prediction):
    mask = ~np.isnan(actual)
    if mask.sum() == 0:
        return 0.0
    actual = actual[mask]
    prediction = prediction[mask]
    denominator = 2.0 * np.sum(actual)
    if denominator == 0:
        return 0.0
    return float((1.0 - np.abs(actual - prediction).sum() / denominator) * 100.0)


def resolve_split_participants(data_cfg, split_name):
    split = str(split_name).strip().lower()
    split_map = data_cfg.get("participants_by_split")
    if isinstance(split_map, dict) and split in split_map:
        return normalize_participant_filter(split_map.get(split))

    split_key = f"participants_{split}"
    if split_key in data_cfg:
        return normalize_participant_filter(data_cfg.get(split_key))

    if split in {"eval", "evaluate"} and "participants_eval" in data_cfg:
        return normalize_participant_filter(data_cfg.get("participants_eval"))

    return normalize_participant_filter(data_cfg.get("participants"))


def _normalize_time_bound(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_date_range(range_cfg):
    if range_cfg is None:
        return None, None

    if isinstance(range_cfg, dict):
        start = (
            range_cfg.get("start")
            or range_cfg.get("from")
            or range_cfg.get("date_start")
            or range_cfg.get("start_timestamp")
        )
        end = (
            range_cfg.get("end")
            or range_cfg.get("to")
            or range_cfg.get("date_end")
            or range_cfg.get("end_timestamp")
        )
        return _normalize_time_bound(start), _normalize_time_bound(end)

    if isinstance(range_cfg, (list, tuple)):
        start = range_cfg[0] if len(range_cfg) > 0 else None
        end = range_cfg[1] if len(range_cfg) > 1 else None
        return _normalize_time_bound(start), _normalize_time_bound(end)

    return None, None


def resolve_split_date_range(data_cfg, split_name):
    split = str(split_name).strip().lower()

    split_map = data_cfg.get("date_range_by_split")
    if isinstance(split_map, dict) and split in split_map:
        start, end = _normalize_date_range(split_map.get(split))
        if start is not None or end is not None:
            return start, end

    start = _normalize_time_bound(data_cfg.get(f"date_start_{split}"))
    end = _normalize_time_bound(data_cfg.get(f"date_end_{split}"))
    if start is not None or end is not None:
        return start, end

    start = _normalize_time_bound(data_cfg.get("date_start"))
    end = _normalize_time_bound(data_cfg.get("date_end"))
    return start, end


def masked_mse(prediction, target, label_mask):
    squared_error = ((prediction - target) ** 2) * label_mask
    denom = label_mask.sum().clamp(min=1.0)
    return squared_error.sum() / denom


def unpack_model_outputs(model_outputs):
    if isinstance(model_outputs, dict):
        power = model_outputs.get("power")
        onoff_logits = model_outputs.get("onoff_logits")
    elif isinstance(model_outputs, (tuple, list)):
        power = model_outputs[0]
        onoff_logits = model_outputs[1] if len(model_outputs) > 1 else None
    else:
        power = model_outputs
        onoff_logits = None

    if power is None:
        raise ValueError("Model output did not contain power predictions.")
    return power, onoff_logits


def center_slice_if_sequence(tensor):
    if tensor is None:
        return None
    if tensor.ndim == 3:
        return tensor[:, tensor.shape[1] // 2, :]
    return tensor


def align_prediction_and_targets(power_pred, onoff_logits, target, label_mask):
    pred = power_pred
    logits = onoff_logits
    y = target
    y_mask = label_mask

    if pred.ndim == 3 and y.ndim == 2:
        pred = center_slice_if_sequence(pred)
        logits = center_slice_if_sequence(logits)
    elif pred.ndim == 2 and y.ndim == 3:
        y = center_slice_if_sequence(y)
        y_mask = center_slice_if_sequence(y_mask)

    if pred.ndim != y.ndim:
        raise ValueError(f"Prediction and target dims do not match after alignment: {pred.ndim} vs {y.ndim}")
    if pred.shape != y.shape:
        raise ValueError(f"Prediction and target shapes do not match after alignment: {pred.shape} vs {y.shape}")
    if y_mask.shape != y.shape:
        raise ValueError(f"Label mask shape {y_mask.shape} must match target shape {y.shape}")

    return pred, logits, y, y_mask


def line_plot(df, x_col, y_cols, title="", filename="line_plot.html"):
    if df[x_col].dtype != "datetime64[ns]":
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
    fig = px.line(df, x=x_col, y=y_cols, title=title)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        ),
    )
    plot(fig, auto_open=False, filename=filename)


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return None
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return None


def safe_div(num, den):
    if den == 0:
        return None
    return float(num / den)


def signal_aggregate_error(y_true, y_pred):
    denominator = float(np.abs(y_true).sum())
    if denominator <= 1e-8:
        return None
    return float(np.abs(y_pred.sum() - y_true.sum()) / denominator)


def on_off_metrics(y_true, y_pred, threshold):
    true_on = y_true > threshold
    pred_on = y_pred > threshold

    tp = int(np.logical_and(true_on, pred_on).sum())
    tn = int(np.logical_and(~true_on, ~pred_on).sum())
    fp = int(np.logical_and(~true_on, pred_on).sum())
    fn = int(np.logical_and(true_on, ~pred_on).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "on_off_precision": precision,
        "on_off_recall": recall,
        "on_off_f1": f1,
        "on_off_accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def on_off_captured(onoff, capture_cfg):
    cfg = capture_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return None

    checks = []
    p_min = cfg.get("precision_min")
    r_min = cfg.get("recall_min")
    f1_min = cfg.get("f1_min")
    acc_min = cfg.get("accuracy_min")

    if p_min is not None:
        checks.append(onoff["on_off_precision"] is not None and onoff["on_off_precision"] >= float(p_min))
    if r_min is not None:
        checks.append(onoff["on_off_recall"] is not None and onoff["on_off_recall"] >= float(r_min))
    if f1_min is not None:
        checks.append(onoff["on_off_f1"] is not None and onoff["on_off_f1"] >= float(f1_min))
    if acc_min is not None:
        checks.append(onoff["on_off_accuracy"] is not None and onoff["on_off_accuracy"] >= float(acc_min))

    if not checks:
        return None
    return all(checks)


def _sanitize_filename_component(value):
    text = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    safe = safe.strip("._")
    return safe or "unknown"


def compute_device_metrics(
    true_unscaled,
    pred_unscaled,
    label_mask,
    threshold_vec,
    device_list,
    capture_cfg=None,
):
    capture_cfg = capture_cfg or {}
    metrics = {}
    for i, dev in enumerate(device_list):
        known = label_mask[:, i] > 0.5
        known_count = int(known.sum())
        threshold = float(threshold_vec[i])

        if known_count == 0:
            metrics[dev] = {
                "known_points": 0,
                "active_points": 0,
                "teca": None,
                "r2": None,
                "mae": None,
                "mse": None,
                "sae": None,
                "on_off_precision": None,
                "on_off_recall": None,
                "on_off_f1": None,
                "on_off_accuracy": None,
                "on_off_captured": None,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }
            continue

        t_known = true_unscaled[known, i]
        p_known = pred_unscaled[known, i]
        active = t_known > threshold
        active_count = int(active.sum())

        onoff = on_off_metrics(t_known, p_known, threshold)
        captured = on_off_captured(onoff, capture_cfg)

        if active_count == 0:
            metrics[dev] = {
                "known_points": known_count,
                "active_points": 0,
                "teca": None,
                "r2": None,
                "mae": None,
                "mse": None,
                "sae": None,
                "on_off_precision": onoff["on_off_precision"],
                "on_off_recall": onoff["on_off_recall"],
                "on_off_f1": onoff["on_off_f1"],
                "on_off_accuracy": onoff["on_off_accuracy"],
                "on_off_captured": captured,
                "tp": onoff["tp"],
                "fp": onoff["fp"],
                "fn": onoff["fn"],
                "tn": onoff["tn"],
            }
            continue

        t = t_known[active]
        p = p_known[active]
        metrics[dev] = {
            "known_points": known_count,
            "active_points": active_count,
            "teca": float(teca(t, p)),
            "r2": safe_r2(t, p),
            "mae": float(mean_absolute_error(t, p)),
            "mse": float(mean_squared_error(t, p)),
            "sae": signal_aggregate_error(t, p),
            "on_off_precision": onoff["on_off_precision"],
            "on_off_recall": onoff["on_off_recall"],
            "on_off_f1": onoff["on_off_f1"],
            "on_off_accuracy": onoff["on_off_accuracy"],
            "on_off_captured": captured,
            "tp": onoff["tp"],
            "fp": onoff["fp"],
            "fn": onoff["fn"],
            "tn": onoff["tn"],
        }

    return metrics


def summarize_device_metrics(metrics):
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    f1_values = []

    weighted_active_points = 0
    weighted_mae_sum = 0.0
    weighted_sae_sum = 0.0

    for m in metrics.values():
        tp_sum += int(m.get("tp", 0) or 0)
        fp_sum += int(m.get("fp", 0) or 0)
        fn_sum += int(m.get("fn", 0) or 0)

        f1 = m.get("on_off_f1")
        if f1 is not None:
            f1_values.append(float(f1))

        active_points = int(m.get("active_points", 0) or 0)
        if active_points > 0:
            mae = m.get("mae")
            if mae is not None:
                weighted_mae_sum += float(mae) * active_points

            sae = m.get("sae")
            if sae is not None:
                weighted_sae_sum += float(sae) * active_points

            weighted_active_points += active_points

    precision_micro = safe_div(tp_sum, tp_sum + fp_sum)
    recall_micro = safe_div(tp_sum, tp_sum + fn_sum)

    if precision_micro is None or recall_micro is None or (precision_micro + recall_micro) <= 0:
        f1_micro = None
    else:
        f1_micro = float(2.0 * precision_micro * recall_micro / (precision_micro + recall_micro))

    f1_macro = float(np.mean(f1_values)) if f1_values else None

    mae_weighted_active = (
        float(weighted_mae_sum / weighted_active_points) if weighted_active_points > 0 else None
    )
    sae_weighted_active = (
        float(weighted_sae_sum / weighted_active_points) if weighted_active_points > 0 else None
    )

    return {
        "on_off_f1_micro": f1_micro,
        "on_off_f1_macro": f1_macro,
        "on_off_precision_micro": precision_micro,
        "on_off_recall_micro": recall_micro,
        "mae_weighted_active": mae_weighted_active,
        "sae_weighted_active": sae_weighted_active,
    }


def print_device_metrics_block(title, metrics):
    print(f"\n{title}")
    summary = summarize_device_metrics(metrics)
    print(
        "  Overall: "
        f"F1_micro={summary['on_off_f1_micro']}, "
        f"F1_macro={summary['on_off_f1_macro']}, "
        f"P_micro={summary['on_off_precision_micro']}, "
        f"R_micro={summary['on_off_recall_micro']}, "
        f"MAE_w_active={summary['mae_weighted_active']}, "
        f"SAE_w_active={summary['sae_weighted_active']}"
    )
    for dev, m in metrics.items():
        if m["known_points"] == 0:
            print(f"  {dev}: known=0 (no labeled points)")
            continue
        print(
            f"  {dev}: known={m['known_points']}, active={m['active_points']}, "
            f"TECA={m['teca']}, R²={m['r2']}, MAE={m['mae']}, MSE={m['mse']}, SAE={m['sae']}, "
            f"ON/OFF(F1={m['on_off_f1']}, P={m['on_off_precision']}, "
            f"R={m['on_off_recall']}, Acc={m['on_off_accuracy']}), "
            f"ON/OFF_CAPTURED={m['on_off_captured']}"
        )


def apply_participant_device_gating(
    predictions,
    participants,
    device_list,
    availability_map,
    unknown_participant_behavior="allow",
):
    if not availability_map:
        return predictions, 0

    gated = predictions.copy()
    gated_values = 0
    unknown_mode = str(unknown_participant_behavior).lower()

    for i, participant in enumerate(participants):
        participant_key = str(participant)
        part_rules = availability_map.get(participant_key)

        if part_rules is None:
            if unknown_mode == "block":
                gated[i, :] = 0.0
                gated_values += len(device_list)
            continue

        for j, dev in enumerate(device_list):
            if not bool(part_rules.get(dev, True)):
                gated[i, j] = 0.0
                gated_values += 1

    return gated, gated_values


def merge_participant_device_availability(base_map, override_map):
    merged = {}
    if isinstance(base_map, dict):
        for participant, device_rules in base_map.items():
            if isinstance(device_rules, dict):
                merged[str(participant)] = {str(dev): bool(flag) for dev, flag in device_rules.items()}

    if isinstance(override_map, dict):
        for participant, device_rules in override_map.items():
            participant_key = str(participant)
            existing = merged.get(participant_key, {}).copy()
            if isinstance(device_rules, dict):
                for dev, flag in device_rules.items():
                    existing[str(dev)] = bool(flag)
            merged[participant_key] = existing
    return merged


def resolve_report_device_subset(device_list, reported_device_list):
    trained_device_list = list(device_list)
    if not reported_device_list:
        return trained_device_list, list(range(len(trained_device_list)))

    index_by_device = {dev: idx for idx, dev in enumerate(trained_device_list)}
    filtered_devices = [dev for dev in reported_device_list if dev in index_by_device]
    if not filtered_devices:
        return trained_device_list, list(range(len(trained_device_list)))
    return filtered_devices, [index_by_device[dev] for dev in filtered_devices]


def evaluate_model(
    model,
    test_data_path,
    device_list,
    reported_device_list,
    thresholds_cfg,
    input_scaler,
    output_scaler,
    window_size,
    batch_size,
    device,
    data_cfg=None,
    eval_cfg=None,
    paths=None,
    participant_device_availability=None,
    participant_gating_cfg=None,
):
    data_cfg = data_cfg or {}
    eval_cfg = eval_cfg or {}
    paths = paths or {}
    max_rows_strategy = data_cfg.get("max_rows_strategy", "head")

    eval_split = str(eval_cfg.get("split_name", "test")).strip().lower()
    participant_filter = resolve_split_participants(data_cfg, eval_split)
    if participant_filter is None and eval_split != "test":
        participant_filter = resolve_split_participants(data_cfg, "test")
    date_start, date_end = resolve_split_date_range(data_cfg, eval_split)
    if date_start is None and date_end is None and eval_split != "test":
        date_start, date_end = resolve_split_date_range(data_cfg, "test")
    print(
        "Evaluation split/participants:",
        {"split": eval_split, "participants": participant_filter if participant_filter is not None else "ALL"},
    )
    print("Evaluation date filter:", {"start": date_start, "end": date_end})

    test_dataset = NILMDataset(
        data_path=test_data_path,
        window_size=window_size,
        device_list=device_list,
        device_thresholds=thresholds_cfg,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        stride=int(eval_cfg.get("stride", 1)),
        is_training=False,
        participant_filter=participant_filter,
        drop_unlabeled_centers=bool(data_cfg.get("drop_unlabeled_centers", True)),
        preprocessing_cfg=data_cfg.get("preprocessing", {}),
        timestamp_col=data_cfg.get("timestamp_col"),
        start_timestamp=date_start,
        end_timestamp=date_end,
        max_rows=data_cfg.get("max_rows_test"),
        max_rows_strategy=max_rows_strategy,
        target_mode=data_cfg.get("target_mode", "point"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true_scaled = []
    all_pred_scaled = []
    all_mask = []
    all_mains_unscaled = []
    all_timestamps = []
    all_participants = []
    all_onoff_prob = []

    max_test_batches = eval_cfg.get("max_test_batches")
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_mask, _, batch_ts, batch_participants) in enumerate(
            test_loader, start=1
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            raw_outputs = model(batch_x)
            pred_scaled, onoff_logits = unpack_model_outputs(raw_outputs)
            pred_scaled, onoff_logits, batch_y, batch_mask = align_prediction_and_targets(
                power_pred=pred_scaled,
                onoff_logits=onoff_logits,
                target=batch_y,
                label_mask=batch_mask,
            )
            loss = masked_mse(pred_scaled, batch_y, batch_mask)
            total_loss += float(loss.item())
            n_batches += 1

            metric_true = center_slice_if_sequence(batch_y)
            metric_pred = center_slice_if_sequence(pred_scaled)
            metric_mask = center_slice_if_sequence(batch_mask)
            all_true_scaled.append(metric_true.cpu().numpy())
            all_pred_scaled.append(metric_pred.cpu().numpy())
            all_mask.append(metric_mask.cpu().numpy())
            if onoff_logits is not None:
                all_onoff_prob.append(torch.sigmoid(center_slice_if_sequence(onoff_logits)).cpu().numpy())

            center_idx = window_size // 2
            center_mains_scaled = batch_x[:, center_idx, :].cpu().numpy()
            center_mains_unscaled = input_scaler.inverse_transform(center_mains_scaled).flatten()
            all_mains_unscaled.append(center_mains_unscaled)
            all_timestamps.extend([str(ts) for ts in batch_ts])
            all_participants.extend([str(p) for p in batch_participants])

            if max_test_batches and batch_idx >= max_test_batches:
                break

    if n_batches == 0:
        print("No evaluation batches processed.")
        return

    scaled_loss = total_loss / n_batches
    print(f"\n[TEST] Masked MSE (scaled): {scaled_loss:.6f}")

    true_scaled = np.concatenate(all_true_scaled, axis=0)
    pred_scaled = np.concatenate(all_pred_scaled, axis=0)
    label_mask = np.concatenate(all_mask, axis=0)
    mains_unscaled = np.concatenate(all_mains_unscaled, axis=0)
    onoff_prob = np.concatenate(all_onoff_prob, axis=0) if all_onoff_prob else None

    true_unscaled = output_scaler.inverse_transform(true_scaled)
    pred_unscaled = output_scaler.inverse_transform(pred_scaled)

    threshold_vec = np.array([test_dataset.threshold_map[d] for d in device_list], dtype=np.float32)
    report_device_list, report_indices = resolve_report_device_subset(device_list, reported_device_list)
    report_threshold_vec = threshold_vec[report_indices]
    participant_arr = np.asarray(all_participants, dtype=object).astype(str)

    gating_cfg = participant_gating_cfg or {}
    if bool(gating_cfg.get("enabled", True)):
        pred_unscaled, gated_values = apply_participant_device_gating(
            predictions=pred_unscaled,
            participants=all_participants,
            device_list=device_list,
            availability_map=participant_device_availability or {},
            unknown_participant_behavior=gating_cfg.get("unknown_participant_behavior", "allow"),
        )
        print(f"[Gating] Zeroed {gated_values} participant-device predictions based on train availability.")

    onoff_gate_cfg = eval_cfg.get("onoff_probability_gating", {})
    if bool(onoff_gate_cfg.get("enabled", False)):
        if onoff_prob is None:
            print("[ON/OFF Gating] Skipped: model does not provide ON/OFF probabilities.")
        else:
            pred_unscaled, onoff_zeroed = apply_onoff_probability_gating(
                predictions=pred_unscaled,
                onoff_prob=onoff_prob,
                device_list=device_list,
                participants=all_participants,
                gating_cfg=onoff_gate_cfg,
            )
            print(f"[ON/OFF Gating] Zeroed {onoff_zeroed} predictions using probability hysteresis.")

    if bool(eval_cfg.get("postprocess_predictions", True)):
        post_params = eval_cfg.get("device_postprocessing_params", {})
        per_participant_post_params = eval_cfg.get("participant_device_postprocessing_params", {}) or {}
        if per_participant_post_params:
            unique_participants = sorted(set(participant_arr.tolist()))
            for participant in unique_participants:
                part_mask = participant_arr == participant
                participant_overrides = per_participant_post_params.get(participant, {}) or {}
                for i, dev in enumerate(device_list):
                    params = dict(post_params.get(dev, {}) or {})
                    dev_override = participant_overrides.get(dev, {}) or {}
                    if dev_override:
                        params.update(dev_override)
                    pred_unscaled[part_mask, i] = advanced_postprocess_predictions(
                        predictions=pred_unscaled[part_mask, i],
                        min_duration=int(params.get("min_duration", 0)),
                        min_energy_value=float(params.get("min_energy_value", test_dataset.threshold_map[dev])),
                        cycle_peak_min=(
                            float(params["cycle_peak_min"])
                            if "cycle_peak_min" in params
                            else None
                        ),
                    )
        else:
            for i, dev in enumerate(device_list):
                params = post_params.get(dev, {})
                pred_unscaled[:, i] = advanced_postprocess_predictions(
                    predictions=pred_unscaled[:, i],
                    min_duration=int(params.get("min_duration", 0)),
                    min_energy_value=float(params.get("min_energy_value", test_dataset.threshold_map[dev])),
                    cycle_peak_min=(
                        float(params["cycle_peak_min"])
                        if "cycle_peak_min" in params
                        else None
                    ),
                )

    if bool(eval_cfg.get("postprocess_ground_truth", False)):
        post_params = eval_cfg.get("device_postprocessing_params", {})
        for i, dev in enumerate(device_list):
            params = post_params.get(dev, {})
            true_unscaled[:, i] = advanced_postprocess_predictions(
                predictions=true_unscaled[:, i],
                min_duration=int(params.get("min_duration", 0)),
                min_energy_value=float(params.get("min_energy_value", test_dataset.threshold_map[dev])),
                cycle_peak_min=(
                    float(params["cycle_peak_min"])
                    if "cycle_peak_min" in params
                    else None
                ),
            )

    mains_budget_cfg = eval_cfg.get("mains_budget_postprocessing", {}) or {}
    if bool(mains_budget_cfg.get("enabled", False)):
        pred_unscaled, adjusted_rows = enforce_mains_power_budget(
            predictions=pred_unscaled,
            mains=mains_unscaled,
            mode=mains_budget_cfg.get("mode", "proportional"),
        )
        print(f"[Mains Budget] Adjusted {adjusted_rows} rows (mode={mains_budget_cfg.get('mode', 'proportional')}).")

    capture_cfg = eval_cfg.get("on_off_capture", {})
    overall_metrics = compute_device_metrics(
        true_unscaled=true_unscaled[:, report_indices],
        pred_unscaled=pred_unscaled[:, report_indices],
        label_mask=label_mask[:, report_indices],
        threshold_vec=report_threshold_vec,
        device_list=report_device_list,
        capture_cfg=capture_cfg,
    )
    print_device_metrics_block("[Overall Device Metrics]", overall_metrics)

    per_participant_metrics = {}
    if bool(eval_cfg.get("report_per_participant", True)):
        unique_participants = sorted(set(participant_arr.tolist()))
        for participant in unique_participants:
            part_mask = participant_arr == participant
            participant_metrics = compute_device_metrics(
                true_unscaled=true_unscaled[part_mask][:, report_indices],
                pred_unscaled=pred_unscaled[part_mask][:, report_indices],
                label_mask=label_mask[part_mask][:, report_indices],
                threshold_vec=report_threshold_vec,
                device_list=report_device_list,
                capture_cfg=capture_cfg,
            )
            per_participant_metrics[participant] = participant_metrics
            print_device_metrics_block(f"[Participant Metrics] {participant}", participant_metrics)

    predictions_csv = paths.get("predictions_csv")
    if predictions_csv:
        parent = os.path.dirname(predictions_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df_save = pd.DataFrame(
            {
                "timestamp": all_timestamps,
                "participant": all_participants,
                "mains": mains_unscaled,
            }
        )
        for report_idx, dev in zip(report_indices, report_device_list):
            df_save[f"{dev}_true"] = true_unscaled[:, report_idx]
            df_save[f"{dev}_pred"] = pred_unscaled[:, report_idx]
            df_save[f"{dev}_known"] = label_mask[:, report_idx]
            if onoff_prob is not None:
                df_save[f"{dev}_onoff_prob"] = onoff_prob[:, report_idx]
        df_save.to_csv(predictions_csv, index=False)
        print(f"Predictions saved to '{predictions_csv}'")

        metrics_json = paths.get("metrics_json")
        if not metrics_json:
            metrics_json = os.path.splitext(predictions_csv)[0] + "_metrics.json"
        metrics_parent = os.path.dirname(metrics_json)
        if metrics_parent:
            os.makedirs(metrics_parent, exist_ok=True)
        with open(metrics_json, "w", encoding="utf-8") as f:
            per_participant_summary = {
                participant: summarize_device_metrics(m) for participant, m in per_participant_metrics.items()
            }
            json.dump(
                {
                    "scaled_test_loss": float(scaled_loss),
                    "overall": overall_metrics,
                    "overall_summary": summarize_device_metrics(overall_metrics),
                    "per_participant": per_participant_metrics,
                    "per_participant_summary": per_participant_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Metrics saved to '{metrics_json}'")

    plots_dir = paths.get("plots_dir")
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plot_head = int(eval_cfg.get("plot_head", 9000))
        plot_include_mains = bool(eval_cfg.get("plot_include_mains", False))
        mask_unknown_for_plots = bool(eval_cfg.get("mask_unknown_for_plots", True))
        for report_idx, dev in zip(report_indices, report_device_list):
            known = label_mask[:, report_idx] > 0.5
            true_plot = true_unscaled[:, report_idx].copy()
            pred_plot = pred_unscaled[:, report_idx].copy()
            if mask_unknown_for_plots:
                true_plot[~known] = np.nan
                pred_plot[~known] = np.nan

            df_plot = pd.DataFrame(
                {
                    "timestamp": all_timestamps,
                    "mains": mains_unscaled,
                    f"{dev}_true": true_plot,
                    f"{dev}_pred": pred_plot,
                }
            ).head(plot_head)
            y_cols = [f"{dev}_true", f"{dev}_pred"]
            if plot_include_mains:
                y_cols = ["mains"] + y_cols
            line_plot(
                df=df_plot,
                x_col="timestamp",
                y_cols=y_cols,
                title=f"Disaggregation for {dev}",
                filename=os.path.join(plots_dir, f"{dev}_test_plot.html"),
            )

        if bool(eval_cfg.get("per_participant_plots", True)):
            participant_plot_head = int(eval_cfg.get("per_participant_plot_head", plot_head))
            participant_plot_dirname = str(eval_cfg.get("per_participant_plots_dirname", "participants"))
            participant_plot_root = os.path.join(plots_dir, participant_plot_dirname)
            os.makedirs(participant_plot_root, exist_ok=True)

            timestamps_arr = np.asarray(all_timestamps, dtype=object)
            min_known_for_plot = int(eval_cfg.get("min_known_points_per_participant_plot", 1))

            for participant in sorted(set(participant_arr.tolist())):
                part_mask = participant_arr == participant
                participant_slug = _sanitize_filename_component(participant)
                participant_dir = os.path.join(participant_plot_root, participant_slug)
                os.makedirs(participant_dir, exist_ok=True)

                part_timestamps = timestamps_arr[part_mask]
                part_mains = mains_unscaled[part_mask]

                for report_idx, dev in zip(report_indices, report_device_list):
                    known = label_mask[part_mask, report_idx] > 0.5
                    if int(known.sum()) < min_known_for_plot:
                        continue

                    part_true = true_unscaled[part_mask, report_idx].copy()
                    part_pred = pred_unscaled[part_mask, report_idx].copy()
                    if mask_unknown_for_plots:
                        part_true[~known] = np.nan
                        part_pred[~known] = np.nan

                    df_part_plot = pd.DataFrame(
                        {
                            "timestamp": part_timestamps,
                            "mains": part_mains,
                            f"{dev}_true": part_true,
                            f"{dev}_pred": part_pred,
                        }
                    )
                    if bool(eval_cfg.get("sort_participant_plots_by_time", True)):
                        df_part_plot["timestamp"] = pd.to_datetime(df_part_plot["timestamp"], errors="coerce")
                        df_part_plot = df_part_plot.sort_values("timestamp")
                    df_part_plot = df_part_plot.head(participant_plot_head)

                    y_cols = [f"{dev}_true", f"{dev}_pred"]
                    if plot_include_mains:
                        y_cols = ["mains"] + y_cols
                    line_plot(
                        df=df_part_plot,
                        x_col="timestamp",
                        y_cols=y_cols,
                        title=f"Disaggregation for {dev} | participant {participant}",
                        filename=os.path.join(participant_dir, f"{dev}_test_plot.html"),
                    )


def main(config):
    paths = config["paths"]
    eval_cfg = config["evaluate"]
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    test_data_path = paths["test_data"]
    window_size = int(eval_cfg["window_size"])
    batch_size = int(eval_cfg["batch_size"])
    device_list = list(eval_cfg["device_list"])
    requested_reported_device_list = list(eval_cfg.get("reported_device_list", device_list))
    thresholds_cfg = eval_cfg.get("active_thresholds", device_thresholds)

    runtime_device = str(eval_cfg.get("runtime_device", "auto")).lower()
    if runtime_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(runtime_device)
    print(f"Runtime device: {device}")

    model_meta_path = paths.get("model_meta")
    use_saved_metadata = bool(eval_cfg.get("use_saved_metadata", True))
    participant_device_availability = {}
    participant_device_availability_override = eval_cfg.get("participant_device_availability_override", {})
    participant_gating_cfg = eval_cfg.get("participant_gating", {})
    if model_meta_path and use_saved_metadata and os.path.exists(model_meta_path):
        with open(model_meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "window_size" in metadata and int(metadata["window_size"]) != window_size:
            print(
                f"Overriding evaluate.window_size={window_size} "
                f"with trained window_size={metadata['window_size']}."
            )
            window_size = int(metadata["window_size"])

        if "device_list" in metadata and metadata["device_list"] != device_list:
            print("Overriding evaluate.device_list with trained device list from metadata.")
            device_list = list(metadata["device_list"])

        if "model" in metadata:
            model_cfg = metadata["model"]

        if "active_thresholds" in metadata:
            thresholds_cfg = metadata["active_thresholds"]
        if "participant_device_availability" in metadata:
            participant_device_availability = metadata["participant_device_availability"]
        if "participant_gating" in metadata and not participant_gating_cfg:
            participant_gating_cfg = metadata["participant_gating"]

    participant_device_availability = merge_participant_device_availability(
        participant_device_availability,
        participant_device_availability_override,
    )
    reported_device_list, report_indices = resolve_report_device_subset(device_list, requested_reported_device_list)
    if reported_device_list != device_list:
        print(f"Reporting subset of devices: {reported_device_list}")
    elif requested_reported_device_list and len(requested_reported_device_list) != len(device_list):
        missing_devices = [dev for dev in requested_reported_device_list if dev not in device_list]
        if missing_devices:
            print(f"Ignoring unknown reported devices not present in trained model: {missing_devices}")

    if not os.path.exists(paths["model_save"]):
        print("No saved model found. Please train first.")
        return
    if not os.path.exists(paths["input_scaler"]) or not os.path.exists(paths["output_scaler"]):
        print("Missing scalers. Please train first.")
        return

    input_scaler = joblib.load(paths["input_scaler"])
    output_scaler = joblib.load(paths["output_scaler"])

    model = build_model(
        model_config=model_cfg,
        output_dim=len(device_list),
        window_size=window_size,
    )
    state_dict = torch.load(paths["model_save"], map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    evaluate_model(
        model=model,
        test_data_path=test_data_path,
        device_list=device_list,
        reported_device_list=reported_device_list,
        thresholds_cfg=thresholds_cfg,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        window_size=window_size,
        batch_size=batch_size,
        device=device,
        data_cfg=data_cfg,
        eval_cfg=eval_cfg,
        paths=paths,
        participant_device_availability=participant_device_availability,
        participant_gating_cfg=participant_gating_cfg,
    )


if __name__ == "__main__":
    print("Please run the program using run.py with a config file.")
