import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data_preprocessing import MaskedStandardScaler, NILMDataset, device_thresholds, normalize_participant_filter
from models import build_model
from postprocessing import advanced_postprocess_predictions, apply_onoff_probability_gating


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


def masked_weighted_mse(
    prediction,
    target,
    label_mask,
    active_mask=None,
    active_loss_weight=1.0,
    device_weights=None,
    regression_on_only=False,
):
    squared_error = (prediction - target) ** 2
    weights = torch.ones_like(squared_error)

    if device_weights is not None:
        device_shape = [1] * prediction.ndim
        device_shape[-1] = -1
        weights = weights * device_weights.view(*device_shape)

    if active_mask is not None:
        if regression_on_only:
            # Restrict regression to ON samples to avoid OFF-state mean collapse.
            weights = weights * active_mask
        elif active_loss_weight > 1.0:
            weights = weights * (1.0 + (active_loss_weight - 1.0) * active_mask)

    valid_weights = weights * label_mask
    if regression_on_only and float(valid_weights.sum().item()) <= 0.0:
        valid_weights = label_mask
    weighted_error = squared_error * valid_weights
    denominator = valid_weights.sum().clamp(min=1.0)
    return weighted_error.sum() / denominator


def masked_multitask_loss(
    prediction,
    target,
    label_mask,
    active_mask=None,
    active_loss_weight=1.0,
    device_weights=None,
    onoff_logits=None,
    onoff_bce_weight=0.0,
    onoff_temperature=30.0,
    output_scale=None,
    output_mean=None,
    onoff_thresholds=None,
    onoff_pos_weight=None,
    regression_on_only=False,
):
    reg_loss = masked_weighted_mse(
        prediction=prediction,
        target=target,
        label_mask=label_mask,
        active_mask=active_mask,
        active_loss_weight=active_loss_weight,
        device_weights=device_weights,
        regression_on_only=regression_on_only,
    )

    if onoff_bce_weight <= 0.0 or active_mask is None:
        return reg_loss

    logits = onoff_logits
    if logits is None:
        if output_scale is None or output_mean is None or onoff_thresholds is None:
            return reg_loss
        temperature = max(float(onoff_temperature), 1e-6)
        if prediction.ndim == 3:
            pred_unscaled = prediction * output_scale.view(1, 1, -1) + output_mean.view(1, 1, -1)
            logits = (pred_unscaled - onoff_thresholds.view(1, 1, -1)) / temperature
        else:
            pred_unscaled = prediction * output_scale.view(1, -1) + output_mean.view(1, -1)
            logits = (pred_unscaled - onoff_thresholds.view(1, -1)) / temperature

    if logits.shape != prediction.shape:
        raise ValueError(
            f"onoff_logits shape {tuple(logits.shape)} must match prediction shape {tuple(prediction.shape)}"
        )

    onoff_target = (active_mask > 0.5).float()

    bce = F.binary_cross_entropy_with_logits(
        logits,
        onoff_target,
        reduction="none",
        pos_weight=onoff_pos_weight,
    )

    valid_weights = label_mask
    if device_weights is not None:
        device_shape = [1] * prediction.ndim
        device_shape[-1] = -1
        valid_weights = valid_weights * device_weights.view(*device_shape)

    onoff_loss = (bce * valid_weights).sum() / valid_weights.sum().clamp(min=1.0)
    return reg_loss + float(onoff_bce_weight) * onoff_loss


def build_onoff_pos_weight(activity_rates, max_pos_weight=20.0, power=0.5):
    rates = np.clip(np.asarray(activity_rates, dtype=np.float32), 1e-4, 1.0 - 1e-6)
    ratios = (1.0 - rates) / rates
    ratios = np.power(ratios, float(power))
    ratios = np.clip(ratios, 1.0, float(max_pos_weight))
    return ratios.astype(np.float32)


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


def align_prediction_and_targets(power_pred, onoff_logits, target, label_mask, active_mask):
    pred = power_pred
    logits = onoff_logits
    y = target
    y_mask = label_mask
    y_active = active_mask

    if pred.ndim == 3 and y.ndim == 2:
        pred = center_slice_if_sequence(pred)
        logits = center_slice_if_sequence(logits)
    elif pred.ndim == 2 and y.ndim == 3:
        y = center_slice_if_sequence(y)
        y_mask = center_slice_if_sequence(y_mask)
        y_active = center_slice_if_sequence(y_active)

    if pred.ndim != y.ndim:
        raise ValueError(f"Prediction and target dims do not match after alignment: {pred.ndim} vs {y.ndim}")
    if pred.shape != y.shape:
        raise ValueError(f"Prediction and target shapes do not match after alignment: {pred.shape} vs {y.shape}")
    if y_mask.shape != y.shape:
        raise ValueError(f"Label mask shape {y_mask.shape} must match target shape {y.shape}")
    if y_active is not None and y_active.shape != y.shape:
        raise ValueError(f"Active mask shape {y_active.shape} must match target shape {y.shape}")

    return pred, logits, y, y_mask, y_active


def ensure_output_paths(paths):
    for key in [
        "model_save",
        "input_scaler",
        "output_scaler",
        "model_meta",
        "train_plot",
    ]:
        value = paths.get(key)
        if value:
            parent = os.path.dirname(value)
            if parent:
                os.makedirs(parent, exist_ok=True)

    plots_dir = paths.get("plots_dir")
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)


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


def compute_metrics(true_unscaled, pred_unscaled, label_mask, device_list, threshold_vec):
    metrics = {}
    for i, dev in enumerate(device_list):
        known = label_mask[:, i] > 0.5
        known_count = int(known.sum())
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
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }
            continue

        t_known = true_unscaled[known, i]
        p_known = pred_unscaled[known, i]
        threshold = threshold_vec[i]
        active = t_known > threshold
        active_points = int(active.sum())

        onoff = on_off_metrics(t_known, p_known, threshold)
        metrics[dev] = {
            "known_points": known_count,
            "active_points": active_points,
            "sae": None,
            "on_off_precision": onoff["on_off_precision"],
            "on_off_recall": onoff["on_off_recall"],
            "on_off_f1": onoff["on_off_f1"],
            "on_off_accuracy": onoff["on_off_accuracy"],
            "tp": onoff["tp"],
            "fp": onoff["fp"],
            "fn": onoff["fn"],
            "tn": onoff["tn"],
        }

        if active_points == 0:
            metrics[dev]["teca"] = None
            metrics[dev]["r2"] = None
            metrics[dev]["mae"] = None
            metrics[dev]["mse"] = None
        else:
            t_active = t_known[active]
            p_active = p_known[active]
            metrics[dev]["teca"] = teca(t_active, p_active)
            metrics[dev]["r2"] = safe_r2(t_active, p_active)
            metrics[dev]["mae"] = float(mean_absolute_error(t_active, p_active))
            metrics[dev]["mse"] = float(mean_squared_error(t_active, p_active))
            metrics[dev]["sae"] = signal_aggregate_error(t_active, p_active)
    return metrics


def compute_metrics_by_participant(
    true_unscaled,
    pred_unscaled,
    label_mask,
    participants,
    device_list,
    threshold_vec,
):
    participant_arr = np.asarray(participants, dtype=object).astype(str)
    per_participant = {}
    for participant in sorted(set(participant_arr.tolist())):
        part_mask = participant_arr == participant
        per_participant[participant] = compute_metrics(
            true_unscaled=true_unscaled[part_mask],
            pred_unscaled=pred_unscaled[part_mask],
            label_mask=label_mask[part_mask],
            device_list=device_list,
            threshold_vec=threshold_vec,
        )
    return per_participant


def summarize_metrics(metrics):
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


def apply_validation_postprocessing(pred_unscaled, device_list, threshold_vec, validation_post_cfg):
    cfg = validation_post_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return pred_unscaled

    processed = np.array(pred_unscaled, copy=True)

    if bool(cfg.get("clip_negative", True)):
        processed[processed < 0] = 0.0

    device_params = cfg.get("device_postprocessing_params", {}) or {}
    default_to_active_threshold = bool(cfg.get("default_to_active_threshold", True))
    default_min_duration = int(cfg.get("default_min_duration", 0))
    default_min_energy_value = float(cfg.get("default_min_energy_value", 0.0))

    for i, dev in enumerate(device_list):
        params = device_params.get(dev, {}) or {}
        min_duration = int(params.get("min_duration", default_min_duration))

        if "min_energy_value" in params:
            min_energy = float(params["min_energy_value"])
        elif default_to_active_threshold:
            min_energy = float(threshold_vec[i])
        else:
            min_energy = default_min_energy_value

        processed[:, i] = advanced_postprocess_predictions(
            predictions=processed[:, i],
            min_duration=min_duration,
            min_energy_value=min_energy,
            cycle_peak_min=(
                float(params["cycle_peak_min"])
                if "cycle_peak_min" in params
                else None
            ),
        )

    return processed


def validate_model(
    model,
    loader,
    output_scaler,
    device_list,
    threshold_vec,
    device,
    device_weights=None,
    active_loss_weight=1.0,
    onoff_bce_weight=0.0,
    onoff_temperature=30.0,
    output_scale=None,
    output_mean=None,
    onoff_thresholds=None,
    onoff_pos_weight=None,
    regression_on_only=False,
    max_batches=None,
    validation_post_cfg=None,
    report_per_participant=False,
):
    model.eval()
    running_loss = 0.0
    n_batches = 0

    all_true = []
    all_pred = []
    all_mask = []
    all_participants = []
    all_onoff_prob = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_mask, batch_active, _, batch_participants) in enumerate(
            loader, start=1
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            batch_active = batch_active.to(device)

            raw_outputs = model(batch_x)
            power_pred, onoff_logits = unpack_model_outputs(raw_outputs)
            power_pred, onoff_logits, batch_y, batch_mask, batch_active = align_prediction_and_targets(
                power_pred=power_pred,
                onoff_logits=onoff_logits,
                target=batch_y,
                label_mask=batch_mask,
                active_mask=batch_active,
            )
            loss = masked_multitask_loss(
                prediction=power_pred,
                target=batch_y,
                label_mask=batch_mask,
                active_mask=batch_active,
                active_loss_weight=active_loss_weight,
                device_weights=device_weights,
                onoff_logits=onoff_logits,
                onoff_bce_weight=onoff_bce_weight,
                onoff_temperature=onoff_temperature,
                output_scale=output_scale,
                output_mean=output_mean,
                onoff_thresholds=onoff_thresholds,
                onoff_pos_weight=onoff_pos_weight,
                regression_on_only=regression_on_only,
            )
            running_loss += float(loss.item())
            n_batches += 1

            metric_true = center_slice_if_sequence(batch_y)
            metric_pred = center_slice_if_sequence(power_pred)
            metric_mask = center_slice_if_sequence(batch_mask)
            all_true.append(metric_true.detach().cpu().numpy())
            all_pred.append(metric_pred.detach().cpu().numpy())
            all_mask.append(metric_mask.detach().cpu().numpy())
            all_participants.extend([str(p) for p in batch_participants])
            if onoff_logits is not None:
                all_onoff_prob.append(torch.sigmoid(center_slice_if_sequence(onoff_logits)).detach().cpu().numpy())

            if max_batches and batch_idx >= max_batches:
                break

    if n_batches == 0:
        return float("inf"), {}, {}

    avg_loss = running_loss / n_batches
    true_scaled = np.concatenate(all_true, axis=0)
    pred_scaled = np.concatenate(all_pred, axis=0)
    label_mask = np.concatenate(all_mask, axis=0)

    true_unscaled = output_scaler.inverse_transform(true_scaled)
    pred_unscaled = output_scaler.inverse_transform(pred_scaled)
    pred_for_metrics = apply_validation_postprocessing(
        pred_unscaled=pred_unscaled,
        device_list=device_list,
        threshold_vec=threshold_vec,
        validation_post_cfg=validation_post_cfg,
    )

    onoff_gate_cfg = (validation_post_cfg or {}).get("onoff_probability_gating", {})
    if bool(onoff_gate_cfg.get("enabled", False)):
        if all_onoff_prob:
            onoff_prob = np.concatenate(all_onoff_prob, axis=0)
            pred_for_metrics, _ = apply_onoff_probability_gating(
                predictions=pred_for_metrics,
                onoff_prob=onoff_prob,
                device_list=device_list,
                participants=all_participants,
                gating_cfg=onoff_gate_cfg,
            )

    metrics = compute_metrics(
        true_unscaled=true_unscaled,
        pred_unscaled=pred_for_metrics,
        label_mask=label_mask,
        device_list=device_list,
        threshold_vec=threshold_vec,
    )
    per_participant_metrics = {}
    if bool(report_per_participant):
        per_participant_metrics = compute_metrics_by_participant(
            true_unscaled=true_unscaled,
            pred_unscaled=pred_for_metrics,
            label_mask=label_mask,
            participants=all_participants,
            device_list=device_list,
            threshold_vec=threshold_vec,
        )
    return avg_loss, metrics, per_participant_metrics


def save_training_artifacts(paths, model, input_scaler, output_scaler, metadata):
    torch.save(model.state_dict(), paths["model_save"])
    joblib.dump(input_scaler, paths["input_scaler"])
    joblib.dump(output_scaler, paths["output_scaler"])

    meta_path = paths.get("model_meta")
    if meta_path:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    output_scaler,
    device_list,
    threshold_vec,
    paths,
    metadata,
    input_scaler,
    device,
    epochs=20,
    patience=5,
    active_loss_weight=1.0,
    device_weights=None,
    onoff_bce_weight=0.0,
    onoff_temperature=30.0,
    onoff_warmup_epochs=0,
    onoff_ramp_epochs=0,
    output_scale=None,
    output_mean=None,
    onoff_thresholds=None,
    onoff_pos_weight=None,
    regression_on_only=False,
    max_train_batches=None,
    max_val_batches=None,
    validation_post_cfg=None,
    validation_report_per_participant=False,
    early_stop_metric="val_loss",
    early_stop_mode=None,
    no_progress=False,
):
    early_stop_metric = str(early_stop_metric).strip().lower()
    if early_stop_metric in {"loss"}:
        early_stop_metric = "val_loss"
    if early_stop_metric in {"f1", "onoff_f1", "on_off_f1"}:
        early_stop_metric = "f1_micro"
    if early_stop_metric in {"mae"}:
        early_stop_metric = "mae_weighted_active"
    if early_stop_metric in {"sae"}:
        early_stop_metric = "sae_weighted_active"

    inferred_mode = "max" if early_stop_metric in {"f1_micro", "f1_macro"} else "min"
    if early_stop_mode is None:
        early_stop_mode = inferred_mode
    else:
        early_stop_mode = str(early_stop_mode).strip().lower()
        if early_stop_mode not in {"min", "max"}:
            early_stop_mode = inferred_mode

    best_metric_score = float("-inf") if early_stop_mode == "max" else float("inf")
    warned_metric_fallback = False
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    model.to(device)
    for epoch in range(1, epochs + 1):
        if onoff_bce_weight <= 0:
            effective_onoff_weight = 0.0
        else:
            warmup = max(0, int(onoff_warmup_epochs))
            ramp = max(0, int(onoff_ramp_epochs))
            if epoch <= warmup:
                effective_onoff_weight = 0.0
            elif ramp > 0:
                alpha = min(1.0, float(epoch - warmup) / float(ramp))
                effective_onoff_weight = float(onoff_bce_weight) * alpha
            else:
                effective_onoff_weight = float(onoff_bce_weight)

        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (batch_x, batch_y, batch_mask, batch_active, _, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=bool(no_progress)),
            start=1,
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            batch_active = batch_active.to(device)

            optimizer.zero_grad()
            raw_outputs = model(batch_x)
            power_pred, onoff_logits = unpack_model_outputs(raw_outputs)
            power_pred, onoff_logits, batch_y, batch_mask, batch_active = align_prediction_and_targets(
                power_pred=power_pred,
                onoff_logits=onoff_logits,
                target=batch_y,
                label_mask=batch_mask,
                active_mask=batch_active,
            )
            loss = masked_multitask_loss(
                prediction=power_pred,
                target=batch_y,
                label_mask=batch_mask,
                active_mask=batch_active,
                active_loss_weight=active_loss_weight,
                device_weights=device_weights,
                onoff_logits=onoff_logits,
                onoff_bce_weight=effective_onoff_weight,
                onoff_temperature=onoff_temperature,
                output_scale=output_scale,
                output_mean=output_mean,
                onoff_thresholds=onoff_thresholds,
                onoff_pos_weight=onoff_pos_weight,
                regression_on_only=regression_on_only,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

            if max_train_batches and batch_idx >= max_train_batches:
                break

        if n_batches == 0:
            print("No training batches were processed.")
            break

        train_loss = running_loss / n_batches
        train_losses.append(train_loss)

        val_loss, val_metrics, val_metrics_by_participant = validate_model(
            model=model,
            loader=val_loader,
            output_scaler=output_scaler,
            device_list=device_list,
            threshold_vec=threshold_vec,
            device=device,
            device_weights=device_weights,
            active_loss_weight=active_loss_weight,
            onoff_bce_weight=effective_onoff_weight,
            onoff_temperature=onoff_temperature,
            output_scale=output_scale,
            output_mean=output_mean,
            onoff_thresholds=onoff_thresholds,
            onoff_pos_weight=onoff_pos_weight,
            regression_on_only=regression_on_only,
            max_batches=max_val_batches,
            validation_post_cfg=validation_post_cfg,
            report_per_participant=validation_report_per_participant,
        )
        val_losses.append(val_loss)
        val_summary = summarize_metrics(val_metrics)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  ON/OFF loss weight (effective) = {effective_onoff_weight:.4f}")
        print(f"  Train Loss (train objective) = {train_loss:.6f}")
        print(f"  Val Loss   (train objective) = {val_loss:.6f}")
        print(
            "  Val Overall: "
            f"F1_micro={val_summary['on_off_f1_micro']}, "
            f"F1_macro={val_summary['on_off_f1_macro']}, "
            f"P_micro={val_summary['on_off_precision_micro']}, "
            f"R_micro={val_summary['on_off_recall_micro']}, "
            f"MAE_w_active={val_summary['mae_weighted_active']}, "
            f"SAE_w_active={val_summary['sae_weighted_active']}"
        )
        for dev, m in val_metrics.items():
            print(
                f"    {dev}: known={m['known_points']}, active={m['active_points']}, "
                f"TECA={m['teca']}, R²={m['r2']}, MAE={m['mae']}, MSE={m['mse']}, "
                f"SAE={m['sae']}, ON/OFF(F1={m['on_off_f1']}, P={m['on_off_precision']}, "
                f"R={m['on_off_recall']}, Acc={m['on_off_accuracy']})"
            )
        if val_metrics_by_participant:
            for participant, part_metrics in val_metrics_by_participant.items():
                part_summary = summarize_metrics(part_metrics)
                print(f"  [Val Participant] {participant}")
                print(
                    "    Overall: "
                    f"F1_micro={part_summary['on_off_f1_micro']}, "
                    f"F1_macro={part_summary['on_off_f1_macro']}, "
                    f"P_micro={part_summary['on_off_precision_micro']}, "
                    f"R_micro={part_summary['on_off_recall_micro']}, "
                    f"MAE_w_active={part_summary['mae_weighted_active']}, "
                    f"SAE_w_active={part_summary['sae_weighted_active']}"
                )
                for dev, m in part_metrics.items():
                    print(
                        f"    {dev}: known={m['known_points']}, active={m['active_points']}, "
                        f"TECA={m['teca']}, R²={m['r2']}, MAE={m['mae']}, MSE={m['mse']}, "
                        f"SAE={m['sae']}, ON/OFF(F1={m['on_off_f1']}, P={m['on_off_precision']}, "
                        f"R={m['on_off_recall']}, Acc={m['on_off_accuracy']})"
                    )

        scheduler.step(val_loss)

        selected_metric_value = None
        if early_stop_metric == "val_loss":
            selected_metric_value = val_loss
        elif early_stop_metric == "f1_micro":
            selected_metric_value = val_summary["on_off_f1_micro"]
        elif early_stop_metric == "f1_macro":
            selected_metric_value = val_summary["on_off_f1_macro"]
        elif early_stop_metric == "mae_weighted_active":
            selected_metric_value = val_summary["mae_weighted_active"]
        elif early_stop_metric == "sae_weighted_active":
            selected_metric_value = val_summary["sae_weighted_active"]

        if selected_metric_value is None or not np.isfinite(float(selected_metric_value)):
            if not warned_metric_fallback:
                print(
                    f"  [!] Early-stop metric '{early_stop_metric}' unavailable/invalid. "
                    "Falling back to val_loss."
                )
                warned_metric_fallback = True
            selected_metric_value = val_loss

        if early_stop_mode == "max":
            improved = float(selected_metric_value) > best_metric_score
        else:
            improved = float(selected_metric_value) < best_metric_score

        print(
            f"  Early-stop monitor: {early_stop_metric}={float(selected_metric_value):.6f} "
            f"(mode={early_stop_mode})"
        )

        if improved:
            best_metric_score = float(selected_metric_value)
            epochs_no_improve = 0
            save_training_artifacts(
                paths=paths,
                model=model,
                input_scaler=input_scaler,
                output_scaler=output_scaler,
                metadata=metadata,
            )
            print("  [*] Best model + scalers + metadata saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    plot_path = paths.get("train_plot")
    if not plot_path:
        plot_path = os.path.join(paths.get("plots_dir", "results/plots"), "training_validation_loss.png")
    parent = os.path.dirname(plot_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main(config):
    paths = config["paths"]
    train_params = config["train"]
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    max_rows_strategy = data_cfg.get("max_rows_strategy", "head")

    ensure_output_paths(paths)

    train_data_path = paths["train_data"]
    val_data_path = paths["val_data"]

    window_size = int(train_params["window_size"])
    batch_size = int(train_params["batch_size"])
    epochs = int(train_params["epochs"])
    learning_rate = float(train_params["learning_rate"])
    weight_decay = float(train_params.get("weight_decay", 1e-4))
    patience = int(train_params["patience"])
    stride = int(train_params.get("stride", 1))
    device_list = list(train_params["device_list"])
    augmentation_cfg = train_params.get("augmentation", {})

    runtime_device = str(train_params.get("runtime_device", "auto")).lower()
    if runtime_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(runtime_device)
    print(f"Runtime device: {device}")

    force_retrain = bool(train_params.get("force_retrain", False))
    if (
        not force_retrain
        and os.path.exists(paths["model_save"])
        and os.path.exists(paths["input_scaler"])
        and os.path.exists(paths["output_scaler"])
    ):
        print("Model/scalers exist. Set train.force_retrain=true to retrain.")
        return

    input_scaler = StandardScaler()
    output_scaler = MaskedStandardScaler()
    thresholds_cfg = train_params.get("active_thresholds", device_thresholds)

    train_participant_filter = resolve_split_participants(data_cfg, "train")
    val_participant_filter = resolve_split_participants(data_cfg, "val")
    train_date_start, train_date_end = resolve_split_date_range(data_cfg, "train")
    val_date_start, val_date_end = resolve_split_date_range(data_cfg, "val")
    print(
        "Participant filters:",
        {
            "train": train_participant_filter if train_participant_filter is not None else "ALL",
            "val": val_participant_filter if val_participant_filter is not None else "ALL",
        },
    )
    print(
        "Date filters:",
        {
            "train": {"start": train_date_start, "end": train_date_end},
            "val": {"start": val_date_start, "end": val_date_end},
        },
    )

    train_dataset = NILMDataset(
        data_path=train_data_path,
        window_size=window_size,
        device_list=device_list,
        device_thresholds=thresholds_cfg,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        stride=stride,
        is_training=True,
        participant_filter=train_participant_filter,
        drop_unlabeled_centers=bool(data_cfg.get("drop_unlabeled_centers", True)),
        preprocessing_cfg=data_cfg.get("preprocessing", {}),
        timestamp_col=data_cfg.get("timestamp_col"),
        start_timestamp=train_date_start,
        end_timestamp=train_date_end,
        max_rows=data_cfg.get("max_rows_train"),
        max_rows_strategy=max_rows_strategy,
        augmentation_cfg=augmentation_cfg,
        target_mode=data_cfg.get("target_mode", "point"),
    )
    val_dataset = NILMDataset(
        data_path=val_data_path,
        window_size=window_size,
        device_list=device_list,
        device_thresholds=thresholds_cfg,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        stride=int(train_params.get("val_stride", stride)),
        is_training=False,
        participant_filter=val_participant_filter,
        drop_unlabeled_centers=bool(data_cfg.get("drop_unlabeled_centers", True)),
        preprocessing_cfg=data_cfg.get("preprocessing", {}),
        timestamp_col=data_cfg.get("timestamp_col"),
        start_timestamp=val_date_start,
        end_timestamp=val_date_end,
        max_rows=data_cfg.get("max_rows_val"),
        max_rows_strategy=max_rows_strategy,
        augmentation_cfg=None,
        target_mode=data_cfg.get("target_mode", "point"),
    )

    train_parts = sorted(set(str(p) for p in train_dataset.participants.tolist()))
    val_parts = sorted(set(str(p) for p in val_dataset.participants.tolist()))
    print("Train participants loaded:", train_parts)
    print("Val participants loaded  :", val_parts)

    pin_memory = device.type == "cuda"
    num_workers = int(train_params.get("num_workers", 0))
    activity_rates = train_dataset.compute_device_activity_rates()

    balance_active_windows = bool(train_params.get("balance_active_windows", True))
    if balance_active_windows:
        device_sampling_boost = None
        if bool(train_params.get("per_device_sampling_boost", True)):
            sampling_power = float(train_params.get("sampling_boost_power", 0.5))
            sampling_max = float(train_params.get("max_sampling_device_boost", 12.0))
            device_sampling_boost = np.power(np.clip(activity_rates, 1e-4, None), -sampling_power)
            device_sampling_boost = np.clip(device_sampling_boost, 1.0, sampling_max)
            print("Device sampling boosts:", dict(zip(device_list, device_sampling_boost.round(3).tolist())))

        weights = train_dataset.get_sample_weights(
            active_boost=float(train_params.get("active_window_boost", 4.0)),
            device_boost=device_sampling_boost,
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        model_config=model_cfg,
        output_dim=len(device_list),
        window_size=window_size,
    ).to(device)

    use_freq_weighting = bool(train_params.get("use_device_frequency_weighting", True))
    device_weight_tensor = None
    if use_freq_weighting:
        raw_weights = 1.0 / np.sqrt(np.clip(activity_rates, 1e-4, None))
        raw_weights /= np.clip(raw_weights.mean(), 1e-8, None)
        min_w = float(train_params.get("min_device_weight", 0.5))
        max_w = float(train_params.get("max_device_weight", 4.0))
        raw_weights = np.clip(raw_weights, min_w, max_w)
        device_weight_tensor = torch.tensor(raw_weights, dtype=torch.float32, device=device)
        print("Device activity rates:", dict(zip(device_list, activity_rates.round(4).tolist())))
        print("Device loss weights :", dict(zip(device_list, raw_weights.round(4).tolist())))
    else:
        print("Device activity rates:", dict(zip(device_list, activity_rates.round(4).tolist())))

    if train_dataset.aug_enabled:
        print(
            "Train augmentation:",
            {
                "active_only": train_dataset.aug_active_only,
                "time_shift": {
                    "enabled": train_dataset.aug_time_shift_enabled,
                    "probability": train_dataset.aug_time_shift_prob,
                    "max_steps": train_dataset.aug_time_shift_max_steps,
                },
                "jitter": {
                    "enabled": train_dataset.aug_jitter_enabled,
                    "probability": train_dataset.aug_jitter_prob,
                    "std": train_dataset.aug_jitter_std,
                },
            },
        )

    onoff_aux_cfg = train_params.get("onoff_aux_loss", {})
    onoff_enabled = bool(onoff_aux_cfg.get("enabled", True))
    onoff_bce_weight = float(onoff_aux_cfg.get("weight", 0.7)) if onoff_enabled else 0.0
    onoff_temperature = float(onoff_aux_cfg.get("temperature", 30.0))
    onoff_warmup_epochs = int(onoff_aux_cfg.get("warmup_epochs", 1))
    onoff_ramp_epochs = int(onoff_aux_cfg.get("ramp_epochs", 2))
    regression_on_only = bool(onoff_aux_cfg.get("regression_on_only", False))
    onoff_use_pos_weight = bool(onoff_aux_cfg.get("use_pos_weight", True))
    onoff_pos_weight_tensor = None
    if onoff_bce_weight > 0 and onoff_use_pos_weight:
        onoff_pos_weights = build_onoff_pos_weight(
            activity_rates=activity_rates,
            max_pos_weight=float(onoff_aux_cfg.get("max_pos_weight", 20.0)),
            power=float(onoff_aux_cfg.get("pos_weight_power", 0.5)),
        )
        onoff_pos_weight_tensor = torch.tensor(onoff_pos_weights, dtype=torch.float32, device=device)

    print(
        "ON/OFF aux loss:",
        {
            "enabled": onoff_bce_weight > 0,
            "weight": onoff_bce_weight,
            "temperature": onoff_temperature,
            "warmup_epochs": onoff_warmup_epochs,
            "ramp_epochs": onoff_ramp_epochs,
            "regression_on_only": regression_on_only,
            "use_pos_weight": onoff_pos_weight_tensor is not None,
            "pos_weights": (
                onoff_pos_weight_tensor.detach().cpu().numpy().round(3).tolist()
                if onoff_pos_weight_tensor is not None
                else None
            ),
        },
    )

    validation_post_cfg = train_params.get("validation_postprocessing", {})
    if bool(validation_post_cfg.get("enabled", False)):
        onoff_gate_cfg = validation_post_cfg.get("onoff_probability_gating", {})
        print(
            "Validation postprocessing:",
            {
                "clip_negative": bool(validation_post_cfg.get("clip_negative", True)),
                "default_to_active_threshold": bool(
                    validation_post_cfg.get("default_to_active_threshold", True)
                ),
                "default_min_duration": int(validation_post_cfg.get("default_min_duration", 0)),
                "device_postprocessing_params": validation_post_cfg.get("device_postprocessing_params", {}),
                "onoff_probability_gating": (
                    onoff_gate_cfg if bool(onoff_gate_cfg.get("enabled", False)) else {"enabled": False}
                ),
            },
        )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_params.get("lr_reduce_factor", 0.1)),
        patience=int(train_params.get("lr_patience", 5)),
    )

    threshold_vec = np.array(
        [train_dataset.threshold_map[d] for d in device_list],
        dtype=np.float32,
    )
    output_mean = np.asarray(getattr(output_scaler, "mean_", np.zeros(len(device_list), dtype=np.float32)))
    output_scale = np.asarray(getattr(output_scaler, "scale_", np.ones(len(device_list), dtype=np.float32)))
    if output_mean.shape[0] != len(device_list):
        output_mean = np.zeros(len(device_list), dtype=np.float32)
    if output_scale.shape[0] != len(device_list):
        output_scale = np.ones(len(device_list), dtype=np.float32)

    output_mean_tensor = torch.tensor(output_mean, dtype=torch.float32, device=device)
    output_scale_tensor = torch.tensor(output_scale, dtype=torch.float32, device=device)
    onoff_threshold_tensor = torch.tensor(threshold_vec, dtype=torch.float32, device=device)

    participant_gating_cfg = train_params.get("participant_gating", {})
    participant_availability = train_dataset.build_participant_availability(
        min_active_points=int(participant_gating_cfg.get("min_active_points", 20)),
        min_known_points=int(participant_gating_cfg.get("min_known_points", 200)),
    )

    if participant_availability:
        print("Participant-device availability (from train data):")
        for participant, devices in participant_availability.items():
            enabled = [d for d, available in devices.items() if available]
            print(f"  {participant}: {enabled}")

    metadata = {
        "window_size": window_size,
        "device_list": device_list,
        "model": model_cfg,
        "active_thresholds": train_dataset.threshold_map,
        "participant_device_availability": participant_availability,
        "participant_gating": {
            "enabled": bool(participant_gating_cfg.get("enabled", True)),
            "unknown_participant_behavior": str(
                participant_gating_cfg.get("unknown_participant_behavior", "allow")
            ).lower(),
            "min_active_points": int(participant_gating_cfg.get("min_active_points", 20)),
            "min_known_points": int(participant_gating_cfg.get("min_known_points", 200)),
        },
    }

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_scaler=output_scaler,
        device_list=device_list,
        threshold_vec=threshold_vec,
        paths=paths,
        metadata=metadata,
        input_scaler=input_scaler,
        device=device,
        epochs=epochs,
        patience=patience,
        active_loss_weight=float(train_params.get("active_loss_weight", 2.0)),
        device_weights=device_weight_tensor,
        onoff_bce_weight=onoff_bce_weight,
        onoff_temperature=onoff_temperature,
        onoff_warmup_epochs=onoff_warmup_epochs,
        onoff_ramp_epochs=onoff_ramp_epochs,
        output_scale=output_scale_tensor,
        output_mean=output_mean_tensor,
        onoff_thresholds=onoff_threshold_tensor,
        onoff_pos_weight=onoff_pos_weight_tensor,
        regression_on_only=regression_on_only,
        max_train_batches=train_params.get("max_train_batches"),
        max_val_batches=train_params.get("max_val_batches"),
        validation_post_cfg=validation_post_cfg,
        validation_report_per_participant=bool(train_params.get("validation_report_per_participant", True)),
        early_stop_metric=train_params.get("early_stop_metric", "val_loss"),
        early_stop_mode=train_params.get("early_stop_mode"),
        no_progress=bool(train_params.get("no_progress", False)),
    )


if __name__ == "__main__":
    print("Please run the program using run.py with a config file.")
