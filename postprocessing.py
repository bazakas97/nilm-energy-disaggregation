import numpy as np


def advanced_postprocess_predictions(
    predictions,
    min_duration=0,
    min_energy_value=0,
    cycle_peak_min=None,
):
    """
    Post-processing for cycle-level denoising:
      1) Zero out negatives.
      2) Form ON blocks using `min_energy_value` as ON threshold.
      3) Drop a whole ON block if:
         - block duration < min_duration, or
         - block peak < cycle_peak_min (if provided).

    """
    x = np.asarray(predictions, dtype=np.float32).copy()
    x[x < 0] = 0.0

    on_th = float(min_energy_value)
    on_mask = x > on_th
    if not on_mask.any():
        x[:] = 0.0
        return x

    min_dur = max(0, int(min_duration))
    peak_th = float(cycle_peak_min) if cycle_peak_min is not None else None

    keep = np.zeros_like(on_mask, dtype=bool)
    padded = np.concatenate(([False], on_mask, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    for s, e in zip(starts, ends):
        dur = int(e - s)
        if min_dur > 0 and dur < min_dur:
            continue
        if peak_th is not None and float(x[s:e].max()) < peak_th:
            continue
        keep[s:e] = True

    x[~keep] = 0.0
    return x


def _contiguous_segments(participants, n_rows):
    if participants is None:
        return [(0, n_rows)]
    if len(participants) != n_rows:
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


def _apply_min_on_duration(state, min_on_duration):
    if min_on_duration <= 1:
        return state

    result = state.copy()
    padded = np.concatenate(([False], result, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    for s, e in zip(starts, ends):
        if (e - s) < min_on_duration:
            result[s:e] = False
    return result


def _hysteresis_states(prob, on_threshold, off_threshold):
    n = len(prob)
    state = np.zeros(n, dtype=bool)
    is_on = False
    for i in range(n):
        p = float(prob[i])
        if is_on:
            if p <= off_threshold:
                is_on = False
        else:
            if p >= on_threshold:
                is_on = True
        state[i] = is_on
    return state


def apply_onoff_probability_gating(
    predictions,
    onoff_prob,
    device_list,
    participants=None,
    gating_cfg=None,
):
    """
    Zero power predictions when ON/OFF probability indicates OFF state.
    Uses hysteresis (on/off thresholds) and optional minimum ON duration.
    """
    if onoff_prob is None:
        return predictions, 0

    pred = np.asarray(predictions)
    prob = np.asarray(onoff_prob)
    if pred.shape != prob.shape:
        return predictions, 0

    cfg = gating_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return predictions, 0

    n_rows, n_dev = pred.shape
    gated = pred.copy()
    segments = _contiguous_segments(participants, n_rows)

    default_on = float(cfg.get("on_threshold", 0.6))
    default_off = float(cfg.get("off_threshold", 0.4))
    default_min_dur = int(cfg.get("min_on_duration", 0))
    device_params = cfg.get("device_params", cfg.get("device_thresholds", {})) or {}

    total_zeroed = 0
    for j in range(n_dev):
        dev = device_list[j] if j < len(device_list) else f"dev_{j}"
        dev_cfg = device_params.get(dev, {}) or {}
        on_th = float(dev_cfg.get("on_threshold", default_on))
        off_th = float(dev_cfg.get("off_threshold", default_off))
        min_dur = int(dev_cfg.get("min_on_duration", default_min_dur))

        on_th = max(0.0, min(1.0, on_th))
        off_th = max(0.0, min(1.0, off_th))
        if off_th > on_th:
            off_th = on_th

        keep_mask = np.zeros(n_rows, dtype=bool)
        for start, end in segments:
            seg_prob = prob[start:end, j]
            seg_state = _hysteresis_states(seg_prob, on_threshold=on_th, off_threshold=off_th)
            seg_state = _apply_min_on_duration(seg_state, min_on_duration=min_dur)
            keep_mask[start:end] = seg_state

        before = gated[:, j]
        zero_mask = ~keep_mask
        total_zeroed += int(np.count_nonzero((before > 0) & zero_mask))
        gated[zero_mask, j] = 0.0

    return gated, total_zeroed


def enforce_mains_power_budget(
    predictions,
    mains,
    mode="proportional",
):
    """
    Enforce simple physical consistency against mains:
      - proportional: if row sum exceeds mains, scale all device predictions proportionally.
      - clip_each: clip each device prediction to mains independently.
    """
    pred = np.asarray(predictions, dtype=np.float32).copy()
    mains_arr = np.asarray(mains, dtype=np.float32).reshape(-1)
    if pred.ndim != 2 or mains_arr.shape[0] != pred.shape[0]:
        return predictions, 0

    pred[pred < 0] = 0.0
    mains_arr = np.clip(mains_arr, 0.0, None)

    mode_norm = str(mode).strip().lower()
    adjusted_rows = 0

    if mode_norm == "clip_each":
        over_mask = pred > mains_arr.reshape(-1, 1)
        adjusted_rows = int(np.count_nonzero(over_mask.any(axis=1)))
        pred = np.minimum(pred, mains_arr.reshape(-1, 1))
        return pred, adjusted_rows

    # default: proportional
    row_sum = pred.sum(axis=1)
    over = row_sum > (mains_arr + 1e-6)
    adjusted_rows = int(np.count_nonzero(over))
    if adjusted_rows == 0:
        return pred, 0

    scale = np.ones_like(row_sum, dtype=np.float32)
    scale[over] = mains_arr[over] / np.maximum(row_sum[over], 1e-6)
    pred *= scale.reshape(-1, 1)
    return pred, adjusted_rows
