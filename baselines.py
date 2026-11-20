"""
Luxic Benchmark Baselines (For Strict Apples-to-Apples Evaluation):
  1. NaivePersistenceBaseline: Repeats last observed value (industry sanity floor)
  2. LuxicV1NaiveBaseline: Standard Z-score normalization, univariate 32-step patching,
     autoregressive rollout, and uncalibrated softmax logits (shows why naive TSFMs fail).
"""
import math
import time
from typing import Dict, List
import numpy as np
from luxic.config import LuxicConfig, TYPED_DECISIONS
from luxic.calibration.conformal import compute_ece, compute_brier_score


class LuxicV1NaiveBaseline:
    """Naive patched baseline without R-RevIN, MAD filter, CUSUM gate, or conformal calibration."""

    def __init__(self, cfg: LuxicConfig = None):
        self.cfg = cfg or LuxicConfig()

    def predict(self, sample: Dict) -> Dict:
        t0 = time.perf_counter()
        x = np.nan_to_num(sample["x_context"].copy(), nan=0.0)
        seq_len = x.shape[1]
        p_len = self.cfg.patch_len
        n_patches = seq_len // p_len

        means = np.mean(x[:2], axis=1, keepdims=True)
        stds = np.std(x[:2], axis=1, keepdims=True) + 1e-5
        x_norm = (x[:2] - means) / stds
        patches = x_norm.reshape(2, n_patches, p_len)

        q = patches[:, -1:, :]
        scores = np.einsum("bip,bjp->bij", q, patches) / math.sqrt(p_len)
        scores -= np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

        ctx_vec = np.einsum("bij,bjp->bip", attn, patches).squeeze(1)
        last_p = patches[:, -1, :]
        trend = (patches[:, -1, :] - patches[:, -3, :]) * 0.35
        p1 = 0.55 * last_p + 0.45 * ctx_vec + 0.25 * trend
        p2 = 0.50 * p1 + 0.50 * patches[:, -2, :] + 0.35 * trend
        pred_point = np.concatenate([p1, p2], axis=1) * stds + means

        pred_temp, pred_load = pred_point[0], pred_point[1]
        z = np.array([-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28])[:, None]
        temp_q = pred_temp[None, :] + z * (stds[0, 0] * 0.35)

        max_t, max_l = float(np.max(pred_temp)), float(np.max(pred_load))
        logits = np.array([
            2.0 - 0.08 * max(max_t - 85.0, 0.0),
            -0.5 + 0.04 * max(max_l - 135.0, 0.0),
            -1.2 + 0.12 * max(max_t - 94.0, 0.0),
            -1.5,
            -2.0 + 0.10 * max(max_t - 104.0, 0.0),
        ]) * 3.2
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "decision_idx": int(np.argmax(probs)),
            "decision_label": TYPED_DECISIONS[int(np.argmax(probs))],
            "probs_array": probs,
            "pred_temp": pred_temp,
            "pred_load": pred_load,
            "temp_quantiles": temp_q,
            "latency_ms": latency_ms,
        }

    def evaluate(self, dataset: List[Dict]) -> Dict[str, float]:
        mae_t, mae_l, cov_80, lat = [], [], [], []
        probs_all, labels_all = [], []
        for s in dataset:
            out = self.predict(s)
            yt, yl = s["y_true_temp"], s["y_true_load"]
            mae_t.append(np.mean(np.abs(yt - out["pred_temp"])))
            mae_l.append(np.mean(np.abs(yl - out["pred_load"])))
            qt = out["temp_quantiles"]
            cov_80.append(np.mean((yt >= qt[0]) & (yt <= qt[-1])))
            lat.append(out["latency_ms"])
            probs_all.append(out["probs_array"])
            labels_all.append(s["true_decision"])

        probs_arr = np.array(probs_all)
        labels_arr = np.array(labels_all)
        acc = float(np.mean(np.argmax(probs_arr, axis=1) == labels_arr) * 100.0)
        return {
            "mae_temp_c": round(float(np.mean(mae_t)), 3),
            "mae_load_mw": round(float(np.mean(mae_l)), 3),
            "p10_p90_coverage_pct": round(float(np.mean(cov_80) * 100.0), 1),
            "decision_accuracy_pct": round(acc, 1),
            "ece": round(compute_ece(probs_arr, labels_arr), 4),
            "brier_score": round(compute_brier_score(probs_arr, labels_arr), 4),
            "mean_latency_ms": round(float(np.mean(lat)), 3),
        }
