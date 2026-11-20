"""
Luxic Industrial Preprocessing Guards:
1. Vectorized 4.8x MAD EMI Spike Suppressor & Cross-Variate NaN/Zero Imputation Gate
2. Robust Huber-Median Reversible Instance Normalization (R-RevIN)
3. Intra-Context CUSUM Regime Step-Shift Localization Gate
"""
from typing import Dict, Tuple
import numpy as np
from luxic.config import LuxicConfig


class TelemetryGuardPipeline:
    def __init__(self, cfg: LuxicConfig):
        self.cfg = cfg

    def clean_and_impute(self, x_context: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Detects NaNs, physically impossible zero-dropouts (on Temp/Load/DGA),
        and >4.8 MAD EMI transients, then repairs via linear interpolation.
        Returns cleaned [5, T] array and diagnostics counts.
        """
        x_fixed = np.array(x_context, dtype=np.float64, copy=True)
        if x_fixed.ndim != 2 or x_fixed.shape[0] != self.cfg.num_channels:
            raise ValueError(
                f"Expected x_context shape ({self.cfg.num_channels}, T), got {x_fixed.shape}"
            )

        n_dropouts = 0
        n_spikes = 0

        for c in range(self.cfg.num_channels):
            s = x_fixed[c]
            invalid = np.isnan(s) | np.isinf(s) | ((s == 0.0) & (c in (0, 1, 2)))
            n_dropouts += int(np.sum(invalid))

            valid_vals = s[~invalid]
            if len(valid_vals) == 0:
                fallback = [75.0, 110.0, 45.0, 3.0, 28.0][c]
                s[:] = fallback
                x_fixed[c] = s
                continue

            med = float(np.median(valid_vals))
            mad = float(np.median(np.abs(valid_vals - med))) + 1e-4
            thresh = max(self.cfg.mad_spike_threshold * mad, self.cfg.mad_abs_floor)
            spikes = (~invalid) & (np.abs(s - med) > thresh)
            n_spikes += int(np.sum(spikes))

            bad = invalid | spikes
            if np.any(bad):
                good_idx = np.where(~bad)[0]
                bad_idx = np.where(bad)[0]
                if len(good_idx) >= 2:
                    s[bad_idx] = np.interp(bad_idx, good_idx, s[good_idx])
                else:
                    s[bad_idx] = med
            x_fixed[c] = s

        return x_fixed, {"repaired_dropouts": n_dropouts, "suppressed_emi_spikes": n_spikes}

    def robust_revin_forward(self, x_fixed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Huber-Median / IQR Reversible Instance Normalization (immune to outliers)."""
        loc = np.median(x_fixed, axis=1, keepdims=True)
        q75 = np.percentile(x_fixed, 75, axis=1, keepdims=True)
        q25 = np.percentile(x_fixed, 25, axis=1, keepdims=True)
        scale = (q75 - q25) / 1.349 + 1e-4
        x_norm = (x_fixed - loc) / scale
        return x_norm, loc, scale

    def detect_cusum_step_shift(
        self,
        load_series: np.ndarray,
        ambient_series: np.ndarray,
        cov_full: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fits a fast ridge-regularized diurnal harmonic + dynamic covariate lookahead basis
        on the stable pre-tail window [0 : T-32], then measures median tail residual over
        the final 20 steps to detect and isolate late-context feeder step-shifts.
        """
        seq_len = len(load_series)
        pred_len = self.cfg.pred_len
        t_hist = np.arange(seq_len, dtype=np.float64)
        t_fut = np.arange(seq_len, seq_len + pred_len, dtype=np.float64)
        omega = 4.0 * np.pi / float(seq_len + pred_len)

        cov_hist = cov_full[:seq_len]
        cov_fut = cov_full[seq_len : seq_len + pred_len]
        stable_end = max(seq_len - 32, 32)

        def build_design(t_vec: np.ndarray, cov_vec: np.ndarray) -> np.ndarray:
            return np.column_stack([
                np.ones_like(t_vec),
                np.sin(omega * t_vec),
                np.cos(omega * t_vec),
                np.sin(2.0 * omega * t_vec),
                np.cos(2.0 * omega * t_vec),
                cov_vec,
            ])

        X_stable = build_design(t_hist[:stable_end], cov_hist[:stable_end])
        X_tail = build_design(t_hist[stable_end:], cov_hist[stable_end:])
        X_fut = build_design(t_fut, cov_fut)

        reg = 1e-3 * np.eye(X_stable.shape[1])
        w_load = np.linalg.solve(X_stable.T @ X_stable + reg, X_stable.T @ load_series[:stable_end])
        w_amb = np.linalg.solve(X_stable.T @ X_stable + reg, X_stable.T @ ambient_series[:stable_end])

        tail_len = min(20, len(X_tail))
        tail_residuals = load_series[-tail_len:] - (X_tail[-tail_len:] @ w_load)
        step_shift_mw = float(np.median(tail_residuals))
        if abs(step_shift_mw) < self.cfg.cusum_shift_threshold_mw:
            step_shift_mw = 0.0

        pred_load_prior = (X_fut @ w_load) + step_shift_mw
        pred_amb_prior = X_fut @ w_amb
        return pred_load_prior, pred_amb_prior, step_shift_mw
