"""
Luxic Physics-Grounded DAE Telemetry Simulator & Adversarial Stress Generator.
Generates 5-channel multivariate substation/industrial streams governed by:
  - IEEE C57.91 non-linear top-oil thermal ODE: dT/dt = (T_ss(L, T_amb) - T) / tau
  - Dissolved Gas Analysis (DGA H2/C2H2) thermal degradation coupling
  - ISO 10816 turbine/compressor bearing vibration RMS
  - Dynamic past-future tariff & heatwave covariate schedule
"""
from typing import Dict, List, Optional
import numpy as np
from luxic.config import LuxicConfig, TYPED_DECISIONS


def evaluate_ground_truth_action(
    horizon_temp: np.ndarray,
    horizon_load: np.ndarray,
    horizon_dga: np.ndarray,
    horizon_vib: np.ndarray,
    horizon_cov: np.ndarray,
    cfg: LuxicConfig,
) -> int:
    """Deterministic physical ground-truth optimal action over the 64-step horizon."""
    max_t = float(np.max(horizon_temp))
    max_l = float(np.max(horizon_load))
    max_dga = float(np.max(horizon_dga))
    max_vib = float(np.max(horizon_vib))
    max_cov = float(np.max(horizon_cov))

    if max_vib > cfg.iso_10816_zone_d_mms:
        return 3  # HARMONIC_DAMPER_ENGAGE
    if max_t > cfg.ieee_c5791_isolate_temp_c and max_dga > cfg.dga_fault_ppm:
        return 4  # EMERGENCY_ISOLATE_BREAKER
    if max_t > cfg.ieee_c5791_shed_temp_c:
        return 2  # LOAD_SHED_THERMAL_PROTECT
    if max_cov >= cfg.arbitrage_cov_threshold or max_l > cfg.arbitrage_load_threshold_mw:
        return 1  # BESS_DISPATCH_ARBITRAGE
    return 0  # NOMINAL_HOLD


def generate_dae_dataset(
    n_samples: int = 200,
    mode: str = "clean",
    seed: int = 42,
    cfg: Optional[LuxicConfig] = None,
) -> List[Dict]:
    """
    Generate multivariate DAE telemetry dataset across 6 operational/adversarial regimes:
      clean | regime_shift | sensor_dropout | harmonic_spikes | covariate_shock | mixed
    """
    cfg = cfg or LuxicConfig()
    rng = np.random.default_rng(seed)
    seq_len = cfg.context_len
    pred_len = cfg.pred_len
    total_len = seq_len + pred_len
    t = np.linspace(0.0, 4.0 * np.pi, total_len)

    modes_cycle = ["clean", "regime_shift", "sensor_dropout", "harmonic_spikes", "covariate_shock"]
    dataset: List[Dict] = []

    for i in range(n_samples):
        active_mode = modes_cycle[i % len(modes_cycle)] if mode == "mixed" else mode
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        base_load = float(rng.uniform(85.0, 140.0))

        # Dynamic past-future scheduled peak tariff / heatwave covariate
        cov_schedule = np.zeros(total_len, dtype=np.float64)
        peak_starts = [int(seq_len * 0.35), int(seq_len * 0.72), seq_len + 10, seq_len + 38]
        for ps in peak_starts:
            if ps + 10 < total_len:
                cov_schedule[ps : ps + 10] = float(rng.uniform(1.0, 2.0))

        ambient = 28.0 + 8.5 * np.sin(t + phase) + 4.0 * cov_schedule + rng.normal(0.0, 0.5, total_len)
        load_mw = (
            base_load
            + 24.0 * np.sin(t + phase - 0.3)
            + 12.0 * np.cos(2.0 * t)
            + 19.5 * cov_schedule
            + rng.normal(0.0, 1.5, total_len)
        )

        # Solve IEEE C57.91 thermal ODE
        top_oil = np.zeros(total_len, dtype=np.float64)
        lr0 = max(load_mw[0] / cfg.rated_transformer_load_mw, 0.3)
        top_oil[0] = ambient[0] + 38.0 * (lr0 ** 1.6)
        for k in range(1, total_len):
            lr = max(load_mw[k] / cfg.rated_transformer_load_mw, 0.3)
            ss = ambient[k] + 42.0 * (lr ** 1.6)
            top_oil[k] = top_oil[k - 1] + (ss - top_oil[k - 1]) / cfg.thermal_time_constant + rng.normal(0.0, 0.3)

        dga_ppm = 42.0 + 0.38 * np.maximum(top_oil - 74.0, 0.0) ** 1.4 + rng.normal(0.0, 0.8, total_len)
        vib_mms = 1.8 + 0.015 * load_mw + rng.normal(0.0, 0.15, total_len)

        inp_temp = top_oil.copy()
        inp_load = load_mw.copy()
        inp_dga = dga_ppm.copy()
        inp_vib = vib_mms.copy()
        inp_amb = ambient.copy()

        if active_mode == "regime_shift":
            shift_idx = seq_len - 24
            inp_load[shift_idx:] += 42.0
            for k in range(shift_idx, total_len):
                lr = max(inp_load[k] / cfg.rated_transformer_load_mw, 0.3)
                ss = inp_amb[k] + 42.0 * (lr ** 1.6)
                inp_temp[k] = inp_temp[k - 1] + (ss - inp_temp[k - 1]) / cfg.thermal_time_constant
            inp_dga = 42.0 + 0.38 * np.maximum(inp_temp - 74.0, 0.0) ** 1.4 + rng.normal(0.0, 0.8, total_len)
            top_oil, load_mw, dga_ppm = inp_temp.copy(), inp_load.copy(), inp_dga.copy()

        elif active_mode == "sensor_dropout":
            drop_temp = rng.random(seq_len) < 0.25
            drop_load = rng.random(seq_len) < 0.20
            inp_temp[:seq_len][drop_temp] = np.nan
            inp_load[:seq_len][drop_load] = 0.0

        elif active_mode == "harmonic_spikes":
            spike_pos = rng.choice(seq_len - 10, size=6, replace=False)
            inp_temp[spike_pos] += rng.choice([-1.0, 1.0], size=6) * 280.0
            inp_load[spike_pos] += rng.choice([-1.0, 1.0], size=6) * 650.0
            inp_vib[seq_len - 32 :] += 5.2
            vib_mms = inp_vib.copy()

        elif active_mode == "covariate_shock":
            cov_schedule[seq_len + 12 : seq_len + 28] = 3.2
            inp_amb = 28.0 + 8.5 * np.sin(t + phase) + 4.0 * cov_schedule + rng.normal(0.0, 0.4, total_len)
            for k in range(seq_len, total_len):
                inp_load[k] = (
                    base_load
                    + 24.0 * np.sin(t[k] + phase - 0.3)
                    + 12.0 * np.cos(2.0 * t[k])
                    + 19.5 * cov_schedule[k]
                    + rng.normal(0.0, 1.2)
                )
                lr = max(inp_load[k] / cfg.rated_transformer_load_mw, 0.3)
                ss = inp_amb[k] + 42.0 * (lr ** 1.6)
                inp_temp[k] = inp_temp[k - 1] + (ss - inp_temp[k - 1]) / cfg.thermal_time_constant
            inp_dga = 42.0 + 0.38 * np.maximum(inp_temp - 74.0, 0.0) ** 1.4 + rng.normal(0.0, 0.8, total_len)
            top_oil, load_mw, dga_ppm = inp_temp.copy(), inp_load.copy(), inp_dga.copy()

        true_action = evaluate_ground_truth_action(
            top_oil[seq_len:],
            load_mw[seq_len:],
            dga_ppm[seq_len:],
            vib_mms[seq_len:],
            cov_schedule[seq_len:],
            cfg,
        )

        x_context = np.vstack([
            inp_temp[:seq_len],
            inp_load[:seq_len],
            inp_dga[:seq_len],
            inp_vib[:seq_len],
            inp_amb[:seq_len],
        ])

        dataset.append({
            "mode": active_mode,
            "x_context": x_context,               # [5, 256]
            "cov_full": cov_schedule.copy(),       # [320]
            "y_true_temp": top_oil[seq_len:],      # [64]
            "y_true_load": load_mw[seq_len:],      # [64]
            "y_true_dga": dga_ppm[seq_len:],       # [64]
            "y_true_vib": vib_mms[seq_len:],       # [64]
            "true_decision": true_action,
            "true_decision_label": TYPED_DECISIONS[true_action],
        })

    return dataset
