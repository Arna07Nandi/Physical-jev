"""
Luxic-330M End-to-End Training, Split-Conformal Calibration, INT8 Quantization & 6-Regime Verification Script.
Trains the model from random initialization (epoch 0) with zero pre-loaded weights.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from luxic.config import LuxicConfig
from luxic.data.dae_simulator import generate_dae_dataset
from luxic.model.backbone import LuxicFoundationModel


def main():
    os.makedirs("artifacts", exist_ok=True)
    cfg = LuxicConfig()

    print("[1/5] Generating physics-grounded DAE datasets (train=600, cal=200, test=120/regime)...")
    train_ds = generate_dae_dataset(n_samples=600, mode="mixed", seed=101, cfg=cfg)
    cal_ds = generate_dae_dataset(n_samples=200, mode="mixed", seed=202, cfg=cfg)

    print("[2/5] Initializing LuxicFoundationModel from random weights (Epoch 0) and training with AdamW + RLCD...")
    model = LuxicFoundationModel(cfg=cfg, seed=42)
    train_summary = model.fit_from_scratch(train_ds, cal_ds, epochs=220, lr=0.008, weight_decay=1e-4)
    print(f"      Epoch 0 Loss: {train_summary['epoch_0_loss']:.4f} (Acc: {train_summary['epoch_0_acc_pct']}%)")
    print(f"      Final Loss:   {train_summary['final_loss']:.4f} (Train Acc: {train_summary['final_train_acc_pct']}%)")
    print(f"      Conformal Calibration: {train_summary['calibration']}")

    print("[3/5] Exporting symmetric per-tensor INT8 edge weights to artifacts/luxic_330m_v3_int8.npz...")
    weight_path = "artifacts/luxic_330m_v3_int8.npz"
    quant_stats = model.export_int8_artifact(weight_path)
    print(f"      Quantization stats: {quant_stats}")

    print("[4/5] Reloading INT8 quantized edge runtime and running 6-regime adversarial verification...")
    edge_model = LuxicFoundationModel.load_int8_artifact(weight_path, cfg=cfg)
    regimes = ["clean", "regime_shift", "sensor_dropout", "harmonic_spikes", "covariate_shock", "mixed"]
    report = {
        "training": train_summary,
        "quantization": quant_stats,
        "fp64_eval": {},
        "int8_edge_eval": {},
    }

    for idx, mode in enumerate(regimes):
        test_ds = generate_dae_dataset(n_samples=120, mode=mode, seed=300 + idx, cfg=cfg)
        m_fp64 = model.evaluate(test_ds)
        m_int8 = edge_model.evaluate(test_ds)
        report["fp64_eval"][mode] = m_fp64
        report["int8_edge_eval"][mode] = m_int8
        print(
            f"  [{mode.upper():15s}] "
            f"Temp MAE: {m_int8['mae_temp_c']:5.2f}°C | "
            f"Load MAE: {m_int8['mae_load_mw']:5.2f} MW | "
            f"Dec Acc (FP64/INT8): {m_fp64['decision_accuracy_pct']:5.1f}% / {m_int8['decision_accuracy_pct']:5.1f}% | "
            f"ECE: {m_int8['ece']:.4f} | "
            f"Cov(80%): {m_int8['p10_p90_coverage_pct']:4.1f}% | "
            f"Lat: {m_int8['mean_latency_ms']:.2f}ms"
        )

    with open("artifacts/verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[5/5] Saved artifacts/verification_report.json successfully.")


if __name__ == "__main__":
    main()
