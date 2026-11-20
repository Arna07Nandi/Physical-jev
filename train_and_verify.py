"""
Luxic-330M Complete Training, Calibration, INT8 Quantization & Head-to-Head Benchmark Script.
Compares Luxic-v1 (Naive Baseline) vs Luxic-v3 (FP64) vs Luxic-v3-INT8 (Edge Quantized)
across all 6 operational & adversarial stress regimes, saving both JSON and Markdown reports.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from luxic.config import LuxicConfig
from luxic.data.dae_simulator import generate_dae_dataset
from luxic.model.backbone import LuxicFoundationModel
from luxic.model.baselines import LuxicV1NaiveBaseline


def main():
    os.makedirs("artifacts", exist_ok=True)
    cfg = LuxicConfig()

    print("[1/5] Generating physics-grounded DAE datasets (train=600, cal=200, test=120/regime)...")
    train_ds = generate_dae_dataset(n_samples=600, mode="mixed", seed=101, cfg=cfg)
    cal_ds = generate_dae_dataset(n_samples=200, mode="mixed", seed=202, cfg=cfg)

    print("[2/5] Initializing LuxicFoundationModel from random weights (Epoch 0) and training with AdamW + RLCD...")
    model = LuxicFoundationModel(cfg=cfg, seed=42)
    train_summary = model.fit_from_scratch(train_ds, cal_ds, epochs=220, lr=0.008)
    print(f"      Epoch 0 Loss: {train_summary['epoch_0_loss']:.4f} (Acc: {train_summary['epoch_0_acc_pct']}%)")
    print(f"      Final Loss:   {train_summary['final_loss']:.4f} (Train Acc: {train_summary['final_train_acc_pct']}%)")
    print(f"      Conformal Calibration: {train_summary['calibration']}")

    print("[3/5] Exporting symmetric per-tensor INT8 edge weights to artifacts/luxic_330m_v3_int8.npz...")
    weight_path = "artifacts/luxic_330m_v3_int8.npz"
    quant_stats = model.export_int8_artifact(weight_path)
    print(f"      Quantization stats: {quant_stats}")

    print("[4/5] Benchmarking Luxic-v1 (Broken) vs Luxic-v3 (FP64) vs Luxic-v3-INT8 across all 6 regimes...")
    v1_baseline = LuxicV1NaiveBaseline(cfg=cfg)
    edge_model = LuxicFoundationModel.load_int8_artifact(weight_path, cfg=cfg)
    regimes = ["clean", "regime_shift", "sensor_dropout", "harmonic_spikes", "covariate_shock", "mixed"]

    report = {
        "training": train_summary,
        "quantization": quant_stats,
        "v1_naive_eval": {},
        "fp64_eval": {},
        "int8_edge_eval": {},
    }

    md_lines = [
        "# Luxic-330M Verified Benchmark Report",
        "",
        f"- **Training From Epoch 0:** Loss `{train_summary['epoch_0_loss']:.4f}` → `{train_summary['final_loss']:.4f}` | Train Acc `{train_summary['epoch_0_acc_pct']}%` → `{train_summary['final_train_acc_pct']}%`",
        f"- **Split-Conformal Calibration:** `q_hat = {train_summary['calibration']['q_hat_temp_c']}°C`, `T = {train_summary['calibration']['optimal_temperature']}`, Calibration `ECE = {train_summary['calibration']['cal_ece']}`",
        f"- **INT8 Edge Compression:** `{quant_stats['fp64_bytes']:,}` bytes → `{quant_stats['int8_bytes']:,}` bytes (`{quant_stats['compression_ratio']}x` compression)",
        "",
        "| Stress Regime | v1 Temp MAE | v3-INT8 Temp MAE | v1 Dec Acc | v3-FP64 Acc | v3-INT8 Acc | v1 ECE | v3-INT8 ECE | v3 P10–P90 Cov | Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for idx, mode in enumerate(regimes):
        test_ds = generate_dae_dataset(n_samples=120, mode=mode, seed=300 + idx, cfg=cfg)
        m_v1 = v1_baseline.evaluate(test_ds)
        m_fp64 = model.evaluate(test_ds)
        m_int8 = edge_model.evaluate(test_ds)

        report["v1_naive_eval"][mode] = m_v1
        report["fp64_eval"][mode] = m_fp64
        report["int8_edge_eval"][mode] = m_int8

        print(
            f"  [{mode.upper():15s}] "
            f"Temp MAE: {m_v1['mae_temp_c']:5.2f}→{m_int8['mae_temp_c']:4.2f}°C | "
            f"Dec Acc: {m_v1['decision_accuracy_pct']:4.1f}%→{m_int8['decision_accuracy_pct']:5.1f}% | "
            f"ECE: {m_v1['ece']:.3f}→{m_int8['ece']:.4f} | "
            f"Cov: {m_int8['p10_p90_coverage_pct']:4.1f}% | "
            f"Lat: {m_int8['mean_latency_ms']:.2f}ms"
        )
        md_lines.append(
            f"| **{mode}** | {m_v1['mae_temp_c']:.2f}°C | **{m_int8['mae_temp_c']:.2f}°C** | "
            f"{m_v1['decision_accuracy_pct']:.1f}% | {m_fp64['decision_accuracy_pct']:.1f}% | "
            f"**{m_int8['decision_accuracy_pct']:.1f}%** | {m_v1['ece']:.4f} | **{m_int8['ece']:.4f}** | "
            f"{m_int8['p10_p90_coverage_pct']:.1f}% | {m_int8['mean_latency_ms']:.2f} ms |"
        )

    with open("artifacts/verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("artifacts/BENCHMARK_RESULTS.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("[5/5] Saved artifacts/verification_report.json and artifacts/BENCHMARK_RESULTS.md.")


if __name__ == "__main__":
    main()
