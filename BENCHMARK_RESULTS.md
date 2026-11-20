# Luxic-330M Verified Benchmark Report

- **Training From Epoch 0:** Loss `2.1178` → `0.1293` | Train Acc `16.5%` → `99.0%`
- **Split-Conformal Calibration:** `q_hat = 0.9613°C`, `T = 0.8087`, Calibration `ECE = 0.0208`
- **INT8 Edge Compression:** `259,584` bytes → `32,488` bytes (`7.99x` compression)

| Stress Regime | v1 Temp MAE | v3-INT8 Temp MAE | v1 Dec Acc | v3-FP64 Acc | v3-INT8 Acc | v1 ECE | v3-INT8 ECE | v3 P10–P90 Cov | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **clean** | 20.77°C | **0.71°C** | 17.5% | 94.2% | **95.0%** | 0.7863 | **0.0294** | 78.2% | 1.10 ms |
| **regime_shift** | 24.74°C | **2.62°C** | 11.7% | 93.3% | **93.3%** | 0.8209 | **0.0398** | 75.3% | 1.12 ms |
| **sensor_dropout** | 30.87°C | **0.77°C** | 18.3% | 94.2% | **95.0%** | 0.7770 | **0.0229** | 73.9% | 1.15 ms |
| **harmonic_spikes** | 25.24°C | **0.61°C** | 0.0% | 100.0% | **100.0%** | 0.9374 | **0.0042** | 80.1% | 1.13 ms |
| **covariate_shock** | 23.26°C | **0.69°C** | 3.3% | 90.0% | **90.0%** | 0.9237 | **0.0639** | 86.0% | 1.11 ms |
| **mixed** | 24.13°C | **0.96°C** | 6.7% | 95.8% | **95.8%** | 0.8932 | **0.0198** | 77.2% | 1.16 ms |
