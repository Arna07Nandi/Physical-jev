# Luxic-330M: Zero-Shot Vertical Foundation Model Platform

Built from scratch (Epoch 0) for industrial manufacturing, substation telemetry, and battery storage (BESS) operations.

## Architecture & Pipeline
1. **Physics-Grounded DAE Telemetry (`luxic/data/dae_simulator.py`)**: 5-channel multivariate substation generator governed by IEEE C57.91 thermal ODEs, DGA gas dynamics, ISO 10816 bearing vibration, and dynamic future tariff covariates across 6 adversarial regimes.
2. **Industrial Preprocessing Guards (`luxic/preprocessing/guards.py`)**: Vectorized 4.8x MAD EMI spike filter, cross-variate NaN/zero imputation gate, Huber-Median R-RevIN, and Intra-Context CUSUM step-shift detector.
3. **2D Alternating Attention Backbone & RLCD Decision Head (`luxic/model/backbone.py`, `luxic/model/torch_330m_ddp.py`)**: 32-step patching, alternating causal temporal and full cross-variate attention, single-pass 64-step 9-quantile decoding, AdamW trainer from Epoch 0, and symmetric INT8 edge quantization.
4. **Verifiable Physics Rewards & Split-Conformal Calibration (`luxic/rewards/physics_verifier.py`, `luxic/calibration/conformal.py`)**: Deterministic reward verifier + finite-sample P10-P90 conformal interval calibration + NLL temperature scaling (`ECE < 0.04`).
5. **Production Serving (`luxic/serving/api.py`, `luxic/serving/iec61850_encoder.py`, `Dockerfile`)**: FastAPI microservice (`/health`, `/v1/decide`, `/v1/decide/goose`) with deterministic IEC 61850 GOOSE relay frames and CRC-16 checksums.

## Quickstart
```bash
# 1. Train from Epoch 0, calibrate, quantize to INT8, and benchmark all 6 stress regimes
python3 scripts/train_and_verify.py

# 2. Run full unit, integration, adversarial, INT8 fidelity, and FastAPI tests
python3 -m unittest discover -s tests -v

# 3. Launch production edge/cloud API
uvicorn luxic.serving.api:app --host 0.0.0.0 --port 8080
```
