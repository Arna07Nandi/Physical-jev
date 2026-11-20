"""
Luxic-330M Production Edge/Cloud FastAPI Inference Microservice.
Endpoints:
  - GET  /health           : Readiness & INT8 weight status
  - POST /v1/decide        : Single-pass zero-shot multivariate forecast + calibrated typed decision
  - POST /v1/decide/goose  : Deterministic IEC 61850 GOOSE substation relay frame
"""
import os
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from luxic.config import LuxicConfig, TYPED_DECISIONS
from luxic.model.backbone import LuxicFoundationModel
from luxic.serving.iec61850_encoder import IEC61850GooseEncoder


class TelemetryRequest(BaseModel):
    asset_id: str = Field(..., example="SUB01_TR1_ERCOT")
    context_5ch: List[List[Optional[float]]] = Field(
        ..., description="5 channels x 256 context steps (null/NaN permitted; auto-imputed)"
    )
    cov_full: List[float] = Field(
        ..., description="320-step dynamic covariate array (256 history + 64 known future)"
    )


class DecisionResponse(BaseModel):
    asset_id: str
    model: str
    forward_pass_ms: float
    typed_decision: str
    calibrated_confidence: float
    action_probabilities: Dict[str, float]
    cusum_step_shift_mw: float
    guard_diagnostics: Dict[str, int]
    horizon_64_top_oil_p50_c: List[float]
    horizon_64_top_oil_p10_c: List[float]
    horizon_64_top_oil_p90_c: List[float]
    horizon_64_load_p50_mw: List[float]


def create_app(weights_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(
        title="Luxic-330M Edge & Cloud Inference API",
        version="3.0.0",
        description="Zero-Shot Multivariate Time-Series & Calibrated Typed Decision Foundation Model",
    )
    cfg = LuxicConfig()
    goose_encoder = IEC61850GooseEncoder()

    if weights_path and os.path.exists(weights_path):
        model = LuxicFoundationModel.load_int8_artifact(weights_path, cfg=cfg)
    else:
        model = LuxicFoundationModel(cfg=cfg)

    app.state.model = model
    app.state.cfg = cfg
    app.state.goose = goose_encoder

    @app.get("/health")
    def health():
        m: LuxicFoundationModel = app.state.model
        return {
            "status": "ok",
            "model": m.cfg.model_name,
            "trained": m.is_trained,
            "int8_quantized": m.is_int8_quantized,
            "typed_actions": TYPED_DECISIONS,
        }

    def _run_inference(req: TelemetryRequest) -> Dict:
        m: LuxicFoundationModel = app.state.model
        if len(req.context_5ch) != m.cfg.num_channels:
            raise HTTPException(status_code=422, detail=f"Expected {m.cfg.num_channels} channels")
        if any(len(ch) != m.cfg.context_len for ch in req.context_5ch):
            raise HTTPException(status_code=422, detail=f"Each channel must have {m.cfg.context_len} steps")
        if len(req.cov_full) != m.cfg.context_len + m.cfg.pred_len:
            raise HTTPException(
                status_code=422,
                detail=f"cov_full must have {m.cfg.context_len + m.cfg.pred_len} steps",
            )

        x_arr = np.array(
            [[np.nan if v is None else float(v) for v in ch] for ch in req.context_5ch],
            dtype=np.float64,
        )
        cov_arr = np.array(req.cov_full, dtype=np.float64)
        return m.predict({"x_context": x_arr, "cov_full": cov_arr})

    @app.post("/v1/decide", response_model=DecisionResponse)
    def decide(req: TelemetryRequest):
        out = _run_inference(req)
        return DecisionResponse(
            asset_id=req.asset_id,
            model=out["model"],
            forward_pass_ms=out["latency_ms"],
            typed_decision=out["decision_label"],
            calibrated_confidence=round(out["confidence"], 4),
            action_probabilities=out["probabilities"],
            cusum_step_shift_mw=out["step_shift_detected_mw"],
            guard_diagnostics=out["guard_diagnostics"],
            horizon_64_top_oil_p50_c=[round(float(x), 2) for x in out["pred_temp"]],
            horizon_64_top_oil_p10_c=[round(float(x), 2) for x in out["temp_quantiles"][0]],
            horizon_64_top_oil_p90_c=[round(float(x), 2) for x in out["temp_quantiles"][-1]],
            horizon_64_load_p50_mw=[round(float(x), 2) for x in out["pred_load"]],
        )

    @app.post("/v1/decide/goose")
    def decide_goose(req: TelemetryRequest):
        out = _run_inference(req)
        frame = app.state.goose.encode_decision(out)
        frame["asset_id"] = req.asset_id
        return frame

    return app


app = create_app(weights_path=os.environ.get("LUXIC_WEIGHTS_PATH", "artifacts/luxic_330m_v3_int8.npz"))
