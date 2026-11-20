"""
Luxic Platform Configuration.
Supports both the full 330M-parameter H100 pre-training profile and the runnable
in-VM Edge-Core profile that shares the exact same mathematical pipeline.
"""
from dataclasses import dataclass, field
from typing import List


TYPED_DECISIONS: List[str] = [
    "NOMINAL_HOLD",
    "BESS_DISPATCH_ARBITRAGE",
    "LOAD_SHED_THERMAL_PROTECT",
    "HARMONIC_DAMPER_ENGAGE",
    "EMERGENCY_ISOLATE_BREAKER",
]

QUANTILES: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass(frozen=True)
class LuxicConfig:
    model_name: str = "luxic-330m-v3"
    num_channels: int = 5
    context_len: int = 256
    patch_len: int = 32
    pred_len: int = 64
    d_model: int = 64              # 1024 in 330M H100 DDP profile
    num_layers: int = 2            # 24 alternating layers in 330M H100 DDP profile
    num_heads: int = 4             # 16 heads in 330M H100 DDP profile
    num_decisions: int = 5
    num_quantiles: int = 9
    mad_spike_threshold: float = 4.8
    mad_abs_floor: float = 45.0
    cusum_shift_threshold_mw: float = 6.0
    rated_transformer_load_mw: float = 135.0
    thermal_time_constant: float = 6.5
    ieee_c5791_shed_temp_c: float = 96.5
    ieee_c5791_isolate_temp_c: float = 106.0
    dga_fault_ppm: float = 56.0
    iso_10816_zone_d_mms: float = 6.2
    arbitrage_cov_threshold: float = 1.5
    arbitrage_load_threshold_mw: float = 152.0
    conformal_target_alpha: float = 0.20  # 80% P10-P90 nominal coverage
    quantiles: List[float] = field(default_factory=lambda: list(QUANTILES))

    @property
    def num_patches(self) -> int:
        if self.context_len % self.patch_len != 0:
            raise ValueError("context_len must be divisible by patch_len")
        return self.context_len // self.patch_len
