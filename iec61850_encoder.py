"""
Luxic Deterministic IEC 61850 GOOSE Substation Relay Frame Encoder.
Converts a calibrated Luxic typed decision into a deterministic substation PDU
with CRC-16/CCITT-FALSE integrity checksum for air-gapped relay gateways.
"""
import json
import struct
import time
from typing import Dict


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class IEC61850GooseEncoder:
    def __init__(self, gocb_ref: str = "SUB01_TR1/LLN0$GO$Luxic_Dispatch"):
        self.gocb_ref = gocb_ref
        self._st_num = 1000
        self._sq_num = 0
        self._last_action = None

    def encode_decision(self, inference_out: Dict) -> Dict:
        action = inference_out["decision_label"]
        if action != self._last_action:
            self._st_num += 1
            self._sq_num = 0
            self._last_action = action
        else:
            self._sq_num += 1

        max_p50 = round(float(max(inference_out["pred_temp"])), 2)
        max_p90 = round(float(max(inference_out["temp_quantiles"][-1])), 2)
        conf = round(float(inference_out["confidence"]), 4)
        shift_mw = round(float(inference_out["step_shift_detected_mw"]), 2)

        pdu = {
            "gocbRef": self.gocb_ref,
            "timeAllowedtoLiveMs": 4,
            "timestampUtcMs": int(time.time() * 1000),
            "stNum": self._st_num,
            "sqNum": self._sq_num,
            "simulation": False,
            "confRev": 3,
            "allData": {
                "typedActionEnum": action,
                "actionIndex": int(inference_out["decision_idx"]),
                "conformalConfidence": conf,
                "ieeeC5791_TopOilP50_C": max_p50,
                "ieeeC5791_TopOilP90_C": max_p90,
                "cusumStepShift_MW": shift_mw,
            },
        }

        # Compact binary wire frame: action_idx (B), conf (f), p50 (f), p90 (f), stNum (I) + CRC16 (H)
        body = struct.pack(">BfffI", int(inference_out["decision_idx"]), conf, max_p50, max_p90, self._st_num)
        checksum = crc16_ccitt(body)
        wire_hex = (body + struct.pack(">H", checksum)).hex().upper()

        return {
            "goose_pdu": pdu,
            "wire_frame_hex": wire_hex,
            "crc16_hex": f"0x{checksum:04X}",
            "forward_pass_ms": inference_out["latency_ms"],
        }
