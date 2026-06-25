# coding: utf-8

"""
Builds a LivePortrait motion template from audio frame params.

Strategy: only use LivePortrait's trained retargeting modules.
- lip sync  → c_lip_lst  → retarget_lip()   (trained model)
- eye blink → c_eyes_lst → retarget_eye()   (trained model)
- head move → R matrix   (simple rotation, no distortion)
- exp       → untouched  (don't guess the latent space)
"""

import numpy as np
from typing import List
from .audio_processor import FrameParams


def build_motion_template(
    source_kp_info: dict,
    frame_params: List[FrameParams],
    fps: float = 25.0,
    device: str = "mps",
) -> dict:
    n_frames = len(frame_params)

    base_exp   = source_kp_info["exp"].cpu().numpy().copy()
    base_kp    = source_kp_info["kp"].cpu().numpy().copy()
    base_scale = source_kp_info["scale"].cpu().numpy().copy()
    base_t     = source_kp_info["t"].cpu().numpy().copy()

    template = {
        "n_frames": n_frames,
        "output_fps": fps,
        "motion": [],
        "c_eyes_lst": [],
        "c_lip_lst": [],
    }

    for fp in frame_params:
        R = _rotation_matrix(fp.head_pitch, fp.head_yaw, fp.head_roll)

        template["motion"].append({
            "scale": base_scale.astype(np.float32),
            "R":     R.astype(np.float32),
            "exp":   base_exp.astype(np.float32),   # source exp — untouched
            "t":     base_t.astype(np.float32),
            "kp":    base_kp.astype(np.float32),
            "x_s":   base_kp.astype(np.float32),
        })

        # LivePortrait retargeting range: lip 0.0–0.20, eye 0.0–0.40
        template["c_lip_lst"].append(
            np.array([[fp.lip_ratio * 0.20]], dtype=np.float32)
        )
        eye_open = 0.38 * (1.0 - fp.eye_blink)
        template["c_eyes_lst"].append(
            np.array([[eye_open]], dtype=np.float32)
        )

    return template


def _rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    p, y, r = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg), np.deg2rad(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]], dtype=np.float32)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]], dtype=np.float32)
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx)[np.newaxis]
