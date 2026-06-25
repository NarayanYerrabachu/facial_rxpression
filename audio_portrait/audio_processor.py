# coding: utf-8

import numpy as np
import librosa
import whisper
from dataclasses import dataclass
from typing import List


@dataclass
class FrameParams:
    lip_ratio: float    # 0.0–1.0 — jaw/lip openness
    energy: float       # 0.0–1.0 — speech energy
    eye_blink: float    # 0.0 open, 1.0 closed
    head_pitch: float   # degrees
    head_yaw: float     # degrees
    head_roll: float    # degrees


# ── Noise helpers ──────────────────────────────────────────────────────────────

def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise — much more natural than sine waves for head drift."""
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n) + 1e-6
    pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(f), n=n)
    pink -= pink.mean()
    std = pink.std()
    if std > 0:
        pink /= std
    return pink.astype(np.float32)


def _smooth(x: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    k = np.hanning(max(3, int(fps * window_s)))
    k /= k.sum()
    return np.convolve(x, k, mode="same").astype(np.float32)


def _spring_damp(signal: np.ndarray, stiffness: float = 0.25, damping: float = 0.55) -> np.ndarray:
    """Simulate spring-mass system so jaw opens/closes with inertia, not instantly."""
    out = np.zeros_like(signal)
    vel = 0.0
    pos = 0.0
    for i, target in enumerate(signal):
        force = stiffness * (target - pos)
        vel = damping * vel + force
        pos += vel
        out[i] = pos
    return out


def _build_blink_schedule(n_frames: int, fps: float, rng: np.random.Generator) -> np.ndarray:
    """Natural blinks: varied interval, occasional double-blink, asymmetric speed."""
    blinks = np.zeros(n_frames, dtype=np.float32)
    t = 0.0
    total = n_frames / fps
    last_was_double = False
    while t < total:
        # varied interval: 2–6s, shorter after a long gap
        interval = rng.uniform(2.0, 5.5)
        t += interval
        frame = int(t * fps)
        if frame >= n_frames:
            break

        # occasionally do a double blink
        do_double = (not last_was_double) and rng.random() < 0.2
        last_was_double = do_double

        for blink_num in range(2 if do_double else 1):
            offset = frame + blink_num * int(fps * 0.22)
            # fast close (5 frames), slow open (8 frames) — asymmetric
            close_f = max(2, int(fps * 0.05))
            open_f  = max(3, int(fps * 0.09))
            for j in range(close_f):
                idx = offset + j
                if 0 <= idx < n_frames:
                    blinks[idx] = max(blinks[idx], j / close_f)
            for j in range(open_f):
                idx = offset + close_f + j
                if 0 <= idx < n_frames:
                    blinks[idx] = max(blinks[idx], 1.0 - j / open_f)
    return blinks


def _build_head_movement(energy: np.ndarray, fps: float, rng: np.random.Generator) -> tuple:
    """
    Realistic head movement: pink noise base + speech-correlated nod.
    No sine waves — real heads drift irregularly.
    """
    n = len(energy)

    # pink noise base (slow drift, 1/f spectrum)
    pitch_noise = _pink_noise(n, rng)
    yaw_noise   = _pink_noise(n, rng)
    roll_noise  = _pink_noise(n, rng)

    # smooth heavily — head moves slowly
    pitch_noise = _smooth(pitch_noise, fps, 0.8)
    yaw_noise   = _smooth(yaw_noise,   fps, 1.0)
    roll_noise  = _smooth(roll_noise,  fps, 1.2)

    # scale: pitch ±2.5°, yaw ±2°, roll ±1°
    pitch = pitch_noise * 1.5 * (0.4 + 0.6 * energy)
    yaw   = yaw_noise   * 1.2 * (0.3 + 0.7 * energy)
    roll  = roll_noise  * 0.4

    # subtle forward nod when speaking louder (natural emphasis)
    nod = _smooth(energy, fps, 0.1) * 1.2
    pitch += nod

    return (
        np.clip(pitch, -4.0, 4.0).astype(np.float32),
        np.clip(yaw,   -3.0, 3.0).astype(np.float32),
        np.clip(roll,  -1.5, 1.5).astype(np.float32),
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def process_audio(
    audio_path: str,
    fps: float = 25.0,
    whisper_model: str = "base",
    use_whisper: bool = True,
) -> List[FrameParams]:
    rng = np.random.default_rng(42)

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    n_frames = int(np.ceil(len(y) / sr * fps))

    # energy envelope
    hop = max(1, int(sr / fps))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_at_fps = np.interp(np.linspace(0, len(rms) - 1, n_frames), np.arange(len(rms)), rms)
    rms_norm = (rms_at_fps / (rms_at_fps.max() + 1e-8)).astype(np.float32)
    energy = _smooth(rms_norm, fps, 0.04)

    # lip ratios
    if use_whisper:
        try:
            model = whisper.load_model(whisper_model)
            result = model.transcribe(audio_path, word_timestamps=True, language=None, fp16=False)
            lip_raw = _whisper_to_lip_ratios(result, n_frames, fps, y, sr)
        except Exception as e:
            print(f"[audio_processor] Whisper failed ({e}), using energy fallback.")
            lip_raw = np.where(energy < 0.04, 0.0, energy ** 0.55).astype(np.float32)
    else:
        lip_raw = np.where(energy < 0.04, 0.0, energy ** 0.55).astype(np.float32)

    # clean silence, then spring-damp for natural inertia
    lip_raw = np.where(energy < 0.03, 0.0, lip_raw).astype(np.float32)
    lip_ratios = _spring_damp(lip_raw, stiffness=0.18, damping=0.52)
    lip_ratios = np.clip(lip_ratios, 0.0, 1.0).astype(np.float32)

    # eyebrows anticipate speech by ~2 frames
    brow_energy = np.roll(energy, -2)
    brow_energy[:2] = energy[:2]

    blinks = _build_blink_schedule(n_frames, fps, rng)
    pitch, yaw, roll = _build_head_movement(energy, fps, rng)

    return [
        FrameParams(
            lip_ratio=float(lip_ratios[i]),
            energy=float(brow_energy[i]),
            eye_blink=float(blinks[i]),
            head_pitch=float(pitch[i]),
            head_yaw=float(yaw[i]),
            head_roll=float(roll[i]),
        )
        for i in range(n_frames)
    ]


def _whisper_to_lip_ratios(result, n_frames, fps, y, sr):
    n_samples = len(y)
    signal = np.zeros(n_samples, dtype=np.float32)
    vowels = {"a": 0.9, "e": 0.6, "i": 0.35, "o": 0.65, "u": 0.4,
              "ä": 0.75, "ö": 0.55, "ü": 0.35}

    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            start = w.get("start", 0.0)
            end   = w.get("end", start + 0.05)
            text  = w.get("word", "").strip().lower()
            scores = [vowels[c] for c in text if c in vowels]
            val = float(np.mean(scores)) if scores else 0.15
            s = int(start * sr)
            e = min(int(end * sr), n_samples)
            if e > s:
                seg_len = e - s
                ramp = min(int(0.02 * sr), seg_len // 4)
                signal[s:s + ramp] = np.linspace(0, val, ramp)
                signal[s + ramp:e - ramp] = val
                if e - ramp > s + ramp:
                    signal[e - ramp:e] = np.linspace(val, 0, ramp)

    hop = sr / fps
    lip = np.array([
        signal[int(i * hop):max(int(i * hop) + 1, int((i + 1) * hop))].mean()
        for i in range(n_frames)
    ], dtype=np.float32)
    return np.clip(lip, 0.0, 1.0)
