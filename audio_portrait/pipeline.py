# coding: utf-8

"""
AudioPortraitPipeline: image + audio -> animated talking-head video.

Replaces LivePortrait's video-driving path with audio-derived motion.
Requires LivePortrait to be cloned alongside this repo (see README).
"""

import sys
import os
import os.path as osp
import cv2
import numpy as np
import torch
import subprocess
from rich.progress import track

from .audio_processor import process_audio
from .expression_generator import build_motion_template


def _lp_path() -> str:
    here = osp.dirname(osp.dirname(osp.realpath(__file__)))
    candidates = [
        osp.join(here, "LivePortrait"),            # sibling: ~/git/LivePortrait
        osp.join(osp.dirname(here), "LivePortrait"), # parent sibling
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    raise RuntimeError(
        f"LivePortrait not found. Expected at one of:\n" +
        "\n".join(f"  {c}" for c in candidates) +
        "\nClone it: git clone https://github.com/KlingAIResearch/LivePortrait.git"
    )


def _ensure_lp_on_path():
    lp = _lp_path()
    if lp not in sys.path:
        sys.path.insert(0, lp)


class AudioPortraitPipeline:
    """
    Animates a portrait image to match speech in an audio file.

    Usage:
        pipeline = AudioPortraitPipeline()
        pipeline.run("face.jpg", "speech.wav", "output/result.mp4")
    """

    def __init__(
        self,
        fps: float = 25.0,
        whisper_model: str = "base",
        use_whisper: bool = True,
        device_id: int = 0,
        flag_pasteback: bool = True,
        flag_stitching: bool = True,
    ):
        _ensure_lp_on_path()

        self.fps = fps
        self.whisper_model = whisper_model
        self.use_whisper = use_whisper
        self.flag_pasteback = flag_pasteback
        self.flag_stitching = flag_stitching

        # import LivePortrait modules after path is set
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.cropper import Cropper
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import prepare_paste_back, paste_back
        from src.utils.io import load_image_rgb, resize_to_limit
        from src.utils.helper import dct2device, mkdir
        from src.utils.video import images2video, add_audio_to_video

        self._get_rotation_matrix = get_rotation_matrix
        self._prepare_paste_back = prepare_paste_back
        self._paste_back = paste_back
        self._load_image_rgb = load_image_rgb
        self._resize_to_limit = resize_to_limit
        self._dct2device = dct2device
        self._mkdir = mkdir
        self._images2video = images2video
        self._add_audio_to_video = add_audio_to_video

        lp_root = _lp_path()
        inference_cfg = InferenceConfig(
            models_config=osp.join(lp_root, "src/config/models.yaml"),
            checkpoint_F=osp.join(lp_root, "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth"),
            checkpoint_M=osp.join(lp_root, "pretrained_weights/liveportrait/base_models/motion_extractor.pth"),
            checkpoint_W=osp.join(lp_root, "pretrained_weights/liveportrait/base_models/warping_module.pth"),
            checkpoint_G=osp.join(lp_root, "pretrained_weights/liveportrait/base_models/spade_generator.pth"),
            checkpoint_S=osp.join(lp_root, "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth"),
            flag_use_half_precision=False,  # MPS doesn't support FP16
            flag_pasteback=flag_pasteback,
            flag_stitching=flag_stitching,
            flag_do_crop=True,
            flag_relative_motion=False,  # we drive absolutely, not relative to a reference frame
            device_id=device_id,
        )
        crop_cfg = CropConfig()

        self.wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.device = self.wrapper.device
        self.inf_cfg = inference_cfg

    def run(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """
        Args:
            image_path: path to portrait image (jpg/png)
            audio_path: path to audio file (wav/mp3/m4a)
            output_path: where to write the output mp4

        Returns:
            output_path
        """
        print(f"[1/4] Processing audio: {audio_path}")
        frame_params = process_audio(
            audio_path,
            fps=self.fps,
            whisper_model=self.whisper_model,
            use_whisper=self.use_whisper,
        )
        n_frames = len(frame_params)
        print(f"      → {n_frames} frames at {self.fps} fps")

        print(f"[2/4] Loading and cropping portrait: {image_path}")
        img_rgb = self._load_image_rgb(image_path)
        img_rgb = self._resize_to_limit(img_rgb, 1280, 2)

        crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
        if crop_info is None:
            raise RuntimeError("No face detected in the source image. Use a clear frontal portrait.")

        source_lmk = crop_info["lmk_crop"]
        img_crop_256 = crop_info["img_crop_256x256"]

        I_s = self.wrapper.prepare_source(img_crop_256)
        x_s_info = self.wrapper.get_kp_info(I_s)
        x_c_s = x_s_info["kp"]
        R_s = self._get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
        f_s = self.wrapper.extract_feature_3d(I_s)
        x_s = self.wrapper.transform_keypoint(x_s_info)

        if self.flag_pasteback and self.inf_cfg.flag_do_crop and self.flag_stitching:
            mask_ori_float = self._prepare_paste_back(
                self.inf_cfg.mask_crop,
                crop_info["M_c2o"],
                dsize=(img_rgb.shape[1], img_rgb.shape[0]),
            )

        print(f"[3/4] Building audio-driven motion template…")
        driving_template = build_motion_template(
            source_kp_info=x_s_info,
            frame_params=frame_params,
            fps=self.fps,
            device=self.device,
        )

        print(f"[4/4] Animating {n_frames} frames…")
        I_p_lst = []
        I_p_pstbk_lst = []

        for i in track(range(n_frames), description="Animating…"):
            motion = driving_template["motion"][i]
            R_i   = torch.from_numpy(motion["R"]).to(self.device)
            scale = x_s_info["scale"]
            t     = x_s_info["t"].clone()
            t[..., 2].fill_(0)

            # head rotation applied to source keypoints
            x_d_i_new = scale * (x_c_s @ R_i + x_s_info["exp"]) + t

            # lip retargeting — trained model, safe to use
            c_d_lip_i    = driving_template["c_lip_lst"][i]
            combined_lip = self.wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
            lip_delta    = self.wrapper.retarget_lip(x_s, combined_lip)
            x_d_i_new    = x_d_i_new + lip_delta

            # eye blink retargeting — trained model, safe to use
            c_d_eyes_i   = driving_template["c_eyes_lst"][i]
            combined_eye = self.wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
            eye_delta    = self.wrapper.retarget_eye(x_s, combined_eye)
            x_d_i_new    = x_d_i_new + eye_delta

            if self.flag_stitching:
                x_d_i_new = self.wrapper.stitching(x_s, x_d_i_new)

            out = self.wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.wrapper.parse_output(out["out"])[0]
            I_p_lst.append(I_p_i)

            if self.flag_pasteback and self.inf_cfg.flag_do_crop and self.flag_stitching:
                I_p_pstbk = self._paste_back(I_p_i, crop_info["M_c2o"], img_rgb, mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
        frames = I_p_pstbk_lst if I_p_pstbk_lst else I_p_lst

        # write silent video first
        silent_path = output_path.replace(".mp4", "_silent.mp4")
        self._images2video(frames, wfp=silent_path, fps=self.fps)

        # merge audio
        cmd = [
            "ffmpeg", "-y",
            "-i", silent_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"[warning] ffmpeg audio merge failed: {result.stderr.decode()}")
            os.rename(silent_path, output_path)
        else:
            os.remove(silent_path)

        print(f"\nDone → {output_path}")
        return output_path
