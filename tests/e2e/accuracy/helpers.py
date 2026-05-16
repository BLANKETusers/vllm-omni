import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

FTFY_SITECUSTOMIZE_MOCK_DIR = Path(__file__).with_name("ftfy_mock")


class IdentityFtfy:
    @staticmethod
    def fix_text(text: str) -> str:
        return text


def apply_ftfy_mock(*, wan_i2v_module: Any | None = None) -> None:
    """Mock ftfy library for test cases in the main process"""
    if wan_i2v_module is None:
        from diffusers.pipelines.wan import pipeline_wan_i2v as wan_i2v_module

    if not hasattr(wan_i2v_module, "ftfy"):
        wan_i2v_module.ftfy = IdentityFtfy()
        print("ftfy (text encoding sanitizer) is not installed. Using mock ftfy implementation (identity function)")
    else:
        print("ftfy (text encoding sanitizer) is installed. Using actual ftfy implementation.")


def env_to_apply_ftfy_mock_in_subproc(env: dict[str, str] | None = None) -> dict[str, str]:
    """Mock ftfy library for test cases in subprocesses, by prepending the sitecustomize.py directory to PYTHONPATH"""
    env_dict = dict(env or {})
    pythonpath = env_dict.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    path_entries = [str(FTFY_SITECUSTOMIZE_MOCK_DIR)]
    if pythonpath:
        path_entries.append(pythonpath)
    env_dict["PYTHONPATH"] = os.pathsep.join(path_entries)
    return env_dict


def reset_artifact_dir(path: Path) -> Path:
    import shutil

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_model_label(model: str) -> str:
    label = Path(model.rstrip("/\\")).name or "model"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)


def model_output_dir(parent_dir: Path, model: str) -> Path:
    safe_model_name = model.split("/")[-1].replace(".", "_")
    path = parent_dir / safe_model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_similarity(
    *,
    model_name: str,
    vllm_image: Image.Image,
    baseline_image: Image.Image,
    ssim_threshold: float,
    psnr_threshold: float,
    width: int | None = None,
    height: int | None = None,
    compare_mode: str = "RGB",
) -> None:
    requested_size = (width, height) if width is not None and height is not None else None
    if requested_size is not None and baseline_image.size != requested_size:
        pytest.skip(
            "Skipping as diffusers baseline output is corrupt and not comparable: "
            f"dimensions do not match requested size; requested={requested_size}, got={baseline_image.size}."
        )

    assert vllm_image.size == baseline_image.size, (
        f"Online and diffusers output sizes mismatch: online={vllm_image.size}, diffusers={baseline_image.size}"
    )

    ssim_score, psnr_score = compute_image_ssim_psnr(
        prediction=vllm_image,
        reference=baseline_image,
        compare_mode=compare_mode,
    )
    print(f"{model_name} similarity metrics:")
    print(f"  SSIM: value={ssim_score:.6f}, threshold>={ssim_threshold:.6f}, range=[-1, 1], higher_is_better=True")
    print(
        f"  PSNR: value={psnr_score:.6f} dB, threshold>={psnr_threshold:.6f} dB, range=[0, +inf), higher_is_better=True"
    )

    assert ssim_score >= ssim_threshold, (
        f"SSIM below threshold for {model_name}: got {ssim_score:.6f}, expected >= {ssim_threshold:.6f}."
    )
    assert psnr_score >= psnr_threshold, (
        f"PSNR below threshold for {model_name}: got {psnr_score:.6f}, expected >= {psnr_threshold:.6f}."
    )


def assert_image_sequence_similarity(
    *,
    model_name: str,
    vllm_images: list[Image.Image],
    baseline_images: list[Image.Image],
    ssim_threshold: float,
    psnr_threshold: float,
    compare_mode: str = "RGB",
) -> None:
    assert len(vllm_images) == len(baseline_images), (
        f"Output image count mismatch for {model_name}: online={len(vllm_images)}, diffusers={len(baseline_images)}"
    )
    for index, (vllm_image, baseline_image) in enumerate(zip(vllm_images, baseline_images, strict=True), start=1):
        assert_similarity(
            model_name=f"{model_name}[layer={index}]",
            vllm_image=vllm_image,
            baseline_image=baseline_image,
            ssim_threshold=ssim_threshold,
            psnr_threshold=psnr_threshold,
            compare_mode=compare_mode,
        )


def compute_image_ssim_psnr(
    *,
    prediction: Image.Image,
    reference: Image.Image,
    compare_mode: str = "RGB",
) -> tuple[float, float]:
    pred_tensor = _pil_to_batched_tensor(prediction, compare_mode=compare_mode)
    ref_tensor = _pil_to_batched_tensor(reference, compare_mode=compare_mode)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)

    ssim_value = float(ssim_metric(pred_tensor, ref_tensor).item())
    psnr_value = float(psnr_metric(pred_tensor, ref_tensor).item())
    return ssim_value, psnr_value


class CLIPScorer:
    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch16"):
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(model_name_or_path)
        self._processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self._model.eval()

    def score(self, image: Image.Image, text: str) -> float:
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        similarity = (img_emb * txt_emb).sum(dim=-1)
        return float(similarity.item() * 100)

    def assert_score(self, *, model_name: str, image: Image.Image, text: str, threshold: float) -> None:
        value = self.score(image, text)
        print(f"{model_name} CLIP score:")
        print(f"  CLIPScore: value={value:.4f}, threshold>={threshold:.4f}, higher_is_better=True")
        assert value >= threshold, (
            f"CLIP score below threshold for {model_name}: got {value:.4f}, expected >= {threshold:.4f}."
        )


def _pil_to_clip_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _pil_to_batched_tensor(image: Image.Image, *, compare_mode: str) -> torch.Tensor:
    array = np.asarray(image.convert(compare_mode), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def assert_images_pixel_close(
    *,
    model_name: str,
    vllm_image: Image.Image,
    baseline_image: Image.Image,
    compare_mode: str = "RGB",
    mean_threshold: float,
    p99_threshold: float,
) -> None:
    """Assert full-image pixel closeness between online serving and diffusers output.

    Match the threshold style used by diffusion parallelism tests: convert both
    images to float32 RGB tensors in the [0, 1] range, then compare mean and p99
    absolute channel differences.

    Accuracy improves as both values move lower:
    - ``mean_abs_diff`` tracks global average drift across the whole image.
    - ``p99_abs_diff`` tracks tail drift while ignoring the noisiest 1% of
      channel samples.

    The printed mismatch ratios are diagnostics only. ``pixel_ratio`` counts
    pixels where any channel exceeds a tolerance, while ``channel_ratio`` counts
    individual channel samples above that tolerance.
    """
    assert vllm_image.size == baseline_image.size, (
        f"Online and diffusers output sizes mismatch: online={vllm_image.size}, diffusers={baseline_image.size}"
    )

    vllm_array = np.asarray(vllm_image.convert(compare_mode), dtype=np.float32) / 255.0
    diffusers_array = np.asarray(baseline_image.convert(compare_mode), dtype=np.float32) / 255.0
    channel_abs_diff = np.abs(vllm_array - diffusers_array)
    mean_abs_diff = float(channel_abs_diff.mean())
    p99_abs_diff = float(np.quantile(channel_abs_diff, 0.99))
    percentiles = {
        percentile: float(np.quantile(channel_abs_diff, percentile / 100.0)) for percentile in (50, 90, 95, 99, 99.9)
    }

    print(f"{model_name} pixel metrics:")
    print(
        f"  mean_abs_diff: value={mean_abs_diff:.6e}, threshold<={mean_threshold:.6e}, "
        "range=[0, 1], lower_is_better=True"
    )
    print(
        f"  p99_abs_diff: value={p99_abs_diff:.6e}, threshold<={p99_threshold:.6e}, range=[0, 1], lower_is_better=True"
    )
    print("  abs_diff_percentiles:")
    for percentile, value in percentiles.items():
        print(f"    p{percentile}: value={value:.6e}, range=[0, 1], lower_is_better=True")
    print("  mismatch_ratios_by_channel_threshold:")
    for tolerance in (0, 1, 2, 4, 8, 16, 32, 64, 128):
        normalized_tolerance = tolerance / 255.0
        pixel_mismatch = np.any(channel_abs_diff > normalized_tolerance, axis=-1)
        pixel_mismatch_ratio = float(np.count_nonzero(pixel_mismatch) / pixel_mismatch.size)
        channel_mismatch_ratio = float(
            np.count_nonzero(channel_abs_diff > normalized_tolerance) / channel_abs_diff.size
        )
        print(
            f"    threshold>{normalized_tolerance:.6e} ({tolerance}/255): pixel_ratio={pixel_mismatch_ratio:.8f}, "
            f"channel_ratio={channel_mismatch_ratio:.8f}"
        )

    assert mean_abs_diff <= mean_threshold and p99_abs_diff <= p99_threshold, (
        f"Image diff exceeded threshold for {model_name}: mean_abs_diff={mean_abs_diff:.6e}, "
        f"p99_abs_diff={p99_abs_diff:.6e} (thresholds: mean<={mean_threshold:.6e}, "
        f"p99<={p99_threshold:.6e})"
    ) 
