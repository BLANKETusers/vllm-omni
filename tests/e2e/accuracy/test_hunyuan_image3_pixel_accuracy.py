from __future__ import annotations

import base64
import copy
import io
import os
import tempfile
import time
from pathlib import Path

import pytest
import requests
import yaml
from PIL import Image

from tests.e2e.accuracy.helpers import assert_images_pixel_close, assert_similarity, model_output_dir
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 2.5
HEIGHT = 1024
WIDTH = 1024
PROMPT = "A brown and white dog is running on the grass."
MEAN_THRESHOLD = 3e-2
P99_THRESHOLD = 3e-1
SSIM_THRESHOLD = 0.97
PSNR_THRESHOLD_CUDA = 30.0
PSNR_THRESHOLD_NPU = 26.0


def _psnr_threshold() -> float:
    from vllm_omni.platforms import current_omni_platform

    return PSNR_THRESHOLD_NPU if current_omni_platform.is_npu() else PSNR_THRESHOLD_CUDA


def _request_timeout() -> int:
    from vllm_omni.platforms import current_omni_platform

    return 1200 if current_omni_platform.is_npu() else 600


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BASELINE_DIR = _REPO_ROOT / "tests" / "assets" / "hunyuan"


def _baseline_path() -> Path:
    from vllm_omni.platforms import current_omni_platform

    suffix = "_npu" if current_omni_platform.is_npu() else ""
    path = _BASELINE_DIR / f"hunyuan_baseline{suffix}.png"
    assert path.exists(), f"Baseline image not found at {path}"
    return path


_OFFLINE_SCRIPT = _REPO_ROOT / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"

# DiT-only deploy config with trust_remote_code (based on hunyuan_image3_dit.yaml).
_DEPLOY_CONFIG = {
    "pipeline": "hunyuan_image3_dit",
    "async_chunk": False,
    "trust_remote_code": True,
    "stages": [
        {
            "stage_id": 0,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
            "trust_remote_code": True,
            "devices": "0,1,2,3",  # set dynamically by _write_deploy_config
            "vae_use_slicing": False,
            "moe_backend": "flashinfer_cutlass",
            "vae_use_tiling": False,
            "parallel_config": {
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "tensor_parallel_size": 4,
                "enable_expert_parallel": True,
                "sequence_parallel_size": 1,
                "ulysses_degree": 1,
                "ring_degree": 1,
                "cfg_parallel_size": 1,
                "vae_patch_parallel_size": 1,
                "use_hsdp": False,
                "hsdp_shard_size": -1,
                "hsdp_replicate_size": 1,
            },
            "default_sampling_params": {
                "seed": SEED,
            },
        },
    ],
    "platforms": {
        "npu": {
            "stages": [
                {
                    "stage_id": 0,
                    "gpu_memory_utilization": 0.65,
                    "devices": "0,1,2,3",
                    "moe_backend": "auto",
                    "max_num_batched_tokens": 32768,
                    "parallel_config": {
                        "tensor_parallel_size": 4,
                        "enable_expert_parallel": True,
                    },
                },
            ],
        },
    },
}


def _model_name() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_MODEL", MODEL_NAME)


def _devices() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_DEVICES", "0,1,2,3")


def _write_deploy_config(path: Path) -> None:
    config = copy.deepcopy(_DEPLOY_CONFIG)
    devices = _devices()
    config["stages"][0]["devices"] = devices
    config["stages"][0]["parallel_config"]["tensor_parallel_size"] = len(devices.split(","))
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _wait_for_server_ready(omni_server: OmniServer, timeout_s: int = 300) -> None:
    """Poll /health until the server responds 200, or raise on timeout/error."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"http://{omni_server.host}:{omni_server.port}/health",
                timeout=10,
            )
            if resp.status_code == 200:
                return
            print(f"[health] Server returned {resp.status_code}: {resp.text[:500]}")
        except requests.RequestException as e:
            print(f"[health] Not ready: {e}")
        time.sleep(5)
    raise RuntimeError(f"Server /health did not return 200 within {timeout_s}s")


def _run_vllm_omni_hunyuan_image3_online(*, model: str, deploy_config: str, output_path: Path) -> Image.Image:
    server_args = [
        "--deploy-config",
        deploy_config,
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
        "--enforce-eager",
        "--trust-remote-code",
    ]
    with OmniServer(model, server_args, use_omni=True) as omni_server:
        _wait_for_server_ready(omni_server)
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "seed": SEED,
                "bot_task": "none",
                "use_system_prompt": "en_unified",
            },
            timeout=_request_timeout(),
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        image.save(output_path)
        return image


def _run_vllm_omni_hunyuan_image3_offline(*, model: str, deploy_config: str, output_path: Path) -> Image.Image:
    import subprocess
    import sys

    output_dir = str(output_path.parent)
    subprocess.run(
        [
            sys.executable,
            str(_OFFLINE_SCRIPT),
            "--modality",
            "text2img",
            "--deploy-config",
            deploy_config,
            "--prompts",
            PROMPT,
            "--output",
            output_dir,
            "--steps",
            str(NUM_INFERENCE_STEPS),
            "--guidance-scale",
            str(GUIDANCE_SCALE),
            "--seed",
            str(SEED),
            "--height",
            str(HEIGHT),
            "--width",
            str(WIDTH),
            "--bot-task",
            "none",
            "--sys-type",
            "en_unified",
            "--model",
            model,
            "--enforce-eager",
        ],
        check=True,
    )
    images = sorted(Path(output_dir).glob("output_*.png"))
    assert images, f"No output image found in {output_dir}"
    image = Image.open(images[0]).convert("RGB")
    image.load()
    image.save(output_path)
    return image


def _assert_against_baseline(image: Image.Image, label: str) -> None:
    baseline_path = _baseline_path()
    baseline_image = Image.open(baseline_path).convert("RGB")

    assert_images_pixel_close(
        model_name=f"{MODEL_NAME} ({label} vs baseline)",
        vllm_image=image,
        diffusers_image=baseline_image,
        mean_threshold=MEAN_THRESHOLD,
        p99_threshold=P99_THRESHOLD,
    )
    assert_similarity(
        model_name=f"{MODEL_NAME} ({label} vs baseline)",
        vllm_image=image,
        diffusers_image=baseline_image,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=_psnr_threshold(),
    )


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_hunyuan_image3_pixel_accuracy_online(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_config_path = Path(tmpdir) / "deploy.yaml"
        _write_deploy_config(deploy_config_path)
        image = _run_vllm_omni_hunyuan_image3_online(
            model=model, deploy_config=str(deploy_config_path), output_path=output_dir / "vllm_omni_online.png"
        )
    _assert_against_baseline(image, "online")


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_hunyuan_image3_pixel_accuracy_offline(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_config_path = Path(tmpdir) / "deploy.yaml"
        _write_deploy_config(deploy_config_path)
        image = _run_vllm_omni_hunyuan_image3_offline(
            model=model, deploy_config=str(deploy_config_path), output_path=output_dir / "vllm_omni_offline.png"
        )
    _assert_against_baseline(image, "offline")
