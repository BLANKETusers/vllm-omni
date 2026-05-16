from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import pytest
import requests
from PIL import Image

from tests.e2e.accuracy.helpers import assert_images_pixel_close, model_output_dir
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
MEAN_THRESHOLD = 0.02
P99_THRESHOLD = 0.10

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASELINE_PATH = _REPO_ROOT / "assets" / "hunyuan" / "hunyuan_baseline.png"
_DEFAULT_DEPLOY_CONFIG = _REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3.yaml"
_OFFLINE_SCRIPT = _REPO_ROOT / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"

def _model_name() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_MODEL", MODEL_NAME)


def _deploy_config_path() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_DEPLOY_CONFIG", str(_DEFAULT_DEPLOY_CONFIG))


def _run_vllm_omni_hunyuan_image3_online(
    *, model: str, deploy_config: str | None = None, output_path: Path
) -> Image.Image:
    deploy_config = deploy_config or _deploy_config_path()
    server_args = [
        "--deploy-config", deploy_config,
        "--stage-init-timeout", "300",
        "--init-timeout", "900",
    ]
    with OmniServer(model, server_args, use_omni=True) as omni_server:
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
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        image.save(output_path)
        return image


def _run_vllm_omni_hunyuan_image3_offline(
    *, model: str, deploy_config: str | None = None, output_path: Path
) -> Image.Image:
    import subprocess

    deploy_config = deploy_config or _deploy_config_path()
    output_dir = str(output_path.parent)
    subprocess.run(
        [
            "python", str(_OFFLINE_SCRIPT),
            "--model", model,
            "--modality", "text2img",
            "--deploy-config", deploy_config,
            "--prompts", PROMPT,
            "--output", output_dir,
            "--guidance-scale", str(GUIDANCE_SCALE),
            "--seed", str(SEED),
        ],
        check=True,
    )
    images = sorted(Path(output_dir).glob("output_*.png"))
    assert images, f"No output image found in {output_dir}"
    image = Image.open(images[0]).convert("RGB")
    image.load()
    image.save(output_path)
    return image


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_hunyuan_image3_pixel_accuracy(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME)

    # online
    # online_output = _run_vllm_omni_hunyuan_image3_online(model=model, output_path=output_dir / "vllm_omni_online.png")
    # offline
    offline_output = _run_vllm_omni_hunyuan_image3_offline(model=model, output_path=output_dir / "vllm_omni_offline.png")

    # online vs offline: same seed / params, different serving paths → must be pixel-close.
    # assert_images_pixel_close(
    #     model_name=f"{MODEL_NAME} (online vs offline)",
    #     vllm_image=online_output,
    #     baseline_image=offline_output,
    #     mean_threshold=MEAN_THRESHOLD,
    #     p99_threshold=P99_THRESHOLD,
    # )

    # Baseline regression check: requires hunyuan_baseline.png generated from the
    # same vllm-omni serving path and seed.
    assert BASELINE_PATH.exists(), f"Baseline image not found at {BASELINE_PATH}"
    baseline_image = Image.open(BASELINE_PATH).convert("RGB")

    assert_images_pixel_close(
        model_name=f"{MODEL_NAME} (offline vs baseline)",
        vllm_image=offline_output,
        baseline_image=baseline_image,
        mean_threshold=MEAN_THRESHOLD,
        p99_threshold=P99_THRESHOLD,
    )

    # online vs baseline_image
    # assert_images_pixel_close(
    #     model_name=f"{MODEL_NAME} (online vs baseline)",
    #     vllm_image=online_output,
    #     baseline_image=baseline_image,
    #     mean_threshold=MEAN_THRESHOLD,
    #     p99_threshold=P99_THRESHOLD,
    # )
