# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests of common diffusion feature combinations in online serving mode
for HunyuanVideo-1.5-T2V (480p).

Coverage (H100, since model cannot fit L4):
- CPU offloading (1 GPU) — ``core_model`` + ``advanced_model``
- CacheDiT + Layerwise CPU offloading (1 GPU) — ``advanced_model``
- CacheDiT + TP=2 + VAE patch parallel=2 (2 GPUs) — ``advanced_model``

HunyuanVideo-1.5 is a high-priority model, so the lightweight single-card row runs on
every PR (L2) and all rows run on merge (L3). The heavyweight video similarity suites
stay nightly-only in ``tests/e2e/accuracy/hunyuanvideo15_{t2v,i2v}/``.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_hunyuan_video_15_expansion.py -m "core_model and diffusion" --run-level=core_model
    pytest -s -v e2e/online_serving/test_hunyuan_video_15_expansion.py -m "advanced_model and diffusion" --run-level=advanced_model
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

PROMPT = "A cat walking across a sunlit garden, cinematic lighting, slow motion."
NEGATIVE_PROMPT = "low quality, blurry, distorted"

MODEL = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

# The plain single-card row is cheap enough for PR-level CI; the CacheDiT / parallel rows
# only run from merge onwards.
_CORE_MARKS = [pytest.mark.core_model, pytest.mark.advanced_model]
_ADVANCED_MARKS = [pytest.mark.advanced_model]


def _get_diffusion_feature_cases(model: str):
    """Return diffusion feature cases for HunyuanVideo-1.5.

    Designed for 2x H100 environment per issue #1832.
    """
    return [
        # (1 GPU) CPU offload
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--enable-cpu-offload",
                ],
            ),
            id="single_card_cpu_offload",
            marks=SINGLE_CARD_MARKS + _CORE_MARKS,
        ),
        # (1 GPU) CacheDiT + Layerwise CPU offloading
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--enable-layerwise-offload",
                ],
            ),
            id="single_card_cachedit_layerwise",
            marks=SINGLE_CARD_MARKS + _ADVANCED_MARKS,
        ),
        # (2 GPUs) CacheDiT + TP=2 + VAE patch parallel=2
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_cachedit_tp2_vae2",
            marks=PARALLEL_MARKS + _ADVANCED_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_hunyuan_video_15_t2v(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Diffusion feature coverage for HunyuanVideo-1.5-T2V on H100."""
    form_data = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": 480,
        "width": 640,
        "num_frames": 5,
        "num_inference_steps": 2,
        "guidance_scale": 6.0,
        "seed": 42,
    }

    request_config = {
        "model": omni_server.model,
        "form_data": form_data,
    }

    openai_client.send_video_diffusion_request(request_config)
