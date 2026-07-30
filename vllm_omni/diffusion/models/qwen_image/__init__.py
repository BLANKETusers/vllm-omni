# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen Image diffusion model components."""

import torch

torch.backends.cudnn.benchmark = True

from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (  # noqa: E402
    QwenImageCFGParallelMixin,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (  # noqa: E402
    QwenImageDMD2Pipeline,
    QwenImagePipeline,
    get_qwen_image_post_process_func,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (  # noqa: E402
    QwenImageTransformer2DModel,
)

__all__ = [
    "QwenImageCFGParallelMixin",
    "QwenImagePipeline",
    "QwenImageDMD2Pipeline",
    "QwenImageTransformer2DModel",
    "get_qwen_image_post_process_func",
]
