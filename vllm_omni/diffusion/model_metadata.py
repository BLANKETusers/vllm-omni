# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionModelMetadata:
    # Keep serving-facing capability metadata in a lightweight shared module so
    # config/model plumbing can read it without importing concrete pipelines.
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None
    # Deferred SHM D2H: offload GPU→CPU copies to a background thread so
    # the inference stream is never blocked by the transfer.
    enable_deferred_shm: bool = False


QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES = 4
# Upstream HunyuanImage-3.0 "Multi-Image Fusion" caps reference images at 3.
HUNYUAN_IMAGE3_MAX_INPUT_IMAGES = 3


_DIFFUSION_MODEL_METADATA: dict[str, DiffusionModelMetadata] = {
    "QwenImageEditPlusPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES,
    ),
    "HunyuanImage3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=HUNYUAN_IMAGE3_MAX_INPUT_IMAGES,
        enable_deferred_shm=True,
    ),
    "HunyuanImage3ForCausalMM": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=HUNYUAN_IMAGE3_MAX_INPUT_IMAGES,
        enable_deferred_shm=True,
    ),
}


def get_diffusion_model_metadata(model_class_name: str | None) -> DiffusionModelMetadata:
    # Unknown models fall back to "no special multimodal capabilities" so new
    # pipelines do not accidentally inherit limits meant for other models.
    if model_class_name is None:
        return DiffusionModelMetadata()
    return _DIFFUSION_MODEL_METADATA.get(model_class_name, DiffusionModelMetadata())
