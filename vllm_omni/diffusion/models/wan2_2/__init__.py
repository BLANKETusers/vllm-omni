# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

torch.backends.cudnn.benchmark = True

from .patch_diffusers import patch_wan_rms_norm  # noqa: E402
from .pipeline_wan2_2 import (  # noqa: E402
    Wan22Pipeline,
    WanT2VDMD2Pipeline,
    create_transformer_from_config,
    get_wan22_post_process_func,
    get_wan22_pre_process_func,
    load_transformer_config,
    retrieve_latents,
)
from .pipeline_wan2_2_i2v import (  # noqa: E402
    Wan22I2VPipeline,
    WanI2VDMD2Pipeline,
    get_wan22_i2v_post_process_func,
    get_wan22_i2v_pre_process_func,
)
from .pipeline_wan2_2_s2v import (  # noqa: E402
    Wan22S2VPipeline,
    get_wan22_s2v_post_process_func,
    get_wan22_s2v_pre_process_func,
)
from .pipeline_wan2_2_vace import (  # noqa: E402
    Wan22VACEPipeline,
    get_wan22_vace_post_process_func,
    get_wan22_vace_pre_process_func,
)
from .wan2_2_transformer import WanTransformer3DModel  # noqa: E402
from .wan2_2_vace_transformer import VaceWanTransformerBlock, WanVACETransformer3DModel  # noqa: E402

__all__ = [
    "WanT2VDMD2Pipeline",
    "Wan22Pipeline",
    "get_wan22_post_process_func",
    "get_wan22_pre_process_func",
    "retrieve_latents",
    "load_transformer_config",
    "create_transformer_from_config",
    "Wan22I2VPipeline",
    "WanI2VDMD2Pipeline",
    "get_wan22_i2v_post_process_func",
    "get_wan22_i2v_pre_process_func",
    "Wan22S2VPipeline",
    "get_wan22_s2v_post_process_func",
    "get_wan22_s2v_pre_process_func",
    "Wan22VACEPipeline",
    "get_wan22_vace_post_process_func",
    "get_wan22_vace_pre_process_func",
    "WanTransformer3DModel",
    "VaceWanTransformerBlock",
    "WanVACETransformer3DModel",
]

patch_wan_rms_norm()
