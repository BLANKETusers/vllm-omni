# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Latch diffusion serving health on an escaped aclnnRmsNorm tiling failure.

Worker processes stay alive after ``npu_rms_norm`` raises: the async video job
is stored as failed and ``GET /v1/videos/{id}`` returns HTTP 500, while
``GET /health`` only checks that those processes are still alive. One escaped
561002 is enough to latch this flag. A later successful ``generate()`` clears
it; until then ``check_health`` raises ``EngineDeadError`` and ``/health``
returns 503.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.errors import is_aclnn_rms_norm_tiling_failure

logger = init_logger(__name__)


class DiffusionServingHealth:
    """In-process latch shared by ``generate()`` and ``check_health()``."""

    def __init__(self) -> None:
        self.unhealthy_reason: str | None = None

    def note_failure(self, exc: BaseException) -> None:
        if self.unhealthy_reason is not None or not is_aclnn_rms_norm_tiling_failure(exc):
            return
        self.unhealthy_reason = (
            "Diffusion workers are wedged after aclnnRmsNorm error 561002 "
            "(ACLNN_ERR_INNER_TILING_ERROR). The worker processes are still alive, "
            "so this latch is what makes GET /health return 503 while video polls "
            "return HTTP 500. Restart the server, or serve one successful request, "
            f"to clear it. Latest error: {type(exc).__name__}: {exc}"
        )
        logger.error("%s", self.unhealthy_reason)

    def note_success(self) -> None:
        if self.unhealthy_reason is None:
            return
        logger.info("Clearing the aclnnRmsNorm 561002 health latch after a successful request")
        self.unhealthy_reason = None

    def raise_if_unhealthy(self) -> None:
        if self.unhealthy_reason is not None:
            raise EngineDeadError(self.unhealthy_reason)
