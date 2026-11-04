# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the aclnnRmsNorm 561002 health latch.

The async video API stores a failed job and returns HTTP 500 on later polls.
``/health`` only used to notice a dead process. These tests cover the latch
that makes ``check_health`` fail closed while the workers are still alive.
"""

from __future__ import annotations

import pytest
from pytest_mock import MockerFixture
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.entrypoints.diffusion_serving_health import DiffusionServingHealth
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.errors import NpuRmsNormError, OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPORTED = (
    "npu_rms_norm: call aclnnRmsNorm failed, error code is 561002 "
    "[pipeline_minimax_h3.py in _prepare_local_conditioning]"
)


def test_client_error_and_unrelated_runtime_error_do_not_latch() -> None:
    health = DiffusionServingHealth()

    health.note_failure(OmniClientError("aclnnRmsNorm error code is 561002"))
    health.note_failure(RuntimeError("video postprocess failed"))
    health.note_failure(RuntimeError("call aclnnMatmul failed, error code is 561002"))
    health.raise_if_unhealthy()


def test_escaped_561002_latches_until_a_successful_request() -> None:
    health = DiffusionServingHealth()
    # Worker boundaries re-raise the text as EngineGenerateError.
    health.note_failure(EngineGenerateError(_REPORTED))

    with pytest.raises(EngineDeadError, match="561002"):
        health.raise_if_unhealthy()

    health.note_failure(EngineGenerateError(_REPORTED))
    health.note_success()
    health.raise_if_unhealthy()


def test_wrapped_npu_rms_norm_error_latches() -> None:
    health = DiffusionServingHealth()
    health.note_failure(NpuRmsNormError("npu_rms_norm/aclnnRmsNorm failed (561002 ACLNN_ERR_INNER_TILING_ERROR)"))

    with pytest.raises(EngineDeadError, match="GET /health"):
        health.raise_if_unhealthy()


def test_omni_check_health_sees_diffusion_latch(mocker: MockerFixture) -> None:
    base = object.__new__(OmniBase)
    base.engine = mocker.MagicMock()
    base.engine.is_alive.return_value = True
    base.engine.stage_pools = None
    base.engine.stage_configs = [mocker.Mock(stage_type="diffusion")]
    base._diffusion_serving_health = DiffusionServingHealth()

    base.check_health()
    base.note_diffusion_infrastructure_failure(RuntimeError(_REPORTED))
    with pytest.raises(EngineDeadError, match="aclnnRmsNorm"):
        base.check_health()

    base.note_diffusion_infrastructure_success()
    base.check_health()


def test_non_diffusion_server_does_not_latch(mocker: MockerFixture) -> None:
    base = object.__new__(OmniBase)
    base.engine = mocker.MagicMock()
    base.engine.is_alive.return_value = True
    base.engine.stage_pools = None
    base.engine.stage_configs = [mocker.Mock(stage_type="llm")]
    base._diffusion_serving_health = DiffusionServingHealth()

    base.note_diffusion_infrastructure_failure(RuntimeError(_REPORTED))
    base.check_health()
