# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.config import VllmConfig

import vllm_omni.platforms.cuda.platform as cuda_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("enabled", [False, True])
def test_diffusion_cuda_initializes_requested_batch_invariance(monkeypatch, mocker, enabled):
    monkeypatch.setattr(cuda_platform.envs, "VLLM_BATCH_INVARIANT", enabled)
    initialize = mocker.patch("vllm.model_executor.determinism.batch_invariant.init_batch_invariance")
    config = mocker.Mock(spec=VllmConfig)

    cuda_platform.CudaOmniPlatform.init_diffusion_worker_vllm_config(config)

    if enabled:
        initialize.assert_called_once_with()
    else:
        initialize.assert_not_called()
