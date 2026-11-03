# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.models import TokenizationResult
from core.service import TokenizerService
from providers.modelscope import ModelScopeProvider
from providers.resolution import ResolvedTokenizer

REAL_MODEL_ID = "Qwen/Qwen3-8B"
REAL_PROMPT = "hello"


class _FakeRenderClient:
    async def render_completion(self, request) -> TokenizationResult:
        del request
        return TokenizationResult(token_ids=[1, 2, 3])

    async def render_chat_completion(self, request) -> TokenizationResult:
        del request
        return TokenizationResult(token_ids=[4, 5, 6])


def _render_factory(resolved: ResolvedTokenizer) -> _FakeRenderClient:
    del resolved
    return _FakeRenderClient()


def _resolved(identifier: str, *, served_model_name: str | None = None) -> ResolvedTokenizer:
    return ResolvedTokenizer(
        served_model_name=served_model_name or REAL_MODEL_ID,
        model_path=identifier,
        tokenizer_path=identifier,
        provider="modelscope",
        trust_remote_code=True,
    )


def assert_real_token_ids(token_ids: list[int]) -> None:
    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_explicit_modelscope_provider_creates_real_tokenizer() -> None:
    provider = ModelScopeProvider(_resolved(REAL_MODEL_ID))

    assert provider.provider_name == "modelscope"
    assert provider.resolved_name == REAL_MODEL_ID
    assert_real_token_ids(provider.encode(REAL_PROMPT))


def test_modelscope_remote_model_uses_cache_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ms_cache"
    service = TokenizerService(
        provider_name="modelscope",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    service.initialize(model=REAL_MODEL_ID)

    resolved = service._resolved_tokenizer
    assert resolved is not None
    assert resolved.provider == "modelscope"
    assert str(cache_dir) in resolved.tokenizer_path


def test_modelscope_cached_tokenizer_reuse(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ms_reuse_cache"
    service = TokenizerService(
        provider_name="modelscope",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    service.initialize(model=REAL_MODEL_ID)
    first_path = service._resolved_tokenizer.tokenizer_path

    service.initialize(model=REAL_MODEL_ID)
    second_path = service._resolved_tokenizer.tokenizer_path

    assert first_path == second_path


def test_modelscope_existing_local_path_is_loaded_directly(tmp_path: Path) -> None:
    provider = ModelScopeProvider(_resolved(REAL_MODEL_ID))
    tokenizer = getattr(provider, "_tokenizer")
    local_path = tmp_path / "local-tokenizer"
    tokenizer.save_pretrained(local_path)

    service = TokenizerService(
        provider_name="modelscope",
        render_factory=_render_factory,
    )
    service.initialize(
        model=REAL_MODEL_ID,
        tokenizer_name=str(local_path),
    )

    assert service._resolved_tokenizer is not None
    assert service._resolved_tokenizer.tokenizer_path == str(local_path)
    assert service._resolved_tokenizer.model_path == str(local_path)


def test_modelscope_incomplete_cache_forces_redownload(tmp_path: Path) -> None:
    cache_dir = tmp_path / "incomplete_cache"
    incomplete_path = cache_dir / "models" / "Qwen" / "Qwen3-8B"
    incomplete_path.mkdir(parents=True)
    (incomplete_path / "config.json").write_text("{}")
    stub_provider = type(
        "_StubProvider",
        (),
        {
            "provider_name": "modelscope",
            "resolved_name": str(tmp_path / "downloaded"),
        },
    )()

    service = TokenizerService(
        provider_name="modelscope",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    with patch("core.service.TokenizerService._download_tokenizer") as mock_download, patch.object(
        service,
        "_create_provider",
        return_value=stub_provider,
    ):
        mock_download.return_value = str(tmp_path / "downloaded")
        service.initialize(model=REAL_MODEL_ID)
        mock_download.assert_called_once()


def test_modelscope_cache_hit_skips_download(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ms_skip_download_cache"
    service = TokenizerService(
        provider_name="modelscope",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    service.initialize(model=REAL_MODEL_ID)

    with patch("core.service.TokenizerService._download_tokenizer") as mock_download:
        service.initialize(model=REAL_MODEL_ID)
        mock_download.assert_not_called()


def test_modelscope_network_failure_during_download(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ms_network_fail_cache"
    service = TokenizerService(
        provider_name="modelscope",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    with patch("modelscope.hub.snapshot_download.snapshot_download", side_effect=Exception("timeout")):
        with pytest.raises(Exception, match="timeout"):
            service.initialize(model="nonexistent/model")


def test_modelscope_invalid_model_id_fails(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ms_invalid_cache"
    service = TokenizerService(
        provider_name="modelscope",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    with pytest.raises(Exception):
        service.initialize(model="InvalidModel/DoesNotExist")