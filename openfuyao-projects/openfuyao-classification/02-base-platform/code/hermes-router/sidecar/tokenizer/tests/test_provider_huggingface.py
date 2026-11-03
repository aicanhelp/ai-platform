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
from huggingface_hub.errors import OfflineModeIsEnabled

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _minimal_tokenizer import VALID_CHAT_TEMPLATE, save_minimal_tokenizer
from core.models import TokenizationResult
from core.service import TokenizerService
from providers.huggingface import HuggingFaceProvider
from providers.resolution import ResolvedTokenizer, TokenizerResolutionError

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
        provider="huggingface",
        trust_remote_code=True,
    )


def assert_real_token_ids(token_ids: list[int]) -> None:
    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_explicit_huggingface_provider_creates_real_tokenizer() -> None:
    provider = HuggingFaceProvider(_resolved(REAL_MODEL_ID))

    assert provider.provider_name == "huggingface"
    assert provider.resolved_name == REAL_MODEL_ID
    assert_real_token_ids(provider.encode(REAL_PROMPT))


def test_has_chat_template_reports_builtin_template(tmp_path: Path) -> None:
    plain_path = save_minimal_tokenizer(tmp_path / "plain")
    provider = HuggingFaceProvider(_resolved(str(plain_path)))

    assert provider.has_chat_template() is False

    templated_path = save_minimal_tokenizer(tmp_path / "templated", chat_template=VALID_CHAT_TEMPLATE)
    provider = HuggingFaceProvider(_resolved(str(templated_path)))

    assert provider.has_chat_template() is True


def test_existing_local_path_is_loaded_directly(tmp_path: Path) -> None:
    provider = HuggingFaceProvider(_resolved(REAL_MODEL_ID))
    tokenizer = getattr(provider, "_tokenizer")
    local_path = tmp_path / "local-tokenizer"
    tokenizer.save_pretrained(local_path)

    service = TokenizerService(
        provider_name="huggingface",
        render_factory=_render_factory,
    )
    service.initialize(
        model=REAL_MODEL_ID,
        tokenizer_name=str(local_path),
    )

    assert service._resolved_tokenizer is not None
    assert service._resolved_tokenizer.tokenizer_path == str(local_path)
    assert service._resolved_tokenizer.model_path == str(local_path)


def test_missing_explicit_local_path_fails_without_remote_fallback(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-qwen-tokenizer"
    service = TokenizerService(
        provider_name="huggingface",
        render_factory=_render_factory,
    )

    with pytest.raises(TokenizerResolutionError, match="local path does not exist"):
        service.initialize(
            model=REAL_MODEL_ID,
            tokenizer_name=str(missing_path),
        )


def test_remote_model_uses_configured_cache_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "model_cache"
    service = TokenizerService(
        provider_name="huggingface",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    service.initialize(model=REAL_MODEL_ID)

    resolved = service._resolved_tokenizer
    assert resolved is not None
    assert resolved.provider == "huggingface"
    assert str(cache_dir) in resolved.tokenizer_path


def test_cached_tokenizer_reuse_skips_download(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hf_cache"
    service = TokenizerService(
        provider_name="huggingface",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    service.initialize(model=REAL_MODEL_ID)
    first_path = service._resolved_tokenizer.tokenizer_path

    service.initialize(model=REAL_MODEL_ID)
    second_path = service._resolved_tokenizer.tokenizer_path

    assert first_path == second_path


def test_incomplete_cache_forces_redownload(tmp_path: Path) -> None:
    cache_dir = tmp_path / "incomplete_cache"
    incomplete_path = cache_dir / "models--Qwen--Qwen3-8B" / "snapshots" / "abc123"
    incomplete_path.mkdir(parents=True)
    (incomplete_path / "config.json").write_text("{}")
    stub_provider = type(
        "_StubProvider",
        (),
        {
            "provider_name": "huggingface",
            "resolved_name": str(tmp_path / "downloaded"),
        },
    )()

    service = TokenizerService(
        provider_name="huggingface",
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


def test_network_failure_during_download(tmp_path: Path) -> None:
    cache_dir = tmp_path / "network_fail_cache"
    service = TokenizerService(
        provider_name="huggingface",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    with patch("huggingface_hub.snapshot_download", side_effect=Exception("timeout")):
        with pytest.raises(Exception, match="timeout"):
            service.initialize(model="nonexistent/model")


def test_invalid_model_id_fails(tmp_path: Path) -> None:
    cache_dir = tmp_path / "invalid_cache"
    service = TokenizerService(
        provider_name="huggingface",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    with pytest.raises(Exception):
        service.initialize(model="InvalidModel/DoesNotExist")


def test_hf_offline_with_cached_tokenizer_succeeds(tmp_path: Path) -> None:
    cache_dir = tmp_path / "offline_cache"
    service = TokenizerService(
        provider_name="huggingface",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    service.initialize(model=REAL_MODEL_ID)

    with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True), patch(
        "huggingface_hub.snapshot_download",
        side_effect=OfflineModeIsEnabled("offline"),
    ) as mock_download:
        service.initialize(model=REAL_MODEL_ID)
        assert service._resolved_tokenizer is not None
        mock_download.assert_not_called()


def test_hf_offline_without_cached_tokenizer_fails(tmp_path: Path) -> None:
    empty_cache_dir = tmp_path / "empty_offline_cache"
    empty_cache_dir.mkdir(parents=True)

    with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True), patch(
        "huggingface_hub.snapshot_download",
        side_effect=OfflineModeIsEnabled("offline"),
    ):
        service = TokenizerService(
            provider_name="huggingface",
            cache_dir=str(empty_cache_dir),
            render_factory=_render_factory,
        )
        with pytest.raises(OfflineModeIsEnabled, match="offline"):
            service.initialize(model=REAL_MODEL_ID)


def test_hf_cache_hit_skips_snapshot_download(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hf_skip_download_cache"
    service = TokenizerService(
        provider_name="huggingface",
        cache_dir=str(cache_dir),
        render_factory=_render_factory,
    )

    service.initialize(model=REAL_MODEL_ID)

    with patch("huggingface_hub.snapshot_download") as mock_download:
        service.initialize(model=REAL_MODEL_ID)
        mock_download.assert_not_called()
