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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.models import TokenizationResult
from core.service import TokenizerService
from providers.huggingface import HuggingFaceProvider
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


def _resolved(identifier: str) -> ResolvedTokenizer:
    return ResolvedTokenizer(
        served_model_name=REAL_MODEL_ID,
        model_path=identifier,
        tokenizer_path=identifier,
        provider="huggingface",
        trust_remote_code=True,
    )


def assert_real_token_ids(token_ids: list[int]) -> None:
    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)


def build_local_tokenizer_dir_from_real_provider(target_dir: Path) -> Path:
    provider = HuggingFaceProvider(_resolved(REAL_MODEL_ID))
    tokenizer = getattr(provider, "_tokenizer")
    tokenizer.save_pretrained(target_dir)
    return target_dir


def test_huggingface_tokenizer_encodes_real_prompt(tmp_path: Path) -> None:
    service = TokenizerService(
        provider_name="huggingface",
        cache_dir=str(tmp_path / "hf_cache"),
        render_factory=_render_factory,
    )
    service.initialize(model=REAL_MODEL_ID)

    with_special = service.tokenize(
        model=REAL_MODEL_ID,
        prompt=REAL_PROMPT,
        add_special_tokens=True,
    ).token_ids
    without_special = service.tokenize(
        model=REAL_MODEL_ID,
        prompt=REAL_PROMPT,
        add_special_tokens=False,
    ).token_ids

    assert_real_token_ids(with_special)
    assert_real_token_ids(without_special)


def test_modelscope_tokenizer_encodes_real_prompt(tmp_path: Path) -> None:
    service = TokenizerService(
        provider_name="modelscope",
        cache_dir=str(tmp_path / "ms_cache"),
        render_factory=_render_factory,
    )
    service.initialize(model=REAL_MODEL_ID)

    token_ids = service.tokenize(
        model=REAL_MODEL_ID,
        prompt=REAL_PROMPT,
        add_special_tokens=True,
    ).token_ids

    assert_real_token_ids(token_ids)


def test_local_tokenizer_encodes_real_prompt(tmp_path: Path) -> None:
    local_path = build_local_tokenizer_dir_from_real_provider(tmp_path / "local-tokenizer")
    service = TokenizerService(
        provider_name="huggingface",
        render_factory=_render_factory,
    )
    service.initialize(
        model=REAL_MODEL_ID,
        tokenizer_name=str(local_path),
    )

    resolved = service._resolved_tokenizer
    assert resolved is not None
    assert resolved.tokenizer_path == str(local_path)
    assert_real_token_ids(service.tokenize(model=REAL_MODEL_ID, prompt=REAL_PROMPT).token_ids)
