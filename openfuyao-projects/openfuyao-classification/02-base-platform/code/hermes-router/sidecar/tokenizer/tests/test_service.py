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

import asyncio
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _minimal_tokenizer import VALID_CHAT_TEMPLATE, save_minimal_tokenizer
from core.errors import NotInitializedError, RenderUnavailableError
from core.models import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    InitializeResult,
    TokenizationResult,
)
from core.service import TokenizerService
from providers.huggingface import HuggingFaceProvider
from providers.resolution import ResolvedTokenizer

REAL_MODEL_ID = "Qwen/Qwen3-8B"
REAL_PROMPT = "hello"


class _FakeRenderClient:
    async def render_completion(self, request: CompletionRequest) -> TokenizationResult:
        del request
        return TokenizationResult(token_ids=[11, 22, 33])

    async def render_chat_completion(self, request) -> TokenizationResult:
        del request
        return TokenizationResult(token_ids=[44, 55, 66])


def _render_factory(resolved: ResolvedTokenizer) -> _FakeRenderClient:
    del resolved
    return _FakeRenderClient()


def _hf_provider(identifier: str) -> HuggingFaceProvider:
    return HuggingFaceProvider(
        ResolvedTokenizer(
            served_model_name=REAL_MODEL_ID,
            model_path=identifier,
            tokenizer_path=identifier,
            provider="huggingface",
            trust_remote_code=True,
        )
    )


class TokenizerServiceTests(IsolatedAsyncioTestCase):
    def test_tokenize_requires_initialize(self) -> None:
        service = TokenizerService(
            provider_name="huggingface",
            render_factory=_render_factory,
        )
        with self.assertRaises(NotInitializedError):
            service.tokenize(model=REAL_MODEL_ID, prompt="hello")

    def test_initialize_requires_render_factory(self) -> None:
        service = TokenizerService(provider_name="huggingface")

        with self.assertRaises(RenderUnavailableError):
            service.initialize(model=REAL_MODEL_ID)

    async def test_render_requires_initialize(self) -> None:
        service = TokenizerService(
            provider_name="huggingface",
            render_factory=_render_factory,
        )

        with self.assertRaises(NotInitializedError):
            await service.render_completion(
                CompletionRequest(model=REAL_MODEL_ID, prompt_text=REAL_PROMPT)
            )

    async def test_render_completion_rejects_model_mismatch(self) -> None:
        service = TokenizerService(
            provider_name="huggingface",
            render_factory=_render_factory,
        )
        service.initialize(model=REAL_MODEL_ID)

        with self.assertRaisesRegex(
            ValueError,
            r"model mismatch: requested='other/model', initialized='Qwen/Qwen3-8B'",
        ):
            await service.render_completion(
                CompletionRequest(model="other/model", prompt_text=REAL_PROMPT)
            )

    async def test_render_chat_completion_rejects_model_mismatch(self) -> None:
        service = TokenizerService(
            provider_name="huggingface",
            render_factory=_render_factory,
        )
        service.initialize(model=REAL_MODEL_ID)

        with self.assertRaisesRegex(
            ValueError,
            r"model mismatch: requested='other/model', initialized='Qwen/Qwen3-8B'",
        ):
            await service.render_chat_completion(
                ChatCompletionRequest(
                    model="other/model",
                    messages=[ChatMessage(role="user", content=REAL_PROMPT)],
                )
            )

    def test_tokenize_rejects_model_mismatch(self) -> None:
        service = TokenizerService(
            provider_name="huggingface",
            render_factory=_render_factory,
        )
        service.initialize(model=REAL_MODEL_ID)

        with self.assertRaisesRegex(
            ValueError,
            r"model mismatch: requested='other/model', initialized='Qwen/Qwen3-8B'",
        ):
            service.tokenize(model="other/model", prompt=REAL_PROMPT)

    def test_initialize_uses_local_tokenizer_path_as_render_input(self) -> None:
        remote_provider = _hf_provider(REAL_MODEL_ID)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_dir = Path(tmp_dir) / "local-tokenizer"
            getattr(remote_provider, "_tokenizer").save_pretrained(local_dir)

            service = TokenizerService(
                provider_name="huggingface",
                render_factory=_render_factory,
            )
            result = service.initialize(
                model="my-custom-model",
                tokenizer_name=str(local_dir),
            )

        self.assertEqual(result.model, "my-custom-model")
        self.assertEqual(result.resolved_provider, "huggingface")
        self.assertIsNotNone(service._resolved_tokenizer)
        self.assertEqual(service._resolved_tokenizer.served_model_name, "my-custom-model")
        self.assertEqual(service._resolved_tokenizer.model_path, str(local_dir))
        self.assertEqual(service._resolved_tokenizer.tokenizer_path, str(local_dir))

    def test_has_usable_chat_template_reflects_builtin_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plain_dir = save_minimal_tokenizer(Path(tmp_dir) / "plain")
            service = TokenizerService(
                provider_name="huggingface",
                render_factory=_render_factory,
            )
            service.initialize(
                model="my-custom-model",
                tokenizer_name=str(plain_dir),
            )
            self.assertFalse(service.has_usable_chat_template())

        with tempfile.TemporaryDirectory() as tmp_dir:
            templated_dir = save_minimal_tokenizer(
                Path(tmp_dir) / "templated",
                chat_template=VALID_CHAT_TEMPLATE,
            )
            service = TokenizerService(
                provider_name="huggingface",
                render_factory=_render_factory,
            )
            service.initialize(
                model="my-custom-model",
                tokenizer_name=str(templated_dir),
            )
            self.assertTrue(service.has_usable_chat_template())

    def test_has_usable_chat_template_false_before_initialize(self) -> None:
        service = TokenizerService(
            provider_name="huggingface",
            render_factory=_render_factory,
        )
        self.assertFalse(service.has_usable_chat_template())

    def test_initialize_and_tokenize_with_real_provider(self) -> None:
        service = TokenizerService(
            provider_name="huggingface",
            render_factory=_render_factory,
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            service = TokenizerService(
                provider_name="huggingface",
                cache_dir=cache_dir,
                render_factory=_render_factory,
            )
            result = service.initialize(
                model=REAL_MODEL_ID,
            )

        self.assertEqual(result.model, REAL_MODEL_ID)
        self.assertEqual(result.resolved_provider, "huggingface")
        tokenized = service.tokenize(model=REAL_MODEL_ID, prompt=REAL_PROMPT, add_special_tokens=True)
        self.assertTrue(tokenized.token_ids)
        self.assertTrue(all(isinstance(token_id, int) for token_id in tokenized.token_ids))

    def test_render_factory_failure_raises_error(self) -> None:
        def failing_render_factory(resolved: ResolvedTokenizer) -> None:
            del resolved
            raise RuntimeError("render factory failed")

        service = TokenizerService(
            provider_name="huggingface",
            render_factory=failing_render_factory,
        )

        with self.assertRaises(RuntimeError):
            service.initialize(model=REAL_MODEL_ID)

    def test_render_factory_returns_none_raises_error(self) -> None:
        def null_render_factory(resolved: ResolvedTokenizer) -> None:
            del resolved
            return None

        service = TokenizerService(
            provider_name="huggingface",
            render_factory=null_render_factory,
        )

        with self.assertRaises(RenderUnavailableError):
            service.initialize(model=REAL_MODEL_ID)


class _GatedInitService(TokenizerService):
    """TokenizerService whose first _create_provider call blocks on a gate."""

    def __init__(self, *, fail_first: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.provider_calls = 0
        self.gate = threading.Event()
        self.fail_first = fail_first

    def _create_provider(self, resolved: ResolvedTokenizer) -> HuggingFaceProvider:
        self.provider_calls += 1
        if self.provider_calls == 1:
            if self.fail_first:
                raise RuntimeError("provider construction failed")
            self.gate.wait(timeout=10)
        return super()._create_provider(resolved)


class TokenizerServiceSingleFlightTests(IsolatedAsyncioTestCase):
    def _local_tokenizer_dir(self) -> str:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        return str(save_minimal_tokenizer(Path(tmp_dir.name) / "tok"))

    def test_initialize_single_flight_under_concurrency(self) -> None:
        service = _GatedInitService(provider_name="huggingface", render_factory=_render_factory)
        tokenizer_dir = self._local_tokenizer_dir()

        results: list[InitializeResult] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                results.append(service.initialize(model="m", tokenizer_name=tokenizer_dir))
            except BaseException as exc:  # noqa: BLE001 - failure capture for assertions
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()

        deadline = time.monotonic() + 2
        while service.provider_calls < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)
        self.assertEqual(service.provider_calls, 1)

        service.gate.set()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 5)
        self.assertEqual(service.provider_calls, 1)
        self.assertEqual(len({id(result) for result in results}), 1)

    def test_initialize_returns_cached_result_without_re_resolution(self) -> None:
        service = _GatedInitService(provider_name="huggingface", render_factory=_render_factory)
        service.gate.set()
        tokenizer_dir = self._local_tokenizer_dir()

        first = service.initialize(model="m", tokenizer_name=tokenizer_dir)
        second = service.initialize(model="m", tokenizer_name=tokenizer_dir)

        self.assertIs(first, second)
        self.assertEqual(service.provider_calls, 1)

    def test_initialize_re_resolves_on_model_change(self) -> None:
        service = _GatedInitService(provider_name="huggingface", render_factory=_render_factory)
        service.gate.set()
        tokenizer_dir = self._local_tokenizer_dir()

        service.initialize(model="a", tokenizer_name=tokenizer_dir)
        service.initialize(model="b", tokenizer_name=tokenizer_dir)

        self.assertEqual(service.provider_calls, 2)

    def test_initialize_normalizes_empty_tokenizer_name(self) -> None:
        # The model itself is a local path so the "" fallback stays offline.
        tokenizer_dir = self._local_tokenizer_dir()
        service = _GatedInitService(provider_name="huggingface", render_factory=_render_factory)
        service.gate.set()

        first = service.initialize(model=tokenizer_dir, tokenizer_name="")
        second = service.initialize(model=tokenizer_dir)

        self.assertIs(first, second)
        self.assertEqual(service.provider_calls, 1)

    def test_initialize_failure_is_not_cached(self) -> None:
        service = _GatedInitService(
            provider_name="huggingface", render_factory=_render_factory, fail_first=True
        )
        tokenizer_dir = self._local_tokenizer_dir()

        with self.assertRaises(RuntimeError):
            service.initialize(model="m", tokenizer_name=tokenizer_dir)

        service.fail_first = False
        service.gate.set()
        result = service.initialize(model="m", tokenizer_name=tokenizer_dir)

        self.assertEqual(result.model, "m")
        self.assertEqual(service.provider_calls, 2)

    async def test_cancelled_initialize_rpc_leaves_service_usable(self) -> None:
        service = _GatedInitService(provider_name="huggingface", render_factory=_render_factory)
        tokenizer_dir = self._local_tokenizer_dir()

        task = asyncio.create_task(
            asyncio.to_thread(service.initialize, model="m", tokenizer_name=tokenizer_dir)
        )
        deadline = time.monotonic() + 2
        while service.provider_calls < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        self.assertEqual(service.provider_calls, 1)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # Cancelled RPC does not abort the worker; release it to finish init.
        service.gate.set()
        deadline = time.monotonic() + 5
        while service._last_init is None and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        self.assertIsNotNone(service._last_init)

        result = await asyncio.to_thread(
            service.initialize, model="m", tokenizer_name=tokenizer_dir
        )
        self.assertEqual(service.provider_calls, 1)
        self.assertIs(result, service._last_init)
