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
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import grpc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "proto"))

from _minimal_tokenizer import VALID_CHAT_TEMPLATE, save_minimal_tokenizer
from core.models import ChatCompletionRequest, CompletionRequest, InitializeResult, TokenizationResult
from core.service import TokenizerService
from proto import tokenizer_pb2 as pb
from transport.grpc_service import TokenizationServiceServicer

REAL_MODEL_ID = "Qwen/Qwen3-8B"
REAL_PROMPT = "hello"


class _FakeRenderClient:
    async def render_completion(self, request: CompletionRequest) -> TokenizationResult:
        del request
        return TokenizationResult(token_ids=[101, 102, 103])

    async def render_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> TokenizationResult:
        del request
        return TokenizationResult(token_ids=[201, 202, 203])


class _CapturingRenderClient:
    def __init__(self) -> None:
        self.completion_requests: list[CompletionRequest] = []
        self.chat_requests: list[ChatCompletionRequest] = []

    async def render_completion(self, request: CompletionRequest) -> TokenizationResult:
        self.completion_requests.append(request)
        return TokenizationResult(token_ids=[301, 302, 303])

    async def render_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> TokenizationResult:
        self.chat_requests.append(request)
        return TokenizationResult(token_ids=[401, 402, 403])


class AbortCalled(Exception):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        super().__init__(details)


class FakeContext:
    async def abort(self, code: grpc.StatusCode, details: str):
        raise AbortCalled(code, details)


def _build_service(*, cache_dir: str | None = None, render_factory=None) -> TokenizerService:
    return TokenizerService(
        provider_name="huggingface",
        cache_dir=cache_dir,
        render_factory=render_factory or (lambda resolved: _FakeRenderClient()),
    )


class GrpcServiceTests(IsolatedAsyncioTestCase):
    async def test_initialize_maps_invalid_identifier_to_invalid_argument(self) -> None:
        service = _build_service()
        servicer = TokenizationServiceServicer(service)

        with self.assertRaises(AbortCalled) as raised:
            await servicer.Initialize(
                pb.InitializeRequest(
                    model="served-model",
                    tokenizer_name="s3://bucket/model",
                ),
                FakeContext(),
            )

        self.assertEqual(raised.exception.code, grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("unsupported URI scheme", raised.exception.details)

    async def test_initialize_warmup_uses_direct_tokenization(self) -> None:
        # The plain-text chat warmup is tokenized directly off the render path,
        # so the vLLM render client is never invoked during initialize.
        render_client = _CapturingRenderClient()
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(
                cache_dir=cache_dir,
                render_factory=lambda resolved: render_client,
            )
            servicer = TokenizationServiceServicer(tokenizer)

            response = await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )

        self.assertEqual(response.model, REAL_MODEL_ID)
        self.assertEqual(len(render_client.chat_requests), 0)

    async def test_initialize_skips_warmup_without_chat_template(self) -> None:
        # A tokenizer without a chat template is a model/artifact property, not
        # a sidecar health problem: warmup would only fail, so initialize
        # succeeds and the plugin marks the tokenizer ready.
        with tempfile.TemporaryDirectory() as model_dir:
            tokenizer_path = save_minimal_tokenizer(Path(model_dir))
            render_client = _CapturingRenderClient()
            tokenizer = _build_service(render_factory=lambda resolved: render_client)
            servicer = TokenizationServiceServicer(tokenizer)

            with self.assertLogs("transport.grpc_service", level="WARNING") as logs:
                response = await servicer.Initialize(
                    pb.InitializeRequest(model=REAL_MODEL_ID, tokenizer_name=str(tokenizer_path)),
                    FakeContext(),
                )

        self.assertEqual(response.model, REAL_MODEL_ID)
        self.assertEqual(
            response.resolved_provider,
            pb.TOKENIZER_PROVIDER_HUGGINGFACE,
        )
        self.assertEqual(len(render_client.chat_requests), 0)
        self.assertEqual(len(render_client.completion_requests), 0)
        self.assertTrue(any("skipping warmup" in line for line in logs.output))

    async def test_initialize_warmup_succeeds_with_local_chat_template(self) -> None:
        # The plain-text chat warmup is tokenized directly off the render path,
        # so the vLLM render client is never invoked during initialize.
        with tempfile.TemporaryDirectory() as model_dir:
            tokenizer_path = save_minimal_tokenizer(Path(model_dir), chat_template=VALID_CHAT_TEMPLATE)
            render_client = _CapturingRenderClient()
            tokenizer = _build_service(render_factory=lambda resolved: render_client)
            servicer = TokenizationServiceServicer(tokenizer)

            response = await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID, tokenizer_name=str(tokenizer_path)),
                FakeContext(),
            )

        self.assertEqual(response.model, REAL_MODEL_ID)
        self.assertEqual(len(render_client.chat_requests), 0)

    async def test_initialize_warmup_failure_with_template_still_aborts(self) -> None:
        # A present-but-broken template is a real fault: warmup must keep
        # aborting initialization instead of being skipped.
        with tempfile.TemporaryDirectory() as model_dir:
            tokenizer_path = save_minimal_tokenizer(Path(model_dir), chat_template="{% if %}")
            tokenizer = _build_service()
            servicer = TokenizationServiceServicer(tokenizer)

            with self.assertRaises(AbortCalled) as raised:
                await servicer.Initialize(
                    pb.InitializeRequest(model=REAL_MODEL_ID, tokenizer_name=str(tokenizer_path)),
                    FakeContext(),
                )

        self.assertEqual(raised.exception.code, grpc.StatusCode.INTERNAL)
        self.assertIn("initialize warmup failed", raised.exception.details)

    async def test_initialize_offloads_blocking_work_from_event_loop(self) -> None:
        class SlowInitializeTokenizer:
            def initialize(
                self,
                *,
                model: str,
                tokenizer_name: str | None = None,
            ) -> InitializeResult:
                del tokenizer_name
                time.sleep(0.2)
                return InitializeResult(
                    model=model,
                    resolved_provider="huggingface",
                    resolved_tokenizer_name=model,
                )

            def has_usable_chat_template(self) -> bool:
                """Report a usable template so Initialize runs the warmup path."""
                return True

            async def render_chat_completion(
                self,
                request: ChatCompletionRequest,
            ) -> TokenizationResult:
                del request
                return TokenizationResult(token_ids=[1])

        servicer = TokenizationServiceServicer(SlowInitializeTokenizer())
        request = pb.InitializeRequest(model=REAL_MODEL_ID)

        started = time.perf_counter()
        first, second = await asyncio.gather(
            servicer.Initialize(request, FakeContext()),
            servicer.Initialize(request, FakeContext()),
        )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.33)
        self.assertEqual(first.model, REAL_MODEL_ID)
        self.assertEqual(second.model, REAL_MODEL_ID)

    async def test_tokenize_returns_proto_result(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(cache_dir=cache_dir)
            servicer = TokenizationServiceServicer(tokenizer)

            initialized = await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )
            self.assertEqual(initialized.model, REAL_MODEL_ID)
            self.assertEqual(
                initialized.resolved_provider,
                pb.TOKENIZER_PROVIDER_HUGGINGFACE,
            )

            expected = tokenizer.tokenize(
                model=REAL_MODEL_ID,
                prompt=REAL_PROMPT,
                add_special_tokens=True,
            ).token_ids
            response = await servicer.Tokenize(
                pb.TokenizeRequest(model=REAL_MODEL_ID, prompt=REAL_PROMPT, add_special_tokens=True),
                FakeContext(),
            )

            self.assertEqual(list(response.result.token_ids), expected)
            self.assertTrue(response.result.token_ids)

    async def test_tokenize_offloads_blocking_work_from_event_loop(self) -> None:
        class SlowTokenizer:
            def tokenize(
                self,
                *,
                model: str,
                prompt: str,
                add_special_tokens: bool = True,
            ) -> TokenizationResult:
                del model, prompt, add_special_tokens
                time.sleep(0.2)
                return TokenizationResult(token_ids=[1, 2, 3])

        servicer = TokenizationServiceServicer(SlowTokenizer())
        request = pb.TokenizeRequest(model=REAL_MODEL_ID, prompt=REAL_PROMPT, add_special_tokens=True)

        started = time.perf_counter()
        first, second = await asyncio.gather(
            servicer.Tokenize(request, FakeContext()),
            servicer.Tokenize(request, FakeContext()),
        )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.33)
        self.assertEqual(list(first.result.token_ids), [1, 2, 3])
        self.assertEqual(list(second.result.token_ids), [1, 2, 3])

    async def test_render_completion_plain_text_tokenizes_directly(self) -> None:
        # Plain-text completions bypass the vLLM render client and tokenize
        # directly off the event loop; token IDs match a direct encode.
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(cache_dir=cache_dir)
            servicer = TokenizationServiceServicer(tokenizer)

            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )
            expected = tokenizer.tokenize(
                model=REAL_MODEL_ID, prompt=REAL_PROMPT, add_special_tokens=False
            ).token_ids
            response = await servicer.RenderCompletion(
                pb.RenderCompletionRequest(model=REAL_MODEL_ID, prompt_text=REAL_PROMPT),
                FakeContext(),
            )

            self.assertEqual(list(response.result.token_ids), list(expected))
            self.assertTrue(response.result.token_ids)

    async def test_render_completion_accepts_prompt_token_ids(self) -> None:
        render_client = _CapturingRenderClient()
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(
                cache_dir=cache_dir,
                render_factory=lambda resolved: render_client,
            )
            servicer = TokenizationServiceServicer(tokenizer)

            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )
            response = await servicer.RenderCompletion(
                pb.RenderCompletionRequest(
                    model=REAL_MODEL_ID,
                    prompt_token_ids=pb.TokenIdList(values=[11, 22, 33]),
                ),
                FakeContext(),
            )

            self.assertEqual(list(response.result.token_ids), [301, 302, 303])
            self.assertEqual(len(render_client.completion_requests), 1)
            self.assertEqual(
                render_client.completion_requests[0].prompt_token_ids,
                [11, 22, 33],
            )
            self.assertIsNone(render_client.completion_requests[0].prompt_text)

    async def test_render_chat_maps_failed_precondition(self) -> None:
        service = _build_service()
        servicer = TokenizationServiceServicer(service)
        with self.assertRaises(AbortCalled) as raised:
            await servicer.RenderChatCompletion(
                pb.RenderChatCompletionRequest(
                    model=REAL_MODEL_ID,
                    messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                ),
                FakeContext(),
            )
        self.assertEqual(raised.exception.code, grpc.StatusCode.FAILED_PRECONDITION)

    async def test_tokenize_before_initialize_raises_error(self) -> None:
        service = _build_service()
        servicer = TokenizationServiceServicer(service)

        with self.assertRaises(AbortCalled) as raised:
            await servicer.Tokenize(
                pb.TokenizeRequest(model=REAL_MODEL_ID, prompt=REAL_PROMPT, add_special_tokens=True),
                FakeContext(),
            )
        self.assertEqual(raised.exception.code, grpc.StatusCode.FAILED_PRECONDITION)

    async def test_tokenize_model_mismatch_maps_to_invalid_argument(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(cache_dir=cache_dir)
            servicer = TokenizationServiceServicer(tokenizer)

            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )

            with self.assertRaises(AbortCalled) as raised:
                await servicer.Tokenize(
                    pb.TokenizeRequest(model="other/model", prompt=REAL_PROMPT, add_special_tokens=True),
                    FakeContext(),
                )

            self.assertEqual(raised.exception.code, grpc.StatusCode.INVALID_ARGUMENT)
            self.assertIn("model mismatch", raised.exception.details)

    async def test_render_completion_before_initialize_raises_error(self) -> None:
        service = _build_service()
        servicer = TokenizationServiceServicer(service)

        with self.assertRaises(AbortCalled) as raised:
            await servicer.RenderCompletion(
                pb.RenderCompletionRequest(model=REAL_MODEL_ID, prompt_text=REAL_PROMPT),
                FakeContext(),
            )
        self.assertEqual(raised.exception.code, grpc.StatusCode.FAILED_PRECONDITION)

    async def test_render_completion_model_mismatch_maps_to_invalid_argument(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(cache_dir=cache_dir)
            servicer = TokenizationServiceServicer(tokenizer)

            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )

            with self.assertRaises(AbortCalled) as raised:
                await servicer.RenderCompletion(
                    pb.RenderCompletionRequest(model="other/model", prompt_text=REAL_PROMPT),
                    FakeContext(),
                )

            self.assertEqual(raised.exception.code, grpc.StatusCode.INVALID_ARGUMENT)
            self.assertIn("model mismatch", raised.exception.details)

    async def test_render_chat_completion_model_mismatch_maps_to_invalid_argument(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(cache_dir=cache_dir)
            servicer = TokenizationServiceServicer(tokenizer)

            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )

            with self.assertRaises(AbortCalled) as raised:
                await servicer.RenderChatCompletion(
                    pb.RenderChatCompletionRequest(
                        model="other/model",
                        messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                    ),
                    FakeContext(),
                )

            self.assertEqual(raised.exception.code, grpc.StatusCode.INVALID_ARGUMENT)
            self.assertIn("model mismatch", raised.exception.details)

    async def test_initialize_with_invalid_model_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            service = _build_service(cache_dir=cache_dir)
            servicer = TokenizationServiceServicer(service)

            with self.assertRaises(AbortCalled) as raised:
                await servicer.Initialize(
                    pb.InitializeRequest(model="InvalidModel/DoesNotExist"),
                    FakeContext(),
                )
            self.assertEqual(raised.exception.code, grpc.StatusCode.INVALID_ARGUMENT)

    async def test_initialize_with_empty_model_raises_error(self) -> None:
        service = _build_service()
        servicer = TokenizationServiceServicer(service)

        with self.assertRaises(AbortCalled) as raised:
            await servicer.Initialize(
                pb.InitializeRequest(model=""),
                FakeContext(),
            )
        self.assertEqual(raised.exception.code, grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("model cannot be empty", raised.exception.details)

    async def test_render_chat_completion_plain_text_tokenizes_directly(self) -> None:
        # Plain-text chat bypasses the vLLM render client and tokenizes through
        # the tokenizer chat template directly; token IDs match that template.
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(cache_dir=cache_dir)
            servicer = TokenizationServiceServicer(tokenizer)

            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )
            response = await servicer.RenderChatCompletion(
                pb.RenderChatCompletionRequest(
                    model=REAL_MODEL_ID,
                    messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                ),
                FakeContext(),
            )

            expected = tokenizer._provider.apply_chat_template(
                [{"role": "user", "content": REAL_PROMPT}],
                add_generation_prompt=False,
            )
            self.assertEqual(list(response.result.token_ids), list(expected))
            self.assertTrue(response.result.token_ids)

    async def test_render_chat_with_tools_uses_render_client(self) -> None:
        # Tool-calling chat is not a plain-text request, so it still flows
        # through the vLLM render client for tokenization fidelity.
        render_client = _CapturingRenderClient()
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(
                cache_dir=cache_dir,
                render_factory=lambda resolved: render_client,
            )
            servicer = TokenizationServiceServicer(tokenizer)

            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )
            response = await servicer.RenderChatCompletion(
                pb.RenderChatCompletionRequest(
                    model=REAL_MODEL_ID,
                    messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                    tools_json='[{"type":"function","function":{"name":"f"}}]',
                ),
                FakeContext(),
            )

            self.assertEqual(list(response.result.token_ids), [401, 402, 403])
            self.assertEqual(len(render_client.chat_requests), 1)

    async def test_render_chat_completion_invalid_json_maps_to_invalid_argument(self) -> None:
        invalid_requests = (
            pb.RenderChatCompletionRequest(
                model=REAL_MODEL_ID,
                messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                chat_template_kwargs_json="{",
            ),
            pb.RenderChatCompletionRequest(
                model=REAL_MODEL_ID,
                messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                tools_json="{",
            ),
            pb.RenderChatCompletionRequest(
                model=REAL_MODEL_ID,
                messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                tool_choice_json="{",
            ),
            pb.RenderChatCompletionRequest(
                model=REAL_MODEL_ID,
                messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                mm_processor_kwargs_json="{",
            ),
            pb.RenderChatCompletionRequest(
                model=REAL_MODEL_ID,
                messages=[pb.ChatMessage(role="user", content=REAL_PROMPT)],
                media_io_kwargs_json="{",
            ),
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = _build_service(cache_dir=cache_dir)
            servicer = TokenizationServiceServicer(tokenizer)

            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )

            for request in invalid_requests:
                with self.subTest(request=request):
                    with self.assertRaises(AbortCalled) as raised:
                        await servicer.RenderChatCompletion(request, FakeContext())
                    self.assertEqual(raised.exception.code, grpc.StatusCode.INVALID_ARGUMENT)
                    self.assertIn("Expecting property name enclosed in double quotes", raised.exception.details)

    async def test_initialize_with_render_factory_failure_raises_internal(self) -> None:
        def failing_render_factory(resolved):
            raise RuntimeError("render factory failed")

        service = TokenizerService(
            provider_name="huggingface",
            render_factory=failing_render_factory,
        )
        servicer = TokenizationServiceServicer(service)

        with self.assertRaises(AbortCalled) as raised:
            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )
        self.assertEqual(raised.exception.code, grpc.StatusCode.INTERNAL)

    async def test_initialize_with_render_factory_returns_none_raises_unimplemented(self) -> None:
        def null_render_factory(resolved):
            return None

        service = TokenizerService(
            provider_name="huggingface",
            render_factory=null_render_factory,
        )
        servicer = TokenizationServiceServicer(service)

        with self.assertRaises(AbortCalled) as raised:
            await servicer.Initialize(
                pb.InitializeRequest(model=REAL_MODEL_ID),
                FakeContext(),
            )
        self.assertEqual(raised.exception.code, grpc.StatusCode.UNIMPLEMENTED)
