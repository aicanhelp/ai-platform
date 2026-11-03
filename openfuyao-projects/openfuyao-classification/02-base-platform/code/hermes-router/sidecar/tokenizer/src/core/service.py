# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Core tokenizer service and tokenizer-path resolution."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from core.errors import NotInitializedError, RenderUnavailableError
from core.models import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    InitializeResult,
    TokenizationResult,
)
from providers.base import TokenizerProvider
from providers.huggingface import HuggingFaceProvider
from providers.modelscope import ModelScopeProvider
from providers.resolution import (
    ResolvedTokenizer,
    TokenizerResolutionError,
    _is_remote_model,
    normalize_provider_name,
)

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_CACHE_DIR = "~/.cache/huggingface/hub"
DEFAULT_TRUST_REMOTE_CODE = True
_CORE_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
_TOKENIZER_MODEL_FILES = (
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
)
_TOKENIZER_FILE_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "chat_template.json",
    "*.jinja",
)


class RenderClient(Protocol):
    """Render-backed tokenization interface."""

    async def render_completion(self, request: CompletionRequest) -> TokenizationResult:
        """Render and tokenize a completion request."""

    async def render_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> TokenizationResult:
        """Render and tokenize a chat-completion request."""


RenderFactory = Callable[[ResolvedTokenizer], RenderClient]


class TokenizerService:
    """Core tokenizer implementation shared by the gRPC service."""

    def __init__(
        self,
        *,
        provider_name: str,
        cache_dir: str | None = None,
        render_factory: RenderFactory | None = None,
    ) -> None:
        self._provider_name = normalize_provider_name(provider_name)
        self._cache_dir = Path(cache_dir or DEFAULT_TOKENIZER_CACHE_DIR).expanduser()
        self._render_factory = render_factory
        self._provider: TokenizerProvider | None = None
        self._resolved_tokenizer: ResolvedTokenizer | None = None
        self._render_client: RenderClient | None = None
        self._init_lock = threading.Lock()
        self._last_init: InitializeResult | None = None
        self._last_init_key: tuple[str, str | None] | None = None

    def initialize(
        self,
        *,
        model: str,
        tokenizer_name: str | None = None,
    ) -> InitializeResult:
        """Single-flight initialize: waiters share one in-flight result.

        Failures are not cached. The lock is held by the worker thread, so a
        cancelled RPC does not abort an in-flight import.
        """

        if self._render_factory is None:
            raise RenderUnavailableError("vLLM render is not configured")

        # "" and None are the same key: the EPP client omits the proto field
        # for empty strings, but other clients may send "" explicitly.
        key = (model, tokenizer_name or None)
        with self._init_lock:
            if self._last_init is not None and self._last_init_key == key:
                return self._last_init

            load_path = self._resolve_load_path(tokenizer_name or model)
            resolved = ResolvedTokenizer(
                served_model_name=model,
                model_path=load_path,
                tokenizer_path=load_path,
                provider=self._provider_name,
                trust_remote_code=DEFAULT_TRUST_REMOTE_CODE,
            )
            provider = self._create_provider(resolved)
            render_client = self._render_factory(resolved)

            if render_client is None:
                raise RenderUnavailableError("render_factory returned None")

            self._provider = provider
            self._resolved_tokenizer = resolved
            self._render_client = render_client

            self._last_init = InitializeResult(
                model=model,
                resolved_provider=provider.provider_name,
                resolved_tokenizer_name=provider.resolved_name,
            )
            self._last_init_key = key
            return self._last_init

    def has_usable_chat_template(self) -> bool:
        """Report whether the initialized tokenizer can render chat templates.

        A missing chat template is a model/artifact property rather than a
        sidecar health problem, so callers can skip the chat warmup instead of
        failing initialization.
        """
        return self._provider is not None and self._provider.has_chat_template()

    def tokenize(
        self,
        *,
        model: str,
        prompt: str,
        add_special_tokens: bool = True,
    ) -> TokenizationResult:
        """Tokenize a plain-text prompt via the configured provider."""

        provider = self._require_provider()
        self._validate_model_consistency(model)
        return TokenizationResult(
            token_ids=provider.encode(prompt, add_special_tokens=add_special_tokens)
        )

    async def render_completion(self, request: CompletionRequest) -> TokenizationResult:
        """Tokenize a completion request, off the event loop when possible.

        Plain-text prompts are tokenized directly in a worker thread so that
        concurrent requests run in parallel. Token-id prompts and truncation
        still go through the vLLM render path, which runs on the event loop.
        """

        self._validate_model_consistency(request.model)
        if self._can_tokenize_completion_directly(request):
            return await asyncio.to_thread(self._tokenize_completion_directly, request)
        render_client = self._require_render_client()
        return await render_client.render_completion(request)

    async def render_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> TokenizationResult:
        """Tokenize a chat request, off the event loop when possible.

        Plain-text chats (no tools, no multimodal, no structured content) are
        rendered through the tokenizer chat template directly in a worker
        thread so that concurrent requests run in parallel; this yields token
        IDs identical to the vLLM render path. Tool/multimodal requests still
        go through the vLLM render path, which runs on the event loop.
        """

        self._validate_model_consistency(request.model)
        if self._can_tokenize_chat_directly(request):
            return await asyncio.to_thread(self._tokenize_chat_directly, request)
        render_client = self._require_render_client()
        return await render_client.render_chat_completion(request)

    def _can_tokenize_completion_directly(self, request: CompletionRequest) -> bool:
        return (
            self._provider is not None
            and request.prompt_text is not None
            and request.prompt_token_ids is None
            and request.truncate_prompt_tokens is None
        )

    def _tokenize_completion_directly(self, request: CompletionRequest) -> TokenizationResult:
        provider = self._require_provider()
        token_ids = provider.encode(
            request.prompt_text,
            add_special_tokens=request.add_special_tokens,
        )
        return TokenizationResult(token_ids=token_ids)

    def _can_tokenize_chat_directly(self, request: ChatCompletionRequest) -> bool:
        if self._provider is None:
            return False
        if request.tools is not None or request.tool_choice is not None:
            return False
        if request.mm_processor_kwargs is not None or request.media_io_kwargs is not None:
            return False
        return all(_is_plain_text_message(message) for message in request.messages)

    def _tokenize_chat_directly(self, request: ChatCompletionRequest) -> TokenizationResult:
        provider = self._require_provider()
        conversation = [_to_template_message(message) for message in request.messages]
        token_ids = provider.apply_chat_template(
            conversation,
            chat_template=request.chat_template or None,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            chat_template_kwargs=request.chat_template_kwargs,
        )
        return TokenizationResult(token_ids=token_ids)

    def _validate_model_consistency(self, requested_model: str) -> None:
        if self._resolved_tokenizer is None:
            raise NotInitializedError("tokenizer has not been initialized")
        if requested_model != self._resolved_tokenizer.served_model_name:
            raise ValueError(
                f"model mismatch: requested={requested_model!r}, "
                f"initialized={self._resolved_tokenizer.served_model_name!r}"
            )

    def _resolve_load_path(self, identifier: str) -> str:
        if not _is_remote_model(identifier):
            local_path = Path(identifier).expanduser()
            if not local_path.exists():
                raise TokenizerResolutionError(
                    model=identifier,
                    provider=self._provider_name,
                    tokenizer_name=identifier,
                    message="local path does not exist",
                )
            return str(local_path)

        cached_path = self._resolve_target_cache_path(identifier)
        if cached_path is not None and self._cache_is_complete(cached_path):
            logger.info("using cached tokenizer", extra={"identifier": identifier, "path": str(cached_path)})
            return str(cached_path)

        return self._download_tokenizer(identifier)

    def _resolve_target_cache_path(self, identifier: str) -> Path | None:
        if self._provider_name == "huggingface":
            snapshots_dir = self._cache_dir / f"models--{identifier.replace('/', '--')}" / "snapshots"
            if not snapshots_dir.is_dir():
                return None
            candidates = sorted(
                (path for path in snapshots_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            return candidates[0] if candidates else None

        if self._provider_name == "modelscope":
            for candidate in (
                self._cache_dir / "models" / identifier,
                self._cache_dir / identifier,
            ):
                if candidate.exists():
                    return candidate
            return None

        raise ValueError(f"unsupported provider {self._provider_name!r}")

    def _cache_is_complete(self, load_path: Path) -> bool:
        required_config = load_path / "tokenizer_config.json"
        if not required_config.is_file():
            return False

        artifact_names = set(_CORE_TOKENIZER_FILES) | set(_TOKENIZER_MODEL_FILES)
        if not any((load_path / name).exists() for name in artifact_names):
            return False

        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(
                str(load_path),
                local_files_only=True,
                trust_remote_code=DEFAULT_TRUST_REMOTE_CODE,
                use_fast=True,
            )
        except Exception:
            logger.warning("cached tokenizer validation failed", extra={"path": str(load_path)}, exc_info=True)
            return False
        return True

    def _download_tokenizer(self, identifier: str) -> str:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if self._provider_name == "huggingface":
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError, RevisionNotFoundError

            try:
                path = snapshot_download(
                    repo_id=identifier,
                    cache_dir=str(self._cache_dir),
                    allow_patterns=list(_TOKENIZER_FILE_PATTERNS),
                )
            except (HFValidationError, RepositoryNotFoundError, RevisionNotFoundError) as exc:
                raise TokenizerResolutionError(
                    model=identifier,
                    provider=self._provider_name,
                    tokenizer_name=identifier,
                    message=str(exc),
                ) from exc
            return str(Path(path).expanduser())

        if self._provider_name == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download

            path = snapshot_download(
                model_id=identifier,
                cache_dir=str(self._cache_dir),
                allow_patterns=list(_TOKENIZER_FILE_PATTERNS),
            )
            return str(Path(path).expanduser())

        raise ValueError(f"unsupported provider {self._provider_name!r}")

    def _create_provider(self, resolved: ResolvedTokenizer) -> TokenizerProvider:
        provider_factories: dict[str, type[TokenizerProvider]] = {
            "huggingface": HuggingFaceProvider,
            "modelscope": ModelScopeProvider,
        }
        provider_factory = provider_factories.get(resolved.provider)
        if provider_factory is None:
            raise ValueError(f"unsupported provider {resolved.provider!r}")
        return provider_factory(resolved)

    def _require_provider(self) -> TokenizerProvider:
        if self._provider is None:
            raise NotInitializedError("tokenizer has not been initialized")
        return self._provider

    def _require_render_client(self) -> RenderClient:
        if self._render_client is None:
            if self._resolved_tokenizer is None:
                raise NotInitializedError("tokenizer has not been initialized")
            raise RenderUnavailableError("vLLM render is not configured")
        return self._render_client


def _is_plain_text_message(message: ChatMessage) -> bool:
    """Report whether a chat message carries only plain text the chat template can render."""
    return (
        not message.content_parts
        and not message.tool_calls_json
        and not message.tool_call_id
    )


def _to_template_message(message: ChatMessage) -> dict[str, str]:
    payload = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    return payload
