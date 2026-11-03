# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""vLLM render adapter for tokenizer sidecar render RPCs."""

from __future__ import annotations

import base64
import contextlib
import json
import os
from dataclasses import dataclass
from typing import Any

from core.models import (
    ChatCompletionRequest,
    ChatContentPart,
    ChatMessage,
    CompletionRequest,
    MediaRef,
    MultiModalFeature,
    TokenizationResult,
)
from providers import ResolvedTokenizer


@dataclass(slots=True)
class RenderSettings:
    """Initialization settings for the vLLM render path."""

    tokenizer: str | None = None
    chat_template: str | None = None
    chat_template_content_format: str = "auto"
    served_model_name: str | None = None
    num_workers: int | None = None
    trust_request_chat_template: bool = False


MAX_DEFAULT_RENDER_WORKERS = 8


def _available_cpu_count() -> int:
    with contextlib.suppress(AttributeError, OSError):
        available = len(os.sched_getaffinity(0))
        if available > 0:
            return available
    return os.cpu_count() or 1


def resolve_render_num_workers(configured_workers: int | None) -> int:
    if configured_workers is not None:
        if configured_workers < 1:
            raise ValueError("render_num_workers must be at least 1")
        return configured_workers

    return min(MAX_DEFAULT_RENDER_WORKERS, _available_cpu_count())


class VLLMRenderClient:
    """Adapter that delegates render-backed tokenization to OpenAIServingRender."""

    def __init__(
        self,
        serving_render: Any,
        *,
        completion_request_cls: type | None = None,
        chat_request_cls: type | None = None,
    ) -> None:
        self._serving_render = serving_render
        self._completion_request_cls = completion_request_cls
        self._chat_request_cls = chat_request_cls

    async def render_completion(self, request: CompletionRequest) -> TokenizationResult:
        """Render a single logical completion prompt through vLLM."""

        completion_request = self._build_completion_request(request)
        result = await self._serving_render.render_completion_request(completion_request)
        self._raise_if_error_response(result)
        if len(result) != 1:
            raise ValueError(
                f"expected 1 render completion result, got {len(result)}"
            )
        return self._to_tokenization_result(result[0])

    async def render_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> TokenizationResult:
        """Render a single chat-completion request through vLLM."""

        chat_request = self._build_chat_request(request)
        result = await self._serving_render.render_chat_request(chat_request)
        self._raise_if_error_response(result)
        return self._to_tokenization_result(result)

    def _build_completion_request(self, request: CompletionRequest) -> Any:
        completion_request_cls, _ = self._request_types()
        return completion_request_cls(
            model=request.model,
            prompt=self._completion_prompt(request),
            add_special_tokens=request.add_special_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )

    def _build_chat_request(self, request: ChatCompletionRequest) -> Any:
        _, chat_request_cls = self._request_types()
        return chat_request_cls(
            model=request.model,
            messages=[self._message_payload(message) for message in request.messages],
            chat_template=request.chat_template or None,
            chat_template_kwargs=request.chat_template_kwargs,
            tools=request.tools,
            tool_choice=request.tool_choice,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            mm_processor_kwargs=request.mm_processor_kwargs,
            media_io_kwargs=request.media_io_kwargs,
        )

    def _request_types(self) -> tuple[type, type]:
        if self._completion_request_cls is not None and self._chat_request_cls is not None:
            return self._completion_request_cls, self._chat_request_cls

        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest as VLLMChatCompletionRequest,
        )
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionRequest as VLLMCompletionRequest,
        )

        return (
            self._completion_request_cls or VLLMCompletionRequest,
            self._chat_request_cls or VLLMChatCompletionRequest,
        )

    def _completion_prompt(self, request: CompletionRequest) -> str | list[int]:
        has_text = request.prompt_text is not None
        has_token_ids = request.prompt_token_ids is not None
        if has_text == has_token_ids:
            raise ValueError("completion request must provide exactly one prompt source")
        if request.prompt_text is not None:
            return request.prompt_text
        return request.prompt_token_ids

    def _message_payload(self, message: ChatMessage) -> dict[str, Any]:
        if message.content and message.content_parts:
            raise ValueError(
                "chat message must use exactly one content form: content or content_parts"
            )

        payload: dict[str, Any] = {"role": message.role}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls_json:
            payload["tool_calls"] = json.loads(message.tool_calls_json)

        if message.content_parts:
            payload["content"] = [self._content_part_payload(part) for part in message.content_parts]
        else:
            payload["content"] = message.content
        return payload

    def _content_part_payload(self, part: ChatContentPart) -> dict[str, Any]:
        if part.media is None:
            return {"type": "text", "text": part.text}
        return self._media_payload(part.media)

    def _media_payload(self, media: MediaRef) -> dict[str, Any]:
        if media.modality != "image":
            raise ValueError(f"unsupported media modality: {media.modality}")

        image_url: dict[str, Any] = {"url": self._media_url(media)}
        if media.detail_json:
            detail = json.loads(media.detail_json)
            if not isinstance(detail, dict):
                raise ValueError("media detail_json must encode a JSON object")
            image_url.update(detail)
        return {"type": "image_url", "image_url": image_url}

    def _media_url(self, media: MediaRef) -> str:
        if media.url:
            return media.url
        if media.inline_data:
            mime_type = media.mime_type or "application/octet-stream"
            encoded = base64.b64encode(media.inline_data).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
        raise ValueError("media reference requires url or inline_data")

    def _raise_if_error_response(self, result: Any) -> None:
        if not self._is_error_response(result):
            return
        error = getattr(result, "error", None)
        if error is not None:
            message = getattr(error, "message", None) or str(result)
        else:
            message = str(result)
        raise ValueError(message)

    def _is_error_response(self, result: Any) -> bool:
        return hasattr(result, "error") and hasattr(result.error, "message")

    def _to_tokenization_result(self, generate_request: Any) -> TokenizationResult:
        token_ids = list(getattr(generate_request, "token_ids", []) or [])
        if not token_ids:
            raise ValueError("render returned no token_ids")
        return TokenizationResult(
            token_ids=token_ids,
            multimodal_features=self._extract_multimodal_features(
                getattr(generate_request, "features", None)
            ),
        )

    def _extract_multimodal_features(self, features: Any) -> list[MultiModalFeature]:
        if features is None:
            return []

        raw_hashes = self._as_mapping_or_attr(features, "mm_hashes") or {}
        raw_placeholders = self._as_mapping_or_attr(features, "mm_placeholders") or {}
        multimodal_features: list[MultiModalFeature] = []

        for modality, hashes in self._items(raw_hashes):
            placeholders = raw_placeholders.get(modality, []) if isinstance(raw_placeholders, dict) else []
            for item_hash, placeholder in zip(list(hashes), list(placeholders)):
                offset = self._value(placeholder, "offset")
                length = self._value(placeholder, "length")
                multimodal_features.append(
                    MultiModalFeature(
                        modality=str(modality),
                        hash=str(item_hash),
                        offset=int(offset),
                        length=int(length),
                    )
                )

        return multimodal_features

    def _as_mapping_or_attr(self, value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    def _items(self, value: Any) -> list[tuple[Any, Any]]:
        if isinstance(value, dict):
            return list(value.items())
        if hasattr(value, "items"):
            return list(value.items())
        return []

    def _value(self, value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value[name]
        return getattr(value, name)


def create_vllm_render_client(
    resolved: ResolvedTokenizer,
    settings: RenderSettings | None = None,
) -> VLLMRenderClient:
    """Create a vLLM render client for a resolved tokenizer/model pair."""

    settings = settings or RenderSettings()
    model_path = resolved.model_path
    tokenizer_path = settings.tokenizer or resolved.tokenizer_path

    try:
        from vllm.config import DeviceConfig, VllmConfig
        from vllm.engine.arg_utils import EngineArgs
        from vllm.entrypoints.chat_utils import load_chat_template
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
        from vllm.entrypoints.serve.render.serving import OpenAIServingRender
        from vllm.plugins.io_processors import get_io_processor
        from vllm.renderers import renderer_from_config
    except Exception as exc:
        raise RuntimeError(
            "failed to import vllm dependencies "
            f"(model={model_path}, tokenizer={tokenizer_path}): {exc}"
        ) from exc

    try:
        engine_args = EngineArgs(
            model=model_path,
            tokenizer=tokenizer_path,
            trust_remote_code=resolved.trust_remote_code,
        )
        model_config = engine_args.create_model_config()
        model_config.quantization = None
        model_config.renderer_num_workers = resolve_render_num_workers(
            settings.num_workers
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            device_config=DeviceConfig(device="cpu"),
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to build vllm engine/model config "
            f"(model={model_path}, tokenizer={tokenizer_path}): {exc}"
        ) from exc

    try:
        renderer = renderer_from_config(vllm_config)
        io_processor = get_io_processor(
            vllm_config,
            renderer,
            model_config.io_processor_plugin,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to create renderer "
            f"(model={model_path}, tokenizer={tokenizer_path}): {exc}"
        ) from exc

    try:
        model_registry = OpenAIModelRegistry(
            model_config=model_config,
            base_model_paths=[
                BaseModelPath(
                    name=settings.served_model_name or resolved.served_model_name,
                    model_path=model_path,
                )
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to build model registry "
            f"(model={model_path}, tokenizer={tokenizer_path}): {exc}"
        ) from exc

    try:
        resolved_chat_template = (
            load_chat_template(settings.chat_template) if settings.chat_template else None
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to load chat template "
            f"(model={model_path}, tokenizer={tokenizer_path}): {exc}"
        ) from exc

    try:
        serving_render = OpenAIServingRender(
            model_config=model_config,
            renderer=renderer,
            io_processor=io_processor,
            model_registry=model_registry,
            request_logger=None,
            chat_template=resolved_chat_template,
            chat_template_content_format=settings.chat_template_content_format,
            trust_request_chat_template=settings.trust_request_chat_template,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to initialize serving render "
            f"(model={model_path}, tokenizer={tokenizer_path}): {exc}"
        ) from exc
    return VLLMRenderClient(serving_render)