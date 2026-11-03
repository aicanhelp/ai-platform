# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared core models for the tokenizer sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class InitializeResult:
    """Result returned by a successful Initialize call."""

    model: str
    resolved_provider: str
    resolved_tokenizer_name: str


@dataclass(slots=True)
class MultiModalFeature:
    """Normalized multimodal placeholder metadata."""

    modality: str
    hash: str
    offset: int
    length: int


@dataclass(slots=True)
class TokenizationResult:
    """Normalized tokenization output shared by all RPC paths."""

    token_ids: list[int] = field(default_factory=list)
    multimodal_features: list[MultiModalFeature] = field(default_factory=list)


@dataclass(slots=True)
class CompletionRequest:
    """Input for render-backed completion tokenization."""

    model: str
    prompt_text: str | None = None
    prompt_token_ids: list[int] | None = None
    add_special_tokens: bool = True
    truncate_prompt_tokens: int | None = None


@dataclass(slots=True)
class MediaRef:
    """Normalized media reference used in structured chat content."""

    modality: str
    url: str = ""
    inline_data: bytes = b""
    mime_type: str = ""
    detail_json: str = ""


@dataclass(slots=True)
class ChatContentPart:
    """One structured content part for a chat message."""

    text: str = ""
    media: MediaRef | None = None


@dataclass(slots=True)
class ChatMessage:
    """Normalized chat message used by the render layer."""

    role: str
    content: str = ""
    content_parts: list[ChatContentPart] = field(default_factory=list)
    name: str = ""
    tool_calls_json: str = ""
    tool_call_id: str = ""


@dataclass(slots=True)
class ChatCompletionRequest:
    """Input for render-backed chat tokenization."""

    model: str
    messages: list[ChatMessage] = field(default_factory=list)
    chat_template: str = ""
    chat_template_kwargs: dict[str, Any] | None = None
    tools: Any = None
    tool_choice: Any = None
    add_generation_prompt: bool = False
    continue_final_message: bool = False
    mm_processor_kwargs: dict[str, Any] | None = None
    media_io_kwargs: dict[str, dict[str, Any]] | None = None
