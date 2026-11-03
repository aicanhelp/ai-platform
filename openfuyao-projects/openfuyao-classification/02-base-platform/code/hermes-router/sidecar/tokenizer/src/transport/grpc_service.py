# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""gRPC adapter for the tokenizer sidecar."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import grpc

from core import (
    ChatCompletionRequest,
    ChatContentPart,
    ChatMessage,
    CompletionRequest,
    MediaRef,
    MultiModalFeature,
    NotInitializedError,
    RenderUnavailableError,
    TokenizationResult,
    TokenizerService,
)
from providers.resolution import TokenizerResolutionError
from proto import tokenizer_pb2 as pb
from proto import tokenizer_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)

CHAT_WARMUP_PROMPT = "warmup"

PROVIDER_STRING_TO_PROTO = {
    "huggingface": pb.TOKENIZER_PROVIDER_HUGGINGFACE,
    "modelscope": pb.TOKENIZER_PROVIDER_MODELSCOPE,
}

MODALITY_STRING_TO_PROTO = {
    "image": pb.MODALITY_IMAGE,
}


class TokenizationServiceServicer(pb_grpc.TokenizationServiceServicer):
    """Thin gRPC adapter over the core tokenizer service."""

    def __init__(self, tokenizer: TokenizerService) -> None:
        self._tokenizer = tokenizer

    async def Initialize(
        self,
        request: pb.InitializeRequest,
        context: grpc.aio.ServicerContext,
    ):  # noqa: N802
        if not request.model:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model cannot be empty")

        tokenizer_name = request.tokenizer_name if request.HasField("tokenizer_name") else None

        try:
            result = await asyncio.to_thread(
                self._tokenizer.initialize,
                model=request.model,
                tokenizer_name=tokenizer_name,
            )
        except (TokenizerResolutionError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RenderUnavailableError as exc:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
        except Exception as exc:
            logger.exception("tokenizer initialization failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"initialize failed: {exc}")

        if self._tokenizer.has_usable_chat_template():
            warmup_started = time.perf_counter()
            try:
                await self._tokenizer.render_chat_completion(
                    ChatCompletionRequest(
                        model=request.model,
                        messages=[ChatMessage(role="user", content=CHAT_WARMUP_PROMPT)],
                        add_generation_prompt=True,
                    )
                )
            except RenderUnavailableError as exc:
                await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
            except Exception as exc:
                logger.exception("tokenizer chat warmup failed")
                await context.abort(grpc.StatusCode.INTERNAL, f"initialize warmup failed: {exc}")

            logger.info(
                "tokenizer chat warmup completed for model %s in %.2f ms",
                request.model,
                (time.perf_counter() - warmup_started) * 1000,
            )
        else:
            # No chat template is a model/artifact property, not a sidecar
            # health problem: warmup would only fail, so skip it and let the
            # runtime path fail open for chat requests without a template.
            logger.warning("tokenizer has no chat template; skipping warmup", extra={"model": request.model})

        return pb.InitializeResponse(
            model=result.model,
            resolved_provider=PROVIDER_STRING_TO_PROTO.get(
                result.resolved_provider,
                pb.TOKENIZER_PROVIDER_UNSPECIFIED,
            ),
            resolved_tokenizer_name=result.resolved_tokenizer_name,
        )

    async def Tokenize(
        self,
        request: pb.TokenizeRequest,
        context: grpc.aio.ServicerContext,
    ):  # noqa: N802
        try:
            result = await asyncio.to_thread(
                self._tokenizer.tokenize,
                model=request.model,
                prompt=request.prompt,
                add_special_tokens=request.add_special_tokens,
            )
        except NotInitializedError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            logger.exception("tokenize request failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"tokenize failed: {exc}")

        return pb.TokenizeResponse(result=to_proto_result(result))

    async def RenderCompletion(
        self,
        request: pb.RenderCompletionRequest,
        context: grpc.aio.ServicerContext,
    ):  # noqa: N802
        prompt_source = request.WhichOneof("prompt_source")
        completion_request = CompletionRequest(
            model=request.model,
            prompt_text=request.prompt_text if prompt_source == "prompt_text" else None,
            prompt_token_ids=(
                list(request.prompt_token_ids.values)
                if prompt_source == "prompt_token_ids"
                else None
            ),
            add_special_tokens=request.add_special_tokens,
            truncate_prompt_tokens=(
                request.truncate_prompt_tokens if request.HasField("truncate_prompt_tokens") else None
            ),
        )

        if completion_request.prompt_text is None and completion_request.prompt_token_ids is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "completion request requires prompt_text or prompt_token_ids",
            )

        try:
            result = await self._tokenizer.render_completion(completion_request)
        except NotInitializedError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except RenderUnavailableError as exc:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            logger.exception("render completion request failed")
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"render completion failed: {exc}",
            )

        return pb.RenderCompletionResponse(result=to_proto_result(result))

    async def RenderChatCompletion(
        self,
        request: pb.RenderChatCompletionRequest,
        context: grpc.aio.ServicerContext,
    ):  # noqa: N802
        if len(request.messages) == 0:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "chat completion request requires at least one message",
            )

        try:
            chat_request = ChatCompletionRequest(
                model=request.model,
                messages=[from_proto_message(message) for message in request.messages],
                chat_template=request.chat_template,
                chat_template_kwargs=parse_json_string(request.chat_template_kwargs_json),
                tools=parse_json_string(request.tools_json),
                tool_choice=parse_json_string(request.tool_choice_json),
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                mm_processor_kwargs=parse_json_string(request.mm_processor_kwargs_json),
                media_io_kwargs=parse_json_string(request.media_io_kwargs_json),
            )
            result = await self._tokenizer.render_chat_completion(chat_request)
        except NotInitializedError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except RenderUnavailableError as exc:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            logger.exception("render chat completion request failed")
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"render chat completion failed: {exc}",
            )

        return pb.RenderChatCompletionResponse(result=to_proto_result(result))


def parse_json_string(value: str) -> Any:
    """Parse a JSON-encoded proto field into Python data."""

    if value == "":
        return None
    return json.loads(value)


def from_proto_message(message: pb.ChatMessage) -> ChatMessage:
    """Convert a protobuf chat message into the domain model."""

    content_parts: list[ChatContentPart] = []
    proto_content_parts = message.content_parts if message.WhichOneof("content_form") == "content_parts" else None
    for part in proto_content_parts.values if proto_content_parts is not None else []:
        media = None
        if part.WhichOneof("part") == "media":
            media = MediaRef(
                modality=from_proto_modality(part.media.modality),
                url=part.media.url,
                inline_data=part.media.inline_data,
                mime_type=part.media.mime_type,
                detail_json=part.media.detail_json,
            )
        content_parts.append(ChatContentPart(text=part.text, media=media))

    return ChatMessage(
        role=message.role,
        content=message.content if message.WhichOneof("content_form") == "content" else "",
        content_parts=content_parts,
        name=message.name,
        tool_calls_json=message.tool_calls_json,
        tool_call_id=message.tool_call_id,
    )


def to_proto_result(result: TokenizationResult) -> pb.TokenizationResult:
    """Convert a normalized tokenization result to protobuf."""

    return pb.TokenizationResult(
        token_ids=result.token_ids,
        multimodal_features=[to_proto_feature(feature) for feature in result.multimodal_features],
    )


def to_proto_feature(feature: MultiModalFeature) -> pb.MultiModalFeature:
    """Convert a normalized multimodal feature to protobuf."""

    return pb.MultiModalFeature(
        modality=MODALITY_STRING_TO_PROTO.get(feature.modality.lower(), pb.MODALITY_UNSPECIFIED),
        hash=feature.hash,
        offset=feature.offset,
        length=feature.length,
    )


def from_proto_modality(modality: int) -> str:
    """Convert a protobuf modality enum into the normalized string form."""

    if modality == pb.MODALITY_IMAGE:
        return "image"
    logger.warning("unknown modality received from protobuf", extra={"modality": modality})
    return ""