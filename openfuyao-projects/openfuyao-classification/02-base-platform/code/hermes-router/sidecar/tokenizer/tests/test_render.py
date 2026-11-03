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

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import render

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "proto"))

from core.models import (
    ChatCompletionRequest,
    ChatContentPart,
    ChatMessage,
    CompletionRequest,
    MediaRef,
)
from providers.huggingface import HuggingFaceProvider
from providers.resolution import ResolvedTokenizer
from render import RenderSettings
from render import VLLMRenderClient
from render import create_vllm_render_client
from render import resolve_render_num_workers

import app

REAL_MODEL_ID = "Qwen/Qwen3-8B"


class _FakeServingRender:
    def __init__(self, *, chat_result: object | None = None) -> None:
        self.chat_requests: list[object] = []
        self._chat_result = chat_result or SimpleNamespace(token_ids=[7, 8, 9], features=None)

    async def render_chat_request(self, request: object) -> object:
        self.chat_requests.append(request)
        return self._chat_result


class _FakeChatRequest:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _build_stubbed_chat_render_client(
    *,
    chat_result: object | None = None,
) -> tuple[VLLMRenderClient, _FakeServingRender]:
    serving_render = _FakeServingRender(chat_result=chat_result)
    client = VLLMRenderClient(
        serving_render,
        completion_request_cls=object,
        chat_request_cls=_FakeChatRequest,
    )
    return client, serving_render


def _build_hf_provider(identifier: str) -> HuggingFaceProvider:
    return HuggingFaceProvider(
        ResolvedTokenizer(
            served_model_name=REAL_MODEL_ID,
            model_path=identifier,
            tokenizer_path=identifier,
            provider="huggingface",
            trust_remote_code=True,
        )
    )


def _build_local_runtime_client(tmp_path: Path):
    remote_provider = _build_hf_provider(REAL_MODEL_ID)
    local_dir = tmp_path / "local-tokenizer"
    tokenizer = getattr(remote_provider, "_tokenizer")
    tokenizer.save_pretrained(local_dir)
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=REAL_MODEL_ID, filename="config.json")
    shutil.copy2(config_path, local_dir / "config.json")

    resolved = ResolvedTokenizer(
        served_model_name=REAL_MODEL_ID,
        model_path=str(local_dir),
        tokenizer_path=str(local_dir),
        provider="huggingface",
        trust_remote_code=True,
    )
    return create_vllm_render_client(resolved)


@pytest.mark.asyncio
async def test_completion_render_accepts_single_prompt(tmp_path: Path) -> None:
    render_client = _build_local_runtime_client(tmp_path)
    result = await render_client.render_completion(
        CompletionRequest(model=REAL_MODEL_ID, prompt_text="hello")
    )
    assert result.token_ids


@pytest.mark.asyncio
async def test_chat_render_builds_openai_content_blocks(tmp_path: Path) -> None:
    render_client = _build_local_runtime_client(tmp_path)
    result = await render_client.render_chat_completion(
        ChatCompletionRequest(
            model=REAL_MODEL_ID,
            messages=[
                ChatMessage(
                    role="user",
                    content_parts=[
                        ChatContentPart(text="describe"),
                        ChatContentPart(text="the scene"),
                    ],
                )
            ],
            add_generation_prompt=True,
        )
    )
    assert result.token_ids


@pytest.mark.asyncio
async def test_chat_render_rejects_mixed_content_forms(tmp_path: Path) -> None:
    render_client = _build_local_runtime_client(tmp_path)

    with pytest.raises(ValueError, match="exactly one content form"):
        await render_client.render_chat_completion(
            ChatCompletionRequest(
                model=REAL_MODEL_ID,
                messages=[
                    ChatMessage(
                        role="user",
                        content="plain text",
                        content_parts=[ChatContentPart(text="structured text")],
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_chat_render_rejects_unsupported_media_modality(tmp_path: Path) -> None:
    render_client = _build_local_runtime_client(tmp_path)

    with pytest.raises(ValueError, match="unsupported media modality"):
        await render_client.render_chat_completion(
            ChatCompletionRequest(
                model=REAL_MODEL_ID,
                messages=[
                    ChatMessage(
                        role="user",
                        content_parts=[
                            ChatContentPart(
                                media=MediaRef(
                                    modality="audio",
                                    url="https://example.com/demo.wav",
                                )
                            )
                        ],
                    )
                ],
            )
        )


def test_render_client_initialization_requires_valid_tokenizer_path(tmp_path: Path) -> None:
    invalid_path = tmp_path / "nonexistent-tokenizer"
    
    resolved = ResolvedTokenizer(
        served_model_name=REAL_MODEL_ID,
        model_path=str(invalid_path),
        tokenizer_path=str(invalid_path),
        provider="huggingface",
        trust_remote_code=True,
    )
    
    with pytest.raises(RuntimeError, match="failed to"):
        create_vllm_render_client(resolved)


def test_render_client_initialization_with_missing_config(tmp_path: Path) -> None:
    incomplete_path = tmp_path / "incomplete-tokenizer"
    incomplete_path.mkdir()
    (incomplete_path / "tokenizer.json").write_text("{}")
    
    resolved = ResolvedTokenizer(
        served_model_name=REAL_MODEL_ID,
        model_path=str(incomplete_path),
        tokenizer_path=str(incomplete_path),
        provider="huggingface",
        trust_remote_code=True,
    )
    
    with pytest.raises(RuntimeError, match="failed to"):
        create_vllm_render_client(resolved)


def test_build_render_factory_allows_request_chat_template_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[RenderSettings] = []

    def fake_create_vllm_render_client(
        resolved: ResolvedTokenizer,
        settings: RenderSettings | None = None,
    ) -> object:
        del resolved
        captured.append(settings or RenderSettings())
        return object()

    monkeypatch.setattr(app, "create_vllm_render_client", fake_create_vllm_render_client)
    factory = app.build_render_factory(
        SimpleNamespace(
            render_tokenizer="",
            render_chat_template="",
            render_chat_template_content_format="auto",
            render_num_workers=6,
            render_trust_request_chat_template=True,
        )
    )

    factory(
        ResolvedTokenizer(
            served_model_name=REAL_MODEL_ID,
            model_path="/models/qwen",
            tokenizer_path="/models/qwen",
            provider="huggingface",
            trust_remote_code=True,
        )
    )

    assert len(captured) == 1
    assert captured[0].trust_request_chat_template is True
    assert captured[0].num_workers == 6


def test_resolve_render_num_workers_uses_available_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_available_cpu_count", lambda: 6)

    assert resolve_render_num_workers(None) == 6


def test_resolve_render_num_workers_caps_default_pool_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_available_cpu_count", lambda: 64)

    assert resolve_render_num_workers(None) == render.MAX_DEFAULT_RENDER_WORKERS


def test_resolve_render_num_workers_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="render_num_workers"):
        resolve_render_num_workers(0)


@pytest.mark.asyncio
async def test_chat_render_with_explicit_chat_template(tmp_path: Path) -> None:
    del tmp_path
    render_client, serving_render = _build_stubbed_chat_render_client()

    result = await render_client.render_chat_completion(
        ChatCompletionRequest(
            model=REAL_MODEL_ID,
            messages=[
                ChatMessage(role="user", content="test"),
            ],
            chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        )
    )

    assert result.token_ids == [7, 8, 9]
    assert serving_render.chat_requests[0].chat_template == "{% for message in messages %}{{ message.content }}{% endfor %}"


@pytest.mark.asyncio
async def test_chat_render_passes_tools_and_tool_choice(tmp_path: Path) -> None:
    del tmp_path
    render_client, serving_render = _build_stubbed_chat_render_client()
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    result = await render_client.render_chat_completion(
        ChatCompletionRequest(
            model=REAL_MODEL_ID,
            messages=[
                ChatMessage(role="user", content="What's the weather?"),
            ],
            tools=tools,
            tool_choice="auto",
        )
    )

    assert result.token_ids == [7, 8, 9]
    assert serving_render.chat_requests[0].tools == tools
    assert serving_render.chat_requests[0].tool_choice == "auto"


@pytest.mark.asyncio
async def test_chat_render_multimodal_happy_path(tmp_path: Path) -> None:
    del tmp_path
    render_client, serving_render = _build_stubbed_chat_render_client(
        chat_result=SimpleNamespace(
            token_ids=[7, 8, 9],
            features=SimpleNamespace(
                mm_hashes={"image": ["img_hash"]},
                mm_placeholders={"image": [SimpleNamespace(offset=1, length=2)]},
            ),
        )
    )
    result = await render_client.render_chat_completion(
        ChatCompletionRequest(
            model=REAL_MODEL_ID,
            messages=[
                ChatMessage(
                    role="user",
                    content_parts=[
                        ChatContentPart(text="Describe this"),
                        ChatContentPart(
                            media=MediaRef(
                                modality="image",
                                url="https://example.com/image.jpg",
                            )
                        ),
                    ],
                )
            ],
            add_generation_prompt=True,
        )
    )

    assert result.token_ids == [7, 8, 9]
    assert result.multimodal_features
    assert result.multimodal_features[0].modality == "image"
    assert result.multimodal_features[0].hash == "img_hash"
    content = serving_render.chat_requests[0].messages[0]["content"]
    assert content == [
        {"type": "text", "text": "Describe this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]


@pytest.mark.asyncio
async def test_chat_render_forwards_optional_request_fields() -> None:
    render_client, serving_render = _build_stubbed_chat_render_client()

    result = await render_client.render_chat_completion(
        ChatCompletionRequest(
            model=REAL_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
            chat_template="{{ messages }}",
            chat_template_kwargs={"enable_tools": True},
            mm_processor_kwargs={"image": {"max_pixels": 512}},
            media_io_kwargs={"image": {"timeout": 3}},
        )
    )

    assert result.token_ids == [7, 8, 9]
    assert len(serving_render.chat_requests) == 1
    request = serving_render.chat_requests[0]
    assert request.chat_template == "{{ messages }}"
    assert request.chat_template_kwargs == {"enable_tools": True}
    assert request.mm_processor_kwargs == {"image": {"max_pixels": 512}}
    assert request.media_io_kwargs == {"image": {"timeout": 3}}


@pytest.mark.asyncio
async def test_chat_render_parses_message_tool_calls_json() -> None:
    render_client, serving_render = _build_stubbed_chat_render_client()

    await render_client.render_chat_completion(
        ChatCompletionRequest(
            model=REAL_MODEL_ID,
            messages=[
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls_json='[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]',
                )
            ],
        )
    )

    assert len(serving_render.chat_requests) == 1
    request = serving_render.chat_requests[0]
    assert request.messages[0]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{}",
            },
        }
    ]