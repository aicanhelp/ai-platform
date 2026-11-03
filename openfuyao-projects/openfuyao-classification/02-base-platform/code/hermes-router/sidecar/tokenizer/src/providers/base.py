# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Provider protocol shared by HuggingFace and ModelScope adapters."""

from __future__ import annotations

from typing import Any, Protocol

from providers.resolution import ResolvedTokenizer


class TokenizerProvider(Protocol):
    """Protocol for a tokenizer backend that can encode text to token IDs."""

    @property
    def provider_name(self) -> str:
        """Return the normalized provider identifier."""

    @property
    def resolved_name(self) -> str:
        """Return the fully resolved tokenizer name used by the backend."""

    @property
    def resolved_tokenizer(self) -> ResolvedTokenizer:
        """Return the resolved tokenizer metadata and load kwargs."""

    def has_chat_template(self) -> bool:
        """Report whether the tokenizer carries a built-in chat template.

        A missing template is a model/artifact property, not a sidecar health
        problem: initialize may skip the chat warmup instead of failing.
        """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode plain text into token IDs."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Render chat messages through the tokenizer chat template into token IDs."""


def render_chat_template_token_ids(
    tokenizer: Any,
    conversation: list[dict[str, Any]],
    *,
    chat_template: str | None,
    add_generation_prompt: bool,
    continue_final_message: bool,
    chat_template_kwargs: dict[str, Any] | None,
) -> list[int]:
    """Render a conversation into token IDs via a transformers-style chat template.

    Shared by the HuggingFace and ModelScope providers; both wrap an
    ``AutoTokenizer`` whose ``apply_chat_template`` returns token IDs directly
    when ``tokenize=True`` and ``return_dict=False``.
    """
    return list(
        tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=True,
            return_dict=False,
            **(chat_template_kwargs or {}),
        )
    )