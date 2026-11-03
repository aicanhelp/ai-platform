# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""HuggingFace tokenizer provider."""

from __future__ import annotations

import logging
from typing import Any

from providers.base import render_chat_template_token_ids
from providers.resolution import ResolvedTokenizer

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """Tokenizer provider backed by transformers.AutoTokenizer."""

    def __init__(self, resolved: ResolvedTokenizer) -> None:
        from transformers import AutoTokenizer

        self._resolved = resolved
        logger.info(
            "initializing HuggingFace tokenizer",
            extra={
                "tokenizer": self._resolved.tokenizer_path,
            },
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._resolved.tokenizer_path,
            trust_remote_code=self._resolved.trust_remote_code,
            padding_side="left",
            truncation_side="left",
            use_fast=True,
        )

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def resolved_name(self) -> str:
        return self._resolved.tokenizer_path

    @property
    def resolved_tokenizer(self) -> ResolvedTokenizer:
        """Return detailed resolved tokenizer metadata."""
        return self._resolved

    def has_chat_template(self) -> bool:
        return bool(getattr(self._tokenizer, "chat_template", None))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        return render_chat_template_token_ids(
            self._tokenizer,
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            chat_template_kwargs=chat_template_kwargs,
        )