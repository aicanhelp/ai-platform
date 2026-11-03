# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tokenizer resolution helpers shared by providers and the core service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_SUPPORTED_REMOTE_SCHEMES: frozenset[str] = frozenset()


@dataclass(slots=True)
class ResolvedTokenizer:
    """Concrete tokenizer/model paths resolved for one initialized service."""

    served_model_name: str
    model_path: str
    tokenizer_path: str
    provider: str
    trust_remote_code: bool


class TokenizerResolutionError(RuntimeError):
    """Error raised when tokenizer resolution cannot proceed."""

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        tokenizer_name: str | None = None,
        message: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.tokenizer_name = tokenizer_name
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        context = (
            f"model={self.model}, "
            f"provider={self.provider}, "
            f"tokenizer_name={self.tokenizer_name}"
        )
        if self.message:
            return f"tokenizer resolution failed: {context}; error={self.message}"
        return f"tokenizer resolution failed: {context}; error=unknown"


def normalize_provider_name(provider: str) -> str:
    """Validate and normalize one configured provider value."""

    if provider in {"huggingface", "modelscope"}:
        return provider
    raise ValueError(
        f"unsupported provider {provider!r}; expected one of huggingface|modelscope"
    )


def _is_remote_model(identifier: str) -> bool:
    """Determine whether an identifier should be treated as a remote model id."""

    parsed = urlparse(identifier)
    if parsed.scheme and parsed.scheme not in _SUPPORTED_REMOTE_SCHEMES:
        raise ValueError(f"unsupported URI scheme: {parsed.scheme}://")

    expanded = Path(identifier).expanduser()
    if expanded.is_absolute():
        return False
    if identifier.startswith("./") or identifier.startswith("../") or identifier.startswith("~"):
        return False
    if expanded.exists():
        return False
    return "/" in identifier
