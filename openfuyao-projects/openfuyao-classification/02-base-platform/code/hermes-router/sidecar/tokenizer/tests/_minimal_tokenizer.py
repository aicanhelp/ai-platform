# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Build tiny local tokenizer directories for network-free tests."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer as TokenizerBackend
from transformers import PreTrainedTokenizerFast

VALID_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
)

# A minimal BPE with a single <unk> token; it loads and encodes without any
# model files, so tests can exercise the sidecar paths without a network.
_MINIMAL_TOKENIZER_JSON = (
    '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],'
    '"normalizer":null,'
    '"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},'
    '"post_processor":null,"decoder":null,'
    '"model":{"type":"BPE","dropout":null,"unk_token":"<unk>",'
    '"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,'
    '"vocab":{"<unk>":0},"merges":[]}}'
)


def save_minimal_tokenizer(directory: Path, *, chat_template: str | None = None) -> Path:
    """Save a minimal loadable fast tokenizer into ``directory``.

    When ``chat_template`` is given it is written into tokenizer_config.json so
    the loaded tokenizer reports it as its built-in chat template.
    """

    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=TokenizerBackend.from_str(_MINIMAL_TOKENIZER_JSON))
    tokenizer.save_pretrained(directory)

    if chat_template is not None:
        config_path = directory / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["chat_template"] = chat_template
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    return directory
