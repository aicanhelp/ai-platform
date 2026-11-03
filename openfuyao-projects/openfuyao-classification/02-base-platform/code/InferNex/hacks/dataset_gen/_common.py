# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Shared helpers for the AIPerf dataset generation scripts."""

import argparse
import json
import random
from pathlib import Path

from aiperf.common import random_generator as rng
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
from aiperf.config.distributions import NormalDistribution
from aiperf.dataset.generator.prompt import PromptGenerator


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return ivalue


def init_seed(seed: int) -> None:
    random.seed(seed)
    rng.init(seed)


def create_prompt_generator(isl: int) -> tuple[PromptGenerator, Tokenizer]:
    print("Creating tokenizer (builtin o200k_base)...")
    tokenizer = Tokenizer.from_pretrained("builtin")

    print("Initializing PromptGenerator with Shakespeare corpus...")
    prompt_config = PromptConfig(
        isl=NormalDistribution(mean=isl, stddev=0),
    )
    generator = PromptGenerator(
        prompts=prompt_config,
        prefix_prompts=PrefixPromptConfig(),
        tokenizer=tokenizer,
    )
    return generator, tokenizer


def generate_unique_prompts(
    generator: PromptGenerator,
    tokenizer: Tokenizer,
    count: int,
    isl: int,
) -> list[str]:
    print(f"Generating {count} unique prompts of {isl} tokens each...")
    unique_prompts: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 10
    while len(unique_prompts) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to generate {count} unique prompts after "
                f"{max_attempts} attempts (got {len(unique_prompts)})"
            )
        prompt_text = generator.generate_prompt(isl)
        if len(tokenizer.encode(prompt_text)) != isl:
            continue
        if prompt_text in seen:
            continue
        seen.add(prompt_text)
        unique_prompts.append(prompt_text)
        if len(unique_prompts) % 40 == 0:
            print(f"  {len(unique_prompts)}/{count} prompts generated...")

    print(f"Unique prompts: {len(unique_prompts)}/{count}")
    return unique_prompts


def write_jsonl(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_profile_usage(output: Path, dataset_type: str) -> None:
    print(f"Usage: aiperf profile --input-file {output} "
          f"--custom-dataset-type {dataset_type} ...")
