# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#!/usr/bin/env python3
"""
Generate a random fixed-length single-turn text dataset in AIPerf JSONL format.

Usage:
    # Default: 120 types × 4 repeats = 480 entries, 8k tokens, shuffled
    python gen_single_turn.py

    # Customize:
    python gen_single_turn.py --num-types 50 --repeat 3 --isl 4096 --output my_data.jsonl
"""

import argparse
import random
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a random fixed-length single-turn dataset"
    )
    parser.add_argument(
        "--num-types", type=_common.positive_int, default=120, help="Number of unique prompt templates"
    )
    parser.add_argument(
        "--repeat", type=_common.positive_int, default=4, help="Repeat count per template"
    )
    parser.add_argument(
        "--isl", type=_common.positive_int, default=8192, help="Input sequence length in tokens"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("random_dataset.jsonl"), help="Output JSONL path"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    _common.init_seed(args.seed)
    generator, tokenizer = _common.create_prompt_generator(args.isl)
    unique_prompts = _common.generate_unique_prompts(
        generator, tokenizer, args.num_types, args.isl
    )

    # Repeat each type
    entries = []
    for prompt_text in unique_prompts:
        for _ in range(args.repeat):
            entries.append({"text": prompt_text})

    # Shuffle
    random.shuffle(entries)

    # Write JSONL
    _common.write_jsonl(args.output, entries)

    print(f"Written {len(entries)} entries to {args.output}")
    _common.print_profile_usage(args.output, "single_turn")


if __name__ == "__main__":
    main()
