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
Generate a random fixed-length multi-turn dataset in AIPerf multi_turn JSONL format.

Each line is one session:
    {"session_id": "...", "turns": [{"text": "...", "output_length": 1024}, ...]}

Usage:
    python gen_multi_turn.py --num-sessions 120 --turns 4 --isl 1024 --osl 1024 \
        --seed 42 --output random_1k_mt_s42.jsonl
"""

import argparse
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a random fixed-length multi-turn dataset"
    )
    parser.add_argument("--num-sessions", type=_common.positive_int, default=120)
    parser.add_argument("--turns", type=_common.positive_int, default=4, help="Turns per session")
    parser.add_argument("--isl", type=_common.positive_int, default=1024, help="Input tokens per turn")
    parser.add_argument("--osl", type=_common.positive_int, default=1024, help="Max output tokens per turn")
    parser.add_argument(
        "--output", type=Path, default=Path("random_multi_turn_dataset.jsonl"),
        help="Output JSONL path"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _common.init_seed(args.seed)
    generator, tokenizer = _common.create_prompt_generator(args.isl)
    total = args.num_sessions * args.turns
    unique_prompts = _common.generate_unique_prompts(generator, tokenizer, total, args.isl)

    idx = 0
    sessions = []
    for s in range(args.num_sessions):
        session = {
            "session_id": f"s{args.seed}_{s:04d}",
            "turns": [
                {"text": unique_prompts[idx + t], "output_length": args.osl}
                for t in range(args.turns)
            ],
        }
        idx += args.turns
        sessions.append(session)

    _common.write_jsonl(args.output, sessions)

    print(f"Written {args.num_sessions} sessions x {args.turns} turns to {args.output}")
    _common.print_profile_usage(args.output, "multi_turn")


if __name__ == "__main__":
    main()
