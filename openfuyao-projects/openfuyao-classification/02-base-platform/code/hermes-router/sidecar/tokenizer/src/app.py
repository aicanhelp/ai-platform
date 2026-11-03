# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Executable entrypoint for the tokenizer gRPC sidecar."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

from core import TokenizerService
from health import start_health_server
from providers import ResolvedTokenizer
from proto import tokenizer_pb2_grpc as pb_grpc
from render import RenderSettings, create_vllm_render_client
from thread_pool import THREAD_POOL_SIZE_ENV, create_thread_pool
from transport.grpc_service import TokenizationServiceServicer

# Defaults align with the kserve scheduler preset: it mounts the shared UDS
# directory at /tmp/tokenizer and probes httpGet /healthz on port 8082, and
# passes no args to the tokenizer container.
DEFAULT_SOCKET_PATH = "/tmp/tokenizer/tokenizer-uds.socket"
DEFAULT_HEALTH_PORT = 8082
DEFAULT_MAX_MESSAGE_SIZE = 16 * 1024 * 1024
TOKENIZER_PROVIDER_ENV = "TOKENIZER_PROVIDER"
TOKENIZER_CACHE_DIR_ENV = "TOKENIZER_CACHE_DIR"
DEFAULT_TOKENIZER_PROVIDER = "huggingface"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Hermes tokenizer gRPC sidecar")
    parser.add_argument("--socket-path", default=DEFAULT_SOCKET_PATH)
    parser.add_argument(
        "--health-port",
        type=int,
        default=DEFAULT_HEALTH_PORT,
        help="Port for the HTTP /healthz probe endpoint; 0 disables it.",
    )
    parser.add_argument("--max-message-size", type=int, default=DEFAULT_MAX_MESSAGE_SIZE)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--thread-pool-size",
        type=int,
        default=None,
        help=(
            "Worker threads for tokenization offloaded via asyncio.to_thread "
            "(direct tokenize/encode/chat-template paths). Overrides the "
            f"{THREAD_POOL_SIZE_ENV} env var; defaults to min(cpu*2, 32)."
        ),
    )
    parser.add_argument("--render-tokenizer", default="")
    parser.add_argument("--render-chat-template", default="")
    parser.add_argument(
        "--render-chat-template-content-format",
        default="auto",
    )
    parser.add_argument("--render-num-workers", type=int, default=None)
    parser.add_argument(
        "--render-trust-request-chat-template",
        action="store_true",
    )
    return parser.parse_args()


def build_render_factory(args: argparse.Namespace) -> Callable[[ResolvedTokenizer], object]:
    """Build the per-model render factory from CLI settings."""

    settings = RenderSettings(
        tokenizer=args.render_tokenizer or None,
        chat_template=args.render_chat_template or None,
        chat_template_content_format=args.render_chat_template_content_format,
        num_workers=args.render_num_workers,
        trust_request_chat_template=args.render_trust_request_chat_template,
    )
    return lambda resolved_tokenizer: create_vllm_render_client(resolved_tokenizer, settings)


def configure_default_executor(
    loop: asyncio.AbstractEventLoop,
    thread_pool_size: int | None = None,
) -> ThreadPoolExecutor:
    executor = create_thread_pool(thread_pool_size)
    loop.set_default_executor(executor)
    return executor


async def serve(args: argparse.Namespace) -> None:
    """Start the tokenizer gRPC server."""

    socket_path = Path(args.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        socket_path.unlink()

    configure_default_executor(asyncio.get_running_loop(), args.thread_pool_size)

    health_server = start_health_server(args.health_port) if args.health_port > 0 else None
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", args.max_message_size),
            ("grpc.max_receive_message_length", args.max_message_size),
        ]
    )
    tokenizer = TokenizerService(
        provider_name=os.getenv(TOKENIZER_PROVIDER_ENV, DEFAULT_TOKENIZER_PROVIDER),
        cache_dir=os.getenv(TOKENIZER_CACHE_DIR_ENV) or None,
        render_factory=build_render_factory(args),
    )
    pb_grpc.add_TokenizationServiceServicer_to_server(
        TokenizationServiceServicer(tokenizer),
        server,
    )
    server.add_insecure_port(f"unix:{socket_path}")
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=None)
        if health_server is not None:
            health_server.shutdown()
        with contextlib.suppress(FileNotFoundError):
            socket_path.unlink()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()