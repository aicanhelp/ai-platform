# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Executable entrypoint for the prediction gRPC sidecar."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_health.v1._async import HealthServicer

from domain.models import PredictInput, PredictOutput
from proto import prediction_pb2_grpc as pb_grpc
from runtime.artifact_predictor import ArtifactPredictor
from runtime.predictor import FailedPreconditionError
from transport.grpc_service import PredictionServiceServicer

DEFAULT_SOCKET_PATH = "/var/run/hermes/prediction.sock"
DEFAULT_MAX_MESSAGE_SIZE = 16 * 1024 * 1024
SERVICE_NAME = "hermes.prediction.v1.PredictionService"
SHUTDOWN_GRACE_SECONDS = 5
# Predict runs off the event loop in this many threads. Two is the measured sweet
# spot: under concurrent RPCs (N=50..200 backends each) going from 1->2 workers
# lifts throughput ~12-27% and cuts p50 ~20% for both XGBoost and LightGBM,
# because one RPC's GIL-releasing C++ inference can overlap another RPC's
# (GIL-bound) Python feature engineering. Beyond 2 the GIL caps useful
# parallelism — 4/8 workers add no throughput and only worsen p99 tail latency
# via CPU oversubscription against the models' own internal threads. Booster
# concurrent-predict safety is not a concern at 2 workers in practice, but keep
# the pool small for this reason too.
#
# Overridable via HERMES_PREDICT_WORKER_THREADS so the worker-count sweep can be
# re-measured without code edits; the default stays at the measured sweet spot.
PREDICT_WORKER_THREADS = int(os.environ.get("HERMES_PREDICT_WORKER_THREADS", "2"))


class FailedPredictor:
    model_version = ""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def predict(self, target_model: str, inputs: list[PredictInput]) -> list[PredictOutput]:
        del target_model, inputs
        raise FailedPreconditionError(self._reason)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hermes prediction gRPC sidecar")
    parser.add_argument("--socket-path", default=DEFAULT_SOCKET_PATH)
    parser.add_argument("--max-message-size", type=int, default=DEFAULT_MAX_MESSAGE_SIZE)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--target-model", default="")
    parser.add_argument("--model-version", default="")
    return parser.parse_args()


def build_predictor(args: argparse.Namespace):
    # A prediction bundle is mandatory. With no --artifact-root/--target-model/
    # --model-version the sidecar has nothing real to serve, so report NOT_SERVING
    # and let the EPP fall back rather than emit fabricated constant latencies.
    if not (args.artifact_root or args.target_model or args.model_version):
        return (
            FailedPredictor(
                "prediction requires --artifact-root, --target-model and --model-version"
            ),
            False,
        )
    try:
        predictor = ArtifactPredictor(
            artifact_root=Path(args.artifact_root),
            target_model=args.target_model,
            model_version=args.model_version,
        )
        return predictor, True
    except Exception as exc:  # pragma: no cover - exercised through app smoke
        logging.exception("failed to load prediction artifact bundle")
        return FailedPredictor(f"artifact bundle cannot be loaded: {exc}"), False


async def serve(args: argparse.Namespace) -> None:
    socket_path = Path(args.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        socket_path.unlink()

    predictor, serving = build_predictor(args)
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", args.max_message_size),
            ("grpc.max_receive_message_length", args.max_message_size),
        ]
    )
    predict_executor = ThreadPoolExecutor(
        max_workers=PREDICT_WORKER_THREADS, thread_name_prefix="predict"
    )
    pb_grpc.add_PredictionServiceServicer_to_server(
        PredictionServiceServicer(predictor, executor=predict_executor), server
    )
    health = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health, server)
    await health.set(
        SERVICE_NAME,
        health_pb2.HealthCheckResponse.SERVING
        if serving
        else health_pb2.HealthCheckResponse.NOT_SERVING,
    )
    server.add_insecure_port(f"unix:{socket_path}")
    await server.start()

    # Kubernetes sends SIGTERM on pod termination; default asyncio.run only
    # catches SIGINT, so without explicit SIGTERM handling the grace period is
    # wasted in a hard kill. Treat both signals as a request to flip the stop
    # event and let the finally block drive ordered shutdown. We track which
    # signals we successfully installed so we can remove them cleanly even
    # when the test event loop refuses one (Windows / restricted env).
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    installed_signals: list[signal.Signals] = []

    def _request_shutdown(sig: signal.Signals) -> None:
        logging.info("received %s, initiating prediction sidecar shutdown", sig.name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig)
            installed_signals.append(sig)
        except (NotImplementedError, RuntimeError):
            # Best-effort: signal handlers may be unsupported on the current
            # platform or already owned by the test harness's event loop.
            pass

    try:
        await stop_event.wait()
    finally:
        for sig in installed_signals:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
        await server.stop(grace=SHUTDOWN_GRACE_SECONDS)
        # Cancel queued-but-unstarted predicts and return immediately. We do NOT
        # wait=True: ThreadPoolExecutor has no shutdown timeout and a running
        # inference cannot be safely killed from outside its thread, so waiting
        # risks blocking termination unboundedly past the grpc grace period. A
        # hard time bound on in-flight inference would need process-level
        # isolation or a watchdog — out of scope for this change.
        predict_executor.shutdown(wait=False, cancel_futures=True)
        with contextlib.suppress(FileNotFoundError):
            socket_path.unlink()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
