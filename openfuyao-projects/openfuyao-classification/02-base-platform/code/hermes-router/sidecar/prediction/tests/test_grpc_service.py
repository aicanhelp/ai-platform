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

import asyncio
import os
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "proto"))

from domain.models import PredictInput, PredictOutput
from runtime.predictor import Predictor
from proto import prediction_pb2 as pb
from proto import prediction_pb2_grpc as pb_grpc
from transport.grpc_service import PredictionServiceServicer


class _ConstantPredictor:
    """Test fake returning fixed ttft/tpot for every input, so the servicer layer
    can be exercised end-to-end without a trained prediction bundle."""

    model_version = ""

    def __init__(self, ttft_millis: float, tpot_millis: float) -> None:
        self._ttft_millis = ttft_millis
        self._tpot_millis = tpot_millis

    def predict(self, target_model: str, inputs: list[PredictInput]) -> list[PredictOutput]:
        del target_model
        return [
            PredictOutput(item.input_id, item.server_ip, self._ttft_millis, self._tpot_millis)
            for item in inputs
        ]


@pytest.mark.asyncio
async def test_constant_predictor_outputs_and_omits_invalid_rows() -> None:
    socket_path, server = await start_server(_ConstantPredictor(12.0, 0.8))
    try:
        async with grpc.aio.insecure_channel(f"unix:{socket_path}") as channel:
            stub = pb_grpc.PredictionServiceStub(channel)
            response = await stub.Predict(
                pb.PredictRequest(
                    target_model="test-model",
                    inputs=[
                        pb.PredictInput(input_id=0, pod_type="aggregate", server_ip="10.0.0.1"),
                        pb.PredictInput(input_id=1, pod_type="bogus", server_ip="10.0.0.2"),
                    ],
                )
            )
    finally:
        await stop_server(server, socket_path)

    assert len(response.outputs) == 1
    assert response.outputs[0].input_id == 0
    assert response.outputs[0].predicted_ttft_millis == 12.0
    assert response.outputs[0].predicted_tpot_millis == 0.8
    assert response.metadata.omitted_inputs == 1


@pytest.mark.asyncio
async def test_empty_target_model_is_invalid_argument() -> None:
    socket_path, server = await start_server(_ConstantPredictor(12.0, 0.8))
    try:
        async with grpc.aio.insecure_channel(f"unix:{socket_path}") as channel:
            stub = pb_grpc.PredictionServiceStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                await stub.Predict(pb.PredictRequest(inputs=[pb.PredictInput(input_id=0, pod_type="aggregate")]))
    finally:
        await stop_server(server, socket_path)

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


class _BarrierPredictor:
    """Blocks in predict() until two calls run concurrently.

    If the servicer ran predict() inline on the event loop, the first call would
    hold the loop thread at the barrier and the second handler could never start,
    so the barrier would time out. Reaching the barrier from both calls proves the
    work was offloaded off the loop.
    """

    model_version = ""

    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def predict(self, target_model: str, inputs: list[PredictInput]) -> list[PredictOutput]:
        del target_model, inputs
        self._barrier.wait()
        return []


@pytest.mark.asyncio
async def test_predict_runs_off_event_loop() -> None:
    barrier = threading.Barrier(2, timeout=5)
    executor = ThreadPoolExecutor(max_workers=2)
    socket_path, server = await start_server(_BarrierPredictor(barrier), executor=executor)
    request = pb.PredictRequest(
        target_model="m", inputs=[pb.PredictInput(input_id=0, pod_type="aggregate")]
    )
    try:
        async with grpc.aio.insecure_channel(f"unix:{socket_path}") as channel:
            stub = pb_grpc.PredictionServiceStub(channel)
            # gather completes only if both predicts reach the barrier together.
            await asyncio.gather(stub.Predict(request), stub.Predict(request))
    finally:
        executor.shutdown(wait=False)
        await stop_server(server, socket_path)


class _DropDecodePredictor:
    """Scores everything except decode rows, mimicking a bundle whose disagg_tpot
    slot is not promoted so valid decode inputs are dropped by the predictor."""

    model_version = "v9"

    def predict(self, target_model: str, inputs: list[PredictInput]) -> list[PredictOutput]:
        del target_model
        return [
            PredictOutput(item.input_id, item.server_ip, 1.0, 2.0)
            for item in inputs
            if item.pod_type != "decode"
        ]


@pytest.mark.asyncio
async def test_omitted_inputs_counts_invalid_and_predictor_dropped_rows() -> None:
    socket_path, server = await start_server(_DropDecodePredictor())
    try:
        async with grpc.aio.insecure_channel(f"unix:{socket_path}") as channel:
            stub = pb_grpc.PredictionServiceStub(channel)
            response = await stub.Predict(
                pb.PredictRequest(
                    target_model="test-model",
                    inputs=[
                        pb.PredictInput(input_id=0, pod_type="aggregate", server_ip="10.0.0.1"),
                        pb.PredictInput(input_id=1, pod_type="decode", server_ip="10.0.0.2"),
                        pb.PredictInput(input_id=2, pod_type="bogus", server_ip="10.0.0.3"),
                    ],
                )
            )
    finally:
        await stop_server(server, socket_path)

    # Only the aggregate row is scored; the decode row (predictor drop) and the
    # bogus row (pod-type filter) both count toward omitted_inputs.
    assert [output.input_id for output in response.outputs] == [0]
    assert response.metadata.omitted_inputs == 2


async def start_server(
    predictor: Predictor,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[str, grpc.aio.Server]:
    temp_dir = tempfile.mkdtemp(prefix="hp-", dir="/tmp")
    socket_path = os.path.join(temp_dir, "p.sock")
    server = grpc.aio.server()
    pb_grpc.add_PredictionServiceServicer_to_server(
        PredictionServiceServicer(predictor, executor=executor), server
    )
    server.add_insecure_port(f"unix:{socket_path}")
    await server.start()
    return socket_path, server


async def stop_server(server, socket_path: str) -> None:
    await server.stop(grace=None)
    shutil.rmtree(os.path.dirname(socket_path), ignore_errors=True)
