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

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

import grpc
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "proto"))

from app import build_predictor, serve
from proto import prediction_pb2 as pb
from proto import prediction_pb2_grpc as pb_grpc
from training.artifacts import TrainOptions, write_artifacts
from training.jsonl_loader import load_completed_jsonl
from training.samples import generate_samples


def test_build_predictor_without_bundle_is_not_serving() -> None:
    # No --artifact-root/--target-model/--model-version means the sidecar has no
    # bundle to load: it must report NOT_SERVING so the EPP falls back instead of
    # routing on fabricated predictions.
    args = argparse.Namespace(artifact_root="", target_model="", model_version="")
    _predictor, serving = build_predictor(args)
    assert serving is False


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["xgboost", "lightgbm"])
async def test_app_serves_artifact_prediction_over_uds(tmp_path: Path, backend: str) -> None:
    version = f"app-{backend}"
    train_fixture_bundle(tmp_path, backend, version)
    temp_dir = tempfile.mkdtemp(prefix="hp-", dir="/tmp")
    socket_path = os.path.join(temp_dir, "p.sock")
    args = argparse.Namespace(
        socket_path=socket_path,
        max_message_size=1024 * 1024,
        artifact_root=str(tmp_path),
        target_model="test-model",
        model_version=version,
    )
    task = asyncio.create_task(serve(args))
    try:
        await wait_for_socket(socket_path)
        async with grpc.aio.insecure_channel(f"unix:{socket_path}") as channel:
            stub = pb_grpc.PredictionServiceStub(channel)
            response = await stub.Predict(
                pb.PredictRequest(
                    target_model="test-model",
                    inputs=[pb.PredictInput(input_id=7, pod_type="aggregate", input_token_length=120, num_request_waiting=0, server_ip="10.0.0.7")],
                )
            )
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert len(response.outputs) == 1
    assert response.outputs[0].input_id == 7
    assert response.outputs[0].predicted_ttft_millis >= 0
    assert response.outputs[0].predicted_tpot_millis >= 0


def train_fixture_bundle(tmp_path: Path, backend: str, version: str) -> None:
    records, load_drops = load_completed_jsonl(ROOT / "tests" / "fixtures" / "completed_requests.jsonl")
    samples, sample_drops = generate_samples(records)
    load_drops.update(sample_drops)
    write_artifacts(
        samples,
        load_drops,
        TrainOptions(
            artifact_root=tmp_path,
            target_model="test-model",
            model_version=version,
            model_backend=backend,
            quantile=0.9,
            min_samples_per_slot=2,
            min_holdout_samples_per_slot=1,
            holdout_ratio=0.2,
            bucket_cap=0,
        ),
    )


async def wait_for_socket(path: str) -> None:
    for _ in range(100):
        if os.path.exists(path):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"socket did not appear: {path}")
