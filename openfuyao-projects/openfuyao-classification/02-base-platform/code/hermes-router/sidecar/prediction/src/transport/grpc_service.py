# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""gRPC adapter for the prediction sidecar."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor

import grpc

from domain.models import PredictInput, PredictOutput
from proto import prediction_pb2 as pb
from proto import prediction_pb2_grpc as pb_grpc
from runtime.predictor import FailedPreconditionError, Predictor
from schema import POD_TYPES

logger = logging.getLogger(__name__)


class PredictionServiceServicer(pb_grpc.PredictionServiceServicer):
    def __init__(self, predictor: Predictor, executor: Executor | None = None) -> None:
        self._predictor = predictor
        # predict() is a synchronous, CPU-bound call. Running it inline on the
        # grpc.aio event loop would block every other RPC for its full duration,
        # starving the health servicer (which shares this loop) and delaying
        # deadline/cancellation handling. Offloading to a thread keeps the loop
        # responsive; None falls back to the loop's default executor.
        self._executor = executor

    async def Predict(
        self,
        request: pb.PredictRequest,
        context: grpc.aio.ServicerContext,
    ):  # noqa: N802
        # grpc.aio's `context.abort` raises AbortError so control does not
        # actually fall through, but explicit `return` after every abort
        # keeps the function readable and survives any future change to the
        # abort raising contract in grpc-py.
        if not request.target_model:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "targetModel cannot be empty")
            return pb.PredictResponse()

        valid_inputs = [from_proto_input(item) for item in request.inputs if item.pod_type in POD_TYPES]
        try:
            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(
                self._executor, self._predictor.predict, request.target_model, valid_inputs
            )
        except FailedPreconditionError as exc:
            logger.error("prediction predictor failed precondition: %s", exc)
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            return pb.PredictResponse()
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return pb.PredictResponse()
        except Exception as exc:
            logger.exception("prediction request failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"predict failed: {exc}")
            return pb.PredictResponse()

        proto_outputs = [to_proto_output(output) for output in outputs]
        # omitted_inputs counts every input that yielded no prediction: rows with
        # an unrecognized pod type (filtered out above) plus rows the predictor
        # dropped because their slot model is not promoted in the loaded bundle.
        # Deriving it from the produced output count folds both drop paths into
        # one number the caller can reconcile against len(inputs).
        omitted_inputs = len(request.inputs) - len(proto_outputs)
        return pb.PredictResponse(
            outputs=proto_outputs,
            metadata=pb.ResponseMetadata(
                target_model=request.target_model,
                model_version=getattr(self._predictor, "model_version", ""),
                omitted_inputs=omitted_inputs,
            ),
        )


def from_proto_input(item: pb.PredictInput) -> PredictInput:
    return PredictInput(
        input_id=item.input_id,
        pod_type=item.pod_type,
        input_token_length=item.input_token_length,
        prefix_cache_score=item.prefix_cache_score,
        num_request_waiting=item.num_request_waiting,
        num_request_running=item.num_request_running,
        kv_cache_usage=item.kv_cache_usage,
        inflight_tokens=item.inflight_tokens,
        hbm_utilization=item.hbm_utilization,
        ai_core_utilization=item.ai_core_utilization,
        bandwidth_utilization=item.bandwidth_utilization,
        server_ip=item.server_ip,
    )


def to_proto_output(output: PredictOutput) -> pb.PredictOutput:
    return pb.PredictOutput(
        input_id=output.input_id,
        server_ip=output.server_ip,
        predicted_ttft_millis=output.predicted_ttft_millis,
        predicted_tpot_millis=output.predicted_tpot_millis,
    )
