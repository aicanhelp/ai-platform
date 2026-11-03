# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import asyncio
import uvicorn
import os

from vllm.config import KVEventsConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

from logger.setup_logger import setup_logger
from routing_metrics.vllm.publisher import CustomStatLogger

logger = setup_logger()

kv_events_config = KVEventsConfig(
    enable_kv_cache_events=True,
    publisher="zmq",
    topic="kv-events"
)

app = FastAPI()
engine = None


@app.post("/generate")
async def generate(request: Request):
    request_dict = await request.json()
    prompt = request_dict.get("prompt")

    if not prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    results_generator = engine.generate(
        prompt,
        sampling_params=sampling_params,
        request_id=f"req-{os.urandom(4).hex()}"
    )

    final_output = ""
    async for request_output in results_generator:
        final_output = request_output.outputs[0].text

    return JSONResponse({"text": final_output})


@app.get("/health")
async def health():
    return {"status": "ok"}


async def main():
    global engine

    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        enable_prefix_caching=True,
        block_size=128,
        kv_events_config=kv_events_config,
        disable_log_stats=False,
    )

    logger.info("Initialize vLLM engine...")

    engine = AsyncLLMEngine.from_engine_args(
        engine_args,
        stat_loggers=[CustomStatLogger]
    )

    logger.info("vLLM engine started...")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
