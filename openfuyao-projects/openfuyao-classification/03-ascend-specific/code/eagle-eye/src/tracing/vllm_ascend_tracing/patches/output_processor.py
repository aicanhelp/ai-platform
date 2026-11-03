# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Patch for OutputProcessor.do_tracing.

Replaces vLLM's native do_tracing. When a request finishes, creates
the full span tree: root → prefill/decode phases → schedule/kv_cache/store.
"""

import json
import logging
import functools

from opentelemetry import trace
from opentelemetry.trace import StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from vllm_ascend_tracing.context import pop_span, get_tracer

logger = logging.getLogger("vllm_ascend.tracing")

_propagator = TraceContextTextMapPropagator()


def patch_output_processor(module):
    output_proc_cls = module.OutputProcessor
    from vllm.utils import length_from_prompt_token_ids_or_embeds

    @functools.wraps(output_proc_cls.do_tracing)
    def patched_do_tracing(self, engine_core_output, req_state, iteration_stats=None):
        span = None
        try:
            trace_headers = engine_core_output.trace_headers
            if not trace_headers:
                return

            span_id = trace_headers.get("x-vllm-ascend-span-id")
            if not span_id:
                return

            span = pop_span(span_id)
            if span is None:
                parent_ctx = _propagator.extract(carrier=dict(trace_headers))

                if "traceparent" not in trace_headers:
                    logger.warning("[tracing] traceparent NOT found in trace_headers!")

                span = get_tracer().start_span(
                    name="vllm_ascend.request",
                    context=parent_ctx,
                    kind=trace.SpanKind.SERVER,
                )
                if not span.is_recording():
                    return

            # Root span attributes
            prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req_state.prompt_token_ids, req_state.prompt_embeds
            )
            attrs = {
                "request.request_id": req_state.external_req_id,
                "request.usage.prompt_tokens": prompt_tokens,
            }

            stats = req_state.stats
            if stats is not None:
                attrs["request.usage.completion_tokens"] = stats.num_generation_tokens
                if iteration_stats is not None:
                    attrs["request.latency.e2e"] = (
                        iteration_stats.iteration_timestamp - stats.arrival_time
                    )
                attrs["request.latency.time_in_queue"] = stats.scheduled_ts - stats.queued_ts
                attrs["request.latency.time_to_first_token"] = stats.first_token_latency
                attrs["request.latency.time_in_model_prefill"] = stats.first_token_ts - stats.scheduled_ts
                attrs["request.latency.time_in_model_decode"] = stats.last_token_ts - stats.first_token_ts
                attrs["request.latency.time_in_model_inference"] = stats.last_token_ts - stats.scheduled_ts

            for key, val in [("request.top_p", req_state.top_p),
                             ("request.max_tokens", req_state.max_tokens_param),
                             ("request.temperature", req_state.temperature),
                             ("request.n", req_state.n)]:
                if val:
                    attrs[key] = val

            span.set_attributes(attrs)

            # Parse accumulated stats and create child spans
            ascend_stats_json = trace_headers.get("x-vllm-ascend-stats")
            if ascend_stats_json:
                try:
                    ascend_stats = json.loads(ascend_stats_json)
                    native_ctx = _build_native_ctx(stats, prompt_tokens, trace_headers, req_state)
                    _create_child_spans(span, ascend_stats, native_ctx)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("[tracing] Failed to parse ascend stats: %s", e)

            span.set_status(StatusCode.OK)
            span.end()
        except Exception as e:  # noqa: BLE001
            logger.warning("[tracing] patched_do_tracing error, ending span best-effort: %s", e)
            if span is not None:
                try:
                    span.set_status(StatusCode.ERROR, description=str(e))
                    span.end()
                except Exception:  # noqa: BLE001
                    pass

    output_proc_cls.do_tracing = patched_do_tracing


def _build_native_ctx(stats, prompt_tokens, trace_headers, req_state):
    """Build context dict with timing/token data for phase spans."""
    # Read num_cached_tokens from req_state, which is populated by OutputProcessor
    # from the first EngineCoreOutput (works for both vLLM 0.18.0 and 0.23.0).
    num_cached_tokens = getattr(req_state, "num_cached_tokens", 0)

    ctx = {
        "num_prompt_tokens": prompt_tokens,
        "num_cached_tokens": num_cached_tokens,
    }
    if stats is not None:
        ctx["prefill_time_ms"] = (stats.first_token_ts - stats.scheduled_ts) * 1000
        ctx["decode_time_ms"] = (stats.last_token_ts - stats.first_token_ts) * 1000
        ctx["num_generation_tokens"] = stats.num_generation_tokens
        # Epoch timestamps derived from arrival_time + monotonic durations
        arrival_ns = int(stats.arrival_time * 1e9)
        queue_dur_ns = int((stats.scheduled_ts - stats.queued_ts) * 1e9)
        prefill_dur_ns = int((stats.first_token_ts - stats.scheduled_ts) * 1e9)
        decode_dur_ns = int((stats.last_token_ts - stats.first_token_ts) * 1e9)
        ctx["scheduled_ts_ns"] = arrival_ns + queue_dur_ns
        ctx["first_token_ts_ns"] = ctx["scheduled_ts_ns"] + prefill_dur_ns
        ctx["last_token_ts_ns"] = ctx["first_token_ts_ns"] + decode_dur_ns

    store_json = trace_headers.get("x-ascend-store-stats")
    if store_json:
        try:
            ctx["store_stats"] = json.loads(store_json)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("[tracing] Failed to parse store stats: %s", e)
    return ctx


def _create_child_spans(parent_span, ascend_stats, native_ctx):
    """Create prefill and decode phase spans with child spans."""
    tracer = get_tracer()
    parent_ctx = trace.set_span_in_context(parent_span)

    prefill_data = ascend_stats.get("prefill", {})
    if prefill_data.get("schedule") or prefill_data.get("kv_cache"):
        _create_phase_span(
            tracer, parent_ctx, "prefill", prefill_data, native_ctx,
            start_ns=native_ctx.get("scheduled_ts_ns"),
            end_ns=native_ctx.get("first_token_ts_ns"),
        )

    decode_data = ascend_stats.get("decode", {})
    if decode_data.get("schedule") or decode_data.get("kv_cache"):
        _create_phase_span(
            tracer, parent_ctx, "decode", decode_data, native_ctx,
            start_ns=native_ctx.get("first_token_ts_ns"),
            end_ns=native_ctx.get("last_token_ts_ns"),
            num_preemptions=ascend_stats.get("num_preemptions", 0),
        )


def _create_phase_span(tracer, parent_ctx, phase, phase_data, native_ctx,
                       start_ns=None, end_ns=None, num_preemptions=0):
    """Create a single phase span with schedule/kv_cache/store children."""
    span_name = "vllm_ascend.prefill_execution" if phase == "prefill" else "vllm_ascend.decode_phase"
    phase_span = tracer.start_span(name=span_name, context=parent_ctx, start_time=start_ns)
    phase_ctx = trace.set_span_in_context(phase_span)

    # Phase-level attributes
    _set_phase_attrs(phase_span, phase, phase_data, native_ctx)

    # Schedule child span
    schedule_data = phase_data.get("schedule")
    if schedule_data:
        _create_schedule_span(tracer, phase_ctx, phase, schedule_data, start_ns)

    # KV cache child span
    kv_data = phase_data.get("kv_cache")
    if kv_data:
        _create_kv_cache_span(tracer, phase_ctx, phase, kv_data, native_ctx, start_ns, num_preemptions)

    # AscendStore child span (prefill only)
    if phase == "prefill":
        _create_store_span(tracer, phase_ctx, native_ctx, start_ns)

    phase_span.end(end_time=end_ns)


def _set_phase_attrs(phase_span, phase, phase_data, native_ctx):
    if phase == "prefill":
        attrs = {"prefill.num_tokens": native_ctx.get("num_prompt_tokens", 0)}
        prefill_time_ms = native_ctx.get("prefill_time_ms")
        if prefill_time_ms is not None:
            attrs["prefill.total_time_ms"] = prefill_time_ms
        phase_span.set_attributes(attrs)
    else:
        schedule_data = phase_data.get("schedule")
        decode_iterations = schedule_data["total_calls"] if schedule_data else 0
        attrs = {"decode.num_iterations": decode_iterations}
        decode_time_ms = native_ctx.get("decode_time_ms")
        if decode_time_ms is not None:
            attrs["decode.total_time_ms"] = decode_time_ms
            if decode_iterations > 0:
                attrs["decode.avg_time_per_token_ms"] = decode_time_ms / decode_iterations
        phase_span.set_attributes(attrs)


def _create_schedule_span(tracer, phase_ctx, phase, data, start_ns):
    span = tracer.start_span(
        name=f"vllm_ascend.{phase}.schedule", context=phase_ctx, start_time=start_ns,
    )
    span.set_attributes({
        "schedule.total_calls": data["total_calls"],
        "schedule.total_tokens": data["total_tokens"],
        "schedule.avg_tokens_per_call": data["avg_tokens_per_call"],
        "schedule.total_time_ms": data["total_time_ms"],
        "schedule.avg_time_ms": data["avg_time_ms"],
        "schedule.max_time_ms": data.get("max_time_ms", 0.0),
        "schedule.avg_queue_length": data["avg_queue_length"],
        "schedule.max_queue_length": data["max_queue_length"],
        "schedule.avg_running_count": data["avg_running_count"],
    })
    span.end(end_time=start_ns)


def _create_kv_cache_span(tracer, phase_ctx, phase, kv_data, native_ctx,
                          start_ns, num_preemptions):
    span = tracer.start_span(
        name=f"vllm_ascend.{phase}.kv_cache", context=phase_ctx, start_time=start_ns,
    )
    attrs = {
        "kv_cache.total_calls": kv_data["total_calls"],
        "kv_cache.total_blocks_allocated": kv_data["total_blocks_allocated"],
    }
    if kv_data["total_calls"] > 0:
        attrs["kv_cache.avg_blocks_per_call"] = (
            kv_data["total_blocks_allocated"] / kv_data["total_calls"]
        )
    for key in ("final_free_blocks", "min_free_blocks", "avg_free_blocks"):
        if kv_data.get(key) is not None:
            attrs[f"kv_cache.{key}"] = kv_data[key]

    if phase == "prefill":
        num_prompt = native_ctx.get("num_prompt_tokens", 0)
        num_cached = native_ctx.get("num_cached_tokens", 0)
        if num_cached > 0:
            attrs["kv_cache.total_cached_tokens"] = num_cached
            if num_prompt > 0:
                attrs["kv_cache.cache_hit_rate"] = num_cached / num_prompt
        ext_tokens = kv_data.get("total_external_computed_tokens", 0)
        if ext_tokens > 0:
            attrs["kv_cache.has_external_cache"] = True
            attrs["kv_cache.num_external_computed_tokens"] = ext_tokens

    if phase == "decode" and num_preemptions > 0:
        attrs["kv_cache.total_preemptions"] = num_preemptions
        if kv_data["total_calls"] > 0:
            attrs["kv_cache.preemption_rate"] = num_preemptions / kv_data["total_calls"]

    span.set_attributes(attrs)
    span.end(end_time=start_ns)


def _create_store_span(tracer, phase_ctx, native_ctx, start_ns):
    """Create AscendStore child span if store stats exist."""
    store_stats = native_ctx.get("store_stats")
    if not store_stats:
        return
    has_save = store_stats.get("save_calls", 0) > 0
    has_load = store_stats.get("load_calls", 0) > 0
    if not (has_save or has_load):
        return

    span = tracer.start_span(
        name="vllm_ascend.prefill.ascend_store", context=phase_ctx, start_time=start_ns,
    )
    attrs = {}
    if has_save:
        attrs["ascend_store.save_calls"] = store_stats["save_calls"]
        if store_stats["save_duration_ms"] > 0:
            attrs["ascend_store.save_duration_ms"] = store_stats["save_duration_ms"]
            attrs["ascend_store.save_max_duration_ms"] = store_stats["save_max_duration_ms"]
            attrs["ascend_store.save_avg_duration_ms"] = (
                store_stats["save_duration_ms"] / store_stats["save_calls"]
            )
    if has_load:
        attrs["ascend_store.load_calls"] = store_stats["load_calls"]
        if store_stats["load_duration_ms"] > 0:
            attrs["ascend_store.load_duration_ms"] = store_stats["load_duration_ms"]
            attrs["ascend_store.load_max_duration_ms"] = store_stats["load_max_duration_ms"]
            attrs["ascend_store.load_avg_duration_ms"] = (
                store_stats["load_duration_ms"] / store_stats["load_calls"]
            )
    span.set_attributes(attrs)
    span.end(end_time=start_ns)