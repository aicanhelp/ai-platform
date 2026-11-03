# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.datastructures import Headers

import k8s_proxy_server_example as proxy


class HeaderRequest:
    """Minimal request object for Header-only routing tests."""

    def __init__(self, headers=None):
        self.headers = Headers(headers or {})


class ProxyRequest(HeaderRequest):
    """In-memory request supporting Header and JSON body access."""

    def __init__(self, body, headers=None):
        super().__init__(headers)
        self._body = body
        self._raw_body = httpx.Request(
            "POST", "http://proxy/request", json=body
        ).content

    async def json(self):
        return self._body

    async def body(self):
        return self._raw_body


class FakeDecoderResponse:
    """Async decoder response with controllable stream behavior."""

    def __init__(self, chunks=None, stream_error=None):
        self.chunks = chunks or []
        self.stream_error = stream_error
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk
        if self.stream_error is not None:
            raise self.stream_error

    async def aclose(self):
        self.closed = True


class FakeBackendClient:
    """In-memory httpx-compatible client used by orchestration tests."""

    def __init__(self, response_json=None, response_content=b"response"):
        self.response_json = response_json or {}
        self.response_content = response_content
        self.post_calls = []
        self.decode_calls = []

    async def post(self, endpoint, **kwargs):
        payload = copy.deepcopy(kwargs["json"])
        headers = dict(kwargs["headers"])
        self.post_calls.append((endpoint, payload, headers))
        request = httpx.Request(
            "POST",
            f"http://prefill{endpoint}",
            json=payload,
            headers=headers,
        )
        return httpx.Response(
            200,
            json=self.response_json,
            request=request,
        )

    def build_request(self, method, endpoint, **kwargs):
        payload = copy.deepcopy(kwargs["json"])
        headers = dict(kwargs["headers"])
        request = httpx.Request(
            method,
            f"http://decode{endpoint}",
            json=payload,
            headers=headers,
        )
        self.decode_calls.append((endpoint, payload, headers))
        return request

    async def send(self, request, stream=False):
        return httpx.Response(
            200,
            content=self.response_content,
            request=request,
        )


def new_fake_router():
    """Build a router mock with one Prefill and one Decode endpoint."""
    return SimpleNamespace(
        prefillers={
            "prefill-key": SimpleNamespace(
                url="http://10.0.0.1:8100",
                client=Mock(),
            )
        },
        decoders={
            "decode-key": SimpleNamespace(
                url="http://10.0.0.2:8200",
                client=Mock(),
            )
        },
        select_prefiller=Mock(return_value="prefill-key"),
        select_prefiller_by_address=Mock(return_value="prefill-key"),
        select_decoder=Mock(return_value="decode-key"),
        select_decoder_by_address=Mock(return_value="decode-key"),
        release_prefiller_tokens=Mock(),
        release_decoder_tokens=Mock(),
        release_kv=Mock(),
        mark_prefill_aborted=Mock(),
        acquire_aborted_for_prefiller=Mock(return_value=set()),
    )


def new_integration_router():
    """Build a router and in-memory backends for end-to-end orchestration."""
    prefill_client = FakeBackendClient(
        response_json={
            "kv_transfer_params": {
                "remote_engine_id": "prefill-engine",
                "remote_port": 12345,
            }
        }
    )
    decode_client = FakeBackendClient(
        response_content=b'{"id":"completion"}'
    )
    test_router = SimpleNamespace(
        prefillers={
            "prefill-key": SimpleNamespace(
                url="http://10.0.0.1:8100",
                client=prefill_client,
            )
        },
        decoders={
            "decode-key": SimpleNamespace(
                url="http://10.0.0.2:8200",
                client=decode_client,
            )
        },
        generate_request_id=AsyncMock(return_value="integration-request"),
        select_prefiller=Mock(return_value="prefill-key"),
        select_prefiller_by_address=Mock(return_value="prefill-key"),
        select_decoder=Mock(return_value="decode-key"),
        select_decoder_by_address=Mock(return_value="decode-key"),
        release_prefiller_tokens=Mock(),
        release_decoder_tokens=Mock(),
        release_kv=Mock(),
        mark_prefill_aborted=Mock(),
        acquire_aborted_for_prefiller=Mock(return_value=set()),
    )
    return test_router, prefill_client, decode_client


class NodeOverrideParserTests(unittest.TestCase):
    def parse(self, headers=None):
        return proxy._parse_node_overrides(HeaderRequest(headers))

    def assert_bad_request(self, headers):
        with self.assertRaises(HTTPException) as raised:
            self.parse(headers)
        self.assertEqual(raised.exception.status_code, 400)

    def test_legacy_without_overrides(self):
        prefill, decode = self.parse()

        self.assertIsNone(prefill)
        self.assertIsNone(decode)

    def test_pd_with_complete_overrides(self):
        prefill, decode = self.parse(
            {
                proxy.HEADER_PREFILL_POD_ADDRESS_PORT: "10.0.0.1:8100",
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
            }
        )

        self.assertEqual(prefill, ("10.0.0.1", 8100))
        self.assertEqual(decode, ("10.0.0.2", 8200))

    def test_legacy_ignores_prefill_only_override(self):
        prefill, decode = self.parse(
            {
                proxy.HEADER_PREFILL_POD_ADDRESS_PORT: "invalid-prefill",
            }
        )

        self.assertIsNone(prefill)
        self.assertIsNone(decode)

    def test_d_parses_decode_only_address(self):
        prefill, decode = self.parse(
            {
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
            }
        )

        self.assertIsNone(prefill)
        self.assertEqual(decode, ("10.0.0.2", 8200))

    def test_required_address_is_strictly_validated(self):
        invalid_addresses = (
            "missing-port",
            ":8200",
            "10.0.0.2:not-a-port",
            "10.0.0.2:0",
            "10.0.0.2:65536",
        )
        for address in invalid_addresses:
            with self.subTest(address=address):
                self.assert_bad_request(
                    {
                        proxy.HEADER_DECODE_POD_ADDRESS_PORT: address,
                    }
                )


class RoutingOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_d_path_skips_prefill_and_preserves_body(self):
        fake_router = new_fake_router()
        body = {
            "model": "test-model",
            "kv_transfer_params": {"from-client": True},
        }
        original_body = body.copy()
        decode_override = ("10.0.0.2", 8200)
        expected_response = object()
        build_response = AsyncMock(return_value=expected_response)

        with (
            patch.object(proxy, "router", fake_router),
            patch.object(
                proxy, "_build_streaming_response", build_response
            ),
        ):
            result = await proxy._proxy_decode_only_request(
                api_path="/v1/chat/completions",
                decode_override=decode_override,
                body=body,
                req_id="request-d",
                decode_cost=12.0,
                is_stream=True,
                media_type="text/event-stream",
                traceparent="00-trace",
            )

        self.assertIs(result, expected_response)
        self.assertEqual(body, original_body)
        fake_router.select_decoder_by_address.assert_called_once_with(
            "10.0.0.2", 8200, 12.0
        )
        fake_router.select_prefiller.assert_not_called()
        fake_router.select_prefiller_by_address.assert_not_called()
        call = build_response.await_args.kwargs
        self.assertNotIn("prefill_lease", call)
        self.assertEqual(call["traceparent"], "00-trace")

    async def test_pd_path_uses_explicit_endpoints_and_prefill_lease(self):
        fake_router = new_fake_router()
        body = {"model": "test-model"}
        request = HeaderRequest(
            {
                proxy.HEADER_PREFILL_POD_ADDRESS_PORT: "10.0.0.1:8100",
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
            }
        )
        read_request = AsyncMock(
            return_value=(
                body,
                False,
                "application/json",
                10.0,
                20.0,
                "request-pd",
            )
        )
        invoke_prefill = AsyncMock()
        build_response = AsyncMock(return_value=object())

        with (
            patch.object(proxy, "router", fake_router),
            patch.object(proxy, "_read_request_and_costs", read_request),
            patch.object(
                proxy,
                "_invoke_prefiller_and_update_body",
                invoke_prefill,
            ),
            patch.object(
                proxy, "_build_streaming_response", build_response
            ),
        ):
            await proxy._proxy_openai_request(
                "/v1/completions", request
            )

        fake_router.select_prefiller_by_address.assert_called_once_with(
            "10.0.0.1", 8100, 10.0
        )
        fake_router.select_decoder_by_address.assert_called_once_with(
            "10.0.0.2", 8200, 20.0
        )
        invoke_prefill.assert_awaited_once()
        call = build_response.await_args.kwargs
        self.assertEqual(
            call["prefill_lease"],
            proxy.PrefillLease(key="prefill-key", cost=10.0),
        )

    async def test_pd_releases_prefill_kv_when_decoder_selection_fails(self):
        fake_router = new_fake_router()
        fake_router.select_decoder_by_address.side_effect = (
            proxy.BackendEndpointUnavailableError(
                "decoder", "10.0.0.2", 8200
            )
        )
        request = HeaderRequest(
            {
                proxy.HEADER_PREFILL_POD_ADDRESS_PORT: "10.0.0.1:8100",
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
            }
        )
        read_request = AsyncMock(
            return_value=(
                {},
                False,
                "application/json",
                10.0,
                20.0,
                "request-pd-failed",
            )
        )

        with (
            patch.object(proxy, "router", fake_router),
            patch.object(proxy, "_read_request_and_costs", read_request),
            patch.object(
                proxy,
                "_invoke_prefiller_and_update_body",
                AsyncMock(),
            ),
            patch.object(
                proxy,
                "global_args",
                SimpleNamespace(discovery_interval=5),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await proxy._proxy_openai_request(
                    "/v1/completions", request
                )

        self.assertEqual(raised.exception.status_code, 503)
        fake_router.mark_prefill_aborted.assert_called_once_with(
            "prefill-key", "request-pd-failed"
        )
        fake_router.release_kv.assert_called_once_with("prefill-key", 10.0)

    async def test_legacy_without_overrides_uses_internal_selection(self):
        fake_router = new_fake_router()
        with patch.object(proxy, "router", fake_router):
            prefiller_key = proxy._choose_prefiller(
                None, 10.0, "request-legacy"
            )
            decoder_key = proxy._choose_decoder(
                None, 20.0, "request-legacy"
            )

        self.assertEqual(prefiller_key, "prefill-key")
        self.assertEqual(decoder_key, "decode-key")
        fake_router.select_prefiller.assert_called_once_with(10.0)
        fake_router.select_decoder.assert_called_once_with(20.0)
        fake_router.select_prefiller_by_address.assert_not_called()
        fake_router.select_decoder_by_address.assert_not_called()

    async def test_explicit_prefiller_none_is_service_unavailable(self):
        fake_router = new_fake_router()
        fake_router.select_prefiller_by_address.return_value = None
        with patch.object(proxy, "router", fake_router):
            with self.assertRaises(
                proxy.BackendEndpointUnavailableError
            ) as raised:
                proxy._choose_prefiller(
                    ("10.0.0.1", 8100), 10.0, "request-pd"
                )

        self.assertEqual(
            str(raised.exception),
            "prefiller endpoint 10.0.0.1:8100 is not currently available",
        )
        fake_router.select_prefiller.assert_not_called()

    async def test_explicit_decoder_none_is_service_unavailable(self):
        fake_router = new_fake_router()
        fake_router.select_decoder_by_address.return_value = None
        with patch.object(proxy, "router", fake_router):
            with self.assertRaises(
                proxy.BackendEndpointUnavailableError
            ) as raised:
                proxy._choose_decoder(
                    ("10.0.0.2", 8200), 20.0, "request-d"
                )

        self.assertEqual(
            str(raised.exception),
            "decoder endpoint 10.0.0.2:8200 is not currently available",
        )
        fake_router.select_decoder.assert_not_called()

    async def test_openai_handler_dispatches_d_before_prefill(self):
        request = HeaderRequest(
            {
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
                "traceparent": "00-trace",
            }
        )
        body = {"model": "test-model"}
        read_request = AsyncMock(
            return_value=(
                body,
                True,
                "text/event-stream",
                10.0,
                20.0,
                "request-d",
            )
        )
        d_handler = AsyncMock(return_value=object())

        with (
            patch.object(proxy, "_read_request_and_costs", read_request),
            patch.object(proxy, "_proxy_decode_only_request", d_handler),
        ):
            await proxy._proxy_openai_request(
                "/v1/chat/completions", request
            )

        d_handler.assert_awaited_once()
        self.assertEqual(
            d_handler.await_args.kwargs["decode_override"],
            ("10.0.0.2", 8200),
        )
        self.assertEqual(
            d_handler.await_args.kwargs["traceparent"], "00-trace"
        )

    async def test_decoder_open_unavailable_is_logged_and_returns_503(self):
        fake_router = new_fake_router()
        request = HeaderRequest(
            {
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
            }
        )
        read_request = AsyncMock(
            return_value=(
                {},
                False,
                "application/json",
                10.0,
                20.0,
                "request-unavailable",
            )
        )
        unavailable = proxy.BackendEndpointUnavailableError(
            "decoder", "10.0.0.2", 8200
        )
        open_decoder = AsyncMock(side_effect=unavailable)

        with (
            patch.object(proxy, "router", fake_router),
            patch.object(proxy, "_read_request_and_costs", read_request),
            patch.object(proxy, "_open_decoder_response", open_decoder),
            patch.object(
                proxy,
                "global_args",
                SimpleNamespace(
                    max_retries=1,
                    retry_interval=0,
                    discovery_interval=5,
                ),
            ),
        ):
            with self.assertLogs(
                proxy.logger.name, level="WARNING"
            ) as captured_logs:
                with self.assertRaises(HTTPException) as raised:
                    await proxy._proxy_openai_request(
                        "/v1/completions", request
                    )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            "decoder endpoint 10.0.0.2:8200 is not currently available",
        )
        self.assertEqual(raised.exception.headers["Retry-After"], "5")
        self.assertTrue(
            any(
                "Backend endpoint currently unavailable for request "
                "request-unavailable: decoder endpoint 10.0.0.2:8200 "
                "is not currently available" in message
                for message in captured_logs.output
            )
        )
        open_decoder.assert_awaited_once()
        fake_router.release_decoder_tokens.assert_called_once_with(
            "decode-key", 20.0
        )


class ProxyOrchestrationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        (
            self.test_router,
            self.prefill_client,
            self.decode_client,
        ) = new_integration_router()
        self.router_patch = patch.object(
            proxy, "router", self.test_router
        )
        self.args_patch = patch.object(
            proxy,
            "global_args",
            SimpleNamespace(
                max_retries=1,
                retry_interval=0,
                discovery_interval=5,
            ),
        )
        self.router_patch.start()
        self.args_patch.start()
        self.addCleanup(self.router_patch.stop)
        self.addCleanup(self.args_patch.stop)

    async def consume(self, response):
        self.assertIsInstance(response, StreamingResponse)
        return b"".join(
            [chunk async for chunk in response.body_iterator]
        )

    async def test_decode_only_request_returns_streaming_response(self):
        body = {
            "model": "test-model",
            "kv_transfer_params": {"from-client": True},
        }
        response = await proxy._proxy_decode_only_request(
            api_path="/v1/chat/completions",
            decode_override=("10.0.0.2", 8200),
            body=body,
            req_id="integration-d",
            decode_cost=20.0,
            is_stream=True,
            media_type="text/event-stream",
            traceparent="00-integration-trace",
        )
        content = await self.consume(response)

        self.assertEqual(content, b'{"id":"completion"}')
        self.assertEqual(self.prefill_client.post_calls, [])
        self.assertEqual(len(self.decode_client.decode_calls), 1)
        _, decode_payload, decode_headers = (
            self.decode_client.decode_calls[0]
        )
        self.assertEqual(decode_payload, body)
        self.assertEqual(
            decode_headers["traceparent"], "00-integration-trace"
        )
        self.test_router.select_decoder_by_address.assert_called_once_with(
            "10.0.0.2", 8200, 20.0
        )
        self.test_router.select_prefiller.assert_not_called()
        self.test_router.release_kv.assert_not_called()
        self.test_router.release_decoder_tokens.assert_called_once_with(
            "decode-key", 20.0
        )

    async def test_d_path_invalid_decode_address_raises_http_400(self):
        request = ProxyRequest(
            body={"model": "test-model"},
            headers={
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "invalid-address",
            },
        )

        with self.assertRaises(HTTPException) as raised:
            await proxy._proxy_openai_request(
                "/v1/chat/completions", request
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.test_router.generate_request_id.assert_not_awaited()
        self.assertEqual(self.prefill_client.post_calls, [])
        self.assertEqual(self.decode_client.decode_calls, [])

    async def test_prefill_decode_request_executes_complete_flow(self):
        body = {"model": "test-model", "stream": False}
        request = ProxyRequest(
            body=body,
            headers={
                proxy.HEADER_PREFILL_POD_ADDRESS_PORT: "10.0.0.1:8100",
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
                "traceparent": "00-integration-trace",
            },
        )
        request_length = len(await request.body())
        expected_prefill_cost = (
            request_length / 4.0 * 0.0345 + 120.0745
        )
        expected_decode_cost = float(request_length)

        response = await proxy._proxy_openai_request(
            "/v1/completions", request
        )
        content = await self.consume(response)

        self.assertEqual(content, b'{"id":"completion"}')
        self.assertEqual(len(self.prefill_client.post_calls), 1)
        self.assertEqual(len(self.decode_client.decode_calls), 1)
        _, prefill_payload, prefill_headers = (
            self.prefill_client.post_calls[0]
        )
        _, decode_payload, decode_headers = (
            self.decode_client.decode_calls[0]
        )
        self.assertFalse(prefill_payload["stream"])
        self.assertEqual(prefill_payload["max_tokens"], 1)
        self.assertEqual(
            decode_payload["kv_transfer_params"]["remote_engine_id"],
            "prefill-engine",
        )
        self.assertEqual(
            prefill_headers["traceparent"], "00-integration-trace"
        )
        self.assertEqual(
            decode_headers["traceparent"], "00-integration-trace"
        )
        self.test_router.release_prefiller_tokens.assert_called_once_with(
            "prefill-key", expected_prefill_cost
        )
        self.test_router.release_kv.assert_called_once_with(
            "prefill-key", expected_prefill_cost
        )
        self.test_router.release_decoder_tokens.assert_called_once_with(
            "decode-key", expected_decode_cost
        )

    async def test_openai_request_routes_d_path(self):
        request = ProxyRequest(
            body={"model": "test-model", "stream": True},
            headers={
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
            },
        )

        response = await proxy._proxy_openai_request(
            "/v1/chat/completions", request
        )
        await self.consume(response)

        self.assertEqual(self.prefill_client.post_calls, [])
        self.assertEqual(len(self.decode_client.decode_calls), 1)
        self.test_router.select_decoder_by_address.assert_called_once()
        self.test_router.select_prefiller.assert_not_called()

    async def test_openai_request_routes_pd_path(self):
        request = ProxyRequest(
            body={"model": "test-model", "stream": False},
            headers={
                proxy.HEADER_PREFILL_POD_ADDRESS_PORT: "10.0.0.1:8100",
                proxy.HEADER_DECODE_POD_ADDRESS_PORT: "10.0.0.2:8200",
            },
        )

        response = await proxy._proxy_openai_request(
            "/v1/completions", request
        )
        await self.consume(response)

        self.assertEqual(len(self.prefill_client.post_calls), 1)
        self.assertEqual(len(self.decode_client.decode_calls), 1)
        self.test_router.select_prefiller_by_address.assert_called_once()
        self.test_router.select_decoder_by_address.assert_called_once()
        self.test_router.select_prefiller.assert_not_called()
        self.test_router.select_decoder.assert_not_called()

    async def test_openai_request_routes_legacy_path(self):
        request = ProxyRequest(
            body={"model": "test-model", "stream": False}
        )

        response = await proxy._proxy_openai_request(
            "/v1/completions", request
        )
        await self.consume(response)

        self.assertEqual(len(self.prefill_client.post_calls), 1)
        self.assertEqual(len(self.decode_client.decode_calls), 1)
        self.test_router.select_prefiller.assert_called_once()
        self.test_router.select_decoder.assert_called_once()
        self.test_router.select_prefiller_by_address.assert_not_called()
        self.test_router.select_decoder_by_address.assert_not_called()


class StreamingResponseResourceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake_router = new_fake_router()
        self.router_patch = patch.object(
            proxy, "router", self.fake_router
        )
        self.args_patch = patch.object(
            proxy,
            "global_args",
            SimpleNamespace(max_retries=1, retry_interval=0),
        )
        self.router_patch.start()
        self.args_patch.start()
        self.addCleanup(self.router_patch.stop)
        self.addCleanup(self.args_patch.stop)

    async def consume(self, response):
        return [chunk async for chunk in response.body_iterator]

    async def test_d_stream_releases_only_decode(self):
        decoder_response = FakeDecoderResponse([b"first", b"second"])
        body = {
            "model": "test-model",
            "kv_transfer_params": {"from-client": True},
        }
        open_decoder = AsyncMock(return_value=decoder_response)

        with patch.object(
            proxy, "_open_decoder_response", open_decoder
        ):
            response = await proxy._build_streaming_response(
                api_path="/v1/chat/completions",
                body=body,
                req_id="request-d",
                decoder_key="decode-key",
                decode_cost=20.0,
                is_stream=True,
                media_type="text/event-stream",
                traceparent="00-trace",
            )
            chunks = await self.consume(response)

        self.assertEqual(chunks, [b"first", b"second"])
        self.assertTrue(decoder_response.closed)
        self.fake_router.release_decoder_tokens.assert_called_once_with(
            "decode-key", 20.0
        )
        self.fake_router.release_kv.assert_not_called()
        self.fake_router.mark_prefill_aborted.assert_not_called()
        service_request = open_decoder.await_args.args[0]
        self.assertIs(service_request.payload, body)
        self.assertIsNone(service_request.prefiller_key)
        self.assertEqual(service_request.traceparent, "00-trace")

    async def test_d_open_failure_releases_decode(self):
        open_decoder = AsyncMock(side_effect=RuntimeError("open failed"))

        with patch.object(
            proxy, "_open_decoder_response", open_decoder
        ):
            with self.assertRaisesRegex(RuntimeError, "open failed"):
                await proxy._build_streaming_response(
                    api_path="/v1/completions",
                    body={},
                    req_id="request-d-failed",
                    decoder_key="decode-key",
                    decode_cost=20.0,
                    is_stream=False,
                    media_type="application/json",
                )

        self.fake_router.release_decoder_tokens.assert_called_once_with(
            "decode-key", 20.0
        )
        self.fake_router.release_kv.assert_not_called()

    async def test_d_stream_failure_closes_response_and_releases_decode(self):
        decoder_response = FakeDecoderResponse(
            stream_error=RuntimeError("stream failed")
        )
        open_decoder = AsyncMock(return_value=decoder_response)

        with patch.object(
            proxy, "_open_decoder_response", open_decoder
        ):
            response = await proxy._build_streaming_response(
                api_path="/v1/completions",
                body={},
                req_id="request-d-stream-failed",
                decoder_key="decode-key",
                decode_cost=20.0,
                is_stream=True,
                media_type="text/event-stream",
            )
            chunks = await self.consume(response)

        self.assertEqual(chunks, [])
        self.assertTrue(decoder_response.closed)
        self.fake_router.release_decoder_tokens.assert_called_once_with(
            "decode-key", 20.0
        )
        self.fake_router.release_kv.assert_not_called()
        self.fake_router.mark_prefill_aborted.assert_not_called()

    async def test_pd_open_failure_releases_both_routes(self):
        open_decoder = AsyncMock(side_effect=RuntimeError("open failed"))

        with patch.object(
            proxy, "_open_decoder_response", open_decoder
        ):
            with self.assertRaisesRegex(RuntimeError, "open failed"):
                await proxy._build_streaming_response(
                    api_path="/v1/completions",
                    body={},
                    req_id="request-pd-open-failed",
                    decoder_key="decode-key",
                    decode_cost=20.0,
                    is_stream=False,
                    media_type="application/json",
                    prefill_lease=proxy.PrefillLease(
                        key="prefill-key", cost=10.0
                    ),
                )

        self.fake_router.mark_prefill_aborted.assert_called_once_with(
            "prefill-key", "request-pd-open-failed"
        )
        self.fake_router.release_kv.assert_called_once_with(
            "prefill-key", 10.0
        )
        self.fake_router.release_decoder_tokens.assert_called_once_with(
            "decode-key", 20.0
        )

    async def test_pd_stream_releases_prefill_kv_and_decode_once(self):
        decoder_response = FakeDecoderResponse([b"response"])
        open_decoder = AsyncMock(return_value=decoder_response)

        with patch.object(
            proxy, "_open_decoder_response", open_decoder
        ):
            response = await proxy._build_streaming_response(
                api_path="/v1/completions",
                body={},
                req_id="request-pd",
                decoder_key="decode-key",
                decode_cost=20.0,
                is_stream=False,
                media_type="application/json",
                prefill_lease=proxy.PrefillLease(
                    key="prefill-key", cost=10.0
                ),
            )
            await self.consume(response)

        self.fake_router.release_kv.assert_called_once_with(
            "prefill-key", 10.0
        )
        self.fake_router.release_decoder_tokens.assert_called_once_with(
            "decode-key", 20.0
        )
        self.fake_router.mark_prefill_aborted.assert_not_called()

    async def test_prefill_request_requires_prefiller_key(self):
        request = proxy.ServiceRequest(
            client=Mock(),
            endpoint="/v1/completions",
            payload={},
            request_id="missing-prefill-key",
        )

        with self.assertRaisesRegex(ValueError, "prefiller_key"):
            await proxy._forward_prefill_once(request)


if __name__ == "__main__":
    unittest.main()
