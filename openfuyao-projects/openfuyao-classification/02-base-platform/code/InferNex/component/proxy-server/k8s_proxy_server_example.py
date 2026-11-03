# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import asyncio
import logging
import heapq
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

MAX_INT = sys.maxsize
MAX_BACKEND_ERROR_BODY_BYTES = 64 * 1024

# Header names for explicitly pinning nodes
HEADER_PREFILL_POD_ADDRESS_PORT = "x-openfuyao-prefill-pod-address-port"
HEADER_DECODE_POD_ADDRESS_PORT = "x-openfuyao-decode-pod-address-port"


NodeAddress = Tuple[str, int]
NodeOverrides = Tuple[Optional[NodeAddress], Optional[NodeAddress]]


@dataclass(frozen=True)
class PrefillLease:
    """Prefill resources that must be released while Decode is streamed."""

    key: str
    cost: float


@dataclass
class ServiceRequest:
    """Request parameters used when forwarding to backend services."""

    client: httpx.AsyncClient
    endpoint: str
    payload: dict
    request_id: str
    prefiller_key: Optional[str] = None
    max_retries: int = 3
    retry_interval: float = 1
    traceparent: Optional[str] = None


class BackendEndpointUnavailableError(RuntimeError):
    """No usable backend endpoint is currently available."""

    def __init__(
        self,
        role: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        self.role = role
        self.host = host
        self.port = port

        if host is not None and port is not None:
            message = (
                f"{role} endpoint {host}:{port} is not currently available"
            )
        else:
            message = f"No {role} endpoints are currently available"

        super().__init__(message)


class BackendHTTPError(Exception):
    """An HTTP error response returned by a prefill or decode backend."""

    def __init__(
        self,
        backend: str,
        response: httpx.Response,
        content: Optional[bytes] = None,
    ):
        self.backend = backend
        self.status_code = response.status_code
        self.content = response.content if content is None else content
        self.headers = {
            name: value
            for name in ("content-type", "retry-after")
            if (value := response.headers.get(name)) is not None
        }
        super().__init__(
            f"{backend} returned HTTP {response.status_code} for "
            f"{response.request.method} {response.request.url}"
        )


@dataclass(frozen=True)
class BackendEndpoint:
    host: str
    port: int
    pod_name: str
    group_id: str
    role: str


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def _parse_excluded_endpoints(raw_endpoints: str) -> frozenset[str]:
    """Return the non-empty, comma-separated endpoint paths."""
    return frozenset(
        endpoint.strip()
        for endpoint in raw_endpoints.split(",")
        if endpoint.strip()
    )


class SuccessfulEndpointAccessLogFilter(logging.Filter):
    """Suppress successful Uvicorn access logs for selected endpoint paths."""

    def __init__(self, excluded_endpoints: str):
        super().__init__()
        self.excluded_endpoints = _parse_excluded_endpoints(excluded_endpoints)

    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn 0.41.0 (pinned in requirements.txt) emits access records
        # with arguments in this order:
        # (client_addr, method, path_with_query, http_version, status_code).
        # Keep unknown records so a Uvicorn logging change cannot hide errors.
        if not isinstance(record.args, tuple) or len(record.args) != 5:
            return True

        try:
            path = str(record.args[2]).partition("?")[0]
            status_code = int(record.args[4])
        except (TypeError, ValueError):
            return True

        return status_code >= 400 or path not in self.excluded_endpoints


def _configure_access_log_filter(excluded_endpoints: str) -> None:
    """Install the endpoint filter on Uvicorn's access logger."""
    access_logger = logging.getLogger("uvicorn.access")
    for existing_filter in list(access_logger.filters):
        if isinstance(existing_filter, SuccessfulEndpointAccessLogFilter):
            access_logger.removeFilter(existing_filter)

    if _parse_excluded_endpoints(excluded_endpoints):
        access_logger.addFilter(
            SuccessfulEndpointAccessLogFilter(excluded_endpoints)
        )


# --- Kubernetes service discovery ----------------------------------------------------------

try:
    from kubernetes import (client as k8s_client, config as k8s_config, watch as k8s_watch)

    K8S_AVAILABLE = True
except ImportError:
    k8s_client = None  # type: ignore
    k8s_config = None  # type: ignore
    k8s_watch = None  # type: ignore
    K8S_AVAILABLE = False
    logger.warning(
        "kubernetes package not available. Install with: pip install kubernetes"
    )


class VllmServiceDiscovery:
    """Watches and updates prefiller / decoder pod endpoints in a namespace."""

    def __init__(self, args):
        self.prefiller_labels = args.prefiller_labels
        self.decoder_labels = args.decoder_labels
        self.namespace = args.namespace
        self.discovery_interval = args.discovery_interval
        self.prefiller_container = args.prefiller_container
        self.decoder_container = args.decoder_container
        self.prefiller_port_name = args.prefiller_port_name
        self.decoder_port_name = args.decoder_port_name

        self._prefiller_endpoints: List[BackendEndpoint] = []
        self._decoder_endpoints: List[BackendEndpoint] = []
        self._lock = threading.Lock()
        self._discovery_lock = threading.Lock()
        self._running = False
        self._watch_threads: List[threading.Thread] = []
        self._watchers = []
        self._last_discovery_counts: Optional[Tuple[int, int]] = None

        if not K8S_AVAILABLE:
            logger.warning("K8s client not available. Service discovery disabled.")
            return

        try:
            # Prefer in-cluster config; fall back to local kubeconfig
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
        except Exception:
            logger.warning(
                "Could not load Kubernetes configuration; "
                "automatic service discovery will not run."
            )
            return

        configuration = k8s_client.Configuration.get_default_copy()
        self._normalize_k8s_bearer_token_config(configuration)
        k8s_client.Configuration.set_default(configuration)
        self._core_v1 = k8s_client.CoreV1Api(k8s_client.ApiClient(configuration))
        self.start()

    @staticmethod
    def _normalize_k8s_bearer_token_config(configuration) -> None:
        token = configuration.api_key.get("authorization")
        if "BearerToken" in configuration.api_key or not token:
            return

        if token.lower().startswith("bearer "):
            token = token.split(" ", 1)[1]

        configuration.api_key["BearerToken"] = token
        configuration.api_key_prefix["BearerToken"] = "Bearer"

    def start(self) -> None:
        if self._running:
            return
        if not K8S_AVAILABLE or not hasattr(self, "_core_v1"):
            return
        self._running = True

        # Build an initial snapshot before watching for subsequent changes.
        self._discovery_once()

        self._watch_threads = [
            threading.Thread(
                target=self._watch_loop,
                args=(self.prefiller_labels, "prefill"),
                daemon=True,
            ),
            threading.Thread(
                target=self._watch_loop,
                args=(self.decoder_labels, "decode"),
                daemon=True,
            ),
        ]
        for thread in self._watch_threads:
            thread.start()
        logger.info("K8s vLLM service discovery started.")

    def stop(self) -> None:
        with self._lock:
            self._running = False
            watchers = list(self._watchers)
        for watcher in watchers:
            watcher.stop()

        for thread in self._watch_threads:
            thread.join(timeout=5)
        logger.info("K8s vLLM service discovery stopped.")

    def get_prefiller_instances(self) -> List[BackendEndpoint]:
        with self._lock:
            return list(self._prefiller_endpoints)

    def get_decoder_instances(self) -> List[BackendEndpoint]:
        with self._lock:
            return list(self._decoder_endpoints)

    def pod_health_status(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[str, str]]]:
        if not K8S_AVAILABLE or not hasattr(self, "_core_v1"):
            return (0, 0), (0, 0), []

        try:
            prefill_pods = self._list_pods(self.prefiller_labels).items
            decode_pods = self._list_pods(self.decoder_labels).items
            pods = list(prefill_pods) + list(decode_pods)
            prefill_ready_count = sum(
                1 for pod in prefill_pods if self._is_pod_ready(pod)
            )
            decode_ready_count = sum(
                1 for pod in decode_pods if self._is_pod_ready(pod)
            )
            not_ready_pods = [
                (
                    getattr(pod.metadata, "name", "unknown"),
                    self._pod_ready_status(pod),
                )
                for pod in pods
                if not self._is_pod_ready(pod)
            ]
            return (
                (prefill_ready_count, len(prefill_pods)),
                (decode_ready_count, len(decode_pods)),
                not_ready_pods,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to collect pod health status: %s", exc)
            return (0, 0), (0, 0), []

    def _list_pods(self, label_selector: str):
        return self._core_v1.list_namespaced_pod(
            namespace=self.namespace, label_selector=label_selector
        )

    @staticmethod
    def _resolve_container_port(pod, container_name: str, port_name: str) -> int:
        """Best-effort extraction of the target container port."""
        default_port = 8100 if "prefill" in pod.metadata.name.lower() else 8200
        try:
            containers = pod.spec.containers or []
            if not containers:
                return default_port

            target = None
            if container_name:
                for c in containers:
                    if c.name == container_name:
                        target = c
                        break
            if target is None:
                target = containers[0]

            if target.ports:
                # prefer named port
                for port in target.ports:
                    if getattr(port, "name", None) and port.name.lower() == port_name:
                        logger.debug("Found named port '%s' with value %s for container %s",
                            port.name, port.container_port, target.name)
                        return port.container_port
                # otherwise first port
                logger.debug("Using first port %s for container %s",
                    target.ports[0].container_port, target.name)
                return target.ports[0].container_port
            return default_port
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to resolve container port for pod %s: %s",
                pod.metadata.name,
                exc,
            )
            return default_port

    @staticmethod
    def _is_pod_ready(pod) -> bool:
        return VllmServiceDiscovery._pod_ready_status(pod) == "Ready=True"

    @staticmethod
    def _pod_ready_status(pod) -> str:
        for condition in pod.status.conditions or []:
            if condition.type == "Ready":
                return f"Ready={condition.status}"
        return "Ready=Unknown"

    @staticmethod
    def _endpoint_from_pod(pod, port: int, role: str) -> BackendEndpoint:
        labels = getattr(pod.metadata, "labels", None) or {}
        return BackendEndpoint(
            host=pod.status.pod_ip,
            port=port,
            pod_name=getattr(pod.metadata, "name", pod.status.pod_ip),
            group_id=labels.get("openfuyao.com/pdGroupID", ""),
            role=role,
        )

    def _log_discovery_if_changed(self) -> None:
        with self._lock:
            counts = (
                len(self._prefiller_endpoints),
                len(self._decoder_endpoints),
            )

            if counts == self._last_discovery_counts:
                return

            self._last_discovery_counts = counts

        logger.info(
            "Service discovery changed: %d prefill, %d decode",
            counts[0],
            counts[1],
        )

    def _discovery_once(self) -> None:
        if not K8S_AVAILABLE or not hasattr(self, "_core_v1"):
            return
        
        # Lock order: _discovery_lock -> _lock.
        # Callers must not hold _lock when entering this method.
        
        # Serialize concurrent discovery triggered by watch events
        # to avoid stale snapshots overwriting newer results.
        with self._discovery_lock:
            try:
                prefiller_eps: List[BackendEndpoint] = []
                decoder_eps: List[BackendEndpoint] = []

                # Prefillers
                pods = self._list_pods(self.prefiller_labels)
                for pod in pods.items:
                    if (
                        pod.status.phase != "Running"
                        or not self._is_pod_ready(pod)
                        or pod.metadata.deletion_timestamp is not None
                    ):
                        continue
                    port = self._resolve_container_port(
                        pod, self.prefiller_container, self.prefiller_port_name
                    )
                    prefiller_eps.append(
                        self._endpoint_from_pod(pod, port, "prefill")
                    )

                # Decoders
                pods = self._list_pods(self.decoder_labels)
                for pod in pods.items:
                    if (
                        pod.status.phase != "Running"
                        or not self._is_pod_ready(pod)
                        or pod.metadata.deletion_timestamp is not None
                    ):
                        continue
                    port = self._resolve_container_port(
                        pod, self.decoder_container, self.decoder_port_name
                    )
                    decoder_eps.append(
                        self._endpoint_from_pod(pod, port, "decode")
                    )

                with self._lock:
                    self._prefiller_endpoints = prefiller_eps
                    self._decoder_endpoints = decoder_eps

                self._log_discovery_if_changed()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Service discovery iteration failed: %s", exc)

    def _watch_loop(self, label_selector: str, role: str) -> None:
        while self._running:
            watcher = k8s_watch.Watch()

            with self._lock:
                if not self._running:
                    return
                self._watchers.append(watcher)

            try:
                for event in watcher.stream(
                    self._core_v1.list_namespaced_pod,
                    namespace=self.namespace,
                    label_selector=label_selector,
                    timeout_seconds=60,
                ):
                    if not self._running:
                        break

                    event_type = event.get("type")
                    if event_type in {"ADDED", "MODIFIED", "DELETED"}:
                        self._discovery_once()
                    elif event_type == "ERROR":
                        logger.warning("%s pod watch returned an error event", role)
                        break

            except Exception as exc:
                if self._running:
                    logger.warning(
                        "%s pod watch disconnected: %s",
                        role,
                        exc,
                    )
                    time.sleep(1)
            
            finally:
                watcher.stop()
                with self._lock:
                    if watcher in self._watchers:
                        self._watchers.remove(watcher)

            if not self._running:
                break
            
            self._discovery_once()


# --- Backend pools and load balancer router --------------------------------------------------------


class ServerState:
    """Single backend node (prefiller / decoder)."""

    def __init__(self, host: str, port: int, pod_name: str = "",
                 group_id: str = "", role: str = ""):
        self.host = host
        self.port = port
        self.pod_name = pod_name or f"{host}:{port}"
        self.group_id = group_id
        self.role = role
        self.url = f"http://{host}:{port}"
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(
                max_connections=100000,
                max_keepalive_connections=100000,
            ),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0
        self.active_requests = 0
        self.aborted_requests = set()


class BackendPool:
    """Manage a homogeneous group of backend nodes and provide load‑balanced selection."""

    def __init__(self, role: str, weight_kv_cache: bool):
        self.role = role
        self.weight_kv_cache = weight_kv_cache
        self.servers: Dict[str, ServerState] = {}
        self._heap: List[Tuple[float, str]] = []
        self._lock = threading.Lock()

    def sync_from_endpoints(self, endpoints: List[BackendEndpoint]) -> None:
        """Reconcile current nodes with the given endpoint list."""
        with self._lock:
            desired = {f"{endpoint.host}:{endpoint.port}" for endpoint in endpoints}

            # Add new nodes
            for endpoint in endpoints:
                key = f"{endpoint.host}:{endpoint.port}"
                if key not in self.servers:
                    self.servers[key] = ServerState(
                        endpoint.host,
                        endpoint.port,
                        endpoint.pod_name,
                        endpoint.group_id,
                        endpoint.role,
                    )
                    logger.info("Added %s node %s", self.role, key)
                    # Directly push into heap since membership changes are infrequent
                    heapq.heappush(self._heap, (0.0, key))
                else:
                    server = self.servers[key]
                    server.pod_name = endpoint.pod_name or key
                    server.group_id = endpoint.group_id
                    server.role = endpoint.role

            existing = set(self.servers.keys())

            # Mark removed nodes as overloaded
            for key in existing - desired:
                server = self.servers.pop(key, None)
                if server is not None:
                    server.active_tokens = MAX_INT
                    logger.info("Removed %s node %s (marked as overloaded)", self.role, key)

    def _priority_for(self, server: ServerState) -> float:
        value = float(server.active_tokens)
        if self.weight_kv_cache:
            value += server.active_kv_cache * 0.3
        return value

    def _prune_stale_head(self) -> None:
        while self._heap:
            _, key = self._heap[0]
            if key in self.servers:
                return
            heapq.heappop(self._heap)

    def choose(self, token_cost: float) -> str:
        """Pick the least‑loaded node and charge it."""
        with self._lock:
            self._prune_stale_head()
            if not self._heap:
                raise BackendEndpointUnavailableError(self.role)
            _, key = heapq.heappop(self._heap)
            server = self.servers.get(key)
            if server is None:
                return self.choose(token_cost)
            server.active_tokens += token_cost
            if self.weight_kv_cache:
                server.active_kv_cache += token_cost
            heapq.heappush(self._heap, (self._priority_for(server), key))
            return key

    def choose_specific(self, host: str, port: int, token_cost: float) -> str:
        """Use a caller‑specified node, if present, and charge it."""
        key = f"{host}:{port}"
        with self._lock:
            server = self.servers.get(key)
            if server is None:
                raise BackendEndpointUnavailableError(self.role, host, port)
            server.active_tokens += token_cost
            if self.weight_kv_cache:
                server.active_kv_cache += token_cost
            heapq.heappush(self._heap, (self._priority_for(server), key))
            return key

    def release_tokens(self, key: str, token_cost: float) -> None:
        with self._lock:
            server = self.servers.get(key)
            if server is None:
                return
            server.active_tokens -= token_cost
            heapq.heappush(self._heap, (self._priority_for(server), key))

    def release_kv(self, key: str, token_cost: float) -> None:
        with self._lock:
            server = self.servers.get(key)
            if server is None:
                return
            if server.active_kv_cache > 0:
                server.active_kv_cache -= token_cost
            heapq.heappush(self._heap, (self._priority_for(server), key))

    def peek_next_key(self) -> str:
        """Return the key of the node that would be selected next."""
        self._prune_stale_head()
        if not self._heap:
            raise RuntimeError(f"No {self.role} servers available")
        return self._heap[0][1]

    def record_aborted(self, key: str, request_id: str) -> None:
        server = self.servers.get(key)
        if server is not None:
            server.aborted_requests.add(request_id)

    def take_aborted(self, key: str):
        server = self.servers.get(key)
        if server is None:
            return set()
        ids = server.aborted_requests.copy()
        server.aborted_requests.clear()
        return ids


class LoadBalanceRouter:
    """Top‑level proxy router: discovery + prefill / decode pools + routing helpers."""

    def __init__(self, discovery: VllmServiceDiscovery):
        self._discovery = discovery
        self._prefill_pool = BackendPool("prefiller", weight_kv_cache=True)
        self._decode_pool = BackendPool("decoder", weight_kv_cache=False)
        self._req_id_lock = asyncio.Lock()
        self._instance_lock = threading.Lock()

        self._sync_from_discovery()

    # Expose read‑only views for diagnostics / healthcheck
    @property
    def prefillers(self) -> Dict[str, ServerState]:
        return self._prefill_pool.servers

    @property
    def decoders(self) -> Dict[str, ServerState]:
        return self._decode_pool.servers

    def _sync_from_discovery(self) -> None:
        with self._instance_lock:
            self._prefill_pool.sync_from_endpoints(
                self._discovery.get_prefiller_instances()
            )
            self._decode_pool.sync_from_endpoints(
                self._discovery.get_decoder_instances()
            )
            logger.debug(
                "Synced instances: prefiller=%d, decoder=%d",
                len(self.prefillers),
                len(self.decoders),
            )

    def refresh_instances(self) -> None:
        self._sync_from_discovery()

    def pod_health_status(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[str, str]]]:
        return self._discovery.pod_health_status()

    def mark_prefill_aborted(self, prefiller_key: str, request_id: str) -> None:
        self._prefill_pool.record_aborted(prefiller_key, request_id)

    def acquire_aborted_for_prefiller(self, prefiller_key: str):
        return self._prefill_pool.take_aborted(prefiller_key)

    async def generate_request_id(self) -> str:
        async with self._req_id_lock:
            return str(uuid.uuid4())

    def next_metrics_prefiller_key(self) -> str:
        return self._prefill_pool.peek_next_key()

    def select_prefiller(self, cost: float) -> str:
        return self._prefill_pool.choose(cost)

    def select_prefiller_by_address(self, host: str, port: int,
                                    cost: float) -> str:
        return self._prefill_pool.choose_specific(host, port, cost)

    def release_prefiller_tokens(self, key: str, cost: float) -> None:
        self._prefill_pool.release_tokens(key, cost)

    def release_kv(self, key: str, cost: float) -> None:
        """Release KV cache usage on a prefiller node."""
        self._prefill_pool.release_kv(key, cost)

    def select_decoder(self, cost: float) -> str:
        return self._decode_pool.choose(cost)

    def select_decoder_by_address(self, host: str, port: int,
                                  cost: float) -> str:
        return self._decode_pool.choose_specific(host, port, cost)

    def release_decoder_tokens(self, key: str, cost: float) -> None:
        self._decode_pool.release_tokens(key, cost)


router: Optional[LoadBalanceRouter] = None
global_args = None  # populated in __main__


async def stop_proxy_server(discovery: VllmServiceDiscovery,
                            state: LoadBalanceRouter,
                            refresh_task: asyncio.Task) -> None:
    """Gracefully stop background tasks and close all backend clients."""
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass

    discovery.stop()
    for server in state.prefillers.values():
        await server.client.aclose()
    for server in state.decoders.values():
        await server.client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router

    discovery = VllmServiceDiscovery(global_args)
    router = LoadBalanceRouter(discovery)
    logger.info(
        "Proxy initialized: %d prefiller nodes, %d decoder nodes.",
        len(router.prefillers),
        len(router.decoders),
    )

    async def refresh_loop():
        while True:
            await asyncio.sleep(global_args.discovery_interval)
            try:
                router.refresh_instances()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to refresh instances: %s", exc)

    refresh_task = asyncio.create_task(refresh_loop())

    try:
        yield
    finally:
        await stop_proxy_server(discovery, router, refresh_task)


async def _listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler):
    import functools
    @functools.wraps(handler)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs["request"]
        task_handler = asyncio.create_task(handler(*args, **kwargs))
        task_cancel = asyncio.create_task(_listen_for_disconnect(request))
        done, pending = await asyncio.wait(
            [task_handler, task_cancel], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        if task_handler in done:
            return task_handler.result()
        return None

    return wrapper


app = FastAPI(lifespan=lifespan)


async def _forward_prefill_once(req: ServiceRequest):
    """Call prefiller with non‑streaming settings (single response)."""
    if req.prefiller_key is None:
        raise ValueError("prefiller_key is required for a Prefill request")

    aborted_ids = router.acquire_aborted_for_prefiller(req.prefiller_key)
    body = req.payload.copy()
    body["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_ids),
    }
    body["stream"] = False
    body["max_tokens"] = 1
    if "stream_options" in body:
        body.pop("stream_options", None)

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": req.request_id,
    }
    if req.traceparent:
        headers["traceparent"] = req.traceparent

    attempt = 0
    while attempt < req.max_retries:
        attempt += 1
        try:
            resp = await req.client.post(req.endpoint,
                                         json=body,
                                         headers=headers)
            if not resp.is_success:
                raise BackendHTTPError("prefiller", resp)
            return resp
        except (httpx.RequestError, BackendHTTPError) as exc:
            logger.warning("Prefiller call failed on attempt %d for %s: %s",
                           attempt, req.endpoint, exc)
            if attempt >= req.max_retries:
                logger.error(
                    "Prefiller failed after %d attempts requesting to %s.",
                    attempt,
                    req.endpoint,
                )
                raise
        await asyncio.sleep(req.retry_interval)


async def _open_decoder_response(req: ServiceRequest) -> httpx.Response:
    """Open a decoder stream before the downstream response is committed."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": req.request_id,
    }
    if req.traceparent:
        headers["traceparent"] = req.traceparent

    attempt = 0
    while attempt < req.max_retries:
        attempt += 1
        try:
            request = req.client.build_request(
                "POST", req.endpoint, json=req.payload, headers=headers
            )
            resp = await req.client.send(request, stream=True)
            if not resp.is_success:
                try:
                    content = bytearray()
                    truncated = False
                    async for chunk in resp.aiter_bytes():
                        remaining = MAX_BACKEND_ERROR_BODY_BYTES - len(content)
                        if len(chunk) > remaining:
                            content.extend(chunk[:remaining])
                            truncated = True
                            break
                        content.extend(chunk)
                    if truncated:
                        logger.warning(
                            "Decoder error response body from %s exceeded %d "
                            "bytes and was truncated.",
                            req.endpoint,
                            MAX_BACKEND_ERROR_BODY_BYTES,
                        )
                    error = BackendHTTPError(
                        "decoder", resp, content=bytes(content)
                    )
                finally:
                    await resp.aclose()
                raise error
            return resp
        except (httpx.RequestError, BackendHTTPError) as exc:
            logger.warning("Decoder stream attempt %d failed for %s: %s",
                           attempt, req.endpoint, exc)
            if attempt >= req.max_retries:
                logger.error(
                    "Decoder failed after %d attempts streaming to %s.",
                    attempt,
                    req.endpoint,
                )
                raise
        await asyncio.sleep(req.retry_interval)

    raise RuntimeError(f"Decoder exhausted retries requesting {req.endpoint}")


def _parse_node_address(raw_header: str, role: str) -> NodeAddress:
    """Parse and validate one required backend address header."""
    try:
        host, port_text = raw_header.strip().split(":")
        host = host.strip()
        port = int(port_text.strip())
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {role} address header: expected <host>:<port> "
                "format."
            ),
        ) from exc

    if not host or port < 1 or port > 65535:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {role} address header: host must be non-empty "
                "and port must be in range 1..65535."
            ),
        )
    return host, port


def _parse_node_overrides(request: Request) -> NodeOverrides:
    """Derive explicit backend overrides from P/D address headers."""
    prefill_header = request.headers.get(HEADER_PREFILL_POD_ADDRESS_PORT)
    decode_header = request.headers.get(HEADER_DECODE_POD_ADDRESS_PORT)

    if prefill_header and decode_header:
        return (
            _parse_node_address(prefill_header, "Prefill"),
            _parse_node_address(decode_header, "Decode"),
        )

    if decode_header:
        return None, _parse_node_address(decode_header, "Decode")

    # Preserve the existing behavior for an incomplete Prefill-only override.
    return None, None


async def _read_request_and_costs(
    request: Request,
) -> Tuple[dict, bool, str, float, float, str]:
    """Read body once and derive basic cost / metadata."""
    body = await request.json()
    raw_bytes = await request.body()
    req_len = len(raw_bytes)

    is_stream = body.get("stream", False)
    media_type = "text/event-stream" if is_stream else "application/json"

    request_length_cost = req_len / 4.0
    estimated_prefill_cost = request_length_cost * 0.0345 + 120.0745
    estimated_decode_cost = float(req_len)
    req_id = await router.generate_request_id()
    return body, is_stream, media_type, estimated_prefill_cost, estimated_decode_cost, req_id


def _choose_prefiller(
    prefill_override: Optional[NodeAddress],
    prefill_cost: float,
    req_id: str,
) -> str:
    """Pick a prefiller node, honoring explicit overrides when provided."""
    if prefill_override is not None:
        prefill_host, prefill_port = prefill_override
        prefiller_key = router.select_prefiller_by_address(
            prefill_host, prefill_port, prefill_cost
        )
        if prefiller_key is None:
            raise BackendEndpointUnavailableError(
                "prefiller",
                prefill_host,
                prefill_port,
            )
        logger.info("Using specified prefiller %s:%d for request %s",
                    prefill_host, prefill_port, req_id)
        return prefiller_key

    return router.select_prefiller(prefill_cost)


async def _invoke_prefiller_and_update_body(
    api_path: str,
    body: dict,
    prefiller_key: str,
    prefill_cost: float,
    req_id: str,
    traceparent: Optional[str] = None,
) -> None:
    """Call the selected prefiller to obtain KV transfer params and patch body."""
    prefiller = router.prefillers[prefiller_key]
    prefill_req = ServiceRequest(
        client=prefiller.client,
        prefiller_key=prefiller_key,
        endpoint=api_path,
        payload=body,
        request_id=req_id,
        max_retries=global_args.max_retries,
        retry_interval=global_args.retry_interval,
        traceparent=traceparent,
    )
    try:
        resp = await _forward_prefill_once(prefill_req)
    finally:
        router.release_prefiller_tokens(prefiller_key, prefill_cost)

    resp_json = resp.json()
    kv_params = resp_json.get("kv_transfer_params", {}) or {}
    if kv_params:
        body["kv_transfer_params"] = kv_params


def _choose_decoder(
    decode_override: Optional[NodeAddress],
    decode_cost: float,
    req_id: str,
) -> str:
    """Pick a decoder node, honoring explicit overrides when provided."""
    if decode_override is not None:
        decode_host, decode_port = decode_override
        decoder_key = router.select_decoder_by_address(
            decode_host, decode_port, decode_cost
        )
        if decoder_key is None:
            raise BackendEndpointUnavailableError(
                "decoder",
                decode_host,
                decode_port,
            )
        logger.info("Using specified decoder %s:%d for request %s",
                    decode_host, decode_port, req_id)
        return decoder_key

    return router.select_decoder(decode_cost)


async def _build_streaming_response(
    api_path: str,
    body: dict,
    req_id: str,
    decoder_key: str,
    decode_cost: float,
    is_stream: bool,
    media_type: str,
    traceparent: Optional[str] = None,
    prefill_lease: Optional[PrefillLease] = None,
) -> StreamingResponse:
    """Construct the StreamingResponse that forwards bytes from decoder to client."""
    decoder = router.decoders[decoder_key]
    if prefill_lease is None:
        logger.debug(
            "Routing request %s directly to %s (decode)",
            req_id,
            decoder.url,
        )
    else:
        prefiller = router.prefillers[prefill_lease.key]
        logger.debug(
            "Routing request %s via %s (prefill) -> "
            "%s (decode)",
            req_id,
            prefiller.url,
            decoder.url,
        )

    released_kv = prefill_lease is None

    stream_req = ServiceRequest(
        client=decoder.client,
        endpoint=api_path,
        payload=body,
        request_id=req_id,
        prefiller_key=(
            prefill_lease.key if prefill_lease is not None else None
        ),
        max_retries=global_args.max_retries,
        retry_interval=global_args.retry_interval,
        traceparent=traceparent,
    )
    try:
        decoder_response = await _open_decoder_response(stream_req)
    except Exception:
        if prefill_lease is not None:
            router.mark_prefill_aborted(prefill_lease.key, req_id)
            router.release_kv(prefill_lease.key, prefill_lease.cost)
        router.release_decoder_tokens(decoder_key, decode_cost)
        raise

    async def generate():
        nonlocal released_kv
        try:
            async for chunk in decoder_response.aiter_bytes():
                if (
                    prefill_lease is not None
                    and not released_kv
                    and chunk
                ):
                    router.release_kv(
                        prefill_lease.key, prefill_lease.cost
                    )
                    released_kv = True
                yield chunk
        except Exception as exc:
            if prefill_lease is not None:
                logger.error(
                    "Error streaming from decoder %s for request %s: %s. "
                    "The aborted request will be recorded for KV release.",
                    decoder.url,
                    req_id,
                    exc,
                )
                router.mark_prefill_aborted(prefill_lease.key, req_id)
            else:
                logger.error(
                    "Error streaming directly from decoder %s for request "
                    "%s: %s.",
                    decoder.url,
                    req_id,
                    exc,
                )
            if prefill_lease is not None and not released_kv:
                router.release_kv(prefill_lease.key, prefill_lease.cost)
                released_kv = True
        finally:
            if prefill_lease is not None and not released_kv:
                router.release_kv(prefill_lease.key, prefill_lease.cost)
                released_kv = True
            await decoder_response.aclose()
            router.release_decoder_tokens(decoder_key, decode_cost)

    headers: Dict[str, str] = {}
    if is_stream:
        headers["Cache-Control"] = "no-cache"
        headers["Connection"] = "keep-alive"
        headers["X-Accel-Buffering"] = "no"

    return StreamingResponse(generate(), media_type=media_type, headers=headers)


async def _proxy_decode_only_request(
    api_path: str,
    decode_override: NodeAddress,
    body: dict,
    req_id: str,
    decode_cost: float,
    is_stream: bool,
    media_type: str,
    traceparent: Optional[str],
) -> StreamingResponse:
    """Forward a D route directly to its explicitly selected Decode Pod."""
    decoder_key = _choose_decoder(decode_override, decode_cost, req_id)
    return await _build_streaming_response(
        api_path=api_path,
        body=body,
        req_id=req_id,
        decoder_key=decoder_key,
        decode_cost=decode_cost,
        is_stream=is_stream,
        media_type=media_type,
        traceparent=traceparent,
    )


async def _proxy_openai_request(api_path: str, request: Request):
    """Core handler that proxies OpenAI-style completion/chat requests."""
    try:
        prefill_override, decode_override = _parse_node_overrides(request)
        (
            body,
            is_stream,
            media_type,
            prefill_cost,
            decode_cost,
            req_id,
        ) = await _read_request_and_costs(request)

        # Extract traceparent from incoming request headers
        traceparent = request.headers.get("traceparent")

        if prefill_override is None and decode_override is not None:
            return await _proxy_decode_only_request(
                api_path=api_path,
                decode_override=decode_override,
                body=body,
                req_id=req_id,
                decode_cost=decode_cost,
                is_stream=is_stream,
                media_type=media_type,
                traceparent=traceparent,
            )

        prefiller_key = _choose_prefiller(
            prefill_override, prefill_cost, req_id
        )
        await _invoke_prefiller_and_update_body(
            api_path, body, prefiller_key, prefill_cost, req_id, traceparent
        )

        try:
            decoder_key = _choose_decoder(
                decode_override, decode_cost, req_id
            )
        except Exception:
            router.mark_prefill_aborted(prefiller_key, req_id)
            router.release_kv(prefiller_key, prefill_cost)
            raise

        return await _build_streaming_response(
            api_path=api_path,
            body=body,
            req_id=req_id,
            decoder_key=decoder_key,
            decode_cost=decode_cost,
            is_stream=is_stream,
            media_type=media_type,
            traceparent=traceparent,
            prefill_lease=PrefillLease(
                key=prefiller_key, cost=prefill_cost
            ),
        )
    except HTTPException:
        raise
    except BackendEndpointUnavailableError as exc:
        logger.warning(
            "Backend endpoint currently unavailable for request %s: %s",
            req_id,
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={
                "Retry-After": str(global_args.discovery_interval),
            },
        ) from exc

    except BackendHTTPError as exc:
        logger.warning(
            "Returning %s response from %s for request %s",
            exc.status_code,
            exc.backend,
            api_path,
        )
        return Response(
            content=exc.content,
            status_code=exc.status_code,
            headers=exc.headers,
        )
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        logger.error("Error in proxy completion handler for %s: %s", api_path, exc)
        logger.error("".join(traceback.format_exception(*sys.exc_info())))
        raise


def _escape_prometheus_label_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _prometheus_sample_name_end(line: str) -> Optional[int]:
    if not line or line.startswith("#"):
        return None
    label_start = line.find("{")
    first_space = line.find(" ")
    if label_start == -1 or (first_space != -1 and first_space < label_start):
        return first_space if first_space != -1 else None

    in_quotes = False
    escaped = False
    for idx in range(label_start + 1, len(line)):
        char = line[idx]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            continue
        if char == "}" and not in_quotes:
            return idx + 1
    return None


def _append_prometheus_labels(line: str, labels: Dict[str, str]) -> str:
    name_end = _prometheus_sample_name_end(line)
    if name_end is None:
        return line

    metric = line[:name_end]
    suffix = line[name_end:]
    extra_labels = ",".join(
        f'{name}="{_escape_prometheus_label_value(value)}"'
        for name, value in labels.items()
    )
    if metric.endswith("}"):
        return f"{metric[:-1]},{extra_labels}}}{suffix}"
    return f"{metric}{{{extra_labels}}}{suffix}"


def _merge_prometheus_metrics(metrics_by_node: List[Tuple[str, str, object]]) -> str:
    output_lines: List[str] = []
    emitted_metadata = set()

    for key, text, node in metrics_by_node:
        labels = {
            "pod": getattr(node, "pod_name", key),
            "group_id": getattr(node, "group_id", ""),
            "role": getattr(node, "role", ""),
        }
        for line in text.splitlines():
            if line.startswith("# HELP ") or line.startswith("# TYPE "):
                parts = line.split(maxsplit=3)
                metadata_key = tuple(parts[:3])
                if metadata_key in emitted_metadata:
                    continue
                emitted_metadata.add(metadata_key)
                output_lines.append(line)
                continue
            output_lines.append(_append_prometheus_labels(line, labels))

    return "\n".join(output_lines) + "\n"


async def _fetch_metrics(key: str, node, headers: Dict[str, str],
                         max_retries: int, retry_interval: float) -> Tuple[str, str, object]:
    """Fetch /metrics from a backend node with the same retry policy as other calls."""
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            resp = await node.client.get("/metrics", headers=headers)
            resp.raise_for_status()
            return key, resp.text, node
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning("Metrics attempt %s failed for node %s: %s",
                           attempt, key, exc)
            if attempt >= max_retries:
                logger.error("Metrics exhausted retries (%s) on node %s",
                             max_retries, key)
                raise
            await asyncio.sleep(retry_interval)

    raise RuntimeError(f"Metrics exhausted retries ({max_retries}) on node {key}")


async def _aggregate_metrics(request: Request, max_retries: int,
                             retry_interval: float) -> Response:
    """Fetch and combine /metrics from all prefiller and decoder nodes."""
    if router is None:
        raise HTTPException(status_code=503, detail="Proxy is initializing")

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("traceparent", None)  # Remove tracing header for metrics scraping
    nodes = list(router.prefillers.items()) + list(router.decoders.items())
    fetch_results = await asyncio.gather(
        *[
            _fetch_metrics(key, node, headers, max_retries, retry_interval)
            for key, node in nodes
        ],
        return_exceptions=True,
    )
    results = []
    failed_endpoints = []
    for (key, _), result in zip(nodes, fetch_results):
        if isinstance(result, Exception):
            failed_endpoints.append((key, result))
            continue
        results.append(result)

    if failed_endpoints:
        logger.warning(
            "Failed to collect /metrics from %d/%d backend endpoints: %s",
            len(failed_endpoints),
            len(nodes),
            ", ".join(f"{key} ({type(exc).__name__}: {exc})"
                      for key, exc in failed_endpoints),
        )

    return Response(
        content=_merge_prometheus_metrics(results),
        media_type="text/plain; version=0.0.4",
    )


# --- FastAPI routes -----------------------------------------------------------------------


@app.post("/v1/completions")
@with_cancellation
async def completions(request: Request):
    return await _proxy_openai_request("/v1/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def chat_completions(request: Request):
    return await _proxy_openai_request("/v1/chat/completions", request)


@app.get("/listEndPoints")
async def listEndPoints():
    """Return current prefiller/decoder endpoints and their approximate load."""
    if not router:
        return {
            "status": "initializing",
            "prefill_nodes": [],
            "decode_nodes": [],
        }

    prefill_nodes = [
        {
            "endpoint": server.url,
            "host": server.host,
            "port": server.port,
            "active_tokens": server.active_tokens,
        }
        for server in router.prefillers.values()
    ]

    decode_nodes = [
        {
            "endpoint": server.url,
            "host": server.host,
            "port": server.port,
            "active_tokens": server.active_tokens,
        }
        for server in router.decoders.values()
    ]

    return {"status": "ok", "prefill_nodes": prefill_nodes, "decode_nodes": decode_nodes}


@app.get("/health")
async def health():
    if not router:
        return PlainTextResponse(
            "0/0 prefill ready, 0/0 decode ready", status_code=500
        )

    prefill_counts, decode_counts, not_ready_pods = router.pod_health_status()
    prefill_ready, prefill_total = prefill_counts
    decode_ready, decode_total = decode_counts

    body_lines = [
        f"{prefill_ready}/{prefill_total} prefill ready, "
        f"{decode_ready}/{decode_total} decode ready"
    ]
    body_lines.extend(f"{name}: {status}" for name, status in not_ready_pods)
    status_code = 200 if prefill_ready > 0 and decode_ready > 0 else 500
    return PlainTextResponse("\n".join(body_lines), status_code=status_code)


@app.get("/metrics")
async def metrics(request: Request):
    """Aggregate Prometheus /metrics from all prefiller and decoder nodes."""
    try:
        logger.debug("Aggregating metrics from prefiller and decoder nodes")
        return await _aggregate_metrics(
            request,
            max_retries=global_args.max_retries,
            retry_interval=global_args.retry_interval,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error in metrics proxy: %s", exc)
        raise


# --- CLI argument parsing -----------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--disable-access-log",
        action="store_true",
        help="Disable all HTTP access logs.",
    )
    parser.add_argument(
        "--disable-access-log-for-endpoints",
        type=str,
        default="",
        help=(
            "Comma-separated endpoint paths whose successful HTTP access logs "
            "are disabled. Responses with status 400 or greater are still logged."
        ),
    )

    # These arguments are kept for compatibility with non‑K8s mode if needed,
    # but in K8s deployments the service discovery will be the main source.
    parser.add_argument("--prefiller-hosts",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        type=int,
                        nargs="+",
                        default=[8001])
    parser.add_argument("--decoder-hosts",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports",
                        type=int,
                        nargs="+",
                        default=[8002])

    parser.add_argument(
        "--prefiller-labels",
        type=str,
        default="app=prefill",
        help="K8s label selector for prefiller pods "
        "(comma-separated, AND semantics: a=b,c=d).",
    )
    parser.add_argument(
        "--decoder-labels",
        type=str,
        default="app=decode",
        help="K8s label selector for decoder pods "
        "(comma-separated, AND semantics: a=b,c=d).",
    )
    parser.add_argument("--prefiller-container",
                        type=str,
                        default="prefill-engine")
    parser.add_argument("--decoder-container",
                        type=str,
                        default="decode-engine")
    parser.add_argument("--prefiller-port-name",
                        type=str,
                        default="prefill-port")
    parser.add_argument("--decoder-port-name",
                        type=str,
                        default="decode-port")
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument("--discovery-interval",
                        type=int,
                        default=10,
                        help="K8s service discovery interval (seconds).")

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts when calling backend.",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=0.001,
        help="Fixed interval (seconds) to wait between retry attempts.",
    )
    return parser


def parse_args():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


if __name__ == "__main__":
    import uvicorn

    global_args = parse_args()
    _configure_access_log_filter(
        global_args.disable_access_log_for_endpoints
    )
    uvicorn.run(
        app,
        host=global_args.host,
        port=global_args.port,
        access_log=not global_args.disable_access_log,
    )
