# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Minimal HTTP /healthz endpoint for Kubernetes HTTP probes.

The gRPC tokenizer cannot answer the kserve scheduler preset's HTTP probes
(httpGet /healthz on port 8082), so the sidecar exposes this shallow health
endpoint alongside the gRPC socket: 200 once the process serves HTTP, which is
sufficient because tokenizer loading is lazy per model.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HEALTH_PATH = "/healthz"
_HEALTH_BODY = b"ok"


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == _HEALTH_PATH:
            self.send_response(200)
            self.send_header("Content-Length", str(len(_HEALTH_BODY)))
            self.end_headers()
            self.wfile.write(_HEALTH_BODY)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Probe traffic would flood stdout every few seconds; stay silent.
        pass


def start_health_server(port: int) -> ThreadingHTTPServer:
    """Serve GET /healthz on a background daemon thread.

    Returns the running server; the caller is responsible for shutdown().
    """
    server = ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
