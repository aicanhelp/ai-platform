#!/bin/sh
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################

set -eu

ARCH=${HOST_ARCH:-$(uname -m)}
case "$ARCH" in
  x86_64|amd64)   SRC=/artifacts/bkeagent_linux_amd64 ;;
  aarch64|arm64)  SRC=/artifacts/bkeagent_linux_arm64 ;;
  *) echo "[deployer] unsupported arch: $ARCH"; exit 1 ;;
esac

HOST_BIN_DIR=${HOST_BIN_DIR:-/etc/openFuyao/bkeagent/bin}
DEST="${HOST_BIN_DIR}/bkeagent"
HEALTH_PORT=${HEALTH_PORT:-8443}
RESPONSE_FILE="/tmp/health_response.txt"
# Use existing TLS server certificate and key from /etc/kubernetes directory
CERT_FILE="/etc/kubernetes/tls-server.crt"
KEY_FILE="/etc/kubernetes/tls-server.key"

mkdir -p "${HOST_BIN_DIR}"

if [ ! -f "${DEST}" ] || ! cmp -s "${SRC}" "${DEST}"; then
  cp -f "${SRC}" "${DEST}"
  chmod +x "${DEST}"
  echo "[deployer] installed ${SRC} -> ${DEST}"
else
  echo "[deployer] host already has same binary, skip"
fi

"${DEST}" -v || true

# Check if bkeagent binary exists and is executable
check_binary() {
  if [ -f "${DEST}" ] && [ -x "${DEST}" ]; then
    return 0
  else
    return 1
  fi
}

# Check if TLS server certificate and key exist
check_tls_certificate() {
  if [ ! -f "${CERT_FILE}" ]; then
    echo "[deployer] ERROR: TLS server certificate not found at ${CERT_FILE}"
    return 1
  fi
  
  if [ ! -f "${KEY_FILE}" ]; then
    echo "[deployer] ERROR: TLS server key not found at ${KEY_FILE}"
    return 1
  fi
  
  # Verify certificate and key match
  CERT_MODULUS=$(openssl x509 -noout -modulus -in "${CERT_FILE}" 2>/dev/null | md5sum 2>/dev/null | cut -d' ' -f1 || openssl x509 -noout -modulus -in "${CERT_FILE}" 2>/dev/null | md5 2>/dev/null | cut -d' ' -f1 || echo "")
  KEY_MODULUS=$(openssl rsa -noout -modulus -in "${KEY_FILE}" 2>/dev/null | md5sum 2>/dev/null | cut -d' ' -f1 || openssl rsa -noout -modulus -in "${KEY_FILE}" 2>/dev/null | md5 2>/dev/null | cut -d' ' -f1 || echo "")
  
  if [ -n "${CERT_MODULUS}" ] && [ -n "${KEY_MODULUS}" ] && [ "${CERT_MODULUS}" = "${KEY_MODULUS}" ]; then
    echo "[deployer] TLS server certificate and key match"
    echo "[deployer] Using existing TLS server certificate: ${CERT_FILE}"
    echo "[deployer] Using existing TLS server key: ${KEY_FILE}"
    return 0
  else
    echo "[deployer] WARNING: TLS server certificate and key may not match"
    if [ -n "${CERT_MODULUS}" ] && [ -n "${KEY_MODULUS}" ]; then
      echo "[deployer] Cert modulus: ${CERT_MODULUS}"
      echo "[deployer] Key modulus: ${KEY_MODULUS}"
    fi
    return 1
  fi
}

# Health check HTTP response handler
# $1: request path (optional, defaults to /healthz)
health_response() {
  local request_path="${1:-/healthz}"
  
  # Only allow /healthz and /readyz paths
  if [ "${request_path}" != "/healthz" ] && [ "${request_path}" != "/readyz" ]; then
    printf "HTTP/1.1 404 Not Found\r\n"
    printf "Content-Type: text/plain\r\n"
    printf "Content-Length: 13\r\n"
    printf "\r\n"
    printf "404 Not Found"
    return
  fi
  
  # Return health status for /healthz and /readyz
  if check_binary; then
    printf "HTTP/1.1 200 OK\r\n"
    printf "Content-Type: text/plain\r\n"
    printf "Content-Length: 2\r\n"
    printf "\r\n"
    printf "ok"
  else
    printf "HTTP/1.1 503 Service Unavailable\r\n"
    printf "Content-Type: text/plain\r\n"
    printf "Content-Length: 18\r\n"
    printf "\r\n"
    printf "binary not ready"
  fi
}

# HTTP request handler - parse request and return appropriate response
handle_http_request() {
  local request_line=""
  local request_path="/healthz"
  
  # Read first line (request line)
  read -r request_line 2>/dev/null || return 1
  
  # Parse request path from "GET /path HTTP/1.1"
  if echo "${request_line}" | grep -qE "^(GET|POST|HEAD|PUT|DELETE|OPTIONS|PATCH) "; then
    request_path=$(echo "${request_line}" | awk '{print $2}' | cut -d'?' -f1)
  fi
  
  # Read remaining headers until empty line
  while read -r line; do
    if [ -z "${line}" ] || [ "${line}" = "$(printf '\r')" ]; then
      break
    fi
  done
  
  # Return appropriate response based on path
  health_response "${request_path}"
}

# Update response file periodically (for /healthz path)
update_response_file() {
  while true; do
    health_response "/healthz" > "${RESPONSE_FILE}"
    sleep 1
  done
}

# Start HTTPS health check server using openssl s_server
if command -v openssl >/dev/null 2>&1; then
  # Check if TLS server certificate and key exist
  if ! check_tls_certificate; then
    echo "[deployer] ERROR: TLS server certificate or key not found or invalid, exiting"
    echo "[deployer] Expected certificate: ${CERT_FILE}"
    echo "[deployer] Expected key: ${KEY_FILE}"
    exit 1
  fi
  
  echo "[deployer] starting HTTPS health check server on port ${HEALTH_PORT}"
  echo "[deployer] NODE_IP=${NODE_IP:-not set}"
  
  # Initialize response file
  health_response "/healthz" > "${RESPONSE_FILE}"
  
  # Start background process to update response file
  update_response_file &
  UPDATE_PID=$!
  
  # Create a wrapper script to handle HTTP requests
  HTTP_HANDLER_SCRIPT="/tmp/http_handler.sh"
  cat > "${HTTP_HANDLER_SCRIPT}" <<'EOFSCRIPT'
#!/bin/sh
# HTTP request handler script
request_line=""
request_path="/healthz"

# Read first line (request line)
read -r request_line 2>/dev/null || exit 1

# Parse request path from "GET /path HTTP/1.1"
if echo "${request_line}" | grep -qE "^(GET|POST|HEAD|PUT|DELETE|OPTIONS|PATCH) "; then
  request_path=$(echo "${request_line}" | awk '{print $2}' | cut -d'?' -f1)
fi

# Read remaining headers until empty line
while read -r line; do
  if [ -z "${line}" ] || [ "${line}" = "$(printf '\r')" ]; then
    break
  fi
done

# Check if binary exists
DEST="${HOST_BIN_DIR:-/etc/openFuyao/bkeagent/bin}/bkeagent"
if [ ! -f "${DEST}" ] || [ ! -x "${DEST}" ]; then
  BINARY_READY=0
else
  BINARY_READY=1
fi

# Return appropriate response based on path
# Allow both /healthz and /readyz paths
if [ "${request_path}" != "/healthz" ] && [ "${request_path}" != "/readyz" ]; then
  printf "HTTP/1.1 404 Not Found\r\n"
  printf "Content-Type: text/plain\r\n"
  printf "Content-Length: 13\r\n"
  printf "\r\n"
  printf "404 Not Found"
elif [ ${BINARY_READY} -eq 1 ]; then
  printf "HTTP/1.1 200 OK\r\n"
  printf "Content-Type: text/plain\r\n"
  printf "Content-Length: 2\r\n"
  printf "\r\n"
  printf "ok"
else
  printf "HTTP/1.1 503 Service Unavailable\r\n"
  printf "Content-Type: text/plain\r\n"
  printf "Content-Length: 18\r\n"
  printf "\r\n"
  printf "binary not ready"
fi
EOFSCRIPT
  chmod +x "${HTTP_HANDLER_SCRIPT}"
  
  # Export HOST_BIN_DIR for the handler script
  export HOST_BIN_DIR
  
  # Start HTTPS server using socat to handle HTTP requests properly
  # socat can parse HTTP requests and execute a script to handle them
  (
    while true; do
      # Use socat to handle HTTPS connections
      # - SSL certificate and key are provided via environment or file
      # - For each connection, execute the handler script to parse request and return response
      socat \
        "OPENSSL-LISTEN:${HEALTH_PORT},reuseaddr,fork,cert=${CERT_FILE},key=${KEY_FILE},verify=0" \
        "EXEC:${HTTP_HANDLER_SCRIPT}" \
        2>&1 | grep -v "^ACCEPT" || true
      # Small delay before restarting to prevent port binding conflicts
      sleep 0.1
    done
  ) &
  
  HEALTH_SERVER_PID=$!
  echo "[deployer] HTTPS health check server started (PID: ${HEALTH_SERVER_PID}, update PID: ${UPDATE_PID})"
  echo "[deployer] listening on 0.0.0.0:${HEALTH_PORT}"
  
  # Give server a moment to bind the port
  # openssl s_server should bind immediately when started
  sleep 3
  
  # Wait for server to start listening
  MAX_WAIT=10
  WAIT_COUNT=0
  PORT_LISTENING=false
  
  while [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
    # Check if process is still running
    if ! kill -0 "${HEALTH_SERVER_PID}" 2>/dev/null; then
      echo "[deployer] ERROR: HTTPS health check server process died"
      kill "${UPDATE_PID}" 2>/dev/null || true
      echo "[deployer] Checking for openssl processes..."
      ps aux | grep openssl | grep -v grep || true
      echo "[deployer] Server will continue running despite process check failure"
      # Don't exit, let the server restart naturally
      break
    fi
    
    # Check if port is listening
    if netstat -tln 2>/dev/null | grep -q ":${HEALTH_PORT} " || \
       ss -tln 2>/dev/null | grep -q ":${HEALTH_PORT} "; then
      PORT_LISTENING=true
      break
    fi
    
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
  done
  
  if [ "${PORT_LISTENING}" = "true" ]; then
    echo "[deployer] Port ${HEALTH_PORT} is listening successfully"
  else
    echo "[deployer] WARNING: Port ${HEALTH_PORT} check timeout, but server process is running"
    echo "[deployer] Server PID: ${HEALTH_SERVER_PID}"
    echo "[deployer] Checking processes..."
    ps aux | grep -E "openssl|${HEALTH_SERVER_PID}" | grep -v grep || true
    echo "[deployer] Checking port status..."
    netstat -tln 2>/dev/null | grep "${HEALTH_PORT}" || ss -tln 2>/dev/null | grep "${HEALTH_PORT}" || true
    echo "[deployer] Note: openssl s_server may not show in netstat until first connection"
    echo "[deployer] Server will be ready when first connection arrives"
  fi
else
  echo "[deployer] ERROR: openssl not found, cannot start HTTPS health check server"
  echo "[deployer] Please install openssl package"
  exit 1
fi

tail -f /dev/null
