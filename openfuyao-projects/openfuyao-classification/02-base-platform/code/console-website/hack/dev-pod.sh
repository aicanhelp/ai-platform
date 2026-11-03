#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NS=openfuyao-system
DEV_POD=console-website-dev
DEV_CONTAINER=console-website
ANNOTATION_MANAGED_NODEPORT="console-website.dev-pod/opened-nodeport"

usage() {
  echo "Usage: $0 {create|delete|activate|deactivate|install|start|build|preview|kill}"
  echo "  create     - Create the console-website-dev Pod and install dependencies"
  echo "  delete     - Delete the console-website-dev Pod"
  echo "  activate   - Apply Service configuration and enable NodePort 30001."
  echo "  deactivate - Restore the Service configuration"
  echo "  install    - Run npm install in the Pod for dependency updates"
  echo "  start      - Run npm run start in the Pod. Optional -d: run in the background"
  echo "  build      - Run npm run build in the Pod. Optional -d: run in the background"
  echo "  preview    - Run npm run preview in the Pod. Requires build output first. Optional -d: run in the background"
  echo "  kill       - Stop all Vite-related processes"
}

require_kubectl() {
  command -v kubectl >/dev/null || {
    echo "Error: kubectl not found" >&2
    exit 1
  }
}

write_dev_pod_manifest() {
  local node="$1"
  awk -v r="$REPO_ROOT" -v n="$node" '
    /^    - name: code$/ { is_code=1; print; next }
    /^    - name:/ { is_code=0; print; next }
    is_code && /^        path:/ { print "        path: " r; is_code=0; next }
    /^  nodeName:/ { print "  nodeName: " n; next }
    { print }
  ' "${SCRIPT_DIR}/console-website-dev.yaml"
}

resolve_dev_pod_node() {
  if [[ -n "${DEV_POD_NODE:-}" ]]; then
    if ! kubectl get node "$DEV_POD_NODE" &>/dev/null; then
      echo "Error: DEV_POD_NODE=$DEV_POD_NODE does not exist in the cluster" >&2
      exit 1
    fi
    echo "$DEV_POD_NODE"
    return
  fi
  local name
  for name in "$(hostname -f 2>/dev/null || true)" "$(hostname)" "$(hostname -s 2>/dev/null || true)"; do
    [[ -z "$name" ]] && continue
    if kubectl get node "$name" &>/dev/null; then
      echo "$name"
      return
    fi
  done
  echo "Error: local hostname does not match any cluster node name. Set DEV_POD_NODE to a node name from kubectl get nodes" >&2
  exit 1
}

cmd_create() {
  require_kubectl
  write_dev_pod_manifest "$(resolve_dev_pod_node)" | kubectl apply -f -

  kubectl wait --for=condition=Ready pod -n "$NS" "$DEV_POD" --timeout=120s

  kubectl exec -n "$NS" "$DEV_POD" -c "$DEV_CONTAINER" -- bash -lc 'cd /app && npm install'
}

cmd_install() {
  require_kubectl
  ensure_dev_pod_exists

  kubectl exec -it -n "$NS" "$DEV_POD" -c "$DEV_CONTAINER" -- bash -lc 'cd /app && npm install'
}

cmd_delete() {
  require_kubectl
  kubectl delete pod -n "$NS" "$DEV_POD"
}

get_svc_nodeport() {
  kubectl get svc console-service -n "$NS" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || true
}

managed_nodeport_annotation() {
  kubectl get svc console-service -n "$NS" -o go-template="{{index .metadata.annotations \"${ANNOTATION_MANAGED_NODEPORT}\"}}" 2>/dev/null || true
}

cmd_activate() {
  require_kubectl
  kubectl patch svc console-website -n "$NS" --type merge \
    -p '{"spec":{"selector":{"app":"console-website-dev"}}}'

  local np
  np="$(get_svc_nodeport)"
  if [[ -z "$np" ]]; then
    kubectl patch svc console-service -n "$NS" --type=json \
      -p='[{"op":"replace","path":"/spec/type","value":"NodePort"},{"op":"add","path":"/spec/ports/0/nodePort","value":30001}]' \
      2>/dev/null \
      || kubectl patch svc console-service -n "$NS" --type=json \
        -p='[{"op":"replace","path":"/spec/type","value":"NodePort"},{"op":"replace","path":"/spec/ports/0/nodePort","value":30001}]'
    kubectl annotate svc console-service -n "$NS" "${ANNOTATION_MANAGED_NODEPORT}=true" --overwrite
  fi

  echo ""
  echo "Run dev-pod.sh start to launch the dev server"
}

cmd_deactivate() {
  require_kubectl
  kubectl patch svc console-website -n "$NS" --type merge \
    -p '{"spec":{"selector":{"app":"console-website"}}}'

  if [[ "$(managed_nodeport_annotation)" == "true" ]]; then
    kubectl patch svc console-service -n "$NS" --type=json \
      -p='[{"op":"remove","path":"/spec/ports/0/nodePort"}]' 2>/dev/null || true
    kubectl patch svc console-service -n "$NS" --type merge -p '{"spec":{"type":"ClusterIP"}}'
    kubectl annotate svc console-service -n "$NS" "${ANNOTATION_MANAGED_NODEPORT}-"
  fi
}

kill_vite_process() {
  pkill -f '[n]ode.*vite' 2>/dev/null || true
  pkill -f '[/]vite' 2>/dev/null || true
  pkill -f 'vite\.mjs' 2>/dev/null || true
  pkill -f 'bin/vite' 2>/dev/null || true
  pkill -f vite 2>/dev/null || true
  sleep 1

  echo "Cleared vite processes"
}

cmd_kill() {
  kill_vite_process
}

ensure_dev_pod_exists() {
  kubectl get pod -n "$NS" "$DEV_POD" &>/dev/null || {
    echo "Error: Pod $DEV_POD does not exist. Run first: $0 create" >&2
    exit 1
  }
}

parse_optional_detach() {
  DETACH=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -d) DETACH=true; shift ;;
      *)
        echo "Error: unknown option: $1 (only -d is supported)" >&2
        exit 1
        ;;
    esac
  done
}

cmd_start() {
  parse_optional_detach "$@"
  require_kubectl
  kill_vite_process
  ensure_dev_pod_exists

  if [[ "$DETACH" == true ]]; then
    kubectl exec -n "$NS" "$DEV_POD" -c "$DEV_CONTAINER" -- bash -lc 'cd /app && nohup npm run start &'
  else
    kubectl exec -it -n "$NS" "$DEV_POD" -c "$DEV_CONTAINER" -- bash -lc 'cd /app && exec npm run start'
  fi
}

cmd_build() {
  parse_optional_detach "$@"
  require_kubectl
  ensure_dev_pod_exists

  if [[ "$DETACH" == true ]]; then
    kubectl exec -n "$NS" "$DEV_POD" -c "$DEV_CONTAINER" -- bash -lc 'cd /app && nohup npm run build &'
  else
    kubectl exec -n "$NS" "$DEV_POD" -c "$DEV_CONTAINER" -- bash -lc 'cd /app && exec npm run build'
  fi
}

cmd_preview() {
  parse_optional_detach "$@"
  require_kubectl
  kill_vite_process
  ensure_dev_pod_exists
  if [[ "$DETACH" == true ]]; then
    kubectl exec -n "$NS" "$DEV_POD" -c "$DEV_CONTAINER" -- bash -lc 'cd /app && nohup npm run preview &'
  else
    kubectl exec -n "$NS" "$DEV_POD" -c "$DEV_CONTAINER" -- bash -lc 'cd /app && exec npm run preview'
  fi
}

case "${1:-}" in
  create) cmd_create ;;
  delete) cmd_delete ;;
  activate) cmd_activate ;;
  deactivate) cmd_deactivate ;;
  install) cmd_install ;;
  start) shift; cmd_start "$@" ;;
  build) shift; cmd_build "$@" ;;
  preview) shift; cmd_preview "$@" ;;
  kill) cmd_kill ;;
  *) usage >&2; exit 1 ;;
esac
