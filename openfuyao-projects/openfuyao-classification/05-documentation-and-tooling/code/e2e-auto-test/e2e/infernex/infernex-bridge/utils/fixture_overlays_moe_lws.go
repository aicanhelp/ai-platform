package utils

import "strings"

// MoEOfflineInferenceModel is the MoE model for FIT 56–59 (ag-03 / pd-05).
// Fixtures use hostPath cache when present; initContainer + main fallback download online on miss.
const MoEOfflineInferenceModel = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

const moeModelCacheDir = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct"

// moeCacheCompleteCheckShell returns 0 only when snapshot weights are fully present (not metadata-only partial cache).
const moeCacheCompleteCheckShell = `moe_resolve_snapshot_dir() {
  local cache_dir="$1"
  if [ -f "${cache_dir}/refs/main" ]; then
    local rev
    rev=$(tr -d '\n' < "${cache_dir}/refs/main")
    echo "${cache_dir}/snapshots/${rev}"
    return 0
  fi
  local snap
  snap=$(ls -d "${cache_dir}"/snapshots/*/ 2>/dev/null | head -1)
  echo "${snap%/}"
}

moe_cache_is_complete() {
  python3 - "$1" <<'PY'
import glob, json, os, sys

cache = sys.argv[1]
ref = os.path.join(cache, "refs", "main")
if os.path.isfile(ref):
    snap = os.path.join(cache, "snapshots", open(ref, encoding="utf-8").read().strip())
else:
    snaps = sorted(glob.glob(os.path.join(cache, "snapshots", "*")))
    snap = snaps[0].rstrip(os.sep) if snaps else ""
if not snap or not os.path.isdir(snap):
    sys.exit(1)

idx = os.path.join(snap, "model.safetensors.index.json")
if os.path.isfile(idx):
    with open(idx, encoding="utf-8") as f:
        data = json.load(f)
    files = set(data.get("weight_map", {}).values())
    if not files or any(not os.path.isfile(os.path.join(snap, name)) for name in files):
        sys.exit(1)
    sys.exit(0)

if os.path.isfile(os.path.join(snap, "model.safetensors")):
    sys.exit(0)

sys.exit(0 if glob.glob(os.path.join(snap, "*.safetensors")) else 1)
PY
}
`

const moeModelPathResolve = moeCacheCompleteCheckShell + `
export HF_HUB_OFFLINE=1
MODEL_CACHE_DIR="` + moeModelCacheDir + `"
MODEL_PATH=$(moe_resolve_snapshot_dir "${MODEL_CACHE_DIR}")
if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ] || ! moe_cache_is_complete "${MODEL_CACHE_DIR}"; then
  echo "MoE cache miss or incomplete under ${MODEL_CACHE_DIR}, falling back to online ` + MoEOfflineInferenceModel + `" >&2
  MODEL_PATH="` + MoEOfflineInferenceModel + `"
  export HF_HUB_OFFLINE=0
fi`

const moeHFInitDownloadScript = moeCacheCompleteCheckShell + `set -e
MODEL="` + MoEOfflineInferenceModel + `"
CACHE_DIR="` + moeModelCacheDir + `"
if moe_cache_is_complete "${CACHE_DIR}"; then
  echo "MoE model cache complete under ${CACHE_DIR}, skip download"
  exit 0
fi
echo "MoE model cache miss or incomplete, downloading ${MODEL}"
export HF_HUB_OFFLINE=0
# Retry in-process: init container restart triggers LWS RecreateGroupOnPodRestart (whole group deleted).
until hf download "${MODEL}"; do
  echo "hf download failed, retrying in 60s..."
  sleep 60
done
`

// vLLM serve scripts synced from InferNex-Bridge ag-03 / pd-05 (leader zmq 5557; LWS overlay bumps worker to 5558).
const (
	moeVLLMScriptAggregate = `set -euo pipefail
export PHYSICAL_DEVICES=$(ls /dev/davinci* 2>/dev/null | grep -o '[0-9]\+' | sort -n | paste -sd',' - || true)
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
START_RANK=$(( ${LWS_WORKER_INDEX:-0} * 1 ))
` + moeModelPathResolve + `
exec vllm serve "${MODEL_PATH}" \
  --served-model-name ` + MoEOfflineInferenceModel + ` \
  --trust-remote-code \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256_cbor \
  --enable-prompt-tokens-details \
  --port 8000 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank ${START_RANK} \
  --data-parallel-address ${LWS_LEADER_ADDRESS} \
  --data-parallel-rpc-port 5434 \
  --data-parallel-hybrid-lb \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.95 \
  --block-size 128 \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher":"zmq", "endpoint":"tcp://*:5557", "topic":"kv-events"}' \
  --kv-transfer-config '{"engine_id":"$(POD_NAME)","kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_port":"20000","kv_connector_extra_config":{"backend":"mooncake","decode":{"dp_size":2,"tp_size":2},"prefill":{"dp_size":2,"tp_size":2},"lookup_rpc_port":"1"}}'
`

	moeVLLMScriptPDDecode = `set -euo pipefail
export PHYSICAL_DEVICES=$(ls /dev/davinci* 2>/dev/null | grep -o '[0-9]\+' | sort -n | paste -sd',' - || true)
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
START_RANK=$(( ${LWS_WORKER_INDEX:-0} * 1 ))
` + moeModelPathResolve + `
exec vllm serve "${MODEL_PATH}" \
  --served-model-name ` + MoEOfflineInferenceModel + ` \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --prefix-caching-hash-algo sha256_cbor \
  --enable-prompt-tokens-details \
  --port 8000 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank ${START_RANK} \
  --data-parallel-address ${LWS_LEADER_ADDRESS} \
  --data-parallel-rpc-port 12777 \
  --data-parallel-hybrid-lb \
  --enable-expert-parallel \
  --enforce-eager \
  --seed 1024 \
  --max-model-len 65536 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 \
  --block-size 128 \
  --kv-transfer-config '{"engine_id":"$(POD_NAME)","kv_connector":"MultiConnector","kv_connector_extra_config":{"connectors":[{"kv_buffer_device":"npu","kv_connector":"MooncakeConnectorV1","kv_connector_extra_config":{"decode":{"dp_size":2,"tp_size":2},"prefill":{"dp_size":2,"tp_size":2},"use_ascend_direct":true},"kv_parallel_size":1,"kv_port":"20001","kv_role":"kv_consumer"},{"kv_buffer_device":"npu","kv_connector":"AscendStoreConnector","kv_connector_extra_config":{"backend":"mooncake","decode":{"dp_size":2,"tp_size":2},"lookup_rpc_port":"0","prefill":{"dp_size":2,"tp_size":2}},"kv_parallel_size":1,"kv_port":"20001","kv_role":"kv_consumer"}],"decode":{"dp_size":2,"tp_size":2},"prefill":{"dp_size":2,"tp_size":2}},"kv_port":"20001","kv_rank":1,"kv_role":"kv_consumer"}'
`

	moeVLLMScriptPDPrefill = `set -euo pipefail
export PHYSICAL_DEVICES=$(ls /dev/davinci* 2>/dev/null | grep -o '[0-9]\+' | sort -n | paste -sd',' - || true)
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
START_RANK=$(( ${LWS_WORKER_INDEX:-0} * 1 ))
` + moeModelPathResolve + `
exec vllm serve "${MODEL_PATH}" \
  --served-model-name ` + MoEOfflineInferenceModel + ` \
  --trust-remote-code \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256_cbor \
  --enable-prompt-tokens-details \
  --port 8000 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank ${START_RANK} \
  --data-parallel-address ${LWS_LEADER_ADDRESS} \
  --data-parallel-rpc-port 12890 \
  --data-parallel-hybrid-lb \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.95 \
  --block-size 128 \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher":"zmq", "topic":"kv-events"}' \
  --kv-transfer-config '{"engine_id":"$(POD_NAME)","kv_connector":"MultiConnector","kv_connector_extra_config":{"connectors":[{"kv_buffer_device":"npu","kv_connector":"MooncakeConnectorV1","kv_connector_extra_config":{"decode":{"dp_size":2,"tp_size":2},"prefill":{"dp_size":2,"tp_size":2},"use_ascend_direct":true},"kv_parallel_size":1,"kv_port":"20000","kv_role":"kv_producer"},{"kv_buffer_device":"npu","kv_connector":"AscendStoreConnector","kv_connector_extra_config":{"backend":"mooncake","decode":{"dp_size":2,"tp_size":2},"lookup_rpc_port":"0","prefill":{"dp_size":2,"tp_size":2}},"kv_parallel_size":1,"kv_port":"20000","kv_role":"kv_producer"}],"decode":{"dp_size":2,"tp_size":2},"prefill":{"dp_size":2,"tp_size":2}},"kv_port":"20000","kv_rank":0,"kv_role":"kv_producer"}'
`
)

func removeInitContainerByName(podSpec map[string]interface{}, name string) {
	inits, ok := podSpec["initContainers"].([]interface{})
	if !ok {
		return
	}
	out := make([]interface{}, 0, len(inits))
	for _, ic := range inits {
		m, ok := ic.(map[string]interface{})
		if !ok {
			out = append(out, ic)
			continue
		}
		n, _ := m["name"].(string)
		if n == name {
			continue
		}
		out = append(out, ic)
	}
	if len(out) == 0 {
		delete(podSpec, "initContainers")
	} else {
		podSpec["initContainers"] = out
	}
}

func upsertContainerEnv(container map[string]interface{}, name, value string) {
	env, _ := container["env"].([]interface{})
	for _, e := range env {
		em, ok := e.(map[string]interface{})
		if !ok {
			continue
		}
		if n, _ := em["name"].(string); n == name {
			em["value"] = value
			return
		}
	}
	container["env"] = append(env, map[string]interface{}{"name": name, "value": value})
}

func patchMoEEngineProbes(container map[string]interface{}, modelsPort, healthPort string) {
	if modelsPort == "" {
		modelsPort = "aggregated-port"
	}
	if healthPort == "" {
		healthPort = modelsPort
	}
	container["startupProbe"] = map[string]interface{}{
		"httpGet": map[string]interface{}{
			"path": "/v1/models",
			"port": modelsPort,
		},
		// MoE 30B + cross-node DP: allow up to ~3h before kubelet restarts main (each restart triggers LWS group recreate).
		"initialDelaySeconds": 300,
		"periodSeconds":       30,
		"timeoutSeconds":      10,
		"failureThreshold":    360,
	}
	// Omit liveness during MoE bring-up: a flaky /health during long load causes restart → LWS deletes the group.
	delete(container, "livenessProbe")
	container["readinessProbe"] = map[string]interface{}{
		"httpGet": map[string]interface{}{
			"path": "/v1/models",
			"port": modelsPort,
		},
		"initialDelaySeconds": 300,
		"periodSeconds":       15,
		"timeoutSeconds":      10,
		"failureThreshold":    360,
	}
}

func mainContainerProbePorts(container map[string]interface{}) (modelsPort, healthPort string) {
	modelsPort = "aggregated-port"
	healthPort = "aggregated-port"
	ports, _ := container["ports"].([]interface{})
	for _, p := range ports {
		pm, ok := p.(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := pm["name"].(string)
		switch name {
		case "decode-port", "prefill-port", "aggregated-port":
			modelsPort = name
			healthPort = name
		}
	}
	return modelsPort, healthPort
}

func patchMoEMainContainer(container map[string]interface{}, script string) {
	container["command"] = []interface{}{"/bin/bash", "-c"}
	container["args"] = []interface{}{script}
	modelsPort, healthPort := mainContainerProbePorts(container)
	patchMoEEngineProbes(container, modelsPort, healthPort)
	// HF_HUB_OFFLINE is set by the serve script: 1 when local cache is complete, 0 on online fallback.
	upsertContainerEnv(container, "HCCL_COMM_ID", "127.0.0.1:5555")
	upsertContainerEnv(container, "VLLM_CACHE_ROOT", "/vllm-workspace")
	upsertContainerEnv(container, "HCCL_OP_EXPANSION_MODE", "AIV")
	upsertContainerEnv(container, "OMP_PROC_BIND", "false")
	upsertContainerEnv(container, "OMP_NUM_THREADS", "10")
	upsertContainerEnv(container, "ASCEND_TRANSPORT_PRINT", "1")
	upsertContainerEnv(container, "ACL_OP_INIT_MODE", "1")
	upsertContainerEnv(container, "VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "300000")
	upsertContainerEnv(container, "HCCL_INTRA_ROCE_ENABLE", "1")
	upsertContainerEnv(container, "VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

	limits := map[string]interface{}{
		"cpu":                  "8",
		"memory":               "128Gi",
		"huawei.com/Ascend910": "2",
	}
	requests := map[string]interface{}{
		"cpu":                  "4",
		"memory":               "64Gi",
		"huawei.com/Ascend910": "2",
	}
	container["resources"] = map[string]interface{}{
		"limits":   limits,
		"requests": requests,
	}

	if env, ok := container["env"].([]interface{}); ok {
		for _, e := range env {
			em, ok := e.(map[string]interface{})
			if !ok {
				continue
			}
			if n, _ := em["name"].(string); n == "HCCL_IF_IP" {
				return
			}
		}
	}
	container["env"] = append(container["env"].([]interface{}), map[string]interface{}{
		"name": "HCCL_IF_IP",
		"valueFrom": map[string]interface{}{
			"fieldRef": map[string]interface{}{"fieldPath": "status.podIP"},
		},
	})
}

func patchHuggingFaceDownloadInitForMoE(podSpec map[string]interface{}) {
	inits, ok := podSpec["initContainers"].([]interface{})
	if !ok {
		return
	}
	for _, ic := range inits {
		m, ok := ic.(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := m["name"].(string)
		if name != "huggingface-download" {
			continue
		}
		m["command"] = []interface{}{"/bin/sh", "-c"}
		m["args"] = []interface{}{moeHFInitDownloadScript}
		upsertInitContainerEnv(m, "HF_HUB_OFFLINE", "0")
		return
	}
}

func upsertInitContainerEnv(container map[string]interface{}, name, value string) {
	env, _ := container["env"].([]interface{})
	for _, e := range env {
		em, ok := e.(map[string]interface{})
		if !ok {
			continue
		}
		if n, _ := em["name"].(string); n == name {
			em["value"] = value
			return
		}
	}
	container["env"] = append(env, map[string]interface{}{"name": name, "value": value})
}

func patchPodSpecMoEOffline(podSpec map[string]interface{}, script string) {
	patchHuggingFaceDownloadInitForMoE(podSpec)
	podSpec["terminationGracePeriodSeconds"] = 600
	containers, ok := podSpec["containers"].([]interface{})
	if !ok {
		return
	}
	for _, c := range containers {
		cm, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := cm["name"].(string)
		if name != "" && name != "main" {
			continue
		}
		patchMoEMainContainer(cm, script)
		return
	}
}

func patchInsvcEngineMoE(engine map[string]interface{}, aggregateScript, decodeScript, prefillScript string) {
	if decodeScript == "" && prefillScript == "" {
		tpl := mustMap(engine["template"])
		podSpec := mustMap(tpl["spec"])
		patchPodSpecMoEOffline(podSpec, aggregateScript)
		tpl["spec"] = podSpec
		engine["template"] = tpl
		return
	}
	tpl := mustMap(engine["template"])
	decodeSpec := mustMap(tpl["spec"])
	patchPodSpecMoEOffline(decodeSpec, decodeScript)
	tpl["spec"] = decodeSpec
	engine["template"] = tpl

	if prefill, ok := engine["prefill"].(map[string]interface{}); ok {
		preTpl := mustMap(prefill["template"])
		preSpec := mustMap(preTpl["spec"])
		patchPodSpecMoEOffline(preSpec, prefillScript)
		preTpl["spec"] = preSpec
		prefill["template"] = preTpl
		engine["prefill"] = prefill
	}
}

func patchLLMISvcConfigMoE(cfgSpec map[string]interface{}, aggregateScript, decodeScript, prefillScript string) {
	if decodeScript == "" && prefillScript == "" {
		tpl := mustMap(cfgSpec["template"])
		patchPodSpecMoEOffline(tpl, aggregateScript)
		cfgSpec["template"] = tpl
		return
	}
	tpl := mustMap(cfgSpec["template"])
	patchPodSpecMoEOffline(tpl, decodeScript)
	cfgSpec["template"] = tpl

	if prefill, ok := cfgSpec["prefill"].(map[string]interface{}); ok {
		preTpl := mustMap(prefill["template"])
		patchPodSpecMoEOffline(preTpl, prefillScript)
		prefill["template"] = preTpl
		cfgSpec["prefill"] = prefill
	}
}

func patchInferNexServiceModelAndRouter(insvc map[string]interface{}) {
	spec := mustMap(insvc["spec"])
	spec["model"] = map[string]interface{}{
		"uri":  "hf://" + MoEOfflineInferenceModel,
		"name": MoEOfflineInferenceModel,
	}
	patchRouterTokenizerModelInSpec(spec)
	insvc["spec"] = spec
}

func patchLLMInferenceServiceModel(llm map[string]interface{}) {
	spec := mustMap(llm["spec"])
	spec["model"] = map[string]interface{}{
		"uri":  "hf://" + MoEOfflineInferenceModel,
		"name": MoEOfflineInferenceModel,
	}
	patchRouterTokenizerModelInSpec(spec)
	llm["spec"] = spec
}

func patchRouterTokenizerModelInSpec(spec map[string]interface{}) {
	const denseModel = "Qwen/Qwen2.5-0.5B"
	igr, ok := spec["intelligentGatewayRouting"].(map[string]interface{})
	if !ok {
		return
	}
	router, ok := igr["router"].(map[string]interface{})
	if !ok {
		return
	}
	tpl, ok := router["template"].(map[string]interface{})
	if !ok {
		return
	}
	podSpec := mustMap(tpl["spec"])
	containers, ok := podSpec["containers"].([]interface{})
	if !ok {
		return
	}
	for _, c := range containers {
		cm, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		if name, _ := cm["name"].(string); name != "main" {
			continue
		}
		args, ok := cm["args"].([]interface{})
		if !ok {
			continue
		}
		for i, a := range args {
			s, ok := a.(string)
			if !ok || !strings.Contains(s, "model:") {
				continue
			}
			args[i] = strings.ReplaceAll(s, denseModel, MoEOfflineInferenceModel)
		}
		cm["args"] = args
	}
	podSpec["containers"] = containers
	tpl["spec"] = podSpec
	router["template"] = tpl
	igr["router"] = router
	spec["intelligentGatewayRouting"] = igr
}

func overlayMoEOfflineInsvcAggregate(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := insvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(cfg["spec"])
	engine := mustMap(spec["engine"])
	patchInsvcEngineMoE(engine, moeVLLMScriptAggregate, "", "")
	spec["engine"] = engine
	cfg["spec"] = spec

	insvc, err := insvcDoc(docs)
	if err != nil {
		return nil, err
	}
	patchInferNexServiceModelAndRouter(insvc)
	return docs, nil
}

func overlayMoEOfflineInsvcPD(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := insvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	spec := mustMap(cfg["spec"])
	engine := mustMap(spec["engine"])
	patchInsvcEngineMoE(engine, "", moeVLLMScriptPDDecode, moeVLLMScriptPDPrefill)
	spec["engine"] = engine
	cfg["spec"] = spec

	insvc, err := insvcDoc(docs)
	if err != nil {
		return nil, err
	}
	patchInferNexServiceModelAndRouter(insvc)
	return docs, nil
}

func overlayMoEOfflineLLMISvcAggregate(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := llmisvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	cfgSpec := mustMap(cfg["spec"])
	patchLLMISvcConfigMoE(cfgSpec, moeVLLMScriptAggregate, "", "")
	cfg["spec"] = cfgSpec

	llm, err := llmisvcDoc(docs)
	if err != nil {
		return nil, err
	}
	patchLLMInferenceServiceModel(llm)
	return docs, nil
}

func overlayMoEOfflineLLMISvcPD(docs []map[string]interface{}) ([]map[string]interface{}, error) {
	cfg, err := llmisvcConfigDoc(docs)
	if err != nil {
		return nil, err
	}
	cfgSpec := mustMap(cfg["spec"])
	patchLLMISvcConfigMoE(cfgSpec, "", moeVLLMScriptPDDecode, moeVLLMScriptPDPrefill)
	cfg["spec"] = cfgSpec

	llm, err := llmisvcDoc(docs)
	if err != nil {
		return nil, err
	}
	patchLLMInferenceServiceModel(llm)
	return docs, nil
}
