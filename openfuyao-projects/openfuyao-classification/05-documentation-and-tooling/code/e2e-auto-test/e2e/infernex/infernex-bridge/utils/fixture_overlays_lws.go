package utils

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	lwsDataParallelSize      = 2
	lwsDataParallelSizeLocal = 1

	lwsLeaderZMQPort = 5557
	lwsWorkerZMQPort = 5558

	// Placeholders resolved by RenderOptions.Name in BuildFixtureFromRecipe.
	lwsNameInsvcAggregate = "__NAME__-engine-aggregate"
	lwsNameInsvcPDDecode  = "__NAME__-engine-pd-decode"
	lwsNameInsvcPDPrefill = "__NAME__-engine-pd-prefill"
	lwsNameLLMISvcDecode  = "__NAME__-kserve-mn"
	lwsNameLLMISvcPrefill = "__NAME__-kserve-mn-prefill"
)

// patchVLLMServeScriptForLWSE2E adjusts MoE offline vLLM scripts for LWS leader/worker kv-events zmq ports.
func patchVLLMServeScriptForLWSE2E(script string, zmqPubPort int) string {
	return patchKVEventsZMQEndpoint(script, zmqPubPort)
}

func patchKVEventsZMQEndpoint(script string, zmqPubPort int) string {
	if !strings.Contains(script, "--kv-events-config") || zmqPubPort <= 0 {
		return script
	}
	want := fmt.Sprintf(`"endpoint":"tcp://*:%d"`, zmqPubPort)
	if strings.Contains(script, want) {
		return script
	}
	const endpointPrefix = `"endpoint":"tcp://*:`
	if i := strings.Index(script, endpointPrefix); i >= 0 {
		j := i + len(endpointPrefix)
		for j < len(script) && script[j] >= '0' && script[j] <= '9' {
			j++
		}
		// want already ends with `"`; skip the closing quote of the replaced port.
		if j < len(script) && script[j] == '"' {
			j++
		}
		return script[:i] + want + script[j:]
	}
	const needle = `"publisher":"zmq", "topic":`
	replacement := fmt.Sprintf(`"publisher":"zmq", "endpoint":"tcp://*:%d", "topic":`, zmqPubPort)
	return strings.Replace(script, needle, replacement, 1)
}

func deepCloneMap(m map[string]interface{}) map[string]interface{} {
	raw, err := yaml.Marshal(m)
	if err != nil {
		return map[string]interface{}{}
	}
	var out map[string]interface{}
	if err := yaml.Unmarshal(raw, &out); err != nil {
		return map[string]interface{}{}
	}
	return out
}

func patchPodSpecMainContainerArgs(podSpec map[string]interface{}, zmqPubPort int) {
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
		args, ok := cm["args"].([]interface{})
		if !ok || len(args) == 0 {
			continue
		}
		s, ok := args[0].(string)
		if !ok || !strings.Contains(s, "vllm serve") {
			continue
		}
		args[0] = patchVLLMServeScriptForLWSE2E(s, zmqPubPort)
		cm["args"] = args
		return
	}
}

// bumpWorkerZMQPubPort matches ag-03/pd-05: leader zmq-pub 5557, worker 5558.
func bumpWorkerZMQPubPort(podSpec map[string]interface{}, workerPort int) {
	containers, ok := podSpec["containers"].([]interface{})
	if !ok {
		return
	}
	for _, c := range containers {
		cm, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		ports, ok := cm["ports"].([]interface{})
		if !ok {
			continue
		}
		for _, p := range ports {
			pm, ok := p.(map[string]interface{})
			if !ok {
				continue
			}
			name, _ := pm["name"].(string)
			if name == "zmq-pub" {
				pm["containerPort"] = workerPort
			}
		}
	}
}

func cloneLWSWorkerPodSpec(leaderPodSpec map[string]interface{}) map[string]interface{} {
	worker := deepCloneMap(leaderPodSpec)
	bumpWorkerZMQPubPort(worker, lwsWorkerZMQPort)
	patchPodSpecMainContainerArgs(worker, lwsWorkerZMQPort)
	return worker
}

const lwsPodAntiAffinityTopologyKey = "kubernetes.io/hostname"

// mergePodSpecLWSPodAntiAffinity spreads LWS leader/worker across nodes (FIT 56–59).
// MoE initContainer downloads to per-node hostPath cache on miss; complete cache skips download.
func mergePodSpecLWSPodAntiAffinity(podSpec map[string]interface{}, lwsName string) {
	delete(podSpec, "nodeName")
	matchLabels := map[string]interface{}{
		"leaderworkerset.sigs.k8s.io/name": lwsName,
	}
	affinity, _ := podSpec["affinity"].(map[string]interface{})
	if affinity == nil {
		affinity = map[string]interface{}{}
	}
	affinity["podAntiAffinity"] = map[string]interface{}{
		"requiredDuringSchedulingIgnoredDuringExecution": []interface{}{
			map[string]interface{}{
				"labelSelector": map[string]interface{}{
					"matchLabels": matchLabels,
				},
				"topologyKey": lwsPodAntiAffinityTopologyKey,
			},
		},
	}
	podSpec["affinity"] = affinity
}

// patchInsvcEngineWorkloadForLWS patches InferNexServiceConfig engine.{template,worker} (PodTemplateSpec).
func patchInsvcEngineWorkloadForLWS(workload map[string]interface{}, lwsName string) {
	workload["dataParallelSize"] = lwsDataParallelSize
	workload["dataParallelSizeLocal"] = lwsDataParallelSizeLocal
	tpl, ok := workload["template"].(map[string]interface{})
	if !ok {
		return
	}
	podSpec := mustMap(tpl["spec"])
	patchPodSpecMainContainerArgs(podSpec, lwsLeaderZMQPort)
	mergePodSpecLWSPodAntiAffinity(podSpec, lwsName)
	tpl["spec"] = podSpec
	workload["template"] = tpl
	workload["worker"] = map[string]interface{}{"spec": cloneLWSWorkerPodSpec(podSpec)}
}

func patchInsvcEngineSpecForLWS(engine map[string]interface{}, decodeLWSName, prefillLWSName string) {
	patchInsvcEngineWorkloadForLWS(engine, decodeLWSName)
	if prefill, ok := engine["prefill"].(map[string]interface{}); ok {
		patchInsvcEngineWorkloadForLWS(prefill, prefillLWSName)
		engine["prefill"] = prefill
	}
}

// patchLLMISvcFlatWorkload patches LLMInferenceServiceConfig template/worker (flat PodSpec, ag-03 shape).
func patchLLMISvcFlatWorkload(block map[string]interface{}, lwsName string) {
	tpl, ok := block["template"].(map[string]interface{})
	if !ok {
		// aggregate/decode: template is the flat PodSpec itself
		if _, hasContainers := block["containers"]; hasContainers {
			patchPodSpecMainContainerArgs(block, lwsLeaderZMQPort)
			mergePodSpecLWSPodAntiAffinity(block, lwsName)
			block["worker"] = cloneLWSWorkerPodSpec(block)
			return
		}
		return
	}
	patchPodSpecMainContainerArgs(tpl, lwsLeaderZMQPort)
	mergePodSpecLWSPodAntiAffinity(tpl, lwsName)
	block["template"] = tpl
	block["worker"] = cloneLWSWorkerPodSpec(tpl)
}

func patchLLMISvcConfigSpecForLWS(spec map[string]interface{}, decodeLWSName, prefillLWSName string) {
	if tpl, ok := spec["template"].(map[string]interface{}); ok {
		patchPodSpecMainContainerArgs(tpl, lwsLeaderZMQPort)
		mergePodSpecLWSPodAntiAffinity(tpl, decodeLWSName)
		spec["template"] = tpl
		spec["worker"] = cloneLWSWorkerPodSpec(tpl)
	}
	if prefill, ok := spec["prefill"].(map[string]interface{}); ok {
		patchLLMISvcFlatWorkload(prefill, prefillLWSName)
		spec["prefill"] = prefill
	}
}

func llmisvcLWSParallelismFields() map[string]interface{} {
	return map[string]interface{}{
		"parallelism": map[string]interface{}{"data": lwsDataParallelSize, "dataLocal": lwsDataParallelSizeLocal},
		"worker":      map[string]interface{}{},
	}
}
