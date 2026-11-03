/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1alpha1

// infernexMutateTargetLLMInferenceServiceConfigNames lists LLMInferenceServiceConfig objects that
// need InferNex patches (clear llm-d initContainers on decode/aggregate templates).
// capabilities.drop ALL is preserved on all templates so workloads stay compatible with restricted
// Pod Security Admission; NPU device access is granted via pod-level supplementalGroups.
//
// Not listed (add if you install and use them in-cluster):
//   - kserve-config-llm-*-pipeline-parallel (PP worker presets from KServe combineBaseRefsConfig)
//   - kserve-config-llm-router-route (HTTPRoute-only; usually no-op for this webhook)
//   - kserve-config-llm-scheduler (Hermes overlays the upstream preset; Bridge must not sanitize it)
//
// Resource names match metadata.name in each YAML (prefix kserve-config-llm-*).
var infernexMutateTargetLLMInferenceServiceConfigNames = []string{
	"kserve-config-llm-template",
	"kserve-config-llm-worker-data-parallel",
	"kserve-config-llm-decode-template",
	"kserve-config-llm-decode-worker-data-parallel",
	"kserve-config-llm-prefill-template",
	"kserve-config-llm-prefill-worker-data-parallel",
}
