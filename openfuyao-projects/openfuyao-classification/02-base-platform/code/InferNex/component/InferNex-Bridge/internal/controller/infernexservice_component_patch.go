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

package controller

import (
	"fmt"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// InferNexService-managed PD labels: openfuyao + infernex branding (no kserve.io/*; distinct from llmisvc KServe templates).
// openfuyao.com/pdRole + pdGroupID align with inference-backend charts and Hermes filter-by-pd-label when using those keys.
const (
	labelAppKubernetesIOName      = "app.kubernetes.io/name"
	labelAppKubernetesIOComponent = "app.kubernetes.io/component"
	labelAppKubernetesIOPartOf    = "app.kubernetes.io/part-of"
	labelInfernexPDEngineGroup    = "infernex.io/pdEngineGroup"    // prefill+decode; cache-indexer insvc PD discovery engineKey
	labelInfernexAggregateGroup   = "infernex.io/aggregateGroup" // aggregate engine; cache-indexer insvc aggregate discovery engineKey
	labelInfernexOwner            = "infernex.io/owner"
	labelInfernexComponent        = "infernex.io/component"
	valueInfernexEngineAggregate  = "engine-aggregate"
	labelProxyService             = "proxy-service"
	labelOpenFuyaoPDRole          = "openfuyao.com/pdRole"
	labelOpenFuyaoPDGroup         = "openfuyao.com/pdGroupID"

	// app.kubernetes.io/part-of value for all Bridge-managed PD pods (engines + proxy).
	valueInfernexAppPartOf = "infernex"
	// app.kubernetes.io/component values (observability; router may instead use openfuyao.com/pdRole).
	valueInfernexPDPrefillComponent = "infernex-pd-prefill"
	valueInfernexPDDecodeComponent  = "infernex-pd-decode"
	valueInfernexPDProxyComponent   = "infernex-pd-proxy-server"

	openfuyaoPDRolePrefill = "prefill"
	openfuyaoPDRoleDecode  = "decode"
	// Proxy-server pod: Hermes filter-by-pd-label leaderValue matches this (not the word "proxy").
	openfuyaoPDRoleLeader = "leader"

	// KServe / LLMInferenceService PD pods (sourceRef-linked mode): engines are not labeled by Bridge.
	labelKServeComponent           = "kserve.io/component"
	labelLLMDRole                  = "llm-d.ai/role"
	// Non-LWS (single-node) KServe PD component labels.
	kserveWorkloadComponentDecode  = "llminferenceservice-workload"
	kserveWorkloadComponentPrefill = "llminferenceservice-workload-prefill"
	// LWS (multi-node) KServe PD component labels — see kserve/pkg/constants/constants.go.
	kserveWorkloadComponentLeader        = "llminferenceservice-workload-leader"
	kserveWorkloadComponentWorker        = "llminferenceservice-workload-worker"
	kserveWorkloadComponentLeaderPrefill = "llminferenceservice-workload-leader-prefill"
	kserveWorkloadComponentWorkerPrefill = "llminferenceservice-workload-worker-prefill"
	valueKServeAppPartOf                 = "llminferenceservice"
	valueKServeProxyComponent      = "pd-proxy"

	envProxyPrefillLabel      = "INFERNEX_PROXY_PREFILLER_LABEL"
	envProxyDecoderLabel      = "INFERNEX_PROXY_DECODER_LABEL"
	envProxyWorkloadNS        = "INFERNEX_PROXY_WORKLOAD_NAMESPACE"
	envProxyDiscoveryInterval = "INFERNEX_PROXY_DISCOVERY_INTERVAL_SEC"
	envProxyLegacyPrefillArg  = "INFERNEX_PROXY_ARG_PREFILLER_LABEL"
	envProxyLegacyDecoderArg  = "INFERNEX_PROXY_ARG_DECODER_LABEL"
	defaultProxyDiscoverySec  = 10
)

// infernexEngineGroupValue is the cache-indexer discovery engineValue for direct InferNexService engines.
// Uses "namespace.name" (not "/") because Kubernetes label values cannot contain slashes.
func infernexEngineGroupValue(namespace, name string) string {
	return fmt.Sprintf("%s.%s", strings.TrimSpace(namespace), strings.TrimSpace(name))
}

func openfuyaoPDPodSelectorString(pdRole, workloadName string) string {
	return fmt.Sprintf("%s=%s,%s=%s",
		labelOpenFuyaoPDRole, pdRole,
		labelOpenFuyaoPDGroup, workloadName,
	)
}

// mergePodTemplateLabelIfAbsent sets tpl.Labels[k]=v only when k is missing or the current value is empty
// (after trim). Non-empty user-supplied labels are kept; controller only appends missing integration keys.
func mergePodTemplateLabelIfAbsent(tpl *corev1.PodTemplateSpec, k, v string) {
	if tpl == nil || strings.TrimSpace(k) == "" || strings.TrimSpace(v) == "" {
		return
	}
	if tpl.Labels == nil {
		tpl.Labels = map[string]string{}
	}
	if existing, ok := tpl.Labels[k]; ok && strings.TrimSpace(existing) != "" {
		return
	}
	tpl.Labels[k] = v
}

// kserveComponentInSelector builds "app.kubernetes.io/component in (a,b,c)" for K8s list pods.
// Comma-separated keys in the full selector are AND; "in" is OR among values of one key.
func kserveComponentInSelector(components ...string) string {
	return fmt.Sprintf("%s in (%s)", labelAppKubernetesIOComponent, strings.Join(components, ","))
}

// llmisvcProxyPrefillSelector discovers all non-LWS prefill pods and LWS prefill leader+worker.
func llmisvcProxyPrefillSelector(workloadName string) string {
	return fmt.Sprintf("%s,%s=%s",
		kserveComponentInSelector(
			kserveWorkloadComponentPrefill,
			kserveWorkloadComponentLeaderPrefill,
			kserveWorkloadComponentWorkerPrefill,
		),
		labelAppKubernetesIOName, workloadName,
	)
}

// llmisvcProxyDecodeSelector discovers all non-LWS decode pods and LWS decode leader+worker.
func llmisvcProxyDecodeSelector(workloadName string) string {
	return fmt.Sprintf("%s,%s=%s",
		kserveComponentInSelector(
			kserveWorkloadComponentDecode,
			kserveWorkloadComponentLeader,
			kserveWorkloadComponentWorker,
		),
		labelAppKubernetesIOName, workloadName,
	)
}

// mergePDInferenceWorkloadPodTemplateLabels sets PD engine pod labels for Hermes EPP, cache-indexer, and
// k8s_proxy_server discovery. Missing keys are filled; existing non-empty user labels are not overwritten.
// Deployment pod selector uses openfuyao.com/pdRole|pdGroupID for direct PD (see deploymentPodMatchLabels);
// infernex.io/* resource labels are applied in reconcile.
func mergePDInferenceWorkloadPodTemplateLabels(tpl *corev1.PodTemplateSpec, namespace, workloadName string, prefill bool) {
	if tpl == nil || strings.TrimSpace(namespace) == "" || strings.TrimSpace(workloadName) == "" {
		return
	}
	if tpl.Labels == nil {
		tpl.Labels = map[string]string{}
	}
	mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOPartOf, valueInfernexAppPartOf)
	mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOName, workloadName)
	mergePodTemplateLabelIfAbsent(tpl, labelInfernexPDEngineGroup, infernexEngineGroupValue(namespace, workloadName))
	if prefill {
		mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOComponent, valueInfernexPDPrefillComponent)
		mergePodTemplateLabelIfAbsent(tpl, labelOpenFuyaoPDRole, openfuyaoPDRolePrefill)
	} else {
		mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOComponent, valueInfernexPDDecodeComponent)
		mergePodTemplateLabelIfAbsent(tpl, labelOpenFuyaoPDRole, openfuyaoPDRoleDecode)
	}
	mergePodTemplateLabelIfAbsent(tpl, labelOpenFuyaoPDGroup, workloadName)
}

func preferredContainer(tpl *corev1.PodTemplateSpec, names ...string) *corev1.Container {
	if tpl == nil || len(tpl.Spec.Containers) == 0 {
		return nil
	}
	for _, name := range names {
		for i := range tpl.Spec.Containers {
			if tpl.Spec.Containers[i].Name == name {
				return &tpl.Spec.Containers[i]
			}
		}
	}
	return &tpl.Spec.Containers[0]
}

// mergeProxyPDLabelsAndEnv sets proxy pod labels and INFERNEX_PROXY_* env to match PD engine labels.
// kserveBackend: true when engines come from LLMInferenceService (sourceRef); keep KServe/llmisvc selectors.
// No chart/CR user input: chart installs controller + default InferNexServiceConfig; this runs at reconcile.
func mergeProxyPDLabelsAndEnv(
	tpl *corev1.PodTemplateSpec,
	workloadName,
	workloadNS string,
	discoveryIntervalSec int32,
	kserveBackend bool,
) {
	if tpl == nil {
		return
	}
	if tpl.Labels == nil {
		tpl.Labels = map[string]string{}
	}
	var pref, dec string
	if kserveBackend {
		// Align with KServe-managed PD pods; proxy discovery uses component "in" + name (see INFERNEX_PROXY_*).
		// Do not add openfuyao.com/pdRole|pdGroupID here — that convention is for InferNexService direct PD only.
		mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOPartOf, valueKServeAppPartOf)
		mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOComponent, valueKServeProxyComponent)
		mergePodTemplateLabelIfAbsent(tpl, labelKServeComponent, "workload")
		mergePodTemplateLabelIfAbsent(tpl, labelProxyService, "true")
		mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOName, workloadName)
		pref = llmisvcProxyPrefillSelector(workloadName)
		dec = llmisvcProxyDecodeSelector(workloadName)
	} else {
		// Hermes filter-by-pd-label: openfuyao.com/pdRole + pdGroupID (see
		// config/examples/pd_infernexservice_example.yaml).
		mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOPartOf, valueInfernexAppPartOf)
		mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOComponent, valueInfernexPDProxyComponent)
		mergePodTemplateLabelIfAbsent(tpl, labelProxyService, "true")
		mergePodTemplateLabelIfAbsent(tpl, labelAppKubernetesIOName, workloadName)
		mergePodTemplateLabelIfAbsent(tpl, labelOpenFuyaoPDRole, openfuyaoPDRoleLeader)
		mergePodTemplateLabelIfAbsent(tpl, labelOpenFuyaoPDGroup, workloadName)
		pref = openfuyaoPDPodSelectorString(openfuyaoPDRolePrefill, workloadName)
		dec = openfuyaoPDPodSelectorString(openfuyaoPDRoleDecode, workloadName)
	}
	interval := discoveryIntervalSec
	if interval <= 0 {
		interval = defaultProxyDiscoverySec
	}

	idx := 0
	for i := range tpl.Spec.Containers {
		if tpl.Spec.Containers[i].Name == "proxy-server" {
			idx = i
			break
		}
	}
	c := &tpl.Spec.Containers[idx]
	mergeEnvVar(c, envProxyPrefillLabel, pref)
	mergeEnvVar(c, envProxyDecoderLabel, dec)
	mergeEnvVar(c, envProxyWorkloadNS, workloadNS)
	mergeEnvVar(c, envProxyDiscoveryInterval, strconv.Itoa(int(interval)))
	// Quoted strings suitable for copying into --prefiller-label / --decoder-label style CLIs.
	mergeEnvVar(c, envProxyLegacyPrefillArg, pref)
	mergeEnvVar(c, envProxyLegacyDecoderArg, dec)
}

func mergeEnvVar(c *corev1.Container, name, value string) {
	if strings.TrimSpace(value) == "" {
		return
	}
	for i := range c.Env {
		if c.Env[i].Name == name {
			c.Env[i].Value = value
			return
		}
	}
	c.Env = append(c.Env, corev1.EnvVar{Name: name, Value: value})
}
