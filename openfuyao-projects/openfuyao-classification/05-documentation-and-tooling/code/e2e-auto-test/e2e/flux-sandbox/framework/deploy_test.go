/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2/
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package framework

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

// TestBuildAgentPod_FixedName verifies the e2e agent is a bare pod with a
// fixed name: reconcile re-adoption keys on the agent-name label, so keeping
// the name across RestartAgentPod recreations mirrors the production
// in-place-restart path (scheduler-managed agent pods keep their names).
func TestBuildAgentPod_FixedName(t *testing.T) {
	cfg := &Config{
		Namespace:  "flux-sandbox",
		AgentImage: "flux-sandbox/agent:latest",
		ExecdImage: "opensandbox/execd:v1.0.14",
		AgentPort:  5758,
	}
	pod := buildAgentPod(cfg)

	if pod.Name != AgentPodName {
		t.Errorf("pod name = %q, want fixed %q", pod.Name, AgentPodName)
	}
	if pod.Namespace != cfg.Namespace {
		t.Errorf("pod namespace = %q, want %q", pod.Namespace, cfg.Namespace)
	}
	// The label must keep both pod discovery (AgentPodLabel selector) and the
	// watcher fast path (app=flux-sandbox-agent dev deployment) working.
	if pod.Labels["app"] != "flux-sandbox-agent" {
		t.Errorf("pod app label = %q, want flux-sandbox-agent", pod.Labels["app"])
	}
	if pod.Spec.ServiceAccountName != ServiceAccountName {
		t.Errorf("service account = %q, want %q", pod.Spec.ServiceAccountName, ServiceAccountName)
	}

	var agent *corev1.Container
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == "agent" {
			agent = &pod.Spec.Containers[i]
		}
	}
	if agent == nil {
		t.Fatal("no agent container in pod spec")
	}
	if agent.Image != cfg.AgentImage {
		t.Errorf("agent image = %q, want %q", agent.Image, cfg.AgentImage)
	}
	if len(pod.Spec.InitContainers) != 1 || pod.Spec.InitContainers[0].Image != cfg.ExecdImage {
		t.Errorf("init containers = %+v, want single execd injector with image %s", pod.Spec.InitContainers, cfg.ExecdImage)
	}
}
