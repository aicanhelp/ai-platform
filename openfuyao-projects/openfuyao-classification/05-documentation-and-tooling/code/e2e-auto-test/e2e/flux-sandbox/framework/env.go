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

package framework

import (
	"fmt"
	"log"
	"time"
)

// Env is the fully wired e2e environment: configuration, k8s clients, the
// agent HTTP client (via port-forward), and node-level access.
type Env struct {
	// Cfg is the loaded e2e configuration.
	Cfg *Config
	// K8s holds the k8s clients.
	K8s *K8s
	// Agent talks to the agent through the port-forward.
	Agent *AgentClient
	// Node runs commands on the agent pod node.
	Node *NodeRunner
	// PF is the port-forward subprocess handle.
	PF *PortForward
	// SmallPF is the port-forward to the suite-managed small agent pod.
	SmallPF *PortForward

	// AgentPodName, AgentPodUID and AgentPodIP identify the current agent pod.
	AgentPodName string
	AgentPodUID  string
	AgentPodIP   string
	// AgentNodeName and AgentNodeIP identify the node running the agent pod.
	AgentNodeName string
	AgentNodeIP   string
}

// agentBaseURL is the local URL of the port-forwarded agent API.
func (e *Env) agentBaseURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d", e.Cfg.AgentPort)
}

// Setup loads the config, deploys components, waits for the agent pod,
// establishes node access and the port-forward, and probes healthz.
func Setup() (*Env, error) {
	cfg, err := Load()
	if err != nil {
		return nil, err
	}

	k8s, err := NewK8s(cfg)
	if err != nil {
		return nil, err
	}

	if !cfg.SkipDeploy {
		if err := DeployComponents(k8s); err != nil {
			return nil, fmt.Errorf("deploy components: %w", err)
		}
	}

	e := &Env{Cfg: cfg, K8s: k8s}
	if err := e.attachToAgentPod(); err != nil {
		return nil, err
	}

	if cfg.SmallCapacityAgentManaged {
		if err := e.setupSmallAgent(); err != nil {
			return nil, err
		}
	}

	log.Printf("e2e env ready: agent=%s uid=%s ip=%s node=%s nodeAccess=%s",
		e.AgentPodName, e.AgentPodUID, e.AgentPodIP, e.AgentNodeName, cfg.NodeAccess)
	return e, nil
}

// setupSmallAgent deploys the suite-managed capacity=2 small agent pod and
// points SmallCapacityAgentURL at a local kubectl port-forward (enables
// DP-E2E-005/029 without an externally deployed agent; port-forward works
// on both standard clusters and local kind, no node port required).
func (e *Env) setupSmallAgent() error {
	if !e.Cfg.SkipDeploy {
		if err := DeploySmallAgentPod(e.K8s); err != nil {
			return err
		}
	}
	info, err := e.K8s.WaitForReadyPod(SmallAgentPodLabel)
	if err != nil {
		return fmt.Errorf("wait small agent pod: %w", err)
	}
	url := fmt.Sprintf("http://127.0.0.1:%d", SmallAgentLocalPort)
	probe := func() error {
		client := NewAgentClient(url, e.Cfg.AgentAuthToken)
		code, _, err := client.Healthz()
		if err != nil {
			return err
		}
		if code != 200 {
			return fmt.Errorf("healthz status %d", code)
		}
		return nil
	}
	if e.SmallPF != nil {
		e.SmallPF.Stop()
	}
	e.SmallPF, err = StartPortForward(e.Cfg.Namespace, info.Name, SmallAgentLocalPort, e.Cfg.AgentPort, probe)
	if err != nil {
		return fmt.Errorf("small agent port-forward: %w", err)
	}
	e.Cfg.SmallCapacityAgentURL = url
	log.Printf("small capacity agent ready: pod=%s node=%s url=%s", info.Name, info.Node, e.Cfg.SmallCapacityAgentURL)
	return nil
}

// attachToAgentPod waits for a ready agent pod, (re)builds the Node runner,
// port-forward and agent client, and probes healthz.
func (e *Env) attachToAgentPod() error {
	info, err := e.K8s.WaitForAgentPod()
	if err != nil {
		return err
	}
	e.AgentPodName, e.AgentPodUID, e.AgentPodIP, e.AgentNodeName = info.Name, info.UID, info.IP, info.Node

	if e.AgentNodeIP, err = e.K8s.NodeIP(info.Node); err != nil {
		return fmt.Errorf("get node %s IP: %w", info.Node, err)
	}
	e.Node = NewNodeRunner(e.Cfg, e.AgentNodeName, e.AgentNodeIP)
	e.Agent = NewAgentClient(e.agentBaseURL(), e.Cfg.AgentAuthToken)

	if e.PF != nil {
		e.PF.Stop()
	}
	probe := func() error {
		code, _, err := e.Agent.Healthz()
		if err != nil {
			return err
		}
		if code != 200 {
			return fmt.Errorf("healthz status %d", code)
		}
		return nil
	}
	e.PF, err = StartPortForward(e.Cfg.Namespace, e.AgentPodName, e.Cfg.AgentPort, e.Cfg.AgentPort, probe)
	if err != nil {
		return fmt.Errorf("port-forward: %w", err)
	}
	return nil
}

// RestartAgentPod deletes the current agent pod and re-attaches to the
// rebuilt one (new identity, restarted port-forward). The pod is a bare pod
// with a fixed name, so it is recreated under the SAME name (only the UID
// changes): this mirrors the production in-place-restart path where the
// agent's name-based reconcile still finds its sandboxes.
func (e *Env) RestartAgentPod() error {
	if err := e.K8s.DeleteAgentPod(e.AgentPodName); err != nil {
		return fmt.Errorf("delete agent pod %s: %w", e.AgentPodName, err)
	}
	// Wait for the old pod object to disappear so the fixed name is free
	// again and WaitForAgentPod does not observe the dying pod.
	if err := WaitFor(fmt.Sprintf("agent pod %s to disappear", e.AgentPodName), 60*time.Second, func() bool {
		_, err := e.K8s.GetPod(e.AgentPodName)
		return err != nil
	}); err != nil {
		return err
	}
	if err := e.K8s.CreateAgentPod(); err != nil {
		return fmt.Errorf("recreate agent pod: %w", err)
	}
	return e.attachToAgentPod()
}

// Teardown stops the port-forward, removes the suite-managed small agent
// pod, and optionally deletes the deployed components. It is a no-op when
// Setup never completed.
func (e *Env) Teardown() {
	if e.Cfg == nil {
		return
	}
	if e.PF != nil {
		e.PF.Stop()
	}
	if e.SmallPF != nil {
		e.SmallPF.Stop()
	}
	if e.K8s != nil && e.Cfg.SmallCapacityAgentManaged {
		// The small pod is suite-managed ephemera: always remove it so the
		// next run recreates it with the current image and settings.
		if err := CleanupSmallAgentPod(e.K8s); err != nil {
			log.Printf("cleanup warning: %v", err)
		}
	}
	if e.Cfg.Cleanup {
		if err := CleanupComponents(e.K8s); err != nil {
			log.Printf("cleanup warning: %v", err)
		}
	}
}
