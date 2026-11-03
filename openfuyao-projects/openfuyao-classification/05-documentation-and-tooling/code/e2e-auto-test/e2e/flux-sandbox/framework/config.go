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

// Package framework provides reusable infrastructure for flux-sandbox e2e
// suites. It targets standard Linux + k8s clusters as the primary
// environment; kind/docker based access is an interchangeable convenience
// implementation for local runs.
package framework

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Node access transports, selected via E2E_NODE_ACCESS.
const (
	NodeAccessSSH    = "ssh"    // standard Linux + k8s: ssh to the node (default)
	NodeAccessDocker = "docker" // local kind dev: docker exec into the kind node
	NodeAccessLocal  = "local"  // test process runs on the node itself
)

// Default component and timing values used by the e2e deployment.
const (
	// DefaultAgentPort is the agent HTTP control port.
	DefaultAgentPort = 5758
	// DefaultNamespace is the namespace the e2e components are deployed into.
	DefaultNamespace = "flux-sandbox"
	// OrphanTimeout is passed to the watcher via --orphan-timeout. It is the
	// age below which a container is never considered an orphan.
	OrphanTimeout = 30 * time.Second
)

// pollInterval is the tick used by the polling helpers (var so tests can
// shorten it).
var pollInterval = 3 * time.Second

// Config holds all environment-driven e2e settings. It is loaded once per run
// via Load.
type Config struct {
	// Kubeconfig is the path to the kubeconfig file for the target cluster.
	Kubeconfig string
	// Namespace receives the e2e agent/watcher deployments.
	Namespace string
	// AgentImage is the container image for the flux-sandbox-agent DaemonSet.
	AgentImage string
	// WatcherImage is the container image for the sandbox-watcher DaemonSet.
	WatcherImage string
	// ExecdImage is the init-container image injecting the execd binary.
	ExecdImage string
	// SkipDeploy skips deploying components (they are already installed).
	SkipDeploy bool
	// Cleanup deletes the deployed components during teardown.
	Cleanup bool
	// NodeAccess selects the node command transport: ssh, docker, or local.
	NodeAccess string
	// NodeSSHUser is the ssh user for node access (ssh mode).
	NodeSSHUser string
	// NodeSSHHost overrides the ssh target; defaults to the agent node IP.
	NodeSSHHost string
	// AgentAuthToken is sent as a Bearer token when set (optional).
	AgentAuthToken string
	// GracePeriod is the watcher orphan grace period (--orphan-grace-period).
	GracePeriod time.Duration
	// ScanInterval is the watcher scan interval (--scan-interval).
	ScanInterval time.Duration
	// SmallCapacityAgentURL points at the capacity=2 agent used by
	// DP-E2E-005/029. It comes from E2E_SMALL_CAPACITY_AGENT_URL (externally
	// deployed agent), or is resolved by the suite itself when
	// SmallCapacityAgentManaged is set. When both are unset the specs skip.
	SmallCapacityAgentURL string
	// SmallCapacityAgentManaged lets the suite deploy (and clean up) its own
	// capacity=2 small agent pod for DP-E2E-005/029, instead of requiring an
	// externally deployed agent behind E2E_SMALL_CAPACITY_AGENT_URL.
	SmallCapacityAgentManaged bool
	// NodeSSHReady gates specs that need node-level (cgroup/FIFO) access.
	NodeSSHReady bool
	// ExecdReachable gates the spec that curls the execd endpoint from a node.
	ExecdReachable bool
	// AgentPort is the local port-forward target port of the agent.
	AgentPort int
}

// Load reads the e2e configuration from environment variables.
func Load() (*Config, error) {
	cfg := &Config{
		Kubeconfig:                envOrDefault("E2E_KUBECONFIG", defaultKubeconfigPath()),
		Namespace:                 envOrDefault("E2E_NAMESPACE", DefaultNamespace),
		AgentImage:                envOrDefault("E2E_AGENT_IMAGE", "flux-sandbox/agent:latest"),
		WatcherImage:              envOrDefault("E2E_WATCHER_IMAGE", "flux-sandbox/sandbox-watcher:latest"),
		ExecdImage:                envOrDefault("E2E_EXECD_IMAGE", "opensandbox/execd:v1.0.14"),
		SkipDeploy:                os.Getenv("E2E_SKIP_DEPLOY") != "",
		Cleanup:                   os.Getenv("E2E_CLEANUP") != "",
		NodeAccess:                envOrDefault("E2E_NODE_ACCESS", NodeAccessSSH),
		NodeSSHUser:               envOrDefault("E2E_NODE_SSH_USER", "root"),
		NodeSSHHost:               os.Getenv("E2E_NODE_SSH_HOST"),
		AgentAuthToken:            os.Getenv("E2E_AGENT_AUTH_TOKEN"),
		SmallCapacityAgentURL:     os.Getenv("E2E_SMALL_CAPACITY_AGENT_URL"),
		SmallCapacityAgentManaged: os.Getenv("E2E_SMALL_CAPACITY_AGENT") != "",
		NodeSSHReady:              os.Getenv("E2E_NODE_SSH_READY") != "",
		ExecdReachable:            os.Getenv("E2E_EXECD_REACHABLE") != "",
		AgentPort:                 DefaultAgentPort,
	}

	switch cfg.NodeAccess {
	case NodeAccessSSH, NodeAccessDocker, NodeAccessLocal:
	default:
		return nil, fmt.Errorf("invalid E2E_NODE_ACCESS %q: want %q, %q, or %q",
			cfg.NodeAccess, NodeAccessSSH, NodeAccessDocker, NodeAccessLocal)
	}

	var err error
	if cfg.GracePeriod, err = parseDurationEnv("E2E_ORPHAN_GRACE_PERIOD", 30*time.Second); err != nil {
		return nil, err
	}
	if cfg.ScanInterval, err = parseDurationEnv("E2E_SCAN_INTERVAL", 30*time.Second); err != nil {
		return nil, err
	}
	return cfg, nil
}

// NotCleanedWindow returns the observation window used to assert that the
// watcher does NOT clean a container. It stays safely below the grace period.
func (c *Config) NotCleanedWindow() time.Duration {
	window := c.GracePeriod - 5*time.Second
	if window < 10*time.Second {
		window = 10 * time.Second
	}
	return window
}

// CleanupTimeout returns the maximum time to wait for the watcher to clean an
// orphan: orphanTimeout + grace + two scan intervals + a safety margin.
func (c *Config) CleanupTimeout() time.Duration {
	return OrphanTimeout + c.GracePeriod + 2*c.ScanInterval + 60*time.Second
}

func envOrDefault(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func parseDurationEnv(key string, defaultVal time.Duration) (time.Duration, error) {
	v := os.Getenv(key)
	if v == "" {
		return defaultVal, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s=%q: %w", key, v, err)
	}
	return d, nil
}

func defaultKubeconfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".kube", "config")
}
