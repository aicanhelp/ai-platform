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

// Package controlplane contains end-to-end tests for the flux-sandbox control
// plane (Controller + Scheduler + CRDs), assuming the components are already
// deployed in the cluster. It exercises SandboxGroup/SubSandboxGroup lifecycle,
// Agent Pod generation, FastPath gRPC sandbox lifecycle, Router load balancing,
// health/readiness gates, the containerd runtime, and fault handling.
package controlplane

import (
	"os"
	"path/filepath"
	"strconv"

	"google.golang.org/grpc"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	sandboxv1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/proto/v1"
)

// Environment-driven configuration. Declared here, populated by initCPEnv
// which runs in BeforeSuite — after the repo-level .env has been loaded — so
// values from .env are honored (package-level init would run too early).
var (
	cpKubeconfig        string
	cpNamespace         string
	cpControllerPod     string
	cpSchedulerReplicas int
	cpAgentImage        string
	cpExecdImage        string
	cpSandboxImage      string
	cpReserveImage      string
	cpDestructive       bool
	cpBigSharding       bool
	cpCleanup           bool
)

func initCPEnv() {
	cpKubeconfig = envOrDefault("E2E_KUBECONFIG", defaultKubeconfigPath())
	cpNamespace = envOrDefault("E2E_CP_NAMESPACE", "flux-system")
	cpControllerPod = os.Getenv("E2E_CONTROLLER_POD") // empty => auto-discover by label
	cpSchedulerReplicas = envOrDefaultInt("E2E_SCHEDULER_REPLICAS", 3)
	cpAgentImage = envOrDefault("E2E_AGENT_IMAGE", "flux-sandbox/agent:latest")
	cpExecdImage = envOrDefault("E2E_EXECD_IMAGE", "opensandbox/execd:v1.0.20")
	cpSandboxImage = envOrDefault("E2E_SANDBOX_IMAGE", "docker.io/library/busybox:latest")
	cpReserveImage = envOrDefault("E2E_RESERVE_IMAGE", "registry.k8s.io/pause:3.10")
	cpDestructive = os.Getenv("E2E_DESTRUCTIVE") != ""  // run destructive cases (030/037)
	cpBigSharding = os.Getenv("E2E_BIG_SHARDING") != "" // run large-scale sharding case (039)
	cpCleanup = os.Getenv("E2E_CLEANUP") != ""          // delete created SandboxGroups in teardown
}

// Local port-forward ports. Chosen to avoid clashing with the data-plane e2e
// package (which uses 5758/44772) and with each other.
const (
	cpControllerHTTPPort = "18080" // controller HTTP (healthz/readyz/debug)
	cpControllerGRPCPort = "19091" // controller gRPC (SandboxProviderService)
)

// Global test environment, initialized by BeforeSuite.
var cpEnv *testEnv

type testEnv struct {
	k8sClient     client.Client
	kubeClient    *kubernetes.Clientset
	grpcConn      *grpc.ClientConn
	grpcClient    sandboxv1.SandboxProviderServiceClient
	controllerPod string
	httpPFCmd     *os.Process // port-forward for controller HTTP 8080
	grpcPFCmd     *os.Process // port-forward for controller gRPC 9091
	crdApplied    bool        // whether CRD apply was attempted in setup
}

func envOrDefault(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func envOrDefaultInt(key string, defaultVal int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return defaultVal
}

func defaultKubeconfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".kube", "config")
}
