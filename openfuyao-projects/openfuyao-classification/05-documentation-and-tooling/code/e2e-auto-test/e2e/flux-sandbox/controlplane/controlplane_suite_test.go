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

// Package controlplane wires the flux-sandbox control-plane e2e cases (ported
// from the flux-sandbox standalone suite) into the e2e-auto-test Ginkgo
// framework: BeforeSuite/AfterSuite replace the original TestMain, and every
// former TestXxx function is exposed as a spec in original run order.
package controlplane

import (
	"log"
	"testing"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	fenv "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func init() {
	// 与本仓库其他 Suite 保持一致：从 e2e 根目录加载 .env（已导出的环境变量优先）。
	if err := fenv.LoadEnv(".env"); err != nil {
		log.Fatalf("Failed to load .env file: %v", err)
	}
}

// TestControlplane is the entry point Ginkgo uses to discover and run every
// spec in this suite.
func TestControlplane(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "Controlplane E2E Suite")
}

var _ = ginkgo.BeforeSuite(func() {
	var err error
	cpEnv, err = setup()
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "control-plane e2e setup failed")
})

var _ = ginkgo.AfterSuite(func() {
	teardown()
})

// specT pairs Ginkgo's testing.T-compatible object with the original go-test
// function name, so cpName(t) derives the exact same SandboxGroup/Sandbox CR
// names as the standalone suite did — several cases pre-clean leftovers of a
// previous run by that stable name.
type specT struct {
	ginkgo.GinkgoTInterface
	name string
}

func (s specT) Name() string { return s.name }

func newSpecT(name string) specT {
	return specT{GinkgoTInterface: ginkgo.GinkgoT(), name: name}
}

var _ = ginkgo.Describe("Agent Pod Generation", ginkgo.Label("flux-sandbox", "agent-pod"), func() {
	ginkgo.It("FS-FUNC-006: agent pod dual-container structure", func() {
		testAgentPod_Structure(newSpecT("TestAgentPod_Structure"))
	})
	ginkgo.It("FS-FUNC-007: agent env injection", func() {
		testAgentPod_EnvInjection(newSpecT("TestAgentPod_EnvInjection"))
	})
	ginkgo.It("FS-FUNC-011: agent pod auto rebuild", func() {
		testAgentPod_AutoRebuild(newSpecT("TestAgentPod_AutoRebuild"))
	})
	ginkgo.It("FS-FUNC-012: execd init container", func() {
		testAgentPod_ExecdInitContainer(newSpecT("TestAgentPod_ExecdInitContainer"))
	})
	ginkgo.It("FS-FUNC-013: E2B runtime with no execd", ginkgo.Label("e2b"), func() {
		testAgentPod_E2BNoExecd(newSpecT("TestAgentPod_E2BNoExecd"))
	})
	ginkgo.It("FS-FUNC-014: containerd runtime with no execd fails to create sandboxes", func() {
		testAgentPod_ContainerdNoExecdFails(newSpecT("TestAgentPod_ContainerdNoExecdFails"))
	})
})

var _ = ginkgo.Describe("Controller ConfigMap", ginkgo.Label("flux-sandbox", "controller-config"), func() {
	ginkgo.It("FS-FUNC-036: default config map drives agent pods", func() {
		testControllerConfig_DefaultConfigMap(newSpecT("TestControllerConfig_DefaultConfigMap"))
	})
	ginkgo.It("FS-FUNC-037: missing config map falls back to built-in defaults", ginkgo.Label("destructive"), func() {
		testControllerConfig_MissingConfigMapFallback(newSpecT("TestControllerConfig_MissingConfigMapFallback"))
	})
})

var _ = ginkgo.Describe("FastPath gRPC Sandbox Lifecycle", ginkgo.Label("flux-sandbox", "fastpath"), func() {
	ginkgo.It("FS-FUNC-015: gRPC create returns Running", func() {
		testFastPath_Create(newSpecT("TestFastPath_Create"))
	})
	ginkgo.It("FS-FUNC-016: sandbox CR written asynchronously", func() {
		testFastPath_CRAsyncWrite(newSpecT("TestFastPath_CRAsyncWrite"))
	})
	ginkgo.It("FS-FUNC-018: delete is idempotent", func() {
		testFastPath_Delete_Idempotent(newSpecT("TestFastPath_Delete_Idempotent"))
	})
	ginkgo.It("FS-FUNC-020: capacity-full create is rejected", func() {
		testFastPath_CapacityFull(newSpecT("TestFastPath_CapacityFull"))
	})
})

var _ = ginkgo.Describe("Sandbox Fault Handling", ginkgo.Label("flux-sandbox", "fault"), func() {
	ginkgo.It("FS-FUNC-022: agent pod deletion moves sandbox to Lost (Manual)", func() {
		testSandboxFault_LostManual(newSpecT("TestSandboxFault_LostManual"))
	})
	ginkgo.It("FS-FUNC-023: AutoRecreate re-schedules a lost sandbox", func() {
		testSandboxFault_AutoRecreate(newSpecT("TestSandboxFault_AutoRecreate"))
	})
	ginkgo.It("FS-FUNC-024: orphan detection does not misdelete live sandboxes", func() {
		testSandboxFault_NoOrphanMisdelete(newSpecT("TestSandboxFault_NoOrphanMisdelete"))
	})
})

var _ = ginkgo.Describe("Health Checks and Readiness Gates", ginkgo.Label("flux-sandbox", "health"), func() {
	ginkgo.It("FS-FUNC-028: controller healthz/readyz", func() {
		testHealthz_Controller(newSpecT("TestHealthz_Controller"))
	})
	ginkgo.It("FS-FUNC-029: scheduler healthz/readyz", func() {
		testHealthz_Scheduler(newSpecT("TestHealthz_Scheduler"))
	})
	ginkgo.It("FS-FUNC-030: router stops routing a removed scheduler", ginkgo.Label("destructive"), func() {
		testReadyz_Gate(newSpecT("TestReadyz_Gate"))
	})
})

var _ = ginkgo.Describe("Resource Reservation Isolation", ginkgo.Label("flux-sandbox", "resource"), func() {
	ginkgo.It("FS-FUNC-038: dual-container resource isolation", func() {
		testResource_DualContainerIsolation(newSpecT("TestResource_DualContainerIsolation"))
	})
})

var _ = ginkgo.Describe("Router Load Balancing", ginkgo.Label("flux-sandbox", "router"), func() {
	ginkgo.It("FS-FUNC-025: allocated-min-first routing", func() {
		testRouter_AllocatedMinFirst(newSpecT("TestRouter_AllocatedMinFirst"))
	})
	ginkgo.It("FS-FUNC-026: scheduler isolation", func() {
		testRouter_SchedulerIsolation(newSpecT("TestRouter_SchedulerIsolation"))
	})
	ginkgo.It("FS-FUNC-027: cross-node routing", func() {
		testRouter_CrossNode(newSpecT("TestRouter_CrossNode"))
	})
})

var _ = ginkgo.Describe("containerd Runtime and Agent API", ginkgo.Label("flux-sandbox", "runtime"), func() {
	ginkgo.It("FS-FUNC-031: execd /ping", func() {
		testRuntime_ExecdPing(newSpecT("TestRuntime_ExecdPing"))
	})
	ginkgo.It("FS-FUNC-032: execd /command", func() {
		testRuntime_ExecdCommand(newSpecT("TestRuntime_ExecdCommand"))
	})
	ginkgo.It("FS-FUNC-033: agent /status", func() {
		testAgent_Status(newSpecT("TestAgent_Status"))
	})
	ginkgo.It("FS-FUNC-035: duplicate create is idempotent", func() {
		testAgent_DuplicateCreate_Idempotent(newSpecT("TestAgent_DuplicateCreate_Idempotent"))
	})
})

var _ = ginkgo.Describe("SandboxGroup Lifecycle", ginkgo.Label("flux-sandbox", "sandbox-group"), func() {
	ginkgo.It("FS-FUNC-001: status writeback", func() {
		testSandboxGroup_StatusWriteback(newSpecT("TestSandboxGroup_StatusWriteback"))
	})
	ginkgo.It("FS-FUNC-002: SSG split", func() {
		testSandboxGroup_SSGSplit(newSpecT("TestSandboxGroup_SSGSplit"))
	})
	ginkgo.It("FS-FUNC-003: round-robin scheduler assignment", func() {
		testSandboxGroup_RoundRobinScheduler(newSpecT("TestSandboxGroup_RoundRobinScheduler"))
	})
	ginkgo.It("FS-FUNC-004: per-slot resources inheritance", func() {
		testSandboxGroup_PerSlotResourcesInherit(newSpecT("TestSandboxGroup_PerSlotResourcesInherit"))
	})
	ginkgo.It("FS-FUNC-005: cascade delete", func() {
		testSandboxGroup_CascadeDelete(newSpecT("TestSandboxGroup_CascadeDelete"))
	})
})

var _ = ginkgo.Describe("Large-scale Sharding", ginkgo.Label("flux-sandbox", "sharding"), func() {
	ginkgo.It("FS-FUNC-039: 4 SSGs x 2 agent pods with sharded sandboxes", func() {
		testSharding_LargeScale(newSpecT("TestSharding_LargeScale"))
	})
})
