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

package dataplane

import (
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	agentapi "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/agentapi"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/framework"
)

// ── Feature group: Agent HTTP API (flux-sandbox-agent) ──

var _ = ginkgo.Describe("Agent HTTP API", ginkgo.Label("flux-sandbox", "agent-api"), func() {

	ginkgo.Describe("healthz", func() {
		ginkgo.It("returns 200 with ok (DP-E2E-001)", func() {
			code, body := 0, ""
			gomega.Eventually(func() error {
				var err error
				code, body, err = env.Agent.Healthz()
				return err
			}, "10s", "1s").Should(gomega.Succeed())

			gomega.Expect(code).To(gomega.Equal(200))
			gomega.Expect(body).To(gomega.ContainSubstring("ok"))
		})
	})

	ginkgo.Describe("create", func() {
		ginkgo.It("rejects malformed JSON with 400 (DP-E2E-002)", func() {
			code, _ := createSandboxRaw(ginkgo.GinkgoT(), "{invalid json")
			gomega.Expect(code).To(gomega.Equal(400))
		})

		ginkgo.It("is idempotent for duplicate ID (DP-E2E-003)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			resp1, code1 := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code1).To(gomega.Equal(200), "first create: %+v", resp1)
			gomega.Expect(resp1.Success).To(gomega.BeTrue())
			gomega.Expect(resp1.SandboxID).To(gomega.Equal(id))

			resp2, code2 := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code2).To(gomega.Equal(200), "second create: %+v", resp2)
			gomega.Expect(resp2.Success).To(gomega.BeTrue())
			// Cache-hit keeps the original creation time.
			gomega.Expect(resp2.CreatedAt).To(gomega.Equal(resp1.CreatedAt))

			// Idempotency must not spawn a duplicate container.
			waitFor(t, "container created", 30*time.Second, func() bool {
				return containerCount(t, id) == 1
			})
			gomega.Expect(containerCount(t, id)).To(gomega.Equal(1))
		})

		ginkgo.DescribeTable("rejects missing required fields with 500 (DP-E2E-004)",
			func(mutate func(spec *agentapi.SandboxSpec), wantMsg string) {
				t := ginkgo.GinkgoT()
				spec := framework.BasicSpec("e2e-missing-field")
				mutate(&spec)
				resp, code := createSandbox(t, spec)
				gomega.Expect(code).To(gomega.Equal(500))
				gomega.Expect(resp.Success).To(gomega.BeFalse())
				gomega.Expect(resp.Message).To(gomega.ContainSubstring(wantMsg))
			},
			ginkgo.Entry("missing image", func(s *agentapi.SandboxSpec) { s.Image = "" }, "image is required"),
			ginkgo.Entry("missing cpu", func(s *agentapi.SandboxSpec) { s.CPU = "" }, "cpu is required"),
			ginkgo.Entry("missing memory", func(s *agentapi.SandboxSpec) { s.Memory = "" }, "memory is required"),
			ginkgo.Entry("empty sandboxId", func(s *agentapi.SandboxSpec) { s.SandboxID = "" }, "sandboxId is required"),
		)

		ginkgo.It("rejects create for a terminating sandbox (DP-E2E-006)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "first create: %+v", resp)
			gomega.Expect(resp.Success).To(gomega.BeTrue())

			delResp, delCode := deleteSandbox(t, id)
			gomega.Expect(delCode).To(gomega.Equal(200))
			gomega.Expect(delResp.Success).To(gomega.BeTrue())

			// The registry entry is terminating while the async delete runs.
			resp, code = createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(500), "re-create during terminating: %+v", resp)
			gomega.Expect(resp.Success).To(gomega.BeFalse())
			gomega.Expect(resp.Message).To(gomega.ContainSubstring("being deleted, retry later"))

			// Leave a clean tail state for the next specs.
			waitForContainerGone(t, id, 30*time.Second)
			waitForSandboxGone(t, id)
		})
	})

	ginkgo.Describe("delete", func() {
		ginkgo.It("returns success for a nonexistent sandbox (DP-E2E-007)", func() {
			resp, code := deleteSandbox(ginkgo.GinkgoT(), "nonexistent-sandbox-id")
			gomega.Expect(code).To(gomega.Equal(200))
			gomega.Expect(resp.Success).To(gomega.BeTrue())
		})

		ginkgo.It("is idempotent when deleting the same sandbox twice (DP-E2E-007)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			del1, code1 := deleteSandbox(t, id)
			gomega.Expect(code1).To(gomega.Equal(200))
			gomega.Expect(del1.Success).To(gomega.BeTrue())

			del2, code2 := deleteSandbox(t, id)
			gomega.Expect(code2).To(gomega.Equal(200))
			gomega.Expect(del2.Success).To(gomega.BeTrue())

			waitForContainerGone(t, id, 30*time.Second)
		})
	})

	ginkgo.Describe("status", func() {
		ginkgo.It("returns complete agent fields (DP-E2E-008)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			resp, code := createSandbox(t, framework.BasicSpec(id))
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitFor(t, "sandbox appears in status", 30*time.Second, func() bool {
				_, ok := sandboxStatus(t, id)
				return ok
			})

			status, code := getStatus(t)
			gomega.Expect(code).To(gomega.Equal(200))
			gomega.Expect(status.FluxSandboxAgentID).To(gomega.Equal(env.AgentPodName))
			gomega.Expect(status.NodeName).To(gomega.Equal(env.AgentNodeName))
			gomega.Expect(status.RuntimeType).To(gomega.Equal("containerd"))
			gomega.Expect(status.Capacity).To(gomega.Equal(10), "capacity from FLUXSANDBOXAGENT_CAPACITY=10")
			gomega.Expect(status.Allocated).To(gomega.BeNumerically(">=", 1))
			gomega.Expect(status.Images).NotTo(gomega.BeEmpty())
			gomega.Expect(status.Images).To(gomega.ContainElement(gomega.ContainSubstring("busybox")))

			s, ok := sandboxStatus(t, id)
			gomega.Expect(ok).To(gomega.BeTrue(), "sandbox %s missing from status", id)
			gomega.Expect(s.Phase).To(gomega.Or(gomega.Equal("running"), gomega.Equal("creating")), "phase: %s", fmtSandbox(s))
			gomega.Expect(s.CreatedAt).To(gomega.BeNumerically(">", 0))
		})
	})

	ginkgo.Describe("logs", func() {
		ginkgo.It("returns container output (DP-E2E-009)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			spec := framework.BasicSpec(id)
			spec.Command = []string{"/bin/sh", "-c", "echo hello-e2e-log; sleep 3600"}
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitFor(t, "logs contain hello-e2e-log", 30*time.Second, func() bool {
				_, body := getLogs(t, id)
				return strings.Contains(body, "hello-e2e-log")
			})
		})

		ginkgo.It("follow streams new lines continuously (DP-E2E-010)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			spec := framework.BasicSpec(id)
			spec.Command = []string{"/bin/sh", "-c", "i=0; while true; do echo line-$i; i=$((i+1)); sleep 1; done"}
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			// Read the follow stream for 5s and count distinct line-N entries.
			code, body := getLogsWithOpts(t, id, true, 0, 5*time.Second)
			gomega.Expect(code).To(gomega.Equal(200))
			count := 0
			for _, l := range strings.Split(body, "\n") {
				if strings.HasPrefix(l, "line-") {
					count++
				}
			}
			gomega.Expect(count).To(gomega.BeNumerically(">=", 2), "follow body: %q", body)
		})

		ginkgo.It("tail limits returned lines (DP-E2E-011)", func() {
			t := ginkgo.GinkgoT()
			id := sbID(t)
			ginkgo.DeferCleanup(cleanupSandbox, t, id)

			spec := framework.BasicSpec(id)
			spec.Command = []string{"/bin/sh", "-c", "for i in $(seq 1 12); do echo tail-line-$i; done; sleep 3600"}
			resp, code := createSandbox(t, spec)
			gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)

			waitFor(t, "12 lines emitted", 30*time.Second, func() bool {
				_, body := getLogs(t, id)
				return strings.Count(body, "tail-line-") >= 12
			})

			code, body := getLogsWithOpts(t, id, false, 5, 5*time.Second)
			gomega.Expect(code).To(gomega.Equal(200))
			nonEmpty := 0
			for _, l := range strings.Split(body, "\n") {
				if l != "" {
					nonEmpty++
				}
			}
			gomega.Expect(nonEmpty).To(gomega.BeNumerically(">", 0), "tail body: %q", body)
			gomega.Expect(nonEmpty).To(gomega.BeNumerically("<=", 5), "tail body: %q", body)
		})

		ginkgo.It("returns log file not found for a nonexistent sandbox (DP-E2E-012)", func() {
			code, body := getLogs(ginkgo.GinkgoT(), "nonexistent-sandbox-id")
			gomega.Expect(code).To(gomega.Equal(200), "HTTP layer must not fail")
			gomega.Expect(body).To(gomega.ContainSubstring("log file not found"))
		})
	})

	ginkgo.Describe("method not allowed", func() {
		ginkgo.DescribeTable("returns 405 for unsupported methods (DP-E2E-013)",
			func(method, path string) {
				resp, err := env.Agent.Do(method, path)
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
				defer resp.Body.Close()
				gomega.Expect(resp.StatusCode).To(gomega.Equal(405))
			},
			ginkgo.Entry("PUT create", "PUT", "/api/v1/fluxagent/create"),
			ginkgo.Entry("PUT delete", "PUT", "/api/v1/fluxagent/delete"),
			ginkgo.Entry("POST status", "POST", "/api/v1/fluxagent/status"),
			ginkgo.Entry("POST logs", "POST", "/api/v1/fluxagent/logs"),
		)
	})
})
