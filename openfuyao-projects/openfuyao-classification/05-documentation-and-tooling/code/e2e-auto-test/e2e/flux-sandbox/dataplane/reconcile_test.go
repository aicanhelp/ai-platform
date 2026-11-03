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
	"fmt"
	"sync"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/framework"
)

// ── Feature group: Reconcile after agent restart, and concurrency ──
//
// These specs delete the agent pod; ginkgo runs them sequentially, so they
// must always leave a healthy agent behind.

var _ = ginkgo.Describe("Reconcile", ginkgo.Label("flux-sandbox", "reconcile"), func() {

	ginkgo.It("re-admits the sandbox into status after agent restart (DP-E2E-026)", func() {
		t := ginkgo.GinkgoT()
		id := sbID(t)
		ginkgo.DeferCleanup(cleanupSandbox, t, id)

		createRestartAndWaitReconcile(t, id)
	})

	ginkgo.It("idempotent create after restart is a cache-hit (DP-E2E-027)", func() {
		t := ginkgo.GinkgoT()
		id := sbID(t)
		ginkgo.DeferCleanup(cleanupSandbox, t, id)

		createRestartAndWaitReconcile(t, id)

		resp2, code2 := createSandbox(t, framework.BasicSpec(id))
		gomega.Expect(code2).To(gomega.Equal(200), "re-create after restart: %+v", resp2)
		gomega.Expect(resp2.Success).To(gomega.BeTrue())
		// No duplicate container.
		gomega.Expect(containerCount(t, id)).To(gomega.Equal(1))
	})

	ginkgo.It("delete after restart goes through the cache-miss path (DP-E2E-028)", func() {
		t := ginkgo.GinkgoT()
		id := sbID(t)
		ginkgo.DeferCleanup(forceDeleteContainer, id)

		resp, code := createSandbox(t, framework.BasicSpec(id))
		gomega.Expect(code).To(gomega.Equal(200), "create: %+v", resp)
		waitForContainer(t, id, 30*time.Second)

		restartAgentPod(t)

		// The new agent registry is rebuilt by reconcile; the delete hits the
		// runtime directly and still succeeds.
		delResp, delCode := deleteSandbox(t, id)
		gomega.Expect(delCode).To(gomega.Equal(200))
		gomega.Expect(delResp.Success).To(gomega.BeTrue())

		waitForContainerGone(t, id, 30*time.Second)
	})

	ginkgo.It("does not oversell capacity after restart (DP-E2E-029)", ginkgo.Label("needs-small-capacity"), func() {
		t := ginkgo.GinkgoT()
		smallAgent := smallCapacityAgent()
		if smallAgent == nil {
			ginkgo.Skip("skipped: set E2E_SMALL_CAPACITY_AGENT_URL to enable (capacity=2 agent)")
		}
		const capacity = 2
		ids := createOnAgent(t, smallAgent, capacity)

		restartAgentPod(t)

		// After restart the reconciled registry must account for both
		// sandboxes, so a third create must be rejected.
		waitFor(t, "both sandboxes reconciled after restart", 60*time.Second, func() bool {
			status, _, err := smallAgent.Status()
			if err != nil {
				return false
			}
			count := 0
			for _, s := range status.SandboxStatuses {
				for _, id := range ids {
					if s.SandboxID == id {
						count++
					}
				}
			}
			return count == capacity
		})

		extraID := framework.NewID(t.Name()) + "-extra"
		resp, code, err := smallAgent.Create(framework.BasicSpec(extraID))
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(code).To(gomega.Equal(500), "over-capacity create after restart: %+v", resp)
		gomega.Expect(resp.Success).To(gomega.BeFalse())
		gomega.Expect(resp.Message).To(gomega.ContainSubstring("capacity exceeded"))
	})
})

var _ = ginkgo.Describe("Concurrency", ginkgo.Label("flux-sandbox", "reconcile"), func() {

	ginkgo.It("concurrent create and delete have no races (DP-E2E-030)", func() {
		t := ginkgo.GinkgoT()
		const n = 10
		var ids []string
		for i := 0; i < n; i++ {
			ids = append(ids, fmt.Sprintf("%s-%d", sbID(t), i))
		}
		ginkgo.DeferCleanup(func() {
			for _, id := range ids {
				_, _, _ = env.Agent.Delete(id)
				_ = env.K8s.DeleteSandboxCR(id)
				forceDeleteContainer(id)
			}
		})
		for _, id := range ids {
			forceDeleteContainer(id)
		}

		// Concurrent creates: all must succeed.
		var wg sync.WaitGroup
		createErrs := make(chan error, n)
		for _, id := range ids {
			wg.Add(1)
			go func(id string) {
				defer wg.Done()
				resp, code, err := env.Agent.Create(framework.BasicSpec(id))
				if err != nil {
					createErrs <- err
					return
				}
				if code != 200 || !resp.Success {
					createErrs <- fmt.Errorf("create %s: code=%d resp=%+v", id, code, resp)
				}
			}(id)
		}
		wg.Wait()
		close(createErrs)
		for err := range createErrs {
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}

		for _, id := range ids {
			waitForContainer(t, id, 30*time.Second)
		}

		// Concurrent deletes: all must succeed.
		deleteErrs := make(chan error, n)
		for _, id := range ids {
			wg.Add(1)
			go func(id string) {
				defer wg.Done()
				resp, code, err := env.Agent.Delete(id)
				if err != nil {
					deleteErrs <- err
					return
				}
				if code != 200 || !resp.Success {
					deleteErrs <- fmt.Errorf("delete %s: code=%d resp=%+v", id, code, resp)
				}
			}(id)
		}
		wg.Wait()
		close(deleteErrs)
		for err := range deleteErrs {
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}

		for _, id := range ids {
			waitForContainerGone(t, id, 30*time.Second)
		}
	})
})
