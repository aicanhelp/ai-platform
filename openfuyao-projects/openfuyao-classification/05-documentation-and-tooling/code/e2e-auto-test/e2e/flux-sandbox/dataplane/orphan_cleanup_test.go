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
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/framework"
)

// ── Feature group: sandbox-watcher orphan cleanup ──
//
// All timings derive from the watcher flags passed at deploy time:
// orphanTimeout (30s, below which a container is never an orphan), grace
// period (E2E_ORPHAN_GRACE_PERIOD, default 30s in e2e vs 3m in production),
// and scan interval (E2E_SCAN_INTERVAL, default 30s). The worst-case wait
// for a cleanup is orphanTimeout + grace + 2*scan.

var _ = ginkgo.Describe("SandboxWatcher", ginkgo.Label("flux-sandbox", "watcher"), func() {

	ginkgo.It("does not clean a CR-less sandbox within the grace period (DP-E2E-031)", func() {
		t := ginkgo.GinkgoT()
		id := sbID(t)
		ginkgo.DeferCleanup(cleanupSandbox, t, id)

		createSandboxNoCR(t, id)
		waitForContainer(t, id, 30*time.Second)

		// Within grace the watcher must not touch the container even though
		// no Sandbox CR exists for it.
		waitForNot(t, "container survives within grace period",
			env.Cfg.NotCleanedWindow(),
			func() bool { return !containerExists(t, id) })
	})

	ginkgo.It("cleans the orphan after the agent pod is gone (fast path, DP-E2E-032)", func() {
		t := ginkgo.GinkgoT()
		id := sbID(t)
		ginkgo.DeferCleanup(cleanupSandbox, t, id)

		createSandboxWithCR(t, id)
		waitForContainer(t, id, 30*time.Second)

		// Deleting the agent pod triggers the watcher fast path: the pod
		// delete event immediately starts grace tracking for containers
		// labeled with that agent uid.
		restartAgentPod(t)

		// After grace (+ scan slack) the container must be cleaned.
		waitFor(t, "orphan container cleaned after agent pod gone",
			env.Cfg.CleanupTimeout(),
			func() bool { return !containerExists(t, id) })

		// The CR still references a dead sandbox: remove it and leave a
		// healthy agent pod behind for the following specs.
		deleteSandboxCR(t, id)
		restartAgentPod(t)
	})

	ginkgo.It("does not clean within grace after the agent pod is gone (DP-E2E-033)", func() {
		t := ginkgo.GinkgoT()
		id := sbID(t)
		ginkgo.DeferCleanup(cleanupSandbox, t, id)

		createSandboxWithCR(t, id)
		waitForContainer(t, id, 30*time.Second)

		restartAgentPod(t)

		// The fast path still honors the grace period.
		waitForNot(t, "container survives within grace after agent pod gone",
			env.Cfg.NotCleanedWindow(),
			func() bool { return !containerExists(t, id) })
	})

	ginkgo.It("CR recovery stops orphan tracking (DP-E2E-034)", func() {
		t := ginkgo.GinkgoT()
		id := sbID(t)
		ginkgo.DeferCleanup(cleanupSandbox, t, id)

		// Start as an orphan: no CR, so once the container is older than
		// orphanTimeout the watcher starts tracking it.
		createSandboxNoCR(t, id)
		waitForContainer(t, id, 30*time.Second)

		// Outlive the orphan timeout, then restore the CR before the grace
		// period can elapse: the watcher must stop tracking the container.
		time.Sleep(framework.OrphanTimeout + 5*time.Second)
		applySandboxCR(t, id)

		waitForNot(t, "container survives after CR recovered",
			env.Cfg.NotCleanedWindow(),
			func() bool { return !containerExists(t, id) })
	})

	// UID mismatch is simulated by overwriting the container label: the
	// watcher sees a stale agent-uid but the agent pod is alive, so the
	// liveness re-check must abandon the cleanup.
	ginkgo.It("does not clean on uid mismatch while the agent is alive (DP-E2E-035)", func() {
		t := ginkgo.GinkgoT()
		id := sbID(t)
		ginkgo.DeferCleanup(cleanupSandbox, t, id)

		createSandboxWithCR(t, id)
		waitForContainer(t, id, 30*time.Second)

		updateContainerLabel(t, id, "flux-sandbox.io/agent-uid", "fake-old-uid-"+id)

		waitForNot(t, "container survives uid mismatch while agent alive",
			env.Cfg.NotCleanedWindow(),
			func() bool { return !containerExists(t, id) })
	})

	ginkgo.It("leaves no runtime residue after cleanup (DP-E2E-036)", ginkgo.Label("needs-ssh"), func() {
		t := ginkgo.GinkgoT()
		skipIfMissing(t, "E2E_NODE_SSH_READY", "needs node-level file access")
		id := sbID(t)
		ginkgo.DeferCleanup(cleanupSandbox, t, id)

		createSandboxNoCR(t, id)
		waitForContainer(t, id, 30*time.Second)

		// containerd v2 keeps per-task state under the runtime task dir; the
		// legacy layout named FIFO files after the container ID.
		taskDir := "/run/containerd/io.containerd.runtime.v2.task/k8s.io/" + id
		legacyFIFO := "/run/containerd/fifo/*" + id + "*"
		waitFor(t, "task dir created", 30*time.Second, func() bool {
			return env.Node.PathExists(taskDir)
		})

		// The watcher eventually cleans the orphan; no residue may survive.
		waitFor(t, "orphan container cleaned (residue check)",
			env.Cfg.CleanupTimeout(),
			func() bool { return !containerExists(t, id) })

		waitFor(t, "task dir removed", 30*time.Second, func() bool {
			return !env.Node.PathExists(taskDir)
		})
		if nodeFileGlobExists(t, legacyFIFO) {
			gomega.Expect(nodeFileGlobExists(t, legacyFIFO)).To(gomega.BeFalse(),
				"legacy FIFO files still present after cleanup")
		}
	})
})
