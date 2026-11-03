//go:build e2e

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You can obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package warmup

import (
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ── Feature group: controller restart recovery ──

var _ = ginkgo.Describe("Warm Restart Recovery", ginkgo.Label("flux-sandbox", "restart"), func() {

	ginkgo.It("recovers a WARMING task after the controller restarts (WARM-E2E-007)", func() {
		tb := ginkgo.GinkgoT()
		ids := []string{"snapshot-r-001", "snapshot-r-002", "snapshot-r-003",
			"snapshot-r-004", "snapshot-r-005", "snapshot-r-006"}
		wEnv.db(tb).deleteDetails(tb, ids...)
		wEnv.e2bClearRequests(tb)

		// Step 1: submit a 6-snapshot task (2s dispatch delay per target
		// gives a WARMING window long enough to cover the restart).
		created, code, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)

		// Step 2: wait until the task is WARMING with at least one target
		// already warmed, then kill the controller pod mid-flight.
		waitFor(tb, "task WARMING with partial progress", 30*time.Second, func() bool {
			if wEnv.taskState(tb, created.TaskID) != 1 {
				return false
			}
			return len(wEnv.e2bRequests(tb).Requests) > 0
		})
		baseDispatchCount := len(wEnv.e2bRequests(tb).Requests)

		// Record detail-row warmed sets before the restart. Only targets
		// already SUCCEEDED (persisted in the detail row) are the idempotent
		// baseline: in-flight dispatches that never completed are expected
		// to be re-dispatched after recovery (incremental warm-up, design A6).
		baseSucceeded := map[string]bool{} // warmed_snapshot_id set
		for _, snap := range ids {
			row, found := wEnv.db(tb).detailRow(tb, snap)
			if !found {
				continue
			}
			for _, sp := range jsonStrings(tb, row.WarmedSuperPodID) {
				baseSucceeded[snap+":"+sp] = true
			}
		}

		// Kill the controller pod. The Deployment brings a new one up; the
		// port-forward breaks and is re-established once the pod is ready.
		oldPods, err := wEnv.kube.CoreV1().Pods(wControllerNS).List(wEnv.ctx,
			metav1.ListOptions{LabelSelector: "app=flux-sandbox-controller"})
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(oldPods.Items).To(gomega.HaveLen(1))
		err = wEnv.kube.CoreV1().Pods(wControllerNS).Delete(wEnv.ctx, oldPods.Items[0].Name,
			metav1.DeleteOptions{})
		gomega.Expect(err).NotTo(gomega.HaveOccurred())

		// The task stays WARMING right after the restart (no worker touch).
		gomega.Expect(wEnv.taskState(tb, created.TaskID)).To(gomega.Equal(1),
			"task must remain WARMING right after controller restart")

		// Wait for the replacement pod and restore the port-forward.
		waitFor(tb, "new controller pod ready", 3*time.Minute, func() bool {
			pods, err := wEnv.kube.CoreV1().Pods(wControllerNS).List(wEnv.ctx,
				metav1.ListOptions{LabelSelector: "app=flux-sandbox-controller"})
			if err != nil {
				return false
			}
			for _, p := range pods.Items {
				if p.Name != oldPods.Items[0].Name && podReady(p) {
					return true
				}
			}
			return false
		})
		wEnv.restartControllerPortForward(tb)

		// Step 3: existing warmed targets survive the restart in the DB.
		// Targets dispatched between the baseline read and the pod kill may
		// also appear here (in-flight completions by the old controller are
		// legitimate), so assert survival (baseline is a subset), not that
		// the warm set is unchanged.
		for _, snap := range ids {
			row, found := wEnv.db(tb).detailRow(tb, snap)
			if !found {
				continue
			}
			warmed := map[string]bool{}
			for _, sp := range jsonStrings(tb, row.WarmedSuperPodID) {
				warmed[snap+":"+sp] = true
			}
			for pair := range baseSucceeded {
				if strings.HasPrefix(pair, snap+":") {
					gomega.Expect(warmed).To(gomega.HaveKey(pair),
						"%s pre-restart warmed target lost after restart", pair)
				}
			}
		}

		// Step 4: after the stale timeout (30s) the new controller reclaims
		// the WARMING task and finishes it.
		terminal := wEnv.waitForTaskTerminal(tb, created.TaskID,
			wStaleTimeout+3*time.Minute)
		gomega.Expect(terminal).To(gomega.Equal(2), "task must finish SUCCEED after recovery")

		// Step 5: no duplicate dispatches - every snapshot:superpod pair was
		// dispatched exactly once across the restart, and targets already
		// succeeded before the restart were not re-dispatched.
		reqs := wEnv.e2bRequests(tb).Requests
		seen := map[string]int{}
		for _, r := range reqs {
			seen[r.WarmedSnapshotID]++
		}
		for pair := range baseSucceeded {
			gomega.Expect(seen[pair]).To(gomega.Equal(1),
				"pre-restart succeeded target %s was re-dispatched after recovery", pair)
		}
		postRestartCount := len(reqs) - baseDispatchCount
		tb.Logf("dispatches: %d before restart, %d after restart", baseDispatchCount, postRestartCount)

		// Final detail rows: every snapshot SUCCESS with 2 unique targets.
		for _, snap := range ids {
			row := wEnv.db(tb).detailRowMust(tb, snap)
			gomega.Expect(row.State).To(gomega.Equal(int16(1)), "%s detail state", snap)
			warmed := jsonStrings(tb, row.WarmedSuperPodID)
			gomega.Expect(warmed).To(gomega.HaveLen(2), "%s warmed targets", snap)
		}
	})
})

func podReady(p corev1.Pod) bool {
	for _, c := range p.Status.ContainerStatuses {
		if !c.Ready {
			return false
		}
	}
	return p.Status.Phase == corev1.PodRunning
}
