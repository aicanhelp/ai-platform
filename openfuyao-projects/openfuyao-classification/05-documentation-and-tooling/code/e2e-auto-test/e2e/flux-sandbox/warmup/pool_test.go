//go:build e2e

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You can obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/Mulan PSL v2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package warmup

import (
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// ── Feature group: available pool exhaustion ──

var _ = ginkgo.Describe("Warm Pool Exhaustion", ginkgo.Label("flux-sandbox", "pool"), func() {

	ginkgo.It("fails the task when too few SuperPods are available (WARM-E2E-013)", func() {
		tb := ginkgo.GinkgoT()
		snap := "snapshot-short-001"
		wEnv.db(tb).deleteDetails(tb, snap)

		// Backdate superpod2 and superpod3 reports by 10 minutes: their
		// etcd records go stale past the 10s staleness threshold, so the
		// available pool shrinks to superpod1 only - below the configured
		// target count of 2.
		ginkgo.DeferCleanup(func() {
			wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod2", "backdate_ms": 0})
			wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod3", "backdate_ms": 0})
		})
		wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod2", "backdate_ms": 600000})
		wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod3", "backdate_ms": 600000})

		// Wait until the storage view marks them UNAVAILABLE.
		waitFor(tb, "stale SuperPods marked UNAVAILABLE", 30*time.Second, func() bool {
			details, _, _ := wEnv.getStorageDetails(tb)
			return details["superpod2"].Health == "UNAVAILABLE" && details["superpod3"].Health == "UNAVAILABLE"
		})

		// Submit: target selection needs 2 pods but the pool has 1; the
		// controller retries pool selection (4 rounds x 10s) then fails.
		created, code, raw := wEnv.createWarmTask(tb, []string{snap})
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, retryWindow*2)).To(gomega.Equal(3),
			"task must fail when the available pool is smaller than the target count")

		row := wEnv.db(tb).detailRowMust(tb, snap)
		gomega.Expect(row.State).To(gomega.Equal(int16(2)))
		// No fallback to warming ALL SuperPods: nothing got warmed.
		gomega.Expect(jsonStrings(tb, row.WarmedSuperPodID)).To(gomega.BeEmpty(),
			"must not fall back to warming every SuperPod")

		// Restore fresh reports: the SuperPods return to READY (non-sticky
		// health, records recover once fresh data arrives).
		wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod2", "backdate_ms": 0})
		wEnv.e2bAdmin(tb, "POST", "/_admin/report", map[string]any{"superpod": "superpod3", "backdate_ms": 0})
		waitFor(tb, "SuperPods recover to READY", 30*time.Second, func() bool {
			details, _, _ := wEnv.getStorageDetails(tb)
			return details["superpod2"].Health == "READY" && details["superpod3"].Health == "READY"
		})
	})
})
