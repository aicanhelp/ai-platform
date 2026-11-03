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
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// retryWindow bounds controller-side retrying: 4 attempts x 10s + slack.
var retryWindow = time.Duration(wRetryAttempts)*wRetryInterval + 30*time.Second

// hasDuplicates reports whether the slice contains repeated elements.
func hasDuplicates(s []string) bool {
	return len(s) != len(unique(s))
}

// unique returns the de-duplicated copy of s, preserving order.
func unique(s []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(s))
	for _, v := range s {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out
}

// ── Feature group: warm failure, retry and recovery ──

var _ = ginkgo.Describe("Warm Failure And Retry", ginkgo.Label("flux-sandbox", "retry"), func() {

	ginkgo.It("incrementally warms only missing targets after a partial failure (WARM-E2E-010)", func() {
		tb := ginkgo.GinkgoT()
		snap := "snapshot-inc-001"
		wEnv.db(tb).deleteDetails(tb, snap)
		wEnv.e2bClearRequests(tb)

		// T1: superpod2 dispatches keep failing; the load balancer plans
		// superpod1+superpod2 for a fresh snapshot, so T1 ends FAILED with
		// only superpod1 warmed.
		wEnv.e2bInjectFault(tb, snap+":superpod2", "always")
		t1, code, raw := wEnv.createWarmTask(tb, []string{snap})
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, t1.TaskID, retryWindow*2)).To(gomega.Equal(3),
			"T1 must end FAILED while superpod2 keeps failing")

		row := wEnv.db(tb).detailRowMust(tb, snap)
		gomega.Expect(row.State).To(gomega.Equal(int16(2)), "detail FAILED")
		warmed1 := jsonStrings(tb, row.WarmedSuperPodID)
		gomega.Expect(warmed1).To(gomega.ConsistOf("superpod1"),
			"T1 must warm only the non-failing target, got %v", warmed1)
		baseSp1 := wEnv.e2bDispatchCount(tb, snap+":superpod1")

		// T2: fault cleared - the task succeeds by warming only the missing
		// targets; superpod1 is not re-dispatched.
		wEnv.e2bInjectFault(tb, snap+":superpod2", "off")
		t2, code2, raw2 := wEnv.createWarmTask(tb, []string{snap})
		gomega.Expect(code2).To(gomega.Equal(201), "body %s", raw2)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, t2.TaskID, 2*time.Minute)).To(gomega.Equal(2),
			"T2 must succeed after the fault is cleared")

		gomega.Expect(wEnv.e2bDispatchCount(tb, snap+":superpod1")).To(gomega.Equal(baseSp1),
			"already-warmed superpod1 must not be re-dispatched in T2")

		// Final detail: SUCCESS, no duplicates. The warmed set is the union
		// of T1's succeeded target (superpod1) and T2's incremental plan
		// (the missing targets), so it may legitimately hold 2 or 3 pods.
		row = wEnv.db(tb).detailRowMust(tb, snap)
		gomega.Expect(row.State).To(gomega.Equal(int16(1)))
		final := jsonStrings(tb, row.WarmedSuperPodID)
		gomega.Expect(final).To(gomega.ContainElement("superpod1"),
			"T1's succeeded target must survive in the union")
		gomega.Expect(hasDuplicates(final)).To(gomega.BeFalse(), "warmed set must not repeat: %v", final)
		for _, sp := range final {
			gomega.Expect(wSuperPodIDs).To(gomega.ContainElement(sp), "unexpected target %s", sp)
		}
		gomega.Expect(final).To(gomega.HaveLen(len(unique(final))), "warmed set has duplicates: %v", final)
	})

	ginkgo.It("retries a single failed target without disturbing others (WARM-E2E-011)", func() {
		tb := ginkgo.GinkgoT()
		snap := "snapshot-retry-001"
		wEnv.db(tb).deleteDetails(tb, snap)
		wEnv.e2bClearRequests(tb)

		// One-shot fault: the first dispatch to superpod1 fails, the retry
		// auto-recovers within the retry window.
		wEnv.e2bInjectFault(tb, snap+":superpod1", "once")

		created, code, raw := wEnv.createWarmTask(tb, []string{snap})
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, retryWindow*2)).To(gomega.Equal(2),
			"task must succeed after the one-shot fault recovers")

		// superpod1 was dispatched twice (first attempt failed, retry
		// succeeded); superpod2 exactly once (not retried, not skipped).
		sp1 := wEnv.e2bDispatchCount(tb, snap+":superpod1")
		sp2 := wEnv.e2bDispatchCount(tb, snap+":superpod2")
		gomega.Expect(sp1).To(gomega.Equal(2), "superpod1: failed once + retry, got %d dispatches", sp1)
		gomega.Expect(sp2).To(gomega.Equal(1), "superpod2 must be dispatched exactly once")

		// Both targets end up warmed.
		row := wEnv.db(tb).detailRowMust(tb, snap)
		gomega.Expect(row.State).To(gomega.Equal(int16(1)))
		warmed := jsonStrings(tb, row.WarmedSuperPodID)
		gomega.Expect(warmed).To(gomega.ConsistOf("superpod1", "superpod2"))
	})

	ginkgo.It("fails the task when the snapshot can never be warmed (WARM-E2E-012)", func() {
		tb := ginkgo.GinkgoT()
		snap := "snapshot-miss-001"
		wEnv.db(tb).deleteDetails(tb, snap)

		// Simulate an S3-missing snapshot: every target fails persistently
		// (the fake E2B accepts any snapshot, so fault injection stands in
		// for the inevitable pull failure).
		for _, sp := range wSuperPodIDs {
			wEnv.e2bInjectFault(tb, snap+":"+sp, "always")
		}
		ginkgo.DeferCleanup(func() {
			wEnv.e2bInjectFault(tb, snap+":superpod1", "off")
			wEnv.e2bInjectFault(tb, snap+":superpod2", "off")
			wEnv.e2bInjectFault(tb, snap+":superpod3", "off")
		})

		created, code, raw := wEnv.createWarmTask(tb, []string{snap})
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)

		// Task-level FAILED is 3 (snapshot-level FAILED is 2 - distinct
		// enums that must be asserted separately).
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, retryWindow*2)).To(gomega.Equal(3),
			"task must end FAILED after retries are exhausted")

		row := wEnv.db(tb).detailRowMust(tb, snap)
		gomega.Expect(row.State).To(gomega.Equal(int16(2)), "snapshot-level FAILED enum is 2")
		gomega.Expect(jsonStrings(tb, row.WarmedSuperPodID)).To(gomega.BeEmpty(),
			"nothing warmed for a missing snapshot")

		progress, pcode, praw := wEnv.getProgress(tb, created.TaskID, "")
		gomega.Expect(pcode).To(gomega.Equal(200), "body %s", praw)
		gomega.Expect(progress.Summary.Total).To(gomega.Equal(int64(1)))
		gomega.Expect(progress.Summary.FailedCount).To(gomega.Equal(int64(1)))
		gomega.Expect(progress.Summary.SuccessCount).To(gomega.Equal(int64(0)))
		gomega.Expect(progress.Snapshots[0].WarmSuccess).To(gomega.BeFalse())
	})
})
