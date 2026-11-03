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
	"fmt"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// ── Feature group: in-progress and mixed progress queries ──

var _ = ginkgo.Describe("Warm Progress Visibility", ginkgo.Label("flux-sandbox", "progress-view"), func() {

	ginkgo.It("shows partial progress while a task is WARMING (WARM-E2E-016)", func() {
		tb := ginkgo.GinkgoT()
		var ids []string
		for i := 1; i <= 20; i++ {
			ids = append(ids, fmt.Sprintf("snapshot-pp-%03d", i))
		}
		wEnv.db(tb).deleteDetails(tb, ids...)

		// 20 snapshots x 2 targets x 2s dispatch delay gives a long WARMING
		// window; poll every second while it runs.
		created, code, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)

		var midResp progressResponse
		captured := false
		waitFor(tb, "task terminal state", 5*time.Minute, func() bool {
			state := wEnv.taskState(tb, created.TaskID)
			if state != 2 && state != 3 {
				// While WARMING, poll progress and capture a mid-flight
				// response that already mixes finished and unfinished
				// snapshots (success > 0 and in_progress > 0).
				progress, _, _ := wEnv.getProgress(tb, created.TaskID, "")
				if progress.Summary.InProgressCount > 0 && progress.Summary.SuccessCount > 0 && !captured {
					midResp = progress
					captured = true
				}
				return false
			}
			return true
		})
		gomega.Expect(captured).To(gomega.BeTrue(),
			"never captured a WARMING-phase response mixing done and in-progress snapshots")

		// The mid-flight response mixes finished and unfinished snapshots.
		var done, undone int
		for _, s := range midResp.Snapshots {
			if s.WarmSuccess {
				done++
				gomega.Expect(s.WarmedSuperPodIDs).To(gomega.HaveLen(2),
					"finished snapshot %s must carry its full warmed ids", s.SnapshotID)
			} else {
				undone++
				// Unfinished snapshots expose their partial warmed ids when
				// present (omitempty: a snapshot not yet started has none).
			}
		}
		gomega.Expect(done).To(gomega.BeNumerically(">", 0), "some snapshots should be done mid-flight")
		gomega.Expect(undone).To(gomega.BeNumerically(">", 0), "some snapshots should still be in progress")
		gomega.Expect(midResp.Summary.Total).To(gomega.Equal(int64(20)))

		// Terminal state: everything succeeded.
		progress, code, raw := wEnv.getProgress(tb, created.TaskID, "")
		gomega.Expect(code).To(gomega.Equal(200), "body %s", raw)
		gomega.Expect(progress.Summary.SuccessCount).To(gomega.Equal(int64(20)))
		gomega.Expect(progress.Summary.InProgressCount).To(gomega.Equal(int64(0)))
		for _, s := range progress.Snapshots {
			gomega.Expect(s.WarmSuccess).To(gomega.BeTrue())
		}
	})

	ginkgo.It("mixes success and failure in one progress response (WARM-E2E-017)", func() {
		tb := ginkgo.GinkgoT()
		ok := "snapshot-mix-001"
		bad := "snapshot-mix-002"
		wEnv.db(tb).deleteDetails(tb, ok, bad)

		// The second snapshot can never be warmed: all targets fail
		// persistently (stands in for a missing S3 object).
		for _, sp := range wSuperPodIDs {
			wEnv.e2bInjectFault(tb, bad+":"+sp, "always")
		}
		ginkgo.DeferCleanup(func() {
			for _, sp := range wSuperPodIDs {
				wEnv.e2bInjectFault(tb, bad+":"+sp, "off")
			}
		})

		created, code, raw := wEnv.createWarmTask(tb, []string{ok, bad})
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, retryWindow*2)).To(gomega.Equal(3),
			"task with one failing snapshot must end FAILED")

		// Both snapshots appear in one array, submission order preserved.
		progress, pcode, praw := wEnv.getProgress(tb, created.TaskID, "")
		gomega.Expect(pcode).To(gomega.Equal(200), "body %s", praw)
		gomega.Expect(progress.Snapshots).To(gomega.HaveLen(2))
		gomega.Expect(progress.Snapshots[0].SnapshotID).To(gomega.Equal(ok))
		gomega.Expect(progress.Snapshots[1].SnapshotID).To(gomega.Equal(bad))
		gomega.Expect(progress.Snapshots[0].WarmSuccess).To(gomega.BeTrue())
		gomega.Expect(progress.Snapshots[0].WarmedSuperPodIDs).To(gomega.HaveLen(2))
		gomega.Expect(progress.Snapshots[1].WarmSuccess).To(gomega.BeFalse())

		// summary partitions the task exactly.
		gomega.Expect(progress.Summary.Total).To(gomega.Equal(int64(2)))
		gomega.Expect(progress.Summary.SuccessCount).To(gomega.Equal(int64(1)))
		gomega.Expect(progress.Summary.FailedCount).To(gomega.Equal(int64(1)))
		gomega.Expect(progress.Summary.InProgressCount).To(gomega.Equal(int64(0)))
	})
})
