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
	"net/http"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// ── Feature group: warm progress query ──

var _ = ginkgo.Describe("Warm Progress Query", ginkgo.Label("flux-sandbox", "progress"), func() {

	ginkgo.It("returns task progress summary and pagination (WARM-E2E-004)", func() {
		tb := ginkgo.GinkgoT()
		ids := []string{"snapshot-p-001", "snapshot-p-002"}

		// Build a SUCCEED task of two snapshots (fresh detail rows so the
		// warm really happens rather than being skipped).
		wEnv.db(tb).deleteDetails(tb, ids...)
		created, code, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code).To(gomega.Equal(http.StatusCreated), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, 2*time.Minute)).To(gomega.Equal(2))

		// Step 1: query the finished task without pagination.
		progress, code, raw := wEnv.getProgress(tb, created.TaskID, "")
		gomega.Expect(code).To(gomega.Equal(http.StatusOK), "body %s", raw)
		gomega.Expect(progress.Summary.Total).To(gomega.Equal(int64(2)))
		gomega.Expect(progress.Summary.SuccessCount).To(gomega.Equal(int64(2)))
		gomega.Expect(progress.Summary.FailedCount).To(gomega.Equal(int64(0)))
		gomega.Expect(progress.Summary.InProgressCount).To(gomega.Equal(int64(0)))
		gomega.Expect(progress.Snapshots).To(gomega.HaveLen(2))
		// Snapshot order matches the submission order.
		gomega.Expect(progress.Snapshots[0].SnapshotID).To(gomega.Equal(ids[0]))
		gomega.Expect(progress.Snapshots[1].SnapshotID).To(gomega.Equal(ids[1]))
		for _, s := range progress.Snapshots {
			gomega.Expect(s.WarmSuccess).To(gomega.BeTrue())
			gomega.Expect(s.WarmedSuperPodIDs).To(gomega.HaveLen(2))
		}

		// Step 2: create a single-snapshot task and query it immediately -
		// with a 2s dispatch delay the task is still in flight, so the
		// snapshot shows as in progress.
		wEnv.db(tb).deleteDetails(tb, "snapshot-p-003")
		created2, code2, raw2 := wEnv.createWarmTask(tb, []string{"snapshot-p-003"})
		gomega.Expect(code2).To(gomega.Equal(http.StatusCreated), "body %s", raw2)
		progress2, code3, raw3 := wEnv.getProgress(tb, created2.TaskID, "")
		gomega.Expect(code3).To(gomega.Equal(http.StatusOK), "body %s", raw3)
		gomega.Expect(progress2.Summary.Total).To(gomega.Equal(int64(1)))
		gomega.Expect(progress2.Snapshots).To(gomega.HaveLen(1))
		gomega.Expect(progress2.Snapshots[0].SnapshotID).To(gomega.Equal("snapshot-p-003"))
		if wEnv.taskState(tb, created2.TaskID) != 2 {
			gomega.Expect(progress2.Summary.InProgressCount).To(gomega.Equal(int64(1)), "body %s", raw3)
			gomega.Expect(progress2.Snapshots[0].WarmSuccess).To(gomega.BeFalse())
		}
		// Let it finish so no lingering task affects later specs.
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created2.TaskID, 2*time.Minute)).To(gomega.Equal(2))

		// Step 3: paginate the finished task with page_size=1 - page 1 and
		// page 2 return the two snapshots in submission order, no overlap.
		p1, c1, r1 := wEnv.getProgress(tb, created.TaskID, "page_size=1&page_num=1")
		gomega.Expect(c1).To(gomega.Equal(http.StatusOK), "body %s", r1)
		gomega.Expect(p1.Snapshots).To(gomega.HaveLen(1))
		gomega.Expect(p1.Snapshots[0].SnapshotID).To(gomega.Equal(ids[0]))

		p2, c2, r2 := wEnv.getProgress(tb, created.TaskID, "page_size=1&page_num=2")
		gomega.Expect(c2).To(gomega.Equal(http.StatusOK), "body %s", r2)
		gomega.Expect(p2.Snapshots).To(gomega.HaveLen(1))
		gomega.Expect(p2.Snapshots[0].SnapshotID).To(gomega.Equal(ids[1]))
	})
})
