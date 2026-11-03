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
	"net/http"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// ── Feature group: warm task creation ──

var _ = ginkgo.Describe("Warm Task Creation", ginkgo.Label("flux-sandbox", "create"), func() {

	ginkgo.It("persists a warm task and returns a unique task id (WARM-E2E-002)", func() {
		tb := ginkgo.GinkgoT()
		ids := []string{"snapshot-001", "snapshot-002"}

		// Submit the creation request twice: creation is intentionally NOT
		// idempotent, each submission yields a fresh task.
		resp1, code1, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code1).To(gomega.Equal(http.StatusCreated), "body %s", raw)
		gomega.Expect(resp1.TaskID).To(gomega.HavePrefix("warm-task-"), "body %s", raw)
		gomega.Expect(resp1.State).To(gomega.Equal("WAIT"), "body %s", raw)
		gomega.Expect(resp1.AcceptedSnapshotCt).To(gomega.Equal(2), "body %s", raw)
		_, err := time.Parse(time.RFC3339, resp1.CreatedAt)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "created_at not RFC3339: %s", resp1.CreatedAt)

		resp2, code2, raw2 := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code2).To(gomega.Equal(http.StatusCreated), "body %s", raw2)
		gomega.Expect(resp2.TaskID).NotTo(gomega.Equal(resp1.TaskID), "task ids must be unique")

		// Direct DB check: persisted with count=2 and JSONB order preserved.
		// The row may already be WARMING(1): workers poll every second, so
		// the creation-to-query window can lose that race. WAIT(0) and
		// WARMING(1) both prove persistence; the terminal states (2/3)
		// would not.
		for _, taskID := range []string{resp1.TaskID, resp2.TaskID} {
			row := wEnv.db(tb).taskRow(tb, taskID)
			gomega.Expect(row.State).To(gomega.BeElementOf(int16(0), int16(1)), "task %s state", taskID)
			gomega.Expect(row.TotalSnapshotCount).To(gomega.Equal(int64(2)), "task %s count", taskID)
			gomega.Expect(jsonStrings(tb, row.SnapshotIDs)).To(gomega.Equal(ids),
				"task %s snapshot_ids order must match submission", taskID)
		}

		// 201 only means the task is persisted; wait for execution to finish
		// so the row does not linger in WARMING across specs (both tasks warm
		// the same snapshots; the second hits the idempotent skip path).
		for _, taskID := range []string{resp1.TaskID, resp2.TaskID} {
			wEnv.waitForTaskTerminal(tb, taskID, 2*time.Minute)
		}
	})
})
