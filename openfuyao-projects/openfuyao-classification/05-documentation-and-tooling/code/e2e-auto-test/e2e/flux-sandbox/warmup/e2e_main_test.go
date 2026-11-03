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
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// progressResponse decodes GET /api/v1/snapshot-warm-tasks/{task_id}.
type progressResponse struct {
	Summary struct {
		Total           int64 `json:"total"`
		SuccessCount    int64 `json:"success_count"`
		FailedCount     int64 `json:"failed_count"`
		InProgressCount int64 `json:"in_progress_count"`
	} `json:"summary"`
	Snapshots []struct {
		SnapshotID        string   `json:"snapshot_id"`
		WarmSuccess       bool     `json:"warm_success"`
		WarmedSuperPodIDs []string `json:"warmed_superpod_ids"`
	} `json:"snapshots"`
}

func (e *warmEnv) getProgress(tb ginkgo.GinkgoTInterface, taskID, query string) (progressResponse, int, []byte) {
	tb.Helper()
	path := fmt.Sprintf("/api/v1/snapshot-warm-tasks/%s", taskID)
	if query != "" {
		path += "?" + query
	}
	resp, raw := e.httpDo(tb, http.MethodGet, path, nil)
	var out progressResponse
	if resp.StatusCode == http.StatusOK {
		if err := json.Unmarshal(raw, &out); err != nil {
			tb.Fatalf("decode progress response %s: %v", raw, err)
		}
	}
	return out, resp.StatusCode, raw
}

// ── Feature group: regular snapshot warm end-to-end ──

var _ = ginkgo.Describe("Warm End-to-End", ginkgo.Label("flux-sandbox", "e2e-main"), func() {

	ginkgo.It("warms regular snapshots end to end (WARM-E2E-003)", func() {
		tb := ginkgo.GinkgoT()
		ids := []string{"snapshot-e2e-001", "snapshot-e2e-002"}

		// Reset cache state for these snapshots so the task really warms
		// them (a re-run would otherwise hit the idempotent skip path).
		wEnv.db(tb).deleteDetails(tb, ids...)

		// Clear the dispatch log so counts reflect only this task.
		wEnv.e2bClearRequests(tb)

		// Step 1: submit the task.
		created, code, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code).To(gomega.Equal(http.StatusCreated), "body %s", raw)
		gomega.Expect(created.State).To(gomega.Equal("WAIT"))
		gomega.Expect(created.AcceptedSnapshotCt).To(gomega.Equal(2))

		// Step 2: poll DB state at high frequency; the task must pass
		// through WARMING(1) to SUCCEED(2).
		observed := wEnv.observeTaskStates(tb, created.TaskID, 2*time.Minute)
		gomega.Expect(observed[2]).To(gomega.BeTrue(), "task must reach SUCCEED(2), observed: %v", observed)
		gomega.Expect(observed[1]).To(gomega.BeTrue(), "task must pass through WARMING(1), observed states: %v", observed)
		gomega.Expect(observed[3]).To(gomega.BeFalse(), "task must not fail, observed: %v", observed)

		// Step 3: detail rows - SUCCESS, planned == warmed, 2 distinct pods.
		for _, snap := range ids {
			row := wEnv.db(tb).detailRowMust(tb, snap)
			gomega.Expect(row.State).To(gomega.Equal(int16(1)), "%s detail state", snap)
			planned := jsonStrings(tb, row.SuperPodID)
			warmed := jsonStrings(tb, row.WarmedSuperPodID)
			gomega.Expect(planned).To(gomega.HaveLen(2), "%s planned targets", snap)
			gomega.Expect(warmed).To(gomega.ConsistOf(planned), "%s warmed must match planned", snap)
		}

		// Step 4: E2B dispatch log - 4 dispatches (2 snapshots x 2 targets),
		// each warmed_snapshot_id formatted "snapshot:superpod", each request
		// targeting exactly one SuperPod.
		reqs := wEnv.e2bRequests(tb).Requests
		gomega.Expect(reqs).To(gomega.HaveLen(4), "dispatch log: %+v", reqs)
		bySnap := map[string]int{}
		for _, r := range reqs {
			gomega.Expect(r.WarmedSnapshotID).To(gomega.Equal(r.SnapshotID+":"+r.TargetSuperPodID),
				"warmed_snapshot_id must be snapshot:superpod, got %+v", r)
			bySnap[r.SnapshotID]++
		}
		gomega.Expect(bySnap).To(gomega.HaveLen(2))
		for _, snap := range ids {
			gomega.Expect(bySnap[snap]).To(gomega.Equal(2), "%s dispatch count", snap)
		}
		// In the fake stack the fake E2B stands in for MoonCakeStore+S3: a
		// dispatch IS the pull, so step 5 (store pulls from S3) is covered by
		// the same 4 dispatch records.

		// Step 6: progress API - both snapshots warm_success=true.
		progress, code, raw := wEnv.getProgress(tb, created.TaskID, "")
		gomega.Expect(code).To(gomega.Equal(http.StatusOK), "body %s", raw)
		gomega.Expect(progress.Summary.SuccessCount).To(gomega.Equal(int64(2)))
		gomega.Expect(progress.Snapshots).To(gomega.HaveLen(2))
		for _, s := range progress.Snapshots {
			gomega.Expect(s.WarmSuccess).To(gomega.BeTrue(), "snapshot %s", s.SnapshotID)
			gomega.Expect(s.WarmedSuperPodIDs).To(gomega.HaveLen(2), "snapshot %s", s.SnapshotID)
		}
	})
})
