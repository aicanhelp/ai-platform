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

// ── Feature group: pagination boundaries and ordering ──

var _ = ginkgo.Describe("Warm Pagination", ginkgo.Label("flux-sandbox", "pagination"), func() {

	ginkgo.It("validates progress pagination boundaries (WARM-E2E-018)", func() {
		tb := ginkgo.GinkgoT()
		var ids []string
		for i := 1; i <= 5; i++ {
			ids = append(ids, fmt.Sprintf("snapshot-pg-%03d", i))
		}
		wEnv.db(tb).deleteDetails(tb, ids...)

		// Build a 5-snapshot SUCCEED task.
		created, code, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, 2*time.Minute)).To(gomega.Equal(2))

		// page_size within [1, 10000] returns all snapshots in one page.
		for _, ps := range []string{"5", "10000"} {
			p, c, r := wEnv.getProgress(tb, created.TaskID, "page_size="+ps+"&page_num=1")
			gomega.Expect(c).To(gomega.Equal(200), "page_size=%s: %s", ps, r)
			gomega.Expect(p.Snapshots).To(gomega.HaveLen(5), "page_size=%s", ps)
		}

		// page_size outside [1, 10000] -> 400 with the documented message.
		for _, ps := range []string{"10001", "0", "-1"} {
			_, c, r := wEnv.getProgress(tb, created.TaskID, "page_size="+ps+"&page_num=1")
			gomega.Expect(c).To(gomega.Equal(400), "page_size=%s: %s", ps, r)
			gomega.Expect(decodeError(tb, r).Message).To(gomega.Equal("page_size must be between 1 and 10000"))
		}

		// page_num < 1 -> 400 with the documented message.
		for _, pn := range []string{"0", "-1"} {
			_, c, r := wEnv.getProgress(tb, created.TaskID, "page_size=5&page_num="+pn)
			gomega.Expect(c).To(gomega.Equal(400), "page_num=%s: %s", pn, r)
			gomega.Expect(decodeError(tb, r).Message).To(gomega.Equal("page_num must be >= 1"))
		}

		// Walk pages of 2: no overlap, no gap, submission order preserved,
		// and the out-of-range page 4 is an empty 200 (not 404).
		var collected []string
		for pn := 1; pn <= 4; pn++ {
			p, c, r := wEnv.getProgress(tb, created.TaskID, fmt.Sprintf("page_size=2&page_num=%d", pn))
			gomega.Expect(c).To(gomega.Equal(200), "page %d: %s", pn, r)
			for _, s := range p.Snapshots {
				collected = append(collected, s.SnapshotID)
			}
			if pn < 3 {
				gomega.Expect(p.Snapshots).To(gomega.HaveLen(2), "page %d", pn)
			} else {
				want := 5 - 2*(pn-1)
				if want < 0 {
					want = 0
				}
				gomega.Expect(p.Snapshots).To(gomega.HaveLen(want), "page %d", pn)
			}
		}
		gomega.Expect(collected).To(gomega.Equal(ids), "paged walk must cover all snapshots in order")

		// No params: default page_size=1000 returns everything at once.
		p, c, r := wEnv.getProgress(tb, created.TaskID, "")
		gomega.Expect(c).To(gomega.Equal(200), "body %s", r)
		gomega.Expect(p.Snapshots).To(gomega.HaveLen(5))
	})

	ginkgo.It("orders cache details by updated_at desc and paginates consistently (WARM-E2E-019)", func() {
		tb := ginkgo.GinkgoT()
		ord := []string{"snapshot-ord-001", "snapshot-ord-002", "snapshot-ord-003"}

		// Warm the three snapshots strictly one after another so their
		// updated_at order is 003 newest, 001 oldest.
		wEnv.db(tb).deleteDetails(tb, ord...)
		for _, snap := range ord {
			created, code, raw := wEnv.createWarmTask(tb, []string{snap})
			gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
			gomega.Expect(wEnv.waitForTaskTerminal(tb, created.TaskID, 2*time.Minute)).To(gomega.Equal(2))
		}

		// The snapshots of this spec, looked up by name in a response page.
		snapshotsIn := func(resp cacheDetailsResponse) []string {
			var found []string
			for _, snaps := range resp {
				for _, s := range snaps {
					for _, want := range ord {
						if s.SnapshotID == want {
							found = appendIfMissing(found, want)
						}
					}
				}
			}
			return found
		}

		// Page 1 of size 1: the newest snapshot (003).
		p1, c1, r1 := wEnv.getCacheDetails(tb, "page_size=1&page_num=1")
		gomega.Expect(c1).To(gomega.Equal(200), "body %s", r1)
		gomega.Expect(snapshotsIn(p1)).To(gomega.Equal([]string{"snapshot-ord-003"}),
			"page 1 must hold only the newest snapshot: %v", snapshotsIn(p1))

		// Walk pages 2..4: 002, 001, then empty object.
		p2, c2, r2 := wEnv.getCacheDetails(tb, "page_size=1&page_num=2")
		gomega.Expect(c2).To(gomega.Equal(200), "body %s", r2)
		gomega.Expect(snapshotsIn(p2)).To(gomega.Equal([]string{"snapshot-ord-002"}))

		p3, c3, r3 := wEnv.getCacheDetails(tb, "page_size=1&page_num=3")
		gomega.Expect(c3).To(gomega.Equal(200), "body %s", r3)
		gomega.Expect(snapshotsIn(p3)).To(gomega.Equal([]string{"snapshot-ord-001"}))

		p4, c4, r4 := wEnv.getCacheDetails(tb, "page_size=1&page_num=4")
		gomega.Expect(c4).To(gomega.Equal(200), "body %s", r4)
		// Other suites' snapshots may occupy this page (shared detail
		// table); the point is none of THIS spec's snapshots reappears.
		gomega.Expect(snapshotsIn(p4)).To(gomega.BeEmpty(),
			"page 4 must not re-show this spec's snapshots: %v", snapshotsIn(p4))

		// Parameter validation matches the progress API.
		_, cBad, rBad := wEnv.getCacheDetails(tb, "page_size=10001")
		gomega.Expect(cBad).To(gomega.Equal(400), "body %s", rBad)
		_, cBad2, rBad2 := wEnv.getCacheDetails(tb, "page_num=0")
		gomega.Expect(cBad2).To(gomega.Equal(400), "body %s", rBad2)
	})
})

// appendIfMissing appends v when not already present (keeps order, dedups).
func appendIfMissing(s []string, v string) []string {
	for _, e := range s {
		if e == v {
			return s
		}
	}
	return append(s, v)
}
