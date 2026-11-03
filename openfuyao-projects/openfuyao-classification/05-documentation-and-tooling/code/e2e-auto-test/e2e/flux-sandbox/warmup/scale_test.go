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
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// wScaleEnabled gates the 10k-snapshot scale spec (WARM_E2E_SCALE). It is
// on by default unless disabled explicitly ("0"/"false"/"no"/"off").
var wScaleEnabled = gateEnabled("WARM_E2E_SCALE")

func envFlag(key string) bool { return os.Getenv(key) != "" }

// ── Feature group: scale persistence and pagination ──

var _ = ginkgo.Describe("Warm Scale", ginkgo.Label("flux-sandbox", "scale"), func() {

	ginkgo.It("persists and paginates a 10k-snapshot task (WARM-E2E-021)", func() {
		if !wScaleEnabled {
			ginkgo.Skip("WARM_E2E_SCALE gate disabled; 10k-snapshot spec skipped")
		}
		tb := ginkgo.GinkgoT()
		const total = 10000

		// Shrink the dispatch delay for the duration of this spec: 10k
		// snapshots x 2 targets at the 2s default would take hours; 5ms
		// keeps the full warm under a couple of minutes with 3 workers.
		// The delay is restored for subsequent specs.
		ginkgo.DeferCleanup(func() {
			wEnv.e2bSetDelay(tb, 2000)
		})
		wEnv.e2bSetDelay(tb, 5)

		// Generate 10000 unique snapshot ids in order.
		ids := make([]string, total)
		for i := range ids {
			ids[i] = fmt.Sprintf("snapshot-scale-%06d", i+1)
		}

		created, code, raw := wEnv.createWarmTask(tb, ids)
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		gomega.Expect(created.AcceptedSnapshotCt).To(gomega.Equal(total))

		// Step 2: direct DB check - count and JSONB persistence.
		row := wEnv.db(tb).taskRow(tb, created.TaskID)
		gomega.Expect(row.TotalSnapshotCount).To(gomega.Equal(int64(total)))
		var stored []string
		gomega.Expect(json.Unmarshal(row.SnapshotIDs, &stored)).To(gomega.Succeed())
		gomega.Expect(stored).To(gomega.HaveLen(total))
		gomega.Expect(stored[0]).To(gomega.Equal(ids[0]))
		gomega.Expect(stored[total-1]).To(gomega.Equal(ids[total-1]))

		// Step 3: page through with page_size=1000 (pages 1..10 full, page
		// 11 empty) and verify the union in order; then one page_size=10000
		// fetch returns everything at once (upper boundary).
		var collected []string
		for pn := 1; pn <= 11; pn++ {
			p, c, r := wEnv.getProgress(tb, created.TaskID, fmt.Sprintf("page_size=1000&page_num=%d", pn))
			gomega.Expect(c).To(gomega.Equal(200), "page %d: %s", pn, r)
			for _, s := range p.Snapshots {
				collected = append(collected, s.SnapshotID)
			}
			if pn <= 10 {
				gomega.Expect(p.Snapshots).To(gomega.HaveLen(1000), "page %d", pn)
			}
		}
		gomega.Expect(collected).To(gomega.HaveLen(total), "paged union size")
		gomega.Expect(collected).To(gomega.Equal(ids), "paged union must match submission order")

		big, c2, r2 := wEnv.getProgress(tb, created.TaskID, "page_size=10000&page_num=1")
		gomega.Expect(c2).To(gomega.Equal(200), "body %s", r2)
		gomega.Expect(big.Snapshots).To(gomega.HaveLen(total), "upper-bound single page")

		// Step 4: wait for the terminal state. With 2s dispatch delay per
		// target and 3 workers this takes a while; the wait window is
		// generous.
		terminal := wEnv.waitForTaskTerminal(tb, created.TaskID, 60*time.Minute)
		gomega.Expect(terminal).To(gomega.Equal(2))
		progress, _, _ := wEnv.getProgress(tb, created.TaskID, "")
		gomega.Expect(progress.Summary.SuccessCount).To(gomega.Equal(int64(total)))

		// Cleanup: drop the 10k detail rows so later runs start clean.
		_, err := wEnv.db(tb).conn.Exec(context.Background(),
			`DELETE FROM superpod_warm_detail WHERE snapshot_id LIKE 'snapshot-scale-%'`)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
	})
})
