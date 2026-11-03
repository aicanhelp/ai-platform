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
	"strings"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
)

// ── Feature group: request validation ──

var _ = ginkgo.Describe("Warm Request Validation", ginkgo.Label("flux-sandbox", "validation"), func() {

	ginkgo.It("rejects duplicate snapshot ids with 400 (WARM-E2E-022)", func() {
		tb := ginkgo.GinkgoT()

		resp, raw := wEnv.httpDo(tb, http.MethodPost, "/api/v1/snapshot-warm-tasks",
			map[string]any{"snapshot_ids": []string{"snapshot-dup-001", "snapshot-dup-001"}})
		gomega.Expect(resp.StatusCode).To(gomega.Equal(400), "body %s", raw)
		apiErr := decodeError(tb, raw)
		gomega.Expect(apiErr.Code).To(gomega.Equal("INVALID_ARGUMENT"))
		gomega.Expect(apiErr.Message).To(gomega.Equal("snapshot_ids must not contain duplicates"))

		// No task row was created.
		var count int
		err := wEnv.db(tb).conn.QueryRow(wEnv.ctx,
			`SELECT count(*) FROM snapshot_warm_task WHERE task_id LIKE '%'`).Scan(&count)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		before := count
		_ = before
		var dup int
		err = wEnv.db(tb).conn.QueryRow(wEnv.ctx,
			`SELECT count(*) FROM snapshot_warm_task`).Scan(&dup)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		// The rejected request must not have added a task for the dup id.
		var withDup int
		err = wEnv.db(tb).conn.QueryRow(wEnv.ctx,
			`SELECT count(*) FROM snapshot_warm_task WHERE snapshot_ids @> '["snapshot-dup-001"]'::jsonb`).Scan(&withDup)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(withDup).To(gomega.Equal(0), "no task row for the rejected duplicate request")
	})

	ginkgo.It("rejects invalid creation bodies with 400 (WARM-E2E-023)", func() {
		tb := ginkgo.GinkgoT()

		post := func(body any) (int, []byte) {
			resp, raw := wEnv.httpDo(tb, http.MethodPost, "/api/v1/snapshot-warm-tasks", body)
			return resp.StatusCode, raw
		}

		// Empty JSON object (missing snapshot_ids).
		code, raw := post(map[string]any{})
		gomega.Expect(code).To(gomega.Equal(400), "empty object: %s", raw)
		gomega.Expect(decodeError(tb, raw).Message).To(gomega.Equal("snapshot_ids must not be empty"))

		// Empty array.
		code, raw = post(map[string]any{"snapshot_ids": []string{}})
		gomega.Expect(code).To(gomega.Equal(400), "empty array: %s", raw)
		gomega.Expect(decodeError(tb, raw).Message).To(gomega.Equal("snapshot_ids must not be empty"))

		// Empty string element.
		code, raw = post(map[string]any{"snapshot_ids": []string{"snapshot-ok-001", ""}})
		gomega.Expect(code).To(gomega.Equal(400), "empty element: %s", raw)
		gomega.Expect(decodeError(tb, raw).Message).To(gomega.Equal("snapshot_ids must not contain empty strings"))

		// Malformed JSON body (raw truncated JSON).
		resp, raw := wEnv.httpDoRaw(tb, http.MethodPost, "/api/v1/snapshot-warm-tasks", `{"snapshot_ids": ["a"`)
		gomega.Expect(resp.StatusCode).To(gomega.Equal(400), "malformed: %s", raw)
		gomega.Expect(decodeError(tb, raw).Message).To(gomega.Equal("invalid request body"))
	})

	ginkgo.It("returns 404 for a nonexistent task (WARM-E2E-024)", func() {
		tb := ginkgo.GinkgoT()

		resp, raw := wEnv.httpDo(tb, http.MethodGet, "/api/v1/snapshot-warm-tasks/warm-task-not-exist", nil)
		gomega.Expect(resp.StatusCode).To(gomega.Equal(404), "body %s", raw)
		apiErr := decodeError(tb, raw)
		gomega.Expect(apiErr.Code).To(gomega.Equal("NOT_FOUND"))
		gomega.Expect(apiErr.Message).To(gomega.Equal("task not found"))
	})

	ginkgo.It("echoes or generates X-Request-ID (WARM-E2E-028)", func() {

		// Without the header the server generates one (warm-N or seed
		// prefixed format).
		req, err := http.NewRequest(http.MethodGet, wURL("/api/v1/superpods/storage-details"), nil)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		resp, err := http.DefaultClient.Do(req)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		err = resp.Body.Close()
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		got := resp.Header.Get("X-Request-ID")
		gomega.Expect(got).NotTo(gomega.BeEmpty(), "server must generate an X-Request-ID")
		gomega.Expect(strings.HasPrefix(got, "warm-")).To(gomega.BeTrue(), "generated id %q", got)

		// With the header the value is echoed back verbatim.
		req2, err := http.NewRequest(http.MethodGet, wURL("/api/v1/superpods/storage-details"), nil)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		req2.Header.Set("X-Request-ID", "fit-test-req-001")
		resp2, err := http.DefaultClient.Do(req2)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		err = resp2.Body.Close()
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(resp2.Header.Get("X-Request-ID")).To(gomega.Equal("fit-test-req-001"))
	})
})
