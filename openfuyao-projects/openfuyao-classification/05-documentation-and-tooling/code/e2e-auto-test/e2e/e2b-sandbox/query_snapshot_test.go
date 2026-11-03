/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 */

package e2b_sandbox

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/e2b-sandbox/utils"
)

var _ = Describe("查询快照",
	Label("e2b-sandbox", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("E2E-IT-009：查询快照——GET /v1/snapshots/{id}", func() {
			By("创建 sandbox 和快照")
			sandboxID := createSandboxAndAwaitRunning("it009")
			snapshotID := createSnapshotAndAwaitReady(sandboxID, "query-it009")

			By("通过 API 查询快照")
			snap, status, err := apiClient.GetSnapshot(ctx, snapshotID)
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusOK))
			Expect(snap.ID).To(Equal(snapshotID))
			Expect(snap.Image).NotTo(BeEmpty(), "image 字段应非空")
			Expect(snap.TeamID).NotTo(BeEmpty(), "teamId 字段应非空")
		})

		It("E2E-IT-010：列出快照——GET /v1/snapshots 分页", func() {
			By("创建 sandbox 和快照以确保列表非空")
			sandboxID := createSandboxAndAwaitRunning("it010")
			createSnapshotAndAwaitReady(sandboxID, "list-it010")

			By("分页查询快照列表")
			resp, status, err := apiClient.ListSnapshots(ctx, 1, 200)
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusOK))
			Expect(resp.Items).NotTo(BeEmpty(), "快照列表不应为空")
			Expect(resp.Pagination).NotTo(BeNil(), "分页信息不应为空")
		})

		It("E2E-IT-011：查询不存在的快照返回 404", func() {
			By("查询不存在的快照 ID")
			_, status, err := apiClient.GetSnapshot(ctx, "nonexistent-snapshot-id")
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusNotFound), "查询不存在的快照应返回 404")
		})
	})
