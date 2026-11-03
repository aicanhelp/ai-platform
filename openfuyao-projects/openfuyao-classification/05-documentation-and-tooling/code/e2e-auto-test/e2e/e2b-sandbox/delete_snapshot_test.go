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

var _ = Describe("删除快照",
	Label("e2b-sandbox", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("E2E-IT-020：删除快照成功", func() {
			By("创建 sandbox 和快照")
			sandboxID := createSandboxAndAwaitRunning("it020")
			snapshotID := createSnapshotAndAwaitReady(sandboxID, "delete-it020")

			By("删除快照")
			status, err := apiClient.DeleteSnapshot(ctx, snapshotID)
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusNoContent), "删除快照应返回 204")

			By("再次查询快照应返回 404")
			_, getStatus, err := apiClient.GetSnapshot(ctx, snapshotID)
			Expect(err).NotTo(HaveOccurred())
			Expect(getStatus).To(Equal(StatusNotFound), "删除后查询应返回 404")
		})

		It("E2E-IT-021：删除不存在的快照返回 404", func() {
			By("删除不存在的快照")
			status, err := apiClient.DeleteSnapshot(ctx, "nonexistent-snapshot-id")
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusNotFound), "删除不存在的快照应返回 404")
		})
	})
