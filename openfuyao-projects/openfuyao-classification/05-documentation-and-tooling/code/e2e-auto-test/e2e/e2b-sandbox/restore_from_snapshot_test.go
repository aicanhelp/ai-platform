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

var _ = Describe("从快照创建 Sandbox",
	Label("e2b-sandbox", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("E2E-IT-012：从快照创建 sandbox 成功", func() {
			By("创建源 sandbox 和快照")
			sandboxID := createSandboxAndAwaitRunning("it012-src")
			snapshotID := createSnapshotAndAwaitReady(sandboxID, "restore-it012")

			By("从快照创建新 sandbox")
			resp, status, err := apiClient.CreateSandbox(ctx, &CreateSandboxRequest{
				SnapshotID: snapshotID,
				Entrypoint: []string{"sleep", "3600"},
				ResourceLimits: &ResourceLimits{
					CPU:    DefaultCPU,
					Memory: DefaultMemory,
				},
				Extensions: map[string]string{"sandboxGroup": sgName},
				Timeout:    DefaultTimeout,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusAccepted), "从快照创建 sandbox 应返回 202")
			Expect(resp.ID).NotTo(BeEmpty())

			By("等待新 sandbox 进入 Running 状态")
			awaitSandboxState(resp.ID, StateRunning)
		})

		It("E2E-IT-013：快照不存在时从快照创建 sandbox 返回 404", func() {
			By("使用不存在的快照 ID 创建 sandbox")
			_, status, err := apiClient.CreateSandbox(ctx, &CreateSandboxRequest{
				SnapshotID: "nonexistent-snapshot-id",
				Entrypoint: []string{"sleep", "3600"},
				ResourceLimits: &ResourceLimits{
					CPU:    DefaultCPU,
					Memory: DefaultMemory,
				},
				Extensions: map[string]string{"sandboxGroup": sgName},
				Timeout:    DefaultTimeout,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusNotFound), "快照不存在应返回 404")
		})

		It("E2E-IT-014：快照删除后从快照创建 sandbox 返回 404", func() {
			By("创建 sandbox 和快照")
			sandboxID := createSandboxAndAwaitRunning("it014")
			snapshotID := createSnapshotAndAwaitReady(sandboxID, "deleted-it014")

			By("删除快照")
			delStatus, err := apiClient.DeleteSnapshot(ctx, snapshotID)
			Expect(err).NotTo(HaveOccurred())
			Expect(delStatus).To(Equal(StatusNoContent))

			By("用已删除的快照创建 sandbox")
			_, status, err := apiClient.CreateSandbox(ctx, &CreateSandboxRequest{
				SnapshotID: snapshotID,
				Entrypoint: []string{"sleep", "3600"},
				ResourceLimits: &ResourceLimits{
					CPU:    DefaultCPU,
					Memory: DefaultMemory,
				},
				Extensions: map[string]string{"sandboxGroup": sgName},
				Timeout:    DefaultTimeout,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusNotFound), "删除后从快照创建应返回 404")
		})
	})
