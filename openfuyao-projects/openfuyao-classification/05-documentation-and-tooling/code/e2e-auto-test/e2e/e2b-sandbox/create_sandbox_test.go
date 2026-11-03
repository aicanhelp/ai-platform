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

var _ = Describe("创建 Sandbox",
	Label("e2b-sandbox", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("E2E-IT-001：创建 sandbox 成功", func() {
			By("通过 API 创建 sandbox")
			sandboxID := createSandboxAndAwaitRunning("it001")

			By("校验 sandbox 状态为 Running")
			sb, status, err := apiClient.GetSandbox(ctx, sandboxID)
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusOK))
			Expect(sb.Status.State).To(Equal(StateRunning), "sandbox 状态应为 Running")
		})

		It("E2E-IT-002：创建 sandbox 失败——未传 sandboxGroup", func() {
			By("不传 extensions.sandboxGroup 创建 sandbox")
			_, status, err := apiClient.CreateSandbox(ctx, &CreateSandboxRequest{
				Image:      &ImageSpec{URI: EnvOr("E2B_SANDBOX_IMAGE", DefaultImage)},
				Entrypoint: []string{"sleep", "3600"},
				ResourceLimits: &ResourceLimits{
					CPU:    DefaultCPU,
					Memory: DefaultMemory,
				},
				Timeout: DefaultTimeout,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(StatusInternalServerError), "未传 sandboxGroup 应返回 500")
		})

		It("E2E-IT-003：创建 sandbox 失败——镜像不存在", func() {
			By("使用不存在的镜像创建 sandbox")
			_, status, err := apiClient.CreateSandbox(ctx, &CreateSandboxRequest{
				Image:      &ImageSpec{URI: "nonexistent-image-12345"},
				Entrypoint: []string{"sleep", "3600"},
				ResourceLimits: &ResourceLimits{
					CPU:    DefaultCPU,
					Memory: DefaultMemory,
				},
				Extensions: map[string]string{"sandboxGroup": sgName},
				Timeout:    DefaultTimeout,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(BeNumerically(">=", 400), "镜像不存在应返回 4xx 或 5xx")
		})
	})
