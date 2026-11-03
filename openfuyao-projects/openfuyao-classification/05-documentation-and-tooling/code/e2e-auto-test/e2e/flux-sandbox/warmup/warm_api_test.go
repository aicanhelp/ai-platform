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

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ── Feature group: warm deployment and readiness ──

var _ = ginkgo.Describe("Warm Deployment", ginkgo.Label("flux-sandbox", "deploy"), func() {

	ginkgo.It("starts the warm subsystem with readiness checks (WARM-E2E-001)", func() {
		tb := ginkgo.GinkgoT()

		// The controller pod is Running, READY 1/1, zero restarts.
		pods, err := wEnv.kube.CoreV1().Pods(wControllerNS).List(wEnv.ctx,
			metav1.ListOptions{LabelSelector: "app=flux-sandbox-controller"})
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(pods.Items).To(gomega.HaveLen(1))
		pod := pods.Items[0]
		for _, c := range pod.Status.ContainerStatuses {
			gomega.Expect(c.State.Running).NotTo(gomega.BeNil(), "container %s not running", c.Name)
			gomega.Expect(c.RestartCount).To(gomega.Equal(int32(0)), "container %s restarted", c.Name)
			gomega.Expect(c.Ready).To(gomega.BeTrue(), "container %s not ready", c.Name)
		}

		// The log must not contain warm init failures.
		logs, err := wEnv.kube.CoreV1().Pods(wControllerNS).GetLogs(pod.Name, &corev1.PodLogOptions{}).DoRaw(wEnv.ctx)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		for _, failure := range []string{"init warm subsystem", "Failed to init warm"} {
			gomega.Expect(string(logs)).NotTo(gomega.ContainSubstring(failure),
				"controller log contains warm init failure %q:\n%.2000s", failure, logs)
		}
		tb.Logf("controller log tail:\n%.1500s", logs)

		// /readyz and /healthz both return 200.
		for _, path := range []string{"/readyz", "/healthz"} {
			resp, raw := wEnv.httpDo(tb, http.MethodGet, path, nil)
			gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK), "GET %s: %s", path, raw)
		}

		// The warm routes are mounted: an unknown task yields a 404 with the
		// documented error envelope.
		resp, raw := wEnv.httpDo(tb, http.MethodGet, "/api/v1/snapshot-warm-tasks/not-exist-task", nil)
		gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusNotFound), "body %s", raw)
		apiErr := decodeError(tb, raw)
		gomega.Expect(apiErr.Code).To(gomega.Equal("NOT_FOUND"))
		gomega.Expect(apiErr.Message).To(gomega.Equal("task not found"))
	})
})
