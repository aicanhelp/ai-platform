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
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ── Feature group: warm fail-fast on missing configuration ──

var _ = ginkgo.Describe("Warm Fail Fast", ginkgo.Label("flux-sandbox", "fail-fast"), func() {

	ginkgo.It("crashes when warm is enabled without the required config (WARM-E2E-029)", func() {
		tb := ginkgo.GinkgoT()
		runFailFastSpecAttach(tb)
	})
})

// runFailFastSpecAttach implements WARM-E2E-029: a disposable pod
// with warm enabled but no DSN must CrashLoopBackOff with the fail-fast log,
// leaving the existing controller untouched.
func runFailFastSpecAttach(tb ginkgo.GinkgoTInterface) {
	name := wEnv.runFailFastPod(tb)
	var crashLog string
	waitFor(tb, "fail-fast pod crashloops with warm init failure", 3*time.Minute, func() bool {
		pod, err := wEnv.kube.CoreV1().Pods(wControllerNS).Get(wEnv.ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		for _, c := range pod.Status.ContainerStatuses {
			if c.RestartCount == 0 {
				continue
			}
			logs, err := wEnv.kube.CoreV1().Pods(wControllerNS).GetLogs(name,
				&corev1.PodLogOptions{Previous: true}).DoRaw(wEnv.ctx)
			if err != nil {
				continue
			}
			if strings.Contains(string(logs), "init warm subsystem") {
				crashLog = string(logs)
				return true
			}
		}
		return false
	})
	gomega.Expect(crashLog).To(gomega.ContainSubstring("warm-enabled requires --warm-postgres-dsn"),
		"crash log must name the missing dependency")
}
