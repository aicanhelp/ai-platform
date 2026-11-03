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
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// ── Feature group: node warm backend ──

var _ = ginkgo.Describe("Warm Node Backend", ginkgo.Label("flux-sandbox", "node-backend"), func() {

	ginkgo.It("warms via the per-node third-party service (WARM-E2E-027)", func() {
		tb := ginkgo.GinkgoT()

		// The node backend discovers target nodes via the superpod label.
		// Label the first Ready node as superpod1.
		nodeList, err := wEnv.kube.CoreV1().Nodes().List(wEnv.ctx, metav1.ListOptions{})
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		nodeIndex := -1
		for i := range nodeList.Items {
			for _, condition := range nodeList.Items[i].Status.Conditions {
				if condition.Type == corev1.NodeReady && condition.Status == corev1.ConditionTrue {
					nodeIndex = i
					break
				}
			}
			if nodeIndex >= 0 {
				break
			}
		}
		if nodeIndex < 0 {
			ginkgo.Fail("no Ready node found for the node warm backend")
		}
		node := &nodeList.Items[nodeIndex]
		nodeName := node.Name
		origLabels := map[string]string{}
		for k, v := range node.Labels {
			origLabels[k] = v
		}
		ginkgo.DeferCleanup(func() {
			node, err := wEnv.kube.CoreV1().Nodes().Get(wEnv.ctx, nodeName, metav1.GetOptions{})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			node.Labels = origLabels
			_, err = wEnv.kube.CoreV1().Nodes().Update(wEnv.ctx, node, metav1.UpdateOptions{})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		})
		node.Labels["unifiedbus.com/superpod"] = "superpod1"
		_, err = wEnv.kube.CoreV1().Nodes().Update(wEnv.ctx, node, metav1.UpdateOptions{})
		gomega.Expect(err).NotTo(gomega.HaveOccurred())

		// Deploy the third-party per-node warm service (DaemonSet,
		// hostPort 44772). The fake-e2b image implements the same
		// warm-snapshot endpoint shape.
		wEnv.deployNodeWarmService(tb)
		ginkgo.DeferCleanup(func() {
			_ = wEnv.kube.AppsV1().DaemonSets(wNamespace).Delete(wEnv.ctx, "node-warm-service",
				metav1.DeleteOptions{})
		})

		// Switch the controller to the node backend (args swap + rollout),
		// restoring the e2b backend afterwards.
		dep, err := wEnv.kube.AppsV1().Deployments(wControllerNS).Get(wEnv.ctx,
			"flux-sandbox-controller", metav1.GetOptions{})
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		origArgs := append([]string(nil), dep.Spec.Template.Spec.Containers[0].Args...)
		ginkgo.DeferCleanup(func() {
			wEnv.restoreControllerArgs(tb, origArgs)
		})
		var nodeArgs []string
		for _, a := range origArgs {
			switch {
			case strings.HasPrefix(a, "--warm-e2b-api-endpoint="):
				continue // e2b endpoint is not used by the node backend
			case strings.HasPrefix(a, "--warm-backend-type="):
				continue // replaced below
			default:
				nodeArgs = append(nodeArgs, a)
			}
		}
		nodeArgs = append(nodeArgs, "--warm-backend-type=node")
		wEnv.setControllerArgs(tb, nodeArgs, true)

		// Submit a task; the only labeled node is superpod1, and the
		// load balancer plans top-2 by capacity, so superpod1 is always a
		// target. superpod2/superpod3 have no labeled nodes on this
		// single-node cluster, so those targets legitimately fail with
		// "no nodes found" - the task may therefore end FAILED even though
		// the node backend path works. What matters here:
		//   1. superpod1 was warmed through nodeIP:44772 (detail row shows
		//      it warmed), and
		//   2. the failure mode for label-less SuperPods is a clean
		//      discovery error, not a crash or a wrong dispatch.
		snap := "snapshot-node-001"
		wEnv.db(tb).deleteDetails(tb, snap)
		created, code, raw := wEnv.createWarmTask(tb, []string{snap})
		gomega.Expect(code).To(gomega.Equal(201), "body %s", raw)
		// Wait for terminal state (either SUCCEED or FAILED - see above).
		wEnv.waitForTaskTerminal(tb, created.TaskID, retryWindow*2)

		// The snapshot warmed through the node service: the superpod1
		// target (the labeled node) landed in warmed_superpod_id.
		row := wEnv.db(tb).detailRowMust(tb, snap)
		warmed := jsonStrings(tb, row.WarmedSuperPodID)
		gomega.Expect(warmed).To(gomega.ContainElement("superpod1"),
			"the labeled node's superpod must be warmed via the node service, warmed=%v", warmed)
	})
})

// deployNodeWarmService runs the per-node warm service as a DaemonSet with
// hostPort 44772 (the third-party service contract).
func (e *warmEnv) deployNodeWarmService(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	dsLabels := map[string]string{"app": "node-warm-service"}
	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: "node-warm-service", Namespace: wNamespace, Labels: dsLabels},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: dsLabels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: dsLabels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "node-warm",
						Image: wFakeE2BIm,
						Env: []corev1.EnvVar{
							{Name: "FAKE_E2B_PORT", Value: "44772"},
							// No FAKE_ETCD_ENDPOINTS: this instance is a
							// plain warm service, not a MoonCakeStore
							// reporter.
						},
						Ports: []corev1.ContainerPort{{
							ContainerPort: 44772,
							HostPort:      44772,
						}},
						ImagePullPolicy: corev1.PullIfNotPresent,
					}},
				},
			},
		},
	}
	err := e.crClient.Get(e.ctx, wObjectKey(wNamespace, "node-warm-service"), ds)
	if err == nil {
		return // already deployed
	}
	if err := e.crClient.Create(e.ctx, ds); err != nil {
		tb.Fatalf("create node-warm-service daemonset: %v", err)
	}
	// Wait for the DaemonSet pod to be ready.
	waitFor(tb, "node-warm-service ready", 3*time.Minute, func() bool {
		pods, err := e.kube.CoreV1().Pods(wNamespace).List(e.ctx,
			metav1.ListOptions{LabelSelector: labels.SelectorFromSet(dsLabels).String()})
		if err != nil {
			return false
		}
		for _, p := range pods.Items {
			if podReady(p) {
				return true
			}
		}
		return false
	})
}
