// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package eagle_eye

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	neturl "net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/eagle-eye/utils"
)

const (
	defaultNamespace = "default"
	npeNamespace     = "eagle-eye"
	natsNamespace    = "nats"
	npeSvcName       = "network-performance-exporter"
	// npeMetricsDefaultPort is the fallback port when the service spec has no port listed.
	npeMetricsDefaultPort = int32(8222)
)

// NPEHelmConfig holds Helm parameters for the standalone network-performance-exporter sub-chart.
var NPEHelmConfig = struct {
	ReleaseName string
	ChartURL    string
	Namespace   string
}{
	ReleaseName: "network-performance-exporter",
	Namespace:   npeNamespace,
}

// npeInstalledNats track whether NPE_deploy_003 installed
// these prerequisites, so the final AfterAll only uninstalls what the test suite added.
var (
	npeInstalledNats bool
)

// NetworkPerformanceExporterPhase registers P0 e2e test cases for network-performance-exporter:
//   - NPE_deploy_001: eagle-eye chart installed with enabled=false must NOT create pods
//   - NPE_deploy_002: eagle-eye chart installed with enabled=true must create all required pods
//   - NPE_deploy_003: standalone NPE sub-chart install deploys the pod successfully
//   - NPE_deploy_004: standalone NPE sub-chart stays in init when NATS is unavailable
//   - NPE_deploy_005: the running NPE exposes a valid Prometheus /metrics endpoint
func NetworkPerformanceExporterPhase() {
	Describe("NPE_deploy_001 eagle-eye installed with enabled=false",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			var hasPrometheusOperator bool

			BeforeAll(func() {
				podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
				if err == nil {
					for _, p := range podList.Items {
						if strings.Contains(p.Name, "prometheus-operator") {
							hasPrometheusOperator = true
							break
						}
					}
				}
				GinkgoWriter.Printf("prometheus-operator present before install: %v\n", hasPrometheusOperator)

				uninstallEagleEye()
				installEagleEyeWithEnabled(false)
			})

			AfterAll(func() {
				uninstallEagleEye()
			})

			It("no NPE, hardware, or NATS pods should be created", func() {
				podList, err := clientset.CoreV1().Pods(npeNamespace).List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())
				for _, p := range podList.Items {
					Expect(p.Name).NotTo(ContainSubstring("network-performance-exporter"),
						"unexpected NPE pod %q found when enabled=false", p.Name)
					Expect(p.Name).NotTo(HavePrefix("hardware-diagnosis-"),
						"unexpected hardware-diagnosis pod %q found when enabled=false", p.Name)
					Expect(p.Name).NotTo(HavePrefix("hardware-monitor-"),
						"unexpected hardware-monitor pod %q found when enabled=false", p.Name)
				}

				natsPods, err := clientset.CoreV1().Pods(natsNamespace).List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())
				for _, p := range natsPods.Items {
					Expect(p.Name).NotTo(HavePrefix("nats"),
						"unexpected nats pod %q found when enabled=false", p.Name)
				}

				if !hasPrometheusOperator {
					defaultPods, err := clientset.CoreV1().Pods(defaultNamespace).List(ctx, metav1.ListOptions{})
					Expect(err).NotTo(HaveOccurred())
					for _, p := range defaultPods.Items {
						Expect(p.Name).NotTo(ContainSubstring("prometheus-operator"),
							"unexpected NPE resource %q found in default namespace when enabled=false", p.Name)
					}
				}
			})
		})

	Describe("NPE_deploy_002 eagle-eye installed with enabled=true",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			var hasPrometheusOperator bool

			BeforeAll(func() {
				podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
				if err == nil {
					for _, p := range podList.Items {
						if strings.Contains(p.Name, "prometheus-operator") ||
							strings.Contains(p.Name, "kube-prometheus") {
							hasPrometheusOperator = true
							break
						}
					}
				}
				GinkgoWriter.Printf("prometheus-operator present before install: %v\n", hasPrometheusOperator)

				uninstallEagleEye()
				installEagleEyeWithEnabled(true)
			})

			AfterAll(func() {
				uninstallEagleEye()
			})

			It("NPE, NATS, hardware-monitor, hardware-diagnosis pods must all be Running", func() {
				waitForNPEPodRunning(clientset, ctx)

				Eventually(func() bool {
					natsPods, err := clientset.CoreV1().Pods(natsNamespace).List(ctx, metav1.ListOptions{})
					if err != nil {
						GinkgoWriter.Printf("failed to list nats pods in namespace %s: %v\n", natsNamespace, err)
						return false
					}
					for _, p := range natsPods.Items {
						if strings.HasPrefix(p.Name, "nats") && p.Status.Phase == corev1.PodRunning {
							return true
						}
					}
					return false
				}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
					"a Running nats pod should exist in %s namespace when enabled=true", natsNamespace)

				var monitorPod *corev1.Pod
				Eventually(func() *corev1.Pod {
					monitorPod = GetHardwarePod(clientset, ctx, "monitor")
					return monitorPod
				}, 5*time.Minute, 10*time.Second).ShouldNot(BeNil(),
					"hardware-monitor pod should exist and be Running when enabled=true")
				Expect(monitorPod.Status.Phase).To(Equal(corev1.PodRunning),
					"hardware-monitor pod should be Running")

				var diagPod *corev1.Pod
				Eventually(func() *corev1.Pod {
					diagPod = GetHardwarePod(clientset, ctx, "diagnosis")
					return diagPod
				}, 5*time.Minute, 10*time.Second).ShouldNot(BeNil(),
					"hardware-diagnosis pod should exist and be Running when enabled=true")
				Expect(diagPod.Status.Phase).To(Equal(corev1.PodRunning),
					"hardware-diagnosis pod should be Running")

				if !hasPrometheusOperator {
					defaultPods, err := clientset.CoreV1().Pods(defaultNamespace).List(ctx, metav1.ListOptions{})
					Expect(err).NotTo(HaveOccurred())
					var found bool
					for _, p := range defaultPods.Items {
						if strings.Contains(p.Name, "prometheus-operator") && p.Status.Phase == corev1.PodRunning {
							found = true
							break
						}
					}
					Expect(found).To(BeTrue(),
						"prometheus-operator pod should be Running in some namespace when enabled=true")
				}
			})
		})

	Describe("NPE_deploy_003 standalone NPE sub-chart deployment",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			BeforeAll(func() {
				// Skip if no Prometheus operator is present in the cluster.
				allPods, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())
				var hasPrometheusOperator bool
				for _, p := range allPods.Items {
					if strings.Contains(p.Name, "prometheus-operator") && p.Status.Phase == corev1.PodRunning {
						hasPrometheusOperator = true
						break
					}
				}
				if !hasPrometheusOperator {
					Skip("No Prometheus operator found in cluster; skipping NPE_deploy_003 standalone deployment test")
				}
				GinkgoWriter.Printf("Prometheus operator found; proceeding with standalone NPE install\n")

				// Pull eagle-eye chart and install NPE sub-chart (nats if missing).
				c := EagleEyeHelmConfig
				tmpDir, err := os.MkdirTemp("", "eagle-eye-chart-*")
				Expect(err).NotTo(HaveOccurred())
				defer os.RemoveAll(tmpDir)

				pullArgs := []string{"pull", c.ChartURL, "--untar", "--destination", tmpDir}
				if c.Version != "" {
					pullArgs = append(pullArgs, "--version", c.Version)
				}
				pullCmd := exec.Command("helm", pullArgs...)
				out, err := pullCmd.CombinedOutput()
				GinkgoWriter.Printf("helm pull eagle-eye output:\n%s\n", string(out))
				Expect(err).NotTo(HaveOccurred(), "helm pull eagle-eye should succeed")

				chartsDir := filepath.Join(tmpDir, "eagle-eye", "charts")

				// Install nats if missing.
				var hasNats bool
				for _, p := range allPods.Items {
					if strings.HasPrefix(p.Name, "nats") && p.Status.Phase == corev1.PodRunning {
						hasNats = true
						break
					}
				}
				if !hasNats {
					HelmInstall("nats", filepath.Join(chartsDir, "nats"), defaultNamespace)
					npeInstalledNats = true
					GinkgoWriter.Printf("nats installed from sub-chart\n")
				}

				// Wait for nats Running before deploying NPE.
				Eventually(func() bool {
					podList, err := clientset.CoreV1().Pods(natsNamespace).List(ctx, metav1.ListOptions{})
					if err != nil {
						return false
					}
					for _, p := range podList.Items {
						if strings.HasPrefix(p.Name, "nats") && p.Status.Phase == corev1.PodRunning {
							return true
						}
					}
					return false
				}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
					"nats pod should be Running before installing NPE standalone")

				HelmInstall(NPEHelmConfig.ReleaseName, filepath.Join(chartsDir, "network-performance-exporter"), NPEHelmConfig.Namespace)
				waitForNPEPodRunning(clientset, ctx)
			})

			AfterAll(func() {
				// Uninstall only what this test block installed: NPE standalone and (if needed) nats.
				uninstallNPEStandalone()
				deleteCmd := exec.Command("kubectl", "delete", "namespace", npeNamespace, "--ignore-not-found")
				if out, err := deleteCmd.CombinedOutput(); err != nil {
					GinkgoWriter.Printf("kubectl delete namespace %s: %v\n%s\n", npeNamespace, err, string(out))
				}
				WaitForNamespaceGone(clientset, ctx, npeNamespace)
				if npeInstalledNats {
					HelmUninstall("nats", defaultNamespace)
					npeInstalledNats = false
				}
			})

			It("NPE pod must be Running without CrashLoopBackOff", func() {
				pod := getNPEPod(clientset, ctx)
				Expect(pod).NotTo(BeNil())
				Expect(pod.Status.Phase).To(Equal(corev1.PodRunning))
				for _, cs := range pod.Status.ContainerStatuses {
					if cs.State.Waiting != nil {
						Expect(cs.State.Waiting.Reason).NotTo(BeElementOf("CrashLoopBackOff", "Error"),
							"container %s should not be in a failed waiting state", cs.Name)
					}
				}
				GinkgoWriter.Printf("NPE pod %s on node %s is Running\n", pod.Name, pod.Spec.NodeName)
			})
		})

	Describe("NPE_deploy_004 NPE waits in init when NATS is unavailable",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			var (
				restoreNats          bool
				originalNatsReplicas int32
			)

			BeforeAll(func() {
				// Prepare the Excel fault scene explicitly: NATS is unavailable,
				// then standalone NPE is installed and should remain in init.
				uninstallNPEStandalone()
				if npeIsNatsAvailable() {
					replicas, ok := scaleNatsStatefulSet(0)
					Expect(ok).To(BeTrue(), "NPE_deploy_004 should be able to stop the running NATS statefulset")
					restoreNats = true
					originalNatsReplicas = replicas
				}
				Eventually(func() bool {
					return !npeIsNatsAvailable()
				}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
					"NATS should be unavailable before installing standalone NPE")

				installNPEStandaloneNoWait()
			})

			AfterAll(func() {
				uninstallNPEStandalone()
				deleteCmd := exec.Command("kubectl", "delete", "namespace", npeNamespace, "--ignore-not-found")
				if out, err := deleteCmd.CombinedOutput(); err != nil {
					GinkgoWriter.Printf("kubectl delete namespace %s: %v\n%s\n", npeNamespace, err, string(out))
				}
				WaitForNamespaceGone(clientset, ctx, npeNamespace)
				if restoreNats {
					restoreNatsStatefulSet(originalNatsReplicas)
				}
			})

			It("NPE pod should stay in wait-for-nats init container", func() {
				var firstPod *corev1.Pod
				Eventually(func() bool {
					firstPod = getNPEPodByLabelOrName()
					return firstPod != nil
				}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
					"network-performance-exporter pod should be created while NATS is unavailable")
				GinkgoWriter.Printf("First NPE pod status: %s/%s phase=%s\n",
					firstPod.Namespace, firstPod.Name, firstPod.Status.Phase)
				time.Sleep(60 * time.Second)

				pod := getNPEPodByLabelOrName()
				Expect(pod).NotTo(BeNil(), "network-performance-exporter pod should still exist after waiting 60s")
				// In the Kubernetes API, kubectl's Init:0/1 state is represented by PodPending
				// with an unfinished initContainer.
				Expect(pod.Status.Phase).To(Equal(corev1.PodPending),
					"NPE pod should remain Pending while init container waits for NATS")

				// Excel expected result 2: describe pod shows initContainer wait-for-nats is Running.
				waitForNats := getContainerStatus(pod.Status.InitContainerStatuses, "wait-for-nats")
				Expect(waitForNats).NotTo(BeNil(), "initContainer wait-for-nats should exist")
				Expect(waitForNats.Ready).To(BeFalse(), "wait-for-nats initContainer should not complete")
				Expect(waitForNats.State.Running).NotTo(BeNil(),
					"wait-for-nats initContainer should be Running while NATS is unavailable")

				// Excel expected result 3: the main NPE container must not start
				// before the wait-for-nats initContainer completes. When a pod is
				// still in init, Kubernetes may not have created main container
				// status yet; nil also means the main container has not started.
				mainContainer := getContainerStatus(pod.Status.ContainerStatuses, "network-performance-exporter")
				if mainContainer != nil {
					Expect(mainContainer.Ready).To(BeFalse(), "main container should not be Ready while init is incomplete")
					Expect(mainContainer.State.Running).To(BeNil(), "main container should not start before wait-for-nats completes")
				}

				// Excel expected result 4: wait-for-nats logs keep printing "Wait NATS service...".
				logs := getPodContainerLogs(clientset, ctx, pod, "wait-for-nats")
				Expect(logs).To(ContainSubstring("Wait NATS service..."),
					"wait-for-nats initContainer logs should keep reporting NATS wait")
			})
		})

	Describe("NPE_deploy_005 NPE metrics endpoint",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			BeforeAll(func() {
				// If NPE is not yet running (e.g. 003 was skipped or cleaned up), install eagle-eye with enabled=true.
				if getNPEPod(clientset, ctx) == nil {
					GinkgoWriter.Printf("NPE pod not found; installing eagle-eye with enabled=true\n")
					installEagleEyeWithEnabled(true)
					waitForNPEPodRunning(clientset, ctx)
				}
			})

			It("/metrics endpoint must return valid Prometheus format", func() {
				svc, err := clientset.CoreV1().Services(npeNamespace).Get(ctx, npeSvcName, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred(),
					"service %s should exist in namespace %s", npeSvcName, npeNamespace)

				clusterIP := svc.Spec.ClusterIP
				Expect(clusterIP).NotTo(BeEmpty())
				Expect(clusterIP).NotTo(Equal("None"))

				// Determine the actual metrics port from the Service spec; fall back to the default.
				metricsPort := npeMetricsDefaultPort
				if len(svc.Spec.Ports) > 0 {
					metricsPort = svc.Spec.Ports[0].Port
				}

				url := fmt.Sprintf("http://%s:%d/metrics", clusterIP, metricsPort)
				GinkgoWriter.Printf("Fetching metrics from %s (via in-cluster pod)\n", url)

				// Access the metrics endpoint from inside the cluster using a temporary pod,
				// because ClusterIP is not routable from outside the cluster.
				var bodyStr string
				Eventually(func() error {
					output := npeHTTPGetInCluster(clientset, ctx, url)
					if output == "" {
						return fmt.Errorf("empty response from %s", url)
					}
					if !strings.Contains(output, "# HELP") {
						return fmt.Errorf("response does not contain Prometheus format: %.200s", output)
					}
					bodyStr = output
					return nil
				}, 60*time.Second, 10*time.Second).Should(Succeed(),
					"GET %s should eventually return valid Prometheus metrics", url)

				Expect(bodyStr).To(ContainSubstring("# HELP"))
				Expect(bodyStr).To(ContainSubstring("# TYPE"))
				GinkgoWriter.Printf("Metrics endpoint returned %d bytes with valid Prometheus format\n", len(bodyStr))
			})
		})

	// NPE_deploy_006 - ServiceMonitor exists and Prometheus target is UP
	Describe("NPE_deploy_006 prometheus scrapes NPE via ServiceMonitor",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			It("ServiceMonitor exists and Prometheus target contains network-performance-exporter", func() {
				smGVR := schema.GroupVersionResource{
					Group:    "monitoring.coreos.com",
					Version:  "v1",
					Resource: "servicemonitors",
				}
				_, err := dynamicClient.Resource(smGVR).Namespace(npeNamespace).Get(
					ctx, "network-performance-exporter", metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred(),
					"ServiceMonitor network-performance-exporter should exist in namespace %s", npeNamespace)
				GinkgoWriter.Printf("ServiceMonitor network-performance-exporter found in %s\n", npeNamespace)

				// Poll until we find a Prometheus instance that has NPE as an active target.
				Eventually(func() string {
					return npeFindPrometheusBase(clientset, ctx)
				}, 5*time.Minute, 10*time.Second).ShouldNot(BeEmpty(),
					"a Prometheus instance scraping network-performance-exporter should be found within 5 minutes")
			})
		})

	// NPE_nats_001 - metrics published to NATS topic
	Describe("NPE_nats_001 metrics published to NATS",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			It("NPE pod logs show successful NATS publish", func() {
				Eventually(func() bool {
					pod := getNPEPod(clientset, ctx)
					return pod != nil && pod.Status.Phase == corev1.PodRunning
				}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
					"NPE pod should be Running before reading logs")

				pod := getNPEPod(clientset, ctx)
				Expect(pod).NotTo(BeNil())
				Eventually(func() string {
					return getNPEPodLogs(clientset, ctx, pod)
				}, 5*time.Minute, 10*time.Second).Should(ContainSubstring("Successfully published metrics to NATS topic metrics.network"),
					"NPE pod logs should contain NATS publish success message")
				GinkgoWriter.Printf("NATS publish success message confirmed in NPE pod logs\n")
			})
		})

	// NPE_nats_002 - Prometheus metrics remain available after NATS is stopped
	Describe("NPE_nats_002 prometheus metrics exported when NATS is unavailable",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			It("NPE /metrics endpoint, pod stability, and NATS error logs should meet expectations", func() {
				// Excel prerequisite: NPE is already Running before NATS is stopped.
				Eventually(func() bool {
					pod := getNPEPod(clientset, ctx)
					return pod != nil && pod.Status.Phase == corev1.PodRunning
				}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
					"NPE pod should be Running before stopping NATS")

				pod := getNPEPod(clientset, ctx)
				Expect(pod).NotTo(BeNil(), "network-performance-exporter pod should be Running before stopping NATS")
				restartCountBefore := npeContainerRestartCount(pod)

				// Excel step: stop NATS service, then wait for the NATS connection to break.
				originalReplicas, ok := scaleNatsStatefulSet(0)
				if !ok {
					Skip("NATS statefulset is not available; NPE_nats_002 requires a stoppable NATS service")
				}
				DeferCleanup(func() {
					restoreNatsStatefulSet(originalReplicas)
				})

				Eventually(func() bool {
					return !npeIsNatsAvailable()
				}, 2*time.Minute, 5*time.Second).Should(BeTrue(),
					"NATS should become unavailable after scaling it down")
				time.Sleep(15 * time.Second)

				// Excel expected result: /metrics still exposes Prometheus-format data.
				bodyStr := npeFetchAndValidateMetrics()
				Expect(bodyStr).NotTo(BeEmpty())
				Expect(bodyStr).To(ContainSubstring("rdma_link_state"))
				Expect(bodyStr).To(ContainSubstring("disk_read_iops"))

				// Excel expected result: NPE remains Running and does not restart.
				podAfter := getNPEPod(clientset, ctx)
				Expect(podAfter).NotTo(BeNil(), "network-performance-exporter pod should remain Running")
				Expect(npeContainerRestartCount(podAfter)).To(Equal(restartCountBefore),
					"NPE pod should not restart when NATS is unavailable")

				// Excel expected result: logs contain NATS unavailable publish errors.
				Eventually(func() bool {
					return npeLogsContainNatsUnavailable(getNPEPodLogs(clientset, ctx, podAfter))
				}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
					"NPE logs should contain NATS unavailable error message")
			})
		})

	// NPE_rdma_001-008 - RDMA metrics: compare node sysfs values with Prometheus
	Describe("NPE RDMA metrics",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			var prometheusBase string

			BeforeAll(func() {
				prometheusBase = waitForNPEPrometheusBase(clientset, ctx, 5*time.Minute, 15*time.Second)
				if prometheusBase == "" {
					Skip("Prometheus service not found; skipping RDMA metrics tests")
				}
				GinkgoWriter.Printf("Waiting 60s for Prometheus to complete initial scrape of NPE...\n")
				time.Sleep(60 * time.Second)
			})

			It("[NPE_rdma_001] rdma_link_state matches actual link state read from each node", func() {
				nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				for _, node := range nodeList.Items {
					nodeName := node.Name
					script := `for dev in $(ls /host/sys/class/infiniband/ 2>/dev/null); do
  for port in $(ls /host/sys/class/infiniband/$dev/ports/ 2>/dev/null); do
    state=$(cat /host/sys/class/infiniband/$dev/ports/$port/state 2>/dev/null || echo "unknown")
    echo "RDMA:$dev:$port:$state"
  done
done`
					output, t1 := npeExecOnNode(clientset, ctx, nodeName, script)
					samples := npeQueryPrometheus(prometheusBase,
						fmt.Sprintf(`rdma_link_state{node="%s"}`, nodeName), t1)

					if strings.TrimSpace(output) == "" {
						// Node has no RDMA devices; Prometheus must also have no data for this node.
						Expect(samples).To(BeEmpty(),
							"node %s has no RDMA devices but rdma_link_state samples found in Prometheus", nodeName)
						GinkgoWriter.Printf("Node %s: no RDMA devices, Prometheus also empty ✓\n", nodeName)
						continue
					}

					// Node has RDMA devices; Prometheus must also have data.
					samples = waitForNodePrometheusSamples(prometheusBase, "rdma_link_state", nodeName)

					for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
						if !strings.HasPrefix(line, "RDMA:") {
							continue
						}
						parts := strings.SplitN(line, ":", 4)
						if len(parts) < 4 {
							continue
						}
						dev, port, stateStr := parts[1], parts[2], parts[3]
						expectedVal := 0.0
						if strings.Contains(stateStr, "ACTIVE") {
							expectedVal = 1.0
						}
						for _, s := range samples {
							if s.Labels["device"] == dev && s.Labels["port"] == port {
								Expect(s.Value).To(Equal(expectedVal),
									"node=%s device=%s port=%s state=%q: expected rdma_link_state=%.0f",
									nodeName, dev, port, stateStr, expectedVal)
							}
						}
					}
				}
			})

			It("[NPE_rdma_002] rdma_physical_link_state matches actual phys_state read from each node", func() {
				nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				for _, node := range nodeList.Items {
					nodeName := node.Name
					script := `for dev in $(ls /host/sys/class/infiniband/ 2>/dev/null); do
  for port in $(ls /host/sys/class/infiniband/$dev/ports/ 2>/dev/null); do
    state=$(cat /host/sys/class/infiniband/$dev/ports/$port/phys_state 2>/dev/null || echo "unknown")
    echo "PHYS:$dev:$port:$state"
  done
done`
					output, t1 := npeExecOnNode(clientset, ctx, nodeName, script)
					nodeSamples := npeQueryPrometheus(prometheusBase,
						fmt.Sprintf(`rdma_physical_link_state{node="%s"}`, nodeName), t1)

					if strings.TrimSpace(output) == "" {
						// Node has no RDMA devices; Prometheus must also have no data for this node.
						Expect(nodeSamples).To(BeEmpty(),
							"node %s has no RDMA devices but rdma_physical_link_state samples found in Prometheus", nodeName)
						continue
					}

					// Node has RDMA devices; Prometheus must also have data.
					nodeSamples = waitForNodePrometheusSamples(prometheusBase, "rdma_physical_link_state", nodeName)

					for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
						if !strings.HasPrefix(line, "PHYS:") {
							continue
						}
						parts := strings.SplitN(line, ":", 4)
						if len(parts) < 4 {
							continue
						}
						dev, port, stateStr := parts[1], parts[2], parts[3]
						// phys_state "5: LinkUp" means link is up
						expectedVal := 0.0
						if strings.Contains(stateStr, "LinkUp") {
							expectedVal = 1.0
						}
						for _, s := range nodeSamples {
							if s.Labels["device"] == dev && s.Labels["port"] == port {
								Expect(s.Value).To(Equal(expectedVal),
									"node=%s device=%s port=%s phys_state=%q: expected rdma_physical_link_state=%.0f",
									nodeName, dev, port, stateStr, expectedVal)
							}
						}
					}
				}
			})

			It("[NPE_rdma_003] DOWN ports must not expose traffic metrics in Prometheus", Label("skip-temporarily"), func() {
				t1 := time.Now()
				// Query all ports that should NOT have traffic metrics: link_state == 0 OR phys_state == 0
				// NPE only exports traffic metrics when BOTH phys_state AND link_state are true (source: collector_for_rdma.py line 252)
				downSamples := npeQueryPrometheus(prometheusBase, "rdma_link_state == 0 or rdma_physical_link_state == 0", t1)
				if len(downSamples) == 0 {
					Skip("no DOWN RDMA ports in Prometheus; skipping")
				}
				trafficMetrics := []string{
					"rdma_tx_bytes_total", "rdma_rx_bytes_total",
					"rdma_tx_rate_mbps", "rdma_rx_rate_mbps",
					"rdma_tx_available_bandwidth_mbps", "rdma_rx_available_bandwidth_mbps",
				}
				for _, down := range downSamples {
					n, dev, port := down.Labels["node"], down.Labels["device"], down.Labels["port"]
					for _, tm := range trafficMetrics {
						query := fmt.Sprintf(`%s{node="%s",device="%s",port="%s"}`, tm, n, dev, port)
						traffic := npeQueryPrometheus(prometheusBase, query, t1)
						Expect(traffic).To(BeEmpty(),
							"DOWN port node=%s device=%s port=%s should not have metric %s", n, dev, port, tm)
					}
				}
				GinkgoWriter.Printf("Verified %d DOWN port(s) have no traffic metrics\n", len(downSamples))
			})

			It("[NPE_rdma_004] RDMA cumulative byte metrics are exposed with valid labels", Label("skip-temporarily"), func() {
				checked := 0
				for _, port := range collectAllRDMAPorts() {
					t1 := time.Now()
					metrics := []string{"rdma_tx_bytes_total", "rdma_rx_bytes_total"}
					if !port.activeAndLinkUp() || !port.countersReadable {
						expectRDMASamplesAbsent(prometheusBase, metrics, port, t1)
						checked++
						continue
					}
					expectRDMASamplesPresent(prometheusBase, "rdma_tx_bytes_total", port, t1)
					expectRDMASamplesPresent(prometheusBase, "rdma_rx_bytes_total", port, t1)
					checked++
				}
				if checked == 0 {
					Skip("no RDMA port found for NPE_rdma_004")
				}
			})

			It("[NPE_rdma_005] RDMA tx/rx rate metrics are exposed with valid labels", Label("skip-temporarily"), func() {
				checked := 0
				for _, port := range collectAllRDMAPorts() {
					t1 := time.Now()
					metrics := []string{"rdma_tx_rate_mbps", "rdma_rx_rate_mbps"}
					if !port.activeAndLinkUp() || !port.countersReadable {
						expectRDMASamplesAbsent(prometheusBase, metrics, port, t1)
						checked++
						continue
					}
					expectRDMASamplesPresent(prometheusBase, "rdma_tx_rate_mbps", port, t1)
					expectRDMASamplesPresent(prometheusBase, "rdma_rx_rate_mbps", port, t1)
					checked++
				}
				if checked == 0 {
					Skip("no RDMA port found for NPE_rdma_005")
				}
			})

			It("[NPE_rdma_006] RDMA available bandwidth metrics are exposed with valid labels", Label("skip-temporarily"), func() {
				checked := 0
				for _, port := range collectAllRDMAPorts() {
					t1 := time.Now()
					metrics := []string{"rdma_tx_available_bandwidth_mbps", "rdma_rx_available_bandwidth_mbps"}
					if !port.activeAndLinkUp() || !port.countersReadable || !port.rateReadable {
						expectRDMASamplesAbsent(prometheusBase, metrics, port, t1)
						checked++
						continue
					}
					expectRDMASamplesPresent(prometheusBase, "rdma_tx_available_bandwidth_mbps", port, t1)
					expectRDMASamplesPresent(prometheusBase, "rdma_rx_available_bandwidth_mbps", port, t1)
					checked++
				}
				if checked == 0 {
					Skip("no RDMA port found for NPE_rdma_006")
				}
			})

			It("[NPE_rdma_007] unreadable RDMA rate file stops available-bandwidth metrics only", Label("skip-temporarily"), func() {
				checked := 0
				t1 := time.Now()
				for _, port := range collectAllRDMAPorts() {
					// This resilience case is read-only. It is verified only when the
					// current environment already has an active port with unreadable rate.
					if !port.activeAndLinkUp() || port.rateReadable || !port.countersReadable {
						continue
					}

					txAvail := npeQueryPrometheus(prometheusBase,
						fmt.Sprintf(`rdma_tx_available_bandwidth_mbps{node="%s",device="%s",port="%s"}`,
							port.node, port.device, port.port), t1)
					rxAvail := npeQueryPrometheus(prometheusBase,
						fmt.Sprintf(`rdma_rx_available_bandwidth_mbps{node="%s",device="%s",port="%s"}`,
							port.node, port.device, port.port), t1)
					Expect(txAvail).To(BeEmpty(),
						"tx available bandwidth should be absent when rate file is unreadable")
					Expect(rxAvail).To(BeEmpty(),
						"rx available bandwidth should be absent when rate file is unreadable")

					for _, metric := range []string{"rdma_tx_rate_mbps", "rdma_rx_rate_mbps", "rdma_tx_bytes_total", "rdma_rx_bytes_total"} {
						expectRDMASamplesPresent(prometheusBase, metric, port, t1)
					}

					pod := getNPEPodForNode(clientset, ctx, port.node)
					Expect(pod).NotTo(BeNil(), "NPE pod should be Running on node %s", port.node)
					logs := getNPEPodLogs(clientset, ctx, pod)
					Expect(logs).To(ContainSubstring("Failed to get bandwidth limit"),
						"NPE logs should contain bandwidth-limit warning")
					Expect(strings.ToLower(logs)).NotTo(ContainSubstring("panic"),
						"NPE should not panic when RDMA rate file is unreadable")
					checked++
				}
				if checked == 0 {
					Skip("current environment has no active RDMA port with unreadable rate file; NPE_rdma_007 requires a real unreadable-rate scenario")
				}
			})

			It("[NPE_rdma_008] nodes without /sys/class/infiniband expose no rdma metrics and keep disk metrics", func() {
				nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				checked := 0
				for _, node := range nodeList.Items {
					nodeName := node.Name
					if npeNodeHasRDMADevice(nodeName) {
						continue
					}

					pod := getNPEPodForNode(clientset, ctx, nodeName)
					Expect(pod).NotTo(BeNil(), "NPE pod should be Running on non-RDMA node %s", nodeName)
					for _, cs := range pod.Status.ContainerStatuses {
						if cs.State.Waiting != nil {
							Expect(cs.State.Waiting.Reason).NotTo(BeElementOf("CrashLoopBackOff", "Error"))
						}
					}

					t1 := time.Now()
					for _, metric := range rdmaMetricNames() {
						samples := npeQueryPrometheus(prometheusBase,
							fmt.Sprintf(`%s{node="%s"}`, metric, nodeName), t1)
						Expect(samples).To(BeEmpty(), "%s should be absent on non-RDMA node %s", metric, nodeName)
					}
					diskSamples := npeQueryPrometheus(prometheusBase,
						fmt.Sprintf(`disk_reads_completed_total{node="%s"}`, nodeName), t1)
					Expect(diskSamples).NotTo(BeEmpty(), "disk metrics should exist on non-RDMA node %s", nodeName)
					checked++
				}
				if checked == 0 {
					Skip("current environment has no non-RDMA node with running NPE pod for NPE_rdma_008")
				}
			})
		})

	// NPE_disk_001-006 - disk metrics and collector resilience
	Describe("NPE disk metrics",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			var prometheusBase string

			BeforeAll(func() {
				prometheusBase = waitForNPEPrometheusBase(clientset, ctx, 5*time.Minute, 15*time.Second)
				if prometheusBase == "" {
					Skip("Prometheus service not found; skipping disk metrics tests")
				}
			})

			It("[NPE_disk_001] disk cumulative metrics match /proc/diskstats", func() {
				nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				checked := 0
				for _, node := range nodeList.Items {
					nodeName := node.Name
					preStats, _ := collectDiskStats(nodeName)
					if len(preStats) == 0 {
						GinkgoWriter.Printf("Node %s: no diskstats output\n", nodeName)
						continue
					}
					waitForDiskMetricsReady(prometheusBase, nodeName, diskCumulativeMetricNames())
					// Excel step: record T1 and read /proc/diskstats from the real node.
					// The helper returns the node-side snapshot and the timestamp closest to it.
					stats, t1 := collectDiskStats(nodeName)
					if len(stats) == 0 {
						GinkgoWriter.Printf("Node %s: no diskstats output\n", nodeName)
						continue
					}

					// Compare each Prometheus cumulative metric with the matching
					// /proc/diskstats field for the same node/device.
					for metric, expectedValue := range diskCumulativeMetrics() {
						samples := npeQueryPrometheus(prometheusBase,
							fmt.Sprintf(`%s{node="%s"}`, metric, nodeName), t1)
						Expect(samples).NotTo(BeEmpty(), "%s should exist on node %s", metric, nodeName)
						for _, sample := range samples {
							dev := expectDiskSampleLabelsAndNonNegative(sample, metric, nodeName)
							devStats, ok := stats[dev]
							Expect(ok).To(BeTrue(),
								"Prometheus %s device=%s on node=%s should exist in /proc/diskstats",
								metric, dev, nodeName)
							expected := expectedValue(devStats)
							expectApproximatelyEqual(sample.Value, expected,
								fmt.Sprintf("%s node=%s device=%s should match /proc/diskstats", metric, nodeName, dev))
						}
					}
					checked++
					GinkgoWriter.Printf("Node %s: verified disk cumulative metrics for %d device(s)\n",
						nodeName, len(stats))
				}
				if checked == 0 {
					Skip("no node with readable /proc/diskstats found for NPE_disk_001")
				}
			})

			It("[NPE_disk_002] disk IOPS metrics are exposed with valid labels", func() {
				// Excel check scope: verify Prometheus exposes the IOPS metric names
				// with node/device labels for real disk devices. No exact value
				// comparison is required here.
				verifyDiskMetricsExposed(prometheusBase, []string{
					"disk_read_iops",
					"disk_write_iops",
				})
			})

			It("[NPE_disk_003] disk throughput metrics are exposed with valid labels", func() {
				// Excel check scope: verify throughput metric names and labels only.
				verifyDiskMetricsExposed(prometheusBase, []string{
					"disk_read_throughput_mb",
					"disk_write_throughput_mb",
				})
			})

			It("[NPE_disk_004] disk utilization metrics are exposed with valid labels", func() {
				// Excel check scope: verify utilization metric names and labels only.
				verifyDiskMetricsExposed(prometheusBase, []string{
					"disk_read_utilization",
					"disk_write_utilization",
				})
			})

			It("[NPE_disk_006] unreadable /proc/diskstats does not crash other collectors", Label("skip-temporarily"), func() {
				nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				checked := 0
				for _, node := range nodeList.Items {
					nodeName := node.Name
					// This fault case is also read-only. If /proc/diskstats is readable,
					// the current node does not satisfy the Excel prerequisite.
					stats, t1 := collectDiskStats(nodeName)
					if len(stats) > 0 {
						continue
					}

					// With unreadable diskstats, the pod should stay healthy and log a warning.
					pod := getNPEPodForNode(clientset, ctx, nodeName)
					Expect(pod).NotTo(BeNil(), "NPE pod should be Running on node %s", nodeName)
					Expect(pod.Status.Phase).To(Equal(corev1.PodRunning))
					for _, cs := range pod.Status.ContainerStatuses {
						if cs.State.Waiting != nil {
							Expect(cs.State.Waiting.Reason).NotTo(BeElementOf("CrashLoopBackOff", "Error"))
						}
					}

					logs := getNPEPodLogs(clientset, ctx, pod)
					Expect(logs).To(ContainSubstring("Failed to read /proc/diskstats"),
						"node %s should log /proc/diskstats read warning", nodeName)
					Expect(strings.ToLower(logs)).NotTo(ContainSubstring("panic"),
						"node %s should not panic when /proc/diskstats is unreadable", nodeName)

					// No disk_* data should be exported for this node, but other collectors
					// such as RDMA should remain normal when the hardware exists.
					for _, metric := range diskAllMetricNames() {
						samples := npeQueryPrometheus(prometheusBase,
							fmt.Sprintf(`%s{node="%s"}`, metric, nodeName), t1)
						Expect(samples).To(BeEmpty(),
							"%s should be absent on node %s when /proc/diskstats is unreadable", metric, nodeName)
					}
					if npeNodeHasRDMADevice(nodeName) {
						rdmaSamples := npeQueryPrometheus(prometheusBase,
							fmt.Sprintf(`rdma_link_state{node="%s"}`, nodeName), t1)
						Expect(rdmaSamples).NotTo(BeEmpty(),
							"rdma metrics should remain available on node %s", nodeName)
					}
					checked++
				}
				if checked == 0 {
					Skip("current environment has readable /proc/diskstats on all nodes; NPE_disk_006 requires a real unreadable /proc/diskstats scenario")
				}
			})
		})

	// NPE_npu_roce_001-007 - NPU RoCE metrics and collector behavior; AfterAll uninstalls NPE
	Describe("NPE NPU RoCE metrics",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			var prometheusBase string
			roceLabels := []string{"node", "logic_id", "phy_id", "card_id", "device_id"}

			BeforeAll(func() {
				prometheusBase = waitForNPEPrometheusBase(clientset, ctx, 5*time.Minute, 15*time.Second)
				if prometheusBase == "" {
					Skip("Prometheus service not found; skipping NPU RoCE tests")
				}
			})

			AfterAll(func() {
				uninstallEagleEye()
			})

			It("[NPE_npu_roce_001] NPU RoCE collector is skipped on nodes without NPU", func() {
				nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				checked := 0
				for _, node := range nodeList.Items {
					nodeName := node.Name
					// Excel prerequisite: run this case only on a real non-NPU node.
					if nodesNPUSituation[nodeName] == "dcmi" {
						continue
					}

					// Excel step: read the NPE pod logs on that node.
					pod := getNPEPodForNode(clientset, ctx, nodeName)
					if pod == nil {
						GinkgoWriter.Printf("Node %s has no NPE pod; skipping node-level NPU RoCE collector log check\n", nodeName)
						continue
					}

					logs := getNPEPodLogs(clientset, ctx, pod)
					Expect(logs).To(ContainSubstring("NPU not detected, skipping NPU RoCE Collector"),
						"node %s should log NPU RoCE collector skip", nodeName)
					Expect(logs).NotTo(ContainSubstring("NPU detected, creating NPU RoCE Collector"),
						"node %s should not create NPU RoCE collector", nodeName)

					// Excel expected result: no npu_roce_* metrics, while disk/RDMA
					// metrics remain normal in Prometheus.
					t1 := time.Now()
					expectNoNpuRoceMetrics(prometheusBase, nodeName, t1)
					expectRDMADiskMetricsNormal(prometheusBase, nodeName, t1)
					checked++
				}
				if checked == 0 {
					Skip("no non-NPU node with a running NPE pod found; NPE_npu_roce_001 requires a real non-NPU node")
				}
			})

			It("[NPE_npu_roce_002] npu_roce_link_state presence matches hccn_tool availability per NPU node", func() {
				nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				for _, node := range nodeList.Items {
					nodeName := node.Name
					// This positive/negative branch is based on the real node:
					// NPU node + hccn_tool should expose link_state; missing hccn_tool should not.
					if nodesNPUSituation[nodeName] != "dcmi" {
						continue
					}

					// Check if hccn_tool exists on this node.
					// npeExecOnNode mounts host root at /host inside the container, so use /host/... prefix.
					checkScript := `test -f /host/usr/local/Ascend/driver/tools/hccn_tool && echo "EXISTS" || echo "NOT_FOUND"`
					checkOutput, _ := npeExecOnNode(clientset, ctx, nodeName, checkScript)
					hasHccnTool := strings.Contains(checkOutput, "EXISTS")

					t1 := time.Now()
					samples := npeQueryPrometheus(prometheusBase,
						fmt.Sprintf(`npu_roce_link_state{node="%s"}`, nodeName), t1)

					if !hasHccnTool {
						// Node has NPU but no hccn_tool -> Prometheus must have no RoCE metrics for this node.
						Expect(samples).To(BeEmpty(),
							"node %s has NPU but no hccn_tool; npu_roce_link_state should be absent in Prometheus", nodeName)
						GinkgoWriter.Printf("Node %s: NPU present, hccn_tool missing, Prometheus has no RoCE metrics ✓\n", nodeName)
						continue
					}

					// Node has NPU and hccn_tool -> Prometheus must have RoCE metrics for this node.
					expectRoceSamplesPresent(samples, roceLabels, "npu_roce_link_state", nodeName)
					GinkgoWriter.Printf("Node %s: NPU + hccn_tool present, Prometheus has %d RoCE link_state samples ✓\n",
						nodeName, len(samples))
				}
			})

			It("[NPE_npu_roce_003] npu_roce_link_up_count presence matches hccn_tool availability per NPU node", func() {
				nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				for _, node := range nodeList.Items {
					nodeName := node.Name
					// link_up_count follows the same real-environment prerequisite:
					// it is expected only when an NPU node can actually run hccn_tool.
					if nodesNPUSituation[nodeName] != "dcmi" {
						continue
					}

					// Check if hccn_tool binary exists on this node
					checkScript := `test -f /host/usr/local/Ascend/driver/tools/hccn_tool && echo "EXISTS" || echo "NOT_FOUND"`
					checkOutput, t1 := npeExecOnNode(clientset, ctx, nodeName, checkScript)
					hasHccnTool := strings.Contains(checkOutput, "EXISTS")

					samples := npeQueryPrometheus(prometheusBase,
						fmt.Sprintf(`npu_roce_link_up_count{node="%s"}`, nodeName), t1)

					if !hasHccnTool {
						Expect(samples).To(BeEmpty(),
							"node %s has NPU but no hccn_tool; npu_roce_link_up_count should be absent in Prometheus", nodeName)
						GinkgoWriter.Printf("Node %s: no hccn_tool, Prometheus also empty\n", nodeName)
						continue
					}
					expectRoceSamplesPresent(samples, roceLabels, "npu_roce_link_up_count", nodeName)
					GinkgoWriter.Printf("Node %s: hccn_tool found, Prometheus has data\n", nodeName)
				}
			})

			It("[NPE_npu_roce_004] npu_roce tx/rx rate metrics are exposed with valid labels", Label("npu"), func() {
				checked := verifyRoceMetricsRelaxed(prometheusBase, roceLabels,
					[]string{"npu_roce_tx_rate_mbps", "npu_roce_rx_rate_mbps"},
					"get_npu_interface_traffic")
				if checked == 0 {
					Skip("no NPU node found with valid npu_roce tx/rx rate metrics for NPE_npu_roce_004")
				}
			})

			It("[NPE_npu_roce_005] NPU RoCE available bandwidth metrics are exposed with valid labels", Label("npu"), func() {
				checked := verifyRoceMetricsRelaxed(prometheusBase, roceLabels,
					[]string{"npu_roce_tx_available_bandwidth_mbps", "npu_roce_rx_available_bandwidth_mbps"},
					"get_npu_link_speed and get_npu_interface_traffic")
				if checked == 0 {
					Skip("no NPU node found with valid npu_roce available_bandwidth metrics for NPE_npu_roce_005")
				}
			})

			It("[NPE_npu_roce_006] NPU RoCE packet loss and retransmit rate metrics are exposed with valid labels", Label("npu"), func() {
				checked := verifyRoceMetricsRelaxed(prometheusBase, roceLabels,
					[]string{"npu_roce_packet_loss_rate", "npu_roce_retransmit_rate"},
					"get_npu_packet_statistics")
				if checked == 0 {
					Skip("no NPU node found with valid npu_roce packet_loss/retransmit_rate metrics for NPE_npu_roce_006")
				}
			})

		})
	// NPE_config_001 - collector-interval helm value is rendered into pod args and takes effect
	Describe("NPE_config_001 collector interval configuration",
		Label("eagle_eye_network_performance_exporter"), Ordered, func() {
			const configuredInterval = 30

			BeforeAll(func() {
				uninstallEagleEye()
				installEagleEyeWithPulledValuesCollectInterval(configuredInterval)
				waitForNPEPodRunning(clientset, ctx)
			})

			AfterAll(func() {
				uninstallEagleEye()
			})

			It("NPE container args and startup logs should use configured collector interval", func() {
				// Verify the rendered container args first; this confirms Helm values
				// reached the workload spec.
				pod := getNPEPod(clientset, ctx)
				Expect(pod).NotTo(BeNil())

				npeContainer := getNPEContainer(pod)
				Expect(npeContainer).NotTo(BeNil(), "network-performance-exporter container should exist")
				Expect(npeContainer.Args).To(ContainElement("--collector-interval"))
				Expect(npeContainer.Args).To(ContainElement(strconv.Itoa(configuredInterval)))

				// Then verify runtime evidence from logs, proving the exporter started
				// collectors with the configured interval.
				Eventually(func() string {
					return getNPEPodLogs(clientset, ctx, pod)
				}, 2*time.Minute, 10*time.Second).Should(ContainSubstring(
					fmt.Sprintf("All collectors started with interval %d seconds", configuredInterval)),
					"NPE logs should show configured collector interval")
			})
		})
}

// getNPEPod returns the first Running network-performance-exporter pod across all namespaces,
// or nil if none is Running. Mirrors the logic of GetHardwarePod.
func getNPEPod(clientset *kubernetes.Clientset, ctx context.Context) *corev1.Pod {
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "list pods should succeed")
	for i := range podList.Items {
		p := &podList.Items[i]
		if strings.HasPrefix(p.Name, "network-performance-exporter") && p.Status.Phase == corev1.PodRunning {
			return p
		}
	}
	return nil
}

func getNPEPodByLabelOrName() *corev1.Pod {
	podList, err := clientset.CoreV1().Pods(npeNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app=network-performance-exporter",
	})
	Expect(err).NotTo(HaveOccurred(),
		"kubectl get pods -n %s -l app=network-performance-exporter should succeed", npeNamespace)
	if len(podList.Items) > 0 {
		return &podList.Items[0]
	}

	podList, err = clientset.CoreV1().Pods(npeNamespace).List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "list NPE namespace pods should succeed")
	for i := range podList.Items {
		p := &podList.Items[i]
		if strings.HasPrefix(p.Name, "network-performance-exporter") {
			return p
		}
	}
	return nil
}

func getContainerStatus(statuses []corev1.ContainerStatus, name string) *corev1.ContainerStatus {
	for i := range statuses {
		if statuses[i].Name == name {
			return &statuses[i]
		}
	}
	return nil
}

// waitForNPEPodRunning blocks until a Running network-performance-exporter pod appears or the timeout elapses.
func waitForNPEPodRunning(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Printf("Waiting for network-performance-exporter pod to reach Running state...\n")
	Eventually(func() bool {
		pod := getNPEPod(clientset, ctx)
		return pod != nil && pod.Status.Phase == corev1.PodRunning
	}, 3*time.Minute, 10*time.Second).Should(BeTrue(),
		"network-performance-exporter pod should reach Running state within 3 minutes")
}

func getNPEPodForNode(clientset *kubernetes.Clientset, ctx context.Context, nodeName string) *corev1.Pod {
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "list pods should succeed")
	for i := range podList.Items {
		p := &podList.Items[i]
		if strings.HasPrefix(p.Name, "network-performance-exporter") &&
			p.Spec.NodeName == nodeName &&
			p.Status.Phase == corev1.PodRunning {
			return p
		}
	}
	return nil
}

type diskStats struct {
	readsCompleted  float64
	readBytes       float64
	readTimeMS      float64
	writesCompleted float64
	writeBytes      float64
	writeTimeMS     float64
}

func collectDiskStats(nodeName string) (map[string]diskStats, time.Time) {
	// Read the node's real /proc/diskstats through a short-lived pod with
	// host root mounted read-only at /host. No disk state is changed.
	script := `awk 'NF >= 11 {print $3, $4, $6 * 512, $7, $8, $10 * 512, $11}' /host/proc/diskstats 2>/dev/null`
	output, t1 := npeExecOnNode(clientset, ctx, nodeName, script)
	stats := map[string]diskStats{}
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 7 {
			continue
		}
		values := make([]float64, 6)
		ok := true
		for i := 1; i < len(fields); i++ {
			value, err := strconv.ParseFloat(fields[i], 64)
			if err != nil {
				ok = false
				break
			}
			values[i-1] = value
		}
		if !ok {
			continue
		}
		stats[fields[0]] = diskStats{
			readsCompleted:  values[0],
			readBytes:       values[1],
			readTimeMS:      values[2],
			writesCompleted: values[3],
			writeBytes:      values[4],
			writeTimeMS:     values[5],
		}
	}
	return stats, t1
}

func diskCumulativeMetrics() map[string]func(diskStats) float64 {
	return map[string]func(diskStats) float64{
		"disk_reads_completed_total":  func(stats diskStats) float64 { return stats.readsCompleted },
		"disk_read_bytes_total":       func(stats diskStats) float64 { return stats.readBytes },
		"disk_read_time_ms_total":     func(stats diskStats) float64 { return stats.readTimeMS },
		"disk_writes_completed_total": func(stats diskStats) float64 { return stats.writesCompleted },
		"disk_write_bytes_total":      func(stats diskStats) float64 { return stats.writeBytes },
		"disk_write_time_ms_total":    func(stats diskStats) float64 { return stats.writeTimeMS },
	}
}

func diskCumulativeMetricNames() []string {
	names := []string{}
	for metric := range diskCumulativeMetrics() {
		names = append(names, metric)
	}
	return names
}

func diskDerivedMetricNames() []string {
	return []string{
		"disk_read_iops",
		"disk_write_iops",
		"disk_read_throughput_mb",
		"disk_write_throughput_mb",
		"disk_read_utilization",
		"disk_write_utilization",
	}
}

func diskAllMetricNames() []string {
	names := []string{}
	for metric := range diskCumulativeMetrics() {
		names = append(names, metric)
	}
	names = append(names, diskDerivedMetricNames()...)
	return names
}

func expectDiskSampleLabelsAndNonNegative(sample PromSample, metricName, nodeName string) string {
	Expect(sample.Labels).To(HaveKeyWithValue("node", nodeName),
		"%s sample should have node=%s label", metricName, nodeName)
	Expect(sample.Labels).To(HaveKey("device"), "%s sample should have device label", metricName)
	Expect(sample.Value).To(BeNumerically(">=", 0),
		"%s value should be non-negative on node %s", metricName, nodeName)
	return sample.Labels["device"]
}

func expectDiskSampleLabels(sample PromSample, metricName, nodeName string) string {
	Expect(sample.Labels).To(HaveKeyWithValue("node", nodeName),
		"%s sample should have node=%s label", metricName, nodeName)
	Expect(sample.Labels).To(HaveKey("device"), "%s sample should have device label", metricName)
	return sample.Labels["device"]
}

func waitForDiskMetricsReady(prometheusBase string, nodeName string, metricNames []string) {
	Eventually(func() bool {
		for _, metric := range metricNames {
			samples := npeQueryPrometheus(prometheusBase,
				fmt.Sprintf(`%s{node="%s"}`, metric, nodeName), time.Now())
			if len(samples) == 0 {
				return false
			}
		}
		return true
	}, 3*time.Minute, 10*time.Second).Should(BeTrue(),
		"disk metrics should be scraped for node %s: %s", nodeName, strings.Join(metricNames, ", "))
}

// only check node and device
func verifyDiskMetricsExposed(prometheusBase string, metricNames []string) {
	nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())

	checked := 0
	for _, node := range nodeList.Items {
		nodeName := node.Name
		preStats, _ := collectDiskStats(nodeName)
		if len(preStats) == 0 {
			continue
		}
		waitForDiskMetricsReady(prometheusBase, nodeName, metricNames)
		stats, t1 := collectDiskStats(nodeName)
		if len(stats) == 0 {
			continue
		}

		for _, metric := range metricNames {
			samples := npeQueryPrometheus(prometheusBase,
				fmt.Sprintf(`%s{node="%s"}`, metric, nodeName), t1)
			Expect(samples).NotTo(BeEmpty(),
				"%s should exist on node %s with node/device labels", metric, nodeName)
			for _, sample := range samples {
				dev := expectDiskSampleLabels(sample, metric, nodeName)
				_, ok := stats[dev]
				Expect(ok).To(BeTrue(),
					"Prometheus %s device=%s on node=%s should exist in current /proc/diskstats",
					metric, dev, nodeName)
				checked++
			}
		}
	}
	if checked == 0 {
		Skip(fmt.Sprintf("no disk metric samples found for %s", strings.Join(metricNames, ", ")))
	}
}

type rdmaPortEvidence struct {
	node             string
	device           string
	port             string
	state            string
	physState        string
	txData           float64
	rxData           float64
	countersReadable bool
	rate             string
	rateReadable     bool
	collectedAt      time.Time
}

func (p rdmaPortEvidence) activeAndLinkUp() bool {
	return strings.Contains(p.state, "ACTIVE") && strings.Contains(p.physState, "LinkUp")
}

func collectAllRDMAPorts() []rdmaPortEvidence {
	nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())

	var ports []rdmaPortEvidence
	for _, node := range nodeList.Items {
		ports = append(ports, collectRDMAPorts(node.Name)...)
	}
	return ports
}

func collectRDMAPorts(nodeName string) []rdmaPortEvidence {
	script := `for dev in $(ls /host/sys/class/infiniband/ 2>/dev/null); do
  for port in $(ls /host/sys/class/infiniband/$dev/ports/ 2>/dev/null); do
    base="/host/sys/class/infiniband/$dev/ports/$port"
    state=$(cat "$base/state" 2>/dev/null || echo "__UNREADABLE__")
    phys=$(cat "$base/phys_state" 2>/dev/null || echo "__UNREADABLE__")
    xmit=$(cat "$base/counters/port_xmit_data" 2>/dev/null || echo "__UNREADABLE__")
    rcv=$(cat "$base/counters/port_rcv_data" 2>/dev/null || echo "__UNREADABLE__")
    rate=$(cat "$base/rate" 2>/dev/null || echo "__UNREADABLE__")
    echo "RDMA_PORT|$dev|$port|$state|$phys|$xmit|$rcv|$rate"
  done
done`
	output, t1 := npeExecOnNode(clientset, ctx, nodeName, script)
	var ports []rdmaPortEvidence
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		if !strings.HasPrefix(line, "RDMA_PORT|") {
			continue
		}
		parts := strings.SplitN(line, "|", 8)
		if len(parts) != 8 {
			continue
		}
		port := rdmaPortEvidence{
			node:         nodeName,
			device:       parts[1],
			port:         parts[2],
			state:        parts[3],
			physState:    parts[4],
			rate:         parts[7],
			rateReadable: parts[7] != "__UNREADABLE__",
			collectedAt:  t1,
		}
		txData, txErr := strconv.ParseFloat(strings.TrimSpace(parts[5]), 64)
		rxData, rxErr := strconv.ParseFloat(strings.TrimSpace(parts[6]), 64)
		if txErr == nil && rxErr == nil {
			port.txData = txData
			port.rxData = rxData
			port.countersReadable = true
		}
		ports = append(ports, port)
	}
	return ports
}

func expectRDMASamplesPresent(prometheusBase, metric string, port rdmaPortEvidence, at time.Time) {
	query := fmt.Sprintf(`%s{node="%s",device="%s",port="%s"}`,
		metric, port.node, port.device, port.port)
	var samples []PromSample
	Eventually(func() []PromSample {
		samples = npeQueryPrometheus(prometheusBase, query, time.Now())
		return samples
	}, 3*time.Minute, 10*time.Second).ShouldNot(BeEmpty(),
		"%s should exist for RDMA port node=%s device=%s port=%s",
		metric, port.node, port.device, port.port)
	for _, sample := range samples {
		Expect(sample.Labels).To(HaveKeyWithValue("node", port.node),
			"%s sample should have node label", metric)
		Expect(sample.Labels).To(HaveKeyWithValue("device", port.device),
			"%s sample should have device label", metric)
		Expect(sample.Labels).To(HaveKeyWithValue("port", port.port),
			"%s sample should have port label", metric)
	}
}

func waitForNodePrometheusSamples(prometheusBase, metric, nodeName string) []PromSample {
	query := fmt.Sprintf(`%s{node="%s"}`, metric, nodeName)
	var samples []PromSample
	Eventually(func() []PromSample {
		samples = npeQueryPrometheus(prometheusBase, query, time.Now())
		return samples
	}, 3*time.Minute, 10*time.Second).ShouldNot(BeEmpty(),
		"node %s has RDMA devices but no %s samples found in Prometheus", nodeName, metric)
	return samples
}

func expectRDMASamplesAbsent(prometheusBase string, metrics []string, port rdmaPortEvidence, at time.Time) {
	for _, metric := range metrics {
		samples := npeQueryPrometheus(prometheusBase,
			fmt.Sprintf(`%s{node="%s",device="%s",port="%s"}`,
				metric, port.node, port.device, port.port), at)
		Expect(samples).To(BeEmpty(),
			"%s should be absent for RDMA port node=%s device=%s port=%s when the port does not satisfy the metric prerequisite",
			metric, port.node, port.device, port.port)
	}
}

func rdmaMetricNames() []string {
	return []string{
		"rdma_link_state",
		"rdma_physical_link_state",
		"rdma_tx_bytes_total",
		"rdma_rx_bytes_total",
		"rdma_tx_rate_mbps",
		"rdma_rx_rate_mbps",
		"rdma_tx_available_bandwidth_mbps",
		"rdma_rx_available_bandwidth_mbps",
	}
}

func expectApproximatelyEqual(actual, expected float64, message string) {
	tolerance := math.Max(1024, math.Abs(expected)*0.10)
	Expect(actual).To(BeNumerically("~", expected, tolerance), message)
}

func npeNodeHasHccnTool(nodeName string) bool {
	checkScript := `test -f /host/usr/local/Ascend/driver/tools/hccn_tool && echo "EXISTS" || echo "NOT_FOUND"`
	checkOutput, _ := npeExecOnNode(clientset, ctx, nodeName, checkScript)
	return strings.Contains(checkOutput, "EXISTS")
}

func expectRoceSamplesPresent(samples []PromSample, labels []string, metricName string, nodeName string) {
	Expect(samples).NotTo(BeEmpty(), "%s samples should exist on NPU node %s", metricName, nodeName)
	for _, sample := range samples {
		if name, ok := sample.Labels["__name__"]; ok {
			Expect(name).To(Equal(metricName), "Prometheus sample metric name should be %s", metricName)
		}
		expectRoceLabels(sample, labels)
	}
}

func verifyRoceMetricsByHccnTool(prometheusBase, nodeName string, labels []string, metricNames []string) {
	hasHccnTool := npeNodeHasHccnTool(nodeName)
	t1 := time.Now()
	for _, metric := range metricNames {
		query := fmt.Sprintf(`%s{node="%s"}`, metric, nodeName)
		samples := npeQueryPrometheus(prometheusBase, query, t1)
		if !hasHccnTool {
			Expect(samples).To(BeEmpty(),
				"node %s has NPU but no hccn_tool; %s should be absent in Prometheus",
				nodeName, metric)
			continue
		}
		Eventually(func() []PromSample {
			samples = npeQueryPrometheus(prometheusBase, query, time.Now())
			return samples
		}, 3*time.Minute, 10*time.Second).ShouldNot(BeEmpty(),
			"%s samples should exist on NPU node %s", metric, nodeName)
		expectRoceSamplesPresent(samples, labels, metric, nodeName)
	}
}

// verifyRoceMetricsRelaxed verifies NPU RoCE metrics with relaxed validation:
// - If hccn_tool is absent, metrics must be absent
// - If hccn_tool is present but metrics are absent, log a warning (hccn_tool subcommand may have failed)
// - If metrics are present, verify labels
// Returns the number of nodes with valid metrics found.
func verifyRoceMetricsRelaxed(prometheusBase string, labels []string, metricNames []string, hccnSubcommand string) int {
	nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())

	checked := 0
	for _, node := range nodeList.Items {
		nodeName := node.Name
		if nodesNPUSituation[nodeName] != "dcmi" {
			continue
		}

		hasHccnTool := npeNodeHasHccnTool(nodeName)
		t1 := time.Now()

		metricsFound := 0
		for _, metric := range metricNames {
			samples := npeQueryPrometheus(prometheusBase,
				fmt.Sprintf(`%s{node="%s"}`, metric, nodeName), t1)

			if !hasHccnTool {
				// hccn_tool not present -> metrics must be absent
				Expect(samples).To(BeEmpty(),
					"node %s has no hccn_tool; %s should be absent", nodeName, metric)
				continue
			}

			// hccn_tool present -> metrics may or may not exist (depends on hccn_tool subcommand success)
			// If present, verify labels
			if len(samples) > 0 {
				expectRoceSamplesPresent(samples, labels, metric, nodeName)
				metricsFound++
				checked++
			}
		}

		if hasHccnTool && metricsFound == 0 {
			GinkgoWriter.Printf("Node %s: hccn_tool exists but no metrics %v found (%s may have failed)\n",
				nodeName, metricNames, hccnSubcommand)
		}
	}
	return checked
}

func expectRoceLabels(sample PromSample, labels []string) {
	for _, label := range labels {
		Expect(sample.Labels).To(HaveKey(label), "NPU RoCE sample should have label %q", label)
	}
}

func npuRoceMetricNames() []string {
	return []string{
		"npu_roce_link_state",
		"npu_roce_link_up_count",
		"npu_roce_tx_rate_mbps",
		"npu_roce_rx_rate_mbps",
		"npu_roce_tx_available_bandwidth_mbps",
		"npu_roce_rx_available_bandwidth_mbps",
		"npu_roce_packet_loss_rate",
		"npu_roce_retransmit_rate",
	}
}

func expectNoNpuRoceMetrics(prometheusBase, nodeName string, at time.Time) {
	// PromQL does not support a wildcard metric name query, so check the known
	// npu_roce_* metric family explicitly.
	for _, metric := range npuRoceMetricNames() {
		samples := npeQueryPrometheus(prometheusBase,
			fmt.Sprintf(`%s{node="%s"}`, metric, nodeName), at)
		Expect(samples).To(BeEmpty(),
			"node %s has no NPU; %s should be absent", nodeName, metric)
	}
}

func expectRDMADiskMetricsNormal(prometheusBase, nodeName string, at time.Time) {
	// disk_* should be present on normal nodes. RDMA is checked only when the
	// node really has /sys/class/infiniband devices.
	diskSamples := npeQueryPrometheus(prometheusBase,
		fmt.Sprintf(`disk_reads_completed_total{node="%s"}`, nodeName), at)
	Expect(diskSamples).NotTo(BeEmpty(), "disk metrics should remain available on node %s", nodeName)
	for _, sample := range diskSamples {
		Expect(sample.Labels).To(HaveKey("node"), "disk sample should have node label")
		Expect(sample.Labels).To(HaveKey("device"), "disk sample should have device label")
		Expect(sample.Value).To(BeNumerically(">=", 0),
			"disk metric value should be non-negative on node %s", nodeName)
	}

	if !npeNodeHasRDMADevice(nodeName) {
		GinkgoWriter.Printf("Node %s has no RDMA device; skipping rdma_* normal check\n", nodeName)
		return
	}

	rdmaSamples := npeQueryPrometheus(prometheusBase,
		fmt.Sprintf(`rdma_link_state{node="%s"}`, nodeName), at)
	Expect(rdmaSamples).NotTo(BeEmpty(), "rdma metrics should remain available on RDMA node %s", nodeName)
	for _, sample := range rdmaSamples {
		Expect(sample.Labels).To(HaveKey("node"), "rdma sample should have node label")
		Expect(sample.Labels).To(HaveKey("device"), "rdma sample should have device label")
		Expect(sample.Labels).To(HaveKey("port"), "rdma sample should have port label")
		Expect(sample.Value).To(BeNumerically(">=", 0),
			"rdma_link_state should be 0 or 1 on node %s", nodeName)
		Expect(sample.Value).To(BeNumerically("<=", 1),
			"rdma_link_state should be 0 or 1 on node %s", nodeName)
	}
}

func npeNodeHasRDMADevice(nodeName string) bool {
	checkScript := `test -d /host/sys/class/infiniband && ls /host/sys/class/infiniband 2>/dev/null | grep -q . && echo "HAS_RDMA" || echo "NO_RDMA"`
	output, _ := npeExecOnNode(clientset, ctx, nodeName, checkScript)
	return strings.Contains(output, "HAS_RDMA")
}

func installEagleEyeWithEnabled(enabled bool) {
	installEagleEyeWithEnabledAndArgs(enabled)
}

func installEagleEyeWithEnabledAndArgs(enabled bool, extraArgs ...string) {
	c := EagleEyeHelmConfig
	args := []string{
		"--set", fmt.Sprintf("enabled=%v", enabled),
		"--set", "tracing.enabled=false", // Always disable tracing to avoid image pull timeout
	}

	args = append(args, extraArgs...)
	if c.Version != "" {
		args = append([]string{"--version", c.Version}, args...)
	}
	HelmInstall(c.ReleaseName, c.ChartURL, c.Namespace, args...)
}

func installEagleEyeWithPulledValuesCollectInterval(interval int) {
	c := EagleEyeHelmConfig
	tmpDir, err := os.MkdirTemp("", "eagle-eye-chart-*")
	Expect(err).NotTo(HaveOccurred())
	defer os.RemoveAll(tmpDir)

	pullArgs := []string{"pull", c.ChartURL, "--untar", "--destination", tmpDir}
	if c.Version != "" {
		pullArgs = append(pullArgs, "--version", c.Version)
	}
	pullCmd := exec.Command("helm", pullArgs...)
	out, err := pullCmd.CombinedOutput()
	GinkgoWriter.Printf("helm pull eagle-eye output:\n%s\n", string(out))
	Expect(err).NotTo(HaveOccurred(), "helm pull eagle-eye should succeed")

	chartDir := filepath.Join(tmpDir, "eagle-eye")
	valuesPath := filepath.Join(chartDir, "values.yaml")
	valuesBytes, err := os.ReadFile(valuesPath)
	Expect(err).NotTo(HaveOccurred(), "read pulled eagle-eye values.yaml should succeed")

	values := map[string]interface{}{}
	Expect(yaml.Unmarshal(valuesBytes, &values)).To(Succeed(), "parse pulled eagle-eye values.yaml should succeed")
	values["enabled"] = true

	npeValues := yamlMapValue(values, "network-performance-exporter")
	npeExporterValues := yamlMapValue(npeValues, "networkPerformanceExporter")
	npeExporterValues["collectInterval"] = interval

	updatedValuesBytes, err := yaml.Marshal(values)
	Expect(err).NotTo(HaveOccurred(), "marshal updated eagle-eye values.yaml should succeed")
	Expect(os.WriteFile(valuesPath, updatedValuesBytes, 0644)).To(Succeed(),
		"write updated eagle-eye values.yaml should succeed")

	HelmInstall(c.ReleaseName, chartDir, c.Namespace)
}

func installNPEStandaloneNoWait() {
	c := EagleEyeHelmConfig
	tmpDir, err := os.MkdirTemp("", "eagle-eye-chart-*")
	Expect(err).NotTo(HaveOccurred())
	defer os.RemoveAll(tmpDir)

	pullArgs := []string{"pull", c.ChartURL, "--untar", "--destination", tmpDir}
	if c.Version != "" {
		pullArgs = append(pullArgs, "--version", c.Version)
	}
	pullCmd := exec.Command("helm", pullArgs...)
	out, err := pullCmd.CombinedOutput()
	GinkgoWriter.Printf("helm pull eagle-eye output:\n%s\n", string(out))
	Expect(err).NotTo(HaveOccurred(), "helm pull eagle-eye should succeed")

	chartDir := filepath.Join(tmpDir, "eagle-eye", "charts", "network-performance-exporter")
	args := []string{"install", NPEHelmConfig.ReleaseName, chartDir, "-n", NPEHelmConfig.Namespace, "--create-namespace", "--set", "tracing.enabled=false"}
	installCmd := exec.Command("helm", args...)
	out, err = installCmd.CombinedOutput()
	GinkgoWriter.Printf("helm install %s without --wait output:\n%s\n", NPEHelmConfig.ReleaseName, string(out))
	Expect(err).NotTo(HaveOccurred(),
		"helm install %s without --wait should create resources even while NATS is unavailable\noutput:\n%s",
		NPEHelmConfig.ReleaseName, string(out))
}

func yamlMapValue(parent map[string]interface{}, key string) map[string]interface{} {
	if value, ok := parent[key].(map[string]interface{}); ok {
		return value
	}
	value := map[string]interface{}{}
	parent[key] = value
	return value
}
func getNPEContainer(pod *corev1.Pod) *corev1.Container {
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == "network-performance-exporter" {
			return &pod.Spec.Containers[i]
		}
	}
	return nil
}
func uninstallNPEStandalone() {
	c := NPEHelmConfig
	HelmUninstall(c.ReleaseName, c.Namespace)
}

func npeIsNatsAvailable() bool {
	natsPods, err := clientset.CoreV1().Pods(natsNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		GinkgoWriter.Printf("failed to list nats pods in namespace %s: %v\n", natsNamespace, err)
		return false
	}
	for _, pod := range natsPods.Items {
		if !strings.HasPrefix(pod.Name, "nats") || pod.Status.Phase != corev1.PodRunning {
			continue
		}
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Ready {
				return true
			}
		}
	}
	return false
}

func scaleNatsStatefulSet(replicas int32) (int32, bool) {
	sts, err := clientset.AppsV1().StatefulSets(natsNamespace).Get(ctx, "nats", metav1.GetOptions{})
	if err != nil {
		GinkgoWriter.Printf("failed to get nats statefulset in namespace %s: %v\n", natsNamespace, err)
		return 0, false
	}
	originalReplicas := int32(1)
	if sts.Spec.Replicas != nil {
		originalReplicas = *sts.Spec.Replicas
	}
	sts.Spec.Replicas = &replicas
	_, err = clientset.AppsV1().StatefulSets(natsNamespace).Update(ctx, sts, metav1.UpdateOptions{})
	Expect(err).NotTo(HaveOccurred(), "scale nats statefulset to %d replicas should succeed", replicas)
	GinkgoWriter.Printf("Scaled nats statefulset from %d to %d replicas\n", originalReplicas, replicas)
	return originalReplicas, true
}

func restoreNatsStatefulSet(replicas int32) {
	_, ok := scaleNatsStatefulSet(replicas)
	if !ok {
		return
	}
	if replicas == 0 {
		return
	}
	Eventually(func() bool {
		return npeIsNatsAvailable()
	}, 3*time.Minute, 10*time.Second).Should(BeTrue(),
		"NATS should become available after restoring replicas")
}

func npeFetchAndValidateMetrics() string {
	svc, err := clientset.CoreV1().Services(npeNamespace).Get(ctx, npeSvcName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(),
		"service %s should exist in namespace %s", npeSvcName, npeNamespace)
	Expect(svc.Spec.ClusterIP).NotTo(BeEmpty())
	Expect(svc.Spec.ClusterIP).NotTo(Equal("None"))

	metricsPort := npeMetricsDefaultPort
	if len(svc.Spec.Ports) > 0 {
		metricsPort = svc.Spec.Ports[0].Port
	}
	url := fmt.Sprintf("http://%s:%d/metrics", svc.Spec.ClusterIP, metricsPort)
	GinkgoWriter.Printf("NATS unavailable; fetching NPE metrics from %s\n", url)

	var bodyStr string
	Eventually(func() error {
		output := npeHTTPGetInCluster(clientset, ctx, url)
		if output == "" {
			return fmt.Errorf("empty response from %s", url)
		}
		if !strings.Contains(output, "# HELP") || !strings.Contains(output, "# TYPE") {
			return fmt.Errorf("response is not Prometheus format: %.200s", output)
		}
		bodyStr = output
		return nil
	}, 60*time.Second, 10*time.Second).Should(Succeed(),
		"GET %s should return Prometheus metrics when NATS is unavailable", url)

	GinkgoWriter.Printf("Metrics endpoint returned %d bytes while NATS is unavailable\n", len(bodyStr))
	return bodyStr
}

func npeContainerRestartCount(pod *corev1.Pod) int32 {
	var total int32
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == "network-performance-exporter" {
			total += cs.RestartCount
		}
	}
	return total
}

func npeLogsContainNatsUnavailable(logs string) bool {
	return strings.Contains(logs, "nats: encountered error") ||
		strings.Contains(logs, "NATS publisher not connected, skipping publish") ||
		strings.Contains(logs, "Failed to publish metrics to NATS") ||
		strings.Contains(logs, "Failed to publish to NATS")
}

// getNPEPodLogs fetches all current logs from the given NPE pod.
func getNPEPodLogs(clientset *kubernetes.Clientset, ctx context.Context, pod *corev1.Pod) string {
	req := clientset.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{})
	stream, err := req.Stream(ctx)
	Expect(err).NotTo(HaveOccurred(), "get NPE pod logs should succeed")
	defer stream.Close()
	logBytes, err := io.ReadAll(stream)
	Expect(err).NotTo(HaveOccurred())
	return string(logBytes)
}

func getPodContainerLogs(clientset *kubernetes.Clientset, ctx context.Context, pod *corev1.Pod, containerName string) string {
	req := clientset.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{
		Container: containerName,
	})
	stream, err := req.Stream(ctx)
	Expect(err).NotTo(HaveOccurred(), "kubectl logs -n %s %s -c %s should succeed",
		pod.Namespace, pod.Name, containerName)
	defer stream.Close()
	logBytes, err := io.ReadAll(stream)
	Expect(err).NotTo(HaveOccurred())
	return string(logBytes)
}

func waitForNPEPrometheusBase(clientset *kubernetes.Clientset, ctx context.Context, timeout, interval time.Duration) string {
	deadline := time.Now().Add(timeout)
	for {
		base := npeFindPrometheusBase(clientset, ctx)
		if base != "" {
			return base
		}
		if time.Now().After(deadline) {
			return ""
		}
		GinkgoWriter.Printf("Prometheus with network-performance-exporter active target not found yet; retrying in %s\n", interval)
		time.Sleep(interval)
	}
}

// npeFindPrometheusBase scans all namespaces for Prometheus services and returns the base URL
// of the instance that actually has network-performance-exporter as an active target.
// HTTP requests are made from inside the cluster via a temporary pod to avoid ClusterIP routing issues.
func npeFindPrometheusBase(clientset *kubernetes.Clientset, ctx context.Context) string {
	svcs, err := clientset.CoreV1().Services("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return ""
	}
	for _, svc := range svcs.Items {
		if !strings.Contains(svc.Name, "prometheus") ||
			strings.Contains(svc.Name, "operator") ||
			strings.Contains(svc.Name, "operated") ||
			svc.Spec.ClusterIP == "" || svc.Spec.ClusterIP == "None" ||
			len(svc.Spec.Ports) == 0 {
			continue
		}
		base := fmt.Sprintf("http://%s:%d", svc.Spec.ClusterIP, svc.Spec.Ports[0].Port)
		body := npeHTTPGetInCluster(clientset, ctx, base+"/api/v1/targets?state=active")
		if strings.Contains(body, "network-performance-exporter") {
			GinkgoWriter.Printf("Found Prometheus with NPE target: %s (svc %s/%s)\n",
				base, svc.Namespace, svc.Name)
			return base
		}
	}
	return ""
}

// npeHTTPGetInCluster performs an HTTP GET request from inside the cluster by launching a temporary
// alpine pod. This avoids ClusterIP routing issues when tests run outside the cluster.
// Returns the response body as a string, or empty string on error.
func npeHTTPGetInCluster(clientset *kubernetes.Clientset, ctx context.Context, url string) string {
	podName := fmt.Sprintf("npe-http-%d", time.Now().UnixNano())
	script := fmt.Sprintf(`wget -q -O- --timeout=10 '%s' 2>/dev/null`, url)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: defaultNamespace},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{{
				Name:    "wget",
				Image:   "alpine:latest",
				Command: []string{"/bin/sh", "-c", script},
			}},
			Tolerations: []corev1.Toleration{{Operator: corev1.TolerationOpExists}},
		},
	}
	if _, err := clientset.CoreV1().Pods(defaultNamespace).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		GinkgoWriter.Printf("npeHTTPGetInCluster: create pod failed: %v\n", err)
		return ""
	}
	defer func() {
		_ = clientset.CoreV1().Pods(defaultNamespace).Delete(ctx, podName, metav1.DeleteOptions{})
	}()
	for i := 0; i < 60; i++ {
		p, err := clientset.CoreV1().Pods(defaultNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			time.Sleep(time.Second)
			continue
		}
		if p.Status.Phase == corev1.PodSucceeded || p.Status.Phase == corev1.PodFailed {
			break
		}
		time.Sleep(time.Second)
	}
	req := clientset.CoreV1().Pods(defaultNamespace).GetLogs(podName, &corev1.PodLogOptions{})
	logBytes, err := req.DoRaw(ctx)
	if err != nil {
		return ""
	}
	return string(logBytes)
}

// PromSample holds a single instant-query result from Prometheus.
type PromSample struct {
	Labels map[string]string
	Value  float64
}

// npeQueryPrometheus executes an instant PromQL query at the given timestamp and returns the matching samples.
// Pass the time recorded immediately after node-side data collection so Prometheus returns the value
// closest to that observation point rather than the current scrape.
// HTTP requests are made from inside the cluster via a temporary pod to avoid ClusterIP routing issues.
func npeQueryPrometheus(prometheusBase, query string, at time.Time) []PromSample {
	url := prometheusBase + "/api/v1/query?query=" + neturl.QueryEscape(query) +
		fmt.Sprintf("&time=%d", at.Unix())
	bodyStr := npeHTTPGetInCluster(clientset, ctx, url)
	if bodyStr == "" {
		return nil
	}
	body := []byte(bodyStr)
	var result struct {
		Data struct {
			Result []struct {
				Metric map[string]string `json:"metric"`
				Value  []interface{}     `json:"value"`
			} `json:"result"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil
	}
	var samples []PromSample
	for _, r := range result.Data.Result {
		if len(r.Value) < 2 {
			continue
		}
		valStr, ok := r.Value[1].(string)
		if !ok {
			continue
		}
		val, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			continue
		}
		samples = append(samples, PromSample{Labels: r.Metric, Value: val})
	}
	return samples
}

// npeExecOnNode creates a short-lived privileged pod on nodeName, mounts the host root at /host,
// runs script, returns stdout and the timestamp at which the output was received, and deletes the pod.
func npeExecOnNode(clientset *kubernetes.Clientset, ctx context.Context, nodeName, script string) (string, time.Time) {
	podName := fmt.Sprintf("npe-exec-%s-%d",
		strings.ReplaceAll(strings.ReplaceAll(nodeName, ".", "-"), "_", "-"),
		time.Now().UnixNano())
	trueVal := true
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: defaultNamespace},
		Spec: corev1.PodSpec{
			NodeName: nodeName, HostPID: true, HostNetwork: true,
			RestartPolicy: corev1.RestartPolicyNever,
			Volumes: []corev1.Volume{{
				Name: "host-root",
				VolumeSource: corev1.VolumeSource{
					HostPath: &corev1.HostPathVolumeSource{Path: "/"},
				},
			}},
			Containers: []corev1.Container{{
				Name:    "runner",
				Image:   "alpine:latest",
				Command: []string{"/bin/sh", "-c", script},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "host-root", MountPath: "/host", ReadOnly: true},
				},
				SecurityContext: &corev1.SecurityContext{Privileged: &trueVal},
			}},
			Tolerations: []corev1.Toleration{{Operator: corev1.TolerationOpExists}},
		},
	}
	if _, err := clientset.CoreV1().Pods(defaultNamespace).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		GinkgoWriter.Printf("npeExecOnNode: create pod on %s failed: %v\n", nodeName, err)
		return "", time.Time{}
	}
	defer func() {
		_ = clientset.CoreV1().Pods(defaultNamespace).Delete(ctx, podName, metav1.DeleteOptions{})
	}()
	for i := 0; i < 60; i++ {
		p, err := clientset.CoreV1().Pods(defaultNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			time.Sleep(time.Second)
			continue
		}
		if p.Status.Phase == corev1.PodSucceeded || p.Status.Phase == corev1.PodFailed {
			break
		}
		time.Sleep(time.Second)
	}
	req := clientset.CoreV1().Pods(defaultNamespace).GetLogs(podName, &corev1.PodLogOptions{})
	logBytes, err := req.DoRaw(ctx)
	t1 := time.Now() // record T1 immediately after receiving the node-side output
	if err != nil {
		return "", time.Time{}
	}
	return string(logBytes), t1
}
