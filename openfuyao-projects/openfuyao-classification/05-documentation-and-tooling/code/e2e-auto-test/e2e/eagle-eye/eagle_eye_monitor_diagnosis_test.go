package eagle_eye

import (
	"context"
	"io"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/eagle-eye/utils"
)

// eagleEyeInstalledByTests is set when this suite installs Eagle-Eye; used to uninstall after all tests.
var eagleEyeInstalledByTests bool

func installEagleEyeIfNeeded(clientset *kubernetes.Clientset, ctx context.Context) {
	c := EagleEyeHelmConfig
	eagleEyeInstalledByTests = InstallEagleEyeIfNeeded(clientset, ctx, c.ReleaseName, c.ChartURL, c.Namespace, c.Version)
}

func uninstallEagleEye() {
	c := EagleEyeHelmConfig
	UninstallEagleEye(clientset, ctx, c.ReleaseName, c.Namespace, "eagle-eye", "nats")
}

// MonitorPhase registers the Eagle-Eye metrics collection specs (HBM, DDR, NPU network).
// If Eagle-Eye is not installed, it is installed before tests and uninstalled after all tests finish.
func MonitorPhase() {
	var _ = Describe("Eagle-Eye Monitor", Label("eagle_eye_monitor", "npu"), func() {
		BeforeAll(func() {
			installEagleEyeIfNeeded(clientset, ctx)
		})

		AfterAll(func() {
			if eagleEyeInstalledByTests {
				uninstallEagleEye()
			}
		})

		BeforeEach(func() {
			if !clusterHasNPU(nodesNPUSituation) {
				Skip("Skipping this test: cluster has no NPU (no DCMI path).")
			}
		})

		It("should collect HBM metrics from hardware-monitor (910A1/910A2/910A3/910A5)", func() {
			pod := ensureHardwareMonitorPodForMetric(
				func() bool { return clusterHasAscendTypeForHBM(nodesAscendChipType) },
				func() *corev1.Pod {
					return getHardwareMonitorPodOnNodeWithAscendTypeForHBM(clientset, ctx, nodesNPUSituation, nodesAscendChipType)
				},
				"no node with Ascend type in {910A1,910A2,910A3,910A5}; skipping HBM metrics-not-NA check.",
				"no hardware-monitor pod on a node with Ascend type 910A1/910A2/910A3/910A5")
			verifyHBMMetricsInHardwareMonitorLogs(clientset, ctx, pod, false,
				"hardware-monitor logs should show collector.npu.hbm and HBM metrics not NA")
		})

		It("should collect HBM metrics from hardware-monitor (other types — metrics NA)", func() {
			pod := ensureHardwareMonitorPodForMetric(
				func() bool { return clusterHasNodeWithAscendTypeNotForHBM(nodesAscendChipType) },
				func() *corev1.Pod {
					return getHardwareMonitorPodOnNodeWithAscendTypeNotForHBM(clientset, ctx, nodesNPUSituation, nodesAscendChipType)
				},
				"no node with Ascend type outside {910A1,910A2,910A3,910A5}; skipping HBM metrics-NA check. Set AscendChipByIP in constants.go for your cluster.",
				"no hardware-monitor pod on a node with Ascend type not in {910A1,910A2,910A3,910A5}")
			verifyHBMMetricsInHardwareMonitorLogs(clientset, ctx, pod, true,
				"hardware-monitor logs should show collector.npu.hbm and HBM metrics NA")
		})

		It("should collect DDR metrics from hardware-monitor (except 910A2/910A3/910A5 — metrics not NA)", func() {
			pod := ensureHardwareMonitorPodForMetric(
				func() bool { return clusterHasNodeForDDR(nodesAscendChipType) },
				func() *corev1.Pod {
					return getHardwareMonitorPodOnNodeForDDR(clientset, ctx, nodesNPUSituation, nodesAscendChipType)
				},
				"no node with Ascend type outside {910A2,910A3,910A5}; skipping DDR metrics. Set AscendChipByIP in constants.go.",
				"no hardware-monitor pod on a node with Ascend type not in {910A2,910A3,910A5}")
			verifyDDRMetricsInHardwareMonitorLogs(clientset, ctx, pod, false,
				"hardware-monitor logs should show collector.npu.ddr and DDR metrics not NA")
		})

		It("should collect DDR metrics from hardware-monitor (910A2/910A3/910A5 — metrics NA)", func() {
			pod := ensureHardwareMonitorPodForMetric(
				func() bool { return clusterHasNodeWithAscendTypeNotForDDR(nodesAscendChipType) },
				func() *corev1.Pod {
					return getHardwareMonitorPodOnNodeWithAscendTypeNotForDDR(clientset, ctx, nodesNPUSituation, nodesAscendChipType)
				},
				"no node with Ascend type in {910A2,910A3,910A5}; skipping DDR metrics-NA check. Set AscendChipByIP in constants.go.",
				"no hardware-monitor pod on a node with Ascend type 910A2/910A3/910A5")
			verifyDDRMetricsInHardwareMonitorLogs(clientset, ctx, pod, true,
				"hardware-monitor logs should show collector.npu.ddr and DDR metrics NA")
		})

		It("should collect network metrics from hardware-monitor (except 310/310B/310P — empty then timeout occurred, or non-empty then not NA)", func() {
			pod := ensureHardwareMonitorPodForMetric(
				func() bool { return clusterHasNodeForNPUNetwork(nodesAscendChipType) },
				func() *corev1.Pod {
					return getHardwareMonitorPodOnNodeForNPUNetwork(clientset, ctx, nodesNPUSituation, nodesAscendChipType)
				},
				"no node with Ascend type outside {310,310B,310P}; skipping NPU network metrics. Set AscendChipByIP in constants.go.",
				"no hardware-monitor pod on a node with Ascend type not in {310,310B,310P}")
			verifyNPUNetworkEmptyOrNonEmptyNotNA(clientset, ctx, pod)
		})

		It("should collect NPU network metrics from hardware-monitor (310/310B/310P — metrics NA)", func() {
			pod := ensureHardwareMonitorPodForMetric(
				func() bool { return clusterHasNodeWithAscendTypeNotForNPUNetwork(nodesAscendChipType) },
				func() *corev1.Pod {
					return getHardwareMonitorPodOnNodeWithAscendTypeNotForNPUNetwork(clientset, ctx, nodesNPUSituation, nodesAscendChipType)
				},
				"no node with Ascend type in {310,310B,310P}; skipping NPU network metrics-NA check. Set AscendChipByIP in constants.go.",
				"no hardware-monitor pod on a node with Ascend type 310/310B/310P")
			verifyNPUNetworkMetricsInHardwareMonitorLogs(clientset, ctx, pod, true,
				"hardware-monitor logs should show collector.npu.npu and NPU network metrics NA")
		})
	})
}

// DiagnosisPhase registers the Eagle-Eye diagnosis specs: NATS subscription and log records.
func DiagnosisPhase() {
	var _ = Describe("Eagle-Eye Diagnosis (NATS subscription)", Label("eagle_eye_diagnosis", "npu"), func() {
		BeforeAll(func() {
			installEagleEyeIfNeeded(clientset, ctx)
		})

		AfterAll(func() {
			if eagleEyeInstalledByTests {
				uninstallEagleEye()
			}
		})

		BeforeEach(func() {
			if !clusterHasNPU(nodesNPUSituation) {
				Skip("Skipping this test: cluster has no NPU (no DCMI path).")
			}
		})

		It("monitor publishes to NATS; diagnosis module receives the messages", func() {
			pod := GetHardwarePod(clientset, ctx, "diagnosis")
			if pod == nil {
				Skip("no hardware-diagnosis pod found (Eagle-Eye chart may not deploy diagnosis component)")
			}
			Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "hardware-diagnosis pod should be Running")
			logs := getPodLogs(clientset, ctx, pod)
			Expect(strings.Contains(logs, "Received message on topic")).To(BeTrue(),
				"logs should show receive record: hardware-diagnosis - INFO - "+
					"Received message on topic hardware.monitor; log snippet: %s", getLogSnippet(logs))
			GinkgoWriter.Printf("Logs show receive record (hardware-diagnosis - INFO - " +
				"Received message on topic hardware.monitor) — verified\n")
		})

		It("monitor publishes to NATS; diagnosis module receives the messages and detects events", func() {
			pod := GetHardwarePod(clientset, ctx, "diagnosis")
			if pod == nil {
				Skip("no hardware-diagnosis pod found (Eagle-Eye chart may not deploy diagnosis component)")
			}
			Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "hardware-diagnosis pod should be Running")
			logs := getPodLogs(clientset, ctx, pod)
			Expect(strings.Contains(logs, "Detected events")).To(BeTrue(),
				"logs should show receive record: hardware-diagnosis - INFO - "+
					"Detected events; log snippet: %s", getLogSnippet(logs))
			GinkgoWriter.Printf("Logs show receive record (hardware-diagnosis - INFO - " +
				"Received message on topic hardware.monitor) — verified\n")
		})
	})
}

// --- Test helpers (skip + get pod, verify) ---

// ensureHardwareMonitorPodForMetric skips if clusterCheck is false or getPod returns nil; otherwise returns the pod.
func ensureHardwareMonitorPodForMetric(clusterCheck func() bool, getPod func() *corev1.Pod,
	skipNoNode, skipNoPod string) *corev1.Pod {
	if !clusterCheck() {
		Skip(skipNoNode)
	}
	pod := getPod()
	if pod == nil {
		Skip(skipNoPod)
	}
	return pod
}

// verifyHBMMetricsInHardwareMonitorLogs fetches pod logs and asserts collector.npu.hbm is present and HBM context contains or does not contain "NA" per expectNA.
func verifyHBMMetricsInHardwareMonitorLogs(clientset *kubernetes.Clientset, ctx context.Context, pod *corev1.Pod, expectNA bool, message string) {
	Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "hardware-monitor pod should be Running")
	GinkgoWriter.Printf("Reading hardware-monitor logs from pod %s on node %s\n", pod.Name, pod.Spec.NodeName)

	logs := getHardwareMonitorPodLogs(clientset, ctx, pod)
	Expect(strings.Contains(logs, "collector.npu.hbm")).To(BeTrue(), "%s. Log snippet: %s", message, getLogSnippet(logs))
	hbmCtx := getHBMLogContext(logs)
	Expect(hbmCtx).NotTo(BeEmpty(), "logs should contain collector.npu.hbm and following context")
	if expectNA {
		Expect(strings.Contains(hbmCtx, "NA")).To(BeTrue(),
			"HBM metrics should be NA. HBM context snippet: %s", getLogSnippet(hbmCtx))
	} else {
		Expect(strings.Contains(hbmCtx, "NA")).To(BeFalse(),
			"HBM metrics should not be NA. HBM context snippet: %s", getLogSnippet(hbmCtx))
	}
	GinkgoWriter.Printf("HBM metrics check passed for pod %s (expectNA=%v)\n", pod.Name, expectNA)
}

// verifyDDRMetricsInHardwareMonitorLogs fetches pod logs and asserts collector.npu.ddr is present and DDR segment contains or does not contain "NA" per expectNA.
func verifyDDRMetricsInHardwareMonitorLogs(clientset *kubernetes.Clientset, ctx context.Context, pod *corev1.Pod, expectNA bool, message string) {
	Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "hardware-monitor pod should be Running")
	GinkgoWriter.Printf("Reading hardware-monitor logs from pod %s on node %s\n", pod.Name, pod.Spec.NodeName)

	logs := getHardwareMonitorPodLogs(clientset, ctx, pod)
	Expect(strings.Contains(logs, "collector.npu.ddr")).To(BeTrue(), "%s. Log snippet: %s", message, getLogSnippet(logs))
	ddrCtx := getDDRLogContext(logs)
	Expect(ddrCtx).NotTo(BeEmpty(), "logs should contain collector.npu.ddr and following context")
	if expectNA {
		Expect(strings.Contains(ddrCtx, "NA")).To(BeTrue(),
			"DDR metrics should be NA. DDR context snippet: %s", getLogSnippet(ddrCtx))
	} else {
		Expect(strings.Contains(ddrCtx, "NA")).To(BeFalse(),
			"DDR metrics should not be NA. DDR context snippet: %s", getLogSnippet(ddrCtx))
	}
	GinkgoWriter.Printf("DDR metrics check passed for pod %s (expectNA=%v)\n", pod.Name, expectNA)
}

// hasEmptyNPUNetworkOutput returns true if, in the same "Collected results" block we use for metrics,
// the collector.npu.network segment is empty ({}). This avoids false positives when an earlier log line
// or another cycle mentions empty network.
func hasEmptyNPUNetworkOutput(logs string) bool {
	ctx := getCollectorLogContext(logs, "collector.npu.network")
	if ctx == "" {
		return false
	}
	// Segment looks like "collector.npu.network': {}" or "collector.npu.network\": {}" (maybe with trailing comma).
	after := strings.TrimPrefix(ctx, "collector.npu.network': ")
	if after == ctx {
		after = strings.TrimPrefix(ctx, "collector.npu.network\": ")
	}
	if after == ctx {
		return false
	}
	after = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(after), ", '"))
	return after == "{}"
}

// verifyNPUNetworkEmptyOrNonEmptyNotNA fetches logs once, then: if network output is {} requires "timeout occurred"; if not {} requires metrics not NA.
func verifyNPUNetworkEmptyOrNonEmptyNotNA(clientset *kubernetes.Clientset, ctx context.Context, pod *corev1.Pod) {
	Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "hardware-monitor pod should be Running")
	GinkgoWriter.Printf("Reading hardware-monitor logs from pod %s on node %s\n", pod.Name, pod.Spec.NodeName)
	logs := getHardwareMonitorPodLogs(clientset, ctx, pod)
	if hasEmptyNPUNetworkOutput(logs) {
		lowerAll := strings.ToLower(logs)
		Expect(strings.Contains(lowerAll, "timeout occurred")).To(BeTrue(),
			"NPU network output is {}; logs must contain 'timeout occurred'. Log snippet: %s", getLogSnippet(logs))
		GinkgoWriter.Printf("NPU network empty + timeout occurred check passed for pod %s\n", pod.Name)
		return
	}
	Expect(strings.Contains(logs, "collector.npu.npu")).To(BeTrue(), "logs should contain collector.npu.npu. Log snippet: %s", getLogSnippet(logs))
	npuCtx := getNPUNetworkLogContext(logs)
	Expect(npuCtx).NotTo(BeEmpty(), "logs should contain collector.npu.npu and following context")
	Expect(strings.Contains(npuCtx, "NA")).To(BeFalse(),
		"when output is not {}, NPU network metrics must not be NA. Snippet: %s", getLogSnippet(npuCtx))
	GinkgoWriter.Printf("NPU network non-empty not-NA check passed for pod %s\n", pod.Name)
}

// verifyNPUNetworkMetricsInHardwareMonitorLogs fetches pod logs and asserts NPU network segment contains "NA" when expectNA is true (e.g. 310/310B/310P).
func verifyNPUNetworkMetricsInHardwareMonitorLogs(clientset *kubernetes.Clientset, ctx context.Context, pod *corev1.Pod, expectNA bool, message string) {
	Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "hardware-monitor pod should be Running")
	GinkgoWriter.Printf("Reading hardware-monitor logs from pod %s on node %s\n", pod.Name, pod.Spec.NodeName)
	logs := getHardwareMonitorPodLogs(clientset, ctx, pod)
	Expect(strings.Contains(logs, "collector.npu.npu")).To(BeTrue(), "%s. Log snippet: %s", message, getLogSnippet(logs))
	npuCtx := getNPUNetworkLogContext(logs)
	Expect(npuCtx).NotTo(BeEmpty(), "logs should contain collector.npu.npu and following context")
	if expectNA {
		Expect(strings.Contains(npuCtx, "NA")).To(BeTrue(),
			"NPU network metrics should be NA. NPU network context snippet: %s", getLogSnippet(npuCtx))
	}
	GinkgoWriter.Printf("NPU network metrics check passed for pod %s (expectNA=%v)\n", pod.Name, expectNA)
}

// --- Eagle-Eye / cluster checks ---

func clusterHasNPU(nodesNPUSituation map[string]string) bool {
	if len(nodesNPUSituation) == 0 {
		return false
	}
	for _, npuType := range nodesNPUSituation {
		if npuType != "none" {
			return true
		}
	}
	return false
}

func isAscendTypeInList(chipType string, list []string) bool {
	for _, t := range list {
		if t == chipType {
			return true
		}
	}
	return false
}

func clusterHasNodeMatching(nodesAscendChipType map[string]string, match func(chipType string) bool) bool {
	for _, chipType := range nodesAscendChipType {
		if match(chipType) {
			return true
		}
	}
	return false
}

func isAscendTypeForHBM(chipType string) bool {
	return isAscendTypeInList(chipType, AscendTypesWithHBM)
}
func isAscendTypeExcludedForDDR(chipType string) bool {
	return isAscendTypeInList(chipType, AscendTypesExcludedForDDR)
}
func isAscendTypeExcludedForNPUNetwork(chipType string) bool {
	return isAscendTypeInList(chipType, AscendTypesExcludedForNetwork)
}

func clusterHasAscendTypeForHBM(nodesAscendChipType map[string]string) bool {
	return clusterHasNodeMatching(nodesAscendChipType, isAscendTypeForHBM)
}
func clusterHasNodeWithAscendTypeNotForHBM(nodesAscendChipType map[string]string) bool {
	return clusterHasNodeMatching(nodesAscendChipType, func(chipType string) bool {
		return chipType != "" && !isAscendTypeForHBM(chipType)
	})
}
func clusterHasNodeForDDR(nodesAscendChipType map[string]string) bool {
	return clusterHasNodeMatching(nodesAscendChipType, func(chipType string) bool {
		return chipType != "" && !isAscendTypeExcludedForDDR(chipType)
	})
}

func clusterHasNodeWithAscendTypeNotForDDR(nodesAscendChipType map[string]string) bool {
	return clusterHasNodeMatching(nodesAscendChipType, isAscendTypeExcludedForDDR)
}
func clusterHasNodeForNPUNetwork(nodesAscendChipType map[string]string) bool {
	return clusterHasNodeMatching(nodesAscendChipType, func(chipType string) bool {
		return chipType != "" && !isAscendTypeExcludedForNPUNetwork(chipType)
	})
}
func clusterHasNodeWithAscendTypeNotForNPUNetwork(nodesAscendChipType map[string]string) bool {
	return clusterHasNodeMatching(nodesAscendChipType, isAscendTypeExcludedForNPUNetwork)
}

// --- Pod getters ---

func getHardwareMonitorPodOnNodeMatching(clientset *kubernetes.Clientset, ctx context.Context,
	nodesNPUSituation, nodesAscendChipType map[string]string, nodeMatch func(nodeName, chipType string) bool) *corev1.Pod {
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "list pods should succeed")

	for i := range podList.Items {
		p := &podList.Items[i]
		if !strings.HasPrefix(p.Name, "hardware-monitor-") || p.Status.Phase != corev1.PodRunning {
			continue
		}
		nodeName := p.Spec.NodeName
		if nodeName == "" {
			continue
		}
		if nodesNPUSituation[nodeName] == "" || nodesNPUSituation[nodeName] == "none" {
			continue
		}
		chipType := nodesAscendChipType[nodeName]
		if nodeMatch(nodeName, chipType) {
			return p
		}
	}
	return nil
}

func getHardwareMonitorPodOnNodeWithAscendTypeForHBM(clientset *kubernetes.Clientset, ctx context.Context,
	nodesNPUSituation, nodesAscendChipType map[string]string) *corev1.Pod {
	return getHardwareMonitorPodOnNodeMatching(clientset, ctx,
		nodesNPUSituation, nodesAscendChipType, func(_, chipType string) bool {
			return isAscendTypeForHBM(chipType)
		})
}

func getHardwareMonitorPodOnNodeWithAscendTypeNotForHBM(clientset *kubernetes.Clientset, ctx context.Context,
	nodesNPUSituation, nodesAscendChipType map[string]string) *corev1.Pod {
	return getHardwareMonitorPodOnNodeMatching(clientset, ctx,
		nodesNPUSituation, nodesAscendChipType, func(_, chipType string) bool {
			return chipType != "" && !isAscendTypeForHBM(chipType)
		})
}

func getHardwareMonitorPodOnNodeForDDR(clientset *kubernetes.Clientset, ctx context.Context,
	nodesNPUSituation, nodesAscendChipType map[string]string) *corev1.Pod {
	return getHardwareMonitorPodOnNodeMatching(clientset, ctx,
		nodesNPUSituation, nodesAscendChipType, func(_, chipType string) bool {
			return chipType != "" && !isAscendTypeExcludedForDDR(chipType)
		})
}

func getHardwareMonitorPodOnNodeWithAscendTypeNotForDDR(clientset *kubernetes.Clientset, ctx context.Context,
	nodesNPUSituation, nodesAscendChipType map[string]string) *corev1.Pod {
	return getHardwareMonitorPodOnNodeMatching(clientset, ctx,
		nodesNPUSituation, nodesAscendChipType, func(_, chipType string) bool {
			return isAscendTypeExcludedForDDR(chipType)
		})
}

func getHardwareMonitorPodOnNodeForNPUNetwork(clientset *kubernetes.Clientset, ctx context.Context,
	nodesNPUSituation, nodesAscendChipType map[string]string) *corev1.Pod {
	return getHardwareMonitorPodOnNodeMatching(clientset, ctx,
		nodesNPUSituation, nodesAscendChipType, func(_, chipType string) bool {
			return chipType != "" && !isAscendTypeExcludedForNPUNetwork(chipType)
		})
}

func getHardwareMonitorPodOnNodeWithAscendTypeNotForNPUNetwork(clientset *kubernetes.Clientset, ctx context.Context,
	nodesNPUSituation, nodesAscendChipType map[string]string) *corev1.Pod {
	return getHardwareMonitorPodOnNodeMatching(clientset, ctx,
		nodesNPUSituation, nodesAscendChipType, func(_, chipType string) bool {
			return isAscendTypeExcludedForNPUNetwork(chipType)
		})
}

// --- Log helpers ---

func getPodLogs(clientset *kubernetes.Clientset, ctx context.Context, pod *corev1.Pod) string {
	req := clientset.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{})
	stream, err := req.Stream(ctx)
	Expect(err).NotTo(HaveOccurred(), "get pod logs should succeed")
	defer stream.Close()
	logsBytes, err := io.ReadAll(stream)
	Expect(err).NotTo(HaveOccurred(), "read pod logs should succeed")
	return string(logsBytes)
}

// getHardwareMonitorPodLogs fetches all hardware-monitor logs (no tail limit) so we can find the last "Collected results" block in code.
func getHardwareMonitorPodLogs(clientset *kubernetes.Clientset, ctx context.Context, pod *corev1.Pod) string {
	return getPodLogs(clientset, ctx, pod)
}

func getLogSnippet(logs string) string {
	if len(logs) == 0 {
		return "[empty]"
	}
	if len(logs) > 300 {
		return "..." + logs[len(logs)-300:]
	}
	return logs
}

// getCollectorLogContext returns the log segment for the given collector only.
// Uses the last "Collected results" block (most recent collection) so all tests see the same cycle;
// within that block finds marker and returns from that marker until the next "collector.npu.".
// If "Collected results" is not found, searches the whole log.
func getCollectorLogContext(logs string, marker string) string {
	searchFrom := 0
	if idxResults := strings.LastIndex(logs, "Collected results"); idxResults >= 0 {
		searchFrom = idxResults
	}
	rest := logs[searchFrom:]
	idx := strings.Index(rest, marker)
	if idx < 0 {
		return ""
	}
	start := searchFrom + idx
	afterMarker := start + len(marker)
	end := len(logs)
	restAfterMarker := logs[afterMarker:]
	if next := strings.Index(restAfterMarker, "collector.npu."); next >= 0 {
		end = afterMarker + next
	}
	return logs[start:end]
}

// getHBMLogContext returns the log segment for the HBM collector only (from collector.npu.hbm until next collector.npu.).
func getHBMLogContext(logs string) string {
	return getCollectorLogContext(logs, "collector.npu.hbm")
}

// getDDRLogContext returns the log segment for the DDR collector only (from collector.npu.ddr until next collector.npu.).
func getDDRLogContext(logs string) string {
	return getCollectorLogContext(logs, "collector.npu.ddr")
}

// getNPUNetworkLogContext returns the log segment for the NPU network collector only (from collector.npu.npu until next collector.npu.).
func getNPUNetworkLogContext(logs string) string {
	return getCollectorLogContext(logs, "collector.npu.npu")
}
