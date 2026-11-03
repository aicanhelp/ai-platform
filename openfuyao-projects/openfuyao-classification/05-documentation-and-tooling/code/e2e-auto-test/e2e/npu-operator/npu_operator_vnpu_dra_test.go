package npu_operator

import (
	"context"
	"fmt"
	"strings"
	"time"

	npu_operator_config "gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/npu-operator-config"
	. "gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/util/retry"
)

const (
	vnpuNamespaceName              = "xpu"
	volcanoNamespaceName           = "volcano-system"
	volcanoMonitoringNamespaceName = "volcano-monitoring"

	vnpuReadyLabelKey   = "huawei.com/vnpu"
	vnpuReadyLabelValue = "ready"

	vnpuTestSelectorKey = "npu.openfuyao.com/e2e-vnpu"
	draTestSelectorKey  = "npu.openfuyao.com/e2e-dra"

	testDeviceClassName  = "test-soft-share.npu.huawei.com"
	vnpuSoftSplitPodName = "vnpu-single-soft-split"

	testExporterPort int32 = 29099
)

func VNPUDRAManagementPhase() {
	var _ = Describe("NPU Operator vNPU and DRA management", Ordered,
		Label("npu_operator_vnpu_dra", "npu"), func() {
			var (
				gvr              schema.GroupVersionResource
				originalSpec     map[string]interface{}
				testBaselineSpec map[string]interface{}
				originalSpecOnce bool
				installedByPhase bool
			)

			BeforeAll(func() {
				installedByPhase = EnsureNPUOperatorInstallation(clientset, ctx, nodesNPUSituation)
				GinkgoWriter.Println()

				gvr = npuClusterPolicyGVR()
				originalSpec = snapshotNPUClusterPolicySpec(gvr)
				testBaselineSpec = cleanNPUOperatorVNPUDRATestBaseline(originalSpec)
				originalSpecOnce = true
				restoreNPUClusterPolicySpec(gvr, testBaselineSpec)
				waitForPolicyReady(gvr)
			})

			AfterAll(func() {
				if originalSpecOnce {
					restoreNPUClusterPolicySpec(gvr, originalSpec)
					waitForPolicySettled(gvr)
				}
				if !installedByPhase {
					GinkgoWriter.Println("NPU Operator reused existing environment in this phase, skip uninstall")
					return
				}
				VerifyNPUOperatorUninstallation(clientset, ctx)
			})

			AfterEach(func() {
				if originalSpecOnce {
					restoreNPUClusterPolicySpec(gvr, testBaselineSpec)
					waitForPolicyReady(gvr)
				}
			})

			Context("vNPU component management", Ordered, func() {
				It("NPUOperatorVNPUDRA_FIT_1/FIT_2/FIT_4/FIT_5/FIT_6 - verifies vNPU components, labels, nodeSelector, exporter port and Volcano flavor", func() {
					nodes := listNodes()
					targetNode := firstSchedulableNode(nodes)
					labelNode(targetNode.Name, vnpuTestSelectorKey, "true")
					DeferCleanup(func() {
						removeNodeLabel(targetNode.Name, vnpuTestSelectorKey)
					})

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, false, "dra", "managed")
						setString(spec, "vnpu", "volcano", "flavor")
						setInt64(spec, int64(0), "volcano", "runAsUser")
						setInt64(spec, int64(0), "volcano", "runAsGroup")
						setBool(spec, true, "volcano", "admission", "enabled")
						setBool(spec, true, "vnpu", "managed")
						setNodeSelector(spec, map[string]string{vnpuTestSelectorKey: "true"}, "vnpu")
						setBool(spec, true, "vnpu", "clientUpdate", "managed")
						setBool(spec, true, "vnpu", "devicePlugin", "managed")
						setBool(spec, true, "vnpu", "exporter", "managed")
						setInt64(spec, int64(testExporterPort), "vnpu", "exporter", "port")
						return nil
					})

					waitForPolicyNoComponentConflict(gvr)
					waitForComponentRunning(gvr, "vnpu-device-plugin")
					waitForComponentRunning(gvr, "vnpu-exporter")
					waitForComponentRunning(gvr, "volcano-vnpu-scheduler")
					waitForComponentRunning(gvr, "volcano-vnpu-controller")
					waitForComponentRunning(gvr, "volcano-vnpu-admission")

					expectNamespace(vnpuNamespaceName)
					expectNamespace(volcanoNamespaceName)
					expectNamespace(volcanoMonitoringNamespaceName)

					devicePluginDS := waitForDaemonSet("npu-device-plugin-daemonset")
					expectDaemonSetNodeSelector(devicePluginDS, vnpuTestSelectorKey, "true")
					expectDaemonSetPodsOnSelectedNodes(devicePluginDS, labels.Set{vnpuTestSelectorKey: "true"})

					exporterDS := waitForDaemonSet("xpu-exporter-daemonset")
					expectContainerArg(exporterDS.Spec.Template.Spec.Containers[0].Args, fmt.Sprintf("--port=%d", testExporterPort))
					expectDaemonSetNodeSelector(exporterDS, vnpuTestSelectorKey, "true")
					expectServiceTargetPort("xpu-exporter-svc", vnpuNamespaceName, testExporterPort)

					waitForNodeLabel(targetNode.Name, vnpuReadyLabelKey, vnpuReadyLabelValue)
					expectDeploymentImageContains("volcano-scheduler", volcanoNamespaceName, "/vnpu/vc-scheduler")
					expectDeploymentImageContains("volcano-controllers", volcanoNamespaceName, "/vnpu/vc-controller-manager")
				})

				It("NPUOperatorVNPUDRA_FIT_8 - verifies vNPU admission can be enabled and disabled", func() {
					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setString(spec, "vnpu", "volcano", "flavor")
						setInt64(spec, int64(0), "volcano", "runAsUser")
						setInt64(spec, int64(0), "volcano", "runAsGroup")
						setBool(spec, true, "vccontroller", "managed")
						setBool(spec, true, "vcscheduler", "managed")
						setBool(spec, true, "volcano", "admission", "enabled")
						return nil
					})

					waitForComponentRunning(gvr, "volcano-vnpu-admission")
					waitForDeployment("volcano-admission", volcanoNamespaceName)
					expectService("volcano-admission-service", volcanoNamespaceName)
					expectJob("volcano-admission-init", volcanoNamespaceName)
					expectClusterResource(mutatingWebhookConfigurationGVR(), "volcano-admission-service-queues-mutate")
					expectClusterResource(mutatingWebhookConfigurationGVR(), "volcano-admission-service-jobs-mutate")
					expectClusterResource(validatingWebhookConfigurationGVR(), "volcano-admission-service-jobs-validate")
					expectClusterResource(validatingWebhookConfigurationGVR(), "volcano-admission-service-podgroups-validate")

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "volcano", "admission", "enabled")
						return nil
					})

					waitForComponentUnmanaged(gvr, "volcano-vnpu-admission")
					waitForDeploymentDeleted("volcano-admission", volcanoNamespaceName)
					waitForServiceDeleted("volcano-admission-service", volcanoNamespaceName)
					waitForJobDeleted("volcano-admission-init", volcanoNamespaceName)
					waitForClusterResourceDeleted(mutatingWebhookConfigurationGVR(), "volcano-admission-service-queues-mutate")
					waitForClusterResourceDeleted(mutatingWebhookConfigurationGVR(), "volcano-admission-service-jobs-mutate")
					waitForClusterResourceDeleted(validatingWebhookConfigurationGVR(), "volcano-admission-service-jobs-validate")
					waitForClusterResourceDeleted(validatingWebhookConfigurationGVR(), "volcano-admission-service-podgroups-validate")
				})

				It("NPUOperatorVNPUDRA_FIT_3 - verifies vNPU disable cleans workloads and ready label", func() {
					nodes := listNodes()
					targetNode := firstSchedulableNode(nodes)
					labelNode(targetNode.Name, vnpuTestSelectorKey, "true")
					DeferCleanup(func() {
						removeNodeLabel(targetNode.Name, vnpuTestSelectorKey)
					})

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, false, "dra", "managed")
						setString(spec, "vnpu", "volcano", "flavor")
						setInt64(spec, int64(0), "volcano", "runAsUser")
						setInt64(spec, int64(0), "volcano", "runAsGroup")
						setBool(spec, true, "vnpu", "managed")
						setNodeSelector(spec, map[string]string{vnpuTestSelectorKey: "true"}, "vnpu")
						setBool(spec, true, "vnpu", "clientUpdate", "managed")
						setBool(spec, true, "vnpu", "devicePlugin", "managed")
						setBool(spec, true, "vnpu", "exporter", "managed")
						return nil
					})

					waitForComponentRunning(gvr, "vnpu-device-plugin")
					waitForDaemonSet("npu-device-plugin-daemonset")

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "vnpu", "managed")
						return nil
					})

					waitForComponentUnmanaged(gvr, "vnpu-device-plugin")
					waitForDaemonSetDeleted("npu-device-plugin-daemonset")
					waitForDaemonSetDeleted("npu-client-update-daemonset")
					waitForDaemonSetDeleted("xpu-exporter-daemonset")
					waitForAllNodesLabelRemoved(vnpuReadyLabelKey)
				})

				It("NPUOperatorVNPUDRA_FIT_7 - verifies Volcano flavor switch replaces vNPU Volcano images", func() {
					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setString(spec, "mindcluster", "volcano", "flavor")
						removeNestedField(spec, "volcano", "runAsUser")
						removeNestedField(spec, "volcano", "runAsGroup")
						setBool(spec, true, "vccontroller", "managed")
						setBool(spec, true, "vcscheduler", "managed")
						return nil
					})

					waitForComponentRunning(gvr, "volcano-controller")
					waitForComponentRunning(gvr, "volcano-scheduler")
					Eventually(func(g Gomega) {
						g.Expect(deploymentImages("volcano-scheduler", volcanoNamespaceName)).NotTo(ContainElement(ContainSubstring("/vnpu/")))
						g.Expect(deploymentImages("volcano-controllers", volcanoNamespaceName)).NotTo(ContainElement(ContainSubstring("/vnpu/")))
					}, 5*time.Minute, 5*time.Second).Should(Succeed())

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setString(spec, "vnpu", "volcano", "flavor")
						setInt64(spec, int64(0), "volcano", "runAsUser")
						setInt64(spec, int64(0), "volcano", "runAsGroup")
						setBool(spec, true, "volcano", "admission", "enabled")
						return nil
					})

					waitForComponentRunning(gvr, "volcano-vnpu-scheduler")
					waitForComponentRunning(gvr, "volcano-vnpu-controller")
					expectDeploymentImageContains("volcano-scheduler", volcanoNamespaceName, "/vnpu/vc-scheduler")
					expectDeploymentImageContains("volcano-controllers", volcanoNamespaceName, "/vnpu/vc-controller-manager")
				})

				It("NPUOperatorVNPUDRA_FIT_15 - verifies vNPU single pod soft split", func() {
					nodes := listNodes()
					targetNode := firstSchedulableNode(nodes)
					labelNode(targetNode.Name, vnpuTestSelectorKey, "true")
					DeferCleanup(func() {
						removeNodeLabel(targetNode.Name, vnpuTestSelectorKey)
					})
					DeferCleanup(func() {
						deletePodIfExists("default", vnpuSoftSplitPodName)
					})

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, false, "dra", "managed")
						setString(spec, "vnpu", "volcano", "flavor")
						setInt64(spec, int64(0), "volcano", "runAsUser")
						setInt64(spec, int64(0), "volcano", "runAsGroup")
						setBool(spec, true, "volcano", "admission", "enabled")
						setBool(spec, true, "vnpu", "managed")
						setNodeSelector(spec, map[string]string{vnpuTestSelectorKey: "true"}, "vnpu")
						setBool(spec, true, "vnpu", "clientUpdate", "managed")
						setBool(spec, true, "vnpu", "devicePlugin", "managed")
						setBool(spec, true, "vnpu", "exporter", "managed")
						return nil
					})

					waitForPolicyNoComponentConflict(gvr)
					waitForComponentRunning(gvr, "vnpu-device-plugin")
					waitForComponentRunning(gvr, "volcano-vnpu-scheduler")
					waitForComponentRunning(gvr, "volcano-vnpu-controller")
					waitForNodeLabel(targetNode.Name, vnpuReadyLabelKey, vnpuReadyLabelValue)

					deletePodIfExists("default", vnpuSoftSplitPodName)
					createVNPUSoftSplitPod(vnpuSoftSplitPodName)

					pod := waitForPodRunning("default", vnpuSoftSplitPodName)
					Expect(pod.Spec.NodeName).To(Equal(targetNode.Name), "vNPU soft split pod should run on selected vNPU node")
					expectVNPUSoftSplitResources(pod)
					expectVNPUSoftSplitAnnotations(pod, targetNode.Name)
					expectPodLogsContain("default", vnpuSoftSplitPodName, "VNPU_SOFT_SPLIT_TEST=started")
				})
			})

			Context("DRA component management", Ordered, func() {
				It("NPUOperatorVNPUDRA_FIT_9/FIT_10/FIT_11/FIT_13 - verifies DRA workloads, args, DeviceClasses and soft-vNPU config", func() {
					nodes := listNodes()
					targetNode := firstSchedulableNode(nodes)
					labelNode(targetNode.Name, draTestSelectorKey, "true")
					DeferCleanup(func() {
						removeNodeLabel(targetNode.Name, draTestSelectorKey)
					})

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, false, "vnpu", "managed")
						setString(spec, "external", "volcano", "flavor")
						setBool(spec, true, "dra", "managed")
						setString(spec, "custom.npu.example.com", "dra", "driverName")
						setString(spec, "/var/run/cdi", "dra", "cdiRoot")
						setString(spec, "/var/lib/kubelet/plugins_registry", "dra", "kubeletRegistrarDirectoryPath")
						setString(spec, "/var/lib/kubelet/plugins", "dra", "kubeletPluginsDirectoryPath")
						setNodeSelector(spec, map[string]string{draTestSelectorKey: "true"}, "dra")
						setBool(spec, true, "dra", "kubeletPlugin", "managed")
						setBool(spec, true, "dra", "deviceClasses", "managed")
						setBool(spec, true, "dra", "softVNPU", "managed")
						setInt64(spec, int64(16), "dra", "softVNPU", "shareCount")
						setString(spec, "/etc/dra/dra.config", "dra", "softVNPU", "npuProfileConfigPath")
						setBool(spec, true, "dra", "softVNPU", "vcannrtInstaller", "managed")
						setBool(spec, false, "dra", "webhook", "managed")
						return nil
					})

					waitForPolicyNoComponentConflict(gvr)
					waitForComponentRunning(gvr, "dra-kubelet-plugin")
					waitForComponentRunning(gvr, "dra-vcannrt-installer")

					kubeletPluginDS := waitForDaemonSet("npu-dra-kubeletplugin")
					expectDaemonSetNodeSelector(kubeletPluginDS, draTestSelectorKey, "true")
					args := kubeletPluginDS.Spec.Template.Spec.Containers[0].Args
					for _, expectedArg := range []string{
						"--node-name=$(NODE_NAME)",
						"--device-profile=npu",
						"--driver-name=custom.npu.example.com",
						"--cdi-root=/var/run/cdi",
						"--healthcheck-port=-1",
						"--kubelet-registrar-directory-path=/var/lib/kubelet/plugins_registry",
						"--kubelet-plugins-directory-path=/var/lib/kubelet/plugins",
						"--share-count=16",
						"--npu-profile-config=/etc/dra/dra.config",
						"--chip-capabilities-configmap=vnpu-template-config",
					} {
						expectContainerArg(args, expectedArg)
					}
					expectHostPath(kubeletPluginDS, "cdi", "/var/run/cdi")
					expectHostPath(kubeletPluginDS, "kubelet-plugins", "/var/lib/kubelet/plugins")
					expectHostPath(kubeletPluginDS, "kubelet-plugins-registry", "/var/lib/kubelet/plugins_registry")

					vcannrtDS := waitForDaemonSet("vcannrt-installer")
					expectDaemonSetNodeSelector(vcannrtDS, draTestSelectorKey, "true")

					for _, name := range []string{
						"full.custom.npu.example.com",
						"hard.custom.npu.example.com",
						"elastic.custom.npu.example.com",
						"fixed.custom.npu.example.com",
						"best-effort.custom.npu.example.com",
					} {
						expectDeviceClassSelector(name, "custom.npu.example.com")
					}
					expectConfigMapData("dra-profile-config", kubeletPluginDS.Namespace, "dra.config")
					expectConfigMapData("vnpu-template-config", kubeletPluginDS.Namespace, "capabilities.yaml")
				})

				It("NPUOperatorVNPUDRA_FIT_12 - verifies custom DeviceClass template create and cleanup", func() {
					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, false, "vnpu", "managed")
						setString(spec, "external", "volcano", "flavor")
						setBool(spec, true, "dra", "managed")
						setString(spec, "npu.huawei.com", "dra", "driverName")
						setBool(spec, true, "dra", "deviceClasses", "managed")
						setSlice(spec, []interface{}{
							map[string]interface{}{
								"name":        testDeviceClassName,
								"selectorCEL": `device.driver == "npu.huawei.com" && device.attributes["npu.huawei.com"].vnpuMode == "soft"`,
							},
						}, "dra", "deviceClasses", "templates")
						return nil
					})

					waitForPolicyNoComponentConflict(gvr)
					waitForDeviceClass(testDeviceClassName)

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setSlice(spec, []interface{}{}, "dra", "deviceClasses", "templates")
						return nil
					})

					waitForDeviceClassDeleted(testDeviceClassName)
				})

				It("NPUOperatorVNPUDRA_DFT_1 - verifies component statuses expose logical components only", func() {
					waitForPolicySettled(gvr)
					expectComponentStatusesInclude(gvr, "dra-kubelet-plugin", "dra-vcannrt-installer")
					expectComponentStatusesExclude(gvr,
						"/dra/deviceclasses",
						"/dra/soft-vnpu",
						"/volcano/vnpu-crds",
						"/volcano/vnpu-namespaces",
						"/vnpu-namespace",
						"exporter-servicemonitor",
					)
					expectComponentStatusTypesValid(gvr)
				})

				It("NPUOperatorVNPUDRA_FIT_14 - verifies vNPU and DRA disable cleans managed resources", func() {
					nodes := listNodes()
					targetNode := firstSchedulableNode(nodes)
					labelNode(targetNode.Name, draTestSelectorKey, "true")
					DeferCleanup(func() {
						removeNodeLabel(targetNode.Name, draTestSelectorKey)
					})

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, false, "vnpu", "managed")
						setString(spec, "external", "volcano", "flavor")
						setBool(spec, true, "dra", "managed")
						setNodeSelector(spec, map[string]string{draTestSelectorKey: "true"}, "dra")
						setBool(spec, true, "dra", "kubeletPlugin", "managed")
						setBool(spec, true, "dra", "deviceClasses", "managed")
						setBool(spec, true, "dra", "softVNPU", "managed")
						setBool(spec, true, "dra", "softVNPU", "vcannrtInstaller", "managed")
						setBool(spec, false, "dra", "webhook", "managed")
						return nil
					})

					waitForComponentRunning(gvr, "dra-kubelet-plugin")
					draNamespace := waitForDaemonSet("npu-dra-kubeletplugin").Namespace

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "vnpu", "managed")
						setBool(spec, false, "dra", "managed")
						setString(spec, "external", "volcano", "flavor")
						return nil
					})

					waitForComponentUnmanaged(gvr, "vnpu-device-plugin")
					waitForComponentUnmanaged(gvr, "dra-kubelet-plugin")
					waitForDaemonSetDeleted("npu-device-plugin-daemonset")
					waitForDaemonSetDeleted("npu-client-update-daemonset")
					waitForDaemonSetDeleted("xpu-exporter-daemonset")
					waitForDaemonSetDeleted("npu-dra-kubeletplugin")
					waitForDaemonSetDeleted("vcannrt-installer")
					waitForConfigMapDeleted("dra-profile-config", draNamespace)
					waitForConfigMapDeleted("vnpu-template-config", draNamespace)
					waitForNoManagedDeviceClasses()
					waitForAllNodesLabelRemoved(vnpuReadyLabelKey)
				})
			})

			Context("component conflict detection", Ordered, func() {
				It("NPUOperatorVNPUDRA_FIT_20 - rejects overlapping non-empty vNPU and DRA node selectors", func() {
					nodes := listNodes()
					targetNode := firstSchedulableNode(nodes)
					labelNode(targetNode.Name, vnpuTestSelectorKey, "true")
					labelNode(targetNode.Name, draTestSelectorKey, "true")
					DeferCleanup(func() {
						removeNodeLabel(targetNode.Name, vnpuTestSelectorKey)
						removeNodeLabel(targetNode.Name, draTestSelectorKey)
					})

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, true, "vnpu", "managed")
						setBool(spec, true, "vnpu", "devicePlugin", "managed")
						setNodeSelector(spec, map[string]string{vnpuTestSelectorKey: "true"}, "vnpu", "devicePlugin")
						setBool(spec, true, "dra", "managed")
						setBool(spec, true, "dra", "kubeletPlugin", "managed")
						setNodeSelector(spec, map[string]string{draTestSelectorKey: "true"}, "dra", "kubeletPlugin")
						return nil
					})

					waitForPolicyComponentConflict(gvr, "vnpu-device-plugin", "dra-kubelet-plugin", targetNode.Name)
				})

				It("NPUOperatorVNPUDRA_FIT_21 - rejects empty node selectors as all-node conflict", func() {
					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, true, "vnpu", "managed")
						setBool(spec, true, "vnpu", "devicePlugin", "managed")
						removeNestedField(spec, "vnpu", "nodeSelector")
						removeNestedField(spec, "vnpu", "devicePlugin", "nodeSelector")
						setBool(spec, true, "dra", "managed")
						setBool(spec, true, "dra", "kubeletPlugin", "managed")
						removeNestedField(spec, "dra", "nodeSelector")
						removeNestedField(spec, "dra", "kubeletPlugin", "nodeSelector")
						return nil
					})

					waitForPolicyComponentConflict(gvr, "vnpu-device-plugin", "dra-kubelet-plugin", "<all nodes>")
				})

				It("NPUOperatorVNPUDRA_DFT_2 - verifies conflict diagnostics and recovery", func() {
					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "devicePlugin", "managed")
						setBool(spec, true, "vnpu", "managed")
						setBool(spec, true, "vnpu", "devicePlugin", "managed")
						removeNestedField(spec, "vnpu", "nodeSelector")
						removeNestedField(spec, "vnpu", "devicePlugin", "nodeSelector")
						setBool(spec, true, "dra", "managed")
						setBool(spec, true, "dra", "kubeletPlugin", "managed")
						removeNestedField(spec, "dra", "nodeSelector")
						removeNestedField(spec, "dra", "kubeletPlugin", "nodeSelector")
						return nil
					})

					waitForPolicyComponentConflict(gvr, "vnpu-device-plugin", "dra-kubelet-plugin", "<all nodes>")
					expectOperatorLogsContain("Component node selector conflict detected", "vnpu-device-plugin", "dra-kubelet-plugin")

					patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
						setBool(spec, false, "dra", "managed")
						return nil
					})

					waitForPolicyNoComponentConflict(gvr)
					waitForComponentUnmanaged(gvr, "dra-kubelet-plugin")
				})

				It("NPUOperatorVNPUDRA_FIT_22 - verifies statically disjoint selectors and selected-node deployment", func() {
					setupDisjointVNPUAndDRAPluginSelectors(gvr)

					waitForPolicyNoComponentConflict(gvr)
					vnpuDS := waitForDaemonSet("npu-device-plugin-daemonset")
					draDS := waitForDaemonSet("npu-dra-kubeletplugin")
					expectDaemonSetPodsOnSelectedNodes(vnpuDS, labels.Set{vnpuTestSelectorKey: "true"})
					expectDaemonSetPodsOnSelectedNodes(draDS, labels.Set{draTestSelectorKey: "true"})
				})
			})
		})
}

func npuClusterPolicyGVR() schema.GroupVersionResource {
	return schema.GroupVersionResource{
		Group:    "npu.openfuyao.com",
		Version:  "v1",
		Resource: "npuclusterpolicies",
	}
}

func snapshotNPUClusterPolicySpec(gvr schema.GroupVersionResource) map[string]interface{} {
	policy, err := GetNPUOperatorCR(dynamicClient, gvr)
	Expect(err).NotTo(HaveOccurred(), "should get NPUClusterPolicy before vNPU/DRA e2e")
	spec, found, err := unstructured.NestedMap(policy.Object, "spec")
	Expect(err).NotTo(HaveOccurred(), "should read NPUClusterPolicy spec")
	Expect(found).To(BeTrue(), "NPUClusterPolicy spec should exist")
	return runtime.DeepCopyJSON(spec)
}

func restoreNPUClusterPolicySpec(gvr schema.GroupVersionResource, spec map[string]interface{}) {
	err := UpdateNPUOperatorCRWithRetry(dynamicClient, ctx, gvr, func(policy *unstructured.Unstructured) error {
		return unstructured.SetNestedMap(policy.Object, runtime.DeepCopyJSON(spec), "spec")
	})
	Expect(err).NotTo(HaveOccurred(), "should restore NPUClusterPolicy spec")
}

func cleanNPUOperatorVNPUDRATestBaseline(spec map[string]interface{}) map[string]interface{} {
	baseline := runtime.DeepCopyJSON(spec)
	setString(baseline, "mindcluster", "volcano", "flavor")
	removeNestedField(baseline, "volcano", "runAsUser")
	removeNestedField(baseline, "volcano", "runAsGroup")
	removeNestedField(baseline, "volcano", "admission", "enabled")
	setBool(baseline, false, "vnpu", "managed")
	setBool(baseline, false, "dra", "managed")
	setBool(baseline, true, "vccontroller", "managed")
	setBool(baseline, true, "vcscheduler", "managed")
	return baseline
}

func setupDisjointVNPUAndDRAPluginSelectors(gvr schema.GroupVersionResource) {
	nodes := listNodes()
	schedulableNodes := schedulableNodes(nodes)
	if len(schedulableNodes) < 2 {
		Skip("当前集群少于2个可调度节点，跳过分节点部署验证")
	}

	vnpuNode := schedulableNodes[0]
	draNode := schedulableNodes[1]
	labelNode(vnpuNode.Name, vnpuTestSelectorKey, "true")
	labelNode(draNode.Name, draTestSelectorKey, "true")
	DeferCleanup(func() {
		removeNodeLabel(vnpuNode.Name, vnpuTestSelectorKey)
		removeNodeLabel(draNode.Name, draTestSelectorKey)
	})

	patchNPUClusterPolicySpec(gvr, func(spec map[string]interface{}) error {
		setBool(spec, false, "devicePlugin", "managed")
		setBool(spec, true, "vnpu", "managed")
		setBool(spec, true, "vnpu", "devicePlugin", "managed")
		setNodeSelector(spec, map[string]string{vnpuTestSelectorKey: "true"}, "vnpu", "devicePlugin")
		setBool(spec, true, "dra", "managed")
		setBool(spec, true, "dra", "kubeletPlugin", "managed")
		setNodeSelector(spec, map[string]string{draTestSelectorKey: "true"}, "dra", "kubeletPlugin")
		return nil
	})
}

func patchNPUClusterPolicySpec(gvr schema.GroupVersionResource, mutator func(map[string]interface{}) error) {
	err := UpdateNPUOperatorCRWithRetry(dynamicClient, ctx, gvr, func(policy *unstructured.Unstructured) error {
		spec, found, err := unstructured.NestedMap(policy.Object, "spec")
		if err != nil {
			return err
		}
		if !found {
			spec = map[string]interface{}{}
		}
		if err := mutator(spec); err != nil {
			return err
		}
		return unstructured.SetNestedMap(policy.Object, spec, "spec")
	})
	Expect(err).NotTo(HaveOccurred(), "should patch NPUClusterPolicy spec")
}

func ensureMap(parent map[string]interface{}, fields ...string) map[string]interface{} {
	current := parent
	for _, field := range fields {
		child, _ := current[field].(map[string]interface{})
		if child == nil {
			child = map[string]interface{}{}
			current[field] = child
		}
		current = child
	}
	return current
}

func setBool(spec map[string]interface{}, value bool, fields ...string) {
	Expect(unstructured.SetNestedField(spec, value, fields...)).To(Succeed())
}

func setString(spec map[string]interface{}, value string, fields ...string) {
	Expect(unstructured.SetNestedField(spec, value, fields...)).To(Succeed())
}

func setInt64(spec map[string]interface{}, value int64, fields ...string) {
	Expect(unstructured.SetNestedField(spec, value, fields...)).To(Succeed())
}

func setSlice(spec map[string]interface{}, value []interface{}, fields ...string) {
	Expect(unstructured.SetNestedSlice(spec, value, fields...)).To(Succeed())
}

func setNodeSelector(spec map[string]interface{}, selector map[string]string, fields ...string) {
	target := ensureMap(spec, fields...)
	selectorObj := make(map[string]interface{}, len(selector))
	for key, value := range selector {
		selectorObj[key] = value
	}
	target["nodeSelector"] = selectorObj
}

func removeNestedField(spec map[string]interface{}, fields ...string) {
	unstructured.RemoveNestedField(spec, fields...)
}

func listNodes() *corev1.NodeList {
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "should list nodes")
	Expect(nodes.Items).NotTo(BeEmpty(), "cluster should have nodes")
	return nodes
}

func firstSchedulableNode(nodes *corev1.NodeList) corev1.Node {
	schedulable := schedulableNodes(nodes)
	if len(schedulable) > 0 {
		return schedulable[0]
	}
	Skip("未找到可调度节点")
	return corev1.Node{}
}

func schedulableNodes(nodes *corev1.NodeList) []corev1.Node {
	var result []corev1.Node
	for _, node := range nodes.Items {
		if !node.Spec.Unschedulable {
			result = append(result, node)
		}
	}
	return result
}

func labelNode(nodeName, key, value string) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		node, err := clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if node.Labels == nil {
			node.Labels = map[string]string{}
		}
		node.Labels[key] = value
		_, err = clientset.CoreV1().Nodes().Update(ctx, node, metav1.UpdateOptions{})
		return err
	})
	Expect(err).NotTo(HaveOccurred(), "should label node %s", nodeName)
}

func removeNodeLabel(nodeName, key string) {
	_ = retry.RetryOnConflict(retry.DefaultRetry, func() error {
		node, err := clientset.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		delete(node.Labels, key)
		_, err = clientset.CoreV1().Nodes().Update(context.Background(), node, metav1.UpdateOptions{})
		return err
	})
}

func expectNamespace(name string) {
	Eventually(func() bool {
		_, err := clientset.CoreV1().Namespaces().Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "namespace %s should exist", name)
}

func waitForDaemonSet(name string) *appsv1.DaemonSet {
	var found *appsv1.DaemonSet
	Eventually(func() bool {
		daemonSets, err := clientset.AppsV1().DaemonSets("").List(ctx, metav1.ListOptions{})
		if err != nil {
			return false
		}
		for i := range daemonSets.Items {
			if daemonSets.Items[i].Name == name {
				found = &daemonSets.Items[i]
				return true
			}
		}
		return false
	}, 3*time.Minute, 3*time.Second).Should(BeTrue(), "DaemonSet %s should exist", name)
	return found
}

func waitForDaemonSetDeleted(name string) {
	Eventually(func() bool {
		daemonSets, err := clientset.AppsV1().DaemonSets("").List(ctx, metav1.ListOptions{})
		if err != nil {
			return false
		}
		for _, daemonSet := range daemonSets.Items {
			if daemonSet.Name == name {
				return false
			}
		}
		return true
	}, 3*time.Minute, 3*time.Second).Should(BeTrue(), "DaemonSet %s should be deleted", name)
}

func expectDaemonSetNodeSelector(daemonSet *appsv1.DaemonSet, key, value string) {
	Expect(daemonSet.Spec.Template.Spec.NodeSelector).To(HaveKeyWithValue(key, value),
		"DaemonSet %s should have expected nodeSelector", daemonSet.Name)
}

func expectDaemonSetPodsOnSelectedNodes(daemonSet *appsv1.DaemonSet, selector labels.Set) {
	Eventually(func(g Gomega) {
		pods, err := clientset.CoreV1().Pods(daemonSet.Namespace).List(ctx, metav1.ListOptions{
			LabelSelector: labels.SelectorFromSet(daemonSet.Spec.Selector.MatchLabels).String(),
		})
		g.Expect(err).NotTo(HaveOccurred(), "should list DaemonSet pods")
		g.Expect(pods.Items).NotTo(BeEmpty(), "DaemonSet %s should have pods", daemonSet.Name)
		for _, pod := range pods.Items {
			node, err := clientset.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
			g.Expect(err).NotTo(HaveOccurred(), "should get pod node")
			g.Expect(labels.SelectorFromSet(selector).Matches(labels.Set(node.Labels))).To(BeTrue(),
				"pod %s/%s should run on selected node", pod.Namespace, pod.Name)
			g.Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "pod %s/%s should be Running", pod.Namespace, pod.Name)
		}
	}, 5*time.Minute, 5*time.Second).Should(Succeed(), "DaemonSet %s pods should match selector", daemonSet.Name)
}

func expectContainerArg(args []string, expected string) {
	Expect(args).To(ContainElement(expected), "container args should contain %s", expected)
}

func expectServiceTargetPort(name, namespace string, port int32) {
	Eventually(func(g Gomega) {
		service, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "should get service %s/%s", namespace, name)
		g.Expect(service.Spec.Ports).NotTo(BeEmpty(), "service should have ports")
		found := false
		for _, servicePort := range service.Spec.Ports {
			if servicePort.Port == port || servicePort.TargetPort.IntVal == port {
				found = true
				break
			}
		}
		g.Expect(found).To(BeTrue(), "service port or targetPort should be %d", port)
	}, 2*time.Minute, 2*time.Second).Should(Succeed())
}

func waitForNodeLabel(nodeName, key, value string) {
	Eventually(func() string {
		node, err := clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return ""
		}
		return node.Labels[key]
	}, 2*time.Minute, 2*time.Second).Should(Equal(value), "node %s should have label %s=%s", nodeName, key, value)
}

func waitForAllNodesLabelRemoved(key string) {
	Eventually(func() bool {
		nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		if err != nil {
			return false
		}
		for _, node := range nodes.Items {
			if _, exists := node.Labels[key]; exists {
				return false
			}
		}
		return true
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "label %s should be removed from all nodes", key)
}

func createVNPUSoftSplitPod(name string) {
	runtimeClassName := "ascend"
	resourceRequests := corev1.ResourceList{
		corev1.ResourceName("huawei.com/vnpu-number"):     resource.MustParse("1"),
		corev1.ResourceName("huawei.com/vnpu-cores"):      resource.MustParse("5"),
		corev1.ResourceName("huawei.com/vnpu-memory.1Gi"): resource.MustParse("8"),
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"npuoperator-manual": "vnpu-single-soft",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy:      corev1.RestartPolicyNever,
			RuntimeClassName:   &runtimeClassName,
			SchedulerName:      "volcano",
			NodeSelector:       map[string]string{vnpuReadyLabelKey: vnpuReadyLabelValue, vnpuTestSelectorKey: "true"},
			ServiceAccountName: "default",
			Containers: []corev1.Container{
				{
					Name:            "app",
					Image:           npu_operator_config.ContainerImage,
					ImagePullPolicy: corev1.PullIfNotPresent,
					Command:         []string{"/bin/sh", "-c"},
					Args: []string{
						"echo VNPU_SOFT_SPLIT_TEST=started; " +
							"env | grep -E 'ASCEND|NPU|VNPU|VISIBLE|LD_LIBRARY_PATH' || true; " +
							"sleep 3600",
					},
					Resources: corev1.ResourceRequirements{
						Requests: resourceRequests,
						Limits:   resourceRequests,
					},
				},
			},
		},
	}

	_, err := clientset.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred(), "should create vNPU soft split pod")
}

func waitForPodRunning(namespace, name string) *corev1.Pod {
	var found *corev1.Pod
	Eventually(func(g Gomega) {
		pod, err := clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "should get pod %s/%s", namespace, name)
		g.Expect(pod.Status.Phase).NotTo(Equal(corev1.PodFailed), "pod %s/%s should not fail", namespace, name)
		g.Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "pod %s/%s should be Running", namespace, name)
		found = pod
	}, 5*time.Minute, 5*time.Second).Should(Succeed(), "pod %s/%s should become Running", namespace, name)
	return found
}

func expectVNPUSoftSplitResources(pod *corev1.Pod) {
	Expect(pod.Spec.Containers).NotTo(BeEmpty(), "vNPU soft split pod should have containers")
	requests := pod.Spec.Containers[0].Resources.Requests
	limits := pod.Spec.Containers[0].Resources.Limits
	for name, expected := range map[corev1.ResourceName]string{
		"huawei.com/vnpu-number":     "1",
		"huawei.com/vnpu-cores":      "5",
		"huawei.com/vnpu-memory.1Gi": "8",
	} {
		expectedQuantity := resource.MustParse(expected)
		Expect(requests[name]).To(Equal(expectedQuantity), "pod request %s should match", name)
		Expect(limits[name]).To(Equal(expectedQuantity), "pod limit %s should match", name)
	}
}

func expectVNPUSoftSplitAnnotations(pod *corev1.Pod, expectedNode string) {
	annotations := pod.GetAnnotations()
	Expect(annotations).To(HaveKeyWithValue("huawei.com/bind-phase", "success"),
		"vNPU pod should be successfully bound by volcano vNPU plugin")
	Expect(annotations).To(HaveKeyWithValue("huawei.com/vnpu-node", expectedNode),
		"vNPU pod should be bound to selected node")
	Expect(strings.TrimSpace(annotations["huawei.com/vnpu-ids-new"])).NotTo(BeEmpty(),
		"vNPU pod should contain allocated virtual device information")
}

func expectPodLogsContain(namespace, name string, expected string) {
	Eventually(func() string {
		raw, err := clientset.CoreV1().Pods(namespace).GetLogs(name, &corev1.PodLogOptions{}).Do(ctx).Raw()
		if err != nil {
			return ""
		}
		return string(raw)
	}, 2*time.Minute, 2*time.Second).Should(ContainSubstring(expected),
		"pod %s/%s logs should contain %s", namespace, name, expected)
}

func deletePodIfExists(namespace, name string) {
	err := clientset.CoreV1().Pods(namespace).Delete(context.Background(), name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		Expect(err).NotTo(HaveOccurred(), "should delete pod %s/%s", namespace, name)
	}
	Eventually(func() bool {
		_, err := clientset.CoreV1().Pods(namespace).Get(context.Background(), name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "pod %s/%s should be deleted", namespace, name)
}

func deploymentImages(name, namespace string) []string {
	deployment, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "should get deployment %s/%s", namespace, name)
	var images []string
	for _, container := range deployment.Spec.Template.Spec.Containers {
		images = append(images, container.Image)
	}
	return images
}

func expectDeploymentImageContains(name, namespace, imagePart string) {
	Eventually(func(g Gomega) {
		g.Expect(deploymentImages(name, namespace)).To(ContainElement(ContainSubstring(imagePart)))
	}, 5*time.Minute, 5*time.Second).Should(Succeed(), "deployment %s/%s image should contain %s", namespace, name, imagePart)
}

func expectHostPath(daemonSet *appsv1.DaemonSet, volumeName, path string) {
	for _, volume := range daemonSet.Spec.Template.Spec.Volumes {
		if volume.Name == volumeName && volume.HostPath != nil {
			Expect(volume.HostPath.Path).To(Equal(path), "hostPath volume %s should match", volumeName)
			return
		}
	}
	Fail(fmt.Sprintf("hostPath volume %s not found in DaemonSet %s", volumeName, daemonSet.Name))
}

func expectConfigMapData(name, namespace, key string) {
	Eventually(func(g Gomega) {
		configMap, err := clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "should get ConfigMap %s/%s", namespace, name)
		g.Expect(configMap.Data).To(HaveKey(key), "ConfigMap should contain %s", key)
		g.Expect(strings.TrimSpace(configMap.Data[key])).NotTo(BeEmpty(), "ConfigMap data %s should not be empty", key)
	}, 2*time.Minute, 2*time.Second).Should(Succeed())
}

func waitForDeployment(name, namespace string) {
	Eventually(func() bool {
		_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "Deployment %s/%s should exist", namespace, name)
}

func waitForDeploymentDeleted(name, namespace string) {
	Eventually(func() bool {
		_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 3*time.Minute, 3*time.Second).Should(BeTrue(), "Deployment %s/%s should be deleted", namespace, name)
}

func expectService(name, namespace string) {
	Eventually(func() bool {
		_, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "Service %s/%s should exist", namespace, name)
}

func waitForServiceDeleted(name, namespace string) {
	Eventually(func() bool {
		_, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 3*time.Minute, 3*time.Second).Should(BeTrue(), "Service %s/%s should be deleted", namespace, name)
}

func expectJob(name, namespace string) {
	Eventually(func() bool {
		_, err := clientset.BatchV1().Jobs(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "Job %s/%s should exist", namespace, name)
}

func waitForJobDeleted(name, namespace string) {
	Eventually(func() bool {
		_, err := clientset.BatchV1().Jobs(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 3*time.Minute, 3*time.Second).Should(BeTrue(), "Job %s/%s should be deleted", namespace, name)
}

func waitForConfigMapDeleted(name, namespace string) {
	Eventually(func() bool {
		_, err := clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 3*time.Minute, 3*time.Second).Should(BeTrue(), "ConfigMap %s/%s should be deleted", namespace, name)
}

func mutatingWebhookConfigurationGVR() schema.GroupVersionResource {
	return schema.GroupVersionResource{
		Group:    "admissionregistration.k8s.io",
		Version:  "v1",
		Resource: "mutatingwebhookconfigurations",
	}
}

func validatingWebhookConfigurationGVR() schema.GroupVersionResource {
	return schema.GroupVersionResource{
		Group:    "admissionregistration.k8s.io",
		Version:  "v1",
		Resource: "validatingwebhookconfigurations",
	}
}

func expectClusterResource(gvr schema.GroupVersionResource, name string) {
	Eventually(func() bool {
		_, err := dynamicClient.Resource(gvr).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "%s %s should exist", gvr.Resource, name)
}

func waitForClusterResourceDeleted(gvr schema.GroupVersionResource, name string) {
	Eventually(func() bool {
		_, err := dynamicClient.Resource(gvr).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 3*time.Minute, 3*time.Second).Should(BeTrue(), "%s %s should be deleted", gvr.Resource, name)
}

func deviceClassGVR() schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: "resource.k8s.io", Version: "v1", Resource: "deviceclasses"}
}

func waitForDeviceClass(name string) *unstructured.Unstructured {
	var deviceClass *unstructured.Unstructured
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(deviceClassGVR()).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		deviceClass = obj
		return true
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "DeviceClass %s should exist", name)
	return deviceClass
}

func waitForDeviceClassDeleted(name string) {
	Eventually(func() bool {
		_, err := dynamicClient.Resource(deviceClassGVR()).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "DeviceClass %s should be deleted", name)
}

func waitForNoManagedDeviceClasses() {
	Eventually(func() bool {
		list, err := dynamicClient.Resource(deviceClassGVR()).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false
		}
		for i := range list.Items {
			if list.Items[i].GetAnnotations()["openfuyao.com/dra-deviceclass-managed"] == "true" {
				return false
			}
		}
		return true
	}, 3*time.Minute, 3*time.Second).Should(BeTrue(), "operator managed DRA DeviceClasses should be deleted")
}

func expectDeviceClassSelector(name, driverName string) {
	deviceClass := waitForDeviceClass(name)
	selectorCELs := collectDeviceClassCELSelectors(deviceClass)
	Expect(selectorCELs).NotTo(BeEmpty(), "DeviceClass %s should have CEL selectors", name)
	Expect(strings.Join(selectorCELs, "\n")).To(ContainSubstring(driverName),
		"DeviceClass %s selectors should contain configured driverName", name)
}

func collectDeviceClassCELSelectors(deviceClass *unstructured.Unstructured) []string {
	var selectors []string
	deviceSelectors, _, _ := unstructured.NestedSlice(deviceClass.Object, "spec", "selectors")
	for _, item := range deviceSelectors {
		selector, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		expression, _, _ := unstructured.NestedString(selector, "cel", "expression")
		if expression != "" {
			selectors = append(selectors, expression)
		}
	}
	return selectors
}

func waitForComponentRunning(gvr schema.GroupVersionResource, componentName string) {
	waitForComponentState(gvr, componentName, "running", "Reconciled")
}

func waitForComponentUnmanaged(gvr schema.GroupVersionResource, componentName string) {
	waitForComponentState(gvr, componentName, "unmanaged", "ComponentUnmanaged")
}

func waitForComponentState(gvr schema.GroupVersionResource, componentName, expectedType, expectedReason string) {
	GinkgoWriter.Printf("Waiting for component %s to be %s/%s\n", componentName, expectedType, expectedReason)
	Eventually(func(g Gomega) {
		policy, err := GetNPUOperatorCR(dynamicClient, gvr)
		g.Expect(err).NotTo(HaveOccurred(), "should get NPUClusterPolicy")
		reason, currentType, err := GetComponentState(componentName, policy)
		g.Expect(err).NotTo(HaveOccurred(),
			"should get component state for %s, current components: %s",
			componentName, componentStatusSummary(policy))
		g.Expect(currentType).To(Equal(expectedType), "component %s type should match", componentName)
		g.Expect(reason).To(Equal(expectedReason), "component %s reason should match", componentName)
	}, 5*time.Minute, 5*time.Second).Should(Succeed(),
		"component %s should become %s/%s", componentName, expectedType, expectedReason)
}

func waitForPolicyComponentConflict(gvr schema.GroupVersionResource, expectedParts ...string) {
	Eventually(func(g Gomega) {
		policy, err := GetNPUOperatorCR(dynamicClient, gvr)
		g.Expect(err).NotTo(HaveOccurred(), "should get NPUClusterPolicy")
		phase, err, found := GetNPUClusterPolicyStatus(policy)
		g.Expect(err).NotTo(HaveOccurred(), "should get policy status")
		g.Expect(found).To(BeTrue(), "policy status should exist")
		g.Expect(phase).To(Equal("NotReady"))

		reason, message := getConditionReasonAndMessage(policy, "Error")
		g.Expect(reason).To(Equal("ComponentConflict"))
		for _, expectedPart := range expectedParts {
			g.Expect(message).To(ContainSubstring(expectedPart))
		}
	}, 2*time.Minute, 2*time.Second).Should(Succeed(), "policy should report ComponentConflict")
}

func waitForPolicyNoComponentConflict(gvr schema.GroupVersionResource) {
	Eventually(func(g Gomega) {
		policy, err := GetNPUOperatorCR(dynamicClient, gvr)
		g.Expect(err).NotTo(HaveOccurred(), "should get NPUClusterPolicy")
		reason, message := getConditionReasonAndMessage(policy, "Error")
		g.Expect(reason).NotTo(Equal("ComponentConflict"), "policy should not report ComponentConflict: %s", message)
	}, 2*time.Minute, 2*time.Second).Should(Succeed())
}

func waitForPolicySettled(gvr schema.GroupVersionResource) {
	Eventually(func() bool {
		policy, err := GetNPUOperatorCR(dynamicClient, gvr)
		if err != nil {
			return false
		}
		phase, _, found := GetNPUClusterPolicyStatus(policy)
		return found && phase != ""
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "policy should expose status after restore")
}

func waitForPolicyReady(gvr schema.GroupVersionResource) {
	GinkgoWriter.Println("Waiting for NPUClusterPolicy phase Ready")
	Eventually(func(g Gomega) {
		policy, err := GetNPUOperatorCR(dynamicClient, gvr)
		g.Expect(err).NotTo(HaveOccurred(), "should get NPUClusterPolicy")
		phase, _, found := GetNPUClusterPolicyStatus(policy)
		g.Expect(found).To(BeTrue(), "policy status should exist")
		g.Expect(phase).To(Equal("Ready"),
			"policy should be Ready, current phase=%s, conditions=%s", phase, policyConditionSummary(policy))
	}, 5*time.Minute, 5*time.Second).Should(Succeed(), "policy should be Ready after restore")
}

func expectComponentStatusesInclude(gvr schema.GroupVersionResource, expectedNames ...string) {
	Eventually(func(g Gomega) {
		names, _ := componentStatusNamesAndTypes(gvr)
		for _, expected := range expectedNames {
			g.Expect(names).To(ContainElement(ContainSubstring(expected)), "componentStatuses should include %s", expected)
		}
	}, 2*time.Minute, 2*time.Second).Should(Succeed())
}

func expectComponentStatusesExclude(gvr schema.GroupVersionResource, unexpectedNames ...string) {
	Eventually(func(g Gomega) {
		names, _ := componentStatusNamesAndTypes(gvr)
		for _, unexpected := range unexpectedNames {
			g.Expect(names).NotTo(ContainElement(ContainSubstring(unexpected)), "componentStatuses should not include %s", unexpected)
		}
	}, 2*time.Minute, 2*time.Second).Should(Succeed())
}

func expectComponentStatusTypesValid(gvr schema.GroupVersionResource) {
	validTypes := map[string]bool{
		"running":    true,
		"deploying":  true,
		"pending":    true,
		"terminated": true,
		"unmanaged":  true,
	}
	Eventually(func(g Gomega) {
		_, stateTypes := componentStatusNamesAndTypes(gvr)
		g.Expect(stateTypes).NotTo(BeEmpty(), "componentStatuses should expose state.type")
		for _, stateType := range stateTypes {
			g.Expect(validTypes).To(HaveKey(stateType), "component status type should be valid")
		}
	}, 2*time.Minute, 2*time.Second).Should(Succeed())
}

func componentStatusNamesAndTypes(gvr schema.GroupVersionResource) ([]string, []string) {
	policy, err := GetNPUOperatorCR(dynamicClient, gvr)
	Expect(err).NotTo(HaveOccurred(), "should get NPUClusterPolicy")
	componentStatuses, _, _ := unstructured.NestedSlice(policy.Object, "status", "componentStatuses")
	names := make([]string, 0, len(componentStatuses))
	stateTypes := make([]string, 0, len(componentStatuses))
	for _, item := range componentStatuses {
		component, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		name, _, _ := unstructured.NestedString(component, "name")
		stateType, _, _ := unstructured.NestedString(component, "state", "type")
		if name != "" {
			names = append(names, name)
		}
		if stateType != "" {
			stateTypes = append(stateTypes, stateType)
		}
	}
	return names, stateTypes
}

func componentStatusSummary(policy unstructured.Unstructured) string {
	componentStatuses, _, _ := unstructured.NestedSlice(policy.Object, "status", "componentStatuses")
	if len(componentStatuses) == 0 {
		return "<empty>"
	}
	summary := make([]string, 0, len(componentStatuses))
	for _, item := range componentStatuses {
		component, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		name, _, _ := unstructured.NestedString(component, "name")
		stateType, _, _ := unstructured.NestedString(component, "state", "type")
		reason, _, _ := unstructured.NestedString(component, "state", "reason")
		if name != "" {
			summary = append(summary, fmt.Sprintf("%s=%s/%s", name, stateType, reason))
		}
	}
	return strings.Join(summary, "; ")
}

func policyConditionSummary(policy unstructured.Unstructured) string {
	conditions, _, _ := unstructured.NestedSlice(policy.Object, "status", "conditions")
	if len(conditions) == 0 {
		return "<empty>"
	}
	summary := make([]string, 0, len(conditions))
	for _, item := range conditions {
		condition, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		conditionType, _, _ := unstructured.NestedString(condition, "type")
		status, _, _ := unstructured.NestedString(condition, "status")
		reason, _, _ := unstructured.NestedString(condition, "reason")
		message, _, _ := unstructured.NestedString(condition, "message")
		if conditionType != "" {
			summary = append(summary, fmt.Sprintf("%s=%s/%s/%s", conditionType, status, reason, message))
		}
	}
	return strings.Join(summary, "; ")
}

func expectOperatorLogsContain(expectedParts ...string) {
	Eventually(func(g Gomega) {
		logs := collectNPUOperatorLogs()
		g.Expect(logs).NotTo(BeEmpty(), "should collect npu-operator logs")
		for _, expected := range expectedParts {
			g.Expect(logs).To(ContainSubstring(expected), "operator logs should contain %s", expected)
		}
	}, 2*time.Minute, 5*time.Second).Should(Succeed())
}

func collectNPUOperatorLogs() string {
	pods, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return ""
	}
	sinceSeconds := int64(600)
	logs := strings.Builder{}
	for _, pod := range pods.Items {
		if !strings.Contains(pod.Name, "npu-operator") {
			continue
		}
		for _, container := range pod.Spec.Containers {
			raw, err := clientset.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{
				Container:    container.Name,
				SinceSeconds: &sinceSeconds,
			}).Do(ctx).Raw()
			if err == nil {
				logs.Write(raw)
			}
		}
	}
	return logs.String()
}

func getConditionReasonAndMessage(policy unstructured.Unstructured, conditionType string) (string, string) {
	conditions, _, _ := unstructured.NestedSlice(policy.Object, "status", "conditions")
	for _, item := range conditions {
		condition, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		currentType, _, _ := unstructured.NestedString(condition, "type")
		if currentType != conditionType {
			continue
		}
		reason, _, _ := unstructured.NestedString(condition, "reason")
		message, _, _ := unstructured.NestedString(condition, "message")
		return reason, message
	}
	return "", ""
}
