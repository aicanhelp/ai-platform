package npu_operator

import (
	"context"
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func InstalledPhase() {
	var _ = Describe("NPU Operator has been installed", Ordered, Label("npu_operator_installed", "npu"), func() {
		var installedByCurrentPhase bool
		BeforeAll(func() {
			// 执行 npu-operator 安装
			installedByCurrentPhase = EnsureNPUOperatorInstallation(clientset, ctx, nodesNPUSituation)

			GinkgoWriter.Println()
		})

		AfterAll(func() {
			GinkgoWriter.Println()

			// 执行 npu-operator 卸载
			if !installedByCurrentPhase {
				GinkgoWriter.Println("NPU Operator reused existing environment in this phase, skip uninstall")
				return
			}

			VerifyNPUOperatorUninstallation(clientset, ctx)
		})

		Context("查询所有的节点", Label("npu_operator_nodes"), func() {
			var nodes *corev1.NodeList

			BeforeEach(func() {
				var err error
				nodes, err = clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})

				Expect(err).NotTo(HaveOccurred(), "'kubectl get nodes' 执行失败")
				Expect(nodes.Items).NotTo(BeEmpty(), "在集群中没有找到任何节点")

				GinkgoWriter.Printf("检测到nodes数量:\n%d\n", len(nodes.Items))
			})

			It("669-检查master和worker节点对应的标签（多节点）", func() {
				for _, node := range nodes.Items {
					// 检测是否为master节点
					_, controlPlaneExists := node.Labels["node-role.kubernetes.io/control-plane"]
					_, masterExists := node.Labels["node-role.kubernetes.io/master"]
					masterLabelExists := controlPlaneExists && masterExists

					if masterLabelExists {
						// master节点
						GinkgoWriter.Printf("%s节点是master节点\n", node.Name)
						label := "masterselector"
						expectValue := "dls-master-node"

						// 检查master节点标签
						VerifyNodeLabels(node, label, expectValue)
					} else {
						// worker节点
						GinkgoWriter.Printf("%s节点是worker节点\n", node.Name)
						_, exists := node.Labels["masterselector"]

						Expect(exists).To(BeFalse(), "worker节点不应存在master节点标签")
						GinkgoWriter.Printf("%s节点不存在masterselector标签\n", node.Name)
					}
				}
			})

			It("670-检查容器运行时标签", func() {
				label := "openfuyao.com/container.runtime"
				expectValue := "containerd"

				// 检查每个节点的标签
				for _, node := range nodes.Items {
					VerifyNodeLabels(node, label, expectValue)
				}
			})

			It("676-检查新增NFD服务发现标签（310P-D100）", func() {
				label := "feature.node.kubernetes.io/pci-1200_19e5_d100.present"
				expectValue := "true"
				npuType := "d100"

				VerifyAllNodesLabels(nodes, nodesNPUSituation, label, expectValue, npuType)
			})

			It("677-检查新增NFD服务发现标签（310-D500）", func() {
				label := "feature.node.kubernetes.io/pci-1200_19e5_d500.present"
				expectValue := "true"
				npuType := "d500"

				VerifyAllNodesLabels(nodes, nodesNPUSituation, label, expectValue, npuType)
			})

			It("678-检查新增NFD服务发现标签（910-D801）", func() {
				label := "feature.node.kubernetes.io/pci-1200_19e5_d801.present"
				expectValue := "true"
				npuType := "d801"

				VerifyAllNodesLabels(nodes, nodesNPUSituation, label, expectValue, npuType)
			})

			It("679-检查新增NFD服务发现标签（910-D802）", func() {
				label := "feature.node.kubernetes.io/pci-1200_19e5_d802.present"
				expectValue := "true"
				npuType := "d802"

				VerifyAllNodesLabels(nodes, nodesNPUSituation, label, expectValue, npuType)
			})

			It("680-检查节点NPU设备标签", func() {
				label := "openfuyao.com/npu.present"
				expectValue := ""

				// 检查每个节点的标签
				for _, node := range nodes.Items {
					foundNPU := nodesNPUSituation[node.Name]

					// 只检测有NPU的节点
					if foundNPU == "none" {
						GinkgoWriter.Printf("%s节点不存在NPU\n", node.Name)
						continue
					} else {
						GinkgoWriter.Printf("%s 节点未找到NPU设备，跳过验证", node.Name)
					}

					VerifyNodeLabels(node, label, expectValue)
				}
			})

			It("681-检查管理节点标签（单节点）", func() {
				if len(nodes.Items) != 1 {
					GinkgoWriter.Printf("当前环境节点数为：%d，不符合单节点环境要求，跳过验证", len(nodes.Items))
				} else {
					node := nodes.Items[0]

					// master节点
					label := "masterselector"
					expectValue := "dls-master-node"

					// 检查master节点标签
					VerifyNodeLabels(node, label, expectValue)
				}
			})

			It("410-npu-operator 节点标签验证（Ascend 310P ARM）-worker标签", func() {
				// 安装流程已在BeforeAll中完成
				var testNpuType = "Ascend310P"
				var testSystemType = "arm"

				// 验证 worker 标签
				VerifyWorkerLabel(nodes, nodesNPUSituation, testNpuType, testSystemType)
			})

			It("411-npu-operator 节点标签验证（Ascend 310P X86）-worker标签", func() {
				// 安装流程已在BeforeAll中完成
				var testNpuType = "Ascend310P"
				var testSystemType = "amd"

				// 验证 worker 标签
				VerifyWorkerLabel(nodes, nodesNPUSituation, testNpuType, testSystemType)
			})

			It("412-npu-operator 节点标签验证（Atlas 800T A2-Ascend 910B/Atlas 900 A2 PoD-Ascend 910B ARM）-worker标签", func() {
				// 安装流程已在BeforeAll中完成
				var testNpuType = "Ascend910B"
				var testSystemType = "arm"

				// 验证 worker 标签
				VerifyWorkerLabel(nodes, nodesNPUSituation, testNpuType, testSystemType)
			})

			It("682-检查NPU-operator节点标签（Atalas 800I-Ascend 910B ARM）-worker标签", func() {
				// 安装流程已在BeforeAll中完成
				var testNpuType = "Ascend910B"
				var testSystemType = "arm"

				// 验证 worker 标签
				VerifyWorkerLabel(nodes, nodesNPUSituation, testNpuType, testSystemType)
			})
		})

		Context("查询所有的服务", Label("npu_operator_services"), func() {
			var services *corev1.ServiceList

			BeforeEach(func() {
				var err error
				services, err = clientset.CoreV1().Services("").List(ctx, metav1.ListOptions{})

				Expect(err).NotTo(HaveOccurred(), "'kubectl get svc -A' 应执行成功")
				Expect(services.Items).NotTo(BeEmpty(), "集群上应发现服务")

				GinkgoWriter.Printf("检测到services数量:\n%d\n", len(services.Items))
			})

			It("NPU Exporter组件功能验证-实际下发使用NPU的任务", func() {
				//查找NPU Exporter服务
				clusterIP, servicePort, found := FindNPUExporterService(services)

				// 首先检查是否找到了服务
				Expect(found).To(BeTrue(), "未找到 npu-exporter/npu-exporter-service 服务")

				// 然后检查 ClusterIP 和 servicePort 是否有效（不是空且不是 "None"）
				Expect(clusterIP).NotTo(BeEmpty(), "服务的 ClusterIP 为空")
				Expect(clusterIP).NotTo(Equal("None"), "服务是 Headless Service，没有 ClusterIP")
				Expect(servicePort).NotTo(BeZero(), "服务没有定义端口")

				GinkgoWriter.Printf("找到服务: ClusterIP=%s, Port=%d\n", clusterIP, servicePort)

				// 创建测试用的Pod
				for nodeName, npuType := range nodesNPUSituation {
					if npuType == "none" {
						GinkgoWriter.Printf("%s 节点没有npu设备，跳过验证\n", nodeName)
					} else {
						GinkgoWriter.Printf("%s 节点有npu设备：%s\n", nodeName, npuType)
						podName, containerName, err := CreateNPUOperatorTestPod(clientset, ctx, npuType)
						Expect(err).NotTo(HaveOccurred(), "应成功创建npu-operator-test-pod")

						// 验证Pod运行状态
						VerifyPodRunning(clientset, ctx, podName)
						GinkgoWriter.Printf("Pod：%s 正在运行!\n", podName)

						// 优先直连测试 Pod 所在节点上的 npu-exporter，避免 Service 负载到其他节点导致漏看指标
						metricsUrl, err := BuildNPUExporterMetricsURLForPodNode(clientset, ctx, podName)
						Expect(err).NotTo(HaveOccurred(), "应成功找到测试 Pod 所在节点上的 npu-exporter")
						VerifyMetricsThroughURL(metricsUrl, podName, containerName)

						GinkgoWriter.Printf("成功验证 container_name 为 %s， pod_name 为 %s\n", containerName, podName)

						// 删除测试用的Pod，恢复原来的环境
						err = DeleteNPUOperatorTestPod(clientset, ctx, podName)
						Expect(err).NotTo(HaveOccurred(), "'kubectl delete npu-operator pod -n "+podName+"' 应执行成功")
						// 验证Pod已删除
						VerifyNPUOperatorTestPodDeleted(clientset, ctx, podName)
						GinkgoWriter.Printf("npu-operator-test-pod已删除，原有环境已恢复")
					}
				}
			})
		})

		Context("查找NPU Operator相关的CR", Ordered, Label("npu_operator_cr"), func() {
			var (
				NPUOperatorCRD apiextensionsv1.CustomResourceDefinition
				NPUOperatorCR  unstructured.Unstructured
				gvr            schema.GroupVersionResource
			)

			BeforeEach(func() {
				// 获取环境中CRD列表
				crds, err := extensionClient.ApiextensionsV1().CustomResourceDefinitions().List(context.TODO(), metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred(), "应该成功获取CRD列表")
				Expect(len(crds.Items)).Should(BeNumerically(">=", 1), "CRD数量应为至少1个")
				GinkgoWriter.Printf("找到%d个CRD\n", len(crds.Items))

				// 过滤出NPU Operator相关的CRD
				NPUOperatorCRDs, found := FilterNPUOperatorCRDs(crds)
				Expect(found).To(BeTrue(), "应该能够找到NPU Operator CRD")
				Expect(len(NPUOperatorCRDs)).Should(BeNumerically("==", 1), "NPU Operator CRD数量应为1个")
				GinkgoWriter.Printf("成功找到NPU Operator CRD\n")

				NPUOperatorCRD = NPUOperatorCRDs[0]

				// 定义gvr结构
				gvr = schema.GroupVersionResource{
					Group:    NPUOperatorCRD.Spec.Group,
					Version:  NPUOperatorCRD.Spec.Versions[0].Name, // 使用第一个可用版本
					Resource: NPUOperatorCRD.Spec.Names.Plural,
				}

				Eventually(func() bool {
					// 找到该CRD对应的CR
					currentNPUOperatorCR, err := GetNPUOperatorCR(dynamicClient, gvr)
					Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
					NPUOperatorCR = currentNPUOperatorCR

					// 检验当前CR状态（可能由于前置test导致状态更新延迟）
					isReady, err := IsNPUOperatorCRReady(currentNPUOperatorCR)
					if err != nil {
						return false
					}
					GinkgoWriter.Printf("当前NPUOperatorCR的状态：%t\n", isReady)
					return isReady
				}, 60*time.Second, 1*time.Second).Should(BeTrue(), "NPUOperatorCR的状态应为Ready")

				GinkgoWriter.Printf("NPUOperatorCR：%s 验证成功，状态为Ready\n", NPUOperatorCR.GetName())
			})

			It("622-MindIO组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "mindioacp",
					ComponentName:    "mindioacp",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Mindioacp组件

				// 获取最新的CR
				currentNPUOperatorCR, err := GetNPUOperatorCR(dynamicClient, gvr)
				Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
				NPUOperatorCR = currentNPUOperatorCR

				verifyModificationANDEffectivenessInfo = VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "mindiotft",
					ComponentName:    "mindiotft",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Mindiotft组件
			})

			It("609-MindIO组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "mindioacp",
					ComponentName:    "mindioacp",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Mindioacp组件

				// 获取最新的CR
				currentNPUOperatorCR, err := GetNPUOperatorCR(dynamicClient, gvr)
				Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
				NPUOperatorCR = currentNPUOperatorCR

				verifyModificationANDEffectivenessInfo = VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "mindiotft",
					ComponentName:    "mindiotft",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Mindiotft组件
			})

			It("611-MindIO组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "mindioacp",
					ComponentName:    "mindioacp",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Mindioacp组件

				// 获取最新的CR
				currentNPUOperatorCR, err := GetNPUOperatorCR(dynamicClient, gvr)
				Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
				NPUOperatorCR = currentNPUOperatorCR

				verifyModificationANDEffectivenessInfo = VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "mindiotft",
					ComponentName:    "mindiotft",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Mindiotft组件
			})

			It("610-MindIO ACP组件managed与镜像异常状态验证", func() {
				info := VerifyManagedAndInvalidImageInfo{
					Clientset:               clientset,
					DynamicClient:           dynamicClient,
					Ctx:                     ctx,
					Gvr:                     gvr,
					ModificationName:        "mindioacp",
					ComponentName:           "mindioacp",
					PodPrefix:               "mindio-acp",
					InvalidRepositorySuffix: "-invalid-610",
				}

				VerifyComponentManagedAndInvalidImageState(&info)
			})

			It("612-NPU Exporter组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "exporter",
					ComponentName:    "npu-exporter",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证NPU Exporter组件
			})

			It("613-NPU Exporter组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "exporter",
					ComponentName:    "npu-exporter",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证NPU Exporter组件
			})

			It("615-NPU Exporter组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "exporter",
					ComponentName:    "npu-exporter",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证NPU Exporter组件
			})

			It("614-NPU Exporter组件managed与镜像异常状态验证", func() {
				info := VerifyManagedAndInvalidImageInfo{
					Clientset:               clientset,
					DynamicClient:           dynamicClient,
					Ctx:                     ctx,
					Gvr:                     gvr,
					ModificationName:        "exporter",
					ComponentName:           "npu-exporter",
					PodPrefix:               "npu-exporter",
					InvalidRepositorySuffix: "-invalid-614",
				}

				VerifyComponentManagedAndInvalidImageState(&info)
			})

			It("616-Volcano Scheduler组件managed与镜像异常状态验证", func() {
				info := VerifyManagedAndInvalidImageInfo{
					Clientset:               clientset,
					DynamicClient:           dynamicClient,
					Ctx:                     ctx,
					Gvr:                     gvr,
					ModificationName:        "vcscheduler",
					ComponentName:           "volcano-scheduler",
					PodPrefix:               "volcano-scheduler",
					InvalidRepositorySuffix: "-invalid-616",
				}

				VerifyComponentManagedAndInvalidImageState(&info)
			})

			It("617-Volcano组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "vccontroller",
					ComponentName:    "volcano-controller",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Volcano Controller组件

				// 获取最新的CR
				currentNPUOperatorCR, err := GetNPUOperatorCR(dynamicClient, gvr)
				Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
				NPUOperatorCR = currentNPUOperatorCR

				verifyModificationANDEffectivenessInfo = VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "vcscheduler",
					ComponentName:    "volcano-scheduler",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Volcano Scheduler组件
			})

			It("628-Volcano组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "vccontroller",
					ComponentName:    "volcano-controller",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Volcano Controller组件

				// 获取最新的CR
				currentNPUOperatorCR, err := GetNPUOperatorCR(dynamicClient, gvr)
				Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
				NPUOperatorCR = currentNPUOperatorCR

				verifyModificationANDEffectivenessInfo = VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "vcscheduler",
					ComponentName:    "volcano-scheduler",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Volcano Scheduler组件
			})

			It("629-Volcano组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "vccontroller",
					ComponentName:    "volcano-controller",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Volcano Controller组件

				// 获取最新的CR
				currentNPUOperatorCR, err := GetNPUOperatorCR(dynamicClient, gvr)
				Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
				NPUOperatorCR = currentNPUOperatorCR

				verifyModificationANDEffectivenessInfo = VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "vcscheduler",
					ComponentName:    "volcano-scheduler",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Volcano Scheduler组件
			})

			It("618-Resilience组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "rscontroller",
					ComponentName:    "resilience-controller",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Resilience组件
			})

			It("619-Resilience组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "rscontroller",
					ComponentName:    "resilience-controller",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Resilience组件
			})

			It("621-Resilience组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "rscontroller",
					ComponentName:    "resilience-controller",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Resilience组件
			})

			It("620-Resilience Controller组件managed与镜像异常状态验证", func() {
				info := VerifyManagedAndInvalidImageInfo{
					Clientset:               clientset,
					DynamicClient:           dynamicClient,
					Ctx:                     ctx,
					Gvr:                     gvr,
					ModificationName:        "rscontroller",
					ComponentName:           "resilience-controller",
					PodPrefix:               "resilience-controller",
					InvalidRepositorySuffix: "-invalid-620",
				}

				VerifyComponentManagedAndInvalidImageState(&info)
			})

			It("623-NodeD组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "nodeD",
					ComponentName:    "noded",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证NodeD组件
			})

			It("634-NodeD组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "nodeD",
					ComponentName:    "noded",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证NodeD组件
			})

			It("635-NodeD组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "nodeD",
					ComponentName:    "noded",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证NodeD组件
			})

			It("624-ClusterD组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "clusterd",
					ComponentName:    "clusterd",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证ClusterD组件
			})

			It("625-ClusterD组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "clusterd",
					ComponentName:    "clusterd",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证ClusterD组件
			})

			It("627-ClusterD组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "clusterd",
					ComponentName:    "clusterd",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证ClusterD组件
			})

			It("630-Ascend Operator组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "trainer",
					ComponentName:    "trainer",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Operator组件
			})

			It("631-Ascend Operator组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "trainer",
					ComponentName:    "trainer",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Operator组件
			})

			It("633-Ascend Operator组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "trainer",
					ComponentName:    "trainer",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Operator组件
			})

			It("638-Ascend Docker Runtime组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "ociRuntime",
					ComponentName:    "oci-runtime",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Docker Runtime组件
			})

			It("648-Ascend Docker Runtime组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "ociRuntime",
					ComponentName:    "oci-runtime",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Docker Runtime组件
			})

			It("649-Ascend Docker Runtime组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "ociRuntime",
					ComponentName:    "oci-runtime",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Docker Runtime组件
			})

			It("639-Ascend Device Plugin组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "devicePlugin",
					ComponentName:    "device-plugin",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Device Plugin组件
			})

			It("640-Ascend Device Plugin组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "devicePlugin",
					ComponentName:    "device-plugin",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Device Plugin组件
			})

			It("642-Ascend Device Plugin组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "devicePlugin",
					ComponentName:    "device-plugin",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Device Plugin组件
			})

			It("644-Ascend Driver & Firmware组件状态验证-prevState.type", func() {
				var verifyField = "type"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "driver",
					ComponentName:    "driver",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Driver & Firmware组件
			})

			It("645-Ascend Driver & Firmware组件状态验证-prevState.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "driver",
					ComponentName:    "driver",
				}

				VerifyComponentManagedFalsePrevState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Driver & Firmware组件
			})

			It("647-Ascend Driver & Firmware组件状态验证-State.reason", func() {
				var verifyField = "reason"

				verifyModificationANDEffectivenessInfo := VerifyModificationANDEffectivenessInfo{
					DynamicClient:    dynamicClient,
					Ctx:              ctx,
					NPUOperatorCR:    NPUOperatorCR,
					Gvr:              gvr,
					ModificationName: "driver",
					ComponentName:    "driver",
				}

				VerifyComponentManagedFalseState(&verifyModificationANDEffectivenessInfo, verifyField) // 验证Ascend Driver & Firmware组件
			})

			It("655-ClusterD组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"clusterd"},
					PodNames:          []string{"clusterd"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("ClusterD组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("656-ClusterD组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"clusterd"},
					PodNames:          []string{"clusterd"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("ClusterD组件修改 managed=true 安装相关Pod验证成功")
			})

			It("657-Resilience组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"rscontroller"},
					PodNames:          []string{"resilience-controller"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("Resilience组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("658-Resilience组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"rscontroller"},
					PodNames:          []string{"resilience-controller"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("Resilience组件修改 managed=true 安装相关Pod验证成功")
			})

			It("659-MingIO组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"mindioacp", "mindiotft"},
					PodNames:          []string{"mindio-acp", "mindio-tft"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("MingIO组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("660-MingIO组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"mindioacp", "mindiotft"},
					PodNames:          []string{"mindio-acp", "mindio-tft"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("MingIO组件修改 managed=true 安装相关Pod验证成功")
			})

			It("661-NPU Exporter组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"exporter"},
					PodNames:          []string{"npu-exporter"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("NPU Exporter组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("650-NPU Exporter组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"exporter"},
					PodNames:          []string{"npu-exporter"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("NPU Exporter组件修改 managed=true 安装相关Pod验证成功")
			})

			It("675-Ascend Docker Runtime组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"ociRuntime"},
					PodNames:          []string{"ascend-runtime-containerd"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("Ascend Docker Runtime组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("662-Ascend Docker Runtime组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"ociRuntime"},
					PodNames:          []string{"ascend-runtime-containerd"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)

				// 验证 Ascend Docker Runtime 功能
				VerifyAscendDockerRuntimeOnNode(nodesNPUSituation)

				GinkgoWriter.Println("Ascend Docker Runtime组件修改 managed=true 安装相关Pod验证成功")
			})

			It("663-Ascend Device Plugin组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"devicePlugin"},
					PodNames:          []string{"ascend-device-plugin"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("Ascend Device Plugin组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("664-Ascend Device Plugin组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"devicePlugin"},
					PodNames:          []string{"ascend-device-plugin"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)

				// 验证 Ascend Device Plugin 功能
				VerifyAscendDevicePlugin(clientset, ctx, nodesNPUSituation)

				GinkgoWriter.Println("Ascend Device Plugin组件修改 managed=true 安装相关Pod验证成功")
			})

			It("665-Ascend Operator组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"trainer"},
					PodNames:          []string{"ascend-operator-manager"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("Ascend Operator组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("666-Ascend Operator组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"trainer"},
					PodNames:          []string{"ascend-operator-manager"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("Ascend Operator组件修改 managed=true 安装相关Pod验证成功")
			})

			It("667-NodeD组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"nodeD"},
					PodNames:          []string{"noded"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("NodeD组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("668-NodeD组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"nodeD"},
					PodNames:          []string{"noded"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("NodeD组件修改 managed=true 安装相关Pod验证成功")
			})

			It("673-Ascend Driver & Firmware组件验证修改 managed=false 卸载相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"driver"},
					PodNames:          []string{"npu-driver"},
				}

				// 验证 managed=false Pod 卸载
				VerifyComponentManagedFalsePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("Ascend Driver & Firmware组件修改 managed=false 卸载相关Pod验证成功")
			})

			It("674-Ascend Driver & Firmware组件验证修改 managed=true 安装相关Pod", func() {
				verifyPodStateInfo := VerifyPodStateInfo{
					Clientset:         clientset,
					DynamicClient:     dynamicClient,
					Ctx:               ctx,
					NPUOperatorCR:     NPUOperatorCR,
					Gvr:               gvr,
					ModificationNames: []string{"driver"},
					PodNames:          []string{"npu-driver"},
				}

				// 验证 managed=true Pod 安装
				VerifyComponentManagedTruePodState(&verifyPodStateInfo)
				GinkgoWriter.Println("Ascend Driver & Firmware组件修改 managed=true 安装相关Pod验证成功")
			})
		})
	})
}
