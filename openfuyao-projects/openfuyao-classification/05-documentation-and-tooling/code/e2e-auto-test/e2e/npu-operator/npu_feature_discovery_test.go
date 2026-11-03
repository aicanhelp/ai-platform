package npu_operator

import (
	"os"
	"strings"
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	keepNPUFeatureDiscoveryOperatorEnv = "KEEP_NPU_FEATURE_DISCOVERY_OPERATOR"
	skipNPUFeatureDiscoveryInstallEnv  = "SKIP_NPU_FEATURE_DISCOVERY_INSTALL"
	nfdLabelSyncTimeout                = 3 * time.Minute
	nfdLabelSyncPolling                = 10 * time.Second
)

func envFlagEnabled(key string) bool {
	value := strings.TrimSpace(os.Getenv(key))
	return strings.EqualFold(value, "true") || value == "1"
}

func eventuallyVerifyFeatureDiscoveryLabels(verify func(nodes *corev1.NodeList)) {
	Eventually(func() []string {
		currentNodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		if err != nil {
			return []string{err.Error()}
		}
		if len(currentNodes.Items) == 0 {
			return []string{"no nodes found in cluster"}
		}

		return InterceptGomegaFailures(func() {
			verify(currentNodes)
		})
	}, nfdLabelSyncTimeout, nfdLabelSyncPolling).Should(BeEmpty(),
		"npu-feature-discovery labels should converge within the retry window")
}

func verifyCommonA3FeatureDiscoveryLabels(node corev1.Node) {
	VerifyNodeLabels(node, "node-role.kubernetes.io/worker", "worker")
	VerifyNodeLabels(node, "workerselector", "dls-worker-node")
	VerifyHostArchLabel(node)
	VerifyNodeLabels(node, "accelerator", "huawei-Ascend910")
}

func FeatureDiscoveryPhase() {
	var _ = Describe("NPU Feature Discovery", Ordered, Label("npu_feature_discovery", "npu"), Label("nfd_label_cases"), func() {
		var installedByCurrentPhase bool
		BeforeAll(func() {
			installedByCurrentPhase = false
			if envFlagEnabled(skipNPUFeatureDiscoveryInstallEnv) {
				GinkgoWriter.Printf("%s 已设置，跳过 NPU Operator 安装，直接复用当前环境\n", skipNPUFeatureDiscoveryInstallEnv)
				return
			}

			installedByCurrentPhase = EnsureNPUOperatorInstallation(clientset, ctx, nodesNPUSituation)
			GinkgoWriter.Println()
		})

		AfterAll(func() {
			GinkgoWriter.Println()

			if envFlagEnabled(keepNPUFeatureDiscoveryOperatorEnv) {
				GinkgoWriter.Printf("%s 已设置，保留当前 NPU Operator 环境，不执行卸载\n", keepNPUFeatureDiscoveryOperatorEnv)
				return
			}

			if !installedByCurrentPhase {
				GinkgoWriter.Println("NPU Operator reused existing environment in this phase, skip uninstall")
				return
			}

			VerifyNPUOperatorUninstallation(clientset, ctx)
		})

		Context("查询所有节点", Label("npu_feature_discovery_nodes"), func() {
			var nodes *corev1.NodeList

			BeforeEach(func() {
				var err error
				nodes, err = clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})

				Expect(err).NotTo(HaveOccurred(), "'kubectl get nodes' 执行失败")
				Expect(nodes.Items).NotTo(BeEmpty(), "在集群中没有找到任何节点")

				GinkgoWriter.Printf("检测到 nodes 数量: %d\n", len(nodes.Items))
			})

			It("676-检查新增NFD服务发现标签（310P-D100）", func() {
				targetNodes := FindNodesByNPUType(nodes, nodesNPUSituation, "d100")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 d100 卡型节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					VerifyAllNodesLabels(currentNodes, nodesNPUSituation,
						"feature.node.kubernetes.io/pci-1200_19e5_d100.present", "true", "d100")
				})
			})

			It("677-检查新增NFD服务发现标签（310-D500）", func() {
				targetNodes := FindNodesByNPUType(nodes, nodesNPUSituation, "d500")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 d500 卡型节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					VerifyAllNodesLabels(currentNodes, nodesNPUSituation,
						"feature.node.kubernetes.io/pci-1200_19e5_d500.present", "true", "d500")
				})
			})

			It("678-检查新增NFD服务发现标签（910-D801）", func() {
				targetNodes := FindNodesByNPUType(nodes, nodesNPUSituation, "d801")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 d801 卡型节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					VerifyAllNodesLabels(currentNodes, nodesNPUSituation,
						"feature.node.kubernetes.io/pci-1200_19e5_d801.present", "true", "d801")
				})
			})

			It("679-检查新增NFD服务发现标签（910-D802）", func() {
				targetNodes := FindNodesByNPUType(nodes, nodesNPUSituation, "d802")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 d802 卡型节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					VerifyAllNodesLabels(currentNodes, nodesNPUSituation,
						"feature.node.kubernetes.io/pci-1200_19e5_d802.present", "true", "d802")
				})
			})

			It("680-检查节点NPU设备标签", func() {
				targetNodes := FindNodesByRoleAndNPU(nodes, nodesNPUSituation, true, true)
				targetNodes = append(targetNodes, FindNodesByRoleAndNPU(nodes, nodesNPUSituation, false, true)...)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在包含昇腾AI处理器的节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					for _, node := range currentNodes.Items {
						npuType := nodesNPUSituation[node.Name]
						if npuType == "none" {
							VerifyNodeLabelAbsent(node, "openfuyao.com/npu.present")
							continue
						}

						VerifyNodeExpectedNPUFeatureLabels(node, npuType)
						VerifyNodeLabels(node, "openfuyao.com/npu.present", "")
					}
				})
			})

			It("683-NPU Feature Discovery-容器运行时标签验证（containerd）", func() {
				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					for _, node := range currentNodes.Items {
						VerifyNodeLabels(node, "openfuyao.com/container.runtime", "containerd")
					}
				})
			})

			It("684-NPU Feature Discovery-管理节点标签验证（节点不含昇腾AI处理器）", func() {
				targetNodes := FindNodesByRoleAndNPU(nodes, nodesNPUSituation, true, false)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在不含昇腾AI处理器的管理节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByRoleAndNPU(currentNodes, nodesNPUSituation, true, false)
					for _, node := range currentTargetNodes {
						VerifyNodeHasNoNPUFeatureLabels(node)
						VerifyNodeRoleLabels(node, true, false, false)
					}
				})
			})

			It("685-NPU Feature Discovery-管理节点标签验证（节点包含昇腾AI处理器）", func() {
				targetNodes := FindNodesByRoleAndNPU(nodes, nodesNPUSituation, true, true)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在包含昇腾AI处理器的管理节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByRoleAndNPU(currentNodes, nodesNPUSituation, true, true)
					for _, node := range currentTargetNodes {
						VerifyNodeExpectedNPUFeatureLabels(node, nodesNPUSituation[node.Name])
						VerifyNodeRoleLabels(node, true, true, true)
					}
				})
			})

			It("686-NPU Feature Discovery-计算节点标签验证（节点不含昇腾AI处理器）", func() {
				targetNodes := FindNodesByRoleAndNPU(nodes, nodesNPUSituation, false, false)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在不含昇腾AI处理器的计算节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByRoleAndNPU(currentNodes, nodesNPUSituation, false, false)
					for _, node := range currentTargetNodes {
						Expect(IsManagerNode(node)).To(BeFalse(), node.Name+" 应为计算节点")
						VerifyNodeHasNoNPUFeatureLabels(node)
						VerifyNodeRoleLabels(node, false, true, true)
					}
				})
			})

			It("687-NPU Feature Discovery-计算节点标签验证（节点包含昇腾AI处理器）", func() {
				targetNodes := FindNodesByRoleAndNPU(nodes, nodesNPUSituation, false, true)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在包含昇腾AI处理器的计算节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByRoleAndNPU(currentNodes, nodesNPUSituation, false, true)
					for _, node := range currentTargetNodes {
						Expect(IsManagerNode(node)).To(BeFalse(), node.Name+" 应为计算节点")
						VerifyNodeExpectedNPUFeatureLabels(node, nodesNPUSituation[node.Name])
						VerifyNodeRoleLabels(node, false, true, true)
					}
				})
			})

			It("688-NPU Feature Discovery-节点上有昇腾AI处理器标签验证", func() {
				targetNodes := FindNodesByRoleAndNPU(nodes, nodesNPUSituation, true, true)
				targetNodes = append(targetNodes, FindNodesByRoleAndNPU(nodes, nodesNPUSituation, false, true)...)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在包含昇腾AI处理器的节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByRoleAndNPU(currentNodes, nodesNPUSituation, true, true)
					currentTargetNodes = append(currentTargetNodes, FindNodesByRoleAndNPU(currentNodes, nodesNPUSituation, false, true)...)
					for _, node := range currentTargetNodes {
						VerifyNodeExpectedNPUFeatureLabels(node, nodesNPUSituation[node.Name])
						VerifyNodeLabels(node, "openfuyao.com/npu.present", "")
					}
				})
			})

			It("689-NPU Feature Discovery-节点上无昇腾AI处理器标签验证", func() {
				targetNodes := FindNodesByRoleAndNPU(nodes, nodesNPUSituation, true, false)
				targetNodes = append(targetNodes, FindNodesByRoleAndNPU(nodes, nodesNPUSituation, false, false)...)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在不含昇腾AI处理器的节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByRoleAndNPU(currentNodes, nodesNPUSituation, true, false)
					currentTargetNodes = append(currentTargetNodes, FindNodesByRoleAndNPU(currentNodes, nodesNPUSituation, false, false)...)
					for _, node := range currentTargetNodes {
						VerifyNodeHasNoNPUFeatureLabels(node)
						VerifyNodeLabelAbsent(node, "openfuyao.com/npu.present")
					}
				})
			})

			It("690-NPU Feature Discovery-架构标签验证", func() {
				targetNodes := FindNodesByRoleAndNPU(nodes, nodesNPUSituation, true, true)
				targetNodes = append(targetNodes, FindNodesByRoleAndNPU(nodes, nodesNPUSituation, false, true)...)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在包含昇腾AI处理器的节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					for _, node := range currentNodes.Items {
						if nodesNPUSituation[node.Name] == "none" {
							continue
						}
						VerifyHostArchLabel(node)
					}
				})
			})

			It("NFD_A3_FIT_1-验证900 A3 SuperPod服务器相关标签", func() {
				nodeCardModels := LoadNodeCardModelsOrSkip(nodes)
				targetNodes := FindNodesByCardModels(nodes, nodeCardModels, "Atlas 900 A3 Pod")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 Atlas 900 A3 Pod 节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByCardModels(currentNodes, nodeCardModels, "Atlas 900 A3 Pod")
					for _, node := range currentTargetNodes {
						verifyCommonA3FeatureDiscoveryLabels(node)
						VerifyNodeLabels(node, "accelerator-type", "module-a3-16-super-pod")
					}
				})
			})

			It("NFD_A3_FIT_2-验证800I A3服务器相关标签", func() {
				nodeCardModels := LoadNodeCardModelsOrSkip(nodes)
				targetNodes := FindNodesByCardModels(nodes, nodeCardModels, "Atlas 800I A3")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 Atlas 800I A3 节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByCardModels(currentNodes, nodeCardModels, "Atlas 800I A3")
					for _, node := range currentTargetNodes {
						verifyCommonA3FeatureDiscoveryLabels(node)
						VerifyNodeLabels(node, "accelerator-type", "module-a3-16")
						VerifyNodeLabels(node, "server-usage", "infer")
					}
				})
			})

			It("691-NPU Feature Discovery-accelerator和accelerator type标签验证（Atlas 800T A2训练服务器）", func() {
				nodeBoardIDs := LoadNodeBoardIDsOrSkip(nodes)
				targetNodes := FindNodesByBoardIDs(nodes, nodeBoardIDs, GetBoardIDsByServerModel("Atlas_800_A2")...)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 board id 命中 Atlas_800_A2 映射的节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByBoardIDs(currentNodes, nodeBoardIDs, GetBoardIDsByServerModel("Atlas_800_A2")...)
					for _, node := range currentTargetNodes {
						VerifyNodeLabels(node, "accelerator", "huawei-Ascend910")
						VerifyNodeLabels(node, "accelerator-type", "module-910b-8")
					}
				})
			})

			It("692-NPU Feature Discovery-accelerator和accelerator type标签验证（Atlas 800I A2推理服务器）", func() {
				nodeCardModels := LoadNodeCardModelsOrSkip(nodes)
				targetNodes := FindNodesByCardModels(nodes, nodeCardModels, "Atlas 800I A2")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 Atlas 800I A2 节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByCardModels(currentNodes, nodeCardModels, "Atlas 800I A2")
					for _, node := range currentTargetNodes {
						VerifyNodeLabels(node, "accelerator", "huawei-Ascend910")
						VerifyNodeLabels(node, "accelerator-type", "module-910b-8")
					}
				})
			})

			It("693-NPU Feature Discovery-server usage标签验证（Atlas 800I A2推理服务器）", func() {
				nodeCardModels := LoadNodeCardModelsOrSkip(nodes)
				targetNodes := FindNodesByCardModels(nodes, nodeCardModels, "Atlas 800I A2")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 Atlas 800I A2 节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByCardModels(currentNodes, nodeCardModels, "Atlas 800I A2")
					for _, node := range currentTargetNodes {
						VerifyNodeLabels(node, "server-usage", "infer")
					}
				})
			})

			It("694-NPU Feature Discovery-server usage标签验证（除Atlas 800I A2/Atlas 800I A3外）", func() {
				nodeCardModels := LoadNodeCardModelsOrSkip(nodes)
				targetNodes := FindNodesExcludingCardModels(nodes, nodeCardModels, "Atlas 800I A2", "Atlas 800I A3")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在除 Atlas 800I A2/Atlas 800I A3 外的节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesExcludingCardModels(currentNodes, nodeCardModels, "Atlas 800I A2", "Atlas 800I A3")
					for _, node := range currentTargetNodes {
						VerifyNodeLabelAbsent(node, "server-usage")
					}
				})
			})

			It("695-NPU Feature Discovery-节点RDMA网卡的理论传输速度", func() {
				nodeRDMAInfo := LoadNodeRDMAInfoOrSkip(nodes)
				targetNodes := FindNodesByRDMASpeedInfo(nodes, nodeRDMAInfo)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在独立探测到活动 RDMA 网卡的节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByRDMASpeedInfo(currentNodes, nodeRDMAInfo)
					for _, node := range currentTargetNodes {
						VerifyNodeExpectedRDMASpeedLabels(node, nodeRDMAInfo[node.Name])
					}
				})
			})

			It("696-NPU Feature Discovery-节点RDMA网卡的PCIe带宽", func() {
				nodeRDMAInfo := LoadNodeRDMAInfoOrSkip(nodes)
				targetNodes := FindNodesByRDMAPCIeInfo(nodes, nodeRDMAInfo)
				if len(targetNodes) == 0 {
					Skip("当前环境不存在独立探测到活动 RDMA 网卡的节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByRDMAPCIeInfo(currentNodes, nodeRDMAInfo)
					for _, node := range currentTargetNodes {
						VerifyNodeExpectedRDMAPCIeLabels(node, nodeRDMAInfo[node.Name])
					}
				})
			})

			It("697-NPU Feature Discovery-NPU卡侧RoCE网卡总带宽（Atlas 800I A2推理服务器）", func() {
				nodeCardModels := LoadNodeCardModelsOrSkip(nodes)
				targetNodes := FindNodesByCardModels(nodes, nodeCardModels, "Atlas 800I A2")
				if len(targetNodes) == 0 {
					Skip("当前环境不存在 Atlas 800I A2 节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					currentTargetNodes := FindNodesByCardModels(currentNodes, nodeCardModels, "Atlas 800I A2")
					for _, node := range currentTargetNodes {
						VerifyNodeLabels(node, "card-side-roce-bandwidth", "200Gbps")
					}
				})
			})

			It("698-NPU Feature Discovery-NPU卡侧RoCE网卡总带宽（除Atlas 800I A2/Atlas 800I A3/Atlas 900 A3 Pod外）", func() {
				nodeCardModels := LoadNodeCardModelsOrSkip(nodes)
				var targetNodes []corev1.Node
				for _, node := range nodes.Items {
					if nodesNPUSituation[node.Name] == "none" {
						continue
					}

					cardModel, exists := nodeCardModels[node.Name]
					if !exists {
						continue
					}
					if cardModel == "Atlas 800I A2" || cardModel == "Atlas 800I A3" || cardModel == "Atlas 900 A3 Pod" {
						continue
					}

					targetNodes = append(targetNodes, node)
				}
				if len(targetNodes) == 0 {
					Skip("当前环境不存在除 Atlas 800I A2/Atlas 800I A3/Atlas 900 A3 Pod 外的昇腾节点")
				}

				eventuallyVerifyFeatureDiscoveryLabels(func(currentNodes *corev1.NodeList) {
					var currentTargetNodes []corev1.Node
					for _, node := range currentNodes.Items {
						if nodesNPUSituation[node.Name] == "none" {
							continue
						}

						cardModel, exists := nodeCardModels[node.Name]
						if !exists {
							continue
						}
						if cardModel == "Atlas 800I A2" || cardModel == "Atlas 800I A3" || cardModel == "Atlas 900 A3 Pod" {
							continue
						}

						currentTargetNodes = append(currentTargetNodes, node)
					}

					for _, node := range currentTargetNodes {
						VerifyNodeLabels(node, "card-side-roce-bandwidth", "unknown")
					}
				})
			})

			It("699-NPU Feature Discovery-手动删除由NPU Feature Discovery添加的节点标签", func() {
				var targetNode *corev1.Node
				for _, node := range nodes.Items {
					if nodesNPUSituation[node.Name] != "none" {
						nodeCopy := node
						targetNode = &nodeCopy
						break
					}
				}
				if targetNode == nil && len(nodes.Items) > 0 {
					nodeCopy := nodes.Items[0]
					targetNode = &nodeCopy
				}
				if targetNode == nil {
					Skip("当前环境不存在可用于标签回补验证的节点")
				}

				currentNode, err := clientset.CoreV1().Nodes().Get(ctx, targetNode.Name, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred(), "should successfully get current node before deleting labels")
				expectedLabels := SnapshotRecoverableNPUFeatureDiscoveryLabels(*currentNode)
				Expect(expectedLabels).NotTo(BeEmpty(), targetNode.Name+" 节点当前不存在可回补的 npu-feature-discovery 标签")

				nodeCopy := currentNode.DeepCopy()
				for key := range expectedLabels {
					delete(nodeCopy.Labels, key)
				}

				_, err = clientset.CoreV1().Nodes().Update(ctx, nodeCopy, metav1.UpdateOptions{})
				Expect(err).NotTo(HaveOccurred(), "should successfully delete managed labels from target node")

				Eventually(func(g Gomega) {
					updatedNode, getErr := clientset.CoreV1().Nodes().Get(ctx, targetNode.Name, metav1.GetOptions{})
					g.Expect(getErr).NotTo(HaveOccurred())
					g.Expect(SnapshotRecoverableNPUFeatureDiscoveryLabels(*updatedNode)).To(Equal(expectedLabels))
				}, 3*time.Minute, 10*time.Second).Should(Succeed())
			})

			It("700-NPU Feature Discovery-重启后标签一致性验证", func() {
				nfdPods, err := GetSpecifiedPods(clientset, ctx, "npu-feature-discovery")
				Expect(err).NotTo(HaveOccurred(), "should successfully find npu-feature-discovery pods")
				Expect(nfdPods).NotTo(BeEmpty(), "should find at least one npu-feature-discovery pod")

				var targetPod *corev1.Pod
				for _, pod := range nfdPods {
					if pod.Status.Phase == corev1.PodRunning {
						podCopy := pod
						targetPod = &podCopy
						break
					}
				}
				if targetPod == nil {
					Skip("当前没有 Running 状态的 npu-feature-discovery pod")
				}

				targetNode, err := clientset.CoreV1().Nodes().Get(ctx, targetPod.Spec.NodeName, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred(), "should successfully get target node before restarting npu-feature-discovery pod")

				expectedLabels := SnapshotManagedNPUFeatureDiscoveryLabels(*targetNode)
				Expect(expectedLabels).NotTo(BeEmpty(), targetNode.Name+" 节点当前不存在受管的 npu-feature-discovery 标签")

				err = clientset.CoreV1().Pods(targetPod.Namespace).Delete(ctx, targetPod.Name, metav1.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred(), "should successfully delete npu-feature-discovery pod")

				Eventually(func() bool {
					_, getErr := clientset.CoreV1().Pods(targetPod.Namespace).Get(ctx, targetPod.Name, metav1.GetOptions{})
					return apierrors.IsNotFound(getErr)
				}, time.Minute, 5*time.Second).Should(BeTrue(), "old npu-feature-discovery pod should be deleted")

				Eventually(func(g Gomega) {
					currentPods, getErr := GetSpecifiedPods(clientset, ctx, "npu-feature-discovery")
					g.Expect(getErr).NotTo(HaveOccurred())

					var replacementPod *corev1.Pod
					for _, pod := range currentPods {
						if pod.Spec.NodeName == targetPod.Spec.NodeName &&
							pod.Name != targetPod.Name &&
							pod.Status.Phase == corev1.PodRunning {
							podCopy := pod
							replacementPod = &podCopy
							break
						}
					}

					g.Expect(replacementPod).NotTo(BeNil(), "should find replacement npu-feature-discovery pod on the same node")
				}, 2*time.Minute, 5*time.Second).Should(Succeed())

				Eventually(func(g Gomega) {
					updatedNode, getErr := clientset.CoreV1().Nodes().Get(ctx, targetPod.Spec.NodeName, metav1.GetOptions{})
					g.Expect(getErr).NotTo(HaveOccurred())
					g.Expect(SnapshotManagedNPUFeatureDiscoveryLabels(*updatedNode)).To(Equal(expectedLabels))
				}, 2*time.Minute, 10*time.Second).Should(Succeed())
			})
		})
	})
}
