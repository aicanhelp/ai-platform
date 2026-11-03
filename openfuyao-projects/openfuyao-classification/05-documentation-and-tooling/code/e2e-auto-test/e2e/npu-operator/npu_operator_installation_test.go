package npu_operator

import (
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func InstallationPhase() {
	var _ = Describe("Install NPU Operator", Ordered, Label("npu_operator_installation", "npu"), func() {
		var needUninstallation bool

		BeforeEach(func() {
			needUninstallation = false
		})

		AfterEach(func() {
			if needUninstallation {
				GinkgoWriter.Println()

				// 执行 npu-operator 卸载
				VerifyNPUOperatorUninstallation(clientset, ctx)
			}
		})

		Context("查找所有NPU Operator相关的Pod", Label("npu_operator_pods"), func() {
			It("162-npu-operator 无驱动部署-ubuntu22.04 arm 910B 单节点", func() {
				verifyEnvironment := VerifyEnvironment{
					NpuType:    "Ascend910B",
					EnvName:    "ubuntu", // 完全小写
					VersionID:  "22.04",
					SystemType: "arm",
				}

				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:          clientset,
					Ctx:                ctx,
					NodesNPUSituation:  nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{&verifyEnvironment},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationSingleNode(&verifyInstallationInfo)
			})

			It("165-npu-operator 无驱动部署-openEuler22.03 arm 310P 单节点", func() {
				verifyEnvironment := VerifyEnvironment{
					NpuType:    "Ascend310P",
					EnvName:    "openeuler", // 完全小写
					VersionID:  "22.03",
					SystemType: "arm",
				}

				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:          clientset,
					Ctx:                ctx,
					NodesNPUSituation:  nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{&verifyEnvironment},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationSingleNode(&verifyInstallationInfo)
			})

			It("166-npu-operator 无驱动部署-openEuler22.03 X86 310P 单节点", func() {
				verifyEnvironment := VerifyEnvironment{
					NpuType:    "Ascend310P",
					EnvName:    "openeuler", // 完全小写
					VersionID:  "22.03",
					SystemType: "amd",
				}

				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:          clientset,
					Ctx:                ctx,
					NodesNPUSituation:  nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{&verifyEnvironment},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationSingleNode(&verifyInstallationInfo)
			})

			It("167-npu-operator 无驱动部署-openEuler22.03 arm 910B 单节点", func() {
				verifyEnvironment := VerifyEnvironment{
					NpuType:    "Ascend910B",
					EnvName:    "openeuler", // 完全小写
					VersionID:  "22.03",
					SystemType: "arm",
				}

				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:          clientset,
					Ctx:                ctx,
					NodesNPUSituation:  nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{&verifyEnvironment},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationSingleNode(&verifyInstallationInfo)
			})

			It("416-npu-operator 有驱动部署-ubuntu22.04 arm 910B 单节点", func() {
				verifyEnvironment := VerifyEnvironment{
					NpuType:    "Ascend910B",
					EnvName:    "ubuntu", // 完全小写
					VersionID:  "22.04",
					SystemType: "arm",
				}

				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:          clientset,
					Ctx:                ctx,
					NodesNPUSituation:  nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{&verifyEnvironment},
				}

				// 验证有驱动部署
				needUninstallation = VerifyComponentsAfterInstallationSingleNode(&verifyInstallationInfo)
			})

			It("418-npu-operator 有驱动部署-openEuler22.03 arm 910B 单节点", func() {
				verifyEnvironment := VerifyEnvironment{
					NpuType:    "Ascend910B",
					EnvName:    "openeuler", // 完全小写
					VersionID:  "22.03",
					SystemType: "arm",
				}

				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:          clientset,
					Ctx:                ctx,
					NodesNPUSituation:  nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{&verifyEnvironment},
				}

				// 验证有驱动部署
				needUninstallation = VerifyComponentsAfterInstallationSingleNode(&verifyInstallationInfo)
			})

			It("419-npu-operator 有驱动部署-openEuler22.03 arm 310P 单节点", func() {
				verifyEnvironment := VerifyEnvironment{
					NpuType:    "Ascend310P",
					EnvName:    "openeuler", // 完全小写
					VersionID:  "22.03",
					SystemType: "arm",
				}

				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:          clientset,
					Ctx:                ctx,
					NodesNPUSituation:  nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{&verifyEnvironment},
				}

				// 验证有驱动部署
				needUninstallation = VerifyComponentsAfterInstallationSingleNode(&verifyInstallationInfo)
			})

			It("420-npu-operator 有驱动部署-openEuler22.03 X86 310P 单节点", func() {
				verifyEnvironment := VerifyEnvironment{
					NpuType:    "Ascend310P",
					EnvName:    "openeuler", // 完全小写
					VersionID:  "22.03",
					SystemType: "amd",
				}

				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:          clientset,
					Ctx:                ctx,
					NodesNPUSituation:  nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{&verifyEnvironment},
				}

				// 验证有驱动部署
				needUninstallation = VerifyComponentsAfterInstallationSingleNode(&verifyInstallationInfo)
			})

			It("156-多节点无驱动部署-openEuler22.03-arm 910B + X86 310P", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend910B",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "arm64",
						},
						{
							NpuType:    "Ascend310P",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "amd64",
						},
					},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("159-多节点无驱动部署-openEuler22.03 310P-arm + X86", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend310P",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "arm64",
						},
						{
							NpuType:    "Ascend310P",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "amd64",
						},
					},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("160-多节点无驱动部署-ubuntu22.04 arm 910B", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend910B",
							EnvName:    "ubuntu", // 完全小写
							VersionID:  "22.04",
							SystemType: "arm64",
						},
					},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("161-多节点无驱动部署-openEuler22.03 arm-910B + 310P", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend910B",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "arm64",
						},
						{
							NpuType:    "Ascend310P",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "arm64",
						},
					},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("163-多节点无驱动部署-openEuler22.03 arm 910B", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend910B",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "arm64",
						},
					},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("164-多节点无驱动部署-openEuler22.03 arm 310P", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend310P",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "arm64",
						},
					},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			// 当前测试环境未包含在测试用例中
			It("test-多节点无驱动部署-openEuler22.03 X86 310P", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend310P",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "amd64",
						},
					},
				}

				// 验证无驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("414-多节点有驱动部署-openEuler22.03 arm 310P", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend310P",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "arm64",
						},
					},
				}

				// 验证有驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("415-多节点有驱动部署-ubuntu22.04 arm 910B", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend910B",
							EnvName:    "ubuntu", // 完全小写
							VersionID:  "22.04",
							SystemType: "arm64",
						},
					},
				}

				// 验证有驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("417-多节点有驱动部署-openEuler22.03 arm 910B", func() {
				verifyInstallationInfo := VerifyInstallationInfo{
					Clientset:         clientset,
					Ctx:               ctx,
					NodesNPUSituation: nodesNPUSituation,
					VerifyEnvironments: []*VerifyEnvironment{
						{
							NpuType:    "Ascend910B",
							EnvName:    "openeuler", // 完全小写
							VersionID:  "22.03",
							SystemType: "arm64",
						},
					},
				}

				// 验证有驱动部署
				needUninstallation = VerifyComponentsAfterInstallationMultipleNodes(&verifyInstallationInfo)
			})

			It("643-查询对应NPUClusterPolicy状态", func() {
				GinkgoWriter.Println("Verifying NPU Cluster Policy status ···")

				// 执行 npu-operator 安装流程
				needUninstallation = true
				var err error

				// 执行 helm install url 指令
				helmReleaseName, helmNamespace, err := ExecuteHelmInstallFromURL()
				Expect(err).NotTo(HaveOccurred(), "should successfully execute helm install")

				// 验证 Helm Release 状态
				VerifyHelmReleaseStatus(helmReleaseName, helmNamespace)

				// 重复验证直到所有 Pod 状态都为 running 或超时
				Eventually(func() bool {
					var allRunning bool

					// 验证所有 Pod 状态是否均为 running
					allRunning, err = JudgePodsAllRunning(clientset, ctx)
					if err != nil {
						GinkgoWriter.Printf("Error verifying pods: %v\n", err)
						return false
					}

					// 验证对应 NPUClusterPolicy 状态是否正常
					VerifyNPUClusterPolicyStatus(dynamicClient, ctx, allRunning)

					return allRunning
				}, 10*time.Minute, 10*time.Second).Should(BeTrue(),
					"All NPU Operator pods should be running in 10 minutes")

				GinkgoWriter.Println("NPUClusterPolicy 状态验证成功")
			})

			It("151-各组件验证", func() {
				// 获取所有的节点
				nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
				Expect(err).NotTo(HaveOccurred(), "'kubectl get nodes' 执行失败")
				Expect(nodes.Items).NotTo(BeEmpty(), "在集群中没有找到任何节点")

				// 验证 npu-operator 安装流程
				needUninstallation = EnsureNPUOperatorInstallation(clientset, ctx, nodesNPUSituation)

				// TODO: 重启环境

				// 验证 npu-operator 各组件安装后状态
				VerifyNPUOperatorComponents(clientset, ctx, nodesNPUSituation, nodes)
			})
		})

		It("153-npu-operator 插件安装", func() {
			// 验证 npu-operator 安装流程
			needUninstallation = EnsureNPUOperatorInstallation(clientset, ctx, nodesNPUSituation)
		})
	})
}
