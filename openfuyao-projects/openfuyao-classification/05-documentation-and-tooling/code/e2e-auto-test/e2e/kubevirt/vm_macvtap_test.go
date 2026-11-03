package kubevirt

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

const macvtapVMName = "vm-macvlan"

func buildMacvtapVM(namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "kubevirt.io/v1",
			"kind":       "VirtualMachine",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"running": false,
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"kubevirt.io/domain": "openeuler",
						},
					},
					"spec": map[string]interface{}{
						"domain": map[string]interface{}{
							"devices": map[string]interface{}{
								"disks": []interface{}{
									map[string]interface{}{
										"name": "rootfs",
										"disk": map[string]interface{}{"bus": "virtio"},
									},
									map[string]interface{}{
										"name": "cloudinit",
										"disk": map[string]interface{}{"bus": "virtio"},
									},
								},
								"interfaces": []interface{}{
									map[string]interface{}{
										"name":      "default",
										"masquerade": map[string]interface{}{},
									},
									map[string]interface{}{
										"name": "macvtap-net",
										"binding": map[string]interface{}{
											"name": "macvtap",
										},
									},
								},
							},
							"resources": map[string]interface{}{
								"requests": map[string]interface{}{"memory": "1Gi"},
								"limits":   map[string]interface{}{"memory": "1Gi"},
							},
						},
						"networks": []interface{}{
							map[string]interface{}{
								"name": "default",
								"pod":  map[string]interface{}{},
							},
							map[string]interface{}{
								"name": "macvtap-net",
								"multus": map[string]interface{}{
									"networkName": "macvtapnetwork",
								},
							},
						},
						"volumes": []interface{}{
							map[string]interface{}{
								"name": "rootfs",
								"containerDisk": map[string]interface{}{
									"image": "openeuler2203-sp3-test:v5",
									"path":  "/disk/disk.qcow2",
								},
							},
							map[string]interface{}{
								"name": "cloudinit",
								"cloudInitNoCloud": map[string]interface{}{
									"userData": "password: openEuler12#$\nchpasswd: { expire: False }\nssh_pwauth: True\npackages:\n  - nginx\nruncmd:\n  - systemctl enable nginx\n  - systemctl start nginx",
								},
							},
						},
					},
				},
			},
		},
	}
}

var _ = Describe("Kubevirt macvtap 虚机测试",
	Label("kubevirt", "with-workload-cluster", "26.06"), Label("P0"), func() {

		It("鲲鹏-kubevirt-IT-004：部署带有 macvtap 设备的虚机", func() {
			By("创建 VirtualMachine vm-macvlan（running=false）")
			vmObj := buildMacvtapVM(	vmNamespace, macvtapVMName)
			_, err := k8sClient.DynamicClient.Resource(vmGVR).Namespace(	vmNamespace).Create(
				ctx, vmObj, metav1.CreateOptions{},
			)
			Expect(err).NotTo(HaveOccurred(), "创建 VM 应当成功")

			DeferCleanup(func() {
				By("清理：删除 VM vm-macvlan")
				_ = k8sClient.DynamicClient.Resource(vmGVR).Namespace(	vmNamespace).Delete(
					ctx, macvtapVMName, metav1.DeleteOptions{},
				)
			})

			By("设置 spec.running=true 启动 VM")
			_, err = k8sClient.DynamicClient.Resource(vmGVR).Namespace(	vmNamespace).Patch(
				ctx, macvtapVMName, "application/merge-patch+json",
				[]byte(`{"spec":{"running":true}}`), metav1.PatchOptions{},
			)
			Expect(err).NotTo(HaveOccurred(), "启动 VM 的 patch 应当成功")

			By("等待 VirtualMachineInstance 达到 Running 阶段")
			Eventually(func() bool {
				vmi, err := k8sClient.DynamicClient.Resource(vmiGVR).Namespace(	vmNamespace).Get(
					ctx, macvtapVMName, metav1.GetOptions{},
				)
				if err != nil {
					return false
				}
				phase, found, _ := unstructured.NestedString(vmi.Object, "status", "phase")
				return found && phase == "Running"
			}, 10*time.Minute, 15*time.Second).Should(BeTrue(),
				"VMI 应当在 10 分钟内达到 Running 状态")
		})
	})
