package kubevirt

import (
	"encoding/json"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

const (
	hotplugVMName = "vm-hotplug"
)

var vmGVR = schema.GroupVersionResource{
	Group:    "kubevirt.io",
	Version:  "v1",
	Resource: "virtualmachines",
}

var vmiGVR = schema.GroupVersionResource{
	Group:    "kubevirt.io",
	Version:  "v1",
	Resource: "virtualmachineinstances",
}

func buildHotplugVM(namespace, name string) *unstructured.Unstructured {
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

func buildStartPatch() []byte {
	patch, _ := json.Marshal(map[string]interface{}{
		"spec": map[string]interface{}{
			"running": true,
		},
	})
	return patch
}

func buildInterfacesNetworksPatch(c *unstructured.Unstructured, ns string,
	newInterfaces []interface{}, newNetworks []interface{}) (string, []byte, error) {

	current, err := k8sClient.DynamicClient.Resource(vmGVR).Namespace(ns).Get(ctx, c.GetName(), metav1.GetOptions{})
	if err != nil {
		return "", nil, fmt.Errorf("获取 VM failed: %w", err)
	}

	template, ok := current.Object["spec"].(map[string]interface{})["template"].(map[string]interface{})
	if !ok {
		return "", nil, fmt.Errorf("无法解析 VM spec.template")
	}

	newTemplate := make(map[string]interface{})
	for k, v := range template {
		newTemplate[k] = v
	}

	spec, ok := newTemplate["spec"].(map[string]interface{})
	if ok {
		newSpec := make(map[string]interface{})
		for k, v := range spec {
			newSpec[k] = v
		}
		domain, ok := newSpec["domain"].(map[string]interface{})
		if ok {
			newDomain := make(map[string]interface{})
			for k, v := range domain {
				newDomain[k] = v
			}
			devices, ok := newDomain["devices"].(map[string]interface{})
			if ok {
				newDevices := make(map[string]interface{})
				for k, v := range devices {
					newDevices[k] = v
				}
				newDevices["interfaces"] = newInterfaces
				newDomain["devices"] = newDevices
			}
			newSpec["domain"] = newDomain
		}
		newSpec["networks"] = newNetworks
		newTemplate["spec"] = newSpec
	}

	patchObj := map[string]interface{}{
		"spec": map[string]interface{}{
			"template": newTemplate,
		},
	}

	patchData, err := json.Marshal(patchObj)
	if err != nil {
		return "", nil, fmt.Errorf("序列化 patch 失败: %w", err)
	}

	return current.GetName(), patchData, nil
}

var _ = Describe("Kubevirt VirtualMachine 热插拔测试",
	Ordered, Label("kubevirt", "with-workload-cluster", "26.06"), Label("P0"), func() {

		AfterAll(func() {
			By("清理：删除 VM vm-hotplug")
			_ = k8sClient.DynamicClient.Resource(vmGVR).Namespace(	vmNamespace).Delete(
				ctx, hotplugVMName, metav1.DeleteOptions{},
			)
		})

		It("鲲鹏-kubevirt-IT-002-Step1：创建并启动 VM", func() {
			By("创建 VirtualMachine（running=false）")
			vmObj := buildHotplugVM(	vmNamespace, hotplugVMName)
			_, err := k8sClient.DynamicClient.Resource(vmGVR).Namespace(	vmNamespace).Create(
				ctx, vmObj, metav1.CreateOptions{},
			)
			Expect(err).NotTo(HaveOccurred(), "创建 VM 应当成功")

			By("设置 spec.running=true 启动 VM")
			_, err = k8sClient.DynamicClient.Resource(vmGVR).Namespace(	vmNamespace).Patch(
				ctx, hotplugVMName, types.MergePatchType, buildStartPatch(), metav1.PatchOptions{},
			)
			Expect(err).NotTo(HaveOccurred(), "启动 VM 的 patch 应当成功")

			By("等待 VirtualMachineInstance 达到 Running 阶段")
			Eventually(func() bool {
				vmi, err := k8sClient.DynamicClient.Resource(vmiGVR).Namespace(	vmNamespace).Get(
					ctx, hotplugVMName, metav1.GetOptions{},
				)
				if err != nil {
					return false
				}
				phase, found, _ := unstructured.NestedString(vmi.Object, "status", "phase")
				return found && phase == "Running"
			}, 10*time.Minute, 15*time.Second).Should(BeTrue(),
				"VMI 应当在 10 分钟内达到 Running 状态")
		})

		It("鲲鹏-kubevirt-IT-002-Step2：热插拔新增接口 dyniface1", func() {
			By("构建含 dyniface1 的 interfaces 和 networks")
			newInterfaces := []interface{}{
				map[string]interface{}{
					"name":      "default",
					"masquerade": map[string]interface{}{},
				},
				map[string]interface{}{
					"name":   "dyniface1",
					"bridge": map[string]interface{}{},
				},
			}
			newNetworks := []interface{}{
				map[string]interface{}{
					"name": "default",
					"pod":  map[string]interface{}{},
				},
				map[string]interface{}{
					"name":  "dyniface1",
					"multus": map[string]interface{}{"networkName": "new-fancy-net"},
				},
			}

			By("构建并应用 merge-patch（含完整 template spec）")
			tempVM := &unstructured.Unstructured{Object: map[string]interface{}{
				"metadata": map[string]interface{}{"name": hotplugVMName},
			}}
			_, patchData, err := buildInterfacesNetworksPatch(tempVM, 	vmNamespace, newInterfaces, newNetworks)
			Expect(err).NotTo(HaveOccurred(), "构建 hotplug patch 应当成功")

			_, err = k8sClient.DynamicClient.Resource(vmGVR).Namespace(	vmNamespace).Patch(
				ctx, hotplugVMName, types.MergePatchType, patchData, metav1.PatchOptions{},
			)
			Expect(err).NotTo(HaveOccurred(),
				"热插拔 dyniface1 接口的 patch 应当成功返回")
		})

		It("鲲鹏-kubevirt-IT-002-Step3：将 dyniface1 标记为 state=absent", func() {
			By("构建 dyniface1/state=absent 的 interfaces 和 networks")
			absentInterfaces := []interface{}{
				map[string]interface{}{
					"name":      "default",
					"masquerade": map[string]interface{}{},
				},
				map[string]interface{}{
					"name":   "dyniface1",
					"bridge": map[string]interface{}{},
					"state":  "absent",
				},
			}
			absentNetworks := []interface{}{
				map[string]interface{}{
					"name": "default",
					"pod":  map[string]interface{}{},
				},
				map[string]interface{}{
					"name":  "dyniface1",
					"multus": map[string]interface{}{"networkName": "new-fancy-net"},
				},
			}

			By("构建并应用 merge-patch（含完整 template spec）")
			tempVM := &unstructured.Unstructured{Object: map[string]interface{}{
				"metadata": map[string]interface{}{"name": hotplugVMName},
			}}
			_, patchData, err := buildInterfacesNetworksPatch(tempVM, 	vmNamespace, absentInterfaces, absentNetworks)
			Expect(err).NotTo(HaveOccurred(), "构建 absent patch 应当成功")

			_, err = k8sClient.DynamicClient.Resource(vmGVR).Namespace(	vmNamespace).Patch(
				ctx, hotplugVMName, types.MergePatchType, patchData, metav1.PatchOptions{},
			)
			Expect(err).NotTo(HaveOccurred(),
				"将 dyniface1 标记为 absent 的 patch 应当成功返回")
		})
	})
