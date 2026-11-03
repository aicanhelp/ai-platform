package npu_dra_plugin

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("NPU DRA Plugin 插件部署测试",
	Ordered, Label("npu-dra-plugin", "26.08"), Label("P0"), Label("npu"), func() {

		It("昇腾-npu-dra-IT-001：插件部署与 ResourceSlice 上报检查", func() {
			By("等待目标节点的 ResourceSlice 出现")
			rs := waitForResourceSlice(localNodeName)
			GinkgoWriter.Printf("ResourceSlice: %s, Driver: %s, Node: %s\n", rs.Name, rs.Spec.Driver, derefStr(rs.Spec.NodeName))

			By("验证 ResourceSlice 的 Driver 和 NodeName")
			Expect(rs.Spec.Driver).To(Equal(driverName))
			if rs.Spec.NodeName != nil {
				Expect(*rs.Spec.NodeName).To(Equal(localNodeName))
			}

			By("验证设备数量大于 0")
			deviceCount := len(rs.Spec.Devices)
			Expect(deviceCount).To(BeNumerically(">", 0))
			GinkgoWriter.Printf("设备数量: %d\n", deviceCount)

			By("验证每个设备包含 physicalId、chipName 属性")
			for _, device := range rs.Spec.Devices {
				_, hasPhysicalID := device.Attributes["physicalID"]
				_, hasChipName := device.Attributes["chipName"]
				Expect(hasPhysicalID).To(BeTrue(), "设备 %s 应当包含 physicalID", device.Name)
				Expect(hasChipName).To(BeTrue(), "设备 %s 应当包含 chipName", device.Name)
			}
			GinkgoWriter.Printf("IT-001 验证通过: %d 个设备上报正常\n", deviceCount)
		})
	})

func derefStr(s *string) string {
	if s == nil { return "<nil>" }
	return *s
}
