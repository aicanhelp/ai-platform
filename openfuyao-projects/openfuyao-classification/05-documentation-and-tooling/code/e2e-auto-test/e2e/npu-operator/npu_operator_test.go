package npu_operator

import (
	. "github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("NPU Operator test cases", Ordered, ContinueOnFailure, Label("npu_operator", "npu"), func() {
	GinkgoWriter.Println("=== Verifying Installation Phase ===")
	InstallationPhase()
	GinkgoWriter.Println("=== Verifying Uninstallation Phase ===")
	UninstallationPhase()
	GinkgoWriter.Println("=== Verifying Installed Phase ===")
	InstalledPhase()
	GinkgoWriter.Println("=== Verifying NPU Feature Discovery Phase ===")
	FeatureDiscoveryPhase()
	GinkgoWriter.Println("=== Verifying vNPU and DRA Management Phase ===")
	VNPUDRAManagementPhase()
})
