package eagle_eye

import (
	. "github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("Eagle-Eye test cases", Ordered, ContinueOnFailure, Label("eagle_eye"), func() {
	GinkgoWriter.Println("=== Verifying Metrics Phase ===")
	MonitorPhase()
	GinkgoWriter.Println("=== Verifying Diagnosis Phase ===")
	DiagnosisPhase()
	GinkgoWriter.Println("=== Verifying Network Performance Exporter Phase ===")
	NetworkPerformanceExporterPhase()
})
