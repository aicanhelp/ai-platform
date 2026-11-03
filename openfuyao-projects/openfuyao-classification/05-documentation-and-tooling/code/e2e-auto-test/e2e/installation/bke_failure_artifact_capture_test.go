package installation

import (
	. "github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
)

var _ = JustAfterEach(func(ctx SpecContext) {
	report := CurrentSpecReport()
	if !report.State.Is(types.SpecStateFailureStates) {
		return
	}

	outputDir, err := utils.BKEFailureArtifactOutputDir(report.FileName(), report.LineNumber())
	if err != nil {
		GinkgoWriter.Printf("[bke-failure-artifact] create output dir failed: %v\n", err)
		return
	}
	utils.CaptureBKEFailureArtifacts(ctx, outputDir, GinkgoWriter.Printf)

	if hasPreInitLabel(report.Labels()) {
		utils.CaptureAbnormalPodDescribeArtifacts(ctx, outputDir, GinkgoWriter.Printf)
	}
})

func hasPreInitLabel(labels []string) bool {
	for _, label := range labels {
		if label == "pre-init" {
			return true
		}
	}
	return false
}
