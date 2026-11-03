package utils

// SkipGate returns a non-empty reason if the test should Skip.
func SkipGate(llmisvc, npuWorkload, inference bool) string {
	if llmisvc && SkipLLMISVCE2E() {
		return "SKIP_LLMISVC_E2E=true"
	}
	if npuWorkload && SkipNPUWorkloadE2E() {
		return "SKIP_NPU_WORKLOAD_E2E=true"
	}
	if inference && SkipInferenceE2E() {
		return "SKIP_INFERENCE_E2E=true"
	}
	return ""
}
