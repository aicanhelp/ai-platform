/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package hardware

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/google/uuid"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/log"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

const (
	// minIdleNPUs is the minimum number of idle NPUs required for slow card detection
	minIdleNPUs = 2
	// podReadyTimeout is the timeout for waiting for the msprof Pod to be ready
	podReadyTimeout = "300s"
	// unknownPhysicalDeviceID indicates that the physical device ID could not be determined
	unknownPhysicalDeviceID = -1
	// expectedSplitParts is the expected number of parts after splitting a key-value pair
	expectedSplitParts = 2
	// detectOutputMinLines is the minimum number of output lines from CANN image detection
	detectOutputMinLines = 2
	// slowRankRegexMatchLen is the expected minimum match length for slow rank regex
	slowRankRegexMatchLen = 3
)

var (
	// affectedRankRe matches the affected rank and wait ratio from msprof suggestion output
	affectedRankRe = regexp.MustCompile(`for rank (\d+)\s*\..*max wait ratio is:\s*([\d.]+)`)
	// rootCauseRankRe matches the root cause rank from msprof suggestion output
	rootCauseRankRe = regexp.MustCompile(`Rank (\d+) has a lot longer computation time`)
)

// rankMetrics represents performance metrics for a single rank
type rankMetrics struct {
	RankID                   int     `json:"rank_id"`
	PhysicalDeviceID         int     `json:"physical_device_id"`
	NodeName                 string  `json:"node_name"`
	WaitTimeMS               float64 `json:"wait_time_ms"`
	TransitTimeMS            float64 `json:"transit_time_ms"`
	SynchronizationTimeMS    float64 `json:"synchronization_time_ms"`
	WaitTimeRatio            float64 `json:"wait_time_ratio"`
	SynchronizationTimeRatio float64 `json:"synchronization_time_ratio"`
}

// checkH08 performs slow card detection using msprof profiling
func checkH08(ssh executor.CommandRunner) types.CheckResult {
	// 1. Get idle NPU device list
	idleDevices, err := getIdleNPUDevices(ssh)
	if err != nil {
		log.Error(fmt.Sprintf("H-08: Failed to query idle NPU devices: %v", err))
		return types.CheckResult{
			ID:         "H-08",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("Cannot determine idle NPU devices: %v", err),
			Suggestion: "Please confirm npu-smi is working properly and NPU devices are accessible",
		}
	}

	if len(idleDevices) < minIdleNPUs {
		log.Info(fmt.Sprintf("H-08: Insufficient idle NPUs (need >= %d, got %d)", minIdleNPUs, len(idleDevices)))
		return types.CheckResult{
			ID:     "H-08",
			Status: types.StatusSkipped,
			Message: fmt.Sprintf("Slow card detection skipped (need >= %d idle NPUs, got %d)",
				minIdleNPUs, len(idleDevices)),
		}
	}

	log.Info(fmt.Sprintf("H-08: Found %d idle NPUs: %v, "+
		"proceeding with msprof check", len(idleDevices), idleDevices))

	// 2. Run msprof check using idle NPUs (collects all data before Pod cleanup)
	result, err := runMsprofCheck(ssh, idleDevices)
	if err != nil {
		log.Error(fmt.Sprintf("H-08: Msprof check failed: %v", err))
		return types.CheckResult{
			ID:         "H-08",
			Status:     types.StatusWarning,
			Message:    fmt.Sprintf("Slow card detection failed: %v", err),
			Suggestion: "Slow card detection is optional and does not block deployment",
		}
	}

	// 3. Parse result
	if strings.Contains(result.Suggestion, "no slow rank") {
		log.Info("H-08: No slow card detected")
		return types.CheckResult{
			ID:      "H-08",
			Status:  types.StatusPassed,
			Message: "All NPU cards have balanced communication performance",
			Detail: map[string]interface{}{
				"metrics": result.Metrics,
			},
		}
	}

	// 4. Slow card detected
	log.Info(fmt.Sprintf("H-08: Slow card detected: %s", result.Suggestion))
	return types.CheckResult{
		ID:      "H-08",
		Status:  types.StatusWarning,
		Message: fmt.Sprintf("Slow NPU card detected: %s", result.Suggestion),
		Detail: map[string]interface{}{
			"metrics": result.Metrics,
		},
		Suggestion: "Check NPU card health status (temperature, ECC errors via npu-smi info); " +
			"slow cards may degrade training/inference efficiency but do not block deployment",
	}
}

// getIdleNPUDevices returns the list of idle NPU device IDs
func getIdleNPUDevices(ssh executor.CommandRunner) ([]int, error) {
	stdout, _, err := ssh.Run("npu-smi info")
	if err != nil {
		return nil, fmt.Errorf("failed to execute npu-smi: %w", err)
	}

	return parseIdleNPUDevices(stdout), nil
}

// parseIdleNPUDevices parses npu-smi output and returns idle device IDs
func parseIdleNPUDevices(output string) []int {
	// Strip ANSI escape sequences
	clean := ansiEscape.ReplaceAllString(output, "")

	// 1. Get all healthy devices
	var allDevices []int
	deviceMatches := npuDeviceRe.FindAllStringSubmatch(clean, -1)
	for _, m := range deviceMatches {
		if id, err := strconv.Atoi(m[npuIDGroupIdx]); err == nil {
			// Only count OK devices
			if m[npuStatusGroupIdx] == "OK" {
				allDevices = append(allDevices, id)
			}
		}
	}

	// 2. Get occupied devices
	occupiedIDs := map[int]bool{}
	matches := npuProcessRe.FindAllStringSubmatch(clean, -1)
	for _, m := range matches {
		if id, err := strconv.Atoi(m[1]); err == nil {
			occupiedIDs[id] = true
		}
	}

	// 3. Calculate idle devices = all healthy devices - occupied devices
	var idleDevices []int
	for _, id := range allDevices {
		if !occupiedIDs[id] {
			idleDevices = append(idleDevices, id)
		}
	}

	sort.Ints(idleDevices)
	return idleDevices
}

// detectNPUResourceName extracts the full NPU resource name from node capacity
func detectNPUResourceName(ssh executor.CommandRunner) (string, error) {
	nodeName := ssh.NodeName()
	cmd := fmt.Sprintf("kubectl get node %s -o jsonpath='{.status.capacity}'", nodeName)
	stdout, _, err := ssh.Run(cmd)
	if err != nil {
		return "", fmt.Errorf("failed to get node capacity: %w", err)
	}

	// Parse JSON capacity map
	var capacity map[string]string
	if err := json.Unmarshal([]byte(strings.Trim(stdout, "'")), &capacity); err != nil {
		return "", fmt.Errorf("failed to parse node capacity: %w", err)
	}

	// Find huawei.com/Ascend* resource key
	for key := range capacity {
		if strings.HasPrefix(key, "huawei.com/Ascend") {
			return key, nil
		}
	}

	return "", fmt.Errorf("no huawei.com/Ascend resource found in node capacity")
}

// msprofCheckResult holds the complete results from msprof profiling
type msprofCheckResult struct {
	Suggestion string
	Metrics    []rankMetrics
}

// runMsprofCheck performs the complete msprof profiling and analysis workflow in a Pod
func runMsprofCheck(ssh executor.CommandRunner, idleDevices []int) (*msprofCheckResult, error) {
	nodeName := ssh.NodeName()
	namespace := "default"
	npuCount := len(idleDevices)
	podName := fmt.Sprintf("msprof-slowcard-%s", uuid.New().String())

	// 1. Detect chip type and select image
	image, chipName, err := detectCANNImage(ssh)
	if err != nil {
		return nil, fmt.Errorf("failed to detect CANN image: %w", err)
	}
	log.Info(fmt.Sprintf("Detected chip: %s, using image: %s", chipName, image))

	// 2. Detect NPU resource name from node capacity
	npuResourceName, err := detectNPUResourceName(ssh)
	if err != nil {
		return nil, fmt.Errorf("failed to detect NPU resource name: %w", err)
	}
	log.Info(fmt.Sprintf("Detected NPU resource: %s", npuResourceName))

	// 3. Create and start msprof Pod
	podName, cleanup, err := createAndStartMsprofPod(ssh, nodeName, npuCount, image, npuResourceName)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	// 3. Prepare scripts in Pod
	if err := prepareScriptsInPod(ssh, podName); err != nil {
		return nil, err
	}

	// 4. Execute msprof workflow with idle devices
	output, err := executeMsprofWorkflow(ssh, podName, idleDevices)
	if err != nil {
		return nil, err
	}

	// 5. Parse result
	suggestion, err := parseSlowCardResult(output)
	if err != nil {
		return nil, fmt.Errorf("failed to parse results: %w", err)
	}

	// 6. Get physical device mapping (before Pod cleanup)
	deviceMapping, err := getPhysicalDeviceMapping(ssh, podName, namespace, npuCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get device mapping: %w", err)
	}

	// 7. Collect performance metrics (before Pod cleanup)
	metrics, err := collectRankMetrics(ssh, podName, namespace, nodeName, deviceMapping)
	if err != nil {
		return nil, fmt.Errorf("failed to collect metrics: %w", err)
	}

	return &msprofCheckResult{
		Suggestion: suggestion,
		Metrics:    metrics,
	}, nil
}

// msprofPodConfig holds configuration for msprof Pod generation
type msprofPodConfig struct {
	NodeName        string
	PodName         string
	Namespace       string
	NPUCount        int
	Image           string
	NPUResourceName string
}

// createAndStartMsprofPod creates a msprof Pod and waits for it to be ready
func createAndStartMsprofPod(ssh executor.CommandRunner,
	nodeName string, npuCount int, image, npuResourceName string) (string, func(), error) {
	podName := fmt.Sprintf("msprof-slowcard-%s", uuid.New().String())
	namespace := "default"

	log.Info("Creating msprof Pod")

	// Generate Pod YAML
	config := msprofPodConfig{
		NodeName:        nodeName,
		PodName:         podName,
		Namespace:       namespace,
		NPUCount:        npuCount,
		Image:           image,
		NPUResourceName: npuResourceName,
	}
	podYAML := generateMsprofPodYAML(config)

	// Write Pod YAML to host
	cmd := fmt.Sprintf("cat > /tmp/msprof-pod.yaml << 'EOF'\n%s\nEOF", podYAML)
	if _, _, err := ssh.Run(cmd); err != nil {
		return "", nil, fmt.Errorf("failed to write pod yaml: %w", err)
	}

	// Apply Pod
	if _, _, err := ssh.Run("kubectl apply -f /tmp/msprof-pod.yaml"); err != nil {
		return "", nil, fmt.Errorf("failed to create pod: %w", err)
	}

	// Cleanup function
	cleanup := func() {
		log.Info("Cleaning up msprof Pod")
		if _, _, err := ssh.Run(fmt.Sprintf(
			"kubectl delete pod %s -n %s --ignore-not-found=true",
			podName, namespace,
		)); err != nil {
			log.Info(fmt.Sprintf("Failed to delete pod during cleanup: %v", err))
		}
		if _, _, err := ssh.Run("rm -f /tmp/msprof-pod.yaml"); err != nil {
			log.Info(fmt.Sprintf("Failed to remove yaml file during cleanup: %v", err))
		}
	}

	// Wait for Pod Ready (timeout 5 minutes)
	log.Info("Waiting for Pod to be ready")
	waitCmd := fmt.Sprintf(
		"kubectl wait --for=condition=Ready pod/%s -n %s --timeout=%s",
		podName, namespace, podReadyTimeout,
	)
	if _, _, err := ssh.Run(waitCmd); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("pod not ready: %w", err)
	}

	return podName, cleanup, nil
}

// prepareScriptsInPod copies benchmark.py and fix_rank.py to the Pod
func prepareScriptsInPod(ssh executor.CommandRunner, podName string) error {
	namespace := "default"

	log.Info("Copying scripts to Pod")

	// Copy benchmark.py using base64 to avoid shell escaping issues
	benchmarkEncoded := base64.StdEncoding.EncodeToString([]byte(benchmarkPyContent))
	copyCmd := fmt.Sprintf(
		"kubectl exec %s -n %s -- bash -c 'echo %s | base64 -d > /tmp/benchmark.py'",
		podName, namespace, benchmarkEncoded,
	)
	if _, _, err := ssh.Run(copyCmd); err != nil {
		return fmt.Errorf("failed to copy benchmark.py: %w", err)
	}

	// Copy fix_rank.py using base64 to avoid shell escaping issues
	fixRankEncoded := base64.StdEncoding.EncodeToString([]byte(fixRankPyContent))
	copyCmd = fmt.Sprintf(
		"kubectl exec %s -n %s -- bash -c 'echo %s | base64 -d > /tmp/fix_rank.py'",
		podName, namespace, fixRankEncoded,
	)
	if _, _, err := ssh.Run(copyCmd); err != nil {
		return fmt.Errorf("failed to copy fix_rank.py: %w", err)
	}

	return nil
}

// executeMsprofWorkflow runs the complete msprof profiling workflow in the Pod
func executeMsprofWorkflow(ssh executor.CommandRunner, podName string, idleDevices []int) (string, error) {
	namespace := "default"

	log.Info("Running msprof check in Pod")

	// Convert idle device IDs to string format
	var deviceStrs []string
	for _, id := range idleDevices {
		deviceStrs = append(deviceStrs, strconv.Itoa(id))
	}
	physicalDevices := strings.Join(deviceStrs, ",")
	log.Info(fmt.Sprintf("Using idle physical devices: %s", physicalDevices))

	// Build exec script with npuCount and set PHYSICAL_DEVICES env var
	npuCount := len(idleDevices)
	execScript := fmt.Sprintf(msprofExecScriptTemplate, npuCount)
	execScriptWithEnv := fmt.Sprintf(`
export ASCEND_RT_VISIBLE_DEVICES="%s"
%s
`, physicalDevices, execScript)

	// Execute in Pod
	execCmd := fmt.Sprintf(
		"kubectl exec %s -n %s -- bash -c %s",
		podName, namespace,
		fmt.Sprintf("'%s'", strings.ReplaceAll(execScriptWithEnv, "'", "'\\'")),
	)

	stdout, stderr, err := ssh.Run(execCmd)
	if err != nil {
		// Log full output for debugging
		log.Error(fmt.Sprintf("msprof workflow failed - stdout: %s", stdout))
		log.Error(fmt.Sprintf("msprof workflow failed - stderr: %s", stderr))

		// Return simplified error message
		errMsg := fmt.Sprintf("msprof check failed: %v", err)
		if strings.TrimSpace(stderr) != "" {
			errMsg = fmt.Sprintf("%s, stderr: %s", errMsg, stderr)
		}
		return "", fmt.Errorf("%s", errMsg)
	}

	return stdout, nil
}

// getPhysicalDevicesFromAnnotation extracts physical device IDs from Pod annotation
func getPhysicalDevicesFromAnnotation(ssh executor.CommandRunner, podName, namespace string) (string, error) {
	cmd := fmt.Sprintf(
		`kubectl get pod %s -n %s -o jsonpath='{.metadata.annotations.huawei\.com/Ascend910}'`,
		podName, namespace,
	)
	output, _, err := ssh.Run(cmd)
	if err != nil {
		return "", fmt.Errorf("failed to get pod annotation: %w", err)
	}

	// Parse "Ascend910-2,Ascend910-4,Ascend910-5" -> "2,4,5"
	var devices []string
	for _, devStr := range strings.Split(strings.TrimSpace(output), ",") {
		parts := strings.Split(devStr, "-")
		if len(parts) == expectedSplitParts {
			devices = append(devices, parts[1])
		}
	}

	if len(devices) == 0 {
		return "", fmt.Errorf("no devices found in annotation")
	}

	return strings.Join(devices, ","), nil
}

// detectCANNImage detects NPU chip type and returns corresponding CANN image
func detectCANNImage(ssh executor.CommandRunner) (image, chipName string, err error) {
	// Execute the detection script from msprof documentation
	detectCmd := `source /dev/stdin <<< "$(` +
		`dev_id=$(lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | head -n1 | cut -d: -f2); ` +
		`case "$dev_id" in ` +
		`'d500' ) echo "export MY_STUDY_VAR_CANN_IMAGE=` +
		`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-openeuler24.03-py3.11-devel; ` +
		`export MY_CHIP_NAME=310P";; ` +
		`'d802' ) echo "export MY_STUDY_VAR_CANN_IMAGE=` +
		`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11-devel; ` +
		`export MY_CHIP_NAME=910B";; ` +
		`'d803' ) echo "export MY_STUDY_VAR_CANN_IMAGE=` +
		`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11-devel; ` +
		`export MY_CHIP_NAME=A3";; ` +
		`'d806' ) echo "export MY_STUDY_VAR_CANN_IMAGE=` +
		`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-950-openeuler24.03-py3.11-devel; ` +
		`export MY_CHIP_NAME=950";; ` +
		`* ) echo "unset MY_STUDY_VAR_CANN_IMAGE MY_CHIP_NAME; echo >&2; ` +
		`echo -e '\''\\033[31m[FAIL] Get device ID: $dev_id. ` +
		`Learning is not supported in the current environment.\\033[0m'\'' >&2";; ` +
		`esac)" && ` +
		`[ -n "$MY_STUDY_VAR_CANN_IMAGE" ] && ` +
		`echo "$MY_STUDY_VAR_CANN_IMAGE" && echo "$MY_CHIP_NAME"`

	stdout, _, err := ssh.Run(detectCmd)
	if err != nil {
		return "", "", fmt.Errorf("failed to detect CANN image: %w", err)
	}

	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	if len(lines) < detectOutputMinLines {
		return "", "", fmt.Errorf("invalid detection output: %s", stdout)
	}

	image = strings.TrimSpace(lines[0])
	chipName = strings.TrimSpace(lines[1])

	if image == "" || chipName == "" {
		return "", "", fmt.Errorf("failed to parse image or chip name from output")
	}

	return image, chipName, nil
}

// generateMsprofPodYAML generates the Pod YAML for msprof profiling
func generateMsprofPodYAML(config msprofPodConfig) string {
	return fmt.Sprintf(
		msprofPodYAMLTemplate,
		config.PodName, config.Namespace, config.NodeName, config.Image,
		config.NPUResourceName, config.NPUCount, config.NPUResourceName, config.NPUCount,
	)
}

// parseSlowCardResult extracts the suggestion from msprof output
func parseSlowCardResult(output string) (string, error) {
	lines := strings.Split(output, "\n")
	for i, line := range lines {
		if !strings.Contains(line, "Suggestion:") {
			continue
		}

		fullSuggestion := line
		for j := i + 1; j < len(lines); j++ {
			trimmed := strings.TrimSpace(lines[j])
			if trimmed == "" || strings.HasPrefix(trimmed, "===") || strings.HasPrefix(trimmed, "{") {
				break
			}
			fullSuggestion += " " + trimmed
		}
		return extractSuggestion(fullSuggestion)
	}

	return "", fmt.Errorf("no suggestion found in msprof output")
}

// extractSuggestion parses the suggestion content from a line containing "Suggestion:"
func extractSuggestion(line string) (string, error) {
	parts := strings.SplitN(line, "Suggestion:", expectedSplitParts)
	if len(parts) != expectedSplitParts {
		return "", fmt.Errorf("no suggestion found in msprof output")
	}

	suggestion := strings.TrimSpace(parts[1])

	// Check if no slow rank
	if strings.Contains(suggestion, "no slow rank") {
		return suggestion, nil
	}

	affectedMatches := affectedRankRe.FindStringSubmatch(suggestion)

	// Extract root cause rank (the actual slow card)
	rootCauseMatches := rootCauseRankRe.FindStringSubmatch(suggestion)

	if len(affectedMatches) >= slowRankRegexMatchLen {
		affectedRank := affectedMatches[1]
		waitRatio := affectedMatches[2]

		if len(rootCauseMatches) >= expectedSplitParts {
			// Root cause is a specific rank (computation imbalance)
			rootRank := rootCauseMatches[1]
			return fmt.Sprintf("Rank %s is the root cause (longer computation time), "+
				"affecting Rank %s (max wait ratio: %s)", rootRank, affectedRank, waitRatio), nil
		}

		if strings.Contains(suggestion, "communication bandwidths between ranks are inconsistent") {
			return fmt.Sprintf("Communication bandwidth inconsistency detected, "+
				"affecting Rank %s (max wait ratio: %s)", affectedRank, waitRatio), nil
		}

		// Fallback: only affected rank info available
		return fmt.Sprintf("Slow rank detected: Rank %s (max wait ratio: %s)",
			affectedRank, waitRatio), nil
	}

	// Return raw suggestion
	return suggestion, nil
}

// getPhysicalDeviceMapping builds the mapping from rank to physical device ID
func getPhysicalDeviceMapping(
	ssh executor.CommandRunner, podName, namespace string, npuCount int) (map[int]int, error) {
	mapping := make(map[int]int)

	// Read each rank's device info file
	for rank := 0; rank < npuCount; rank++ {
		cmd := fmt.Sprintf(
			"kubectl exec %s -n %s -- cat /tmp/rank_%d_device.txt 2>/dev/null",
			podName, namespace, rank,
		)
		output, _, err := ssh.Run(cmd)
		if err != nil {
			log.Info(fmt.Sprintf("Failed to read rank %d device info: %v", rank, err))
			continue
		}

		physicalDevice, ok := parseRankDeviceInfo(output)
		if ok {
			mapping[rank] = physicalDevice
		} else {
			log.Info(fmt.Sprintf("Rank %d: failed to determine physical device from output", rank))
		}
	}

	return mapping, nil
}

// parseRankDeviceInfo parses a rank device file and returns the physical device ID
func parseRankDeviceInfo(output string) (int, bool) {
	var actualDevice int
	var physicalDevices []int

	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		if strings.HasPrefix(line, "actual_device=") {
			if _, err := fmt.Sscanf(line, "actual_device=%d", &actualDevice); err != nil {
				log.Info(fmt.Sprintf("Failed to parse actual_device from line '%s': %v", line, err))
			}
		} else if strings.HasPrefix(line, "physical_devices=") {
			devicesStr := strings.TrimPrefix(line, "physical_devices=")
			physicalDevices = parsePhysicalDevices(devicesStr)
		} else {
			// skip unrecognized lines (e.g. "rank=X")
		}
	}

	if actualDevice < len(physicalDevices) {
		return physicalDevices[actualDevice], true
	}
	return 0, false
}

// parsePhysicalDevices parses comma-separated device IDs string into int slice
func parsePhysicalDevices(devicesStr string) []int {
	if devicesStr == "NOT_SET" {
		return nil
	}
	var devices []int
	for _, d := range strings.Split(devicesStr, ",") {
		deviceID, err := strconv.Atoi(strings.TrimSpace(d))
		if err == nil {
			devices = append(devices, deviceID)
		}
	}
	return devices
}

// collectRankMetrics extracts performance metrics from communication JSON file
func collectRankMetrics(ssh executor.CommandRunner,
	podName, namespace, nodeName string, deviceMapping map[int]int) ([]rankMetrics, error) {
	// Find communication JSON file
	cmd := fmt.Sprintf(
		"kubectl exec %s -n %s -- find /tmp/prof_data -path '*/query/communication_*.json' -type f 2>/dev/null | head -1",
		podName, namespace,
	)
	jsonPath, _, err := ssh.Run(cmd)
	if err != nil || strings.TrimSpace(jsonPath) == "" {
		return nil, fmt.Errorf("communication JSON not found")
	}

	// Read JSON content
	cmd = fmt.Sprintf(
		"kubectl exec %s -n %s -- cat '%s' 2>/dev/null",
		podName, namespace, strings.TrimSpace(jsonPath),
	)
	content, _, err := ssh.Run(cmd)
	if err != nil {
		return nil, fmt.Errorf("failed to read communication JSON: %w", err)
	}

	// Parse JSON
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(content), &data); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Extract metrics from "Total HCCL Operators"
	totalOpsRaw, exists := data["Total HCCL Operators"]
	if !exists {
		return nil, fmt.Errorf("total HCCL Operators not found in JSON")
	}
	totalOps, ok := totalOpsRaw.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("total HCCL Operators has unexpected type")
	}

	var metrics []rankMetrics
	for key, value := range totalOps {
		if key == "Slow Rank Suggestion" {
			continue
		}

		// Parse rank ID from key (should be "0", "1", "2", etc.)
		rankID, err := strconv.Atoi(key)
		if err != nil {
			continue
		}

		rankData, ok := value.(map[string]interface{})
		if !ok {
			continue
		}

		commInfo, ok := rankData["Communication Time Info"].(map[string]interface{})
		if !ok {
			continue
		}

		// Get physical device ID from mapping
		physicalDeviceID := unknownPhysicalDeviceID
		if devID, exists := deviceMapping[rankID]; exists {
			physicalDeviceID = devID
		}

		// Extract metrics
		metric := rankMetrics{
			RankID:           rankID,
			PhysicalDeviceID: physicalDeviceID,
			NodeName:         nodeName,
		}

		if val, ok := commInfo["Wait Time(ms)"].(float64); ok {
			metric.WaitTimeMS = val
		}
		if val, ok := commInfo["Transit Time(ms)"].(float64); ok {
			metric.TransitTimeMS = val
		}
		if val, ok := commInfo["Synchronization Time(ms)"].(float64); ok {
			metric.SynchronizationTimeMS = val
		}
		if val, ok := commInfo["Wait Time Ratio"].(float64); ok {
			metric.WaitTimeRatio = val
		}
		if val, ok := commInfo["Synchronization Time Ratio"].(float64); ok {
			metric.SynchronizationTimeRatio = val
		}

		metrics = append(metrics, metric)
	}

	return metrics, nil
}
