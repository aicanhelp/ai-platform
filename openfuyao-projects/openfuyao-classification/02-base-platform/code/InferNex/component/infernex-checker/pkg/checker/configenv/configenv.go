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

// Package configenv implements the Config-Env Layer checks (B-01~B-02),
// validating that the host environment matches the InferNex Helm deployment
// configuration, including model cache path availability and driver version
// compatibility with the vllm-ascend image.
package configenv

import (
	"fmt"
	"strings"
	"sync"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/log"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/parser"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// compatMatrix maps vllm-ascend image version -> compatible Driver version prefix list.
// Driver version prefixes are derived from the official Ascend HDK compatibility matrix.
var compatMatrix = map[string][]string{
	"v0.13.0": {"25.5.", "25.3.", "25.2.", "25.0."},
	"v0.18.0": {"26.0.RC1", "25.7.RC1", "25.5.", "25.3.", "25.2.", "25.0."},
}

// newSSHRunner is the default factory for creating a CommandRunner from a NodeInfo.
// Tests override this variable to inject a fake runner without a real SSH connection.
var newSSHRunner = executor.NewSSHRunner

// CheckConfigEnv runs Config-Env Layer checks
func CheckConfigEnv(nodes []types.NodeInfo, values map[string]interface{}) *types.ConfigEnvResult {
	result := &types.ConfigEnvResult{Status: types.StatusPassed}
	var mu sync.Mutex
	var wg sync.WaitGroup

	cachePath, _ := parser.GetNestedString(values, "global.cachePath")
	imageTag, _ := parser.GetNestedString(values, "inference-backend.images.inferenceEngine.tag")

	for _, node := range nodes {
		wg.Add(1)
		go func(n types.NodeInfo) {
			defer wg.Done()
			checks := checkSingleNode(n, cachePath, imageTag)

			mu.Lock()
			result.Checks = append(result.Checks, checks...)
			for _, c := range checks {
				if c.Status == types.StatusFailed {
					result.Status = types.StatusFailed
				}
			}
			mu.Unlock()
		}(node)
	}
	wg.Wait()

	return result
}

func checkSingleNode(node types.NodeInfo, cachePath, imageTag string) []types.CheckResult {
	var results []types.CheckResult

	runner, cleanup, err := newSSHRunner(node)
	if err != nil {
		log.Error(fmt.Sprintf("node %s SSH connection failed: %v", node.Name, err))
		return []types.CheckResult{{
			ID:         "B-00",
			Node:       node.Name,
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("SSH connection failed: %v", err),
			Suggestion: "Please check node connection configuration",
		}}
	}
	defer cleanup()

	// B-01 model cache path availability
	b01 := checkB01(runner, node.Name, cachePath)
	results = append(results, b01)
	if b01.Status == types.StatusFailed {
		return results
	}

	// B-02 Driver and vllm-ascend version compatibility
	b02 := checkB02(runner, node.Name, imageTag)
	results = append(results, b02)

	return results
}

// checkB01 checks model cache path availability
func checkB01(ssh executor.CommandRunner, nodeName, cachePath string) types.CheckResult {
	if cachePath == "" {
		return types.CheckResult{
			ID:         "B-01",
			Node:       nodeName,
			Status:     types.StatusFailed,
			Message:    "global.cachePath is not configured in values.yaml",
			Suggestion: "Please configure the global.cachePath field in values.yaml",
		}
	}

	// step 1: check if directory exists
	_, _, err := ssh.Run(fmt.Sprintf("ls %s", cachePath))
	if err != nil {
		return types.CheckResult{
			ID:      "B-01",
			Node:    nodeName,
			Status:  types.StatusFailed,
			Message: fmt.Sprintf("Step 1: directory %s does not exist", cachePath),
			Suggestion: fmt.Sprintf("Create the directory and grant write permission: "+
				"mkdir -p %s && chmod 755 %s", cachePath, cachePath),
		}
	}

	// step 2: check if writable
	testFile := fmt.Sprintf("%s/.infernex_write_test", cachePath)
	_, _, err = ssh.Run(fmt.Sprintf("touch %s && rm %s", testFile, testFile))
	if err != nil {
		return types.CheckResult{
			ID:         "B-01",
			Node:       nodeName,
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("Step 2: directory %s exists but is not writable", cachePath),
			Suggestion: fmt.Sprintf("Fix directory permissions: chmod 755 %s", cachePath),
		}
	}

	return types.CheckResult{
		ID:      "B-01",
		Node:    nodeName,
		Status:  types.StatusPassed,
		Message: fmt.Sprintf("%s directory exists and is writable", cachePath),
	}
}

// checkB02 checks Driver version compatibility with the vllm-ascend image
func checkB02(ssh executor.CommandRunner, nodeName, imageTag string) types.CheckResult {
	driverVersion, checkResult, ok := readDriverVersion(ssh, nodeName)
	if !ok {
		return checkResult
	}
	return checkDriverCompatibility(nodeName, driverVersion, imageTag)
}

func readDriverVersion(ssh executor.CommandRunner, nodeName string) (string, types.CheckResult, bool) {
	stdout, _, err := ssh.Run("cat /usr/local/Ascend/driver/version.info")
	if err != nil {
		return "", types.CheckResult{
			ID:         "B-02",
			Node:       nodeName,
			Status:     types.StatusFailed,
			Message:    "Step 1: failed to read /usr/local/Ascend/driver/version.info",
			Suggestion: "Please confirm the driver is installed and the file exists",
		}, false
	}

	driverVersion := parseDriverVersion(stdout)
	if driverVersion == "" {
		return "", types.CheckResult{
			ID:         "B-02",
			Node:       nodeName,
			Status:     types.StatusFailed,
			Message:    "Step 1: unable to parse Driver version from version.info",
			Suggestion: "Please confirm the version.info file format is correct",
		}, false
	}
	return driverVersion, types.CheckResult{}, true
}

// parseDriverVersion extracts the version number from version.info content
func parseDriverVersion(content string) string {
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToLower(line), "version=") {
			return line[len("version="):]
		}
	}
	return ""
}

func checkDriverCompatibility(nodeName, driverVersion, imageTag string) types.CheckResult {
	if imageTag == "" {
		return types.CheckResult{
			ID:         "B-02",
			Node:       nodeName,
			Status:     types.StatusFailed,
			Message:    "Step 2: inference-backend.images.inferenceEngine.tag is not configured in values.yaml",
			Suggestion: "Please configure the inference engine image version in values.yaml",
			Detail:     map[string]interface{}{"driver_version": driverVersion},
		}
	}

	compatDriverPrefixes, ok := compatMatrix[imageTag]
	if !ok {
		return types.CheckResult{
			ID:     "B-02",
			Node:   nodeName,
			Status: types.StatusWarning,
			Message: fmt.Sprintf("Step 2: image version %s is not in the built-in compatibility matrix, "+
				"please refer to the official Ascend compatibility matrix to confirm", imageTag),
			Suggestion: "Please refer to the latest official Ascend compatibility matrix to confirm whether " +
				"the current Driver version is compatible; if confirmed incompatible, upgrade or downgrade " +
				"the Driver version, or switch to a matching vllm-ascend image version",
			Detail: map[string]interface{}{"driver_version": driverVersion},
		}
	}

	// check if driver version matches any compatible prefix (case-insensitive)
	driverVersionLower := strings.ToLower(driverVersion)
	for _, prefix := range compatDriverPrefixes {
		if strings.HasPrefix(driverVersionLower, strings.ToLower(prefix)) {
			return types.CheckResult{
				ID:      "B-02",
				Node:    nodeName,
				Status:  types.StatusPassed,
				Message: fmt.Sprintf("Driver %s is compatible with image %s", driverVersion, imageTag),
				Detail:  map[string]interface{}{"driver_version": driverVersion},
			}
		}
	}
	return types.CheckResult{
		ID:     "B-02",
		Node:   nodeName,
		Status: types.StatusWarning,
		Message: fmt.Sprintf("Step 2: Driver %s is not in the built-in compatibility matrix for image %s "+
			"(compatible driver prefixes: %s), please refer to the official Ascend compatibility matrix to confirm",
			driverVersion, imageTag, strings.Join(compatDriverPrefixes, ", ")),
		Suggestion: "Please refer to the latest official Ascend compatibility matrix to confirm whether " +
			"the current Driver version is compatible; if confirmed incompatible, upgrade or downgrade " +
			"the Driver version, or switch to a matching vllm-ascend image version",
		Detail: map[string]interface{}{"driver_version": driverVersion},
	}
}
