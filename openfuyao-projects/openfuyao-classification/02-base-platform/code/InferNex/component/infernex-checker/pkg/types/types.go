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

// Package types defines the shared data structures used across all checker packages,
// including check results, status codes, node info, and report formats.
package types

// CheckStatus represents the status of a check result
type CheckStatus string

const (
	// StatusPassed indicates the check passed successfully.
	StatusPassed CheckStatus = "passed"
	// StatusFailed indicates the check failed and requires attention.
	StatusFailed CheckStatus = "failed"
	// StatusWarning indicates the check found a potential issue.
	StatusWarning CheckStatus = "warning"
	// StatusInfo indicates the check result is informational only.
	StatusInfo CheckStatus = "info"
	// StatusSkipped indicates the check was skipped due to prerequisites not being met.
	StatusSkipped CheckStatus = "skipped"
)

// CheckResult represents the result of a single check item
type CheckResult struct {
	ID         string                 `json:"id"`
	Node       string                 `json:"node,omitempty"`
	Status     CheckStatus            `json:"status"`
	Message    string                 `json:"message"`
	Suggestion string                 `json:"suggestion,omitempty"`
	Detail     map[string]interface{} `json:"detail,omitempty"`
}

// NodeResult represents the check result for a single node
type NodeResult struct {
	Name        string        `json:"name"`
	Status      CheckStatus   `json:"status"`
	Is910Series bool          `json:"is910Series,omitempty"`
	SkippedHCCS bool          `json:"skippedHccs,omitempty"`
	SkipReason  string        `json:"skipReason,omitempty"`
	Checks      []CheckResult `json:"checks"`
}

// HardwareResult represents the check result for the Hardware Layer
type HardwareResult struct {
	Nodes      []NodeResult  `json:"nodes"`
	CrossNodes []CheckResult `json:"cross_nodes,omitempty"`
}

// K8sResult represents the check result for the K8s Layer
type K8sResult struct {
	Status CheckStatus   `json:"status"`
	Checks []CheckResult `json:"checks"`
}

// ConfigEnvResult represents the check result for the Config-Env Layer
type ConfigEnvResult struct {
	Status CheckStatus   `json:"status"`
	Checks []CheckResult `json:"checks"`
}

// Summary holds aggregated check statistics
type Summary struct {
	Total   int `json:"total"`
	Passed  int `json:"passed"`
	Failed  int `json:"failed"`
	Warning int `json:"warning"`
	Info    int `json:"info"`
	Skipped int `json:"skipped"`
}

// Report represents the complete check report
type Report struct {
	Summary            Summary          `json:"summary"`
	Hardware           *HardwareResult  `json:"hardware,omitempty"`
	K8s                *K8sResult       `json:"k8s,omitempty"`
	ConfigEnv          *ConfigEnvResult `json:"config_env,omitempty"`
	HardwareTerminated bool             `json:"hardwareTerminated,omitempty"`
}

// NodeInfo holds connection information for a node (from --nodes file)
type NodeInfo struct {
	Name     string `yaml:"name"`
	IP       string `yaml:"ip"`
	Port     int    `yaml:"port"`
	User     string `yaml:"user"`
	Password string `yaml:"password,omitempty"`
	KeyFile  string `yaml:"keyFile,omitempty"`
}

// NodesConfig is the top-level structure of nodes.yaml
type NodesConfig struct {
	Nodes []NodeInfo `yaml:"nodes"`
}
