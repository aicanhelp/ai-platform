/******************************************************************
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package config

import (
	"encoding/json"
	"time"
)

// OperationMode operation mode
type OperationMode string

const (
	ModeQuery         OperationMode = "fileQuery"     // File query
	ModeClean         OperationMode = "fileClean"     // File clean
	ModeProgramCheck  OperationMode = "programCheck"  // Program check
	ModeClockCheck    OperationMode = "clockCheck"    // Clock synchronized check
	ModeRouteCheck    OperationMode = "routeCheck"    // Route check
	ModeIptablesCheck OperationMode = "iptablesCheck" // Iptables check

	DirectoryMode = 0755 // Directory permissions: owner read, write, execute; group and others read and execute
	FileMode      = 0644 // File permissions: owner read, write; group and others read only
	// SecretFileMode for private files (owner read/write only)
	SecretFileMode = 0600
	// ExecutableFileMode for executable files
	ExecutableFileMode = 0755
)

// AppConfig application configuration
type AppConfig struct {
	Mode         OperationMode `json:"mode" yaml:"mode"`                   // Operation mode: query, clean
	LogFile      string        `json:"log_file" yaml:"log_file"`           // Log file path
	OutputFormat string        `json:"output_format" yaml:"output_format"` // Output format
	Paths        []string      `json:"paths" yaml:"paths"`                 // List of paths to operate on

	// Clean related configuration
	CleanForce bool `json:"clean_force" yaml:"clean_force"` // Force clean (no prompt)

	ProgramList []ProgramConfig `json:"program_list" yaml:"program_list"` // List of programs to check

	Hosts          []Host `json:"hosts" yaml:"hosts"` // host list
	ClockThreshold int64  `json:"clock_threshold" yaml:"clock_threshold"`

	// New check configurations
	KernelCheck *KernelCheckConfig `json:"kernel_check,omitempty" yaml:"kernel_check,omitempty"`
	PortCheck   *PortCheckConfig   `json:"port_check,omitempty" yaml:"port_check,omitempty"`
	DiskCheck   *DiskCheckConfig   `json:"disk_check,omitempty" yaml:"disk_check,omitempty"`
	Dispatch    *DispatchConfig    `json:"dispatch,omitempty" yaml:"dispatch,omitempty"`
}

// Host info of each host to be provided
type Host struct {
	IP       string   `json:"ip" yaml:"ip"`
	UserName string   `json:"username" yaml:"username"`
	Password string   `json:"password" yaml:"password"`
	Port     string   `json:"port" yaml:"port"`
	Role     []string `json:"role" yaml:"role"`
	// Target architecture (amd64/arm64), auto-detected if empty.
	Arch string `json:"arch,omitempty" yaml:"arch,omitempty"`
}

// FileInfo file/directory information
type FileInfo struct {
	Path        string `json:"path"`
	Exists      bool   `json:"exists"`
	IsDir       bool   `json:"is_dir"`
	Owner       string `json:"owner,omitempty"` // File owner user
	Group       string `json:"group,omitempty"` // File owner group
	Permissions string `json:"permissions"`     // File permissions
	Error       string `json:"error,omitempty"`
}

// QueryResult query result
type QueryResult struct {
	Timestamp time.Time    `json:"timestamp"`
	Files     []FileInfo   `json:"files"`
	Summary   QuerySummary `json:"summary"`
}

// QuerySummary query summary information
type QuerySummary struct {
	TotalChecked int `json:"total_checked"`
	TotalExists  int `json:"total_exists"`
	TotalMissing int `json:"total_missing"`
	TotalDirs    int `json:"total_dirs"`
	TotalFiles   int `json:"total_files"`
}

// CleanResult clean result
type CleanResult struct {
	Timestamp time.Time    `json:"timestamp"`
	Deleted   []FileInfo   `json:"deleted"`
	Failed    []FileInfo   `json:"failed"`
	Skipped   []FileInfo   `json:"skipped"`
	Summary   CleanSummary `json:"summary"`
}

// CleanSummary clean summary information
type CleanSummary struct {
	TotalChecked int `json:"total_checked"`
	TotalDeleted int `json:"total_deleted"`
	TotalFailed  int `json:"total_failed"`
	TotalSkipped int `json:"total_skipped"`
}

// ProgramConfig program check configuration
type ProgramConfig struct {
	Name        string `json:"name" yaml:"name"`                 // Program name
	ShouldExist bool   `json:"should_exist" yaml:"should_exist"` // Expected existence status
}

// ProgramInfo program information
type ProgramInfo struct {
	Name        string `json:"name"`
	ShouldExist bool   `json:"should_exist"`
	Installed   bool   `json:"installed"`
	Passed      bool   `json:"passed"`
	Version     string `json:"version,omitempty"`
	Path        string `json:"path,omitempty"`
	Error       string `json:"error,omitempty"`
}

// ProgramCheckResult program check result
type ProgramCheckResult struct {
	Timestamp time.Time           `json:"timestamp"`
	Programs  []ProgramInfo       `json:"programs"`
	Summary   ProgramCheckSummary `json:"summary"`
}

// ProgramCheckSummary program check summary
type ProgramCheckSummary struct {
	TotalChecked   int `json:"total_checked"`
	TotalInstalled int `json:"total_installed"`
	TotalMissing   int `json:"total_missing"`
	TotalPassed    int `json:"total_passed"`
	TotalFailed    int `json:"total_failed"`
}

// ClockCheckInfo clock check info of each host
type ClockCheckInfo struct {
	Host          string   `json:"host"`
	Role          []string `json:"role"`
	IsSynced      bool     `json:"is_synced"`
	TimeDiff      int64    `json:"time_diff"`
	RemoteTime    int64    `json:"remote_time"`
	ReferenceTime int64    `json:"reference_time"`
	Error         string   `json:"error,omitempty"`
	IsLocal       bool     `json:"is_local,omitempty"`
}

// ClockCheckResult clock check result
type ClockCheckResult struct {
	Timestamp time.Time        `json:"timestamp"`
	Clocks    []ClockCheckInfo `json:"clocks"`
	Result    string           `json:"Result"`
}

// RouteInfo route information
type RouteInfo struct {
	Exists      bool   `json:"exists"`
	Destination string `json:"destination"`
	Gateway     string `json:"gateway"`
	Interface   string `json:"interface"`
	Metric      int    `json:"metric"`
}

// RouteCheckResult route check result
type RouteCheckResult struct {
	Timestamp     time.Time    `json:"timestamp"`
	DefaultRoutes []RouteInfo  `json:"default_routes"`
	Summary       RouteSummary `json:"summary"`
}

// RouteSummary route check summary
type RouteSummary struct {
	TotalRoutes int    `json:"total_routes"`
	CheckPassed bool   `json:"check_passed"`
	Message     string `json:"message"`
}

// ForwardPolicy iptables forward chain policy
type ForwardPolicy struct {
	Chain    string `json:"chain"`
	Policy   string `json:"policy"`
	IsAccept bool   `json:"is_accept"`
}

// IptablesCheckResult iptables check result
type IptablesCheckResult struct {
	Timestamp     time.Time       `json:"timestamp"`
	ForwardPolicy ForwardPolicy   `json:"forward_policy"`
	Summary       IptablesSummary `json:"summary"`
}

// IptablesSummary iptables check summary
type IptablesSummary struct {
	CheckPassed bool   `json:"check_passed"`
	Message     string `json:"message"`
}

// ========== Kernel Check Types ==========

// KernelCheckConfig kernel check configuration
type KernelCheckConfig struct {
	MinVersion string `json:"min_version" yaml:"min_version"`
	Operator   string `json:"operator" yaml:"operator"`
}

// KernelCheckResult kernel check result
type KernelCheckResult struct {
	Timestamp  time.Time     `json:"timestamp"`
	KernelInfo KernelInfo    `json:"kernel_info"`
	Summary    KernelSummary `json:"summary"`
}

// KernelInfo kernel basic information
type KernelInfo struct {
	Version     string `json:"version"`
	OS          string `json:"os"`
	Arch        string `json:"arch"`
	ExpectedVer string `json:"expected_version"`
	Operator    string `json:"operator"`
	IsValid     bool   `json:"is_valid"`
	Error       string `json:"error,omitempty"`
}

// KernelSummary kernel check summary
type KernelSummary struct {
	TotalChecked int `json:"total_checked"`
	Passed       int `json:"passed"`
	Failed       int `json:"failed"`
}

// ========== Port Check Types ==========

// PortCheckConfig port check configuration
type PortCheckConfig struct {
	Ports   map[string][]string `json:"ports" yaml:"ports"`     // Ports mapped by role: bootstrap, master, worker
	Timeout int                 `json:"timeout" yaml:"timeout"` // TCP dial timeout in seconds
}

// PortCheckResult port check result
type PortCheckResult struct {
	Timestamp time.Time       `json:"timestamp"`
	Ports     []PortCheckInfo `json:"ports"`
	Summary   PortSummary     `json:"summary"`
}

// PortCheckInfo single port check information
type PortCheckInfo struct {
	Port     string `json:"port"`
	Protocol string `json:"protocol"`
	IsUsed   bool   `json:"is_used"`
	Process  string `json:"process,omitempty"`
	Error    string `json:"error,omitempty"`
}

// PortSummary port check summary
type PortSummary struct {
	TotalChecked int `json:"total_checked"`
	Used         int `json:"used"`
	Free         int `json:"free"`
}

// ========== Disk Check Types ==========

// DiskCheckConfig disk check configuration
type DiskCheckConfig struct {
	CheckItems []DiskCheckItem `json:"check_items" yaml:"check_items"`
}

// DiskCheckItem disk check item configuration
type DiskCheckItem struct {
	Path      string   `json:"path" yaml:"path"`
	MinFreeGB uint64   `json:"min_free_gb" yaml:"min_free_gb"`
	Roles     []string `json:"roles,omitempty" yaml:"roles,omitempty"`
}

// DiskCheckResult disk check result
type DiskCheckResult struct {
	Timestamp time.Time   `json:"timestamp"`
	Spaces    []DiskSpace `json:"spaces"`
	Summary   DiskSummary `json:"summary"`
}

// DiskSpace disk space check item
type DiskSpace struct {
	Path         string  `json:"path"`
	Total        uint64  `json:"total_bytes"`
	Free         uint64  `json:"free_bytes"`
	Used         uint64  `json:"used_bytes"`
	UsedPercent  float64 `json:"used_percent"`
	MinFree      uint64  `json:"min_free_bytes"`
	IsSufficient bool    `json:"is_sufficient"`
	Filesystem   string  `json:"filesystem"`
	Error        string  `json:"error,omitempty"`
}

// DiskSummary disk check summary
type DiskSummary struct {
	TotalChecked     int `json:"total_checked"`
	SufficientPath   int `json:"sufficient_paths"`
	InsufficientPath int `json:"insufficient_paths"`
}

// ========== Dispatch Types ==========

// DispatchConfig dispatch configuration
type DispatchConfig struct {
	Timeout         int      `json:"timeout" yaml:"timeout"`
	PollInterval    int      `json:"poll_interval" yaml:"poll_interval"`
	WorkDir         string   `json:"work_dir" yaml:"work_dir"`
	ResultDir       string   `json:"result_dir" yaml:"result_dir"`
	ConcurrentLimit int      `json:"concurrent_limit" yaml:"concurrent_limit"`
	Checks          []string `json:"checks" yaml:"checks"`
	SkipChecks      []string `json:"skip_checks" yaml:"skip_checks"`
}

// DispatchResult dispatch execution result
type DispatchResult struct {
	Timestamp time.Time       `json:"timestamp"`
	Duration  string          `json:"duration"`
	Nodes     []NodeResult    `json:"nodes"`
	Summary   DispatchSummary `json:"summary"`
}

// NodeResult node execution result
type NodeResult struct {
	IP         string        `json:"ip"`
	Role       []string      `json:"role"`
	Status     string        `json:"status"`
	StartTime  time.Time     `json:"start_time"`
	EndTime    time.Time     `json:"end_time,omitempty"`
	ResultFile string        `json:"result_file,omitempty"`
	ErrorFile  string        `json:"error_file,omitempty"`
	Error      string        `json:"error,omitempty"`
	Results    []CheckResult `json:"results,omitempty"`
}

// CheckResult single check result
type CheckResult struct {
	CheckType  string          `json:"check_type"`
	Status     string          `json:"status"`
	Detail     string          `json:"detail"`
	DetailData json.RawMessage `json:"detail_data,omitempty"` // Detailed result data (port list, disk info, etc.)
}

// DispatchSummary dispatch execution summary
type DispatchSummary struct {
	TotalNodes   int    `json:"total_nodes"`
	SuccessNodes int    `json:"success_nodes"`
	FailedNodes  int    `json:"failed_nodes"`
	RunningNodes int    `json:"running_nodes"`
	Result       string `json:"result,omitempty"`
}
