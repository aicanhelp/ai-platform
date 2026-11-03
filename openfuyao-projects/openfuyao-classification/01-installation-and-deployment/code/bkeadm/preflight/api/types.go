/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package api

// Operation enumerates the preflight operations supported in this release.
// Adding an operation (e.g. upgrade, scale) only extends the set; the core
// model does not branch on a specific product.
type Operation string

// Supported preflight operations.
const (
	OperationInit          Operation = "init"           // init checks the bootstrap host.
	OperationClusterCreate Operation = "cluster-create" // cluster-create checks the management cluster and nodes.
)

// IsSupportedOperation reports whether op is enabled in this release.
func IsSupportedOperation(op Operation) bool {
	switch op {
	case OperationInit, OperationClusterCreate:
		return true
	default:
		return false
	}
}

// Digest is a content digest, always formatted as "algo:hex". It is used for
// PlanHash, RequestHash and artifact digests. A zero-value
// Digest is the empty string and compares unequal to any real digest.
type Digest string

// String returns the digest token.
func (d Digest) String() string { return string(d) }

// IsZero reports whether the digest is unset.
func (d Digest) IsZero() bool { return d == "" }

// ValueType is the type tag for plan, fact, and displayed values. A malformed
// rule expectation is a BUNDLE_ERROR; an available collector fact whose value
// is incompatible with the rule is an INTERNAL_ERROR, never an environment
// FAIL/WARN.
type ValueType string

// Supported value types.
const (
	ValueTypeBool     ValueType = "bool"     // ValueTypeBool carries a boolean.
	ValueTypeInteger  ValueType = "integer"  // ValueTypeInteger carries an integer.
	ValueTypeBytes    ValueType = "bytes"    // ValueTypeBytes carries a byte count.
	ValueTypeDuration ValueType = "duration" // ValueTypeDuration carries a duration value.
	ValueTypeString   ValueType = "string"   // ValueTypeString carries text.
	ValueTypeEnum     ValueType = "enum"     // ValueTypeEnum carries an enumerated string.
	ValueTypeAddress  ValueType = "address"  // ValueTypeAddress carries a normalized address.
	ValueTypeSet      ValueType = "set"      // ValueTypeSet carries a set of values.
	ValueTypeRange    ValueType = "range"    // ValueTypeRange carries an inclusive range.
	ValueTypeRelation ValueType = "relation" // ValueTypeRelation carries a relation result.
	ValueTypeObject   ValueType = "object"   // ValueTypeObject carries a bounded JSON object.
)

// Valid reports whether t is a supported non-empty value type.
func (t ValueType) Valid() bool {
	switch t {
	case ValueTypeBool, ValueTypeInteger, ValueTypeBytes, ValueTypeDuration,
		ValueTypeString, ValueTypeEnum, ValueTypeAddress, ValueTypeSet,
		ValueTypeRange, ValueTypeRelation:
		return true
	case ValueTypeObject:
		return true
	default:
		return false
	}
}

// InferValueType derives a public display type from a concrete non-sensitive
// value. Product-specific units such as bytes are supplied by the rule or an
// explicit Fact.Type.
func InferValueType(value any) ValueType {
	switch value.(type) {
	case bool:
		return ValueTypeBool
	case int, int32, int64, uint, uint32, uint64, float32, float64:
		return ValueTypeInteger
	case string:
		return ValueTypeString
	case []string, []any:
		return ValueTypeSet
	case map[string]any:
		return ValueTypeObject
	default:
		return ""
	}
}

// Value is a single typed semantic field of an OperationPlan. Adapter fills
// these; the SDK consumes them. Comparator/severity/unknownPolicy are NOT
// carried here — they live in the Bundle rule.
type Value struct {
	Key   string    `json:"key"   yaml:"key"`
	Type  ValueType `json:"type"  yaml:"type"`
	Value any       `json:"value" yaml:"value"`
}

// Subject is a target of checks: the local host, a remote node, a source or
// the management cluster. Roles and metadata are declared/desired, never
// observed; observed values only enter Facts.
type Subject struct {
	ID       string            `json:"id"                 yaml:"id"`
	Role     []string          `json:"role,omitempty"     yaml:"role,omitempty"`
	Address  string            `json:"address,omitempty"  yaml:"address,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// Source is an installation source (image, package, chart, ntp, api) the plan
// actually consumes. AuthRef names a credential without carrying the secret.
// Endpoint is the operator-facing address (usually domain:port). Address is
// an explicit IP:port the operator supplied alongside the domain; when set,
// connectivity checks connect to Address while keeping the domain as the
// HTTP Host and TLS SNI.
type Source struct {
	ID         string `json:"id"                   yaml:"id"`
	Endpoint   string `json:"endpoint,omitempty"   yaml:"endpoint,omitempty"`
	Address    string `json:"address,omitempty"    yaml:"address,omitempty"`
	Kind       string `json:"kind,omitempty"       yaml:"kind,omitempty"`
	Scheme     string `json:"scheme,omitempty"     yaml:"scheme,omitempty"`
	ObjectPath string `json:"objectPath,omitempty" yaml:"objectPath,omitempty"`
	AuthRef    string `json:"authRef,omitempty"    yaml:"authRef,omitempty"`
}

// PlanOrigin records the input files (desensitized) behind a plan. Absolute
// paths and raw file contents do not enter PlanHash.
type PlanOrigin struct {
	Files []string `json:"files,omitempty" yaml:"files,omitempty"`
}

// OperationPlan describes what an operation will actually act on. It holds
// final business semantics only — no comparators, severities or collection
// commands. Passwords, tokens, private keys, full kubeconfig and credentialed
// URLs MUST NOT appear here.
type OperationPlan struct {
	SchemaVersion string           `json:"schemaVersion"      yaml:"schemaVersion"`
	Operation     Operation        `json:"operation"          yaml:"operation"`
	Subjects      []Subject        `json:"subjects,omitempty" yaml:"subjects,omitempty"`
	Values        map[string]Value `json:"values,omitempty"   yaml:"values,omitempty"`
	Sources       []Source         `json:"sources,omitempty"  yaml:"sources,omitempty"`
	Origin        PlanOrigin       `json:"origin"             yaml:"origin"`
	PlanHash      Digest           `json:"planHash,omitempty" yaml:"planHash,omitempty"`
}

// FactError describes why a fact could not be obtained. It is distinct from a
// rule mismatch: a present-but-unsatisfying fact is FAIL/WARN per severity,
// never a FactError.
type FactError struct {
	Code    string `json:"code,omitempty"    yaml:"code,omitempty"`
	Message string `json:"message,omitempty" yaml:"message,omitempty"`
}

// Fact error codes. These are the only values FactError.Code may carry.
const (
	FactErrNotFound         = "NOT_FOUND"
	FactErrPermissionDenied = "PERMISSION_DENIED"
	FactErrUnreachable      = "UNREACHABLE"
	FactErrTimeout          = "TIMEOUT"
	FactErrUnsupported      = "UNSUPPORTED"
	FactErrParseError       = "PARSE_ERROR"
)

// Evidence is a desensitized observation backing a fact. Locator and Observed
// MUST NOT contain passwords, tokens, private keys, full kubeconfig or
// credentialed URLs; they hold only the minimum needed to render a report.
type Evidence struct {
	Kind     string `json:"kind"               yaml:"kind"` // command|file|api|socket|dns|tcp|tls
	Locator  string `json:"locator,omitempty"  yaml:"locator,omitempty"`
	Observed string `json:"observed,omitempty" yaml:"observed,omitempty"`
	Source   string `json:"source,omitempty"   yaml:"source,omitempty"` // collector ID
}

// Fact is a collector result. Collectors return facts; they do not decide the
// public status. Available=false with a non-nil Error means the fact could not
// be obtained and the unknown-policy applies.
type Fact struct {
	Key       string     `json:"key"                yaml:"key"`
	Subject   string     `json:"subject,omitempty"  yaml:"subject,omitempty"`
	Value     any        `json:"value,omitempty"    yaml:"value,omitempty"`
	Type      ValueType  `json:"type,omitempty"     yaml:"type,omitempty"`
	Available bool       `json:"available"          yaml:"available"`
	Evidence  []Evidence `json:"evidence,omitempty" yaml:"evidence,omitempty"`
	Error     *FactError `json:"error,omitempty"    yaml:"error,omitempty"`
	// Display is an optional safe projection used in reports. The evaluator
	// always receives Value; renderers use Display when collectors must keep
	// raw content or private object fields out of reports.
	Display *DisplayValue `json:"display,omitempty" yaml:"display,omitempty"`
}

// IssueSeverity is INFO, WARN or ERROR for a run-level issue.
type IssueSeverity string

// Supported run issue severities.
const (
	IssueInfo  IssueSeverity = "INFO"  // IssueInfo is observation-only.
	IssueWarn  IssueSeverity = "WARN"  // IssueWarn is non-blocking.
	IssueError IssueSeverity = "ERROR" // IssueError is blocking or interrupting.
)

// IssueStage identifies where a run-level problem occurred.
type IssueStage string

// Supported run issue stages.
const (
	IssueStageInput     IssueStage = "INPUT"     // IssueStageInput covers input resolution.
	IssueStagePlan      IssueStage = "PLAN"      // IssueStagePlan covers plan construction.
	IssueStageSetup     IssueStage = "SETUP"     // IssueStageSetup covers runtime setup.
	IssueStageExecution IssueStage = "EXECUTION" // IssueStageExecution covers fact collection and evaluation.
)

// IssueEffect records what the issue did to execution. It is deliberately
// separate from severity: an ERROR may block startup, interrupt an active run,
// or allow independent checks to continue while still blocking installation.
type IssueEffect string

// Supported run issue effects.
const (
	IssueBlockStart IssueEffect = "BLOCK_START" // IssueBlockStart prevents checks from starting.
	IssueContinue   IssueEffect = "CONTINUE"    // IssueContinue allows independent checks to continue.
	IssueInterrupt  IssueEffect = "INTERRUPT"   // IssueInterrupt stops an active run.
)

// RunIssue describes a run-level problem that is not a check result.
type RunIssue struct {
	Code     string        `json:"code"                 yaml:"code"`
	Severity IssueSeverity `json:"severity"             yaml:"severity"`
	Stage    IssueStage    `json:"stage"                yaml:"stage"`
	Effect   IssueEffect   `json:"effect"               yaml:"effect"`
	Field    string        `json:"field,omitempty"      yaml:"field,omitempty"`
	Cause    string        `json:"cause"                yaml:"cause"`
	Action   string        `json:"action,omitempty"     yaml:"action,omitempty"`
}

// ProgressPhase identifies the non-result phase reported to a host while a
// preflight run is in progress. Progress is advisory and is never persisted
// in Report or used to determine the overall result.
type ProgressPhase string

// Supported progress phases.
const (
	ProgressPlan       ProgressPhase = "plan"       // ProgressPlan reports plan construction.
	ProgressCollecting ProgressPhase = "collecting" // ProgressCollecting reports fact collection.
	ProgressEvaluating ProgressPhase = "evaluating" // ProgressEvaluating reports rule evaluation.
	ProgressComplete   ProgressPhase = "complete"   // ProgressComplete reports a completed run.
	ProgressBlocked    ProgressPhase = "blocked"    // ProgressBlocked reports a run that could not start.
	// ProgressIncomplete means execution ended with timeout or cancellation;
	// it must not be rendered as a completed 100% check percentage.
	ProgressIncomplete ProgressPhase = "incomplete"
)

// ProgressEvent reports deterministic work counts. During planning, Total is
// the number of applicable check instances in the resolved plan. During fact
// collection, Completed/Total count unique fact probes and are not a check
// percentage. During evaluation, each applicable check instance advances
// exactly once when it reaches a terminal public status; that is the only
// phase suitable for a check percentage.
type ProgressEvent struct {
	Phase     ProgressPhase `json:"phase"             yaml:"phase"`
	Completed int           `json:"completed"         yaml:"completed"`
	Total     int           `json:"total"             yaml:"total"`
	Current   string        `json:"current,omitempty" yaml:"current,omitempty"`
	Status    Status        `json:"status,omitempty"  yaml:"status,omitempty"`
}

// ProgressFunc is invoked synchronously by the SDK. Implementations should
// be non-blocking and must not mutate the Report or external installation
// state.
type ProgressFunc func(ProgressEvent)

// HasStartBlocker reports whether planning must stop before checks begin.
func HasStartBlocker(issues []RunIssue) bool {
	for _, issue := range issues {
		if issue.Effect == IssueBlockStart {
			return true
		}
	}
	return false
}

// HasErrorIssue reports whether any run-level issue blocks installation.
func HasErrorIssue(issues []RunIssue) bool {
	for _, issue := range issues {
		if issue.Severity == IssueError {
			return true
		}
	}
	return false
}

// DisplayValue is a desensitized, human-readable expected/actual value carried
// in a Result. It never holds secrets.
type DisplayValue struct {
	Type  ValueType `json:"type,omitempty" yaml:"type,omitempty"`
	Value any       `json:"value"          yaml:"value"`
}

// Result is one check instance's outcome. One Finding = one logical line; it is
// not split by multiple evidence. InstanceID is stable and unique within a run.
type Result struct {
	InstanceID  string       `json:"instanceID"            yaml:"instanceID"`
	RuleID      string       `json:"ruleID"                yaml:"ruleID"`
	Name        string       `json:"name,omitempty"        yaml:"name,omitempty"`
	Status      Status       `json:"status"                yaml:"status"`
	Subject     string       `json:"subject,omitempty"     yaml:"subject,omitempty"`
	Expected    DisplayValue `json:"expected,omitempty"    yaml:"expected,omitempty"`
	Actual      DisplayValue `json:"actual,omitempty"      yaml:"actual,omitempty"`
	Evidence    []Evidence   `json:"evidence,omitempty"    yaml:"evidence,omitempty"`
	Message     string       `json:"message,omitempty"     yaml:"message,omitempty"`
	Remediation string       `json:"remediation,omitempty" yaml:"remediation,omitempty"`
	CauseCode   string       `json:"causeCode,omitempty"   yaml:"causeCode,omitempty"`
	DependsOn   string       `json:"dependsOn,omitempty"   yaml:"dependsOn,omitempty"`
}

// Report is the single source of truth consumed by both text and JSON
// renderers. Renderers MUST NOT recompute status, severity or overall.
type Report struct {
	SchemaVersion string     `json:"schemaVersion"          yaml:"schemaVersion"`
	Operation     Operation  `json:"operation,omitempty"    yaml:"operation,omitempty"`
	Overall       Overall    `json:"overall"                yaml:"overall"`
	ExitCode      int        `json:"exitCode"               yaml:"exitCode"`
	Summary       Summary    `json:"summary"                yaml:"summary"`
	Issues        []RunIssue `json:"issues,omitempty"       yaml:"issues,omitempty"`
	Results       []Result   `json:"results,omitempty"      yaml:"results,omitempty"`
}
