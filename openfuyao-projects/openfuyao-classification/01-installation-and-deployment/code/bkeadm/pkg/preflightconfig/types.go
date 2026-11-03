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

// Package preflightconfig contains the human-authored preflight policy.
// Reviewed checks have no rule IDs or expressions. Optional custom checks use
// bounded CEL over facts from explicitly supported collectors.
package preflightconfig

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"gopkg.openfuyao.cn/preflight/api"
)

const (
	// APIVersion is the only policy API version accepted by this package.
	APIVersion = "preflight.openfuyao.io/v1"
	// Kind identifies a preflight policy document.
	Kind = "PreflightConfig"
)

// Config is the release/runtime policy document. Release builds embed the
// reviewed default alongside its matching Bundle. A bke runtime may accept a
// complete external document while keeping rule code and source evidence in
// the release Bundle fixed.
type Config struct {
	APIVersion string     `yaml:"apiVersion" json:"apiVersion"`
	Kind       string     `yaml:"kind"       json:"kind"`
	Metadata   Metadata   `yaml:"metadata"   json:"metadata"`
	Operations Operations `yaml:"operations" json:"operations"`
}

// Metadata carries the human-readable policy description.
type Metadata struct {
	Description string `yaml:"description,omitempty" json:"description,omitempty"`
}

// Operations groups the policies supported by the bke host adapter.
type Operations struct {
	Init          OperationPolicy `yaml:"init"          json:"init"`
	ClusterCreate OperationPolicy `yaml:"clusterCreate" json:"clusterCreate"`
}

// OperationPolicy is intentionally small and generic.  New checks are added
// as data to one of these lists; the evaluator remains unchanged.
type OperationPolicy struct {
	Platform          PlatformPolicy          `yaml:"platform,omitempty"          json:"platform,omitempty"`
	Ports             []PortCheck             `yaml:"ports,omitempty"             json:"ports,omitempty"`
	Resources         []ResourceCheck         `yaml:"resources,omitempty"         json:"resources,omitempty"`
	Paths             []PathCheck             `yaml:"paths,omitempty"             json:"paths,omitempty"`
	Tools             []ToolCheck             `yaml:"tools,omitempty"             json:"tools,omitempty"`
	Topology          TopologyPolicy          `yaml:"topology,omitempty"          json:"topology,omitempty"`
	ManagementCluster ManagementClusterPolicy `yaml:"managementCluster,omitempty" json:"managementCluster,omitempty"`
	CustomChecks      []CustomCheck           `yaml:"customChecks,omitempty"      json:"customChecks,omitempty"`
}

// PlatformPolicy lists the operating-system variants accepted by an operation.
type PlatformPolicy struct {
	Distributions []PlatformDistribution `yaml:"distributions,omitempty" json:"distributions,omitempty"`
}

// PlatformDistribution describes one supported distribution and architecture set.
type PlatformDistribution struct {
	Name           string   `yaml:"name"                     json:"name"`
	Version        string   `yaml:"version"                  json:"version"`
	Architectures  []string `yaml:"architectures"            json:"architectures"`
	KernelPrefixes []string `yaml:"kernelPrefixes,omitempty" json:"kernelPrefixes,omitempty"`
}

// TopologyPolicy defines reviewed cluster role-count and uniqueness constraints.
type TopologyPolicy struct {
	Masters NodeCountPolicy `yaml:"masters,omitempty" json:"masters,omitempty"`
	Etcd    NodeCountPolicy `yaml:"etcd,omitempty"    json:"etcd,omitempty"`
	Workers NodeCountPolicy `yaml:"workers,omitempty" json:"workers,omitempty"`
	Unique  []string        `yaml:"unique,omitempty"  json:"unique,omitempty"`
}

// NodeCountPolicy defines a minimum role count and optional odd-count requirement.
type NodeCountPolicy struct {
	Minimum int  `yaml:"minimum,omitempty" json:"minimum,omitempty"`
	Odd     bool `yaml:"odd,omitempty"     json:"odd,omitempty"`
}

// ManagementClusterPolicy defines read-only management-cluster prerequisites.
type ManagementClusterPolicy struct {
	API                bool     `yaml:"api,omitempty"                json:"api,omitempty"`
	Discovery          bool     `yaml:"discovery,omitempty"          json:"discovery,omitempty"`
	CRDs               []string `yaml:"crds,omitempty"               json:"crds,omitempty"`
	RequiredKubeconfig string   `yaml:"requiredKubeconfig,omitempty" json:"requiredKubeconfig,omitempty"`
}

// CheckCommon contains presentation and result-policy fields shared by checks.
type CheckCommon struct {
	Description string   `yaml:"description"           json:"description"`
	When        string   `yaml:"when,omitempty"        json:"when,omitempty"`
	Roles       []string `yaml:"roles,omitempty"       json:"roles,omitempty"`
	Failure     string   `yaml:"failure,omitempty"     json:"failure,omitempty"`
	Unknown     string   `yaml:"unknown,omitempty"     json:"unknown,omitempty"`
	Remediation string   `yaml:"remediation,omitempty" json:"remediation,omitempty"`
}

// CustomCheck defines one operation-scoped check over a supported collector.
// CEL receives only the collected value as the variable "actual".
type CustomCheck struct {
	CheckCommon `             yaml:",inline"    json:",inline"`
	Key         string          `yaml:"key"        json:"key"`
	Expected    string          `yaml:"expected"   json:"expected"`
	Collector   CustomCollector `yaml:"collector"  json:"collector"`
	Expression  string          `yaml:"expression" json:"expression"`
}

// CustomCollector selects a bounded built-in fact collector. For fileContent,
// Path is the file to read. For exec, Path is an executable script and
// ResultType describes its Fact JSON value.
type CustomCollector struct {
	Type         string   `yaml:"type"                   json:"type"`
	Path         string   `yaml:"path"                   json:"path"`
	Args         []string `yaml:"args,omitempty"         json:"args,omitempty"`
	ResultType   string   `yaml:"resultType,omitempty"   json:"resultType,omitempty"`
	PublicFields []string `yaml:"publicFields,omitempty" json:"publicFields,omitempty"`
}

// PortCheck defines one reviewed network-port check.
type PortCheck struct {
	CheckCommon `       yaml:",inline"            json:",inline"`
	Key         string `yaml:"key"                json:"key"`
	DefaultPort int    `yaml:"defaultPort"        json:"defaultPort"`
	PortFrom    string `yaml:"portFrom,omitempty" json:"portFrom,omitempty"`
	Protocol    string `yaml:"protocol,omitempty" json:"protocol,omitempty"`
}

// ResourceCheck defines one reviewed minimum-resource check.
type ResourceCheck struct {
	CheckCommon `       yaml:",inline"        json:",inline"`
	Key         string `yaml:"key"            json:"key"`
	// Minimum accepts a plain byte count or a string with a binary unit
	// suffix (K/M/G/T, case-insensitive; 1024 base), e.g. 4294967296, "4G".
	Minimum ByteSize `yaml:"minimum"            json:"minimum"`
}

// ByteSize is a byte quantity that unmarshals from a YAML integer (raw bytes)
// or a string with a K/M/G/T suffix. Marshaling back emits the raw byte count.
type ByteSize int64

func (b *ByteSize) UnmarshalYAML(node *yaml.Node) error {
	text := strings.TrimSpace(node.Value)
	if text == "" {
		return fmt.Errorf("size must be a byte count or a K/M/G/T suffixed string, got empty value")
	}
	var value int64
	if node.Kind == yaml.ScalarNode && node.Tag != "!!str" {
		parsed, err := strconv.ParseInt(text, 10, 64)
		if err != nil {
			return fmt.Errorf("size %q must be a byte count or a K/M/G/T suffixed string", text)
		}
		if parsed < 0 {
			return fmt.Errorf("size %q must be a non-negative number with optional B/K/M/G/T suffix", text)
		}
		value = parsed
	} else {
		numeric, multiplier, ok := splitSizeSuffix(text)
		if !ok {
			return fmt.Errorf("size %q must be a non-negative number with optional B/K/M/G/T suffix", text)
		}
		parsed, err := strconv.ParseFloat(numeric, 64)
		if err != nil || parsed < 0 {
			return fmt.Errorf("size %q must be a non-negative number with optional B/K/M/G/T suffix", text)
		}
		scaled := parsed * float64(multiplier)
		if scaled > math.MaxInt64 {
			return fmt.Errorf("size %q overflows the supported range", text)
		}
		value = int64(scaled)
	}
	*b = ByteSize(value)
	return nil
}

// splitSizeSuffix splits "4G"/"512m"/"1024K"/"100B" into the numeric part and
// its 1024-base multiplier. The unit is case-insensitive; B means bytes.
func splitSizeSuffix(text string) (string, int64, bool) {
	if text == "" {
		return "", 0, false
	}
	unit := strings.ToLower(text[len(text)-1:])
	numeric := text[:len(text)-1]
	switch unit {
	case "b":
		return numeric, 1, numeric != ""
	case "k":
		return numeric, 1024, numeric != ""
	case "m":
		return numeric, 1024 * 1024, numeric != ""
	case "g":
		return numeric, 1024 * 1024 * 1024, numeric != ""
	case "t":
		return numeric, 1024 * 1024 * 1024 * 1024, numeric != ""
	default:
		return "", 0, false
	}
}

// PathCheck defines reviewed requirements for one filesystem path. The
// built-in "workspace" key keeps its semantic checks; any other key requires
// Path (an absolute filesystem path) and is checked by existence facts.
type PathCheck struct {
	CheckCommon `         yaml:",inline"           json:",inline"`
	Key         string   `yaml:"key"               json:"key"`
	Require     []string `yaml:"require,omitempty" json:"require,omitempty"`
	Path        string   `yaml:"path,omitempty"    json:"path,omitempty"`
}

// ToolCheck defines one reviewed command-availability check.
type ToolCheck struct {
	CheckCommon `         yaml:",inline"         json:",inline"`
	Key         string   `yaml:"key"             json:"key"`
	Names       []string `yaml:"names,omitempty" json:"names,omitempty"`
}

// EffectivePort records the value after plan override resolution.  The plan
// always wins over defaultPort; defaultPort is only a policy fallback.
type EffectivePort struct {
	Key         string
	Port        int
	Protocol    string
	ValueSource string // plan | config-default
}

// Operation returns the policy for op.
func (c Config) Operation(op api.Operation) OperationPolicy {
	if op == api.OperationClusterCreate {
		return c.Operations.ClusterCreate
	}
	return c.Operations.Init
}

// CheckID derives an internal stable key. It is intentionally not part of the
// YAML contract; changing a description does not change the identity.
func CheckID(operation api.Operation, kind, key string) string {
	return fmt.Sprintf(
		"CONFIG-%s-%s-%s",
		strings.ToUpper(string(operation)),
		strings.ToUpper(kind),
		strings.ToUpper(strings.ReplaceAll(key, ".", "-")),
	)
}
