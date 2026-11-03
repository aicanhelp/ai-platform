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

package preflightconfig

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/celcheck"
)

const (
	minimumNetworkPort  = 1
	maximumNetworkPort  = 65535
	maximumCustomChecks = 64
	maximumPublicFields = 32
)

// LoadFile reads exactly one external configuration document.
func LoadFile(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read preflight config %q: %w", path, err)
	}
	c, err := Load(b)
	if err != nil {
		return Config{}, fmt.Errorf("parse preflight config %q: %w", path, err)
	}
	return c, nil
}

// Load decodes and validates a human-authored policy.  KnownFields prevents
// misspelled fields from silently changing the effective policy.
func Load(raw []byte) (Config, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return Config{}, fmt.Errorf("config is empty")
	}
	var c Config
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return Config{}, fmt.Errorf("yaml: %w", err)
	}
	var extra Config
	if err := dec.Decode(&extra); err == nil {
		return Config{}, fmt.Errorf("config must be a single YAML document")
	} else if err != io.EOF {
		return Config{}, fmt.Errorf("trailing YAML: %w", err)
	}
	if err := Validate(c); err != nil {
		return Config{}, err
	}
	return c, nil
}

// Validate checks a decoded policy against the supported static contract.
func Validate(c Config) error {
	if c.APIVersion != APIVersion {
		return fmt.Errorf("apiVersion must be %q", APIVersion)
	}
	if c.Kind != Kind {
		return fmt.Errorf("kind must be %q", Kind)
	}
	if err := validateOperation("init", c.Operations.Init); err != nil {
		return err
	}
	if err := validateOperation("clusterCreate", c.Operations.ClusterCreate); err != nil {
		return err
	}
	return nil
}

func validateOperation(name string, p OperationPolicy) error {
	knownPorts, err := reviewedPorts(name)
	if err != nil {
		return err
	}
	seen := map[string]struct{}{}
	if err := validatePorts(name, p.Ports, knownPorts, seen); err != nil {
		return err
	}
	if err := validateResources(name, p.Resources, seen); err != nil {
		return err
	}
	if err := validatePaths(name, p.Paths, seen); err != nil {
		return err
	}
	if err := validateTools(name, p.Tools, seen); err != nil {
		return err
	}
	if err := validateCustomChecks(name, p.CustomChecks, seen); err != nil {
		return err
	}
	if err := validateTopology(name, p.Topology); err != nil {
		return err
	}
	return validateManagementCluster(name, p.ManagementCluster)
}

var customKeyPattern = regexp.MustCompile(`^[a-z][a-z0-9-]{0,62}$`)
var publicFieldPattern = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]{0,62}$`)

func validateCustomChecks(operation string, checks []CustomCheck, seen map[string]struct{}) error {
	if len(checks) > maximumCustomChecks {
		return fmt.Errorf("operations.%s.customChecks exceeds %d items", operation, maximumCustomChecks)
	}
	for _, check := range checks {
		if !customKeyPattern.MatchString(check.Key) {
			return fmt.Errorf("operations.%s custom check key %q is invalid", operation, check.Key)
		}
		if strings.TrimSpace(check.Description) == "" || strings.TrimSpace(check.Expected) == "" ||
			strings.TrimSpace(check.Remediation) == "" {
			return fmt.Errorf(
				"operations.%s custom check %s requires description, expected and remediation",
				operation,
				check.Key,
			)
		}
		duplicate, err := recordValidationKey(seen, "custom:"+check.Key)
		if err != nil {
			return fmt.Errorf("operations.%s customChecks: %w", operation, err)
		}
		if duplicate {
			return fmt.Errorf("operations.%s duplicate custom check key %q", operation, check.Key)
		}
		if err := validateCommon(operation, check.CheckCommon); err != nil {
			return err
		}
		if operation == "init" && len(check.Roles) != 0 {
			return fmt.Errorf("operations.init custom check %s does not support roles", check.Key)
		}
		if err := validateRoles(operation, check.Key, check.Roles); err != nil {
			return err
		}
		if _, err := celcheck.Compile(check.Expression); err != nil {
			return fmt.Errorf("operations.%s custom check %s: %w", operation, check.Key, err)
		}
		if err := validateCustomCollector(operation, check); err != nil {
			return err
		}
	}
	return nil
}

func validateCustomCollector(operation string, check CustomCheck) error {
	collector := check.Collector
	if !filepath.IsAbs(collector.Path) || filepath.Clean(collector.Path) != collector.Path {
		return fmt.Errorf(
			"operations.%s custom check %s collector.path must be a clean absolute path",
			operation,
			check.Key,
		)
	}
	switch collector.Type {
	case "fileContent":
		if len(collector.Args) != 0 || collector.ResultType != "" || len(collector.PublicFields) != 0 {
			return fmt.Errorf(
				"operations.%s custom check %s fileContent only supports path",
				operation,
				check.Key,
			)
		}
	case "exec":
		if !validResultType(collector.ResultType) {
			return fmt.Errorf(
				"operations.%s custom check %s has unsupported resultType %q",
				operation,
				check.Key,
				collector.ResultType,
			)
		}
		if err := validatePublicFields(operation, check.Key, collector); err != nil {
			return err
		}
		for _, argument := range collector.Args {
			if strings.ContainsRune(argument, 0) {
				return fmt.Errorf("operations.%s custom check %s args contain NUL", operation, check.Key)
			}
		}
	default:
		return fmt.Errorf(
			"operations.%s custom check %s has unsupported collector type %q",
			operation,
			check.Key,
			collector.Type,
		)
	}
	return nil
}

func validatePublicFields(operation, key string, collector CustomCollector) error {
	if collector.ResultType != "object" && len(collector.PublicFields) != 0 {
		return fmt.Errorf("operations.%s custom check %s publicFields requires object resultType", operation, key)
	}
	if collector.ResultType == "object" && len(collector.PublicFields) == 0 {
		return fmt.Errorf("operations.%s custom check %s object resultType requires publicFields", operation, key)
	}
	if len(collector.PublicFields) > maximumPublicFields {
		return fmt.Errorf("operations.%s custom check %s publicFields exceeds %d", operation, key, maximumPublicFields)
	}
	seen := map[string]struct{}{}
	for _, field := range collector.PublicFields {
		if !publicFieldPattern.MatchString(field) {
			return fmt.Errorf("operations.%s custom check %s has invalid public field %q", operation, key, field)
		}
		if _, duplicate := seen[field]; duplicate {
			return fmt.Errorf("operations.%s custom check %s has duplicate public field %q", operation, key, field)
		}
		seen[field] = struct{}{}
	}
	return nil
}

func validResultType(value string) bool {
	switch value {
	case "bool", "integer", "string", "stringList", "integerList", "object":
		return true
	default:
		return false
	}
}

func reviewedPorts(name string) (map[string]bool, error) {
	knownPorts := map[string]bool{}
	switch name {
	case "init":
		for _, key := range []string{"kubernetes", "imageRepo", "chart", "yum", "console", "ntp"} {
			knownPorts[key] = true
		}
	case "clusterCreate":
		for _, key := range []string{
			"apiServer", "etcdClient", "etcdPeer", "etcdMetrics", "kubelet",
			"kubeletHealth", "controllerManager", "scheduler", "localHA", "bkeAgent",
		} {
			knownPorts[key] = true
		}
	default:
		return nil, fmt.Errorf("unsupported operation policy %q", name)
	}
	return knownPorts, nil
}

func validatePorts(name string, items []PortCheck, knownPorts map[string]bool, seen map[string]struct{}) error {
	for _, item := range items {
		if strings.TrimSpace(item.Key) == "" || strings.TrimSpace(item.Description) == "" {
			return fmt.Errorf("operations.%s.ports requires key and description", name)
		}
		if item.DefaultPort < minimumNetworkPort || item.DefaultPort > maximumNetworkPort {
			return fmt.Errorf("operations.%s port %s defaultPort is outside 1..65535", name, item.Key)
		}
		if !knownPorts[item.Key] {
			return fmt.Errorf("operations.%s port %s has no reviewed rule/source", name, item.Key)
		}
		if item.Protocol == "" {
			item.Protocol = "tcp"
		}
		if item.Protocol != "tcp" && item.Protocol != "udp" {
			return fmt.Errorf("operations.%s port %s protocol must be tcp or udp", name, item.Key)
		}
		if err := validatePortProjection(name, item, item.Protocol); err != nil {
			return err
		}
		duplicate, err := recordValidationKey(seen, "port:"+item.Key)
		if err != nil {
			return fmt.Errorf("operations.%s ports: %w", name, err)
		}
		if duplicate {
			return fmt.Errorf("operations.%s duplicate port key %q", name, item.Key)
		}
		if err := validateCommon(name, item.CheckCommon); err != nil {
			return err
		}
		if err := validateRoles(name, item.Key, item.Roles); err != nil {
			return err
		}
	}
	return nil
}

func validateResources(name string, items []ResourceCheck, seen map[string]struct{}) error {
	for _, item := range items {
		if item.Key == "" || item.Description == "" || item.Minimum < 0 {
			return fmt.Errorf("operations.%s resource requires key, description and non-negative minimum", name)
		}
		if item.Key != "cpu" && item.Key != "memory" && item.Key != "systemDisk" {
			return fmt.Errorf("operations.%s resource %s has no supported rule", name, item.Key)
		}
		duplicate, err := recordValidationKey(seen, "resource:"+item.Key)
		if err != nil {
			return fmt.Errorf("operations.%s resources: %w", name, err)
		}
		if duplicate {
			return fmt.Errorf("operations.%s duplicate resource key %q", name, item.Key)
		}
		if err := validateCommon(name, item.CheckCommon); err != nil {
			return err
		}
		if err := validateNonPortSelectors(name, "resource", item.Key, item.CheckCommon); err != nil {
			return err
		}
	}
	return nil
}

func validatePaths(name string, items []PathCheck, seen map[string]struct{}) error {
	for _, item := range items {
		if item.Key == "" || item.Description == "" {
			return fmt.Errorf("operations.%s path requires key and description", name)
		}
		if item.Key != "workspace" && name != "init" {
			return fmt.Errorf("operations.%s path %s has no supported rule", name, item.Key)
		}
		if item.Key != "workspace" {
			if !filepath.IsAbs(item.Path) || filepath.Clean(item.Path) != item.Path {
				return fmt.Errorf("operations.%s path %s requires path to be a clean absolute filesystem path", name, item.Key)
			}
			if len(item.Require) != 0 {
				return fmt.Errorf("operations.%s path %s does not support require; it checks path existence", name, item.Key)
			}
		}
		duplicate, err := recordValidationKey(seen, "path:"+item.Key)
		if err != nil {
			return fmt.Errorf("operations.%s paths: %w", name, err)
		}
		if duplicate {
			return fmt.Errorf("operations.%s duplicate path key %q", name, item.Key)
		}
		if err := validateCommon(name, item.CheckCommon); err != nil {
			return err
		}
		if err := validateNonPortSelectors(name, "path", item.Key, item.CheckCommon); err != nil {
			return err
		}
		for _, requirement := range item.Require {
			if requirement != "create" && requirement != "notReadOnly" {
				return fmt.Errorf("operations.%s path %s has unsupported requirement %q", name, item.Key, requirement)
			}
		}
	}
	return nil
}

func validateTools(name string, items []ToolCheck, seen map[string]struct{}) error {
	for _, item := range items {
		if item.Key == "" || item.Description == "" {
			return fmt.Errorf("operations.%s tool requires key and description", name)
		}
		if name != "init" {
			return fmt.Errorf("operations.%s tool %s has no supported rule", name, item.Key)
		}
		if len(item.Names) != 1 || strings.TrimSpace(item.Names[0]) == "" ||
			strings.ContainsAny(item.Names[0], `/\\`) {
			return fmt.Errorf("operations.%s tool %s requires one executable basename", name, item.Key)
		}
		duplicate, err := recordValidationKey(seen, "tool:"+item.Key)
		if err != nil {
			return fmt.Errorf("operations.%s tools: %w", name, err)
		}
		if duplicate {
			return fmt.Errorf("operations.%s duplicate tool key %q", name, item.Key)
		}
		if err := validateCommon(name, item.CheckCommon); err != nil {
			return err
		}
		if err := validateNonPortSelectors(name, "tool", item.Key, item.CheckCommon); err != nil {
			return err
		}
	}
	return nil
}

func validateNonPortSelectors(operation, kind, key string, common CheckCommon) error {
	if common.When != "" {
		return fmt.Errorf("operations.%s %s %s does not support when", operation, kind, key)
	}
	if len(common.Roles) != 0 {
		return fmt.Errorf("operations.%s %s %s does not support roles", operation, kind, key)
	}
	return nil
}

func recordValidationKey(seen map[string]struct{}, key string) (bool, error) {
	if seen == nil {
		return false, fmt.Errorf("validation key set is nil")
	}
	if _, duplicate := seen[key]; duplicate {
		return true, nil
	}
	seen[key] = struct{}{}
	return false, nil
}

func validateTopology(name string, topology TopologyPolicy) error {
	if name == "init" && (topology.Masters.Minimum != 0 || topology.Masters.Odd ||
		topology.Etcd.Minimum != 0 || topology.Workers.Minimum != 0 || len(topology.Unique) != 0) {
		return fmt.Errorf("operations.init topology has no supported rules")
	}
	if name == "clusterCreate" {
		for _, field := range topology.Unique {
			if field != "hostname" && field != "ip" {
				return fmt.Errorf("operations.clusterCreate topology unique field %q has no reviewed rule", field)
			}
		}
	}
	return nil
}

func validateManagementCluster(name string, policy ManagementClusterPolicy) error {
	if name == "init" {
		if policy.API || policy.Discovery || len(policy.CRDs) > 0 || policy.RequiredKubeconfig != "" {
			return fmt.Errorf("operations.init managementCluster has no supported rules")
		}
		return nil
	}
	for _, crd := range policy.CRDs {
		if crd != "bkeclusters" && crd != "bkenodes" {
			return fmt.Errorf("operations.clusterCreate managementCluster CRD %q is unsupported", crd)
		}
	}
	if policy.RequiredKubeconfig != "" && policy.RequiredKubeconfig != "kube-system/localkubeconfig" {
		return fmt.Errorf(
			"operations.clusterCreate requiredKubeconfig only supports kube-system/localkubeconfig",
		)
	}
	return nil
}

func validatePortProjection(operation string, item PortCheck, protocol string) error {
	expectedProtocol := "tcp"
	if operation == "init" && item.Key == "ntp" {
		expectedProtocol = "udp"
	}
	if protocol != expectedProtocol {
		return fmt.Errorf(
			"operations.%s port %s protocol must be %s for the supported collector",
			operation,
			item.Key,
			expectedProtocol,
		)
	}
	expectedPortFrom := reviewedPortFrom(operation, item.Key)
	if item.PortFrom != expectedPortFrom {
		return fmt.Errorf(
			"operations.%s port %s portFrom must be %q",
			operation,
			item.Key,
			expectedPortFrom,
		)
	}
	return nil
}

func reviewedPortFrom(operation, key string) string {
	if operation == "init" {
		return map[string]string{
			"kubernetes": "port.kubernetes", "imageRepo": "port.imageRepo",
			"chart": "port.chart", "yum": "port.yum", "console": "port.console",
		}[key]
	}
	if key == "apiServer" {
		return "cluster.apiPort"
	}
	return ""
}

func validateRoles(operation, key string, roles []string) error {
	for _, role := range roles {
		if role != "master" && role != "worker" && role != "etcd" && role != "all" {
			return fmt.Errorf("operations.%s check %s has unsupported role %q", operation, key, role)
		}
		if role == "all" && len(roles) != 1 {
			return fmt.Errorf("operations.%s check %s role all cannot be combined with other roles", operation, key)
		}
	}
	return nil
}

func validateCommon(operation string, c CheckCommon) error {
	if c.Failure != "" && !validFailure(c.Failure) {
		return fmt.Errorf("operations.%s has invalid failure policy", operation)
	}
	if c.Unknown != "" && !validUnknown(c.Unknown) {
		return fmt.Errorf("operations.%s has invalid unknown policy", operation)
	}
	if c.When != "" && !planKeyPattern.MatchString(c.When) {
		return fmt.Errorf("operations.%s when must be a typed plan key", operation)
	}
	return nil
}

var planKeyPattern = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_.-]*$`)

func validFailure(v string) bool { return v == "fail" || v == "warn" || v == "skip" }
func validUnknown(v string) bool { return v == "fail" || v == "warn" || v == "skip" }

// EffectivePorts resolves plan values without mutating either the plan or the
// configuration.  portFrom is a typed plan key; arbitrary expressions are not
// accepted by this package.
func EffectivePorts(c Config, op api.Operation, plan api.OperationPlan) ([]EffectivePort, error) {
	return EffectivePortsForPolicy(c.Operation(op), plan)
}

// EffectivePortsForPolicy resolves one operation policy against typed plan values.
func EffectivePortsForPolicy(policy OperationPolicy, plan api.OperationPlan) ([]EffectivePort, error) {
	out := make([]EffectivePort, 0, len(policy.Ports))
	for _, item := range policy.Ports {
		value, source, err := resolvePortValue(item, plan)
		if err != nil {
			return nil, err
		}
		protocol := item.Protocol
		if protocol == "" {
			protocol = "tcp"
		}
		out = append(out, EffectivePort{Key: item.Key, Port: value, Protocol: protocol, ValueSource: source})
	}
	return out, nil
}

func resolvePortValue(item PortCheck, plan api.OperationPlan) (int, string, error) {
	if item.PortFrom == "" {
		return item.DefaultPort, "config-default", nil
	}
	value, ok := plan.Values[item.PortFrom]
	if !ok {
		return item.DefaultPort, "config-default", nil
	}
	port, ok := integerPort(value.Value)
	if !ok {
		return 0, "", fmt.Errorf("portFrom %s is not integer", item.PortFrom)
	}
	if port < minimumNetworkPort || port > maximumNetworkPort {
		return 0, "", fmt.Errorf("portFrom %s is outside 1..65535", item.PortFrom)
	}
	return port, "plan", nil
}

func integerPort(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), true
	case float64:
		return int(typed), true
	default:
		return 0, false
	}
}

// ValidateConditions checks that every configured condition resolves to a bool plan value.
func ValidateConditions(policy OperationPolicy, plan api.OperationPlan) error {
	for _, item := range policy.Ports {
		if err := validateConditionValue(item.When, plan); err != nil {
			return err
		}
	}
	for _, item := range policy.CustomChecks {
		if err := validateConditionValue(item.When, plan); err != nil {
			return err
		}
	}
	return nil
}

func validateConditionValue(condition string, plan api.OperationPlan) error {
	if condition == "" {
		return nil
	}
	value, ok := plan.Values[condition]
	if !ok {
		return fmt.Errorf("condition %q is not present in the operation plan", condition)
	}
	if value.Type != api.ValueTypeBool {
		return fmt.Errorf("condition %q must be a bool plan value", condition)
	}
	if _, ok := value.Value.(bool); !ok {
		return fmt.Errorf("condition %q has a non-bool value", condition)
	}
	return nil
}
