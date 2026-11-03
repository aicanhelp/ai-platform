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

// Package preflightbundlecompiler applies the human-authored preflight policy
// onto the static rule catalog. It intentionally imports no bke
// command/provider packages so policy application has no logger, filesystem
// or cluster initialization side effects.
package preflightbundlecompiler

import (
	"fmt"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/adapterutil"
	"gopkg.openfuyao.cn/preflight/api"
	customcollector "gopkg.openfuyao.cn/preflight/collectors/custom"
)

// CustomRules compiles operation-scoped custom checks into CEL rules.
func CustomRules(config preflightconfig.Config) []api.RuleTemplate {
	var rules []api.RuleTemplate
	for _, operation := range []api.Operation{api.OperationInit, api.OperationClusterCreate} {
		for _, check := range config.Operation(operation).CustomChecks {
			rule := api.RuleTemplate{
				ID:   preflightconfig.CheckID(operation, "custom", check.Key),
				Name: check.Description, Operation: operation,
				Fact: customcollector.FactKey(operation, check.Key),
				Op:   api.OpCEL, Expected: true, ExpectedText: check.Expected,
				Expression: check.Expression, Level: api.SeverityError,
				Unknown: api.UnknownBlock, Fix: check.Remediation,
			}
			if operation == api.OperationClusterCreate {
				rule.DependsOn = []string{"CLUSTER-SSH-PRIVILEGE"}
			}
			applyCommon(&rule, check.CheckCommon)
			rules = append(rules, rule)
		}
	}
	return rules
}

// HasCustomChecks reports whether a config contributes runtime rules.
func HasCustomChecks(config preflightconfig.Config) bool {
	return len(config.Operations.Init.CustomChecks) != 0 || len(config.Operations.ClusterCreate.CustomChecks) != 0
}

// HasExtensionChecks reports whether a config contributes tool/path rules
// beyond the reviewed built-in keys.
func HasExtensionChecks(config preflightconfig.Config) bool {
	return len(ExtensionRules(config)) != 0
}

// ExtensionRules compiles operator-added tool and path checks into rules.
// Built-in keys (tar, systemctl, workspace) are applied to the reviewed
// catalog rules instead; everything else becomes a config-sourced rule whose
// ID is derived from the policy key. Only the init policy carries tool/path
// checks: validation rejects them for other operations, and the bridge only
// binds extension checks for init.
func ExtensionRules(config preflightconfig.Config) []api.RuleTemplate {
	var rules []api.RuleTemplate
	policy := config.Operation(api.OperationInit)
	for _, tool := range policy.Tools {
		if tool.Key == "tar" || tool.Key == "systemctl" {
			continue
		}
		rule := api.RuleTemplate{
			ID:   preflightconfig.CheckID(api.OperationInit, "tool", tool.Key),
			Name: tool.Description, Operation: api.OperationInit,
			Fact: "tool." + tool.Names[0], Op: api.OpEq, Expected: true,
			Level: api.SeverityError, Unknown: api.UnknownBlock,
			Fix: fmt.Sprintf("install %s and ensure it is executable on PATH", tool.Names[0]),
		}
		applyCommon(&rule, tool.CheckCommon)
		rules = append(rules, rule)
	}
	for _, path := range policy.Paths {
		if path.Key == "workspace" {
			continue
		}
		rule := api.RuleTemplate{
			ID:   preflightconfig.CheckID(api.OperationInit, "path", path.Key),
			Name: path.Description, Operation: api.OperationInit,
			Fact: "path." + path.Path, Op: api.OpCEL,
			Expected: true, ExpectedText: "path exists",
			Expression: "actual.exists == true",
			Level:      api.SeverityError, Unknown: api.UnknownBlock,
			Fix: fmt.Sprintf("create the required path %s", path.Path),
		}
		applyCommon(&rule, path.CheckCommon)
		rules = append(rules, rule)
	}
	return rules
}

// ApplyPolicy applies the human policy's reviewed thresholds and presentation
// metadata to a copy of the static catalog.
func ApplyPolicy(config preflightconfig.Config, rules []api.RuleTemplate) {
	for i := range rules {
		policy := config.Operation(rules[i].Operation)
		applyResourcePolicy(policy, &rules[i])
		applyPortPolicy(policy, &rules[i])
		applyPlatformPolicy(policy, &rules[i])
	}
}

// ApplyRuntimePolicy applies the release policy fields plus runtime-only
// presentation and topology values. Keeping this separate preserves the
// deterministic release Bundle while allowing an explicit external policy to
// adjust existing checks in memory.
func ApplyRuntimePolicy(config preflightconfig.Config, rules []api.RuleTemplate) {
	ApplyPolicy(config, rules)
	for i := range rules {
		policy := config.Operation(rules[i].Operation)
		applyToolPolicy(policy, &rules[i])
		applyPathPolicy(policy, &rules[i])
		applyTopologyPolicy(policy, &rules[i])
	}
}

func applyResourcePolicy(policy preflightconfig.OperationPolicy, rule *api.RuleTemplate) {
	for _, resource := range policy.Resources {
		if !matchesResource(*rule, resource.Key) {
			continue
		}
		rule.Name = resource.Description
		rule.Expected = int64(resource.Minimum)
		// Keep the remediation aligned with the effective threshold; an
		// operator-adjusted minimum must not leave the reviewed default fix.
		if dynamic := resourceRuleFix(resource.Key, rule.Expected); dynamic != "" {
			rule.Fix = dynamic
		}
		applyCommon(rule, resource.CheckCommon)
		return
	}
}

// resourceRuleFix renders a resource remediation from the actual expected
// value. It covers both INIT- and CLUSTER- prefixed resource rules.
func resourceRuleFix(key string, expected any) string {
	switch key {
	case "cpu":
		if cpus, ok := expected.(int); ok && cpus > 0 {
			return fmt.Sprintf("use a host with at least %d logical CPUs", cpus)
		}
		// applyResourcePolicy overwrites Expected with int64 on every
		// resource rule, so accept the overwritten type here as well.
		if cpus, ok := expected.(int64); ok && cpus > 0 {
			return fmt.Sprintf("use a host with at least %d logical CPUs", cpus)
		}
	case "memory":
		if bytes, ok := expected.(int64); ok && bytes > 0 {
			return fmt.Sprintf("use a host with at least %s memory", adapterutil.FormatGiB(bytes))
		}
	case "systemDisk":
		if bytes, ok := expected.(int64); ok && bytes > 0 {
			return fmt.Sprintf("use a host with at least %s system disk capacity", adapterutil.FormatGiB(bytes))
		}
	default:
	}
	return ""
}

func applyPortPolicy(policy preflightconfig.OperationPolicy, rule *api.RuleTemplate) {
	for _, port := range policy.Ports {
		if !bindingMatchesPort(rule.ID, port.Key, rule.Operation) {
			continue
		}
		rule.Name = port.Description
		applyCommon(rule, port.CheckCommon)
		if port.Remediation == "" {
			rule.Fix = fmt.Sprintf(
				"Make sure port %d is available and rerun preflight.",
				port.DefaultPort,
			)
		}
		return
	}
}

func applyPlatformPolicy(policy preflightconfig.OperationPolicy, rule *api.RuleTemplate) {
	if len(policy.Platform.Distributions) == 0 ||
		(rule.ID != "INIT-PLAT-DISTRO" && rule.ID != "CLUSTER-PLAT-DISTRO") {
		return
	}
	expected := make([]any, 0, len(policy.Platform.Distributions))
	for _, platform := range policy.Platform.Distributions {
		expected = append(expected, []any{platform.Name, platform.Version})
	}
	rule.Expected = expected
}

func applyToolPolicy(policy preflightconfig.OperationPolicy, rule *api.RuleTemplate) {
	for _, tool := range policy.Tools {
		if !bindingMatchesTool(rule.ID, tool.Key) {
			continue
		}
		rule.Name = tool.Description
		applyCommon(rule, tool.CheckCommon)
		return
	}
}

func applyPathPolicy(policy preflightconfig.OperationPolicy, rule *api.RuleTemplate) {
	for _, path := range policy.Paths {
		if !bindingMatchesPath(rule.ID, path.Key) {
			continue
		}
		rule.Name = path.Description
		applyCommon(rule, path.CheckCommon)
		return
	}
}

func applyTopologyPolicy(policy preflightconfig.OperationPolicy, rule *api.RuleTemplate) {
	switch rule.ID {
	case "CLUSTER-TOPO-MASTER":
		rule.Expected = policy.Topology.Masters.Minimum
	case "CLUSTER-TOPO-ETCD":
		rule.Expected = policy.Topology.Etcd.Minimum
	case "CLUSTER-TOPO-WORKER":
		rule.Expected = policy.Topology.Workers.Minimum
	default:
		return
	}
}

func applyCommon(rule *api.RuleTemplate, common preflightconfig.CheckCommon) {
	applyFailurePolicy(rule, common.Failure)
	applyUnknownPolicy(rule, common.Unknown)
	if common.Remediation != "" {
		rule.Fix = common.Remediation
	}
}

func applyFailurePolicy(rule *api.RuleTemplate, policy string) {
	switch policy {
	case "warn":
		rule.Level = api.SeverityWarning
	case "fail":
		rule.Level = api.SeverityError
	case "skip":
		rule.Level = api.SeverityInfo
	default:
		return
	}
}

func applyUnknownPolicy(rule *api.RuleTemplate, policy string) {
	switch policy {
	case "warn":
		rule.Unknown = api.UnknownWarn
	case "skip":
		rule.Unknown = api.UnknownIgnore
	case "fail":
		rule.Unknown = api.UnknownBlock
	default:
		return
	}
}

func matchesResource(rule api.RuleTemplate, key string) bool {
	switch key {
	case "cpu":
		return rule.ID == "INIT-RES-CPU" || rule.ID == "CLUSTER-RES-CPU"
	case "memory":
		return rule.ID == "INIT-RES-MEMORY" || rule.ID == "CLUSTER-RES-MEMORY"
	case "systemDisk":
		return rule.ID == "INIT-RES-SYSTEM-DISK" || rule.ID == "CLUSTER-RES-SYSTEM-DISK"
	default:
		return false
	}
}

func bindingMatchesTool(ruleID, key string) bool {
	return (ruleID == "INIT-TOOL-TAR" && key == "tar") ||
		(ruleID == "INIT-TOOL-SYSTEMCTL" && key == "systemctl")
}

func bindingMatchesPath(ruleID, key string) bool {
	return key == "workspace" && (ruleID == "INIT-PATH-CREATE" || ruleID == "INIT-PATH-MOUNT")
}

func bindingMatchesPort(ruleID, key string, operation api.Operation) bool {
	if operation != api.OperationInit {
		return bindingMatchesClusterPort(ruleID, key)
	}
	switch key {
	case "kubernetes":
		return ruleID == "INIT-PORT-K8S"
	case "imageRepo":
		return ruleID == "INIT-PORT-REGISTRY"
	case "chart":
		return ruleID == "INIT-PORT-CHART"
	case "yum":
		return ruleID == "INIT-PORT-HTTP"
	case "console":
		return ruleID == "INIT-PORT-CONSOLE"
	case "ntp":
		return ruleID == "INIT-PORT-NTP"
	default:
		return false
	}
}

func bindingMatchesClusterPort(ruleID, key string) bool {
	switch key {
	case "apiServer":
		return ruleID == "CLUSTER-PORT-APISERVER"
	case "etcdClient":
		return ruleID == "CLUSTER-PORT-ETCD-CLIENT"
	case "etcdPeer":
		return ruleID == "CLUSTER-PORT-ETCD-PEER"
	case "etcdMetrics":
		return ruleID == "CLUSTER-PORT-ETCD-METRICS"
	case "kubelet":
		return ruleID == "CLUSTER-PORT-KUBELET-SECURE"
	case "kubeletHealth":
		return ruleID == "CLUSTER-PORT-KUBELET-HEALTH"
	case "controllerManager":
		return ruleID == "CLUSTER-PORT-CONTROLLER"
	case "scheduler":
		return ruleID == "CLUSTER-PORT-SCHEDULER"
	case "localHA":
		return ruleID == "CLUSTER-PORT-LOCAL-HA"
	case "bkeAgent":
		return ruleID == "CLUSTER-PORT-BKEAGENT"
	default:
		return false
	}
}
