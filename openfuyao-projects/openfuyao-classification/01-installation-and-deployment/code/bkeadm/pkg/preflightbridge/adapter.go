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

package preflightbridge

import (
	"context"
	"fmt"
	"net"
	"path/filepath"
	"strings"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	adapterutil "gopkg.openfuyao.cn/preflight/adapter/openfuyao/adapterutil"
	clusteradapter "gopkg.openfuyao.cn/preflight/adapter/openfuyao/clustercreate"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
	customcollector "gopkg.openfuyao.cn/preflight/collectors/custom"
	"gopkg.openfuyao.cn/preflight/normalize"
)

const (
	minimumPathFactParts = 4
	pathKindOffset       = 2
	oddDivisor           = 2
	oddRemainder         = 1
)

// OpenFuyaoAdapter is the BKE host adapter. SDK core receives only its generic
// ResolveResult output.
type OpenFuyaoAdapter struct {
	Provider ProviderResolver
	Policy   *preflightconfig.Config
}

// NewOpenFuyaoAdapter creates the BKE adapter with an optional reviewed policy.
func NewOpenFuyaoAdapter(provider ProviderResolver, policies ...preflightconfig.Config) *OpenFuyaoAdapter {
	a := &OpenFuyaoAdapter{Provider: provider}
	if len(policies) > 0 {
		a.Policy = &policies[0]
	}
	return a
}

// Resolve converts one BKE operation input into the SDK operation plan and bindings.
func (a *OpenFuyaoAdapter) Resolve(ctx context.Context, in AdapterInput) (api.ResolveResult, error) {
	switch in.Operation {
	case api.OperationInit:
		return resolveInitWithPolicy(in.Init, a.Policy), nil
	case api.OperationClusterCreate:
		return a.resolveClusterCreate(ctx, in.ClusterCreate)
	default:
		return api.ResolveResult{Issues: []api.RunIssue{{
			Code: "INPUT-UNSUPPORTED-OPERATION", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
			Cause:  fmt.Sprintf("operation %q is not supported", in.Operation),
			Action: "use init or cluster-create, then run preflight again",
		}}}, nil
	}
}

func resolveInit(in *initadapter.Input) api.ResolveResult {
	result := resolveInitRaw(in)
	decorateBindingNames(&result)
	return result
}

func resolveInitRaw(in *initadapter.Input) api.ResolveResult {
	plan, issues := initadapter.Resolve(in)
	result := api.ResolveResult{Plan: plan, Issues: issues}
	if api.HasStartBlocker(issues) {
		return result
	}
	subjects := subjectsByID(plan.Subjects)
	for _, instance := range initadapter.Instances(plan) {
		subject := subjects[instance.Subject]
		result.Bindings = append(result.Bindings, api.RuleBinding{
			TemplateID: instance.Rule.ID, InstanceID: instance.InstanceID,
			Name:    instance.Rule.Name,
			Fix:     instance.Rule.Fix,
			Subject: subject, Fact: instance.FactKey, Applicable: instance.Applicable,
		})
	}
	result.Facts = initPlanFacts(plan)
	return result
}

func resolveInitWithPolicy(in *initadapter.Input, policy *preflightconfig.Config) api.ResolveResult {
	if in == nil || policy == nil {
		return resolveInit(in)
	}
	resolvedInput := *in
	applyInitPortDefaults(&resolvedInput, policy.Operations.Init.Ports)
	result := resolveInitRaw(&resolvedInput)
	if err := preflightconfig.ValidateConditions(policy.Operations.Init, result.Plan); err != nil {
		result.Issues = append(result.Issues, api.RunIssue{Code: "INIT-CONFIG-CONDITION", Severity: api.IssueError,
			Stage: api.IssueStagePlan, Effect: api.IssueBlockStart, Cause: err.Error(),
			Action: "correct the preflight condition, then run preflight again"})
		return result
	}
	applyInitNTPPort(&result, policy.Operations.Init.Ports)
	if err := refreshPlanHash(&result); err != nil {
		result.Issues = append(result.Issues, api.RunIssue{
			Code: "INIT-CONFIG-PLAN", Severity: api.IssueError,
			Stage: api.IssueStagePlan, Effect: api.IssueBlockStart, Cause: err.Error(),
			Action: "correct the preflight policy, then run preflight again",
		})
		return result
	}
	applyPolicyBindings(&result, policy.Operations.Init)
	decorateBindingNames(&result)
	return result
}

func refreshPlanHash(result *api.ResolveResult) error {
	hash, err := normalize.ComputePlanHash(&result.Plan)
	if err != nil {
		return fmt.Errorf("compute policy-adjusted plan hash: %w", err)
	}
	result.Plan.PlanHash = hash
	return nil
}

func applyInitPortDefaults(input *initadapter.Input, ports []preflightconfig.PortCheck) {
	for _, item := range ports {
		if item.DefaultPort == 0 {
			continue
		}
		target := initPortTarget(input, item.Key)
		if target != nil && !input.ExplicitPorts[item.Key] {
			*target = item.DefaultPort
		}
	}
}

func initPortTarget(input *initadapter.Input, key string) *int {
	switch key {
	case "kubernetes":
		return &input.KubernetesPort
	case "imageRepo":
		return &input.ImageRepoPort
	case "chart":
		return &input.ChartRepoPort
	case "yum":
		return &input.YumRepoPort
	case "console":
		return &input.ConsolePort
	default:
		return nil
	}
}

func applyInitNTPPort(result *api.ResolveResult, ports []preflightconfig.PortCheck) {
	for _, item := range ports {
		if item.Key != "ntp" || item.DefaultPort <= 0 {
			continue
		}
		value, applicable := result.Plan.Values["port.ntp"]
		if !applicable || value.Value == nil {
			return
		}
		result.Plan.Values["port.ntp"] = api.Value{Key: "port.ntp", Type: api.ValueTypeInteger, Value: item.DefaultPort}
		for i := range result.Bindings {
			if result.Bindings[i].TemplateID == "INIT-PORT-NTP" {
				result.Bindings[i].Fact = fmt.Sprintf("port.%d.udp.free", item.DefaultPort)
			}
		}
		return
	}
}

func (a *OpenFuyaoAdapter) resolveClusterCreate(
	ctx context.Context,
	in *ClusterCreateInput,
) (api.ResolveResult, error) {
	if in == nil {
		return api.ResolveResult{Issues: []api.RunIssue{{
			Code: "CLUSTER-INPUT-REQUIRED", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
			Cause: "cluster-create input is required", Action: "provide cluster-create input, then run preflight again",
		}}}, nil
	}
	if a.Provider == nil {
		return api.ResolveResult{}, fmt.Errorf("provider resolver is not configured")
	}
	decoded, field, err := decodeClusterCreateInput(in)
	if err != nil {
		return clusterInputError(field, err), nil
	}
	resolved, err := a.Provider.Resolve(ctx, decoded.cluster, decoded.nodes)
	if err != nil {
		return api.ResolveResult{}, fmt.Errorf("resolve Provider plan: %w", err)
	}
	result := api.ResolveResult{Issues: providerResolveIssues(resolved.Issues)}
	if api.HasStartBlocker(result.Issues) {
		return result, nil
	}
	plan, issues := buildClusterPlan(in, decoded, resolved)
	result.Plan = plan
	result.Issues = append(result.Issues, issues...)
	if api.HasStartBlocker(result.Issues) {
		return result, nil
	}
	if a.Policy != nil {
		if err := preflightconfig.ValidateConditions(a.Policy.Operations.ClusterCreate, plan); err != nil {
			result.Issues = append(
				result.Issues,
				api.RunIssue{Code: "CLUSTER-CONFIG-CONDITION", Severity: api.IssueError,
					Stage: api.IssueStagePlan, Effect: api.IssueBlockStart, Cause: err.Error(),
					Action: "correct the preflight condition, then run preflight again"},
			)
			return result, nil
		}
	}
	result.Bindings = clusterBindings(plan, clusterPortDefaults(a.Policy))
	suppressDuplicateClusterValidation(&result, resolved.ValidationFields)
	if a.Policy != nil {
		applyPolicyBindings(&result, a.Policy.Operations.ClusterCreate)
	}
	decorateBindingNames(&result)
	result.Facts = clusterPlanFacts(decoded.nodes, resolved)
	return result, nil
}

func suppressDuplicateClusterValidation(
	result *api.ResolveResult,
	findings map[string]ValidationFinding,
) {
	finding, exists := findings["local.cluster.valid"]
	if !exists || !hasDedicatedClusterFieldRule(finding.Field) {
		return
	}
	for i := range result.Bindings {
		if result.Bindings[i].TemplateID == "CLUSTER-MGMT-LOCAL-CLUSTER" {
			result.Bindings[i].Applicable = false
			return
		}
	}
}

func hasDedicatedClusterFieldRule(field string) bool {
	switch field {
	case "cluster.spec.clusterConfig.cluster.networking.podSubnet",
		"cluster.spec.clusterConfig.cluster.networking.serviceSubnet":
		return true
	default:
		return false
	}
}

type decodedClusterCreate struct {
	cluster *configv1beta1.BKECluster
	nodes   []configv1beta1.BKENode
}

func decodeClusterCreateInput(input *ClusterCreateInput) (decodedClusterCreate, string, error) {
	decoded := decodedClusterCreate{cluster: &configv1beta1.BKECluster{}}
	if err := yaml2.UnmarshalStrict(input.ClusterYAML, decoded.cluster); err != nil {
		return decodedClusterCreate{}, "cluster", err
	}
	if err := yaml2.UnmarshalStrict(input.NodesYAML, &decoded.nodes); err == nil {
		return decoded, "", nil
	}
	var list configv1beta1.BKENodeList
	if err := yaml2.UnmarshalStrict(input.NodesYAML, &list); err != nil {
		return decodedClusterCreate{}, "nodes", err
	}
	decoded.nodes = list.Items
	return decoded, "", nil
}

func providerResolveIssues(messages []string) []api.RunIssue {
	issues := make([]api.RunIssue, 0, len(messages))
	for _, message := range messages {
		issues = append(issues, api.RunIssue{
			Code: "CLUSTER-PROVIDER-RESOLVE", Severity: api.IssueError,
			Stage: api.IssueStagePlan, Effect: api.IssueBlockStart,
			Cause: message, Action: "correct the cluster topology and Provider inputs, then run preflight again",
		})
	}
	return issues
}

func buildClusterPlan(
	input *ClusterCreateInput,
	decoded decodedClusterCreate,
	resolved ResolveResult,
) (api.OperationPlan, []api.RunIssue) {
	return clusteradapter.Resolve(&clusteradapter.ResolvedCluster{
		APIBindPort: resolved.APIBindPort, MasterCount: resolved.MasterCount,
		EtcdCount: resolved.EtcdCount, WorkerCount: resolved.WorkerCount,
		BootstrapNodeID: resolved.BootstrapNodeID, EndpointExplicit: resolved.EndpointExplicit,
		PodSubnet: resolved.PodSubnet, ServiceSubnet: resolved.ServiceSubnet,
		Runtime: resolved.Runtime, EtcdDataDir: resolved.EtcdDataDir,
		KubeletRoot: resolved.KubeletRoot, DataRoot: resolved.DataRoot,
		NodeRoles: resolved.NodeRoles, NodeAddresses: resolved.NodeAddresses,
		NodeMetadata: resolved.NodeMetadata, Sources: resolved.Sources, Workspace: input.Workspace,
		OutputNames: clusterOutputNames(decoded.cluster, decoded.nodes),
	})
}

func clusterPortDefaults(policy *preflightconfig.Config) map[string]int {
	defaults := map[string]int{}
	if policy == nil {
		return defaults
	}
	for _, port := range policy.Operations.ClusterCreate.Ports {
		defaults[port.Key] = port.DefaultPort
	}
	return defaults
}

func clusterBindings(plan api.OperationPlan, portDefaults map[string]int) []api.RuleBinding {
	subjects := subjectsByID(plan.Subjects)
	instances := clusteradapter.InstancesWithPorts(plan, portDefaults)
	bindings := make([]api.RuleBinding, 0, len(instances))
	for _, instance := range instances {
		subject := subjects[instance.Subject]
		if subject.ID == "" {
			subject = api.Subject{ID: instance.Subject, Role: []string{"cluster"}}
		}
		bindings = append(bindings, api.RuleBinding{
			TemplateID: instance.Rule.ID, InstanceID: instance.InstanceID,
			Name: instance.Rule.Name, Fix: instance.Rule.Fix, Subject: subject,
			Fact: instance.FactKey, Applicable: instance.Applicable,
		})
	}
	return bindings
}

// applyPolicyBindings applies only declarative applicability and role gates.
// The adapter still owns schema/business validation; the policy cannot add a
// shell expression or replace a fact collector.
func applyPolicyBindings(result *api.ResolveResult, policy preflightconfig.OperationPolicy) {
	for i := range result.Bindings {
		binding := &result.Bindings[i]
		applyPortBindingPolicy(binding, result.Plan, policy.Ports)
		if result.Plan.Operation == api.OperationInit {
			applyInitBindingPolicy(binding, policy)
			continue
		}
		applyClusterBindingPolicy(binding, policy)
	}
	appendCustomBindings(result, policy)
	appendExtensionBindings(result, policy)
}

func appendCustomBindings(result *api.ResolveResult, policy preflightconfig.OperationPolicy) {
	for _, check := range policy.CustomChecks {
		if check.When != "" && !planBool(result.Plan, check.When) {
			continue
		}
		ruleID := preflightconfig.CheckID(result.Plan.Operation, "custom", check.Key)
		factKey := customcollector.FactKey(result.Plan.Operation, check.Key)
		for _, subject := range customSubjects(result.Plan) {
			if len(check.Roles) != 0 && !subjectHasRole(subject, check.Roles) {
				continue
			}
			result.Bindings = append(result.Bindings, api.RuleBinding{
				TemplateID: ruleID, InstanceID: ruleID + "@" + subject.ID,
				Name: check.Description, Fix: check.Remediation,
				Subject: subject, Fact: factKey, Applicable: true,
			})
		}
	}
}

func customSubjects(plan api.OperationPlan) []api.Subject {
	if plan.Operation == api.OperationInit {
		for _, subject := range plan.Subjects {
			if subject.ID == "bootstrap" {
				return []api.Subject{subject}
			}
		}
		return nil
	}
	out := make([]api.Subject, 0, len(plan.Subjects))
	for _, subject := range plan.Subjects {
		if subjectHasRole(subject, []string{"master", "worker", "etcd"}) {
			out = append(out, subject)
		}
	}
	return out
}

// appendExtensionBindings binds operator-added tool and path checks. Their
// rules are compiled into the runtime Bundle by ExtensionRules; the fact keys
// must stay in sync with those rules.
func appendExtensionBindings(result *api.ResolveResult, policy preflightconfig.OperationPolicy) {
	if result.Plan.Operation != api.OperationInit {
		return
	}
	var bootstrap *api.Subject
	for i := range result.Plan.Subjects {
		if result.Plan.Subjects[i].ID == "bootstrap" {
			bootstrap = &result.Plan.Subjects[i]
			break
		}
	}
	if bootstrap == nil {
		return
	}
	appendToolBindings(result, policy.Tools, *bootstrap)
	appendPathBindings(result, policy.Paths, *bootstrap)
}

func appendToolBindings(result *api.ResolveResult, tools []preflightconfig.ToolCheck, subject api.Subject) {
	for _, tool := range tools {
		if tool.Key == "tar" || tool.Key == "systemctl" || len(tool.Names) == 0 {
			continue
		}
		ruleID := preflightconfig.CheckID(api.OperationInit, "tool", tool.Key)
		result.Bindings = append(result.Bindings, api.RuleBinding{
			TemplateID: ruleID, InstanceID: ruleID + "@" + subject.ID,
			Name: tool.Description, Fix: tool.Remediation,
			Subject: subject, Fact: "tool." + tool.Names[0], Applicable: true,
		})
	}
}

func appendPathBindings(result *api.ResolveResult, paths []preflightconfig.PathCheck, subject api.Subject) {
	for _, path := range paths {
		if path.Key == "workspace" || path.Path == "" {
			continue
		}
		ruleID := preflightconfig.CheckID(api.OperationInit, "path", path.Key)
		result.Bindings = append(result.Bindings, api.RuleBinding{
			TemplateID: ruleID, InstanceID: ruleID + "@" + subject.ID,
			Name: path.Description, Fix: path.Remediation,
			Subject: subject, Fact: "path." + path.Path, Applicable: true,
		})
	}
}

func applyPortBindingPolicy(binding *api.RuleBinding, plan api.OperationPlan, ports []preflightconfig.PortCheck) {
	for _, port := range ports {
		if !bindingMatchesPort(binding.TemplateID, port.Key, plan.Operation) {
			continue
		}
		if port.When != "" && !planBool(plan, port.When) {
			binding.Applicable = false
		}
		if len(port.Roles) > 0 && !subjectHasRole(binding.Subject, port.Roles) {
			binding.Applicable = false
		}
		applyBindingPresentation(binding, port.Description, port.Remediation)
		return
	}
	if isPolicyPortRule(binding.TemplateID, plan.Operation) {
		binding.Applicable = false
	}
}

func applyInitBindingPolicy(binding *api.RuleBinding, policy preflightconfig.OperationPolicy) {
	switch binding.TemplateID {
	case "INIT-PLAT-DISTRO", "INIT-PLAT-KERNEL", "INIT-PLAT-ARCH":
		binding.Applicable = binding.Applicable && len(policy.Platform.Distributions) > 0
	case "INIT-RES-CPU":
		applyResourceBindingPolicy(binding, policy.Resources, "cpu")
	case "INIT-RES-MEMORY":
		applyResourceBindingPolicy(binding, policy.Resources, "memory")
	case "INIT-RES-SYSTEM-DISK":
		applyResourceBindingPolicy(binding, policy.Resources, "systemDisk")
	case "INIT-TOOL-TAR":
		applyToolBindingPolicy(binding, policy.Tools, "tar")
	case "INIT-TOOL-SYSTEMCTL":
		applyToolBindingPolicy(binding, policy.Tools, "systemctl")
	case "INIT-PATH-CREATE":
		applyPathBindingPolicy(binding, policy.Paths, "workspace", "create")
	case "INIT-PATH-MOUNT":
		applyPathBindingPolicy(binding, policy.Paths, "workspace", "notReadOnly")
	default:
		return
	}
}

func applyClusterBindingPolicy(binding *api.RuleBinding, policy preflightconfig.OperationPolicy) {
	switch binding.TemplateID {
	case "CLUSTER-PLAT-DISTRO", "CLUSTER-PLAT-KERNEL":
		binding.Applicable = binding.Applicable && len(policy.Platform.Distributions) > 0
	case "CLUSTER-RES-CPU":
		applyResourceBindingPolicy(binding, policy.Resources, "cpu")
	case "CLUSTER-RES-MEMORY":
		applyResourceBindingPolicy(binding, policy.Resources, "memory")
	case "CLUSTER-RES-SYSTEM-DISK":
		applyResourceBindingPolicy(binding, policy.Resources, "systemDisk")
	case "CLUSTER-UNIQUE-IP":
		binding.Applicable = binding.Applicable && containsPolicyValue(policy.Topology.Unique, "ip")
	case "CLUSTER-UNIQUE-HOSTNAME":
		binding.Applicable = binding.Applicable && containsPolicyValue(policy.Topology.Unique, "hostname")
	case "CLUSTER-TOPO-MASTER-ODD":
		binding.Applicable = binding.Applicable && policy.Topology.Masters.Odd
	default:
		applyManagementBindingPolicy(binding, policy.ManagementCluster)
	}
}

func applyManagementBindingPolicy(binding *api.RuleBinding, policy preflightconfig.ManagementClusterPolicy) {
	switch binding.TemplateID {
	case "CLUSTER-MGMT-API":
		binding.Applicable = binding.Applicable && policy.API
	case "CLUSTER-MGMT-DISCOVERY":
		binding.Applicable = binding.Applicable && policy.API && policy.Discovery
	case "CLUSTER-MGMT-CRD-CLUSTER":
		binding.Applicable = binding.Applicable && policy.Discovery && containsPolicyValue(policy.CRDs, "bkeclusters")
	case "CLUSTER-MGMT-CRD-NODE":
		binding.Applicable = binding.Applicable && policy.Discovery && containsPolicyValue(policy.CRDs, "bkenodes")
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-GET", "CLUSTER-MGMT-AGENT-KUBECONFIG-DATA",
		"CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE", "CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE":
		binding.Applicable = binding.Applicable && policy.API && policy.RequiredKubeconfig != ""
	default:
		return
	}
}

func applyResourceBindingPolicy(
	binding *api.RuleBinding,
	resources []preflightconfig.ResourceCheck,
	key string,
) {
	for _, resource := range resources {
		if resource.Key == key {
			applyBindingPresentation(binding, resource.Description, resource.Remediation)
			applyResourceBindingFix(binding, resource)
			return
		}
	}
	binding.Applicable = false
}

// applyResourceBindingFix re-renders the binding Fix from a policy-adjusted
// minimum. Bindings override the Bundle template's Fix, so without this the
// runtime report keeps the adapter's hardcoded default (e.g. 100 GiB disk).
// An operator-provided remediation always wins.
func applyResourceBindingFix(binding *api.RuleBinding, resource preflightconfig.ResourceCheck) {
	if resource.Remediation != "" {
		return
	}
	if dynamic := resourceBindingFix(binding.TemplateID, resource.Key, int64(resource.Minimum)); dynamic != "" {
		binding.Fix = dynamic
	}
}

// resourceBindingFix renders the resource remediation from the effective policy
// minimum, mirroring the compiler's resourceRuleFix on the runtime binding path.
func resourceBindingFix(templateID, key string, minimum int64) string {
	if minimum <= 0 {
		return ""
	}
	subject := "host"
	if strings.HasPrefix(templateID, "CLUSTER-") {
		subject = "node"
	}
	switch key {
	case "cpu":
		return fmt.Sprintf("use a %s with at least %d logical CPUs", subject, minimum)
	case "memory":
		return fmt.Sprintf("use a %s with at least %s memory", subject, adapterutil.FormatGiB(minimum))
	case "systemDisk":
		return fmt.Sprintf("use a %s with at least %s system disk capacity", subject, adapterutil.FormatGiB(minimum))
	default:
		return ""
	}
}

func applyToolBindingPolicy(binding *api.RuleBinding, tools []preflightconfig.ToolCheck, key string) {
	for _, tool := range tools {
		if tool.Key == key {
			if len(tool.Names) == 1 {
				binding.Fact = "tool." + tool.Names[0]
			}
			applyBindingPresentation(binding, tool.Description, tool.Remediation)
			return
		}
	}
	binding.Applicable = false
}

func applyPathBindingPolicy(
	binding *api.RuleBinding,
	paths []preflightconfig.PathCheck,
	key string,
	requirement string,
) {
	for _, path := range paths {
		if path.Key == key {
			binding.Applicable = binding.Applicable && containsPolicyValue(path.Require, requirement)
			applyBindingPresentation(binding, path.Description, path.Remediation)
			return
		}
	}
	binding.Applicable = false
}

func applyBindingPresentation(binding *api.RuleBinding, name, remediation string) {
	if name != "" {
		binding.Name = name
	}
	if remediation != "" {
		binding.Fix = remediation
	}
}

func containsPolicyValue(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func isPolicyPortRule(ruleID string, operation api.Operation) bool {
	keys := []string{"kubernetes", "imageRepo", "chart", "yum", "console", "ntp"}
	if operation == api.OperationClusterCreate {
		keys = []string{
			"apiServer", "etcdClient", "etcdPeer", "etcdMetrics", "kubelet",
			"kubeletHealth", "controllerManager", "scheduler", "localHA", "bkeAgent",
		}
	}
	for _, key := range keys {
		if bindingMatchesPort(ruleID, key, operation) {
			return true
		}
	}
	return false
}

// decorateBindingNames adds the concrete object examined by a check to its
// existing operator-facing name. Keeping the target in Name preserves the
// stable public report schema while making PASS, FAIL and WARN equally
// actionable in text and JSON output.
func decorateBindingNames(result *api.ResolveResult) {
	if result == nil {
		return
	}
	for i := range result.Bindings {
		binding := &result.Bindings[i]
		name := binding.Name
		if name == "" {
			name = binding.TemplateID
		}
		target := targetForBinding(binding.TemplateID, binding.Fact, binding.Subject, result.Plan)
		if target == "" {
			binding.Name = name
			continue
		}
		// The adapter is called at most once for a normal run. The guard keeps
		// policy re-resolution and tests from appending the same target twice.
		suffix := " [" + target + "]"
		if !strings.HasSuffix(name, suffix) {
			name += suffix
		}
		binding.Name = name
	}
}

func targetForBinding(ruleID, fact string, subject api.Subject, plan api.OperationPlan) string {
	// Custom collector locators are runtime evidence. Keeping the adapter name
	// free of a generic fallback lets the renderer show the exact file/script.
	if strings.HasPrefix(fact, "custom.") {
		return ""
	}
	if target := networkFactTarget(fact, subject); target != "" {
		return target
	}
	if target := fileFactTarget(fact, subject, plan); target != "" {
		return target
	}
	if target := hostFactTarget(fact); target != "" {
		return target
	}
	return ruleFallbackTarget(ruleID, fact, subject)
}

func networkFactTarget(fact string, subject api.Subject) string {
	if strings.HasPrefix(fact, "endpoint.") {
		return endpointTargetForFact(fact)
	}
	if strings.HasPrefix(fact, "port.") {
		parts := strings.Split(fact, ".")
		if len(parts) > 1 && parts[1] != "" {
			address := subject.Address
			if address == "" {
				address = "0.0.0.0"
			}
			return net.JoinHostPort(address, parts[1])
		}
	}
	if strings.HasPrefix(fact, "ssh.") {
		return sshTarget(subject)
	}
	if strings.HasPrefix(fact, "systemd.") {
		return "systemctl --version"
	}
	if strings.HasPrefix(fact, "time.skew.") {
		return "date +%s on " + subject.Address
	}
	return ""
}

func fileFactTarget(fact string, subject api.Subject, plan api.OperationPlan) string {
	if strings.HasPrefix(fact, "path.") {
		return pathTarget(subject, fact)
	}
	if strings.HasPrefix(fact, "artifact.runc.") {
		return "current bkeadm executable"
	}
	if strings.HasPrefix(fact, "artifact.") {
		if target := artifactTarget(plan, fact); target != "" {
			return target
		}
		return subject.Metadata["offlineArtifact"]
	}
	if strings.HasPrefix(fact, "runtime.socket.") {
		if plan.Operation == api.OperationInit {
			return "/run/containerd/containerd.sock and /var/run/docker.sock"
		}
		return "/run/containerd/containerd.sock"
	}
	if strings.HasPrefix(fact, "runtime.dataRoot.") {
		if value := subject.Metadata["runtimeDataRoot"]; value != "" {
			return value
		}
		return subject.Metadata["runtimeStorage"]
	}
	if strings.HasPrefix(fact, "certificate.") || strings.HasPrefix(fact, "privateKey.") {
		return subject.Metadata["imageRepoCAFile"]
	}
	if strings.HasPrefix(fact, "offline.") {
		return subject.Metadata["offlineArtifact"]
	}
	return ""
}

func hostFactTarget(fact string) string {
	switch {
	case strings.HasPrefix(fact, "sysctl."):
		return "/proc/sys/" + strings.ReplaceAll(strings.TrimPrefix(fact, "sysctl."), ".", "/")
	case strings.HasPrefix(fact, "swap."):
		return "/proc/swaps"
	case strings.HasPrefix(fact, "firewall."):
		return "firewalld (systemctl is-active firewalld)"
	case strings.HasPrefix(fact, "selinux."):
		return "getenforce"
	case strings.HasPrefix(fact, "platform."):
		return platformTarget(fact)
	case strings.HasPrefix(fact, "resources."):
		return resourceTarget(fact)
	case strings.HasPrefix(fact, "exec."):
		return "current preflight process"
	case strings.HasPrefix(fact, "tool."):
		return "PATH: " + strings.TrimPrefix(fact, "tool.")
	case strings.HasPrefix(fact, "filesystem.root."):
		return "/"
	case strings.HasPrefix(fact, "plan.customRepo."):
		return "configured repository endpoints"
	case strings.HasPrefix(fact, "plan.colocate"):
		return "bootstrap node placement"
	default:
		return ""
	}
}

func ruleFallbackTarget(ruleID, fact string, subject api.Subject) string {
	if strings.HasPrefix(ruleID, "CLUSTER-MGMT-") {
		return "management cluster API/kubeconfig"
	}
	if strings.HasPrefix(ruleID, "CLUSTER-NET-") {
		return "cluster network configuration"
	}
	if strings.HasPrefix(ruleID, "CLUSTER-LOCAL-") {
		if value := subject.Metadata["outputDir"]; value != "" {
			return value
		}
		if value := subject.Metadata["workspace"]; value != "" {
			return value
		}
	}
	if strings.HasPrefix(ruleID, "CLUSTER-") && subject.Address != "" {
		return subject.Address
	}
	if subject.ID == "bootstrap" {
		return "bootstrap host"
	}
	if subject.ID == "cluster" {
		return "cluster configuration"
	}
	if fact != "" {
		return fact
	}
	return "operation plan"
}

func endpointTargetForFact(fact string) string {
	prefixes := []string{
		"endpoint.dns.",
		"endpoint.tcp.",
		"endpoint.tls.",
		"endpoint.auth.",
		"endpoint.object.",
	}
	for _, prefix := range prefixes {
		if strings.HasPrefix(fact, prefix) {
			target := strings.TrimPrefix(fact, prefix)
			if strings.HasPrefix(prefix, "endpoint.dns.") {
				return target
			}
			if strings.Contains(target, "://") {
				return target
			}
			return "https://" + target
		}
	}
	return ""
}

func sshTarget(subject api.Subject) string {
	address := subject.Address
	if address == "" {
		return "SSH target"
	}
	port := subject.Metadata["sshPort"]
	if port != "" {
		address = net.JoinHostPort(address, port)
	}
	if user := subject.Metadata["sshUser"]; user != "" {
		return user + "@" + address
	}
	return address
}

func pathTarget(subject api.Subject, fact string) string {
	if strings.HasPrefix(fact, "path.local.") {
		if value := subject.Metadata["outputDir"]; value != "" {
			return value
		}
		return subject.Metadata["workspace"]
	}
	if strings.HasPrefix(fact, "path.workspace.") {
		return subject.Metadata["workspace"]
	}
	if strings.HasPrefix(fact, "path.runc-dest.") {
		return "/usr/local/bin"
	}
	if strings.HasPrefix(fact, "path.customRepoCA.") || strings.HasPrefix(fact, "path.caCert.") ||
		strings.HasPrefix(fact, "path.caKey.") {
		return subject.Metadata["imageRepoCAFile"]
	}
	parts := strings.Split(fact, ".")
	if len(parts) >= minimumPathFactParts {
		paths := map[string]string{
			"RUNTIME":          subject.Metadata["runtimeDataRoot"],
			"KUBELET":          subject.Metadata["kubeletRoot"],
			"CERT":             subject.Metadata["certificatesDir"],
			"ETCD":             subject.Metadata["etcdDataDir"],
			"BKEAGENT-BIN":     "/usr/local/bin",
			"BKEAGENT-SERVICE": "/etc/systemd/system",
			"BKEAGENT-CONFIG":  "/etc/openFuyao",
		}
		return paths[parts[len(parts)-pathKindOffset]]
	}
	return ""
}

func artifactTarget(plan api.OperationPlan, fact string) string {
	if value, ok := plan.Values["offline.artifact"]; ok {
		if path, ok := value.Value.(string); ok && path != "" {
			return path
		}
	}
	return ""
}

func platformTarget(fact string) string {
	switch {
	case strings.Contains(fact, "distribution"):
		return "/etc/os-release"
	case strings.Contains(fact, "kernel"):
		return "uname -r"
	case strings.Contains(fact, "architecture"):
		return "uname -m"
	default:
		return "uname -s"
	}
}

func resourceTarget(fact string) string {
	switch {
	case strings.Contains(fact, "cpu"):
		return "logical CPU count (nproc)"
	case strings.Contains(fact, "memory"):
		return "MemTotal (free -b)"
	case strings.Contains(fact, "disk"):
		return "/ (df -B1)"
	default:
		return "host resources"
	}
}

func bindingMatchesPort(ruleID, key string, operation api.Operation) bool {
	if operation == api.OperationInit {
		return map[string]string{
			"kubernetes": "INIT-PORT-K8S",
			"imageRepo":  "INIT-PORT-REGISTRY",
			"chart":      "INIT-PORT-CHART",
			"yum":        "INIT-PORT-HTTP",
			"console":    "INIT-PORT-CONSOLE",
			"ntp":        "INIT-PORT-NTP",
		}[key] == ruleID
	}
	return map[string]string{
		"apiServer":         "CLUSTER-PORT-APISERVER",
		"etcdClient":        "CLUSTER-PORT-ETCD-CLIENT",
		"etcdPeer":          "CLUSTER-PORT-ETCD-PEER",
		"etcdMetrics":       "CLUSTER-PORT-ETCD-METRICS",
		"kubelet":           "CLUSTER-PORT-KUBELET-SECURE",
		"kubeletHealth":     "CLUSTER-PORT-KUBELET-HEALTH",
		"controllerManager": "CLUSTER-PORT-CONTROLLER",
		"scheduler":         "CLUSTER-PORT-SCHEDULER",
		"localHA":           "CLUSTER-PORT-LOCAL-HA",
		"bkeAgent":          "CLUSTER-PORT-BKEAGENT",
	}[key] == ruleID
}

func planBool(plan api.OperationPlan, key string) bool {
	v, ok := plan.Values[key]
	if !ok {
		return false
	}
	b, ok := v.Value.(bool)
	return ok && b
}

func subjectHasRole(subject api.Subject, roles []string) bool {
	for _, want := range roles {
		if want == "all" {
			return true
		}
		for _, have := range subject.Role {
			if want == have {
				return true
			}
		}
	}
	return false
}

func clusterOutputNames(cluster *configv1beta1.BKECluster, nodes []configv1beta1.BKENode) []string {
	namespace := cluster.Namespace
	if namespace == "" {
		namespace = "default"
	}
	out := []string{filepath.Base(fmt.Sprintf("%s-%s.yaml", namespace, cluster.Name))}
	for _, node := range nodes {
		out = append(out, filepath.Base(fmt.Sprintf("%s-%s-node.yaml", namespace, node.Name)))
	}
	return out
}

func clusterInputError(field string, err error) api.ResolveResult {
	return api.ResolveResult{Issues: []api.RunIssue{{
		Code: "CLUSTER-INPUT-SCHEMA", Severity: api.IssueError,
		Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
		Field: field, Cause: err.Error(), Action: "correct the input YAML, then run preflight again",
	}}}
}

func subjectsByID(subjects []api.Subject) map[string]api.Subject {
	out := make(map[string]api.Subject, len(subjects))
	for _, subject := range subjects {
		out[subject.ID] = subject
	}
	return out
}

func initPlanFacts(plan api.OperationPlan) []api.Fact {
	return []api.Fact{{
		Key: "plan.colocate", Subject: "bootstrap", Available: true, Value: false,
		Evidence: []api.Evidence{
			{Kind: "plan", Source: "openfuyao-adapter", Observed: "no workload nodes in init input"},
		},
	}, planFact("bootstrap", "plan.customRepo.valid", sourcesValid(plan.Sources))}
}

func sourcesValid(sources []api.Source) bool {
	for _, source := range sources {
		if source.Endpoint == "" || (source.Scheme != "http" && source.Scheme != "https") {
			return false
		}
	}
	return true
}

func clusterPlanFacts(nodes []configv1beta1.BKENode, resolved ResolveResult) []api.Fact {
	uniqueIP, uniqueHostname := true, true
	ips, hostnames := map[string]bool{}, map[string]bool{}
	for _, node := range nodes {
		uniqueIP = uniqueIP && node.Spec.IP != "" && !ips[node.Spec.IP]
		if hostname := strings.TrimSpace(node.Spec.Hostname); hostname != "" {
			uniqueHostname = uniqueHostname && !hostnames[hostname]
			hostnames[hostname] = true
		}
		ips[node.Spec.IP] = true
	}
	podValid, serviceValid := cidrRelationship(resolved.PodSubnet, resolved.ServiceSubnet)
	facts := []api.Fact{
		planFact("cluster", "unique.ip", uniqueIP),
		planFact("cluster", "unique.hostname", uniqueHostname),
		planFact("cluster", "cluster.topology.masterCount", resolved.MasterCount),
		planFact(
			"cluster", "cluster.topology.masterCount.odd",
			resolved.MasterCount%oddDivisor == oddRemainder,
		),
		planFact("cluster", "cluster.topology.etcdCount", resolved.EtcdCount),
		planFact("cluster", "cluster.topology.workerCount", resolved.WorkerCount),
		networkPlanFact(
			"cluster.podSubnet.valid", podValid,
			"cluster.spec.clusterConfig.cluster.networking.podSubnet", resolved.PodSubnet,
		),
		networkPlanFact(
			"cluster.serviceSubnet.valid", serviceValid,
			"cluster.spec.clusterConfig.cluster.networking.serviceSubnet", resolved.ServiceSubnet,
		),
		validationPlanFact(resolved, "local.cluster.valid", resolved.ClusterValid),
		validationPlanFact(resolved, "local.node.valid", resolved.NodesValid),
		validationPlanFact(resolved, "cluster.ref.versionSupported", resolved.VersionSupported),
		validationPlanFact(resolved, "cluster.addon.closure", resolved.AddonValid),
		validationPlanFact(resolved, "cluster.addon.source", resolved.AddonSourceValid),
	}
	// Only a proven bootstrap identity may drive the HA advisory. A normal
	// master+worker role never sets BootstrapNodeID and therefore cannot warn.
	if resolved.BootstrapNodeID != "" && resolved.MasterCount > 1 {
		facts = append(
			facts,
			planFact(
				"cluster",
				"cluster.ha.colocated",
				contains(resolved.NodeRoles[resolved.BootstrapNodeID], "master"),
			),
		)
	}
	return facts
}

func validationPlanFact(resolved ResolveResult, key string, value bool) api.Fact {
	fact := planFact("cluster", key, value)
	if finding, ok := resolved.ValidationFields[key]; ok {
		fact.Evidence[0].Locator = finding.Field
		fact.Evidence[0].Observed = finding.Cause
	}
	return fact
}

func networkPlanFact(key string, valid bool, field, value string) api.Fact {
	fact := planFact("cluster", key, valid)
	fact.Evidence[0].Locator = field
	if !valid {
		fact.Evidence[0].Observed = value
	}
	return fact
}

func cidrRelationship(pod, service string) (bool, bool) {
	_, _, podErr := net.ParseCIDR(pod)
	_, _, serviceErr := net.ParseCIDR(service)
	if podErr != nil || serviceErr != nil {
		return podErr == nil, serviceErr == nil
	}
	// The Provider contract currently requires parseable CIDRs. It does not
	// define a non-overlap gate, so keep this adapter fact limited to syntax.
	return true, true
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func planFact(subject, key string, value any) api.Fact {
	return api.Fact{Key: key, Subject: subject, Available: true, Value: value,
		Evidence: []api.Evidence{{Kind: "plan", Source: "openfuyao-adapter"}}}
}
