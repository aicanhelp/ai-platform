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

// Package clustercreate turns a resolved cluster-create plan into an
// OperationPlan and rule instances. It is pure: no I/O, no API reads, no
// resource creation. The host resolves cluster/nodes and the Provider resolver
// first, then calls Resolve.
package clustercreate

import (
	"fmt"
	"net/url"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/adapterutil"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/normalize"
)

const (
	clusterMinimumCPU    = 8
	clusterMinimumMemory = int64(16 << 30)
	clusterMinimumDisk   = int64(100 << 30)
	minimumRoleCount     = 1
	singleMasterCount    = 1

	defaultControllerManagerPort = 10257
	defaultSchedulerPort         = 10259
	defaultLocalHAPort           = 36443
	defaultEtcdClientPort        = 2379
	defaultEtcdPeerPort          = 2380
	defaultEtcdMetricsPort       = 2381
	defaultKubeletPort           = 10250
	defaultKubeletHealthPort     = 10248
	defaultBKEAgentPort          = 58080
)

// ResolvedCluster is the pure, resolved cluster plan from the host's Provider
// resolver. It carries only plan-relevant facts; no secrets.
type ResolvedCluster struct {
	APIBindPort      int
	MasterCount      int
	EtcdCount        int
	WorkerCount      int
	EndpointExplicit bool
	// BootstrapNodeID is set only when the host can prove that the machine
	// running bke is one of the declared cluster nodes. An empty value means the
	// identity is unknown and the HA colocation advisory is not instantiated.
	BootstrapNodeID string
	PodSubnet       string
	ServiceSubnet   string
	Runtime         string
	EtcdDataDir     string
	KubeletRoot     string
	DataRoot        string
	// Nodes with roles and addresses (nodeID -> roles, nodeID -> addr).
	NodeRoles     map[string][]string
	NodeAddresses map[string]string
	NodeMetadata  map[string]map[string]string
	Sources       []api.Source
	Workspace     string
	OutputNames   []string
}

// Resolve builds an OperationPlan from a resolved cluster. It projects the
// topology, ports, paths, runtime and sources, then computes PlanHash.
func Resolve(rc *ResolvedCluster) (api.OperationPlan, []api.RunIssue) {
	if rc == nil {
		return api.OperationPlan{}, missingClusterIssues()
	}
	plan := api.OperationPlan{
		SchemaVersion: api.SchemaVersion,
		Operation:     api.OperationClusterCreate,
		Subjects:      clusterSubjects(rc),
		Values:        clusterValues(rc),
		Sources:       append([]api.Source(nil), rc.Sources...),
	}
	return finalizeClusterPlan(plan)
}

func missingClusterIssues() []api.RunIssue {
	return []api.RunIssue{{
		Code:     "CLUSTER-INPUT-REQUIRED",
		Severity: api.IssueError,
		Stage:    api.IssueStageInput,
		Effect:   api.IssueBlockStart,
		Cause:    "resolved cluster is nil",
		Action:   "provide resolved cluster input, then run preflight again",
	}}
}

func clusterSubjects(rc *ResolvedCluster) []api.Subject {
	outputDir := filepath.Join(rc.Workspace, "cluster")
	subjects := []api.Subject{{
		ID:   "cluster",
		Role: []string{"cluster"},
		Metadata: map[string]string{
			"workspace":       rc.Workspace,
			"outputDir":       outputDir,
			"expectedOutputs": strings.Join(rc.OutputNames, ","),
		},
	}}
	for id, roles := range rc.NodeRoles {
		subjects = append(subjects, api.Subject{
			ID: id, Role: roles, Address: rc.NodeAddresses[id], Metadata: rc.NodeMetadata[id],
		})
	}
	sort.Slice(subjects, func(i, j int) bool { return subjects[i].ID < subjects[j].ID })
	return subjects
}

func clusterValues(rc *ResolvedCluster) map[string]api.Value {
	values := map[string]api.Value{
		"cluster.apiPort": clusterValue(
			"cluster.apiPort", api.ValueTypeInteger, rc.APIBindPort,
		),
		"cluster.topology.masterCount": clusterValue(
			"cluster.topology.masterCount", api.ValueTypeInteger, rc.MasterCount,
		),
		"cluster.topology.etcdCount": clusterValue(
			"cluster.topology.etcdCount", api.ValueTypeInteger, rc.EtcdCount,
		),
		"cluster.topology.workerCount": clusterValue(
			"cluster.topology.workerCount", api.ValueTypeInteger, rc.WorkerCount,
		),
		"cluster.topology.multiMaster": clusterValue(
			"cluster.topology.multiMaster", api.ValueTypeBool, rc.MasterCount > singleMasterCount,
		),
		"cluster.podSubnet": clusterValue(
			"cluster.podSubnet", api.ValueTypeString, rc.PodSubnet,
		),
		"cluster.serviceSubnet": clusterValue(
			"cluster.serviceSubnet", api.ValueTypeString, rc.ServiceSubnet,
		),
		"cluster.runtime": clusterValue(
			"cluster.runtime", api.ValueTypeEnum, rc.Runtime,
		),
		"node.etcdDataDir": clusterValue(
			"node.etcdDataDir", api.ValueTypeString, rc.EtcdDataDir,
		),
		"node.kubeletRoot": clusterValue(
			"node.kubeletRoot", api.ValueTypeString, rc.KubeletRoot,
		),
		"node.dataRoot": clusterValue(
			"node.dataRoot", api.ValueTypeString, rc.DataRoot,
		),
		"node.cpu.logical": clusterValue(
			"node.cpu.logical", api.ValueTypeInteger, clusterMinimumCPU,
		),
		"node.memory.totalBytes": clusterValue(
			"node.memory.totalBytes", api.ValueTypeBytes, clusterMinimumMemory,
		),
	}
	if rc.BootstrapNodeID != "" {
		values["cluster.bootstrapNodeID"] = clusterValue(
			"cluster.bootstrapNodeID", api.ValueTypeString, rc.BootstrapNodeID,
		)
	}
	return values
}

func clusterValue(key string, valueType api.ValueType, value any) api.Value {
	return api.Value{Key: key, Type: valueType, Value: value}
}

func finalizeClusterPlan(plan api.OperationPlan) (api.OperationPlan, []api.RunIssue) {
	h, err := normalize.ComputePlanHash(&plan)
	if err != nil {
		return plan, []api.RunIssue{{
			Code:     "CLUSTER-PLAN-HASH",
			Severity: api.IssueError,
			Stage:    api.IssueStagePlan,
			Effect:   api.IssueBlockStart,
			Cause:    fmt.Sprintf("compute plan hash: %v", err),
			Action:   "correct the reported cluster input, then run preflight again",
		}}
	}
	plan.PlanHash = h
	return plan, nil
}

// Instances builds the rule instances for the cluster-create plan using the
// compatibility port defaults. New hosts should call InstancesWithPorts.
func Instances(plan api.OperationPlan) []ClusterInstance {
	return InstancesWithPorts(plan, map[string]int{
		"controllerManager": defaultControllerManagerPort,
		"scheduler":         defaultSchedulerPort,
		"localHA":           defaultLocalHAPort,
		"etcdClient":        defaultEtcdClientPort,
		"etcdPeer":          defaultEtcdPeerPort,
		"etcdMetrics":       defaultEtcdMetricsPort,
		"kubelet":           defaultKubeletPort,
		"kubeletHealth":     defaultKubeletHealthPort,
		"bkeAgent":          defaultBKEAgentPort,
	})
}

// InstancesWithPorts expands dynamic node/role checks with policy-provided
// default ports. A non-zero plan value is always used first.
func InstancesWithPorts(plan api.OperationPlan, defaults map[string]int) []ClusterInstance {
	return buildInstancesWithPorts(plan, defaults)
}

// buildInstancesWithPorts projects the supported cluster-create checks into
// instances per-node/role/port/addon/source. It intentionally does not probe
// paths that the installation creates, nor ambient node state that the
// Provider can safely reuse or reconcile.
//
// Rule grouping:
//
//  1. Input/uniqueness: UNIQUE-IP/HOSTNAME
//
//  2. Topology: TOPO-MASTER/ODD/ETCD/WORKER
//
//  3. Mgmt cluster: API, DISCOVERY, CRD, AGENT-KUBECONFIG, LOCAL
//
//  4. SSH: TRANSPORT, AUTH, PRIVILEGE, AGENT-SYSTEMD
//
//  5. Node resources/platform (§8.6): CPU, MEMORY, SYSTEM-DISK, PLAT-OS/DISTRO/KERNEL
//
//  6. BKE local path: CREATE, ACCESS, MOUNT
//
//  7. Node ports: APISERVER, ETCD-CLIENT/PEER/METRICS, KUBELET-SECURE/HEALTH, CONTROLLER, SCHEDULER, BKEAGENT, HA
//
//  8. Source comms: one TCP reachability check per source per node
//
//  9. Network/HA: POD-CIDR, SERVICE-CIDR, COLOCATE
//
// 10. Custom refs/addons: REF-VERSION, ADDON-CLOSURE/SOURCE
func buildInstancesWithPorts(plan api.OperationPlan, defaults map[string]int) []ClusterInstance {
	out := clusterLevelInstances(plan)
	for _, sub := range plan.Subjects {
		if sub.ID != "cluster" {
			out = append(out, nodeInstances(sub, plan, defaults)...)
		}
	}
	return out
}

func clusterLevelInstances(plan api.OperationPlan) []ClusterInstance {
	out := []ClusterInstance{
		clusterCheck("CLUSTER-UNIQUE-IP", "unique.ip", api.OpEq, true),
		clusterCheck("CLUSTER-UNIQUE-HOSTNAME", "unique.hostname", api.OpEq, true),
		clusterCheck("CLUSTER-TOPO-MASTER", "cluster.topology.masterCount", api.OpGte, minimumRoleCount),
		clusterCheck("CLUSTER-TOPO-MASTER-ODD", "cluster.topology.masterCount.odd", api.OpRelation, true),
		clusterCheck("CLUSTER-TOPO-ETCD", "cluster.topology.etcdCount", api.OpGte, minimumRoleCount),
		clusterCheck("CLUSTER-TOPO-WORKER", "cluster.topology.workerCount", api.OpGte, minimumRoleCount),
		clusterCheck("CLUSTER-MGMT-API", "kubernetes.api.reachable", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-DISCOVERY", "kubernetes.discovery.complete", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-CRD-CLUSTER", "crd.bkeclusters", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-CRD-NODE", "crd.bkenodes", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-AGENT-KUBECONFIG-GET", "auth.get.localkubeconfig", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-AGENT-KUBECONFIG-DATA", "secret.localkubeconfig.dataNonEmpty", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE", "kubeconfig.parseable", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE", "kubeconfig.sourceEndpoint", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-LOCAL-CLUSTER", "local.cluster.valid", api.OpEq, true),
		clusterCheck("CLUSTER-MGMT-LOCAL-NODE", "local.node.valid", api.OpEq, true),
		clusterCheck("CLUSTER-LOCAL-PATH-CREATE", "path.local.creatable", api.OpEq, true),
		clusterCheck("CLUSTER-LOCAL-PATH-ACCESS", "path.local.accessible", api.OpEq, true),
		clusterCheck("CLUSTER-LOCAL-PATH-MOUNT", "path.local.mountReadOnly", api.OpEq, false),
		clusterCheck("CLUSTER-NET-POD-CIDR", "cluster.podSubnet.valid", api.OpEq, true),
		clusterCheck("CLUSTER-NET-SERVICE-CIDR", "cluster.serviceSubnet.valid", api.OpEq, true),
	}
	if planString(plan, "cluster.bootstrapNodeID") != "" &&
		planInt(plan, "cluster.topology.masterCount") > singleMasterCount {
		out = append(out, newClusterInstance(clusterInstanceSpec{
			id: "CLUSTER-HA-COLOCATE", fact: "cluster.ha.colocated",
			op: api.OpEq, expected: false, level: api.SeverityWarning,
			unknown: api.UnknownWarn, subject: "cluster",
		}))
	}
	out = append(out,
		clusterCheck("CLUSTER-REF-VERSION", "cluster.ref.versionSupported", api.OpEq, true),
		clusterCheck("CLUSTER-ADDON-CLOSURE", "cluster.addon.closure", api.OpEq, true),
		clusterCheck("CLUSTER-ADDON-SOURCE", "cluster.addon.source", api.OpEq, true),
	)
	return out
}

func clusterCheck(id, fact string, op api.Op, expected any) ClusterInstance {
	return newClusterInstance(clusterInstanceSpec{
		id: id, fact: fact, op: op, expected: expected,
		level: api.SeverityError, unknown: api.UnknownBlock, subject: "cluster",
	})
}

func nodeInstances(
	sub api.Subject,
	plan api.OperationPlan,
	defaults map[string]int,
) []ClusterInstance {
	out := []ClusterInstance{
		ciNode("CLUSTER-SSH-TRANSPORT", sub.ID, "ssh.transport."+sub.ID, api.OpReachable, true),
		ciNode("CLUSTER-SSH-AUTH", sub.ID, "ssh.auth."+sub.ID, api.OpEq, true),
		ciNode("CLUSTER-SSH-PRIVILEGE", sub.ID, "ssh.privilege."+sub.ID, api.OpEq, true),
		ciNode("CLUSTER-AGENT-SYSTEMD", sub.ID, "systemd."+sub.ID+".available", api.OpEq, true),
	}
	out = append(out, nodeResourceInstances(sub.ID)...)
	out = append(out, nodePortInstances(sub, plan, defaults)...)
	out = append(out, nodeSourceInstances(sub.ID, plan.Sources)...)
	return out
}

func nodeResourceInstances(subject string) []ClusterInstance {
	return []ClusterInstance{
		ciNode("CLUSTER-RES-CPU", subject, "node.cpu.logical", api.OpGte, clusterMinimumCPU),
		ciNode("CLUSTER-RES-MEMORY", subject, "node.memory.totalBytes", api.OpGte, clusterMinimumMemory),
		ciNode("CLUSTER-RES-SYSTEM-DISK", subject, "node.systemDisk.totalBytes", api.OpGte, clusterMinimumDisk),
		ciNode("CLUSTER-PLAT-OS", subject, "platform.os."+subject, api.OpEq, "linux"),
		ciNode("CLUSTER-PLAT-DISTRO", subject, "platform.distribution."+subject, api.OpIn, supportedDistributions()),
		ciNode("CLUSTER-PLAT-KERNEL", subject, "platform.kernel."+subject+".matched", api.OpEq, true),
	}
}

func supportedDistributions() []any {
	return []any{
		[]any{"openeuler", "20.03"},
		[]any{"openeuler", "22.03"},
		[]any{"openeuler", "24.03"},
		[]any{"ubuntu", "22.04"},
		[]any{"uos", "20 1070e"},
	}
}

func nodePortInstances(sub api.Subject, plan api.OperationPlan, defaults map[string]int) []ClusterInstance {
	var out []ClusterInstance
	if hasRole(sub.Role, "master") {
		out = append(out, masterPortInstances(sub.ID, plan, defaults)...)
	}
	if hasRole(sub.Role, "etcd") {
		out = append(out, etcdPortInstances(sub.ID, plan, defaults)...)
	}
	if hasRole(sub.Role, "master") || hasRole(sub.Role, "worker") {
		out = append(out, kubernetesPortInstances(sub.ID, plan, defaults)...)
	}
	out = append(out, defaultPortInstance("CLUSTER-PORT-BKEAGENT", sub.ID, "bkeAgent", plan, defaults))
	return out
}

func masterPortInstances(subject string, plan api.OperationPlan, defaults map[string]int) []ClusterInstance {
	out := []ClusterInstance{
		ciPort("CLUSTER-PORT-APISERVER", subject, plan, "cluster.apiPort"),
		defaultPortInstance("CLUSTER-PORT-CONTROLLER", subject, "controllerManager", plan, defaults),
		defaultPortInstance("CLUSTER-PORT-SCHEDULER", subject, "scheduler", plan, defaults),
	}
	if planInt(plan, "cluster.topology.masterCount") > singleMasterCount {
		out = append(out, defaultPortInstance("CLUSTER-PORT-LOCAL-HA", subject, "localHA", plan, defaults))
	}
	return out
}

func etcdPortInstances(subject string, plan api.OperationPlan, defaults map[string]int) []ClusterInstance {
	return []ClusterInstance{
		defaultPortInstance("CLUSTER-PORT-ETCD-CLIENT", subject, "etcdClient", plan, defaults),
		defaultPortInstance("CLUSTER-PORT-ETCD-PEER", subject, "etcdPeer", plan, defaults),
		defaultPortInstance("CLUSTER-PORT-ETCD-METRICS", subject, "etcdMetrics", plan, defaults),
	}
}

func kubernetesPortInstances(subject string, plan api.OperationPlan, defaults map[string]int) []ClusterInstance {
	return []ClusterInstance{
		defaultPortInstance("CLUSTER-PORT-KUBELET-SECURE", subject, "kubelet", plan, defaults),
		defaultPortInstance("CLUSTER-PORT-KUBELET-HEALTH", subject, "kubeletHealth", plan, defaults),
	}
}

func defaultPortInstance(
	id, subject, key string,
	plan api.OperationPlan,
	defaults map[string]int,
) ClusterInstance {
	return ciStatic(id, subject, planPortDefault(plan, key, defaults), api.OpFree)
}

func nodeSourceInstances(subject string, sources []api.Source) []ClusterInstance {
	out := make([]ClusterInstance, 0, len(sources))
	for _, src := range sources {
		out = append(out, sourceNodeInstance(src, subject))
	}
	return out
}

func planPortDefault(plan api.OperationPlan, key string, defaults map[string]int) int {
	if value := planInt(plan, "cluster.port."+key); value > 0 {
		return value
	}
	return defaults[key]
}

func planInt(plan api.OperationPlan, key string) int {
	return adapterutil.PlanInt(plan, key)
}

func planString(plan api.OperationPlan, key string) string {
	if value, ok := plan.Values[key]; ok {
		if typed, ok := value.Value.(string); ok {
			return typed
		}
	}
	return ""
}

// ClusterInstance is the local instance type (avoids an engine import cycle).
type ClusterInstance struct {
	Rule       api.RuleTemplate
	Subject    string
	FactKey    string
	InstanceID string
	Applicable bool
}

func hasRole(roles []string, role string) bool {
	for _, r := range roles {
		if r == role {
			return true
		}
	}
	return false
}

type clusterInstanceSpec struct {
	id       string
	fact     string
	op       api.Op
	expected any
	level    api.FailureSeverity
	unknown  api.UnknownPolicy
	subject  string
}

func newClusterInstance(spec clusterInstanceSpec) ClusterInstance {
	name, fix := clusterRuleMetadata(spec.id, spec.fact)
	return ClusterInstance{
		Rule: api.RuleTemplate{ID: spec.id, Name: name, Operation: api.OperationClusterCreate,
			Fact: spec.fact, Op: spec.op, Expected: spec.expected, Level: spec.level,
			Unknown: spec.unknown, Fix: fix,
			DependsOn: managementRuleDependencies(spec.id)},
		Subject: spec.subject, FactKey: spec.fact, Applicable: true,
	}
}

func managementRuleDependencies(id string) []string {
	switch id {
	case "CLUSTER-MGMT-DISCOVERY":
		return []string{"CLUSTER-MGMT-API"}
	case "CLUSTER-MGMT-CRD-CLUSTER", "CLUSTER-MGMT-CRD-NODE":
		return []string{"CLUSTER-MGMT-DISCOVERY"}
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-GET":
		return []string{"CLUSTER-MGMT-API"}
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-DATA":
		return []string{"CLUSTER-MGMT-AGENT-KUBECONFIG-GET"}
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE":
		return []string{"CLUSTER-MGMT-AGENT-KUBECONFIG-DATA"}
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE":
		return []string{"CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE"}
	default:
		return nil
	}
}

func ciNode(id, subject, fact string, op api.Op, expected any) ClusterInstance {
	name, fix := clusterRuleMetadata(id, fact)
	if dynamic := resourceFix(id, expected); dynamic != "" {
		fix = dynamic
	}
	return ClusterInstance{
		Rule: api.RuleTemplate{ID: id, Name: name, Operation: api.OperationClusterCreate, Fact: fact, Op: op,
			Expected: expected, Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: fix,
			DependsOn: nodeRuleDependencies(id)},
		Subject: subject, FactKey: fact, Applicable: true,
	}
}

// resourceFix renders the node-resource remediation from the actual expected
// value so a policy-adjusted threshold never leaves a stale hardcoded fix.
func resourceFix(id string, expected any) string {
	switch id {
	case "CLUSTER-RES-CPU":
		if cpus, ok := expected.(int); ok && cpus > 0 {
			return fmt.Sprintf("use a node with at least %d logical CPUs", cpus)
		}
	case "CLUSTER-RES-MEMORY":
		if bytes, ok := expected.(int64); ok && bytes > 0 {
			return fmt.Sprintf("use a node with at least %s memory", adapterutil.FormatGiB(bytes))
		}
	case "CLUSTER-RES-SYSTEM-DISK":
		if bytes, ok := expected.(int64); ok && bytes > 0 {
			return fmt.Sprintf("use a node with at least %s system disk capacity", adapterutil.FormatGiB(bytes))
		}
	default:
	}
	return ""
}

func sourceNodeInstance(source api.Source, subject string) ClusterInstance {
	const id = "CLUSTER-SRC-REACHABLE"
	fact := "endpoint.tcp." + source.Scheme + "://" + source.Endpoint
	// A configured IP rides along in the fact key after "@" so the node-side
	// probe connects to the IP while keeping the domain as Host and TLS SNI.
	if source.Address != "" && source.Address != source.Endpoint {
		fact += "@" + source.Address
	}
	instance := ciNode(id, subject, fact, api.OpEq, true)
	endpoint := sourceEndpointFromFact(instance.FactKey)
	instance.Rule.Name = adapterutil.SourceCheckName(sourceName(source.ID), endpoint)
	instance.Rule.Fix = sourceCheckFix(source.ID, endpoint)
	instance.InstanceID = id + "@" + subject + ":" + source.ID
	return instance
}

func sourceEndpointFromFact(fact string) string {
	prefixes := []string{
		"endpoint.dns.",
		"endpoint.tcp.",
		"endpoint.tls.",
		"endpoint.auth.",
		"endpoint.object.",
	}
	for _, prefix := range prefixes {
		if strings.HasPrefix(fact, prefix) {
			value := strings.TrimPrefix(fact, prefix)
			// Drop a trailing "@ip:port" resolve override; the operator-facing
			// endpoint stays the domain.
			if at := strings.LastIndex(value, "@"); at >= 0 {
				value = value[:at]
			}
			if parsed, err := url.Parse(value); err == nil && parsed.Host != "" {
				return parsed.Host
			}
			return value
		}
	}
	return ""
}

func sourceName(sourceID string) string {
	switch sourceID {
	case "image-repo", "imageRepo":
		return "Image repository"
	case "chart-repo", "chartRepo":
		return "Chart repository"
	case "yum-repo", "yumRepo", "package-repo":
		return "Package repository"
	case "online-image":
		return "Online image source"
	default:
		return adapterutil.HumanizeSourceID(sourceID)
	}
}

func sourceCheckFix(sourceID, endpoint string) string {
	return adapterutil.SourceUnreachableFix(sourceName(sourceID), endpoint)
}

func ciPort(id, subject string, plan api.OperationPlan, key string) ClusterInstance {
	port := 0
	if v, ok := plan.Values[key]; ok {
		port, _ = v.Value.(int)
	}
	return ciStatic(id, subject, port, api.OpFree)
}

// CatalogRules returns every stable cluster-create RuleTemplate. A synthetic
// all-role node enables each conditional branch; runtime bindings carry the
// actual node, port, source and path fact keys.
func CatalogRules() []api.RuleTemplate {
	plan := api.OperationPlan{
		Operation: api.OperationClusterCreate,
		Subjects:  []api.Subject{{ID: "catalog-node", Role: []string{"master", "etcd", "worker"}}},
		Values: map[string]api.Value{
			"cluster.apiPort":              {Key: "cluster.apiPort", Type: api.ValueTypeInteger, Value: 0},
			"cluster.topology.masterCount": {Key: "cluster.topology.masterCount", Type: api.ValueTypeInteger, Value: 3},
			"cluster.bootstrapNodeID": {
				Key:   "cluster.bootstrapNodeID",
				Type:  api.ValueTypeString,
				Value: "catalog-node",
			},
		},
		Sources: []api.Source{{ID: "release-source", Endpoint: "registry.example.invalid:443", Scheme: "https"}},
	}
	instances := InstancesWithPorts(plan, nil)
	return adapterutil.UniqueRules(instances, func(instance ClusterInstance) api.RuleTemplate {
		return instance.Rule
	})
}

func ciStatic(id, subject string, port int, op api.Op) ClusterInstance {
	factKey := fmt.Sprintf("port.%d.free", port)
	name, fix := clusterRuleMetadata(id, factKey)
	if name == "" {
		name = fmt.Sprintf("Required host port %d", port)
	}
	if fix == "" {
		fix = fmt.Sprintf(
			"free host port %d on node %s or change the corresponding cluster configuration",
			port,
			subject,
		)
	}
	return ClusterInstance{
		Rule: api.RuleTemplate{ID: id, Name: name, Operation: api.OperationClusterCreate, Fact: factKey, Op: op,
			Expected: true, Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: fix, DependsOn: nodeRuleDependencies(id)},
		Subject: subject, FactKey: factKey, Applicable: true,
	}
}

// nodeRuleDependencies make SSH transport/auth/privilege the per-node gate
// for all remote checks. The engine uses these edges both for result
// propagation and to avoid collecting remote facts after a node is known to
// be unreachable. Once transport/authentication/privilege is unavailable,
// dependent node facts are skipped and the root SSH finding carries the
// actionable failure.
func nodeRuleDependencies(id string) []string {
	switch id {
	case "CLUSTER-SSH-TRANSPORT":
		return nil
	case "CLUSTER-SSH-AUTH":
		return []string{"CLUSTER-SSH-TRANSPORT"}
	case "CLUSTER-SSH-PRIVILEGE":
		return []string{"CLUSTER-SSH-AUTH"}
	default:
		return []string{"CLUSTER-SSH-PRIVILEGE"}
	}
}

// clusterRuleMetadata keeps the operator-facing explanation beside the
// adapter rule that knows the concrete BKE field or host fact. The Bundle
// still carries stable IDs, while every bound instance gets a name and a
// remediation that points to the relevant YAML field, flag, or node action.
type ruleMetadata struct {
	name string
	fix  string
}

func clusterRuleMetadata(id, fact string) (string, string) {
	if strings.HasPrefix(id, "CLUSTER-PORT-") {
		return portRuleMetadata(id, fact)
	}
	if item, ok := topologyRuleMetadata(id); ok {
		return item.name, item.fix
	}
	if item, ok := managementMetadata(id); ok {
		return item.name, item.fix
	}
	if item, ok := clusterConfigMetadata(id); ok {
		return item.name, item.fix
	}
	if item, ok := nodeMetadata(id); ok {
		return item.name, item.fix
	}
	return humanizeClusterID(id),
		"review the corresponding field in cluster.yaml, nodes.yaml, or the resolved Provider configuration"
}

func portRuleMetadata(id, fact string) (string, string) {
	port := resolvedPort(fact)
	label := portRuleLabel(id)
	if label == "" {
		label = humanizeClusterID(id)
	}
	return fmt.Sprintf("%s host port %s", label, port), fmt.Sprintf(
		"free host port %s on the node or change the corresponding port "+
			"in cluster.yaml/Provider configuration",
		port,
	)
}

func resolvedPort(fact string) string {
	const resolved = "the resolved"
	if !strings.HasPrefix(fact, "port.") {
		return resolved
	}
	value := strings.TrimSuffix(strings.TrimPrefix(fact, "port."), ".free")
	if _, err := strconv.Atoi(value); err != nil {
		return resolved
	}
	return value
}

func portRuleLabel(id string) string {
	switch id {
	case "CLUSTER-PORT-APISERVER":
		return "Kubernetes API"
	case "CLUSTER-PORT-ETCD-CLIENT":
		return "etcd client"
	case "CLUSTER-PORT-ETCD-PEER":
		return "etcd peer"
	case "CLUSTER-PORT-ETCD-METRICS":
		return "etcd metrics"
	case "CLUSTER-PORT-KUBELET-SECURE":
		return "Kubelet secure API"
	case "CLUSTER-PORT-KUBELET-HEALTH":
		return "Kubelet health"
	case "CLUSTER-PORT-CONTROLLER":
		return "Controller Manager"
	case "CLUSTER-PORT-SCHEDULER":
		return "Scheduler"
	case "CLUSTER-PORT-LOCAL-HA":
		return "Local HA"
	case "CLUSTER-PORT-BKEAGENT":
		return "BKE Agent"
	default:
		return ""
	}
}

func foundMetadata(name, fix string) (ruleMetadata, bool) {
	return ruleMetadata{name: name, fix: fix}, true
}

func topologyRuleMetadata(id string) (ruleMetadata, bool) {
	switch id {
	case "CLUSTER-UNIQUE-IP":
		return foundMetadata(
			"Node IP addresses are unique (nodes.yaml spec.ip)",
			"assign a unique nodes.yaml spec.ip to every node",
		)
	case "CLUSTER-UNIQUE-HOSTNAME":
		return foundMetadata(
			"Non-empty node hostnames are unique (nodes.yaml spec.hostname)",
			"assign distinct nodes.yaml spec.hostname values where hostnames are explicitly configured",
		)
	case "CLUSTER-TOPO-MASTER":
		return foundMetadata(
			"At least one master node is configured",
			"add a node with the master role in nodes.yaml",
		)
	case "CLUSTER-TOPO-MASTER-ODD":
		return foundMetadata(
			"Master node count is odd",
			"change nodes.yaml roles so the master count is 1, 3, 5, ...",
		)
	case "CLUSTER-TOPO-ETCD":
		return foundMetadata(
			"At least one etcd node is configured",
			"add a node with the etcd role in nodes.yaml",
		)
	case "CLUSTER-TOPO-WORKER":
		return foundMetadata(
			"At least one Kubernetes worker node is configured",
			"add a node with the worker role in nodes.yaml",
		)
	default:
		return ruleMetadata{}, false
	}
}

func managementMetadata(id string) (ruleMetadata, bool) {
	switch id {
	case "CLUSTER-MGMT-API":
		return foundMetadata(
			"Management Kubernetes API is reachable (--kubeconfig)",
			"check --kubeconfig and verify the management API endpoint is reachable",
		)
	case "CLUSTER-MGMT-DISCOVERY":
		return foundMetadata(
			"Management Kubernetes API discovery succeeds",
			"check --kubeconfig credentials and API discovery permissions",
		)
	case "CLUSTER-MGMT-CRD-CLUSTER":
		return foundMetadata(
			"BKECluster CRD is installed in the management cluster",
			"install the BKECluster CRD in the management cluster before cluster-create",
		)
	case "CLUSTER-MGMT-CRD-NODE":
		return foundMetadata(
			"BKENode CRD is installed in the management cluster",
			"install the BKENode CRD in the management cluster before cluster-create",
		)
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-GET":
		return foundMetadata(
			"kube-system/localkubeconfig Secret is readable",
			"grant the kubeconfig used by --kubeconfig permission to get the Secret",
		)
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-DATA":
		return foundMetadata(
			"localkubeconfig Secret contains kubeconfig data",
			"populate the expected kubeconfig data key in kube-system/localkubeconfig",
		)
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE":
		return foundMetadata(
			"localkubeconfig data is valid kubeconfig",
			"replace the Secret data with a valid kubeconfig document",
		)
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE":
		return foundMetadata(
			"localkubeconfig contains a cluster server endpoint",
			"correct the server field in kube-system/localkubeconfig",
		)
	case "CLUSTER-MGMT-LOCAL-CLUSTER":
		return foundMetadata(
			"cluster.yaml passes local BKECluster schema and business validation",
			"fix the reported BKECluster field in cluster.yaml",
		)
	case "CLUSTER-MGMT-LOCAL-NODE":
		return foundMetadata(
			"nodes.yaml passes local BKENode schema and business validation",
			"fix the reported BKENode field in nodes.yaml",
		)
	default:
		return ruleMetadata{}, false
	}
}

func clusterConfigMetadata(id string) (ruleMetadata, bool) {
	switch id {
	case "CLUSTER-LOCAL-PATH-CREATE":
		return foundMetadata(
			"Cluster output directory is creatable",
			"choose a writable workspace/output directory for cluster-create",
		)
	case "CLUSTER-LOCAL-PATH-ACCESS":
		return foundMetadata(
			"Cluster output directory is accessible",
			"grant the installation user access to the workspace/output directory",
		)
	case "CLUSTER-LOCAL-PATH-MOUNT":
		return foundMetadata(
			"Cluster output filesystem is writable",
			"remount the workspace filesystem read-write or choose another workspace",
		)
	case "CLUSTER-NET-POD-CIDR":
		return foundMetadata(
			"Pod CIDR is valid (cluster.yaml cluster configuration)",
			"correct the Pod CIDR in cluster.yaml",
		)
	case "CLUSTER-NET-SERVICE-CIDR":
		return foundMetadata(
			"Service CIDR is valid (cluster.yaml cluster configuration)",
			"correct the Service CIDR in cluster.yaml",
		)
	case "CLUSTER-HA-COLOCATE":
		return foundMetadata(
			"HA nodes are not unintentionally colocated",
			"review node roles in nodes.yaml and distribute HA roles across failure domains",
		)
	case "CLUSTER-REF-VERSION":
		return foundMetadata(
			"Referenced cluster configuration version is supported",
			"set the supported version in cluster.yaml",
		)
	case "CLUSTER-ADDON-CLOSURE":
		return foundMetadata(
			"Configured addon dependencies are complete",
			"add the missing addon dependency or disable the addon in cluster.yaml",
		)
	case "CLUSTER-ADDON-SOURCE":
		return foundMetadata(
			"Configured addon source information is complete",
			"configure the chart repository required by chart addons in cluster.yaml",
		)
	default:
		return ruleMetadata{}, false
	}
}

func nodeMetadata(id string) (ruleMetadata, bool) {
	metadata := map[string]ruleMetadata{
		"CLUSTER-SSH-TRANSPORT": {
			name: "SSH transport to node",
			fix:  "verify nodes.yaml spec.ip, SSH port and network access to the node",
		},
		"CLUSTER-SSH-AUTH": {
			name: "SSH credentials for node",
			fix:  "correct nodes.yaml SSH username/credential reference and verify authentication",
		},
		"CLUSTER-SSH-PRIVILEGE": {
			name: "Root privilege on node",
			fix:  "configure nodes.yaml SSH access as root or non-interactive sudo",
		},
		"CLUSTER-AGENT-SYSTEMD": {
			name: "systemd is available on node",
			fix:  "install/enable systemd on the node",
		},
		"CLUSTER-RES-CPU": {
			name: "Node CPU capacity",
			fix:  "use a node with at least 8 logical CPUs",
		},
		"CLUSTER-RES-MEMORY": {
			name: "Node memory capacity",
			fix:  "use a node with at least 16 GiB memory",
		},
		"CLUSTER-RES-SYSTEM-DISK": {
			name: "Node system disk capacity",
			fix:  "use a node with at least 100 GiB system disk capacity",
		},
		"CLUSTER-PLAT-OS": {
			name: "Node operating system",
			fix:  "use a supported Linux operating system on the node",
		},
		"CLUSTER-PLAT-DISTRO": {
			name: "Node Linux distribution and version",
			fix:  "use a node distribution/version listed in the preflight policy",
		},
		"CLUSTER-PLAT-KERNEL": {
			name: "Node kernel compatibility",
			fix:  "boot a node kernel covered by the release compatibility matrix",
		},
	}
	value, ok := metadata[id]
	return value, ok
}

func humanizeClusterID(id string) string {
	value := strings.TrimPrefix(strings.TrimPrefix(id, "CLUSTER-"), "LEGACY-")
	value = strings.ReplaceAll(strings.ReplaceAll(value, "-", " "), ":", " ")
	lower := strings.ToLower(value)
	if lower == "" {
		return ""
	}
	return strings.ToUpper(lower[:1]) + lower[1:]
}
