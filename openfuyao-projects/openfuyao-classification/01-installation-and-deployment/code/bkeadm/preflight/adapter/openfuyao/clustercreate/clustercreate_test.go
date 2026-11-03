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

package clustercreate

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

func mkResolved() *ResolvedCluster {
	return &ResolvedCluster{
		APIBindPort: 6443, MasterCount: 1, EtcdCount: 1, WorkerCount: 1,
		EndpointExplicit: false, PodSubnet: "10.244.0.0/16", ServiceSubnet: "10.96.0.0/12",
		Runtime: "containerd", EtcdDataDir: "/var/lib/openFuyao/etcd",
		KubeletRoot: "/var/lib/kubelet", DataRoot: "/var/lib/containerd",
		NodeRoles: map[string][]string{
			"master-01": {"master", "etcd", "worker"},
			"worker-01": {"worker"},
		},
		NodeAddresses: map[string]string{
			"master-01": "192.0.2.10",
			"worker-01": "192.0.2.11",
		},
	}
}

func TestResolveBuildsPlan(t *testing.T) {
	rc := mkResolved()
	plan, diags := Resolve(rc)
	require.Empty(t, diags)
	assert.Equal(t, api.OperationClusterCreate, plan.Operation)
	assert.True(t, strings.HasPrefix(plan.PlanHash.String(), "sha256:"))
	// Management/local subject plus node subjects, sorted by ID.
	assert.Len(t, plan.Subjects, 3)
	assert.Equal(t, "cluster", plan.Subjects[0].ID)
	assert.Equal(t, "master-01", plan.Subjects[1].ID)
	assert.Equal(t, "worker-01", plan.Subjects[2].ID)
	// Values.
	assert.Equal(t, 6443, plan.Values["cluster.apiPort"].Value)
	assert.Equal(t, 1, plan.Values["cluster.topology.masterCount"].Value)
	assert.Equal(t, "containerd", plan.Values["cluster.runtime"].Value)
}

func TestResolveNilInput(t *testing.T) {
	plan, diags := Resolve(nil)
	// Nil input: plan is zero-value (empty operation), with a startup issue.
	assert.Empty(t, plan.Operation)
	require.Len(t, diags, 1)
	assert.Equal(t, api.IssueError, diags[0].Severity)
}

func TestResolveDeterministicPlanHash(t *testing.T) {
	rc := mkResolved()
	p1, _ := Resolve(rc)
	p2, _ := Resolve(rc)
	assert.Equal(t, p1.PlanHash, p2.PlanHash)
}

func TestInstancesTopology(t *testing.T) {
	rc := mkResolved()
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	ids := instanceIDs(insts)
	// Topology checks always present.
	for _, id := range []string{"CLUSTER-TOPO-MASTER", "CLUSTER-TOPO-MASTER-ODD",
		"CLUSTER-TOPO-ETCD", "CLUSTER-TOPO-WORKER"} {
		assert.True(t, ids[id], "missing %s", id)
	}
	// Network checks.
	assert.True(t, ids["CLUSTER-NET-POD-CIDR"])
	assert.True(t, ids["CLUSTER-NET-SERVICE-CIDR"])
	assert.False(t, ids["CLUSTER-UNIQUE-NAME"], "Provider has no standalone node-name preflight rule")
	assert.False(t, ids["CLUSTER-HA-COLOCATE"], "unknown bootstrap identity must not create an HA advisory")
}

func TestHAColocationRequiresProvenBootstrapMaster(t *testing.T) {
	rc := mkResolved()
	rc.MasterCount = 3
	rc.BootstrapNodeID = "master-01"
	plan, diags := Resolve(rc)
	require.Empty(t, diags)
	ids := instanceIDs(Instances(plan))
	assert.True(t, ids["CLUSTER-HA-COLOCATE"])
}

func TestNodeChecksAreGatedBySSHReadiness(t *testing.T) {
	plan, diags := Resolve(mkResolved())
	require.Empty(t, diags)
	byID := map[string]api.RuleTemplate{}
	for _, instance := range Instances(plan) {
		if _, exists := byID[instance.Rule.ID]; !exists {
			byID[instance.Rule.ID] = instance.Rule
		}
	}
	assert.Empty(t, byID["CLUSTER-SSH-TRANSPORT"].DependsOn)
	assert.Equal(t, []string{"CLUSTER-SSH-TRANSPORT"}, byID["CLUSTER-SSH-AUTH"].DependsOn)
	assert.Equal(t, []string{"CLUSTER-SSH-AUTH"}, byID["CLUSTER-SSH-PRIVILEGE"].DependsOn)
	assert.Equal(t, []string{"CLUSTER-SSH-PRIVILEGE"}, byID["CLUSTER-AGENT-SYSTEMD"].DependsOn)
}

func TestManagementChecksFormOneRootCauseChain(t *testing.T) {
	plan, diags := Resolve(mkResolved())
	require.Empty(t, diags)
	byID := map[string]api.RuleTemplate{}
	for _, instance := range Instances(plan) {
		byID[instance.Rule.ID] = instance.Rule
	}
	assert.Empty(t, byID["CLUSTER-MGMT-API"].DependsOn)
	assert.Equal(t, []string{"CLUSTER-MGMT-API"}, byID["CLUSTER-MGMT-DISCOVERY"].DependsOn)
	assert.Equal(t, []string{"CLUSTER-MGMT-DISCOVERY"}, byID["CLUSTER-MGMT-CRD-CLUSTER"].DependsOn)
	assert.Equal(t, []string{"CLUSTER-MGMT-API"}, byID["CLUSTER-MGMT-AGENT-KUBECONFIG-GET"].DependsOn)
	assert.Equal(t, []string{"CLUSTER-MGMT-AGENT-KUBECONFIG-GET"}, byID["CLUSTER-MGMT-AGENT-KUBECONFIG-DATA"].DependsOn)
	assert.Equal(t, []string{"CLUSTER-MGMT-AGENT-KUBECONFIG-DATA"}, byID["CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE"].DependsOn)
}

func TestSystemDiskThresholdUses100GiB(t *testing.T) {
	plan, diags := Resolve(mkResolved())
	require.Empty(t, diags)
	for _, instance := range Instances(plan) {
		if instance.Rule.ID == "CLUSTER-RES-SYSTEM-DISK" {
			assert.Equal(t, int64(100*1024*1024*1024), instance.Rule.Expected)
			return
		}
	}
	t.Fatal("system disk check missing")
}

func TestSourceInstancesHaveReadableUniqueNamesPerNode(t *testing.T) {
	rc := mkResolved()
	rc.Sources = []api.Source{
		{ID: "image-repo", Kind: "image", Endpoint: "images.example.com:5443", Scheme: "https", ObjectPath: "/v2/"},
		{ID: "chart-repo", Kind: "chart", Endpoint: "charts.example.com:38080", Scheme: "http"},
	}
	plan, diags := Resolve(rc)
	require.Empty(t, diags)
	seen := map[string]bool{}
	for _, instance := range Instances(plan) {
		if !strings.HasPrefix(instance.Rule.ID, "CLUSTER-SRC-") {
			continue
		}
		assert.NotEmpty(t, instance.Rule.Name)
		assert.NotContains(t, instance.Rule.Name, "Src ")
		assert.Contains(t, instance.Rule.Name, "(")
		assert.Contains(t, instance.Rule.Fact, "endpoint.tcp.")
		assert.NotContains(t, instance.Rule.Fix, "node resources")
		key := instance.Rule.Name + " — " + instance.Subject
		assert.False(t, seen[key], "duplicate source check name %q", key)
		seen[key] = true
	}
	assert.Contains(t, seen, "Image repository is accessible (images.example.com:5443) — master-01")
	assert.Contains(t, seen, "Chart repository is accessible (charts.example.com:38080) — worker-01")
}

func TestEveryClusterInstanceHasActionableMetadata(t *testing.T) {
	rc := mkResolved()
	rc.Sources = []api.Source{{ID: "image-repo", Kind: "image", Endpoint: "images.example.com:5443", Scheme: "https"}}
	plan, diags := Resolve(rc)
	require.Empty(t, diags)
	for _, instance := range Instances(plan) {
		assert.NotEmpty(t, instance.Rule.Name, instance.Rule.ID)
		assert.NotEmpty(t, instance.Rule.Fix, instance.Rule.ID)
		assert.NotContains(t, strings.ToLower(instance.Rule.Fix), "see provider", instance.Rule.ID)
	}
}

func TestInstancesPerNodePorts(t *testing.T) {
	rc := mkResolved()
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	// master-01 has master+etcd+worker roles: should get apiserver, etcd client/peer,
	// kubelet secure/health, bkeagent ports.
	masterPorts := subjectInstances(insts, "master-01")
	masterIDs := instanceIDs(masterPorts)
	assert.True(t, masterIDs["CLUSTER-PORT-APISERVER"])
	assert.True(t, masterIDs["CLUSTER-PORT-ETCD-CLIENT"])
	assert.True(t, masterIDs["CLUSTER-PORT-ETCD-PEER"])
	assert.True(t, masterIDs["CLUSTER-PORT-KUBELET-SECURE"])
	assert.True(t, masterIDs["CLUSTER-PORT-KUBELET-HEALTH"])
	assert.True(t, masterIDs["CLUSTER-PORT-BKEAGENT"])

	// worker-01 has only worker role: apiserver NOT present, kubelet present.
	workerPorts := subjectInstances(insts, "worker-01")
	workerIDs := instanceIDs(workerPorts)
	assert.False(t, workerIDs["CLUSTER-PORT-APISERVER"])
	assert.True(t, workerIDs["CLUSTER-PORT-KUBELET-SECURE"])
	assert.True(t, workerIDs["CLUSTER-PORT-BKEAGENT"])
	// etcd ports NOT present for worker-only.
	assert.False(t, workerIDs["CLUSTER-PORT-ETCD-CLIENT"])
}

func TestInstancesPortNumbersInFactKey(t *testing.T) {
	rc := mkResolved()
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	// APISERVER port should be 6443 (from plan.Values cluster.apiPort).
	var apiInst *ClusterInstance
	for i := range insts {
		if insts[i].Rule.ID == "CLUSTER-PORT-APISERVER" {
			apiInst = &insts[i]
			break
		}
	}
	require.NotNil(t, apiInst)
	assert.Contains(t, apiInst.FactKey, "port.6443.free")
	// ETCD client should be 2379.
	var etcdClient *ClusterInstance
	for i := range insts {
		if insts[i].Rule.ID == "CLUSTER-PORT-ETCD-CLIENT" {
			etcdClient = &insts[i]
			break
		}
	}
	require.NotNil(t, etcdClient)
	assert.Contains(t, etcdClient.FactKey, "port.2379.free")
}

func TestInstancesBKEAgentAllNodes(t *testing.T) {
	rc := mkResolved()
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	// BKEAgent port 58080 on every node.
	agentCount := 0
	for _, in := range insts {
		if in.Rule.ID == "CLUSTER-PORT-BKEAGENT" {
			agentCount++
			assert.Contains(t, in.FactKey, "port.58080.free")
		}
	}
	assert.Equal(t, 2, agentCount)
}

func TestInstancesEmptyCluster(t *testing.T) {
	rc := &ResolvedCluster{NodeRoles: map[string][]string{}}
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	// No nodes: only cluster-level topology checks.
	for _, in := range insts {
		assert.Equal(t, "cluster", in.Subject)
	}
}

// TestInstancesCoreDomains freezes the intentionally small cluster-create
// check set. Installation-created node paths and ambient node state are not
// part of preflight.
func TestInstancesAll35Domains(t *testing.T) {
	rc := mkResolved()
	rc.MasterCount = 3
	rc.BootstrapNodeID = "master-01"
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	ids := make(map[string]bool)
	for _, i := range insts {
		ids[i.Rule.ID] = true
		ids[i.InstanceID] = true
	}

	// The 35 logical check items from the design, mapped to representative
	// RuleIDs. Each must be present.
	required := []string{
		// 1. Provider-backed address/optional-hostname uniqueness
		"CLUSTER-UNIQUE-IP", "CLUSTER-UNIQUE-HOSTNAME",
		// 2. Topology
		"CLUSTER-TOPO-MASTER", "CLUSTER-TOPO-MASTER-ODD", "CLUSTER-TOPO-ETCD", "CLUSTER-TOPO-WORKER",
		// 3. Management cluster API
		"CLUSTER-MGMT-API", "CLUSTER-MGMT-DISCOVERY", "CLUSTER-MGMT-CRD-CLUSTER", "CLUSTER-MGMT-CRD-NODE",
		// 4. Agent kubeconfig
		"CLUSTER-MGMT-AGENT-KUBECONFIG-GET",
		"CLUSTER-MGMT-AGENT-KUBECONFIG-DATA", "CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE",
		// 5. Pre-submission validation
		"CLUSTER-MGMT-LOCAL-CLUSTER", "CLUSTER-MGMT-LOCAL-NODE",
		// 6. SSH connection
		"CLUSTER-SSH-TRANSPORT",
		"CLUSTER-SSH-AUTH", "CLUSTER-SSH-PRIVILEGE",
		// 7. systemd and CPU/memory/disk/OS/distro/kernel/arch (per node)
		"CLUSTER-AGENT-SYSTEMD",
		"CLUSTER-RES-CPU", "CLUSTER-RES-MEMORY", "CLUSTER-RES-SYSTEM-DISK",
		"CLUSTER-PLAT-OS", "CLUSTER-PLAT-DISTRO", "CLUSTER-PLAT-KERNEL",
		// 8. BKE local path
		"CLUSTER-LOCAL-PATH-CREATE", "CLUSTER-LOCAL-PATH-ACCESS", "CLUSTER-LOCAL-PATH-MOUNT",
		// 9. Ports (per node/role)
		"CLUSTER-PORT-APISERVER", "CLUSTER-PORT-ETCD-CLIENT", "CLUSTER-PORT-ETCD-PEER",
		"CLUSTER-PORT-KUBELET-SECURE", "CLUSTER-PORT-KUBELET-HEALTH",
		"CLUSTER-PORT-CONTROLLER", "CLUSTER-PORT-SCHEDULER", "CLUSTER-PORT-BKEAGENT",
		// 10. Node-to-source comms (per source per node, if sources present)
		// 11. Pod/Service CIDR + overlap
		"CLUSTER-NET-POD-CIDR", "CLUSTER-NET-SERVICE-CIDR",
		// 12. HA colocation warning
		"CLUSTER-HA-COLOCATE",
		// 13. Custom refs/addons
		"CLUSTER-REF-VERSION",
		"CLUSTER-ADDON-CLOSURE", "CLUSTER-ADDON-SOURCE",
		// 14. Agent kubeconfig source
		"CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE",
	}
	for _, id := range required {
		assert.True(t, ids[id], "missing required check: %s", id)
	}
}

// TestInstancesRuleCount freezes the minimum rule count for a 2-node cluster.
func TestInstancesRuleCount(t *testing.T) {
	rc := mkResolved()
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	// 2 nodes with the minimal per-node and cluster-level checks.
	assert.GreaterOrEqual(t, len(insts), 45, "2-node cluster should produce >=45 atomic rules, got %d", len(insts))
}

// TestInstancesMetricsPortConditional verifies etcd metrics port appears only
// for etcd nodes.
func TestInstancesMetricsPortConditional(t *testing.T) {
	rc := mkResolved()
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	// master-01 has etcd role → should have ETCD-METRICS.
	masterInsts := subjectInstances(insts, "master-01")
	masterIDs := instanceIDs(masterInsts)
	assert.True(t, masterIDs["CLUSTER-PORT-ETCD-METRICS"])
	// worker-01 has no etcd role → should NOT have ETCD-METRICS.
	workerInsts := subjectInstances(insts, "worker-01")
	workerIDs := instanceIDs(workerInsts)
	assert.False(t, workerIDs["CLUSTER-PORT-ETCD-METRICS"])
}

// TestInstancesControllerSchedulerMasterOnly verifies controller/scheduler ports
// appear only for master nodes.
func TestInstancesControllerSchedulerMasterOnly(t *testing.T) {
	rc := mkResolved()
	plan, _ := Resolve(rc)
	insts := Instances(plan)
	masterIDs := instanceIDs(subjectInstances(insts, "master-01"))
	workerIDs := instanceIDs(subjectInstances(insts, "worker-01"))
	assert.True(t, masterIDs["CLUSTER-PORT-CONTROLLER"])
	assert.True(t, masterIDs["CLUSTER-PORT-SCHEDULER"])
	assert.False(t, workerIDs["CLUSTER-PORT-CONTROLLER"])
	assert.False(t, workerIDs["CLUSTER-PORT-SCHEDULER"])
}

// --- helpers ---

func instanceIDs(insts []ClusterInstance) map[string]bool {
	m := make(map[string]bool, len(insts))
	for _, i := range insts {
		m[i.Rule.ID] = true
	}
	return m
}

func subjectInstances(insts []ClusterInstance, subject string) []ClusterInstance {
	var out []ClusterInstance
	for _, i := range insts {
		if i.Subject == subject {
			out = append(out, i)
		}
	}
	return out
}

func TestCatalogRulesAreValidAndUnique(t *testing.T) {
	rules := CatalogRules()
	require.NotEmpty(t, rules)
	seen := map[string]bool{}
	for _, rule := range rules {
		assert.Nil(t, rule.Validate())
		assert.False(t, seen[rule.ID], "duplicate rule %s", rule.ID)
		seen[rule.ID] = true
		assert.Equal(t, api.OperationClusterCreate, rule.Operation)
	}
}

func TestNodeResourceFixTracksExpectedValues(t *testing.T) {
	plan, diags := Resolve(mkResolved())
	require.Empty(t, diags)
	fixes := map[string]string{}
	for _, instance := range Instances(plan) {
		switch instance.Rule.ID {
		case "CLUSTER-RES-CPU", "CLUSTER-RES-MEMORY", "CLUSTER-RES-SYSTEM-DISK":
			fixes[instance.Rule.ID] = instance.Rule.Fix
		}
	}
	assert.Equal(t, "use a node with at least 8 logical CPUs", fixes["CLUSTER-RES-CPU"])
	assert.Equal(t, "use a node with at least 16 GiB memory", fixes["CLUSTER-RES-MEMORY"])
	assert.Equal(t, "use a node with at least 100 GiB system disk capacity", fixes["CLUSTER-RES-SYSTEM-DISK"])

	// The fix must derive from the expected value, not from a static string.
	assert.Contains(t, fixes["CLUSTER-RES-MEMORY"], fmt.Sprintf("%d", 16))
	assert.Contains(t, fixes["CLUSTER-RES-SYSTEM-DISK"], fmt.Sprintf("%d", 100))
}

func TestSourceNodeInstanceEncodesConfiguredIPInFactKey(t *testing.T) {
	withIP := sourceNodeInstance(api.Source{
		ID: "image-repo", Kind: "image", Endpoint: "images.example.com:443",
		Address: "192.0.2.10:443", Scheme: "https",
	}, "node-1")
	// The fact key carries the resolve override so the node-side probe
	// connects to the configured IP while the domain stays Host/SNI.
	assert.Equal(t, "endpoint.tcp.https://images.example.com:443@192.0.2.10:443", withIP.FactKey)
	assert.Contains(t, withIP.Rule.Name, "images.example.com:443")
	assert.NotContains(t, withIP.Rule.Name, "192.0.2.10")

	withoutIP := sourceNodeInstance(api.Source{
		ID: "image-repo", Kind: "image", Endpoint: "images.example.com:443", Scheme: "https",
	}, "node-1")
	assert.Equal(t, "endpoint.tcp.https://images.example.com:443", withoutIP.FactKey)
}
