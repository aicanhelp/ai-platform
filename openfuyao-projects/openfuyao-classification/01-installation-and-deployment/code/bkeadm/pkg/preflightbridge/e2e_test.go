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
	"bytes"
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	providernode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	preflight "gopkg.openfuyao.cn/preflight"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/render"
)

type recordingPassingCollector struct {
	mu   sync.Mutex
	keys []string
}

func TestInitCustomFileContentRunsThroughCELAndRedactsReport(t *testing.T) {
	config := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion, Kind: preflightconfig.Kind,
	}
	config.Operations.Init.CustomChecks = []preflightconfig.CustomCheck{{
		CheckCommon: preflightconfig.CheckCommon{
			Description: "/etc/hosts loopback", Failure: "fail", Unknown: "fail",
			Remediation: "add a loopback mapping and rerun preflight",
		},
		Key: "hosts-loopback", Expected: "contains 127.0.0.1 or ::1",
		Collector:  preflightconfig.CustomCollector{Type: "fileContent", Path: "/etc/hosts"},
		Expression: `actual.matches("(?m)^\\s*(127\\.0\\.0\\.1|::1)(\\s|$)")`,
	}}
	require.NoError(t, preflightconfig.Validate(config))
	resolved := resolveInitWithPolicy(&initadapter.Input{HostIP: "127.0.0.1"}, &config)
	bundleBytes, err := BuildBundle(config)
	require.NoError(t, err)
	collector, err := NewInitRuntimeCollectorWithCustom(InitOptions{}, nil, config.Operations.Init)
	require.NoError(t, err)
	report, err := preflight.New(preflight.Dependencies{Collector: collector}).Check(
		context.Background(),
		preflight.CheckInput{Resolved: resolved, Bundle: bundleBytes},
	)
	require.NoError(t, err)
	for _, result := range report.Results {
		if result.RuleID != "CONFIG-INIT-CUSTOM-HOSTS-LOOPBACK" {
			continue
		}
		assert.Equal(t, api.StatusPass, result.Status)
		assert.Equal(t, "contains 127.0.0.1 or ::1", result.Expected.Value)
		assert.Equal(t, "已读取（内容不展示）", result.Actual.Value)
		assert.NotContains(t, result.Actual.Value, "localhost")
		assert.Equal(t, "/etc/hosts", result.Evidence[0].Locator)
		var output bytes.Buffer
		require.NoError(t, render.Text(&output, report))
		assert.Contains(t, output.String(), "/etc/hosts")
		return
	}
	t.Fatal("custom check result not found")
}

func (*recordingPassingCollector) ID() string { return "e2e-readonly-fake" }

func (c *recordingPassingCollector) Collect(_ context.Context, request api.FactRequest) ([]api.Fact, error) {
	c.mu.Lock()
	c.keys = append(c.keys, request.Keys...)
	c.mu.Unlock()
	results := make([]api.Fact, 0, len(request.Keys))
	subject := ""
	if len(request.Subjects) > 0 {
		subject = request.Subjects[0].ID
	}
	for _, key := range request.Keys {
		results = append(results, api.Fact{Key: key, Subject: subject, Available: true, Value: passingValue(key)})
	}
	return results, nil
}

func passingValue(key string) any {
	switch {
	case key == "node.cpu.logical":
		return 16
	case key == "node.memory.totalBytes":
		return int64(32 * 1024 * 1024 * 1024)
	case key == "node.systemDisk.totalBytes":
		return int64(200 * 1000 * 1000 * 1000)
	case strings.HasPrefix(key, "platform.os."):
		return "linux"
	case strings.HasPrefix(key, "platform.distribution."):
		return []any{"openeuler", "22.03"}
	case strings.HasPrefix(key, "node.artifact.arch."):
		return "amd64"
	case strings.HasSuffix(key, ".mountReadOnly"):
		return false
	default:
		return true
	}
}

func (c *recordingPassingCollector) requested(fragment string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, key := range c.keys {
		if strings.Contains(key, fragment) {
			return true
		}
	}
	return false
}

func TestClusterCreateEndToEndUsesDynamicProviderPlan(t *testing.T) {
	cluster := &configv1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "demo"}}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	cluster.Spec.ClusterConfig.Cluster.APIServer = &configv1beta1.APIServer{APIEndpoint: configv1beta1.APIEndpoint{Port: 7443}}
	cluster.Spec.ControlPlaneEndpoint.Host = "192.0.2.100"
	node := configv1beta1.BKENode{ObjectMeta: metav1.ObjectMeta{
		Name: "node-1", Namespace: "demo", Labels: map[string]string{"cluster.x-k8s.io/cluster-name": "demo"},
	}, Spec: configv1beta1.BKENodeSpec{
		IP: "192.0.2.11", Hostname: "node-1", Role: []string{providernode.MasterWorkerNodeRole, providernode.EtcdNodeRole},
	}}
	resources := &ClusterResources{BKECluster: cluster, BKENodes: []configv1beta1.BKENode{node}}
	require.NoError(t, ApplyProviderManagementDefaults(resources, map[string]string{
		"host": "192.0.2.20", "imageRepoPort": "5443", "yumRepoPort": "8080", "imageRepoPrefix": "openfuyao",
	}))
	input, err := resources.ToAdapterInput()
	require.NoError(t, err)
	collector := &recordingPassingCollector{}
	resolver := &ProviderResolverV122{}
	resolved, err := NewOpenFuyaoAdapter(resolver).Resolve(context.Background(), input)
	require.NoError(t, err)
	bundleBytes, err := BuildBundle(preflightconfig.Config{APIVersion: preflightconfig.APIVersion, Kind: preflightconfig.Kind, Metadata: preflightconfig.Metadata{}})
	require.NoError(t, err)
	checker := preflight.New(preflight.Dependencies{
		Collector: collector,
	})
	report, err := checker.Check(context.Background(), preflight.CheckInput{Resolved: resolved, Bundle: bundleBytes})
	require.NoError(t, err)
	for _, result := range report.Results {
		if result.Status == api.StatusFail {
			t.Logf("unexpected FAIL: %s expected=%v actual=%v cause=%s", result.RuleID, result.Expected.Value, result.Actual.Value, result.CauseCode)
		}
	}
	assert.Equal(t, api.OverallPass, report.Overall, "ordinary master-worker roles are not proof of bootstrap/HA colocation")
	for _, result := range report.Results {
		assert.NotEqual(t, "CLUSTER-HA-COLOCATE", result.RuleID)
	}
	assert.True(t, collector.requested("port.7443.free"), "custom Provider API port must drive the node plan")
	assert.True(t, collector.requested(":5443"), "management bke-config defaults must drive source probes")
	t.Logf("dynamic plan verified: apiPort=7443 managementRepoPort=5443 overall=%s results=%d", report.Overall, len(report.Results))
}

func TestClusterCreateInvalidPodSubnetIsActionableInFinalReport(t *testing.T) {
	cluster := &configv1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "demo"}}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	cluster.Spec.ClusterConfig.Cluster.Networking.PodSubnet = "not-a-cidr"
	node := configv1beta1.BKENode{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}, Spec: configv1beta1.BKENodeSpec{
		IP: "192.0.2.11", Hostname: "node-1",
		Role: []string{providernode.MasterWorkerNodeRole, providernode.EtcdNodeRole},
	}}
	input, err := (ClusterResources{BKECluster: cluster, BKENodes: []configv1beta1.BKENode{node}}).ToAdapterInput()
	require.NoError(t, err)
	resolver := &ProviderResolverV122{}
	resolved, err := NewOpenFuyaoAdapter(resolver).Resolve(context.Background(), input)
	require.NoError(t, err)
	bundleBytes, err := BuildBundle(preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion, Kind: preflightconfig.Kind,
	})
	require.NoError(t, err)
	report, err := preflight.New(preflight.Dependencies{
		Collector: &recordingPassingCollector{},
	}).Check(context.Background(), preflight.CheckInput{
		Resolved: resolved, Bundle: bundleBytes,
	})
	require.NoError(t, err)
	assertReportHasInputField(t, report, "cluster.spec.clusterConfig.cluster.networking.podSubnet")
	assertSingleFieldFailure(
		t,
		report,
		"cluster.spec.clusterConfig.cluster.networking.podSubnet",
		"CLUSTER-NET-POD-CIDR",
	)
	var output bytes.Buffer
	require.NoError(t, render.Summary(&output, report, ""))
	assert.Contains(t, output.String(), "字段：podSubnet")
	assert.Contains(t, output.String(), "not-a-cidr")
}

func assertSingleFieldFailure(t *testing.T, report api.Report, field, ruleID string) {
	t.Helper()
	var failures []api.Result
	for _, result := range report.Results {
		if result.Status != api.StatusFail {
			continue
		}
		for _, evidence := range result.Evidence {
			if evidence.Locator == field {
				failures = append(failures, result)
				break
			}
		}
	}
	require.Len(t, failures, 1)
	assert.Equal(t, ruleID, failures[0].RuleID)
}

func assertReportHasInputField(t *testing.T, report api.Report, field string) {
	t.Helper()
	for _, result := range report.Results {
		for _, evidence := range result.Evidence {
			if evidence.Locator == field {
				return
			}
		}
	}
	t.Fatalf("report does not contain input field %q", field)
}

func TestInitToolAndPathExtensionChecksRunEndToEnd(t *testing.T) {
	config := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion, Kind: preflightconfig.Kind,
	}
	config.Operations.Init.Tools = append(config.Operations.Init.Tools,
		preflightconfig.ToolCheck{
			CheckCommon: preflightconfig.CheckCommon{Description: "curl executable"},
			Key:         "curl", Names: []string{"curl"},
		})
	dir := t.TempDir()
	config.Operations.Init.Paths = append(config.Operations.Init.Paths,
		preflightconfig.PathCheck{
			CheckCommon: preflightconfig.CheckCommon{Description: "extension dir exists"},
			Key:         "ext-dir", Path: dir,
		})
	require.NoError(t, preflightconfig.Validate(config))
	resolved := resolveInitWithPolicy(&initadapter.Input{HostIP: "127.0.0.1"}, &config)
	var toolBinding, pathBinding *api.RuleBinding
	for i := range resolved.Bindings {
		switch resolved.Bindings[i].TemplateID {
		case "CONFIG-INIT-TOOL-CURL":
			toolBinding = &resolved.Bindings[i]
		case "CONFIG-INIT-PATH-EXT-DIR":
			pathBinding = &resolved.Bindings[i]
		}
	}
	require.NotNil(t, toolBinding, "tool extension binding missing")
	require.NotNil(t, pathBinding, "path extension binding missing")

	bundleBytes, err := BuildBundle(config)
	require.NoError(t, err)
	collector := NewInitRuntimeCollector(InitOptions{})
	report, err := preflight.New(preflight.Dependencies{Collector: collector}).Check(
		context.Background(),
		preflight.CheckInput{Resolved: resolved, Bundle: bundleBytes},
	)
	require.NoError(t, err)
	statuses := map[string]api.Status{}
	for _, result := range report.Results {
		statuses[result.RuleID] = result.Status
	}
	// The test binary runs in an environment where curl exists and the temp
	// dir exists, so both extension checks must evaluate as real facts.
	assert.Equal(t, api.StatusPass, statuses["CONFIG-INIT-TOOL-CURL"])
	assert.Equal(t, api.StatusPass, statuses["CONFIG-INIT-PATH-EXT-DIR"])
}
