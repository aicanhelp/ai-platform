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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	providernode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
)

// The runtime report renders the binding Fix, which overrides the Bundle
// template's Fix. A policy-adjusted resource minimum must therefore re-render
// the binding Fix too; otherwise a 200 GiB disk policy still reports the
// adapter's hardcoded 100 GiB remediation.
func TestResourceBindingFixTracksPolicyMinimum(t *testing.T) {
	policy := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion, Kind: preflightconfig.Kind,
	}
	policy.Operations.ClusterCreate.Resources = []preflightconfig.ResourceCheck{
		{CheckCommon: preflightconfig.CheckCommon{Description: "Node CPU capacity"},
			Key: "cpu", Minimum: 8},
		{CheckCommon: preflightconfig.CheckCommon{Description: "Node memory capacity"},
			Key: "memory", Minimum: 16 * 1024 * 1024 * 1024},
		{CheckCommon: preflightconfig.CheckCommon{Description: "Node system disk capacity"},
			Key: "systemDisk", Minimum: 200 * 1024 * 1024 * 1024},
	}
	policy.Operations.Init.Resources = []preflightconfig.ResourceCheck{
		{CheckCommon: preflightconfig.CheckCommon{Description: "Host system disk capacity"},
			Key: "systemDisk", Minimum: 300 * 1024 * 1024 * 1024},
	}
	require.NoError(t, preflightconfig.Validate(policy))

	cluster := &configv1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{
		Name: "demo", Namespace: "demo", Labels: map[string]string{"cluster.x-k8s.io/cluster-name": "demo"},
	}}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	node := configv1beta1.BKENode{ObjectMeta: metav1.ObjectMeta{Name: "node-1", Namespace: "demo"}, Spec: configv1beta1.BKENodeSpec{
		IP: "192.0.2.11", Hostname: "node-1",
		Role: []string{providernode.MasterWorkerNodeRole, providernode.EtcdNodeRole},
	}}
	input, err := (&ClusterResources{BKECluster: cluster, BKENodes: []configv1beta1.BKENode{node}}).ToAdapterInput()
	require.NoError(t, err)

	resolved, err := NewOpenFuyaoAdapter(&ProviderResolverV122{}, policy).Resolve(context.Background(), input)
	require.NoError(t, err)

	fixes := map[string]string{}
	for _, binding := range resolved.Bindings {
		fixes[binding.TemplateID] = binding.Fix
	}
	assert.Equal(t, "use a node with at least 8 logical CPUs", fixes["CLUSTER-RES-CPU"])
	assert.Equal(t, "use a node with at least 16 GiB memory", fixes["CLUSTER-RES-MEMORY"])
	assert.Equal(t, "use a node with at least 200 GiB system disk capacity", fixes["CLUSTER-RES-SYSTEM-DISK"])
	assert.NotContains(t, fixes["CLUSTER-RES-SYSTEM-DISK"], "100 GiB")
}

// An explicit operator remediation still wins over the dynamic fix.
func TestResourceBindingOperatorRemediationOverridesDynamicFix(t *testing.T) {
	policy := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion, Kind: preflightconfig.Kind,
	}
	policy.Operations.Init.Resources = []preflightconfig.ResourceCheck{
		{CheckCommon: preflightconfig.CheckCommon{
			Description: "Host system disk capacity",
			Remediation: "attach a bigger disk per ops runbook OP-42",
		}, Key: "systemDisk", Minimum: 300 * 1024 * 1024 * 1024},
	}
	require.NoError(t, preflightconfig.Validate(policy))

	resolved := resolveInitWithPolicy(&initadapter.Input{HostIP: "127.0.0.1"}, &policy)
	for _, binding := range resolved.Bindings {
		if binding.TemplateID == "INIT-RES-SYSTEM-DISK" {
			assert.True(t, strings.Contains(binding.Fix, "OP-42"))
			assert.NotContains(t, binding.Fix, "300 GiB")
			return
		}
	}
	t.Fatal("INIT-RES-SYSTEM-DISK binding not found")
}
