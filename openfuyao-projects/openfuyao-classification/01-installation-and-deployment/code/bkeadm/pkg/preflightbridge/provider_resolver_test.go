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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	providernode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/preflight/api"
)

func TestProviderResolverContractVersion(t *testing.T) {
	r := &ProviderResolverV122{}
	assert.Equal(t, "preflight-resolver/v1", r.ContractVersion())
}

func TestProviderResolverTopology(t *testing.T) {
	r := &ProviderResolverV122{}
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	cluster.Spec.ControlPlaneEndpoint.Host = ""
	nodes := []configv1beta1.BKENode{
		{Spec: configv1beta1.BKENodeSpec{IP: "192.0.2.10", Hostname: "master-01", Role: []string{providernode.MasterWorkerNodeRole, providernode.EtcdNodeRole}}},
		{Spec: configv1beta1.BKENodeSpec{IP: "192.0.2.11", Hostname: "worker-01", Role: []string{providernode.WorkerNodeRole}}},
	}
	nodes[0].Name = "master-01"
	nodes[1].Name = "worker-01"
	res, err := r.Resolve(context.Background(), cluster, nodes)
	require.NoError(t, err)
	assert.Equal(t, 1, res.MasterCount)
	assert.Equal(t, 1, res.EtcdCount)
	assert.Equal(t, 2, res.WorkerCount)
	assert.False(t, res.EndpointExplicit)
	assert.Equal(t, configinit.DefaultAPIBindPort, res.APIBindPort)
	assert.Len(t, res.NodeRoles, 2)
	assert.Equal(t, "192.0.2.10", res.NodeAddresses["master-01"])
}

func TestProviderResolverNilCluster(t *testing.T) {
	r := &ProviderResolverV122{}
	_, err := r.Resolve(context.Background(), nil, nil)
	require.Error(t, err)
}

func TestProviderResolverExplicitEndpoint(t *testing.T) {
	r := &ProviderResolverV122{}
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	cluster.Spec.ControlPlaneEndpoint.Host = "vip.example.com"
	res, _ := r.Resolve(context.Background(), cluster, nil)
	assert.True(t, res.EndpointExplicit)
}

func TestExplicitClusterAPIPortOverridesPolicyFallback(t *testing.T) {
	policy := preflightconfig.Config{Operations: preflightconfig.Operations{
		ClusterCreate: preflightconfig.OperationPolicy{Ports: []preflightconfig.PortCheck{{
			Key: "apiServer", DefaultPort: 8443,
		}}},
	}}
	resolver := &ProviderResolverV122{Policy: &policy}
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}

	defaulted, err := resolver.Resolve(context.Background(), cluster, nil)
	require.NoError(t, err)
	assert.Equal(t, 8443, defaulted.APIBindPort)

	cluster.Spec.ClusterConfig.Cluster.APIServer = &configv1beta1.APIServer{
		APIEndpoint: configv1beta1.APIEndpoint{Port: 7443},
	}
	explicit, err := resolver.Resolve(context.Background(), cluster, nil)
	require.NoError(t, err)
	assert.Equal(t, 7443, explicit.APIBindPort)
}

func TestProviderResolverReportsMissingRoleAsBusinessFinding(t *testing.T) {
	r := &ProviderResolverV122{}
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	nodes := []configv1beta1.BKENode{
		{Spec: configv1beta1.BKENodeSpec{IP: "192.0.2.10"}}, // no role -> worker
	}
	res, _ := r.Resolve(context.Background(), cluster, nodes)
	assert.Empty(t, res.Issues)
	assert.False(t, res.NodesValid)
	assert.Equal(t, 0, res.WorkerCount)
	assert.Equal(t, "nodes[*].spec.role", res.ValidationFields["local.node.valid"].Field)
}

func TestProviderValidationFindingCarriesClusterFieldPathIntoFact(t *testing.T) {
	r := &ProviderResolverV122{}
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	cluster.Spec.ClusterConfig.Cluster.ContainerRuntime.CRI = "unsupported"
	resolved, err := r.Resolve(context.Background(), cluster, nil)
	require.NoError(t, err)
	require.False(t, resolved.ClusterValid)
	finding := resolved.ValidationFields["local.cluster.valid"]
	assert.Equal(t, "cluster.spec.clusterConfig.cluster.containerRuntime.cri", finding.Field)
	assert.NotEmpty(t, finding.Cause)

	facts := clusterPlanFacts(nil, resolved)
	for _, fact := range facts {
		if fact.Key == "local.cluster.valid" {
			require.Len(t, fact.Evidence, 1)
			assert.Equal(t, finding.Field, fact.Evidence[0].Locator)
			assert.Equal(t, finding.Cause, fact.Evidence[0].Observed)
			return
		}
	}
	t.Fatal("local.cluster.valid fact is missing")
}

func TestProviderValidationFindingIdentifiesInvalidPodSubnet(t *testing.T) {
	r := &ProviderResolverV122{}
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	cluster.Spec.ClusterConfig.Cluster.Networking.PodSubnet = "not-a-cidr"

	resolved, err := r.Resolve(context.Background(), cluster, nil)
	require.NoError(t, err)
	require.False(t, resolved.ClusterValid)
	finding := resolved.ValidationFields["local.cluster.valid"]
	assert.Equal(t, "cluster.spec.clusterConfig.cluster.networking.podSubnet", finding.Field)
	assert.Contains(t, finding.Cause, "not-a-cidr")

	facts := clusterPlanFacts(nil, resolved)
	for _, fact := range facts {
		if fact.Key != "local.cluster.valid" {
			continue
		}
		require.Len(t, fact.Evidence, 1)
		assert.Equal(t, finding.Field, fact.Evidence[0].Locator)
		assert.Equal(t, finding.Cause, fact.Evidence[0].Observed)
		return
	}
	t.Fatal("local.cluster.valid fact is missing")
}

func TestProviderSourcesExcludeUnconfiguredEndpoints(t *testing.T) {
	cluster := configv1beta1.Cluster{}
	cluster.ImageRepo.Domain = "images.example.com"
	cluster.ImageRepo.Port = "443"
	assert.Equal(t, []api.Source{{
		ID: "image-repo", Kind: "image", Endpoint: "images.example.com:443", Scheme: "https",
	}}, providerSources(cluster))
	assert.Empty(t, providerSources(configv1beta1.Cluster{}))
}

func TestProviderSourcesDeduplicateTCPDestination(t *testing.T) {
	cluster := configv1beta1.Cluster{}
	cluster.ImageRepo.Domain, cluster.ImageRepo.Port = "repo.example.com", "443"
	cluster.ChartRepo.Domain, cluster.ChartRepo.Port = "repo.example.com", "443"
	assert.Len(t, providerSources(cluster), 1)
}

func TestProviderSourceKeepsConfiguredIPAsResolveAddress(t *testing.T) {
	cluster := configv1beta1.Cluster{}
	cluster.ImageRepo.Domain = "images.example.com"
	cluster.ImageRepo.Ip = "192.0.2.10"
	cluster.ImageRepo.Port = "443"
	sources := providerSources(cluster)
	require.Len(t, sources, 1)
	assert.Equal(t, "images.example.com:443", sources[0].Endpoint)
	assert.Equal(t, "192.0.2.10:443", sources[0].Address)

	// IP-only configuration keeps the IP as the endpoint, no split address.
	ipOnly := configv1beta1.Cluster{}
	ipOnly.ImageRepo.Ip = "192.0.2.11"
	ipOnly.ImageRepo.Port = "40443"
	ipSources := providerSources(ipOnly)
	require.Len(t, ipSources, 1)
	assert.Equal(t, "192.0.2.11:40443", ipSources[0].Endpoint)
	assert.Empty(t, ipSources[0].Address)
}
