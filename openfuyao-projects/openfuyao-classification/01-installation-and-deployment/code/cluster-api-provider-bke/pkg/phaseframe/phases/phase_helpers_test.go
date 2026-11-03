/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phases

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
)

func TestFetchBKENodesIfCPInitialized_RefreshError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        createTestLogger(),
	}

	patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext) error {
		return assert.AnError
	})

	nodes, ok := fetchBKENodesIfCPInitialized(ctx, bkeCluster)
	assert.True(t, ok)
	assert.NotNil(t, nodes)
}

func TestFetchBKENodesIfCPInitialized_CPNotInitialized(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	cluster := &clusterv1.Cluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster, cluster).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Cluster:    cluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        createTestLogger(),
	}

	patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext) error {
		return nil
	})

	patches.ApplyFunc(conditions.IsTrue, func(getter conditions.Getter, conditionType clusterv1.ConditionType) bool {
		return false
	})

	nodes, ok := fetchBKENodesIfCPInitialized(ctx, bkeCluster)
	assert.False(t, ok)
	assert.Nil(t, nodes)
}

func TestGetDeleteTargetNodesIfDeployed_NotDeployed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	cluster := &clusterv1.Cluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster, cluster).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Cluster:    cluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        createTestLogger(),
	}

	patches.ApplyFunc(phaseutil.ClusterEndDeployedWithContext, func(ctx context.Context, c any, cluster *clusterv1.Cluster, bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bool {
		return false
	})

	nodes, ok := getDeleteTargetNodesIfDeployed(ctx, bkeCluster)
	assert.False(t, ok)
	assert.Nil(t, nodes)
}
