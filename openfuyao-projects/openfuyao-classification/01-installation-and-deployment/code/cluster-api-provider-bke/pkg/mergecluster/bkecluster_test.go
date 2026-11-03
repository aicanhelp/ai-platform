/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package mergecluster

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

func setupTestScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = v1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	return scheme
}

func createTestBKECluster(name, namespace string) *v1beta1.BKECluster {
	return &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
		Status: confv1beta1.BKEClusterStatus{},
	}
}

func createTestConfigMap(name, namespace string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string]string{
			"nodes":  "[]",
			"status": "[]",
		},
	}
}

func TestBkeNodes_toCMData(t *testing.T) {
	tests := []struct {
		name    string
		nodes   *bkeNodes
		wantErr bool
	}{
		{
			name:    "nil spec",
			nodes:   &bkeNodes{spec: nil},
			wantErr: false,
		},
		{
			name:    "empty spec",
			nodes:   &bkeNodes{spec: []confv1beta1.Node{}},
			wantErr: false,
		},
		{
			name: "with nodes",
			nodes: &bkeNodes{spec: []confv1beta1.Node{
				{IP: "192.168.1.1", Hostname: "node1"},
			}},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := tt.nodes.toCMData()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, data)
			}
		})
	}
}

func TestGetCombinedBKECluster(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test-cluster", "default")
	cm := createTestConfigMap("test-cluster", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	result, err := GetCombinedBKECluster(context.Background(), fakeClient, "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "test-cluster", result.Name)
}

func TestGetCombinedBKECluster_NotFound(t *testing.T) {
	scheme := setupTestScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	_, err := GetCombinedBKECluster(context.Background(), fakeClient, "default", "nonexistent")
	assert.Error(t, err)
	assert.True(t, apierrors.IsNotFound(err))
}

func TestCombinedBKECluster(t *testing.T) {
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	result, err := CombinedBKECluster(cluster, cm)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotNil(t, result.Spec.ClusterConfig.CustomExtra)
}

func TestCombinedBKECluster_WithAnnotations(t *testing.T) {
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	clusterJSON := `{"metadata":{"name":"test","namespace":"default"},"spec":{"clusterConfig":{}}}`
	cmJSON := `{"metadata":{"name":"test","namespace":"default"},"data":{"nodes":"[]","status":"[]"}}`

	annotation.SetAnnotation(cluster, annotation.LastUpdateConfigurationAnnotationKey, clusterJSON)
	annotation.SetAnnotation(cm, annotation.LastUpdateConfigurationAnnotationKey, cmJSON)

	result, err := CombinedBKECluster(cluster, cm)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetCombinedBKEClusterCM(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	result, err := GetCombinedBKEClusterCM(context.Background(), fakeClient, cluster)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "test", result.Name)
}

func TestGetCombinedBKEClusterCM_CreateIfNotFound(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster).Build()

	result, err := GetCombinedBKEClusterCM(context.Background(), fakeClient, cluster)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotNil(t, result.Data)
}

func TestGetCombinedBKEClusterCM_NilData(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Data: nil,
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	result, err := GetCombinedBKEClusterCM(context.Background(), fakeClient, cluster)
	assert.NoError(t, err)
	assert.NotNil(t, result.Data)
}

func TestDeleteCombinedBKECluster(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	err := DeleteCombinedBKECluster(context.Background(), fakeClient, "default", "test")
	assert.NoError(t, err)

	var deletedCluster v1beta1.BKECluster
	err = fakeClient.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: "test"}, &deletedCluster)
	assert.True(t, apierrors.IsNotFound(err))
}

func TestListCombinedBKECluster(t *testing.T) {
	scheme := setupTestScheme()
	cluster1 := createTestBKECluster("test1", "default")
	cluster2 := createTestBKECluster("test2", "default")
	cm1 := createTestConfigMap("test1", "default")
	cm2 := createTestConfigMap("test2", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster1, cluster2, cm1, cm2).Build()

	result, err := ListCombinedBKECluster(context.Background(), fakeClient)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, 2, len(result.Items))
}

func TestNewTmpBkeNodesCluster(t *testing.T) {
	cluster := createTestBKECluster("test", "default")
	result := newTmpBkeNodesCluster(cluster)
	assert.NotNil(t, result)
	assert.NotNil(t, result.spec)
	assert.Equal(t, 0, len(result.spec))
}

func TestCleanBkeCluster(t *testing.T) {
	cluster := createTestBKECluster("test", "default")
	cluster.Status.ClusterStatus = "Running"

	result := cleanBkeCluster(cluster)
	assert.NotNil(t, result)
	assert.Equal(t, "test", result.Name)
	assert.Equal(t, "default", result.Namespace)
	assert.Empty(t, result.Status.ClusterStatus)
}

func TestGetLastUpdatedBKECluster(t *testing.T) {
	cluster := createTestBKECluster("test", "default")

	result, err := GetLastUpdatedBKECluster(cluster)
	assert.NoError(t, err)
	assert.Nil(t, result)

	clusterJSON := `{"metadata":{"name":"test","namespace":"default"},"spec":{"clusterConfig":{}}}`
	annotation.SetAnnotation(cluster, annotation.LastUpdateConfigurationAnnotationKey, clusterJSON)
	result, err = GetLastUpdatedBKECluster(cluster)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestNewBlankBKEConfigMap(t *testing.T) {
	cluster := createTestBKECluster("test", "default")
	result := newBlankBKEConfigMap(cluster)
	assert.NotNil(t, result)
	assert.Equal(t, "test", result.Name)
	assert.Equal(t, "default", result.Namespace)
	assert.NotNil(t, result.Data)
}

func TestFixPhaseStatus(t *testing.T) {
	tests := []struct {
		name     string
		input    confv1beta1.PhaseStatus
		expected int
	}{
		{
			name:     "empty",
			input:    confv1beta1.PhaseStatus{},
			expected: 0,
		},
		{
			name: "no duplicates",
			input: confv1beta1.PhaseStatus{
				{Name: "Phase1", Status: v1beta1.PhaseRunning, Message: "msg1"},
			},
			expected: 1,
		},
		{
			name: "with duplicates",
			input: confv1beta1.PhaseStatus{
				{Name: "Phase1", Status: v1beta1.PhaseRunning, Message: "msg1"},
				{Name: "Phase1", Status: v1beta1.PhaseRunning, Message: "msg1"},
			},
			expected: 1,
		},
		{
			name: "multiple failed EnsureCluster",
			input: confv1beta1.PhaseStatus{
				{Name: "EnsureCluster", Status: v1beta1.PhaseFailed, Message: "fail1"},
				{Name: "EnsureCluster", Status: v1beta1.PhaseFailed, Message: "fail2"},
				{Name: "EnsureCluster", Status: v1beta1.PhaseFailed, Message: "fail3"},
				{Name: "EnsureCluster", Status: v1beta1.PhaseFailed, Message: "fail4"},
			},
			expected: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fixPhaseStatus(tt.input)
			assert.Equal(t, tt.expected, len(result))
		})
	}
}

func TestDeduplicatePhaseStatus(t *testing.T) {
	input := confv1beta1.PhaseStatus{
		{Name: "Phase1", Status: v1beta1.PhaseRunning, Message: "msg1"},
		{Name: "Phase1", Status: v1beta1.PhaseRunning, Message: "msg1"},
		{Name: "Phase2", Status: v1beta1.PhaseRunning, Message: "msg2"},
	}

	result := deduplicatePhaseStatus(input)
	assert.Equal(t, 2, len(result))
}

func TestGetCurrentBkeClusterPatches(t *testing.T) {
	old := createTestBKECluster("test", "default")
	new := createTestBKECluster("test", "default")
	new.Labels = map[string]string{"key": "value"}

	patches, err := GetCurrentBkeClusterPatches(old, new)
	assert.NoError(t, err)
	if len(new.Labels) != len(old.Labels) {
		assert.NotNil(t, patches)
	}
}

func TestNewTmpBkeCluster(t *testing.T) {
	combined := createTestBKECluster("test", "default")
	combined.Labels = map[string]string{"key": "value"}
	combined.Annotations = map[string]string{"anno": "val"}

	current := createTestBKECluster("test", "default")
	current.ResourceVersion = "123"

	result := newTmpBkeCluster(combined, current)
	assert.NotNil(t, result)
	assert.Equal(t, "123", result.ResourceVersion)
	assert.Equal(t, "value", result.Labels["key"])
}

func TestUpdateCombinedBKECluster(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	err := UpdateCombinedBKECluster(context.Background(), fakeClient, cluster, []string{})
	assert.NoError(t, err)
}

func TestSyncStatusUntilComplete(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	err := SyncStatusUntilComplete(fakeClient, cluster)
	assert.NoError(t, err)
}

func TestSyncStatusUntilComplete_NotFound(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	err := SyncStatusUntilComplete(fakeClient, cluster)
	assert.NoError(t, err)
}

func TestPrepareClusterData(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	params := PrepareClusterDataParams{
		Ctx:             context.Background(),
		Client:          fakeClient,
		CombinedCluster: cluster,
		Patchs:          []PatchFunc{},
	}

	result, err := prepareClusterData(params)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestPreserveDeclarativeUpgradeFromFresh(t *testing.T) {
	now := metav1.Now()
	fresh := createTestBKECluster("test", "default")
	fresh.Status.DeclarativeUpgrade = &confv1beta1.DeclarativeUpgradeStatus{
		TargetVersion: "v26.06",
		Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{
			{Name: "bkeagent", Version: "v26.05", CompletedAt: now},
			{Name: "containerd", Version: "v2.1.2", CompletedAt: now},
		},
	}

	stale := createTestBKECluster("test", "default")
	stale.Status.DeclarativeUpgrade = &confv1beta1.DeclarativeUpgradeStatus{
		TargetVersion: "v26.06",
		Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{
			{Name: "bkeagent", Version: "v26.05", CompletedAt: now},
		},
	}

	PreserveDeclarativeUpgradeFromFresh(fresh, stale)

	assert.Len(t, stale.Status.DeclarativeUpgrade.Completed, 2)
	assert.Equal(t, "containerd", stale.Status.DeclarativeUpgrade.Completed[1].Name)
}

func TestPrepareClusterData_PreservesDeclarativeUpgradeFromAPI(t *testing.T) {
	scheme := setupTestScheme()
	now := metav1.Now()

	stale := createTestBKECluster("test", "default")
	stale.Status.DeclarativeUpgrade = &confv1beta1.DeclarativeUpgradeStatus{
		TargetVersion: "v26.06",
		Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{
			{Name: "bkeagent", Version: "v26.05", CompletedAt: now},
		},
	}

	apiCluster := stale.DeepCopy()
	apiCluster.Status.DeclarativeUpgrade = &confv1beta1.DeclarativeUpgradeStatus{
		TargetVersion: "v26.06",
		Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{
			{Name: "bkeagent", Version: "v26.05", CompletedAt: now},
			{Name: "containerd", Version: "v2.1.2", CompletedAt: now},
		},
	}

	cm := createTestConfigMap("test", "default")
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(apiCluster, cm).Build()

	_, err := prepareClusterData(PrepareClusterDataParams{
		Ctx:             context.Background(),
		Client:          fakeClient,
		CombinedCluster: stale,
	})
	assert.NoError(t, err)
	assert.Len(t, stale.Status.DeclarativeUpgrade.Completed, 2)
}

func TestHandleExternalUpdates(t *testing.T) {
	combined := createTestBKECluster("test", "default")
	current := createTestBKECluster("test", "default")

	err := handleExternalUpdates(combined, current)
	assert.NoError(t, err)
}

func TestInitializePatchHelper(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster).Build()

	result, helper, err := initializePatchHelper(context.Background(), fakeClient, cluster)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotNil(t, helper)
}

func TestInitializePatchHelper_NotFound(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	result, helper, err := initializePatchHelper(context.Background(), fakeClient, cluster)
	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.Nil(t, helper)
}

func TestProcessNodeData(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	params := ProcessNodeDataParams{
		Ctx:               context.Background(),
		Client:            fakeClient,
		CombinedCluster:   cluster,
		CurrentBkeCluster: cluster,
		DeleteNodes:       []string{},
	}

	result, err := processNodeData(params)
	assert.NoError(t, err)
	assert.NotNil(t, result.CM)
	assert.NotNil(t, result.NodesCM)
}

func TestGetBkeClusterAssociateNodesCM(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster, cm).Build()

	resultCM, nodes, err := getBkeClusterAssociateNodesCM(context.Background(), fakeClient, cluster)
	assert.NoError(t, err)
	assert.NotNil(t, resultCM)
	assert.NotNil(t, nodes)
}

func TestSetDefaultCustomExtra(t *testing.T) {
	cluster := createTestBKECluster("test", "default")
	cluster.Spec.ClusterConfig.CustomExtra = nil

	setDefaultCustomExtra(cluster)
	assert.NotNil(t, cluster.Spec.ClusterConfig.CustomExtra)
}

func TestEnsureConfigMapData(t *testing.T) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Data:       nil,
	}

	ensureConfigMapData(cm)
	assert.NotNil(t, cm.Data)
	assert.Equal(t, "[]", cm.Data["nodes"])
}

func TestCreateDefaultConfigMap(t *testing.T) {
	scheme := setupTestScheme()
	cluster := createTestBKECluster("test", "default")

	cm, err := createDefaultConfigMap("default", "test", cluster, scheme)
	assert.NoError(t, err)
	assert.NotNil(t, cm)
	assert.Equal(t, "test", cm.Name)
	assert.NotNil(t, cm.Data)
}

func TestGetLastFailedEnsureClusterIndices(t *testing.T) {
	ph := confv1beta1.PhaseStatus{
		{Name: "EnsureCluster", Status: v1beta1.PhaseFailed},
		{Name: "OtherPhase", Status: v1beta1.PhaseRunning},
		{Name: "EnsureCluster", Status: v1beta1.PhaseFailed},
		{Name: "EnsureCluster", Status: v1beta1.PhaseFailed},
	}

	indices := getLastFailedEnsureClusterIndices(ph, 2)
	assert.Equal(t, 2, len(indices))
}

func TestRemoveOldFailedEnsureCluster(t *testing.T) {
	ph := confv1beta1.PhaseStatus{
		{Name: "EnsureCluster", Status: v1beta1.PhaseFailed},
		{Name: "OtherPhase", Status: v1beta1.PhaseRunning},
		{Name: "EnsureCluster", Status: v1beta1.PhaseFailed},
	}

	indices := []int{2, 0}
	result := removeOldFailedEnsureCluster(ph, indices)
	assert.NotNil(t, result)
}

func TestHandleLastUpdateConfiguration(t *testing.T) {
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	err := handleLastUpdateConfiguration(cluster, cm)
	assert.NoError(t, err)
}

func TestHandleLastUpdateConfiguration_WithAnnotations(t *testing.T) {
	cluster := createTestBKECluster("test", "default")
	cm := createTestConfigMap("test", "default")

	clusterJSON := `{"metadata":{"name":"test","namespace":"default"},"spec":{"clusterConfig":{}}}`
	cmJSON := `{"metadata":{"name":"test","namespace":"default"},"data":{"nodes":"[]","status":"[]"}}`

	annotation.SetAnnotation(cluster, annotation.LastUpdateConfigurationAnnotationKey, clusterJSON)
	annotation.SetAnnotation(cm, annotation.LastUpdateConfigurationAnnotationKey, cmJSON)

	err := handleLastUpdateConfiguration(cluster, cm)
	assert.NoError(t, err)
}

func TestUpdateModifiedBKENodes_Empty(t *testing.T) {
	scheme := setupTestScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	err := UpdateModifiedBKENodes(context.Background(), fakeClient, v1beta1.BKENodes{})
	assert.NoError(t, err)
}
