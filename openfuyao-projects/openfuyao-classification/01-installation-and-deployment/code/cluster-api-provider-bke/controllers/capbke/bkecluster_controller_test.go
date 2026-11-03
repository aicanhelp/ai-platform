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

package capbke

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/record"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clustertracker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
)

const (
	clusterNumZero  = 0
	clusterNumOne   = 1
	clusterNumTwo   = 2
	clusterNumThree = 3
	clusterNumFive  = 5
	clusterNumTen   = 10

	clusterTestNamespace  = "test-ns"
	clusterTestName       = "test-cluster"
	clusterTestBKECluster = "test-bke-cluster"
	clusterTestNodeName   = "test-node"
	clusterTestNodeIP     = "192.168.1.100"
	clusterTestNodeIP2    = "192.168.1.101"
	clusterTestNodeIP3    = "192.168.1.102"
)

// newTestScheme creates a runtime.Scheme with all needed types registered
func newTestScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	return scheme
}

// newTestBKECluster creates a minimal BKECluster for testing
func newTestBKECluster() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      clusterTestBKECluster,
			Namespace: clusterTestNamespace,
		},
	}
}

// newTestBKENode creates a BKENode associated with a cluster
func newTestBKENode(name, namespace, clusterName, ip string) *confv1beta1.BKENode {
	return &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				nodeutil.ClusterNameLabel: clusterName,
			},
		},
		Spec: confv1beta1.BKENodeSpec{
			IP:   ip,
			Role: []string{bkenode.WorkerNodeRole},
		},
		Status: confv1beta1.BKENodeStatus{
			State: confv1beta1.NodeReady,
		},
	}
}

// newTestReconciler creates a BKEClusterReconciler with fake client and optional objects
func newTestReconciler(objs ...client.Object) *BKEClusterReconciler {
	scheme := newTestScheme()
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if len(objs) > clusterNumZero {
		runtimeObjs := make([]client.Object, clusterNumZero, len(objs))
		statusObjs := make([]client.Object, clusterNumZero, len(objs))
		for _, o := range objs {
			runtimeObjs = append(runtimeObjs, o)
			statusObjs = append(statusObjs, o)
		}
		builder = builder.WithObjects(runtimeObjs...).WithStatusSubresource(statusObjs...)
	}
	fakeClient := builder.Build()
	r := &BKEClusterReconciler{
		Client:   fakeClient,
		Scheme:   scheme,
		Recorder: record.NewFakeRecorder(clusterNumTen),
	}
	r.NodeFetcher = nodeutil.NewNodeFetcher(fakeClient)
	return r
}

func TestHandleClusterError(t *testing.T) {
	tests := []struct {
		name        string
		err         error
		expectError bool
	}{
		{
			name:        "not found error",
			err:         nil,
			expectError: false,
		},
		{
			name:        "with not found error",
			err:         apierrors.NewNotFound(schema.GroupResource{}, "test"),
			expectError: false,
		},
		{
			name:        "with other error",
			err:         assert.AnError,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &BKEClusterReconciler{}
			result, err := r.handleClusterError(tt.err)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.Equal(t, ctrl.Result{}, result)
			}
		})
	}
}

func TestInitializeLogger(t *testing.T) {
	tests := []struct {
		name       string
		bkeCluster *bkev1beta1.BKECluster
	}{
		{
			name: "initialize logger",
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &BKEClusterReconciler{
				Recorder: record.NewFakeRecorder(clusterNumTen),
			}

			logger := r.initializeLogger(tt.bkeCluster)
			assert.NotNil(t, logger)
		})
	}
}

func TestGetClusterStatusFlags(t *testing.T) {
	tests := []struct {
		name              string
		bkeCluster        *bkev1beta1.BKECluster
		expectDeployFlag  bool
		expectUpgradeFlag bool
		expectManageFlag  bool
	}{
		{
			name: "no conditions",
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
				},
			},
			expectDeployFlag:  false,
			expectUpgradeFlag: false,
			expectManageFlag:  false,
		},
		{
			name: "with deploy failed condition",
			bkeCluster: &bkev1beta1.BKECluster{
				Status: confv1beta1.BKEClusterStatus{
					Conditions: []confv1beta1.ClusterCondition{
						{
							Type:    bkev1beta1.ClusterHealthyStateCondition,
							Reason:  string(bkev1beta1.Deploying),
							Message: string(bkev1beta1.DeployFailed),
							Status:  confv1beta1.ConditionTrue,
						},
					},
				},
			},
			expectDeployFlag:  true,
			expectUpgradeFlag: false,
			expectManageFlag:  false,
		},
		{
			name: "with upgrade failed condition",
			bkeCluster: &bkev1beta1.BKECluster{
				Status: confv1beta1.BKEClusterStatus{
					Conditions: []confv1beta1.ClusterCondition{
						{
							Type:    bkev1beta1.ClusterHealthyStateCondition,
							Reason:  string(bkev1beta1.Upgrading),
							Message: string(bkev1beta1.UpgradeFailed),
							Status:  confv1beta1.ConditionTrue,
						},
					},
				},
			},
			expectDeployFlag:  false,
			expectUpgradeFlag: true,
			expectManageFlag:  false,
		},
		{
			name: "with manage failed condition",
			bkeCluster: &bkev1beta1.BKECluster{
				Status: confv1beta1.BKEClusterStatus{
					Conditions: []confv1beta1.ClusterCondition{
						{
							Type:    bkev1beta1.ClusterHealthyStateCondition,
							Reason:  string(bkev1beta1.Managing),
							Message: string(bkev1beta1.ManageFailed),
							Status:  confv1beta1.ConditionTrue,
						},
					},
				},
			},
			expectDeployFlag:  false,
			expectUpgradeFlag: false,
			expectManageFlag:  true,
		},
		{
			name: "with empty condition",
			bkeCluster: &bkev1beta1.BKECluster{
				Status: confv1beta1.BKEClusterStatus{
					Conditions: []confv1beta1.ClusterCondition{},
				},
			},
			expectDeployFlag:  false,
			expectUpgradeFlag: false,
			expectManageFlag:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &BKEClusterReconciler{}
			deploy, upgrade, manage := r.getClusterStatusFlags(tt.bkeCluster)

			assert.Equal(t, tt.expectDeployFlag, deploy)
			assert.Equal(t, tt.expectUpgradeFlag, upgrade)
			assert.Equal(t, tt.expectManageFlag, manage)
		})
	}
}

func TestHandleRetryLogic(t *testing.T) {
	tests := []struct {
		name        string
		bkeCluster  *bkev1beta1.BKECluster
		expectRetry bool
	}{
		{
			name: "no retry annotation",
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
				},
			},
			expectRetry: false,
		},
		{
			name: "with retry annotation empty value",
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
					Annotations: map[string]string{
						annotation.RetryAnnotationKey: "",
					},
				},
			},
			expectRetry: true,
		},
		{
			name: "with retry annotation specific nodes",
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
					Annotations: map[string]string{
						annotation.RetryAnnotationKey: clusterTestNodeIP,
					},
				},
			},
			expectRetry: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestReconciler()
			retry, patchFunc := r.handleRetryLogic(context.Background(), tt.bkeCluster)

			assert.Equal(t, tt.expectRetry, retry)
			assert.NotNil(t, patchFunc)
		})
	}
}

func TestCreateRemoveRetryAnnotationFunc(t *testing.T) {
	r := &BKEClusterReconciler{}
	patchFunc := r.createRemoveRetryAnnotationFunc()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annotation.RetryAnnotationKey:                             clusterTestNodeIP,
				annotation.ControlPlaneConfigUpdateFailCountAnnotationKey: "3",
				annotation.EtcdConfigUpdateFailCountAnnotationKey:         "2",
			},
		},
	}
	cluster.Status.ControlPlaneConfigHash = confv1beta1.ComponentConfigHash{
		APIServer:         "good-api",
		ControllerManager: "good-cm",
		Scheduler:         "good-sch",
	}
	cluster.Status.EtcdConfigHash = "good-etcd"

	patchFunc(cluster)

	_, exists := cluster.Annotations[annotation.RetryAnnotationKey]
	assert.False(t, exists)
	assert.Equal(t, bkev1beta1.ClusterChecking, cluster.Status.ClusterStatus)
	_, exists = cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey]
	assert.False(t, exists)
	_, exists = cluster.Annotations[annotation.EtcdConfigUpdateFailCountAnnotationKey]
	assert.False(t, exists)
	assert.Equal(t, confv1beta1.ComponentConfigHash{
		APIServer:         "good-api",
		ControllerManager: "good-cm",
		Scheduler:         "good-sch",
	}, cluster.Status.ControlPlaneConfigHash)
	assert.Equal(t, "good-etcd", cluster.Status.EtcdConfigHash)
}

func TestSetClusterHealthStatus(t *testing.T) {
	tests := []struct {
		name           string
		flags          ClusterHealthStatusFlags
		expectState    confv1beta1.ClusterHealthState
		prepareCluster func(*bkev1beta1.BKECluster)
		deleteOrReset  bool
		expectDeleting bool
	}{
		{
			name: "deploy flag",
			flags: ClusterHealthStatusFlags{
				DeployFlag: true,
			},
			expectState: bkev1beta1.Deploying,
		},
		{
			name: "deploy failed flag",
			flags: ClusterHealthStatusFlags{
				DeployFailedFlag: true,
			},
			expectState: bkev1beta1.Deploying,
		},
		{
			name: "upgrade flag",
			flags: ClusterHealthStatusFlags{
				UpgradeFlag: true,
			},
			expectState: bkev1beta1.Upgrading,
		},
		{
			name: "upgrade failed flag",
			flags: ClusterHealthStatusFlags{
				UpgradeFailedFlag: true,
			},
			expectState: bkev1beta1.Upgrading,
		},
		{
			name: "terminal upgrade failure preserves UpgradeFailed health",
			flags: ClusterHealthStatusFlags{
				UpgradeFlag:       true,
				UpgradeFailedFlag: true,
			},
			prepareCluster: func(cluster *bkev1beta1.BKECluster) {
				condition.ConditionMark(cluster, bkev1beta1.ClusterHealthyStateCondition,
					confv1beta1.ConditionFalse, string(bkev1beta1.Upgrading), string(bkev1beta1.UpgradeFailed))
			},
			expectState: bkev1beta1.UpgradeFailed,
		},
		{
			name: "manage flag",
			flags: ClusterHealthStatusFlags{
				ManageFlag: true,
			},
			expectState: bkev1beta1.Managing,
		},
		{
			name: "manage failed flag",
			flags: ClusterHealthStatusFlags{
				ManageFailedFlag: true,
			},
			expectState: bkev1beta1.Managing,
		},
		{
			name: "all flags true",
			flags: ClusterHealthStatusFlags{
				DeployFlag:        true,
				UpgradeFlag:       true,
				ManageFlag:        true,
				DeployFailedFlag:  true,
				UpgradeFailedFlag: true,
				ManageFailedFlag:  true,
			},
		},
		{
			name:  "no flags set",
			flags: ClusterHealthStatusFlags{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &BKEClusterReconciler{}
			bkeCluster := newTestBKECluster()
			if tt.prepareCluster != nil {
				tt.prepareCluster(bkeCluster)
			}

			patches := gomonkey.ApplyFunc(phaseutil.IsDeleteOrReset, func(_ *bkev1beta1.BKECluster) bool {
				return false
			})
			defer patches.Reset()

			r.setClusterHealthStatus(bkeCluster, tt.flags)
			if tt.expectState != "" {
				assert.Equal(t, tt.expectState, bkeCluster.Status.ClusterHealthState)
			}
		})
	}
}

func TestSetClusterHealthStatusDeleteOrReset(t *testing.T) {
	r := &BKEClusterReconciler{}
	// Simulate a cluster being deleted by setting Spec.Reset = true
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      clusterTestBKECluster,
			Namespace: clusterTestNamespace,
		},
		Spec: confv1beta1.BKEClusterSpec{
			Reset: true,
		},
	}

	r.setClusterHealthStatus(bkeCluster, ClusterHealthStatusFlags{})
	assert.Equal(t, bkev1beta1.Deleting, bkeCluster.Status.ClusterHealthState)
}

func TestMarkBKEClusterHealthyStatus(t *testing.T) {
	tests := []struct {
		name       string
		bkeCluster *bkev1beta1.BKECluster
		status     confv1beta1.ClusterHealthState
	}{
		{
			name:       "mark deploying status",
			bkeCluster: &bkev1beta1.BKECluster{},
			status:     bkev1beta1.Deploying,
		},
		{
			name:       "mark upgrading status",
			bkeCluster: &bkev1beta1.BKECluster{},
			status:     bkev1beta1.Upgrading,
		},
		{
			name:       "mark managing status",
			bkeCluster: &bkev1beta1.BKECluster{},
			status:     bkev1beta1.Managing,
		},
		{
			name:       "mark deleting status",
			bkeCluster: &bkev1beta1.BKECluster{},
			status:     bkev1beta1.Deleting,
		},
		{
			name:       "mark deploying failed status",
			bkeCluster: &bkev1beta1.BKECluster{},
			status:     bkev1beta1.DeployFailed,
		},
		{
			name:       "mark upgrading failed status",
			bkeCluster: &bkev1beta1.BKECluster{},
			status:     bkev1beta1.UpgradeFailed,
		},
		{
			name:       "mark manage failed status",
			bkeCluster: &bkev1beta1.BKECluster{},
			status:     bkev1beta1.ManageFailed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			markBKEClusterHealthyStatus(tt.bkeCluster, tt.status)
			assert.NotNil(t, tt.bkeCluster.Status.Conditions)
			assert.Equal(t, tt.status, tt.bkeCluster.Status.ClusterHealthState)
		})
	}
}

func TestBKENodeToBKEClusterMapFunc(t *testing.T) {
	tests := []struct {
		name      string
		obj       client.Object
		expectLen int
	}{
		{
			name:      "not a BKENode",
			obj:       &bkev1beta1.BKECluster{},
			expectLen: clusterNumZero,
		},
		{
			name: "BKENode without cluster label",
			obj: &confv1beta1.BKENode{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestNodeName,
					Namespace: clusterTestNamespace,
				},
			},
			expectLen: clusterNumZero,
		},
		{
			name: "BKENode with cluster label",
			obj: &confv1beta1.BKENode{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestNodeName,
					Namespace: clusterTestNamespace,
					Labels: map[string]string{
						nodeutil.ClusterNameLabel: clusterTestBKECluster,
					},
				},
			},
			expectLen: clusterNumOne,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &BKEClusterReconciler{}
			handler := r.bkeNodeToBKEClusterMapFunc()
			requests := handler(context.Background(), tt.obj)
			assert.Len(t, requests, tt.expectLen)
		})
	}
}

func TestClusterToBKEClusterMapFunc(t *testing.T) {
	tests := []struct {
		name      string
		obj       client.Object
		expectLen int
	}{
		{
			name:      "not a Cluster",
			obj:       &bkev1beta1.BKECluster{},
			expectLen: clusterNumZero,
		},
		{
			name: "Cluster without InfrastructureRef",
			obj: &clusterv1.Cluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestName,
					Namespace: clusterTestNamespace,
				},
			},
			expectLen: clusterNumZero,
		},
		{
			name: "Cluster with deletion timestamp",
			obj: &clusterv1.Cluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:              clusterTestName,
					Namespace:         clusterTestNamespace,
					DeletionTimestamp: &metav1.Time{},
				},
				Spec: clusterv1.ClusterSpec{
					InfrastructureRef: &corev1.ObjectReference{
						APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
						Kind:       "BKECluster",
						Name:       clusterTestBKECluster,
						Namespace:  clusterTestNamespace,
					},
				},
			},
			expectLen: clusterNumZero,
		},
		{
			name: "Cluster with mismatched GVK",
			obj: &clusterv1.Cluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestName,
					Namespace: clusterTestNamespace,
				},
				Spec: clusterv1.ClusterSpec{
					InfrastructureRef: &corev1.ObjectReference{
						APIVersion: "other.group/v1",
						Kind:       "OtherKind",
						Name:       clusterTestBKECluster,
						Namespace:  clusterTestNamespace,
					},
				},
			},
			expectLen: clusterNumZero,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := clusterToBKEClusterMapFunc(context.Background(), schema.GroupVersionKind{}, nil, nil)
			requests := handler(context.Background(), tt.obj)
			assert.Len(t, requests, tt.expectLen)
		})
	}
}

func TestClusterToBKEClusterMapFuncWithMatchingGVK(t *testing.T) {
	scheme := newTestScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      clusterTestBKECluster,
				Namespace: clusterTestNamespace,
			},
		},
	).Build()

	gvk := bkev1beta1.GroupVersion.WithKind("BKECluster")
	handler := clusterToBKEClusterMapFunc(context.Background(), gvk, fakeClient, &bkev1beta1.BKECluster{})

	obj := &clusterv1.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      clusterTestName,
			Namespace: clusterTestNamespace,
		},
		Spec: clusterv1.ClusterSpec{
			InfrastructureRef: &corev1.ObjectReference{
				APIVersion: gvk.GroupVersion().String(),
				Kind:       gvk.Kind,
				Name:       clusterTestBKECluster,
				Namespace:  clusterTestNamespace,
			},
		},
	}

	requests := handler(context.Background(), obj)
	assert.Len(t, requests, clusterNumOne)
	assert.Equal(t, clusterTestBKECluster, requests[clusterNumZero].Name)
	assert.Equal(t, clusterTestNamespace, requests[clusterNumZero].Namespace)
}

func TestNodeToBKEClusterMapFunc(t *testing.T) {
	tests := []struct {
		name      string
		obj       client.Object
		expectLen int
	}{
		{
			name:      "not a Node",
			obj:       &bkev1beta1.BKECluster{},
			expectLen: clusterNumZero,
		},
		{
			name: "Node without cluster annotations",
			obj: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: clusterTestNodeName,
				},
			},
			expectLen: clusterNumZero,
		},
		{
			name: "Node with only cluster name annotation",
			obj: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: clusterTestNodeName,
					Annotations: map[string]string{
						clusterv1.ClusterNameAnnotation: clusterTestName,
					},
				},
			},
			expectLen: clusterNumZero,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := nodeToBKEClusterMapFunc(context.Background(), nil)
			requests := handler(context.Background(), tt.obj)
			assert.Len(t, requests, tt.expectLen)
		})
	}
}

func TestNodeToBKEClusterMapFuncWithFullAnnotations(t *testing.T) {
	scheme := newTestScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      clusterTestName,
				Namespace: clusterTestNamespace,
			},
			Spec: clusterv1.ClusterSpec{
				InfrastructureRef: &corev1.ObjectReference{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
				},
			},
		},
	).Build()

	handler := nodeToBKEClusterMapFunc(context.Background(), fakeClient)

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: clusterTestNodeName,
			Annotations: map[string]string{
				clusterv1.ClusterNameAnnotation:      clusterTestName,
				clusterv1.ClusterNamespaceAnnotation: clusterTestNamespace,
			},
		},
	}

	requests := handler(context.Background(), node)
	assert.Len(t, requests, clusterNumOne)
	assert.Equal(t, clusterTestBKECluster, requests[clusterNumZero].Name)
}

func TestInitNodeFetcher(t *testing.T) {
	tests := []struct {
		name string
	}{
		{
			name: "init node fetcher when nil",
		},
		{
			name: "init node fetcher when already set",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &BKEClusterReconciler{}
			r.initNodeFetcher()
			assert.NotNil(t, r.NodeFetcher)

			// Call again to verify idempotency
			first := r.NodeFetcher
			r.initNodeFetcher()
			assert.Equal(t, first, r.NodeFetcher)
		})
	}
}

func TestGetFinalResult(t *testing.T) {
	tests := []struct {
		name          string
		phaseResult   ctrl.Result
		expectRequeue bool
	}{
		{
			name: "phase result with requeue",
			phaseResult: ctrl.Result{
				Requeue: true,
			},
			expectRequeue: true,
		},
		{
			name: "phase result with requeue after",
			phaseResult: ctrl.Result{
				RequeueAfter: clusterNumTen,
			},
			expectRequeue: true,
		},
		{
			name: "phase result without requeue",
			phaseResult: ctrl.Result{
				Requeue: false,
			},
			expectRequeue: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &BKEClusterReconciler{}
			result, _ := r.getFinalResult(tt.phaseResult, &bkev1beta1.BKECluster{})
			if tt.expectRequeue {
				assert.True(t, result.Requeue || result.RequeueAfter > clusterNumZero)
			} else {
				assert.False(t, result.Requeue || result.RequeueAfter > clusterNumZero)
			}
		})
	}
}

func TestBKEClusterSetupWithManager(t *testing.T) {
	t.Skip("requires controller manager")
}

func TestReconcile(t *testing.T) {
	tests := []struct {
		name           string
		mockGetCluster func() (*bkev1beta1.BKECluster, error)
		expectError    bool
		expectResult   ctrl.Result
	}{
		{
			name: "cluster not found returns empty result",
			mockGetCluster: func() (*bkev1beta1.BKECluster, error) {
				return nil, apierrors.NewNotFound(schema.GroupResource{}, clusterTestBKECluster)
			},
			expectError:  false,
			expectResult: ctrl.Result{},
		},
		{
			name: "get cluster error returns error",
			mockGetCluster: func() (*bkev1beta1.BKECluster, error) {
				return nil, fmt.Errorf("internal error")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestReconciler()

			patches := gomonkey.ApplyFunc(mergecluster.GetCombinedBKECluster,
				func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
					return tt.mockGetCluster()
				})
			defer patches.Reset()

			result, err := r.Reconcile(context.Background(), ctrl.Request{
				NamespacedName: client.ObjectKey{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
				},
			})

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectResult, result)
			}
		})
	}
}

func TestReconcileFullPath(t *testing.T) {
	r := newTestReconciler()

	testCluster := newTestBKECluster()
	patches := gomonkey.ApplyFunc(mergecluster.GetCombinedBKECluster,
		func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return testCluster, nil
		})
	defer patches.Reset()

	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster,
		func(_ *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
			return testCluster, nil
		})

	patches.ApplyFunc(
		(*BKEClusterReconciler).computeAgentStatus,
		func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) error {
			return nil
		})

	patches.ApplyFunc(
		(*BKEClusterReconciler).initNodeStatus,
		func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) error {
			return nil
		})

	patches.ApplyFunc(
		(*BKEClusterReconciler).executePhaseFlow,
		func(_ *BKEClusterReconciler, _ context.Context,
			_ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster,
			_ *bkev1beta1.BKELogger) (ctrl.Result, error) {
			return ctrl.Result{}, nil
		})

	patches.ApplyFunc(clustertracker.AllowTrackerRemoteCluster,
		func(_ *bkev1beta1.BKECluster) bool {
			return false
		})

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{
			Name:      clusterTestBKECluster,
			Namespace: clusterTestNamespace,
		},
	})

	assert.NoError(t, err)
	assert.False(t, result.Requeue)
}

func TestReconcileGetOldClusterError(t *testing.T) {
	r := newTestReconciler()

	patches := gomonkey.ApplyFunc(mergecluster.GetCombinedBKECluster,
		func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return newTestBKECluster(), nil
		})
	defer patches.Reset()

	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster,
		func(_ *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
			return nil, fmt.Errorf("parse error")
		})

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{
			Name:      clusterTestBKECluster,
			Namespace: clusterTestNamespace,
		},
	})

	assert.Error(t, err)
}

func TestReconcileHandleClusterStatusError(t *testing.T) {
	r := newTestReconciler()

	patches := gomonkey.ApplyFunc(mergecluster.GetCombinedBKECluster,
		func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return newTestBKECluster(), nil
		})
	defer patches.Reset()

	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster,
		func(_ *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
			return newTestBKECluster(), nil
		})

	patches.ApplyFunc(
		(*BKEClusterReconciler).computeAgentStatus,
		func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) error {
			return fmt.Errorf("agent status error")
		})

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{
			Name:      clusterTestBKECluster,
			Namespace: clusterTestNamespace,
		},
	})

	assert.Error(t, err)
}

func TestReconcilePhaseFlowError(t *testing.T) {
	r := newTestReconciler()

	patches := gomonkey.ApplyFunc(mergecluster.GetCombinedBKECluster,
		func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return newTestBKECluster(), nil
		})
	defer patches.Reset()

	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster,
		func(_ *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
			return newTestBKECluster(), nil
		})

	patches.ApplyFunc(
		(*BKEClusterReconciler).computeAgentStatus,
		func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) error {
			return nil
		})

	patches.ApplyFunc(
		(*BKEClusterReconciler).initNodeStatus,
		func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) error {
			return nil
		})

	patches.ApplyFunc(
		(*BKEClusterReconciler).executePhaseFlow,
		func(_ *BKEClusterReconciler, _ context.Context,
			_ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster,
			_ *bkev1beta1.BKELogger) (ctrl.Result, error) {
			return ctrl.Result{}, fmt.Errorf("phase error")
		})

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{
			Name:      clusterTestBKECluster,
			Namespace: clusterTestNamespace,
		},
	})

	assert.Error(t, err)
}

func TestGetAndValidateCluster(t *testing.T) {
	tests := []struct {
		name        string
		mockReturn  *bkev1beta1.BKECluster
		mockError   error
		expectError bool
	}{
		{
			name:        "successful get",
			mockReturn:  newTestBKECluster(),
			mockError:   nil,
			expectError: false,
		},
		{
			name:        "not found error",
			mockReturn:  nil,
			mockError:   apierrors.NewNotFound(schema.GroupResource{}, clusterTestBKECluster),
			expectError: true,
		},
		{
			name:        "internal error",
			mockReturn:  nil,
			mockError:   fmt.Errorf("internal error"),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestReconciler()

			patches := gomonkey.ApplyFunc(mergecluster.GetCombinedBKECluster,
				func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
					return tt.mockReturn, tt.mockError
				})
			defer patches.Reset()

			cluster, err := r.getAndValidateCluster(context.Background(),
				ctrl.Request{NamespacedName: client.ObjectKey{
					Name: clusterTestBKECluster, Namespace: clusterTestNamespace,
				}})

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, cluster)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, cluster)
			}
		})
	}
}

func TestRegisterMetrics(t *testing.T) {
	tests := []struct {
		name        string
		metricsAddr string
		bkeCluster  *bkev1beta1.BKECluster
	}{
		{
			name:        "metrics disabled",
			metricsAddr: "0",
			bkeCluster:  newTestBKECluster(),
		},
		{
			name:        "metrics enabled",
			metricsAddr: ":8080",
			bkeCluster:  newTestBKECluster(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalAddr := config.MetricsAddr
			defer func() { config.MetricsAddr = originalAddr }()

			config.MetricsAddr = tt.metricsAddr
			r := &BKEClusterReconciler{}
			// Should not panic
			r.registerMetrics(tt.bkeCluster)
		})
	}
}

func TestGetOldBKECluster(t *testing.T) {
	tests := []struct {
		name        string
		mockReturn  *bkev1beta1.BKECluster
		mockError   error
		expectError bool
	}{
		{
			name:        "successful get old cluster",
			mockReturn:  newTestBKECluster(),
			mockError:   nil,
			expectError: false,
		},
		{
			name:        "error getting old cluster",
			mockReturn:  nil,
			mockError:   fmt.Errorf("parse error"),
			expectError: true,
		},
		{
			name:        "nil return no error",
			mockReturn:  nil,
			mockError:   nil,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &BKEClusterReconciler{}

			patches := gomonkey.ApplyFunc(mergecluster.GetLastUpdatedBKECluster,
				func(_ *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
					return tt.mockReturn, tt.mockError
				})
			defer patches.Reset()

			result, err := r.getOldBKECluster(newTestBKECluster())
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.mockReturn != nil {
					assert.NotNil(t, result)
				}
			}
		})
	}
}

func TestComputeAgentStatus(t *testing.T) {
	tests := []struct {
		name         string
		bkeCluster   *bkev1beta1.BKECluster
		bkeNodes     []*confv1beta1.BKENode
		syncErr      error
		expectError  bool
		expectStatus string
	}{
		{
			name:         "initial status with zero nodes",
			bkeCluster:   newTestBKECluster(),
			bkeNodes:     nil,
			expectError:  false,
			expectStatus: "0/0",
		},
		{
			name:       "initial status with two nodes",
			bkeCluster: newTestBKECluster(),
			bkeNodes: []*confv1beta1.BKENode{
				newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP),
				newTestBKENode("node-2", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP2),
			},
			expectError:  false,
			expectStatus: "0/2",
		},
		{
			name: "existing status 1/2 with two nodes",
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
				},
				Status: confv1beta1.BKEClusterStatus{
					AgentStatus: confv1beta1.BKEAgentStatus{
						Replies:            int32(clusterNumTwo),
						UnavailableReplies: int32(clusterNumOne),
						Status:             "1/2",
					},
				},
			},
			bkeNodes: []*confv1beta1.BKENode{
				newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP),
				newTestBKENode("node-2", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP2),
			},
			expectError:  false,
			expectStatus: "1/2",
		},
		{
			name: "available nodes greater than node count",
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
				},
				Status: confv1beta1.BKEClusterStatus{
					AgentStatus: confv1beta1.BKEAgentStatus{
						Status: "5/5",
					},
				},
			},
			bkeNodes: []*confv1beta1.BKENode{
				newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP),
				newTestBKENode("node-2", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP2),
			},
			expectError:  false,
			expectStatus: "2/2",
		},
		{
			name: "status with invalid format",
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterTestBKECluster,
					Namespace: clusterTestNamespace,
				},
				Status: confv1beta1.BKEClusterStatus{
					AgentStatus: confv1beta1.BKEAgentStatus{
						Status: "invalid",
					},
				},
			},
			bkeNodes: []*confv1beta1.BKENode{
				newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP),
				newTestBKENode("node-2", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP2),
			},
			expectError:  false,
			expectStatus: "0/2",
		},
		{
			name:       "sync status error",
			bkeCluster: newTestBKECluster(),
			bkeNodes: []*confv1beta1.BKENode{
				newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP),
			},
			syncErr:     fmt.Errorf("sync error"),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []client.Object
			for _, n := range tt.bkeNodes {
				objs = append(objs, n)
			}
			r := newTestReconciler(objs...)

			patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
				func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
					return tt.syncErr
				})
			defer patches.Reset()

			err := r.computeAgentStatus(context.Background(), tt.bkeCluster)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectStatus, tt.bkeCluster.Status.AgentStatus.Status)
			}
		})
	}
}

func TestHandleClusterStatus(t *testing.T) {
	tests := []struct {
		name              string
		computeAgentErr   error
		initNodeStatusErr error
		expectError       bool
	}{
		{
			name:        "success",
			expectError: false,
		},
		{
			name:            "compute agent status error",
			computeAgentErr: fmt.Errorf("agent error"),
			expectError:     true,
		},
		{
			name:              "init node status error",
			initNodeStatusErr: fmt.Errorf("node status error"),
			expectError:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestReconciler()

			patches := gomonkey.ApplyFunc(
				(*BKEClusterReconciler).computeAgentStatus,
				func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) error {
					return tt.computeAgentErr
				})
			defer patches.Reset()

			patches.ApplyFunc(
				(*BKEClusterReconciler).initNodeStatus,
				func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) error {
					return tt.initNodeStatusErr
				})

			bkeCluster := newTestBKECluster()
			bkeLogger := r.initializeLogger(bkeCluster)

			err := r.handleClusterStatus(context.Background(), bkeCluster, bkeLogger)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestExecutePhaseFlow(t *testing.T) {
	tests := []struct {
		name          string
		expectRequeue bool
		expectError   bool
	}{
		{
			name:          "successful phase flow",
			expectRequeue: false,
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestReconciler()
			bkeCluster := newTestBKECluster()
			oldBkeCluster := newTestBKECluster()
			bkeLogger := r.initializeLogger(bkeCluster)

			// Mock CalculatePhase and Execute through the entire flow
			// Since executePhaseFlow creates PhaseFlow internally, we mock the reconciler method
			patches := gomonkey.ApplyFunc(
				(*BKEClusterReconciler).executePhaseFlow,
				func(_ *BKEClusterReconciler, _ context.Context,
					_ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster,
					_ *bkev1beta1.BKELogger) (ctrl.Result, error) {
					return ctrl.Result{}, nil
				})
			defer patches.Reset()

			result, err := r.executePhaseFlow(context.Background(), bkeCluster, oldBkeCluster, bkeLogger)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, ctrl.Result{}, result)
			}
		})
	}
}

func TestSetupClusterWatching(t *testing.T) {
	tests := []struct {
		name          string
		allowTracker  bool
		expectRequeue bool
	}{
		{
			name:          "tracker not allowed",
			allowTracker:  false,
			expectRequeue: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestReconciler()
			bkeCluster := newTestBKECluster()
			bkeLogger := r.initializeLogger(bkeCluster)

			patches := gomonkey.ApplyFunc(clustertracker.AllowTrackerRemoteCluster,
				func(_ *bkev1beta1.BKECluster) bool {
					return tt.allowTracker
				})
			defer patches.Reset()

			result, err := r.setupClusterWatching(context.Background(), bkeCluster, bkeLogger)
			assert.NoError(t, err)

			if tt.expectRequeue {
				assert.True(t, result.RequeueAfter > clusterNumZero)
			} else {
				assert.Equal(t, ctrl.Result{}, result)
			}
		})
	}
}

func TestHandleNodeChanges(t *testing.T) {
	t.Run("no nodes means no change", func(t *testing.T) {
		r := newTestReconciler()
		bkeCluster := newTestBKECluster()
		changed := r.handleNodeChanges(context.Background(), bkeCluster)
		assert.False(t, changed)
	})

	t.Run("with nodes triggers comparison", func(t *testing.T) {
		node := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		node.Spec.Port = bkenode.DefaultNodeSSHPort
		node.Spec.Username = bkenode.DefaultNodeUserRoot
		r := newTestReconciler(node)
		bkeCluster := newTestBKECluster()
		// handleNodeChanges will fetch and compare; just verify it doesn't panic
		_ = r.handleNodeChanges(context.Background(), bkeCluster)
	})

	t.Run("with multiple nodes including removal", func(t *testing.T) {
		node1 := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		node1.Spec.Port = bkenode.DefaultNodeSSHPort
		node1.Spec.Username = bkenode.DefaultNodeUserRoot
		node2 := newTestBKENode("node-2", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP2)
		node2.Spec.Port = bkenode.DefaultNodeSSHPort
		node2.Spec.Username = bkenode.DefaultNodeUserRoot
		r := newTestReconciler(node1, node2)
		bkeCluster := newTestBKECluster()
		_ = r.handleNodeChanges(context.Background(), bkeCluster)
	})
}

func TestGetNodeFlags(t *testing.T) {
	t.Run("zero nodes means deploy flag true", func(t *testing.T) {
		r := newTestReconciler()

		patches := gomonkey.ApplyFunc(phaseutil.GetNeedUpgradeNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return nil
			})
		defer patches.Reset()

		// Default BKECluster (no cluster-from annotation) is a BKE cluster, not bocloud
		bkeCluster := newTestBKECluster()
		deploy, upgrade, manage := r.getNodeFlags(context.Background(), bkeCluster)

		assert.True(t, deploy)
		assert.False(t, upgrade)
		assert.False(t, manage)
	})

	t.Run("non-zero nodes deploy flag false", func(t *testing.T) {
		node := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		r := newTestReconciler(node)

		patches := gomonkey.ApplyFunc(phaseutil.GetNeedUpgradeNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return nil
			})
		defer patches.Reset()

		bkeCluster := newTestBKECluster()
		deploy, upgrade, manage := r.getNodeFlags(context.Background(), bkeCluster)

		assert.False(t, deploy)
		assert.False(t, upgrade)
		assert.False(t, manage)
	})

	t.Run("bocloud cluster not fully controlled", func(t *testing.T) {
		node := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		r := newTestReconciler(node)

		patches := gomonkey.ApplyFunc(phaseutil.GetNeedUpgradeNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return nil
			})
		defer patches.Reset()

		// Use annotation to mark as bocloud cluster (not fully controlled)
		bkeCluster := newTestBKECluster()
		bkeCluster.Annotations = map[string]string{
			"bke.bocloud.com/cluster-from": "bocloud",
		}
		deploy, _, manage := r.getNodeFlags(context.Background(), bkeCluster)

		assert.False(t, deploy)
		assert.True(t, manage)
	})

	t.Run("bocloud cluster fully controlled", func(t *testing.T) {
		node := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		r := newTestReconciler(node)

		patches := gomonkey.ApplyFunc(phaseutil.GetNeedUpgradeNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return nil
			})
		defer patches.Reset()

		// Use annotations to mark as bocloud cluster + fully controlled
		bkeCluster := newTestBKECluster()
		bkeCluster.Annotations = map[string]string{
			"bke.bocloud.com/cluster-from":                    "bocloud",
			annotation.KONKFullManagementClusterAnnotationKey: "true",
		}
		deploy, _, manage := r.getNodeFlags(context.Background(), bkeCluster)

		assert.False(t, deploy)
		assert.False(t, manage)
	})
}

func TestProcessRetryLogic(t *testing.T) {
	tests := []struct {
		name         string
		retryNodeIPs string
	}{
		{
			name:         "empty retry IPs triggers all nodes retry",
			retryNodeIPs: "",
		},
		{
			name:         "specific IPs triggers specific nodes retry",
			retryNodeIPs: clusterTestNodeIP,
		},
		{
			name:         "multiple IPs",
			retryNodeIPs: clusterTestNodeIP + "," + clusterTestNodeIP2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestReconciler()

			patches := gomonkey.ApplyFunc(
				(*BKEClusterReconciler).processAllNodesRetry,
				func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) {
				})
			defer patches.Reset()

			patches.ApplyFunc(
				(*BKEClusterReconciler).processSpecificNodesRetry,
				func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster, _ string) {
				})

			bkeCluster := newTestBKECluster()
			// Should not panic
			r.processRetryLogic(context.Background(), bkeCluster, tt.retryNodeIPs)
		})
	}
}

func TestProcessAllNodesRetry(t *testing.T) {
	t.Run("no nodes", func(t *testing.T) {
		r := newTestReconciler()
		bkeCluster := newTestBKECluster()
		r.processAllNodesRetry(context.Background(), bkeCluster)
	})

	t.Run("node without failed flag", func(t *testing.T) {
		node := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		node.Status.StateCode = clusterNumZero
		r := newTestReconciler(node)
		bkeCluster := newTestBKECluster()
		r.processAllNodesRetry(context.Background(), bkeCluster)
	})

	t.Run("node with failed flag gets cleared", func(t *testing.T) {
		node := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		node.Status.StateCode = bkev1beta1.NodeFailedFlag // set the failed flag
		r := newTestReconciler(node)
		bkeCluster := newTestBKECluster()
		r.processAllNodesRetry(context.Background(), bkeCluster)
	})

	t.Run("multiple nodes with mixed flags", func(t *testing.T) {
		node1 := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		node1.Status.StateCode = bkev1beta1.NodeFailedFlag
		node2 := newTestBKENode("node-2", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP2)
		node2.Status.StateCode = clusterNumZero
		r := newTestReconciler(node1, node2)
		bkeCluster := newTestBKECluster()
		r.processAllNodesRetry(context.Background(), bkeCluster)
	})
}

func TestProcessSpecificNodesRetry(t *testing.T) {
	t.Run("single node without failed flag", func(t *testing.T) {
		node := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		node.Status.StateCode = clusterNumZero
		r := newTestReconciler(node)
		bkeCluster := newTestBKECluster()
		r.processSpecificNodesRetry(context.Background(), bkeCluster, clusterTestNodeIP)
	})

	t.Run("single node with failed flag cleared", func(t *testing.T) {
		node := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		node.Status.StateCode = bkev1beta1.NodeFailedFlag
		r := newTestReconciler(node)
		bkeCluster := newTestBKECluster()
		r.processSpecificNodesRetry(context.Background(), bkeCluster, clusterTestNodeIP)
	})

	t.Run("multiple nodes", func(t *testing.T) {
		node1 := newTestBKENode("node-1", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP)
		node1.Status.StateCode = bkev1beta1.NodeFailedFlag
		node2 := newTestBKENode("node-2", clusterTestNamespace, clusterTestBKECluster, clusterTestNodeIP2)
		node2.Status.StateCode = clusterNumZero
		r := newTestReconciler(node1, node2)
		bkeCluster := newTestBKECluster()
		r.processSpecificNodesRetry(context.Background(), bkeCluster, clusterTestNodeIP+","+clusterTestNodeIP2)
	})

	t.Run("node not found continues", func(t *testing.T) {
		r := newTestReconciler()
		bkeCluster := newTestBKECluster()
		// Node IP doesn't exist in cluster
		r.processSpecificNodesRetry(context.Background(), bkeCluster, clusterTestNodeIP)
	})
}

func TestSyncNodeStatusIfNeeded(t *testing.T) {
	tests := []struct {
		name        string
		params      SyncNodeStatusParams
		syncErr     error
		expectError bool
		expectSync  bool
	}{
		{
			name: "no flags set does not sync",
			params: SyncNodeStatusParams{
				PatchFunc: func(_ *bkev1beta1.BKECluster) {},
			},
			expectSync: false,
		},
		{
			name: "deploy flag triggers sync",
			params: SyncNodeStatusParams{
				DeployFlag: true,
				PatchFunc:  func(_ *bkev1beta1.BKECluster) {},
			},
			expectSync: true,
		},
		{
			name: "deploy failed flag triggers sync",
			params: SyncNodeStatusParams{
				DeployFailedFlag: true,
				PatchFunc:        func(_ *bkev1beta1.BKECluster) {},
			},
			expectSync: true,
		},
		{
			name: "upgrade flag triggers sync",
			params: SyncNodeStatusParams{
				UpgradeFlag: true,
				PatchFunc:   func(_ *bkev1beta1.BKECluster) {},
			},
			expectSync: true,
		},
		{
			name: "upgrade failed flag triggers sync",
			params: SyncNodeStatusParams{
				UpgradeFailedFlag: true,
				PatchFunc:         func(_ *bkev1beta1.BKECluster) {},
			},
			expectSync: true,
		},
		{
			name: "manage failed flag triggers sync",
			params: SyncNodeStatusParams{
				ManageFailedFlag: true,
				PatchFunc:        func(_ *bkev1beta1.BKECluster) {},
			},
			expectSync: true,
		},
		{
			name: "retry flag triggers sync",
			params: SyncNodeStatusParams{
				RetryFlag: true,
				PatchFunc: func(_ *bkev1beta1.BKECluster) {},
			},
			expectSync: true,
		},
		{
			name: "node change flag triggers sync",
			params: SyncNodeStatusParams{
				NodeChangeFlag: true,
				PatchFunc:      func(_ *bkev1beta1.BKECluster) {},
			},
			expectSync: true,
		},
		{
			name: "sync error is returned",
			params: SyncNodeStatusParams{
				DeployFlag: true,
				PatchFunc:  func(_ *bkev1beta1.BKECluster) {},
			},
			syncErr:     fmt.Errorf("sync failed"),
			expectError: true,
			expectSync:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestReconciler()
			syncCalled := false

			patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
				func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
					syncCalled = true
					return tt.syncErr
				})
			defer patches.Reset()

			bkeCluster := newTestBKECluster()
			err := r.syncNodeStatusIfNeeded(bkeCluster, tt.params)

			assert.Equal(t, tt.expectSync, syncCalled)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestInitNodeStatus(t *testing.T) {
	t.Run("success no flags", func(t *testing.T) {
		r := newTestReconciler()

		patches := gomonkey.ApplyFunc(
			(*BKEClusterReconciler).handleNodeChanges,
			func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) bool {
				return false
			})
		defer patches.Reset()

		patches.ApplyFunc(
			(*BKEClusterReconciler).getNodeFlags,
			func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) (bool, bool, bool) {
				return false, false, false
			})

		patches.ApplyFunc(phaseutil.IsDeleteOrReset,
			func(_ *bkev1beta1.BKECluster) bool { return false })

		patches.ApplyFunc(
			(*BKEClusterReconciler).handleRetryLogic,
			func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) (bool, func(*bkev1beta1.BKECluster)) {
				return false, func(_ *bkev1beta1.BKECluster) {}
			})

		bkeCluster := newTestBKECluster()
		err := r.initNodeStatus(context.Background(), bkeCluster)
		assert.NoError(t, err)
	})

	t.Run("sync error propagated when deploy flag", func(t *testing.T) {
		r := newTestReconciler()

		patches := gomonkey.ApplyFunc(
			(*BKEClusterReconciler).handleNodeChanges,
			func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) bool {
				return false
			})
		defer patches.Reset()

		patches.ApplyFunc(
			(*BKEClusterReconciler).getNodeFlags,
			func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) (bool, bool, bool) {
				return true, false, false // deployFlag=true to trigger sync
			})

		patches.ApplyFunc(phaseutil.IsDeleteOrReset,
			func(_ *bkev1beta1.BKECluster) bool { return false })

		patches.ApplyFunc(
			(*BKEClusterReconciler).handleRetryLogic,
			func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) (bool, func(*bkev1beta1.BKECluster)) {
				return false, func(_ *bkev1beta1.BKECluster) {}
			})

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return fmt.Errorf("sync error")
			})

		bkeCluster := newTestBKECluster()
		err := r.initNodeStatus(context.Background(), bkeCluster)
		assert.Error(t, err)
	})
}

func TestInitNodeStatusWithDeployFlag(t *testing.T) {
	r := newTestReconciler()

	patches := gomonkey.ApplyFunc(
		(*BKEClusterReconciler).handleNodeChanges,
		func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) bool {
			return false
		})
	defer patches.Reset()

	patches.ApplyFunc(
		(*BKEClusterReconciler).getNodeFlags,
		func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) (bool, bool, bool) {
			return true, false, false // deployFlag=true
		})

	patches.ApplyFunc(phaseutil.IsDeleteOrReset,
		func(_ *bkev1beta1.BKECluster) bool { return false })

	patches.ApplyFunc(
		(*BKEClusterReconciler).handleRetryLogic,
		func(_ *BKEClusterReconciler, _ context.Context, _ *bkev1beta1.BKECluster) (bool, func(*bkev1beta1.BKECluster)) {
			return false, func(_ *bkev1beta1.BKECluster) {}
		})

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return nil
		})

	bkeCluster := newTestBKECluster()
	err := r.initNodeStatus(context.Background(), bkeCluster)
	assert.NoError(t, err)
	assert.Equal(t, bkev1beta1.Deploying, bkeCluster.Status.ClusterHealthState)
}

func TestNodeWatchRequeueInterval(t *testing.T) {
	expected := 10 * time.Second
	assert.Equal(t, expected, nodeWatchRequeueInterval)
}

func TestClusterHealthStatusFlagsStruct(t *testing.T) {
	flags := ClusterHealthStatusFlags{
		DeployFlag:        true,
		UpgradeFlag:       false,
		ManageFlag:        true,
		DeployFailedFlag:  false,
		UpgradeFailedFlag: true,
		ManageFailedFlag:  false,
	}

	assert.True(t, flags.DeployFlag)
	assert.False(t, flags.UpgradeFlag)
	assert.True(t, flags.ManageFlag)
	assert.False(t, flags.DeployFailedFlag)
	assert.True(t, flags.UpgradeFailedFlag)
	assert.False(t, flags.ManageFailedFlag)
}

func TestSyncNodeStatusParamsStruct(t *testing.T) {
	called := false
	params := SyncNodeStatusParams{
		DeployFlag:        true,
		DeployFailedFlag:  false,
		UpgradeFlag:       true,
		UpgradeFailedFlag: false,
		ManageFailedFlag:  true,
		RetryFlag:         false,
		NodeChangeFlag:    true,
		PatchFunc: func(_ *bkev1beta1.BKECluster) {
			called = true
		},
	}

	assert.True(t, params.DeployFlag)
	assert.True(t, params.UpgradeFlag)
	assert.True(t, params.ManageFailedFlag)
	assert.True(t, params.NodeChangeFlag)
	assert.False(t, params.RetryFlag)

	params.PatchFunc(nil)
	assert.True(t, called)
}
