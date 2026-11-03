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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type fakeClient struct {
	client.Client
}

type fakeRecorder struct{}

func (f *fakeRecorder) Event(object runtime.Object, eventtype, reason, message string) {}
func (f *fakeRecorder) Eventf(object runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
}
func (f *fakeRecorder) AnnotatedEventf(object runtime.Object, annotations map[string]string, eventtype, reason, messageFmt string, args ...interface{}) {
}

func (f *fakeClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	return nil
}

func (f *fakeClient) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	return nil
}

func (f *fakeClient) Delete(ctx context.Context, obj client.Object, opts ...client.DeleteOption) error {
	return nil
}

func (f *fakeClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	return nil
}

func createTestLogger() *bkev1beta1.BKELogger {
	sugar := (*log.Logger)(nil)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	return bkev1beta1.NewBKELogger(sugar, &fakeRecorder{}, bkeCluster)
}

func TestCommonFunctions(t *testing.T) {
	assert.Equal(t, 2, MasterInitSleepSeconds)
}

func TestProcessNodeMachineMapping_EmptyNodes(t *testing.T) {
	logger := createTestLogger()
	params := ProcessNodeMachineMappingParams{
		Ctx:        context.Background(),
		BKECluster: &bkev1beta1.BKECluster{},
		Nodes:      []confv1beta1.Node{},
		Log:        logger,
	}

	result, err := ProcessNodeMachineMapping(params)
	assert.NoError(t, err)
	assert.Empty(t, result.DeleteMap)
	assert.Empty(t, result.WaitDeleteMap)
	assert.Equal(t, 0, result.NodesCount)
}

func TestProcessCommandFailure_RefreshError(t *testing.T) {
	params := ProcessCommandFailureParams{
		RefreshContext: func() error {
			return assert.AnError
		},
	}

	result := ProcessCommandFailure(params)
	assert.False(t, result.Done)
	assert.False(t, result.Success)
	assert.Error(t, result.Err)
}

func TestProcessNodeMachineMapping_WithNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	logger := createTestLogger()
	params := ProcessNodeMachineMappingParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Nodes:      []confv1beta1.Node{{IP: "192.168.1.1"}},
		Log:        logger,
	}

	patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
		return &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}}, nil
	})

	result, err := ProcessNodeMachineMapping(params)
	assert.NoError(t, err)
	assert.Len(t, result.DeleteMap, 1)
}

func TestProcessNodeMachineMapping_NilParams(t *testing.T) {
	logger := createTestLogger()
	params := ProcessNodeMachineMappingParams{
		Ctx:        context.Background(),
		BKECluster: &bkev1beta1.BKECluster{},
		Nodes:      nil,
		Log:        logger,
	}

	result, err := ProcessNodeMachineMapping(params)
	assert.NoError(t, err)
	assert.Equal(t, 0, result.NodesCount)
}

func TestProcessCommandFailureParams_Structure(t *testing.T) {
	params := ProcessCommandFailureParams{
		Context:    context.Background(),
		BKECluster: &bkev1beta1.BKECluster{},
		InitCommand: &agentv1beta1.Command{
			ObjectMeta: metav1.ObjectMeta{Name: "test"},
		},
	}
	assert.NotNil(t, params.Context)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.InitCommand)
}

func TestProcessNodeMachineMappingResult_Structure(t *testing.T) {
	result := ProcessNodeMachineMappingResult{
		DeleteMap: map[string]phaseutil.MachineAndNode{
			"node1": {Machine: &clusterv1.Machine{}},
		},
		WaitDeleteMap: map[string]phaseutil.MachineAndNode{
			"node2": {Machine: &clusterv1.Machine{}},
		},
		NodesCount: 2,
	}
	assert.Len(t, result.DeleteMap, 1)
	assert.Len(t, result.WaitDeleteMap, 1)
	assert.Equal(t, 2, result.NodesCount)
}

func TestProcessCommandFailureResult_Structure(t *testing.T) {
	result := ProcessCommandFailureResult{
		Done:    true,
		Success: false,
		Err:     assert.AnError,
	}
	assert.True(t, result.Done)
	assert.False(t, result.Success)
	assert.Error(t, result.Err)
}

func TestProcessNodeMachineMapping_NodeFetcherNil(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	logger := createTestLogger()
	params := ProcessNodeMachineMappingParams{
		Ctx:               context.Background(),
		Client:            &fakeClient{},
		BKECluster:        &bkev1beta1.BKECluster{},
		NodeFetcher:       nil,
		Nodes:             []confv1beta1.Node{{IP: "192.168.1.1"}},
		Log:               logger,
		NodeDeletedReason: "NodeDeleted",
		NodeJoinedReason:  "NodeJoined",
	}

	patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
		return nil, assert.AnError
	})
	patches.ApplyFunc(shouldSkipOrphanNodeCleanup, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeIP string) (bool, string) {
		return false, ""
	})

	patches.ApplyMethod(&statusmanage.StatusManager{}, "RemoveSingleNodeStatusCache", func(_ *statusmanage.StatusManager, bkeCluster *bkev1beta1.BKECluster, nodeIP string) {
	})

	patches.ApplyFunc(phaseutil.RemoveAppointmentDeletedNodes, func(cluster *bkev1beta1.BKECluster, nodeIP string) {
	})

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return nil
	})

	result, err := ProcessNodeMachineMapping(params)
	assert.NoError(t, err)
	assert.Empty(t, result.DeleteMap)
}

func TestProcessNodeMachineMapping_SyncStatusError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	logger := createTestLogger()
	params := ProcessNodeMachineMappingParams{
		Ctx:               context.Background(),
		Client:            &fakeClient{},
		BKECluster:        &bkev1beta1.BKECluster{},
		NodeFetcher:       nil,
		Nodes:             []confv1beta1.Node{{IP: "192.168.1.1"}},
		Log:               logger,
		NodeDeletedReason: "NodeDeleted",
		NodeJoinedReason:  "NodeJoined",
	}

	patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
		return nil, assert.AnError
	})
	// Allow orphan cleanup to proceed past the target-cluster guard so SyncStatusUntilComplete is reached.
	patches.ApplyFunc(shouldSkipOrphanNodeCleanup, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeIP string) (bool, string) {
		return false, ""
	})

	patches.ApplyMethod(&statusmanage.StatusManager{}, "RemoveSingleNodeStatusCache", func(_ *statusmanage.StatusManager, bkeCluster *bkev1beta1.BKECluster, nodeIP string) {
	})

	patches.ApplyFunc(phaseutil.RemoveAppointmentDeletedNodes, func(cluster *bkev1beta1.BKECluster, nodeIP string) {
	})

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return assert.AnError
	})

	result, err := ProcessNodeMachineMapping(params)
	assert.Error(t, err)
	assert.Empty(t, result.DeleteMap)
}

func TestProcessNodeMachineMapping_MachineDeleting(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	logger := createTestLogger()
	params := ProcessNodeMachineMappingParams{
		Ctx:               context.Background(),
		Client:            &fakeClient{},
		BKECluster:        &bkev1beta1.BKECluster{},
		Nodes:             []confv1beta1.Node{{IP: "192.168.1.1"}},
		Log:               logger,
		NodeDeletedReason: "NodeDeleted",
	}

	patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
		return &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
			Status: clusterv1.MachineStatus{
				Phase: string(clusterv1.MachinePhaseDeleting),
			},
		}, nil
	})

	result, err := ProcessNodeMachineMapping(params)
	assert.NoError(t, err)
	assert.Len(t, result.WaitDeleteMap, 1)
	assert.Empty(t, result.DeleteMap)
}

func TestProcessNodeMachineMapping_MachineDeleted(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	logger := createTestLogger()
	params := ProcessNodeMachineMappingParams{
		Ctx:               context.Background(),
		Client:            &fakeClient{},
		BKECluster:        &bkev1beta1.BKECluster{},
		Nodes:             []confv1beta1.Node{{IP: "192.168.1.1"}},
		Log:               logger,
		NodeDeletedReason: "NodeDeleted",
	}

	patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
		return &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
			Status: clusterv1.MachineStatus{
				Phase: string(clusterv1.MachinePhaseDeleted),
			},
		}, nil
	})

	result, err := ProcessNodeMachineMapping(params)
	assert.NoError(t, err)
	assert.Empty(t, result.WaitDeleteMap)
	assert.Empty(t, result.DeleteMap)
}

func TestGetTargetClusterNodes_GetClientError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(kube.GetTargetClusterClient, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
		return nil, nil, assert.AnError
	})

	nodes, err := GetTargetClusterNodes(context.Background(), &fakeClient{}, &bkev1beta1.BKECluster{})
	assert.Error(t, err)
	assert.Nil(t, nodes)
}

// newFakeTargetBKENodeClient 构造一个 dynamic 假客户端，模拟目标集群中指定状态的 BKENode
func newFakeTargetBKENodeClient(clusterName string, state confv1beta1.NodeState) dynamic.Interface {
	bn := &unstructured.Unstructured{}
	bn.SetUnstructuredContent(map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKENode",
		"metadata": map[string]interface{}{
			"name":      "bn-node1",
			"namespace": "test-ns",
			"labels":    map[string]interface{}{clusterv1.ClusterNameLabel: clusterName},
		},
		"spec":   map[string]interface{}{"ip": "192.168.1.1", "hostname": "node1"},
		"status": map[string]interface{}{"state": string(state)},
	})
	return dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
		runtime.NewScheme(),
		map[schema.GroupVersionResource]string{bkenodeGVR: "BKENodeList"},
		bn,
	)
}

// TestProcessNodeMachineMapping_TargetClusterGuard 验证孤儿节点清理前的二次确认护栏：
// 目标集群仍存在活跃 BKENode 时必须跳过清理，防止第三方视角（如自纳管后引导集群
// 残留控制器）的陈旧状态误删目标集群中刚加入的 Node。
func TestProcessNodeMachineMapping_TargetClusterGuard(t *testing.T) {
	newParams := func() ProcessNodeMachineMappingParams {
		return ProcessNodeMachineMappingParams{
			Ctx:               context.Background(),
			Client:            &fakeClient{},
			BKECluster:        &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "test-ns"}},
			NodeFetcher:       nil,
			Nodes:             []confv1beta1.Node{{IP: "192.168.1.1", Hostname: "node1"}},
			Log:               createTestLogger(),
			NodeDeletedReason: "NodeDeleted",
			NodeJoinedReason:  "NodeJoined",
		}
	}

	t.Run("active BKENode in target cluster blocks orphan cleanup", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(kube.GetTargetClusterClient, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return nil, newFakeTargetBKENodeClient("test", confv1beta1.NodeReady), nil
		})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
			return nil, assert.AnError
		})
		deleteCalls := 0
		patches.ApplyFunc(DeleteTargetClusterNode, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeName string) error {
			deleteCalls++
			return nil
		})
		syncCalls := 0
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			syncCalls++
			return nil
		})

		result, err := ProcessNodeMachineMapping(newParams())
		assert.NoError(t, err)
		assert.Empty(t, result.DeleteMap)
		assert.Equal(t, 0, syncCalls, "orphan cleanup must be skipped when the target cluster still manages the node")
		assert.Equal(t, 0, deleteCalls, "must not delete target Node when active BKENode exists")
	})

	t.Run("BKENode for other cluster label does not block orphan cleanup", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(kube.GetTargetClusterClient, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return nil, newFakeTargetBKENodeClient("other-cluster", confv1beta1.NodeReady), nil
		})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
			return nil, assert.AnError
		})
		patches.ApplyFunc(DeleteTargetClusterNode, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeName string) error {
			return nil
		})
		patches.ApplyMethod(&statusmanage.StatusManager{}, "RemoveSingleNodeStatusCache", func(_ *statusmanage.StatusManager, bkeCluster *bkev1beta1.BKECluster, nodeIP string) {
		})
		patches.ApplyFunc(phaseutil.RemoveAppointmentDeletedNodes, func(cluster *bkev1beta1.BKECluster, nodeIP string) {
		})
		syncCalls := 0
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			syncCalls++
			return nil
		})

		_, err := ProcessNodeMachineMapping(newParams())
		assert.NoError(t, err)
		assert.Equal(t, 1, syncCalls, "orphan cleanup should proceed when BKENode belongs to another cluster")
	})

	t.Run("Deleting-state BKENode in target cluster does not block orphan cleanup", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(kube.GetTargetClusterClient, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return nil, newFakeTargetBKENodeClient("test", confv1beta1.NodeDeleting), nil
		})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
			return nil, assert.AnError
		})
		patches.ApplyFunc(DeleteTargetClusterNode, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeName string) error {
			return nil
		})
		patches.ApplyMethod(&statusmanage.StatusManager{}, "RemoveSingleNodeStatusCache", func(_ *statusmanage.StatusManager, bkeCluster *bkev1beta1.BKECluster, nodeIP string) {
		})
		patches.ApplyFunc(phaseutil.RemoveAppointmentDeletedNodes, func(cluster *bkev1beta1.BKECluster, nodeIP string) {
		})
		syncCalls := 0
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			syncCalls++
			return nil
		})

		_, err := ProcessNodeMachineMapping(newParams())
		assert.NoError(t, err)
		assert.Equal(t, 1, syncCalls, "orphan cleanup should proceed when the target BKENode is already Deleting")
	})

	t.Run("transient list error fail-closed skips orphan cleanup", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(kube.GetTargetClusterClient, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return nil, nil, assert.AnError
		})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
			return nil, assert.AnError
		})
		deleteCalls := 0
		patches.ApplyFunc(DeleteTargetClusterNode, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeName string) error {
			deleteCalls++
			return nil
		})
		syncCalls := 0
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			syncCalls++
			return nil
		})

		_, err := ProcessNodeMachineMapping(newParams())
		assert.NoError(t, err)
		assert.Equal(t, 0, syncCalls, "fail-closed must skip orphan cleanup on target client errors")
		assert.Equal(t, 0, deleteCalls)
	})

	t.Run("missing BKENode CRD fail-open allows orphan cleanup", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(kube.GetTargetClusterClient, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return nil, &crdMissingDynamic{err: apierrors.NewNotFound(schema.GroupResource{Group: "bke.bocloud.com", Resource: "bkenodes"}, "")}, nil
		})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
			return nil, assert.AnError
		})
		patches.ApplyFunc(DeleteTargetClusterNode, func(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeName string) error {
			return nil
		})
		patches.ApplyMethod(&statusmanage.StatusManager{}, "RemoveSingleNodeStatusCache", func(_ *statusmanage.StatusManager, bkeCluster *bkev1beta1.BKECluster, nodeIP string) {
		})
		patches.ApplyFunc(phaseutil.RemoveAppointmentDeletedNodes, func(cluster *bkev1beta1.BKECluster, nodeIP string) {
		})
		syncCalls := 0
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			syncCalls++
			return nil
		})

		_, err := ProcessNodeMachineMapping(newParams())
		assert.NoError(t, err)
		assert.Equal(t, 1, syncCalls, "missing BKENode CRD must fail-open and keep orphan cleanup")
	})
}

// crdMissingDynamic returns NotFound on every List to simulate missing BKENode CRD.
type crdMissingDynamic struct {
	dynamic.Interface
	err error
}

func (c *crdMissingDynamic) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	return &crdMissingNamespaceable{err: c.err}
}

type crdMissingNamespaceable struct {
	dynamic.NamespaceableResourceInterface
	err error
}

func (c *crdMissingNamespaceable) Namespace(string) dynamic.ResourceInterface {
	return &crdMissingResource{err: c.err}
}

type crdMissingResource struct {
	dynamic.ResourceInterface
	err error
}

func (c *crdMissingResource) List(ctx context.Context, opts metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	return nil, c.err
}

func TestIsBKENodeCRDUnavailable(t *testing.T) {
	assert.True(t, isBKENodeCRDUnavailable(apierrors.NewNotFound(schema.GroupResource{Group: "bke.bocloud.com", Resource: "bkenodes"}, "")))
	assert.False(t, isBKENodeCRDUnavailable(assert.AnError))
	assert.False(t, isBKENodeCRDUnavailable(nil))
}
