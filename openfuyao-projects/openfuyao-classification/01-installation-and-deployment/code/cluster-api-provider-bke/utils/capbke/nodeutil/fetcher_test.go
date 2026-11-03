/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package nodeutil

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

type mockClient struct {
	client.Client
}

func TestNewNodeFetcher(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)
	assert.NotNil(t, fetcher)
	assert.NotNil(t, fetcher.client)
}

func TestFetchNodesForCluster_Success(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	result, err := fetcher.FetchNodesForCluster(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestFetchNodesForCluster_Error(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		return assert.AnError
	})
	defer patches.Reset()

	result, err := fetcher.FetchNodesForCluster(context.Background(), "default", "test-cluster")
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestFetchNodesForBKECluster(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	result, err := fetcher.FetchNodesForBKECluster(context.Background(), cluster)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetNodes(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	nodes, err := fetcher.GetNodes(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, nodes)
}

func TestGetNodesForBKECluster(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	nodes, err := fetcher.GetNodesForBKECluster(context.Background(), cluster)
	assert.NoError(t, err)
	assert.NotNil(t, nodes)
}

func TestGetBKENodes(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	bkeNodes, err := fetcher.GetBKENodes(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, bkeNodes)
}

func TestGetBKENodesForBKECluster(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	bkeNodes, err := fetcher.GetBKENodesForBKECluster(context.Background(), cluster)
	assert.NoError(t, err)
	assert.NotNil(t, bkeNodes)
}

func TestGetNodeByIP_NotFound(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	node, err := fetcher.GetNodeByIP(context.Background(), "default", "test-cluster", "192.168.1.1")
	assert.Error(t, err)
	assert.Nil(t, node)
	assert.Contains(t, err.Error(), "not found")
}

func TestGetNodeByIP_Found(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}},
		}
		return nil
	})
	defer patches.Reset()

	node, err := fetcher.GetNodeByIP(context.Background(), "default", "test-cluster", "192.168.1.1")
	assert.NoError(t, err)
	assert.NotNil(t, node)
	assert.Equal(t, "192.168.1.1", node.Spec.IP)
}

func TestGetNodesFromClient(t *testing.T) {
	c := &mockClient{}

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	nodes, err := GetNodesFromClient(context.Background(), c, "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, nodes)
}

func TestGetNodesForBKEClusterFromClient(t *testing.T) {
	c := &mockClient{}

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	nodes, err := GetNodesForBKEClusterFromClient(context.Background(), c, cluster)
	assert.NoError(t, err)
	assert.NotNil(t, nodes)
}

func TestGetBKENodesFromClient(t *testing.T) {
	c := &mockClient{}

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	bkeNodes, err := GetBKENodesFromClient(context.Background(), c, "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, bkeNodes)
}

func TestGetNodeStates(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}},
		}
		return nil
	})
	defer patches.Reset()

	states, err := fetcher.GetNodeStates(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, states)
	assert.Len(t, states, 1)
}

func TestGetNodeStatesForBKECluster(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	states, err := fetcher.GetNodeStatesForBKECluster(context.Background(), cluster)
	assert.NoError(t, err)
	assert.NotNil(t, states)
}

func TestGetDeletingNodes(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	now := metav1.Now()
	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{
			{
				ObjectMeta: metav1.ObjectMeta{DeletionTimestamp: &now},
				Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
			},
		}
		return nil
	})
	defer patches.Reset()

	nodes, err := fetcher.GetDeletingNodes(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
}

func TestGetDeletingNodesForBKECluster(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	nodes, err := fetcher.GetDeletingNodesForBKECluster(context.Background(), cluster)
	assert.NoError(t, err)
	assert.Empty(t, nodes)
}

func TestFilterNodesByState(t *testing.T) {
	states := []NodeStateInfo{
		{State: confv1beta1.NodeReady},
		{State: confv1beta1.NodeNotReady},
		{State: confv1beta1.NodeReady},
	}

	filtered := FilterNodesByState(states, confv1beta1.NodeReady)
	assert.Len(t, filtered, 2)
}

func TestGetReadyNodes(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{
			{
				Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
				Status: confv1beta1.BKENodeStatus{State: confv1beta1.NodeReady},
			},
		}
		return nil
	})
	defer patches.Reset()

	nodes, err := fetcher.GetReadyNodes(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
}

func TestGetNodesExcludingSkipped(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{
			{
				Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
				Status: confv1beta1.BKENodeStatus{NeedSkip: false},
			},
		}
		return nil
	})
	defer patches.Reset()

	nodes, err := fetcher.GetNodesExcludingSkipped(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
}

func TestGetAllNodes(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	nodes, err := fetcher.GetAllNodes(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, nodes)
}

func TestGetReadyBootstrapNodes(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	nodes, err := fetcher.GetReadyBootstrapNodes(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.Empty(t, nodes)
}

func TestCompareNodes(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}},
		}
		return nil
	})
	defer patches.Reset()

	desiredNodes := []confv1beta1.Node{
		{IP: "192.168.1.1"},
		{IP: "192.168.1.2"},
	}

	added, removed, err := fetcher.CompareNodes(context.Background(), "default", "test-cluster", desiredNodes)
	assert.NoError(t, err)
	assert.Len(t, added, 1)
	assert.Empty(t, removed)
}

func TestGetBKENodesWrapper(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	wrapper, err := fetcher.GetBKENodesWrapper(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.NotNil(t, wrapper)
}

func TestGetBKENodesWrapperForCluster(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	wrapper, err := fetcher.GetBKENodesWrapperForCluster(context.Background(), cluster)
	assert.NoError(t, err)
	assert.NotNil(t, wrapper)
}

func TestGetNodeCount(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{{}, {}}
		return nil
	})
	defer patches.Reset()

	count, err := fetcher.GetNodeCount(context.Background(), "default", "test-cluster")
	assert.NoError(t, err)
	assert.Equal(t, 2, count)
}

func TestGetNodeCountForCluster(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	count, err := fetcher.GetNodeCountForCluster(context.Background(), cluster)
	assert.NoError(t, err)
	assert.Equal(t, 0, count)
}

func TestDeleteBKENode_NotFound(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	err := fetcher.DeleteBKENode(context.Background(), "default", "test-cluster", "192.168.1.1")
	assert.NoError(t, err)
}

func TestDeleteBKENodeForCluster(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "List", func(_ *mockClient, ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
		nodeList := list.(*confv1beta1.BKENodeList)
		nodeList.Items = []confv1beta1.BKENode{}
		return nil
	})
	defer patches.Reset()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	err := fetcher.DeleteBKENodeForCluster(context.Background(), cluster, "192.168.1.1")
	assert.NoError(t, err)
}

func TestUpdateModifiedNodes(t *testing.T) {
	c := &mockClient{}
	fetcher := NewNodeFetcher(c)

	patches := gomonkey.ApplyMethod(c, "Status", func(_ *mockClient) client.StatusWriter {
		return &mockStatusWriter{}
	})
	defer patches.Reset()

	nodes := bkev1beta1.BKENodes{}
	err := fetcher.UpdateModifiedNodes(context.Background(), nodes)
	assert.NoError(t, err)
}

type mockStatusWriter struct {
	client.StatusWriter
}
