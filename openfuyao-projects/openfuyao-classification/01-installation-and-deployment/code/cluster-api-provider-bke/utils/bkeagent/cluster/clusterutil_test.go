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

package cluster

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/kubeclient"
)

func TestGetClusterData_InvalidInput(t *testing.T) {
	_, err := GetClusterData("", "")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid namespace or name")
}

func TestGetClusterData_EmptyNamespace(t *testing.T) {
	_, err := GetClusterData("", "test")
	assert.Error(t, err)
}

func TestGetClusterData_EmptyName(t *testing.T) {
	_, err := GetClusterData("default", "")
	assert.Error(t, err)
}

func TestGetClusterData_ClientError(t *testing.T) {
	patches := gomonkey.ApplyFunc(kubeclient.NewClient, func(path string) (*kubeclient.Client, error) {
		return nil, assert.AnError
	})
	defer patches.Reset()

	_, err := GetClusterData("default", "test")
	assert.Error(t, err)
}

func TestGetNodesData_Success(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetClusterData, func(namespace, name string) (*ClusterData, error) {
		return &ClusterData{
			Cluster: &bkev1beta1.BKECluster{},
			Nodes:   bkenode.Nodes{},
		}, nil
	})
	defer patches.Reset()

	nodes, err := GetNodesData("default", "test")
	assert.NoError(t, err)
	assert.NotNil(t, nodes)
}

func TestGetNodesData_Error(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetClusterData, func(namespace, name string) (*ClusterData, error) {
		return nil, assert.AnError
	})
	defer patches.Reset()

	nodes, err := GetNodesData("default", "test")
	assert.Error(t, err)
	assert.Nil(t, nodes)
}

func TestClusterDataStruct(t *testing.T) {
	cd := &ClusterData{
		Cluster: &bkev1beta1.BKECluster{},
		Nodes:   bkenode.Nodes{},
	}
	assert.NotNil(t, cd.Cluster)
	assert.NotNil(t, cd.Nodes)
}

func TestGetClusterData_ValidInput(t *testing.T) {
	patches := gomonkey.ApplyFunc(kubeclient.NewClient, func(path string) (*kubeclient.Client, error) {
		return nil, assert.AnError
	})
	defer patches.Reset()

	result, err := GetClusterData("test-ns", "test-cluster")
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetNodesData_ValidInput(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetClusterData, func(namespace, name string) (*ClusterData, error) {
		return &ClusterData{
			Cluster: &bkev1beta1.BKECluster{},
			Nodes:   bkenode.Nodes{{}, {}},
		}, nil
	})
	defer patches.Reset()

	nodes, err := GetNodesData("test-ns", "test-cluster")
	assert.NoError(t, err)
	assert.Len(t, nodes, 2)
}
