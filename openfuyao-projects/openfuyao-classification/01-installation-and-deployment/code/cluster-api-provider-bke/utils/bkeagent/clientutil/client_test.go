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

package clientutil

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/kubeclient"
)

func TestClientFields(t *testing.T) {
	client := &Client{
		ClientSet:     &kubernetes.Clientset{},
		DynamicClient: nil,
	}

	assert.NotNil(t, client.ClientSet)
	assert.Nil(t, client.DynamicClient)
}

func TestWorkspaceConstant(t *testing.T) {
	expected := "/etc/openFuyao/bkeagent"
	assert.Equal(t, expected, utils.Workspace)
}

func TestNamespaceAndNameLenConstant(t *testing.T) {
	assert.Equal(t, 2, utils.NamespaceAndNameLen)
}

func TestNewKubernetesClient_Success(t *testing.T) {
	patches := gomonkey.ApplyFunc(kubeclient.NewClient, func(path string) (*kubeclient.Client, error) {
		return &kubeclient.Client{}, nil
	})
	defer patches.Reset()

	client, err := NewKubernetesClient("/tmp/kubeconfig")
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestNewKubernetesClient_Error(t *testing.T) {
	patches := gomonkey.ApplyFunc(kubeclient.NewClient, func(path string) (*kubeclient.Client, error) {
		return nil, assert.AnError
	})
	defer patches.Reset()

	client, err := NewKubernetesClient("/tmp/kubeconfig")
	assert.Error(t, err)
	assert.Nil(t, client)
}

func TestClientSetFromManagerClusterSecret_InvalidPath(t *testing.T) {
	_, err := ClientSetFromManagerClusterSecret()
	assert.Error(t, err)
}
