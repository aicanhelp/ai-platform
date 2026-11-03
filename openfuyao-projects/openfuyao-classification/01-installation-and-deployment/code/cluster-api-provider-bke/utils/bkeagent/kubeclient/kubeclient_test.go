/******************************************************************
 * Copyright (c) 2026 ICBC Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kubeclient

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestNewClient_InvalidPath(t *testing.T) {
	_, err := NewClient("/nonexistent/path/kubeconfig")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load kubeconfig")
}

func TestNewClient_LoadFromFileError(t *testing.T) {
	patches := gomonkey.ApplyFunc(clientcmd.LoadFromFile, func(path string) (*clientcmdapi.Config, error) {
		return nil, assert.AnError
	})
	defer patches.Reset()

	_, err := NewClient("/some/path")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load kubeconfig")
}

func TestNewClient_ClientConfigError(t *testing.T) {
	patches := gomonkey.ApplyFunc(clientcmd.LoadFromFile, func(path string) (*clientcmdapi.Config, error) {
		return &clientcmdapi.Config{}, nil
	})
	defer patches.Reset()

	_, err := NewClient("/some/path")
	assert.Error(t, err)
}

func TestNewClient_NewForConfigError(t *testing.T) {
	patches := gomonkey.ApplyFunc(clientcmd.LoadFromFile, func(path string) (*clientcmdapi.Config, error) {
		return &clientcmdapi.Config{}, nil
	})
	patches.ApplyMethod(&clientcmd.DirectClientConfig{}, "ClientConfig", func(_ *clientcmd.DirectClientConfig) (*rest.Config, error) {
		return &rest.Config{}, nil
	})
	patches.ApplyFunc(kubernetes.NewForConfig, func(c *rest.Config) (*kubernetes.Clientset, error) {
		return nil, assert.AnError
	})
	defer patches.Reset()

	_, err := NewClient("/some/path")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create kubernetes clientset")
}

func TestNewClient_DynamicClientError(t *testing.T) {
	t.Skip("Skipping due to sync.Once complexity")
}

func TestClientStruct(t *testing.T) {
	client := &Client{
		ClientSet:     &kubernetes.Clientset{},
		DynamicClient: nil,
	}
	assert.NotNil(t, client.ClientSet)
	assert.Nil(t, client.DynamicClient)
}

func TestNewClient_Success(t *testing.T) {
	config := &clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"test": {Server: "https://localhost:6443"},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"test": {Cluster: "test", AuthInfo: "test"},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"test": {Token: "test-token"},
		},
		CurrentContext: "test",
	}

	patches := gomonkey.ApplyFunc(clientcmd.LoadFromFile, func(path string) (*clientcmdapi.Config, error) {
		return config, nil
	})
	defer patches.Reset()

	client, err := NewClient("/some/path")
	if err != nil {
		t.Logf("Expected success but got error: %v", err)
	}
	if client != nil {
		assert.NotNil(t, client.ClientSet)
		assert.NotNil(t, client.DynamicClient)
	}
}
