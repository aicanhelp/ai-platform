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

package switchcluster

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	testWorkspace   = "/etc/bkeagent"
	testNamespace   = "test-namespace"
	testSecretName  = "test-secret"
	testNodeName    = "test-node"
	testClusterName = "test-cluster"
	testHostName    = "test-hostname"
)

type mockK8sClientForSwitch struct {
	client.Client
	getCalled  bool
	getError   error
	secretData map[string][]byte
}

func (m *mockK8sClientForSwitch) Get(_ context.Context, _ client.ObjectKey, obj client.Object, _ ...client.GetOption) error {
	m.getCalled = true
	if obj.GetObjectKind().GroupVersionKind().Kind == "Secret" {
		secret := obj.(*corev1.Secret)
		secret.Data = m.secretData
	}
	return m.getError
}

func TestSwitchClusterPluginName(t *testing.T) {
	pluginObj := &SwitchClusterPlugin{}
	assert.Equal(t, Name, pluginObj.Name())
}

func TestSwitchClusterPluginConstantName(t *testing.T) {
	assert.Equal(t, "SwitchCluster", Name)
}

func TestNewSwitchClusterPlugin(t *testing.T) {
	mockClient := &mockK8sClientForSwitch{}
	pluginObj := New(mockClient)
	assert.NotNil(t, pluginObj)
	assert.Equal(t, Name, pluginObj.Name())
}

func TestSwitchClusterPluginParam(t *testing.T) {
	pluginObj := &SwitchClusterPlugin{}
	params := pluginObj.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "kubeconfig")
	assert.Contains(t, params, "nodeName")
	assert.Contains(t, params, "clusterName")
	assert.True(t, params["kubeconfig"].Required)
	assert.False(t, params["nodeName"].Required)
	assert.False(t, params["clusterName"].Required)
}

func TestSwitchClusterPluginParamDefaults(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostName
	})

	pluginObj := &SwitchClusterPlugin{}
	params := pluginObj.Param()
	assert.Equal(t, testHostName, params["nodeName"].Default)
}

func TestSwitchClusterPluginKubeconfigParam(t *testing.T) {
	pluginObj := &SwitchClusterPlugin{}
	params := pluginObj.Param()
	assert.Equal(t, "Kubeconfig in Secret, format ns/secret", params["kubeconfig"].Description)
}

func TestSwitchClusterPluginNodeNameParam(t *testing.T) {
	pluginObj := &SwitchClusterPlugin{}
	params := pluginObj.Param()
	assert.Equal(t, "Switch the cluster to the target cluster nodeName, default os.hostname", params["nodeName"].Description)
}

func TestSwitchClusterPluginClusterNameParam(t *testing.T) {
	pluginObj := &SwitchClusterPlugin{}
	params := pluginObj.Param()
	assert.Equal(t, "Switch the cluster to the target cluster clusterName, default os.hostname", params["clusterName"].Description)
}

func TestSwitchClusterPluginExecuteWithParseCommandsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	parseErr := errors.New("parse commands failed")
	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return nil, parseErr
	})

	mockClient := &mockK8sClientForSwitch{}
	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	_, err := pluginObj.Execute(context.Background(), []string{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parse commands failed")
	assert.False(t, mockClient.getCalled)
}

func TestSwitchClusterPluginExecuteWithK8sClientGetError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return map[string]string{
			"kubeconfig":  fmt.Sprintf("%s/%s", testNamespace, testSecretName),
			"nodeName":    testNodeName,
			"clusterName": testClusterName,
		}, nil
	})

	getErr := errors.New("failed to get secret")
	mockClient := &mockK8sClientForSwitch{}
	mockClient.getError = getErr

	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	_, err := pluginObj.Execute(context.Background(), []string{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get secret")
	assert.True(t, mockClient.getCalled)
}

func TestSwitchClusterPluginExecuteMkdirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mkdirErr := errors.New("mkdir failed")

	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return map[string]string{
			"kubeconfig":  fmt.Sprintf("%s/%s", testNamespace, testSecretName),
			"nodeName":    testNodeName,
			"clusterName": testClusterName,
		}, nil
	})

	patches.ApplyFunc(cache.SplitMetaNamespaceKey, func(key string) (string, string, error) {
		return testNamespace, testSecretName, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(_ string, _ os.FileMode) error {
		return mkdirErr
	})

	mockClient := &mockK8sClientForSwitch{}
	mockClient.getError = nil
	mockClient.secretData = map[string][]byte{
		"kubeconfig": []byte("test-content"),
	}

	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	_, err := pluginObj.Execute(context.Background(), []string{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "mkdir failed")
}

func TestSwitchClusterPluginImplementsPluginInterface(t *testing.T) {
	var _ plugin.Plugin = &SwitchClusterPlugin{}
}

func TestSwitchClusterPluginExecuteWithNodeName(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return map[string]string{
			"kubeconfig":  fmt.Sprintf("%s/%s", testNamespace, testSecretName),
			"nodeName":    "custom-node",
			"clusterName": testClusterName,
		}, nil
	})

	patches.ApplyFunc(cache.SplitMetaNamespaceKey, func(key string) (string, string, error) {
		return testNamespace, testSecretName, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.Exit, func(code int) {
	})

	patches.ApplyFunc(time.AfterFunc, func(d time.Duration, f func()) *time.Timer {
		f()
		return nil
	})

	mockClient := &mockK8sClientForSwitch{}
	mockClient.getError = nil
	mockClient.secretData = map[string][]byte{
		"kubeconfig": []byte("kubeconfig-data"),
	}

	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	result, err := pluginObj.Execute(context.Background(), []string{})

	assert.NoError(t, err)
	assert.LessOrEqual(t, len(result), 1)
}

func TestSwitchClusterPluginExecuteWithClusterName(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return map[string]string{
			"kubeconfig":  fmt.Sprintf("%s/%s", testNamespace, testSecretName),
			"nodeName":    testNodeName,
			"clusterName": "custom-cluster",
		}, nil
	})

	patches.ApplyFunc(cache.SplitMetaNamespaceKey, func(key string) (string, string, error) {
		return testNamespace, testSecretName, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.Exit, func(code int) {
	})

	patches.ApplyFunc(time.AfterFunc, func(d time.Duration, f func()) *time.Timer {
		f()
		return nil
	})

	mockClient := &mockK8sClientForSwitch{}
	mockClient.getError = nil
	mockClient.secretData = map[string][]byte{
		"kubeconfig": []byte("kubeconfig-data"),
	}

	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	result, err := pluginObj.Execute(context.Background(), []string{})

	assert.NoError(t, err)
	assert.LessOrEqual(t, len(result), 1)
}

func TestSwitchClusterPluginExecuteWithSplitMetaNamespaceKeyError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	splitErr := errors.New("invalid namespace key")
	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return map[string]string{
			"kubeconfig":  fmt.Sprintf("%s/%s", testNamespace, testSecretName),
			"nodeName":    testNodeName,
			"clusterName": testClusterName,
		}, nil
	})

	patches.ApplyFunc(cache.SplitMetaNamespaceKey, func(key string) (string, string, error) {
		return "", "", splitErr
	})

	mockClient := &mockK8sClientForSwitch{}

	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	_, err := pluginObj.Execute(context.Background(), []string{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid namespace key")
}

func TestSwitchClusterPluginExecuteWithEmptySecretData(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return map[string]string{
			"kubeconfig":  fmt.Sprintf("%s/%s", testNamespace, testSecretName),
			"nodeName":    testNodeName,
			"clusterName": testClusterName,
		}, nil
	})

	patches.ApplyFunc(cache.SplitMetaNamespaceKey, func(key string) (string, string, error) {
		return testNamespace, testSecretName, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.Exit, func(code int) {
	})

	mockClient := &mockK8sClientForSwitch{}
	mockClient.getError = nil
	mockClient.secretData = map[string][]byte{}

	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	result, err := pluginObj.Execute(context.Background(), []string{})

	assert.NoError(t, err)
	assert.LessOrEqual(t, len(result), 1)
}

func TestSwitchClusterPluginExecuteMultipleSecrets(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return map[string]string{
			"kubeconfig":  fmt.Sprintf("%s/%s", testNamespace, testSecretName),
			"nodeName":    testNodeName,
			"clusterName": testClusterName,
		}, nil
	})

	patches.ApplyFunc(cache.SplitMetaNamespaceKey, func(key string) (string, string, error) {
		return testNamespace, testSecretName, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.Exit, func(code int) {
	})

	mockClient := &mockK8sClientForSwitch{}
	mockClient.getError = nil
	mockClient.secretData = map[string][]byte{
		"kubeconfig": []byte("kubeconfig-content"),
		"ca.crt":     []byte("ca-cert-data"),
		"token":      []byte("token-data"),
	}

	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	result, err := pluginObj.Execute(context.Background(), []string{})

	assert.NoError(t, err)
	assert.LessOrEqual(t, len(result), 1)
}

func TestSwitchClusterPluginExecuteValidKubeconfigKey(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(plugin.ParseCommands, func(_ plugin.Plugin, _ []string) (map[string]string, error) {
		return map[string]string{
			"kubeconfig":  "production/cluster-kubeconfig",
			"nodeName":    "prod-node-01",
			"clusterName": "prod-cluster",
		}, nil
	})

	patches.ApplyFunc(cache.SplitMetaNamespaceKey, func(key string) (string, string, error) {
		return "production", "cluster-kubeconfig", nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.Exit, func(code int) {
	})

	mockClient := &mockK8sClientForSwitch{}
	mockClient.getError = nil
	mockClient.secretData = map[string][]byte{
		"kubeconfig": []byte("prod-kubeconfig"),
	}

	pluginObj := &SwitchClusterPlugin{K8sClient: mockClient}

	result, err := pluginObj.Execute(context.Background(), []string{})

	assert.NoError(t, err)
	assert.LessOrEqual(t, len(result), 1)
}
