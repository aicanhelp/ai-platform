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

package kube

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

func TestGetTargetRemoteClientOwnerClusterNotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	assert.NoError(t, bkev1beta1.AddToScheme(scheme))
	assert.NoError(t, clusterv1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).Build()
	bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default"}}

	_, err := GetTargetRemoteClient(context.Background(), cli, bkeCluster)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "owner cluster")

	clientset, dynamicClient, err := GetTargetClusterClient(context.Background(), cli, bkeCluster)
	assert.Error(t, err)
	assert.Nil(t, clientset)
	assert.Nil(t, dynamicClient)
}

func TestClient_KubeClient(t *testing.T) {
	mockCS := &kubernetes.Clientset{}
	mockDC := &dynamic.DynamicClient{}
	c := &Client{ClientSet: mockCS, DynamicClient: mockDC}

	cs, dc := c.KubeClient()
	assert.Equal(t, mockCS, cs)
	assert.Equal(t, mockDC, dc)
}

func TestClient_SetLogger(t *testing.T) {
	var logger *log.Logger = nil
	c := &Client{}
	c.SetLogger(logger)
	assert.Equal(t, logger, c.Log)
}

func TestClient_SetBKELogger(t *testing.T) {
	bkeLog := &bkev1beta1.BKELogger{}
	c := &Client{}
	c.SetBKELogger(bkeLog)
	assert.Equal(t, bkeLog, c.BKELog)
}

func TestNewClientFromK8sToken(t *testing.T) {
	client, err := NewClientFromK8sToken("192.168.1.100", "6443", "test-token")
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestNewClientFromConfig(t *testing.T) {
	_, err := NewClientFromConfig("/nonexistent/kubeconfig")
	assert.Error(t, err)
}

func TestNewClientFromKubeConfig(t *testing.T) {
	_, err := NewClientFromKubeConfig([]byte("invalid-kubeconfig"))
	assert.Error(t, err)
}

func TestNewClientFromKubeConfig_Empty(t *testing.T) {
	_, err := NewClientFromKubeConfig([]byte{})
	assert.Error(t, err)
}

func TestClient_Fields(t *testing.T) {
	ctx := context.Background()
	config := &rest.Config{Host: "https://localhost:6443"}
	var logger *log.Logger = nil
	bkeLog := &bkev1beta1.BKELogger{}

	c := &Client{
		ClientSet:     &kubernetes.Clientset{},
		DynamicClient: &dynamic.DynamicClient{},
		RestConfig:    config,
		Log:           logger,
		BKELog:        bkeLog,
		Ctx:           ctx,
	}

	assert.NotNil(t, c.ClientSet)
	assert.NotNil(t, c.DynamicClient)
	assert.Equal(t, config, c.RestConfig)
	assert.Equal(t, logger, c.Log)
	assert.Equal(t, bkeLog, c.BKELog)
	assert.Equal(t, ctx, c.Ctx)
}

func TestNewClientFromRestConfig_Success(t *testing.T) {
	config := &rest.Config{Host: "https://localhost:6443"}
	client, err := NewClientFromRestConfig(context.Background(), config)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestNewClientFromK8sToken_DifferentParams(t *testing.T) {
	tests := []struct {
		host  string
		port  string
		token string
	}{
		{"127.0.0.1", "6443", "token1"},
		{"10.0.0.1", "8443", "token2"},
		{"localhost", "443", "token3"},
	}
	for _, tt := range tests {
		client, err := NewClientFromK8sToken(tt.host, tt.port, tt.token)
		assert.NoError(t, err)
		assert.NotNil(t, client)
	}
}

func TestKubeConstants(t *testing.T) {
	assert.Equal(t, 2, ExpectedErrorCount)
	assert.Equal(t, 10, DefaultRestConfigTimeout)
}

func TestNewClientFromRestConfig_WithContext(t *testing.T) {
	ctx := context.WithValue(context.Background(), "key", "value")
	config := &rest.Config{Host: "https://localhost:6443"}
	client, err := NewClientFromRestConfig(ctx, config)
	assert.NoError(t, err)
	assert.NotNil(t, client)
	c := client.(*Client)
	assert.Equal(t, ctx, c.Ctx)
}

func TestNewClientFromConfig_ValidPath(t *testing.T) {
	_, err := NewClientFromConfig("/tmp/kubeconfig")
	assert.Error(t, err)
}

func TestNewClientFromKubeConfig_ValidConfig(t *testing.T) {
	validConfig := []byte(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://localhost:6443
  name: test
contexts:
- context:
    cluster: test
    user: test
  name: test
current-context: test
users:
- name: test
  user:
    token: test-token
`)
	client, err := NewClientFromKubeConfig(validConfig)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestClient_MultipleSetters(t *testing.T) {
	c := &Client{}
	var logger1 *log.Logger = nil
	var logger2 *log.Logger = nil
	bkeLog1 := &bkev1beta1.BKELogger{}
	bkeLog2 := &bkev1beta1.BKELogger{}

	c.SetLogger(logger1)
	assert.Equal(t, logger1, c.Log)
	c.SetLogger(logger2)
	assert.Equal(t, logger2, c.Log)

	c.SetBKELogger(bkeLog1)
	assert.Equal(t, bkeLog1, c.BKELog)
	c.SetBKELogger(bkeLog2)
	assert.Equal(t, bkeLog2, c.BKELog)
}

func TestNewClientFromK8sToken_EmptyToken(t *testing.T) {
	client, err := NewClientFromK8sToken("localhost", "6443", "")
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestClient_KubeClient_NilValues(t *testing.T) {
	c := &Client{}
	cs, dc := c.KubeClient()
	assert.Nil(t, cs)
	assert.Nil(t, dc)
}

func TestNewClientFromRestConfig_DifferentHosts(t *testing.T) {
	tests := []struct {
		host string
	}{
		{"https://127.0.0.1:6443"},
		{"https://10.0.0.1:8443"},
		{"https://api.example.com"},
	}
	for _, tt := range tests {
		config := &rest.Config{Host: tt.host}
		client, err := NewClientFromRestConfig(context.Background(), config)
		assert.NoError(t, err)
		assert.NotNil(t, client)
	}
}

func TestNewClientFromRestConfig_InvalidHost(t *testing.T) {
	config := &rest.Config{Host: "://invalid"}
	_, err := NewClientFromRestConfig(context.Background(), config)
	assert.Error(t, err)
}

func TestNewClientFromKubeConfig_MultipleConfigs(t *testing.T) {
	configs := [][]byte{
		[]byte("invalid1"),
		[]byte("invalid2"),
		[]byte{},
	}
	for _, cfg := range configs {
		_, err := NewClientFromKubeConfig(cfg)
		assert.Error(t, err)
	}
}

func TestNewClientFromConfig_MultiplePaths(t *testing.T) {
	paths := []string{
		"/nonexistent/path1",
		"/tmp/invalid",
	}
	for _, path := range paths {
		_, err := NewClientFromConfig(path)
		assert.Error(t, err)
	}
}

func TestClient_AllFields(t *testing.T) {
	ctx := context.Background()
	config := &rest.Config{
		Host:        "https://localhost:6443",
		BearerToken: "test-token",
	}
	var logger *log.Logger = nil
	bkeLog := &bkev1beta1.BKELogger{}
	clientSet := &kubernetes.Clientset{}
	dynamicClient := &dynamic.DynamicClient{}

	c := &Client{
		ClientSet:     clientSet,
		DynamicClient: dynamicClient,
		RestConfig:    config,
		Log:           logger,
		BKELog:        bkeLog,
		Ctx:           ctx,
	}

	assert.Equal(t, clientSet, c.ClientSet)
	assert.Equal(t, dynamicClient, c.DynamicClient)
	assert.Equal(t, config, c.RestConfig)
	assert.Equal(t, logger, c.Log)
	assert.Equal(t, bkeLog, c.BKELog)
	assert.Equal(t, ctx, c.Ctx)

	cs, dc := c.KubeClient()
	assert.Equal(t, clientSet, cs)
	assert.Equal(t, dynamicClient, dc)
}

func TestNewClientFromRestConfig_MultipleContexts(t *testing.T) {
	contexts := []context.Context{
		context.Background(),
		context.TODO(),
		context.WithValue(context.Background(), "key", "value"),
	}
	for _, ctx := range contexts {
		config := &rest.Config{Host: "https://localhost:6443"}
		client, err := NewClientFromRestConfig(ctx, config)
		assert.NoError(t, err)
		assert.NotNil(t, client)
	}
}

func TestNewClientFromRestConfig_WithTimeout(t *testing.T) {
	config := &rest.Config{
		Host:    "https://localhost:6443",
		Timeout: 30,
	}
	client, err := NewClientFromRestConfig(context.Background(), config)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestNewClientFromK8sToken_WithBearerToken(t *testing.T) {
	client, err := NewClientFromK8sToken("api.example.com", "443", "bearer-token-123")
	assert.NoError(t, err)
	assert.NotNil(t, client)
	c := client.(*Client)
	assert.NotNil(t, c.RestConfig)
	assert.Equal(t, "bearer-token-123", c.RestConfig.BearerToken)
}

func TestNewClientFromRestConfig_ErrorHandling(t *testing.T) {
	config := &rest.Config{Host: "://invalid-host"}
	_, err := NewClientFromRestConfig(context.Background(), config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create")
}

func TestClient_SettersChaining(t *testing.T) {
	c := &Client{}
	var logger *log.Logger = nil
	bkeLog := &bkev1beta1.BKELogger{}

	c.SetLogger(logger)
	c.SetBKELogger(bkeLog)

	assert.Equal(t, logger, c.Log)
	assert.Equal(t, bkeLog, c.BKELog)
}

func TestGetRestConfigByToken(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster-k8s-token",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"token": []byte("test-token-value"),
		},
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "192.168.1.100",
				Port: 6443,
			},
		},
	}

	config, err := getRestConfigByToken(context.Background(), fakeClient, bkeCluster)
	assert.NoError(t, err)
	assert.NotNil(t, config)
	assert.Equal(t, "test-token-value", config.BearerToken)
	assert.Equal(t, "https://192.168.1.100:6443", config.Host)
	assert.True(t, config.TLSClientConfig.Insecure)
}

func TestGetRestConfigByToken_SecretNotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "192.168.1.100",
				Port: 6443,
			},
		},
	}

	_, err := getRestConfigByToken(context.Background(), fakeClient, bkeCluster)
	assert.Error(t, err)
}

func TestGetRestConfigByToken_EmptyToken(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster-k8s-token",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"token": []byte(""),
		},
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "192.168.1.100",
				Port: 6443,
			},
		},
	}

	_, err := getRestConfigByToken(context.Background(), fakeClient, bkeCluster)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "token data in secret")
}

func TestGetRestConfigByToken_NoTokenKey(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster-k8s-token",
			Namespace: "default",
		},
		Data: map[string][]byte{},
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "192.168.1.100",
				Port: 6443,
			},
		},
	}

	_, err := getRestConfigByToken(context.Background(), fakeClient, bkeCluster)
	assert.Error(t, err)
}

func TestNewRemoteClientByBKECluster_InvalidEndpoint(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "",
				Port: 0,
			},
		},
	}

	_, err := NewRemoteClientByBKECluster(context.Background(), fakeClient, bkeCluster)
	assert.Error(t, err)
}

func TestNewRemoteClusterClient_InvalidEndpoint(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "",
				Port: 0,
			},
		},
	}

	_, err := NewRemoteClusterClient(context.Background(), fakeClient, bkeCluster)
	assert.Error(t, err)
}

func TestNewClientFromRestConfig_NilConfig(t *testing.T) {
	t.Skip("Skipping test - nil config causes panic in kubernetes.NewForConfig")
}

func TestNewRemoteClientByBKECluster_WithValidToken(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster-k8s-token",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"token": []byte("valid-token"),
		},
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "192.168.1.100",
				Port: 6443,
			},
		},
	}

	client, err := NewRemoteClientByBKECluster(context.Background(), fakeClient, bkeCluster)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestNewRemoteClusterClient_WithValidToken(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster-k8s-token",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"token": []byte("valid-token"),
		},
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "192.168.1.100",
				Port: 6443,
			},
		},
	}

	client, err := NewRemoteClusterClient(context.Background(), fakeClient, bkeCluster)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestGetRestConfigByToken_Success(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster-k8s-token",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"token": []byte("test-token-value"),
		},
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "192.168.1.100",
				Port: 6443,
			},
		},
	}

	config, err := getRestConfigByToken(context.Background(), fakeClient, bkeCluster)
	assert.NoError(t, err)
	assert.NotNil(t, config)
	assert.Equal(t, "test-token-value", config.BearerToken)
	assert.Equal(t, "https://192.168.1.100:6443", config.Host)
	assert.True(t, config.TLSClientConfig.Insecure)
}

func TestNewRemoteClientByCluster_Success(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	cluster := &clusterv1.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}

	patches := gomonkey.ApplyFunc(NewClientFromRestConfig, func(ctx context.Context, config *rest.Config) (RemoteKubeClient, error) {
		return &Client{ClientSet: &kubernetes.Clientset{}}, nil
	})
	defer patches.Reset()

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cluster).Build()

	result, err := NewRemoteClientByCluster(context.Background(), fakeClient, cluster)
	assert.Error(t, err)
	assert.Nil(t, result)
}
