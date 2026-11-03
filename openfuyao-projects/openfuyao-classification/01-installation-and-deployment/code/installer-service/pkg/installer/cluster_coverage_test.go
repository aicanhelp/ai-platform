/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package installer

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	k8sfake "k8s.io/client-go/kubernetes/fake"

	"installer-service/pkg/utils/k8sutil"
)

func makeCoverageBKECluster(name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKECluster",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": name,
		},
		"spec": map[string]interface{}{
			"controlPlaneEndpoint": map[string]interface{}{"host": "1.2.3.4", "port": float64(6443)},
			"clusterConfig": map[string]interface{}{
				"cluster": map[string]interface{}{
					"openFuyaoVersion":  "v0",
					"kubernetesVersion": "v1.26.0",
					"containerRuntime":  map[string]interface{}{"CRI": "containerd"},
				},
			},
		},
		"status": map[string]interface{}{
			"clusterHealthState": "Healthy",
			"clusterStatus":      "Managing",
		},
	}}
}

func newCoverageClient(cluster *unstructured.Unstructured) (*installerClient, *gomonkey.Patches) {
	scheme := runtime.NewScheme()
	listKinds := map[schema.GroupVersionResource]string{
		testGVR:            "BKEClusterList",
		bkeNodeGVR:         "BKENodeList",
	}
	dynamicClient := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds, cluster)
	clientset := k8sfake.NewSimpleClientset()
	patches := gomonkey.ApplyGlobalVar(&gvr, testGVR)
	client := &installerClient{
		dynamicClient: dynamicClient,
		clientset:     clientset,
	}
	return client, patches
}

func TestGetAllClusters(t *testing.T) {
	cluster := makeCoverageBKECluster("list-cluster")
	client, patches := newCoverageClient(cluster)
	defer patches.Reset()

	resp, err := client.GetAllClusters()
	require.NoError(t, err)
	require.Len(t, resp.Items, 1)
	assert.Equal(t, "list-cluster", resp.Items[0].Name)
}

func TestGetClusterFull(t *testing.T) {
	clusterName := "full-cluster"
	cluster := makeCoverageBKECluster(clusterName)
	client, patches := newCoverageClient(cluster)
	defer patches.Reset()

	patches.ApplyFunc(k8sutil.GetConfigMap, func(_ kubernetes.Interface, _, _ string) (*corev1.ConfigMap, error) {
		return &corev1.ConfigMap{
			Data: map[string]string{
				"domain": "docker.io",
				"host":   "10.0.0.1",
			},
		}, nil
	})

	resp, err := client.GetClusterFull(clusterName)
	require.NoError(t, err)
	assert.Equal(t, clusterName, resp.ClusterConfig.ClusterName)
}

func TestUploadPatchFile(t *testing.T) {
	validYAML := "openFuyaoVersion: v25.09.1sp2\nkubernetesVersion: v1.28.0\ncontainerdVersion: v1.6.0\netcdVersion: v3.5.0\n"

	t.Run("online模式拒绝上传", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: BKEConfigCmKey().Name, Namespace: BKEConfigCmKey().Namespace},
			Data:       map[string]string{"otherRepo": "repo.example.com", "onlineImage": ""},
		}
		client := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}
		err := client.UploadPatchFile("v1.yaml", validYAML)
		assert.ErrorContains(t, err, "online mode detected")
	})

	t.Run("离线模式上传成功", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: BKEConfigCmKey().Name, Namespace: BKEConfigCmKey().Namespace},
			Data:       map[string]string{"otherRepo": "", "onlineImage": ""},
		}
		client := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}
		patches := gomonkey.ApplyFunc(ensureNsExists, func(client kubernetes.Interface, ns string) error {
			_, err := client.CoreV1().Namespaces().Get(context.TODO(), ns, metav1.GetOptions{})
			if err != nil {
				_, err = client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
					ObjectMeta: metav1.ObjectMeta{Name: ns},
				}, metav1.CreateOptions{})
			}
			return err
		})
		defer patches.Reset()

		err := client.UploadPatchFile("v25.09.1sp2.yaml", validYAML)
		assert.NoError(t, err)
	})

	t.Run("无效YAML返回错误", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: BKEConfigCmKey().Name, Namespace: BKEConfigCmKey().Namespace},
			Data:       map[string]string{"otherRepo": "", "onlineImage": ""},
		}
		client := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}
		err := client.UploadPatchFile("bad.yaml", "{invalid yaml")
		assert.Error(t, err)
	})
}

func TestGetDefaultConfigFromClientset(t *testing.T) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: BKEConfigCmKey().Name, Namespace: BKEConfigCmKey().Namespace},
		Data: map[string]string{
			"domain": configinit.DefaultYumRepo,
			"host":   "192.168.1.1",
		},
	}
	client := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}

	resp, err := client.GetDefaultConfig()
	require.NoError(t, err)
	assert.Equal(t, "192.168.1.1", resp.Ip)
}

func TestGetInstallableVersionsServiceError(t *testing.T) {
	client := &installerClient{clientset: k8sfake.NewSimpleClientset()}
	_, err := client.GetInstallableVersions()
	assert.Error(t, err)
}

func TestGetOpenFuyaoVersionsServiceError(t *testing.T) {
	client := &installerClient{clientset: k8sfake.NewSimpleClientset()}
	_, err := client.GetOpenFuyaoVersions()
	assert.Error(t, err)
}
