/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	fakedynamic "k8s.io/client-go/dynamic/fake"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestResolveKubeletConfigNamespace(t *testing.T) {
	assert.Equal(t, "default", resolveKubeletConfigNamespace(&confv1beta1.KubeletConfigRef{Name: "k"}, "default"))
	assert.Equal(t, "custom", resolveKubeletConfigNamespace(&confv1beta1.KubeletConfigRef{Name: "k", Namespace: "custom"}, "default"))
}

func TestResolveContainerdConfigNamespace(t *testing.T) {
	assert.Equal(t, "default", resolveContainerdConfigNamespace(&confv1beta1.ContainerdConfigRef{Name: "c"}, "default"))
	assert.Equal(t, "custom", resolveContainerdConfigNamespace(&confv1beta1.ContainerdConfigRef{Name: "c", Namespace: "custom"}, "default"))
}

func TestSyncReferencedKubeletAndContainerdConfig_NoRefs(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
	}
	err := SyncReferencedKubeletAndContainerdConfig(context.Background(), nil, fakeclientset.NewSimpleClientset(), fakedynamic.NewSimpleDynamicClient(runtime.NewScheme()), cluster)
	require.NoError(t, err)
}

func TestSyncReferencedKubeletAndContainerdConfig_SyncsReferencedCRs(t *testing.T) {
	kct := &unstructured.Unstructured{}
	kct.SetGroupVersionKind(confv1beta1.KubeletConfigGVK())
	kct.SetName("my-kubelet")
	kct.SetNamespace("default")
	require.NoError(t, unstructured.SetNestedMap(kct.Object, map[string]interface{}{
		"raw": "apiVersion: kubelet.config.k8s.io/v1beta1",
	}, "spec", "kubeletConfig", "kubelet.conf"))

	cct := &unstructured.Unstructured{}
	cct.SetGroupVersionKind(confv1beta1.ContainerdConfigGVK())
	cct.SetName("my-containerd")
	cct.SetNamespace("default")
	require.NoError(t, unstructured.SetNestedMap(cct.Object, map[string]interface{}{
		"configType": "main",
	}, "spec"))

	scheme := runtime.NewScheme()
	localClient := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(kct, cct).Build()

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			KubeletConfigRef: &confv1beta1.KubeletConfigRef{Name: "my-kubelet"},
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					ContainerdConfigRef: &confv1beta1.ContainerdConfigRef{Name: "my-containerd"},
				},
			},
		},
	}

	remoteScheme := runtime.NewScheme()
	remoteDynamic := fakedynamic.NewSimpleDynamicClient(remoteScheme)
	remoteClientSet := fakeclientset.NewSimpleClientset()

	err := SyncReferencedKubeletAndContainerdConfig(context.Background(), localClient, remoteClientSet, remoteDynamic, cluster)
	require.NoError(t, err)

	gotKCT, err := remoteDynamic.Resource(confv1beta1.KubeletConfigGVR()).Namespace("default").Get(context.Background(), "my-kubelet", metav1.GetOptions{})
	require.NoError(t, err)
	spec, found, err := unstructured.NestedMap(gotKCT.Object, "spec")
	require.NoError(t, err)
	require.True(t, found)
	assert.NotEmpty(t, spec)

	gotCCT, err := remoteDynamic.Resource(confv1beta1.ContainerdConfigGVR()).Namespace("default").Get(context.Background(), "my-containerd", metav1.GetOptions{})
	require.NoError(t, err)
	spec, found, err = unstructured.NestedMap(gotCCT.Object, "spec")
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, "main", spec["configType"])
}

func TestSyncReferencedKubeletAndContainerdConfig_MissingLocalCR(t *testing.T) {
	scheme := runtime.NewScheme()
	localClient := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			KubeletConfigRef: &confv1beta1.KubeletConfigRef{Name: "missing", Namespace: "bke-system"},
		},
	}
	err := SyncReferencedKubeletAndContainerdConfig(
		context.Background(),
		localClient,
		fakeclientset.NewSimpleClientset(&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "bke-system"}}),
		fakedynamic.NewSimpleDynamicClient(runtime.NewScheme()),
		cluster,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "KubeletConfig")
}

func TestPrepareConfigCRForApply(t *testing.T) {
	obj := &unstructured.Unstructured{}
	obj.SetName("test")
	obj.SetNamespace("default")
	obj.SetResourceVersion("1")
	obj.SetUID("uid")
	obj.SetGeneration(1)
	obj.SetManagedFields([]metav1.ManagedFieldsEntry{{Manager: "test"}})
	require.NoError(t, unstructured.SetNestedField(obj.Object, "ready", "status", "phase"))

	prepareConfigCRForApply(obj)

	assert.Empty(t, obj.GetResourceVersion())
	assert.Empty(t, obj.GetUID())
	assert.Zero(t, obj.GetGeneration())
	assert.Empty(t, obj.GetManagedFields())
	_, found, err := unstructured.NestedFieldCopy(obj.Object, "status")
	require.NoError(t, err)
	assert.False(t, found)
}
