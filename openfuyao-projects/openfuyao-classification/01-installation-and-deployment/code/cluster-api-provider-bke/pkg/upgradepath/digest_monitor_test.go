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

package upgradepath

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	upv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/oci"
)

func newFakeK8sClient(objs ...client.Object) client.Client {
	scheme := runtime.NewScheme()
	_ = upv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if len(objs) > 0 {
		builder = builder.WithObjects(objs...).WithStatusSubresource(objs...)
	}
	return builder.Build()
}

func newTestBKEConfigCM(data map[string]string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      common.BKEClusterConfigFileName,
			Namespace: "cluster-system",
		},
		Data: data,
	}
}

func TestDigestMonitorCreation(t *testing.T) {
	k8sClient := newFakeK8sClient()
	monitor := NewDigestMonitor("registry/test:latest", nil, k8sClient, time.Second)
	assert.NotNil(t, monitor)
	assert.Equal(t, time.Second, monitor.checkInterval)
	assert.Equal(t, "registry/test:latest", monitor.ociRef)
}

func TestDigestMonitorDefaultInterval(t *testing.T) {
	k8sClient := newFakeK8sClient()
	monitor := NewDigestMonitor("registry/test:latest", nil, k8sClient, 0)
	assert.Equal(t, DefaultCheckInterval, monitor.checkInterval)
}

func TestDigestMonitorLastDigestAndCheckedAt(t *testing.T) {
	k8sClient := newFakeK8sClient()
	monitor := NewDigestMonitor("registry/test:latest", nil, k8sClient, time.Second)

	monitor.mu.Lock()
	monitor.lastKnownDigest = "sha256:abc123"
	monitor.lastCheckedAt = time.Now()
	monitor.mu.Unlock()

	assert.Equal(t, "sha256:abc123", monitor.LastDigest())
	require.NotNil(t, monitor.LastCheckedAt())
}

func TestUpgradePathExists(t *testing.T) {
	k8sClient := newFakeK8sClient()
	monitor := NewDigestMonitor("registry/test:latest", nil, k8sClient, time.Second)

	exists, err := monitor.upgradePathExists(context.Background())
	require.NoError(t, err)
	assert.False(t, exists)

	up := &upv1alpha1.UpgradePath{ObjectMeta: metav1.ObjectMeta{Name: "openfuyao-upgrade-paths"}}
	k8sClient = newFakeK8sClient(up)
	monitor = NewDigestMonitor("registry/test:latest", nil, k8sClient, time.Second)

	exists, err = monitor.upgradePathExists(context.Background())
	require.NoError(t, err)
	assert.True(t, exists)
}

func TestDigestMonitorPatchSpecUpdatesSpecOnly(t *testing.T) {
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "test-cr"},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{{From: "v1.0.0", To: "v1.1.0"}},
		},
	}

	k8sClient := newFakeK8sClient(up)
	monitor := NewDigestMonitor("cr.openfuyao.cn/openfuyao/upgrade-path:latest", nil, k8sClient, time.Second)

	parsed := &upv1alpha1.UpgradePath{
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
				{From: "v1.1.0", To: "v1.2.0"},
			},
			Versions: []upv1alpha1.VersionEntry{
				{Version: "v1.0.0", Installable: true},
			},
		},
	}

	err := monitor.patchSpec(context.Background(), up, parsed, "sha256:newdigest")
	require.NoError(t, err)

	updated := &upv1alpha1.UpgradePath{}
	require.NoError(t, k8sClient.Get(context.Background(), client.ObjectKey{Name: "test-cr"}, updated))
	assert.Len(t, updated.Spec.Paths, 2)
	assert.Len(t, updated.Spec.Versions, 1)
	assert.Equal(t, "sha256:newdigest", updated.Annotations[OCIDigestAnnotation])
	assert.Empty(t, updated.Status.LastDigest)
}

func TestDigestMonitorPatchSpecFullOverwriteWhenEmpty(t *testing.T) {
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "test-cr"},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
			},
			Versions: []upv1alpha1.VersionEntry{
				{Version: "v1.0.0", Installable: true},
			},
		},
	}

	k8sClient := newFakeK8sClient(up)
	monitor := NewDigestMonitor("cr.openfuyao.cn/openfuyao/upgrade-path:latest", nil, k8sClient, time.Second)

	parsed := &upv1alpha1.UpgradePath{
		Spec: upv1alpha1.UpgradePathSpec{},
	}

	err := monitor.patchSpec(context.Background(), up, parsed, "sha256:newdigest")
	require.NoError(t, err)

	updated := &upv1alpha1.UpgradePath{}
	require.NoError(t, k8sClient.Get(context.Background(), client.ObjectKey{Name: "test-cr"}, updated))
	assert.Empty(t, updated.Spec.Paths)
	assert.Empty(t, updated.Spec.Versions)
	assert.Equal(t, "sha256:newdigest", updated.Annotations[OCIDigestAnnotation])
}

func TestDigestMonitorCreateCR(t *testing.T) {
	k8sClient := newFakeK8sClient()
	monitor := NewDigestMonitor("cr.openfuyao.cn/openfuyao/upgrade-path:latest", nil, k8sClient, time.Second)

	parsed := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "openfuyao-upgrade-paths"},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{{From: "v1.0.0", To: "v1.1.0"}},
		},
	}

	err := monitor.createCR(context.Background(), parsed, "sha256:newdigest")
	require.NoError(t, err)

	created := &upv1alpha1.UpgradePath{}
	require.NoError(t, k8sClient.Get(context.Background(), client.ObjectKey{Name: "openfuyao-upgrade-paths"}, created))
	assert.Equal(t, "sha256:newdigest", created.Annotations[OCIDigestAnnotation])
	assert.Len(t, created.Spec.Paths, 1)
}

func TestLayerUnmarshalUpgradePathCR(t *testing.T) {
	crYAML := []byte(`apiVersion: config.openfuyao.com/v1alpha1
kind: UpgradePath
metadata:
  name: openfuyao-upgrade-paths
spec:
  paths:
    - from: v2.5.0
      to: v2.6.0
      blocked: false
  versions:
    - version: v2.5.0
      installable: true
`)

	layer := &oci.Layer{Path: "paths.yaml", Content: crYAML}
	var upgradePathCR upv1alpha1.UpgradePath
	err := layer.UnmarshalYAML(&upgradePathCR)
	require.NoError(t, err)
	assert.Len(t, upgradePathCR.Spec.Paths, 1)
	assert.Equal(t, "v2.5.0", upgradePathCR.Spec.Paths[0].From)
}

func TestDefaultUpgradePathOCIRef(t *testing.T) {
	assert.Equal(t, "cr.openfuyao.cn/openfuyao/upgrade-path:latest", DefaultUpgradePathOCIRef())
}

func TestResolveImageOCIRefsFromCluster_DomainFirstThenIP(t *testing.T) {
	bc := newTestBKEClusterWithImageRepo("registry.local", "192.168.1.10", "5443", "self")

	refs, err := ResolveImageOCIRefsFromCluster(bc, upgradePathImageName, upgradePathImageTag)
	require.NoError(t, err)
	require.Len(t, refs, 2)
	assert.Equal(t, "registry.local:5443/self/upgrade-path:latest", refs[0])
	assert.Equal(t, "192.168.1.10:5443/self/upgrade-path:latest", refs[1])
}

func TestResolveImageOCIRefsFromCluster_Port443WithoutExplicitPort(t *testing.T) {
	bc := newTestBKEClusterWithImageRepo("cr.openfuyao.cn", "119.3.216.97", "443", "openfuyao")

	refs, err := ResolveImageOCIRefsFromCluster(bc, upgradePathImageName, upgradePathImageTag)
	require.NoError(t, err)
	require.Len(t, refs, 2)
	assert.Equal(t, "cr.openfuyao.cn/openfuyao/upgrade-path:latest", refs[0])
	assert.Equal(t, "119.3.216.97/openfuyao/upgrade-path:latest", refs[1])
}

func TestResolveImageOCIRefsFromCluster_EmptyPrefixReturnsError(t *testing.T) {
	bc := newTestBKEClusterWithImageRepo("registry.local", "192.168.1.10", "5443", "")

	refs, err := ResolveImageOCIRefsFromCluster(bc, upgradePathImageName, upgradePathImageTag)
	require.Error(t, err)
	assert.Nil(t, refs)
	assert.Contains(t, err.Error(), "imageRepo prefix is empty")
}

func TestResolveImageOCIRefsFromCluster_EmptyDomainAndIPReturnsError(t *testing.T) {
	bc := newTestBKEClusterWithImageRepo("", "", "5443", "self")

	refs, err := ResolveImageOCIRefsFromCluster(bc, upgradePathImageName, upgradePathImageTag)
	require.Error(t, err)
	assert.Nil(t, refs)
	assert.Contains(t, err.Error(), "imageRepo domain and ip are empty")
}

func TestResolveUpgradePathOCIRefs_FromBKEConfig_Offline(t *testing.T) {
	cm := newTestBKEConfigCM(map[string]string{
		"domain":        "deploy.bocloud.k8s",
		"host":          "192.168.1.10",
		"imageRepoPort": "40443",
	})
	k8sClient := newFakeK8sClient(cm)

	refs, err := resolveUpgradePathOCIRefs(context.Background(), k8sClient)
	require.NoError(t, err)
	require.Len(t, refs, 2)
	assert.Equal(t, "deploy.bocloud.k8s:40443/kubernetes/upgrade-path:latest", refs[0])
	assert.Equal(t, "192.168.1.10:40443/kubernetes/upgrade-path:latest", refs[1])
}

func TestResolveUpgradePathOCIRefs_FromBKEConfig_OtherRepo(t *testing.T) {
	cm := newTestBKEConfigCM(map[string]string{
		"otherRepo":   "registry.example.com:5000/openfuyao/",
		"otherRepoIp": "10.0.0.2",
	})
	k8sClient := newFakeK8sClient(cm)

	refs, err := resolveUpgradePathOCIRefs(context.Background(), k8sClient)
	require.NoError(t, err)
	require.Len(t, refs, 2)
	assert.Equal(t, "registry.example.com:5000/openfuyao/upgrade-path:latest", refs[0])
	assert.Equal(t, "10.0.0.2:5000/openfuyao/upgrade-path:latest", refs[1])
}

func TestResolveUpgradePathOCIRefs_FromBKEConfig_OnlineImageFallback(t *testing.T) {
	cm := newTestBKEConfigCM(map[string]string{
		"domain":        "deploy.bocloud.k8s",
		"host":          "10.0.0.1",
		"imageRepoPort": "40443",
		"onlineImage":   "cr.openfuyao.cn/openfuyao/bke-online-installed:latest",
	})
	k8sClient := newFakeK8sClient(cm)

	refs, err := resolveUpgradePathOCIRefs(context.Background(), k8sClient)
	require.NoError(t, err)
	require.Len(t, refs, 1)
	assert.Equal(t, DefaultUpgradePathOCIRef(), refs[0])
}

func TestResolveUpgradePathOCIRefs_BKEConfigMissing(t *testing.T) {
	k8sClient := newFakeK8sClient()
	_, err := resolveUpgradePathOCIRefs(context.Background(), k8sClient)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "bke-config")
}

func newTestBKEClusterWithImageRepo(domain, ip, port, prefix string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-bc",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					ImageRepo: confv1beta1.Repo{
						Domain: domain,
						Ip:     ip,
						Port:   port,
						Prefix: prefix,
					},
				},
			},
		},
	}
}
