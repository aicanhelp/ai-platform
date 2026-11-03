package imageref

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestReleaseImageRefs_FromBKEClusterImageRepo(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(testBKECluster("default", "cluster-a")).
		Build()

	releaseRefs, err := ReleaseImageRefs(context.Background(), k8sClient, "default", "cluster-a", "v26.03")
	require.NoError(t, err)
	require.Len(t, releaseRefs, 1)
	assert.Equal(t, "registry.example.com:5000/openfuyao/release-image:v26.03", releaseRefs[0])
}

func TestReleaseImageRefs_DomainFirstThenIP(t *testing.T) {
	bc := testBKECluster("default", "cluster-a")
	bc.Spec.ClusterConfig.Cluster.ImageRepo.Ip = "192.168.1.10"

	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).Build()

	refs, err := ReleaseImageRefs(context.Background(), k8sClient, "default", "cluster-a", "v26.03")
	require.NoError(t, err)
	require.Len(t, refs, 2)
	assert.Equal(t, "registry.example.com:5000/openfuyao/release-image:v26.03", refs[0])
	assert.Equal(t, "192.168.1.10:5000/openfuyao/release-image:v26.03", refs[1])
}

func TestReleaseImageRefs_PreferredClusterByName(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(testBKECluster("default", "cluster-a")).
		Build()

	refs, err := ReleaseImageRefs(context.Background(), k8sClient, "", "cluster-a", "v26.03")
	require.NoError(t, err)
	require.Len(t, refs, 1)
	assert.Equal(t, "registry.example.com:5000/openfuyao/release-image:v26.03", refs[0])
}

func TestReleaseImageRequiresVersion(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	_, err := ReleaseImageRefs(context.Background(), k8sClient, "default", "cluster-a", "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "release version is empty")
}

func TestReleaseImageRefs_UsesFirstClusterWhenMultipleWithoutPreferredName(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(testBKECluster("default", "cluster-a"), testBKECluster("default", "cluster-b")).
		Build()

	refs, err := ReleaseImageRefs(context.Background(), k8sClient, "default", "", "v26.03")
	require.NoError(t, err)
	require.Len(t, refs, 1)
	assert.Equal(t, "registry.example.com:5000/openfuyao/release-image:v26.03", refs[0])
}

func TestReleaseImageRefs_RejectsMissingClusterConfig(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(&bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "cluster-a"}}).
		Build()

	_, err := ReleaseImageRefs(context.Background(), k8sClient, "default", "cluster-a", "v26.03")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "spec.clusterConfig is empty")
}

func TestReleaseImageRefs_FindsClusterByNameAcrossNamespaces(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(testBKECluster("ns-a", "cluster-a")).
		Build()

	refs, err := ReleaseImageRefs(context.Background(), k8sClient, "", "cluster-a", "v26.03")
	require.NoError(t, err)
	require.Len(t, refs, 1)
	assert.Equal(t, "registry.example.com:5000/openfuyao/release-image:v26.03", refs[0])
}

func testBKECluster(namespace, name string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					ImageRepo: confv1beta1.Repo{
						Domain: "registry.example.com",
						Port:   "5000",
						Prefix: "openfuyao",
					},
				},
			},
		},
	}
}
