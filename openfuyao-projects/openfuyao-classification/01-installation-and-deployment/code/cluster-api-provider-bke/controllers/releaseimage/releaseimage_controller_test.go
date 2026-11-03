package releaseimage

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	riv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/compatibility"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
)

type controllerPuller struct {
	files  *manifest.BundleFiles
	digest string
	ref    manifest.ReleaseRef
	calls  int
}

func (p *controllerPuller) Pull(_ context.Context, ref manifest.ReleaseRef) (*manifest.BundleFiles, string, error) {
	p.calls++
	p.ref = ref
	return p.files, p.digest, nil
}

func TestReleaseImageReconcilerMarksValid(t *testing.T) {
	reconciler, key, puller, store := newTestReconciler(t, releaseBundleFiles("v3.5.21-of.1"))

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)

	got := &riv1beta1.ReleaseImage{}
	require.NoError(t, reconciler.Get(context.Background(), key, got))
	assert.Equal(t, riv1beta1.ReleaseImagePhaseValid, got.Status.Phase)
	assert.Equal(t, 2, got.Status.ComponentCount)
	assert.Equal(t, []riv1beta1.ComponentStatus{
		{Name: "etcd", Version: "v3.5.21-of.1", Type: riv1beta1.ComponentTypeBinary},
		{Name: "kubernetes", Version: "v1.29.1-of.1", Type: riv1beta1.ComponentTypeBinary},
	}, got.Status.Components)
	assert.Equal(t, manifest.SourceOCI, got.Status.Source)
	assert.Equal(t, "registry.example.com:5000/openfuyao/release-image:v26.03", puller.ref.OCIRef)
	assert.Equal(t, 1, puller.calls)
	assert.Contains(t, got.Finalizers, releaseImageFinalizer)

	ref := manifest.ReleaseRef{Version: "v26.03"}
	cached, err := store.ResolveRelease(context.Background(), ref)
	require.NoError(t, err)
	assert.Equal(t, puller.digest, cached.Digest)

	digestRef := manifest.ReleaseRef{Digest: got.Status.Digest}
	_, err = store.ResolveRelease(context.Background(), digestRef)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not cached")
}

func TestReleaseImageReconcilerAlwaysRefreshesOnUpdate(t *testing.T) {
	reconciler, key, puller, _ := newTestReconciler(t, releaseBundleFiles("v3.5.21-of.1"))

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)
	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)

	assert.Equal(t, 2, puller.calls)
}

func TestReleaseImageReconcilerMarksCompatibilityFailed(t *testing.T) {
	reconciler, key, puller, store := newTestReconciler(t, releaseBundleFiles("v3.4.0"))

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)

	got := &riv1beta1.ReleaseImage{}
	require.NoError(t, reconciler.Get(context.Background(), key, got))
	assert.Equal(t, riv1beta1.ReleaseImagePhaseCompatibilityFailed, got.Status.Phase)
	assert.Contains(t, got.Status.CompatibilityReport, "etcd")
	assert.Equal(t, 1, puller.calls)

	ref := manifest.ReleaseRef{Version: "v26.03"}
	_, err = store.ResolveRelease(context.Background(), ref)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not cached")
}

func TestReleaseImageReconcilerCompatibilityFailedDoesNotOverwriteValidCache(t *testing.T) {
	goodFiles := releaseBundleFiles("v3.5.21-of.1")
	reconciler, key, puller, store := newTestReconciler(t, goodFiles)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)

	ref := manifest.ReleaseRef{Version: "v26.03"}
	cachedBefore, err := store.ResolveRelease(context.Background(), ref)
	require.NoError(t, err)

	puller.files = releaseBundleFiles("v3.4.0")
	puller.digest = manifest.DigestFiles(puller.files)

	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)

	got := &riv1beta1.ReleaseImage{}
	require.NoError(t, reconciler.Get(context.Background(), key, got))
	assert.Equal(t, riv1beta1.ReleaseImagePhaseCompatibilityFailed, got.Status.Phase)

	cachedAfter, err := store.ResolveRelease(context.Background(), ref)
	require.NoError(t, err)
	assert.Equal(t, cachedBefore.Digest, cachedAfter.Digest)
}

func TestReleaseImageReconcilerMarksInvalidWhenImageRepoMissing(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, riv1beta1.AddToScheme(scheme))
	key := types.NamespacedName{Namespace: "default", Name: "release-v26-03"}
	ri := &riv1beta1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Namespace: key.Namespace, Name: key.Name},
		Spec:       riv1beta1.ReleaseImageSpec{Version: "v26.03"},
	}
	client := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ri).
		WithStatusSubresource(ri).
		Build()
	reconciler := &ReleaseImageReconciler{Client: client, Scheme: scheme}

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)

	got := &riv1beta1.ReleaseImage{}
	require.NoError(t, reconciler.Get(context.Background(), key, got))
	assert.Equal(t, riv1beta1.ReleaseImagePhaseInvalid, got.Status.Phase)
	assert.Contains(t, got.Status.Message, "BKECluster not found")
}

func TestReleaseImageReconcilerDeleteEvictsBundleCache(t *testing.T) {
	reconciler, key, _, store := newTestReconciler(t, releaseBundleFiles("v3.5.21-of.1"))

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)

	ref := manifest.ReleaseRef{Version: "v26.03"}
	_, err = store.ResolveRelease(context.Background(), ref)
	require.NoError(t, err)

	ri := &riv1beta1.ReleaseImage{}
	require.NoError(t, reconciler.Get(context.Background(), key, ri))
	require.NoError(t, reconciler.Delete(context.Background(), ri))

	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)

	got := &riv1beta1.ReleaseImage{}
	if err := reconciler.Get(context.Background(), key, got); err == nil {
		assert.NotContains(t, got.Finalizers, releaseImageFinalizer)
	}

	_, err = store.ResolveRelease(context.Background(), ref)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not cached")
}

func TestComponentStatusesHandlesNilBundle(t *testing.T) {
	assert.Nil(t, componentStatuses(nil))
	assert.Nil(t, componentStatuses(&manifest.Bundle{}))
}

func newTestReconciler(
	t *testing.T,
	files *manifest.BundleFiles,
) (*ReleaseImageReconciler, types.NamespacedName, *controllerPuller, *manifest.Store) {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, riv1beta1.AddToScheme(scheme))
	key := types.NamespacedName{Namespace: "default", Name: "release-v26-03"}
	ri := &riv1beta1.ReleaseImage{}
	ri.Name = key.Name
	ri.Namespace = key.Namespace
	ri.Spec.Version = "v26.03"

	digest := manifest.DigestFiles(files)
	puller := &controllerPuller{files: files, digest: digest}
	store := manifest.NewStore(t.TempDir(), puller, nil)
	client := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ri, testBKECluster(key.Namespace, "cluster-a")).
		WithStatusSubresource(ri).
		Build()
	reconciler := &ReleaseImageReconciler{
		Client:        client,
		Scheme:        scheme,
		Store:         store,
		Compatibility: compatibility.NewEngine(),
	}
	return reconciler, key, puller, store
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

func releaseBundleFiles(etcdVersion string) *manifest.BundleFiles {
	return &manifest.BundleFiles{Files: map[string][]byte{
		"release.yaml": []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: ReleaseImage
metadata:
  name: release-v26-03
spec:
  version: v26.03
  ociRef: registry/release:v26.03
  upgrade:
    components:
    - name: kubernetes
      version: v1.29.1-of.1
    - name: etcd
      version: ` + etcdVersion + `
`),
		"components/kubernetes/component.yaml": []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: ComponentVersion
metadata:
  name: kubernetes
spec:
  name: kubernetes
  version: v1.29.1-of.1
  type: binary
  compatibility:
    constraints:
    - component: etcd
      rule: ">=3.5.10 <3.6.0"
`),
		"components/etcd/component.yaml": []byte(`apiVersion: bke.bocloud.com/v1beta1
kind: ComponentVersion
metadata:
  name: etcd
spec:
  name: etcd
  version: ` + etcdVersion + `
  type: binary
`),
	}}
}
