package phases

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apixv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	api "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestEnsurePreUpgradeResourcesNeedExecuteUsesVersionContext(t *testing.T) {
	phase := newTestPreUpgradeResourcesPhase(t, nil, api.ComponentVersion{})
	phase.Ctx.SetVersionContext(&upgrade.VersionContext{
		Current: map[string]string{upgrade.ComponentPreUpgradeResources: "v1.0.0"},
		Target:  map[string]string{upgrade.ComponentPreUpgradeResources: "v1.0.1"},
	})

	if !phase.NeedExecute(nil, testBKECluster()) {
		t.Fatal("expected phase to execute when pre-upgrade resource version changes")
	}

	phase.Ctx.SetVersionContext(&upgrade.VersionContext{
		Current: map[string]string{upgrade.ComponentPreUpgradeResources: "v1.0.1"},
		Target:  map[string]string{upgrade.ComponentPreUpgradeResources: "v1.0.1"},
	})

	if phase.NeedExecute(nil, testBKECluster()) {
		t.Fatal("expected phase to skip when pre-upgrade resource version is unchanged")
	}
}

func TestEnsurePreUpgradeResourcesVersionAndGetter(t *testing.T) {
	component := api.ComponentVersion{Spec: api.ComponentVersionSpec{Version: "v1.2.3"}}
	phase := newTestPreUpgradeResourcesPhase(t, nil, component)

	require.NotNil(t, phase.GetComponentVersion())
	assert.Equal(t, "v1.2.3", phase.Version())

	phase.componentVersion = nil
	assert.Nil(t, phase.GetComponentVersion())
	assert.Empty(t, phase.Version())
}

func TestEnsurePreUpgradeResourcesNeedExecuteWithoutVersionDecision(t *testing.T) {
	phase := newTestPreUpgradeResourcesPhase(t, nil, api.ComponentVersion{})
	assert.False(t, phase.NeedExecute(nil, testBKECluster()))
}

func TestEnsurePreUpgradeResourcesExecuteCreatesConfigMapAndSecret(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
	component := api.ComponentVersion{Spec: api.ComponentVersionSpec{Resources: []api.ResourceSpec{
		{
			Kind:       "ConfigMap",
			APIVersion: "v1",
			Namespace:  "kube-system",
			Name:       "bke-new-feature-config",
			Labels:     map[string]string{"managed-by": "declarative-upgrade"},
			Data:       map[string]string{"feature-flag": "enabled"},
		},
		{
			Kind:       "Secret",
			APIVersion: "v1",
			Namespace:  "kube-system",
			Name:       "bke-new-cert",
			StringData: map[string]string{"tls.crt": "crt", "tls.key": "key"},
		},
	}}}
	phase := newTestPreUpgradeResourcesPhase(t, c, component)

	_, err := phase.Execute()
	require.NoError(t, err)

	cm := &corev1.ConfigMap{}
	require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: "kube-system", Name: "bke-new-feature-config"}, cm))
	assert.Equal(t, "enabled", cm.Data["feature-flag"])
	assert.Equal(t, "declarative-upgrade", cm.Labels["managed-by"])

	secret := &corev1.Secret{}
	require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: "kube-system", Name: "bke-new-cert"}, secret))
	assert.Equal(t, []byte("crt"), secret.Data["tls.crt"])
	assert.Equal(t, []byte("key"), secret.Data["tls.key"])

	_, err = phase.Execute()
	require.NoError(t, err, "pre-upgrade resource creation must be idempotent")
}

func TestEnsurePreUpgradeResourcesExecuteRequiresComponentVersion(t *testing.T) {
	phase := newTestPreUpgradeResourcesPhase(t, nil, api.ComponentVersion{})
	phase.componentVersion = nil

	_, err := phase.Execute()
	require.Error(t, err)
	assert.Contains(t, err.Error(), string(EnsurePreUpgradeResourcesName))
}

func TestEnsurePreUpgradeResourcesExecuteWithNoResources(t *testing.T) {
	phase := newTestPreUpgradeResourcesPhase(t, nil, api.ComponentVersion{Spec: api.ComponentVersionSpec{Version: "v1.0.0"}})

	_, err := phase.Execute()
	require.NoError(t, err)
}

func TestEnsurePreUpgradeResourcesSortsResourcesByKind(t *testing.T) {
	phase := newTestPreUpgradeResourcesPhase(t, nil, api.ComponentVersion{})
	sorted := phase.sortResourcesByKind([]api.ResourceSpec{
		{Kind: "Secret", Name: "secret"},
		{Kind: "ConfigMap", Name: "cm"},
		{Kind: "CustomResourceDefinition", Name: "crd"},
	})

	assert.Equal(t, []string{"CustomResourceDefinition", "ConfigMap", "Secret"}, []string{
		sorted[0].Kind,
		sorted[1].Kind,
		sorted[2].Kind,
	})
}

func TestEnsurePreUpgradeResourcesCreateCRDFromManifest(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, apixv1.AddToScheme(scheme))
	c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
	component := api.ComponentVersion{Spec: api.ComponentVersionSpec{Resources: []api.ResourceSpec{
		{
			Kind:       "CustomResourceDefinition",
			APIVersion: "apiextensions.k8s.io/v1",
			Name:       "widgets.example.com",
			Manifest: `apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: widgets.example.com
spec:
  group: example.com
  names:
    kind: Widget
    plural: widgets
  scope: Namespaced
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
`,
		},
	}}}
	phase := newTestPreUpgradeResourcesPhase(t, c, component)

	_, err := phase.Execute()
	require.NoError(t, err)

	crd := &apixv1.CustomResourceDefinition{}
	err = c.Get(context.Background(), client.ObjectKey{Name: "widgets.example.com"}, crd)
	if apierrors.IsNotFound(err) {
		t.Fatal("expected CRD manifest to be created")
	}
	require.NoError(t, err)
}

func TestEnsurePreUpgradeResourcesProvisionResourceRejectsUnsupportedKindWithoutManifest(t *testing.T) {
	phase := newTestPreUpgradeResourcesPhase(t, nil, api.ComponentVersion{})

	err := phase.provisionResource(api.ResourceSpec{Kind: "Deployment", Name: "unsupported"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported pre-upgrade resource kind")
}

func TestEnsurePreUpgradeResourcesProvisionersRequireClient(t *testing.T) {
	phase := newTestPreUpgradeResourcesPhase(t, nil, api.ComponentVersion{})
	phase.Ctx.Client = nil

	assert.EqualError(t, phase.provisionConfigMap(api.ResourceSpec{Kind: "ConfigMap"}), "phase client is not configured")
	assert.EqualError(t, phase.provisionSecret(api.ResourceSpec{Kind: "Secret"}), "phase client is not configured")
	assert.EqualError(t, phase.provisionFromManifest("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: test\n"), "phase client is not configured")
}

func TestEnsurePreUpgradeResourcesProvisionFromManifestInvalidYAML(t *testing.T) {
	scheme := runtime.NewScheme()
	c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
	phase := newTestPreUpgradeResourcesPhase(t, c, api.ComponentVersion{})

	err := phase.provisionFromManifest("not: [valid")
	require.Error(t, err)
}

func newTestPreUpgradeResourcesPhase(t *testing.T, c client.Client, cv api.ComponentVersion) *EnsurePreUpgradeResources {
	t.Helper()
	if c == nil {
		c = fakeclient.NewClientBuilder().WithScheme(runtime.NewScheme()).Build()
	}
	bc := testBKECluster()
	pc := phaseframe.NewReconcilePhaseCtx(context.Background()).
		SetClient(c).
		SetBKECluster(bc).
		SetLogger(bkev1beta1.NewBKELogger(nil, nil, bc))
	phase, ok := NewEnsurePreUpgradeResources(pc).(*EnsurePreUpgradeResources)
	require.True(t, ok)
	phase.SetComponentVersion(cv)
	return phase
}

func testBKECluster() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "cluster", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
	}
}
