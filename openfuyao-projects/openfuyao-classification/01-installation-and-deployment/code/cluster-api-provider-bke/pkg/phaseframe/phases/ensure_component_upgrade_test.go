package phases

import (
	"context"
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureComponentUpgradeConstants(t *testing.T) {
	assert.Equal(t, "EnsureComponentUpgrade", string(EnsureComponentUpgradeName))
}

func TestNewEnsureComponentUpgrade(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := NewEnsureComponentUpgrade(ctx)
	assert.NotNil(t, phase)
	assert.IsType(t, &EnsureComponentUpgrade{}, phase)
}

func TestEnsureComponentUpgrade_IsPatchVersion_Valid(t *testing.T) {
	e := &EnsureComponentUpgrade{}

	tests := []struct {
		name    string
		version string
		want    bool
	}{
		{"patch version with v", "v1.2.3", true},
		{"patch version without v", "1.2.3", true},
		{"minor version", "v1.2.0", false},
		{"major version", "v1.0.0", false},
		{"invalid version", "invalid", false},
		{"empty version", "", false},
		{"prerelease version", "v1.2.3-alpha", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := e.isPatchVersion(tt.version)
			assert.Equal(t, tt.want, result)
		})
	}
}

func TestEnsureComponentUpgrade_IsMatchingImage(t *testing.T) {
	e := &EnsureComponentUpgrade{}

	tests := []struct {
		name            string
		currentImage    string
		targetImageName string
		want            bool
	}{
		{"exact match with tag", "registry.io/myimage:v1.0", "myimage", true},
		{"match without tag", "registry.io/myimage", "myimage", true},
		{"no match", "registry.io/other:v1.0", "myimage", false},
		{"partial match", "registry.io/prefix-myimage:v1.0", "myimage", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := e.isMatchingImage(tt.currentImage, tt.targetImageName)
			assert.Equal(t, tt.want, result)
		})
	}
}

func TestEnsureComponentUpgrade_BuildNewImage(t *testing.T) {
	e := &EnsureComponentUpgrade{}

	tests := []struct {
		name         string
		currentImage string
		newTag       string
		want         string
	}{
		{"with existing tag", "registry.io/myimage:v1.0", "v2.0", "registry.io/myimage:v2.0"},
		{"without tag", "registry.io/myimage", "v1.0", "registry.io/myimage:v1.0"},
		{"with port in registry", "registry.io:5000/myimage:v1.0", "v2.0", "registry.io:5000/myimage:v2.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := e.buildNewImage(tt.currentImage, tt.newTag)
			assert.Equal(t, tt.want, result)
		})
	}
}

func TestEnsureComponentUpgrade_GetNamespace(t *testing.T) {
	e := &EnsureComponentUpgrade{}

	tests := []struct {
		name string
		pod  corev1.Pod
		want string
	}{
		{"with namespace", corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "test-ns"}}, "test-ns"},
		{"without namespace", corev1.Pod{}, metav1.NamespaceDefault},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := e.getNamespace(tt.pod)
			assert.Equal(t, tt.want, result)
		})
	}
}

func TestEnsureComponentUpgrade_ProcessRepoImages_IsKubernetes(t *testing.T) {
	e := &EnsureComponentUpgrade{}
	repo := phaseutil.Repo{IsKubernetes: true}
	err := e.processRepoImages(repo)
	assert.NoError(t, err)
}

func TestEnsureComponentUpgrade_ProcessRepoImages_NotKubernetes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := &EnsureComponentUpgrade{}
	repo := phaseutil.Repo{
		IsKubernetes: false,
		SubImages:    []phaseutil.SubImage{{Images: []phaseutil.Image{}}},
	}

	patches.ApplyPrivateMethod(e, "processSubImage", func(_ *EnsureComponentUpgrade, _ phaseutil.SubImage) error {
		return nil
	})

	err := e.processRepoImages(repo)
	assert.NoError(t, err)
}

func TestEnsureComponentUpgrade_ProcessSubImage_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := &EnsureComponentUpgrade{}
	subImage := phaseutil.SubImage{
		Images: []phaseutil.Image{{Name: "test-image", Tag: []string{"v1.0"}}},
	}

	patches.ApplyPrivateMethod(e, "updateSingleImage", func(_ *EnsureComponentUpgrade, _ phaseutil.Image) error {
		return nil
	})

	err := e.processSubImage(subImage)
	assert.NoError(t, err)
}

func TestEnsureComponentUpgrade_ProcessImageUpdates_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := &EnsureComponentUpgrade{}
	patchCfg := &phaseutil.PatchConfig{
		Repos: []phaseutil.Repo{{IsKubernetes: true}},
	}

	patches.ApplyPrivateMethod(e, "processRepoImages", func(_ *EnsureComponentUpgrade, _ phaseutil.Repo) error {
		return nil
	})

	err := e.processImageUpdates(patchCfg)
	assert.NoError(t, err)
}

func TestEnsureComponentUpgrade_HandleOwnerReference_ReplicaSet(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := &EnsureComponentUpgrade{}
	ownerRef := metav1.OwnerReference{Kind: "ReplicaSet", Name: "test-rs"}

	patches.ApplyPrivateMethod(e, "handleReplicaSet", func(_ *EnsureComponentUpgrade, _ context.Context, _ kubernetes.Interface, _ string, _ metav1.OwnerReference) (metav1.Object, string, error) {
		return &appsv1.ReplicaSet{}, "ReplicaSet", nil
	})

	obj, kind, err := e.handleOwnerReference(context.Background(), nil, "default", ownerRef)
	assert.NoError(t, err)
	assert.Equal(t, "ReplicaSet", kind)
	assert.NotNil(t, obj)
}

func TestEnsureComponentUpgrade_HandleOwnerReference_Unknown(t *testing.T) {
	e := &EnsureComponentUpgrade{}
	ownerRef := metav1.OwnerReference{Kind: "Unknown", Name: "test"}

	obj, kind, err := e.handleOwnerReference(context.Background(), nil, "default", ownerRef)
	assert.NoError(t, err)
	assert.Equal(t, "", kind)
	assert.Nil(t, obj)
}

func TestEnsureComponentUpgrade_UpdateSingleImage_NoTags(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	image := phaseutil.Image{Name: "test-image", Tag: []string{}}
	err := e.updateSingleImage(image)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "has no tags")
}

func TestEnsureComponentUpgrade_UpdateSingleImage_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	image := phaseutil.Image{
		Name: "test-image",
		Tag:  []string{"v1.0"},
		UsedPodInfo: []phaseutil.PodInfo{
			{PodPrefix: "test-pod", NameSpace: "default"},
		},
	}

	patches.ApplyPrivateMethod(e, "updatePodImageTag", func(_ *EnsureComponentUpgrade, _ *phaseutil.ImageUpdate) error {
		return nil
	})

	err := e.updateSingleImage(image)
	assert.NoError(t, err)
}

func TestEnsureComponentUpgrade_FindMatchingPods_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{
		BasePhase:    phaseframe.BasePhase{Ctx: ctx},
		remoteClient: &kubernetes.Clientset{},
	}

	patches.ApplyPrivateMethod(e, "findMatchingPods", func(_ *EnsureComponentUpgrade, _, _ string) ([]corev1.Pod, error) {
		return []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "test-pod-123"}}}, nil
	})

	pods, err := e.findMatchingPods("default", "test-pod")
	assert.NoError(t, err)
	assert.Len(t, pods, 1)
}

func TestEnsureComponentUpgrade_NeedExecute_DefaultFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureComponentUpgradeName}}

	patches.ApplyMethod(&e.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return false
	})

	result := e.NeedExecute(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureComponentUpgrade_IsComponentNeedUpgrade_InitialPatchVersion(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v1.2.3"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: ""},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "isPatchVersion", func(_ *EnsureComponentUpgrade, _ string) bool {
		return true
	})

	result := e.isComponentNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.True(t, result)
}

func TestEnsureComponentUpgrade_IsComponentNeedUpgrade_NotPatchVersion(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v1.2.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: ""},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "isPatchVersion", func(_ *EnsureComponentUpgrade, _ string) bool {
		return false
	})

	result := e.isComponentNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureComponentUpgrade_Execute_GetRemoteClientError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "getRemoteClient", func(_ *EnsureComponentUpgrade) error {
		return errors.New("get remote client error")
	})

	_, err := e.Execute()
	assert.Error(t, err)
}

func TestEnsureComponentUpgrade_HandleReplicaSet_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	ownerRef := metav1.OwnerReference{Kind: "Deployment", Name: "test-deploy"}
	mockRS := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test-rs",
			Namespace:       "default",
			OwnerReferences: []metav1.OwnerReference{ownerRef},
		},
	}

	patches.ApplyPrivateMethod(e, "handleReplicaSet", func(_ *EnsureComponentUpgrade, _ context.Context, _ kubernetes.Interface, _ string, _ metav1.OwnerReference) (metav1.Object, string, error) {
		return mockRS, "Deployment", nil
	})

	obj, kind, err := e.handleReplicaSet(context.Background(), nil, "default", metav1.OwnerReference{Name: "test-rs"})
	assert.NoError(t, err)
	assert.Equal(t, "Deployment", kind)
	assert.NotNil(t, obj)
}

func TestEnsureComponentUpgrade_HandleStatefulSet_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	mockSS := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "test-ss", Namespace: "default"},
	}

	patches.ApplyPrivateMethod(e, "handleStatefulSet", func(_ *EnsureComponentUpgrade, _ context.Context, _ kubernetes.Interface, _ string, _ metav1.OwnerReference) (metav1.Object, string, error) {
		return mockSS, "StatefulSet", nil
	})

	obj, kind, err := e.handleStatefulSet(context.Background(), nil, "default", metav1.OwnerReference{Name: "test-ss"})
	assert.NoError(t, err)
	assert.Equal(t, "StatefulSet", kind)
	assert.NotNil(t, obj)
}

func TestEnsureComponentUpgrade_HandleDaemonSet_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	mockDS := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: "test-ds", Namespace: "default"},
	}

	patches.ApplyPrivateMethod(e, "handleDaemonSet", func(_ *EnsureComponentUpgrade, _ context.Context, _ kubernetes.Interface, _ string, _ metav1.OwnerReference) (metav1.Object, string, error) {
		return mockDS, "DaemonSet", nil
	})

	obj, kind, err := e.handleDaemonSet(context.Background(), nil, "default", metav1.OwnerReference{Name: "test-ds"})
	assert.NoError(t, err)
	assert.Equal(t, "DaemonSet", kind)
	assert.NotNil(t, obj)
}

func TestNewEnsureComponentUpgrade_Creation(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureComponentUpgrade(ctx)
	assert.NotNil(t, phase)

	e, ok := phase.(*EnsureComponentUpgrade)
	assert.True(t, ok)
	assert.Equal(t, EnsureComponentUpgradeName, e.PhaseName)
}

func TestEnsureComponentUpgrade_IsMatchingImage_ColonBeforeSlash(t *testing.T) {
	e := &EnsureComponentUpgrade{}
	result := e.isMatchingImage("registry.io:5000/myimage", "myimage")
	assert.True(t, result)
}

func TestEnsureComponentUpgrade_HandleOwnerReference_StatefulSet(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	ownerRef := metav1.OwnerReference{Kind: "StatefulSet", Name: "test-ss"}

	patches.ApplyPrivateMethod(e, "handleStatefulSet", func(_ *EnsureComponentUpgrade, _ context.Context, _ kubernetes.Interface, _ string, _ metav1.OwnerReference) (metav1.Object, string, error) {
		return &appsv1.StatefulSet{}, "StatefulSet", nil
	})

	obj, kind, err := e.handleOwnerReference(context.Background(), nil, "default", ownerRef)
	assert.NoError(t, err)
	assert.Equal(t, "StatefulSet", kind)
	assert.NotNil(t, obj)
}

func TestEnsureComponentUpgrade_HandleOwnerReference_DaemonSet(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	ownerRef := metav1.OwnerReference{Kind: "DaemonSet", Name: "test-ds"}

	patches.ApplyPrivateMethod(e, "handleDaemonSet", func(_ *EnsureComponentUpgrade, _ context.Context, _ kubernetes.Interface, _ string, _ metav1.OwnerReference) (metav1.Object, string, error) {
		return &appsv1.DaemonSet{}, "DaemonSet", nil
	})

	obj, kind, err := e.handleOwnerReference(context.Background(), nil, "default", ownerRef)
	assert.NoError(t, err)
	assert.Equal(t, "DaemonSet", kind)
	assert.NotNil(t, obj)
}

func TestEnsureComponentUpgrade_GetNamespace_MultipleScenarios(t *testing.T) {
	e := &EnsureComponentUpgrade{}

	tests := []struct {
		name string
		pod  corev1.Pod
		want string
	}{
		{"custom namespace", corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "custom"}}, "custom"},
		{"kube-system namespace", corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "kube-system"}}, "kube-system"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := e.getNamespace(tt.pod)
			assert.Equal(t, tt.want, result)
		})
	}
}

func TestEnsureComponentUpgrade_BuildNewImage_MultipleScenarios(t *testing.T) {
	e := &EnsureComponentUpgrade{}

	tests := []struct {
		name         string
		currentImage string
		newTag       string
		want         string
	}{
		{"simple image", "myimage:v1.0", "v2.0", "myimage:v2.0"},
		{"image with path", "registry.io/path/myimage:v1.0", "v2.0", "registry.io/path/myimage:v2.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := e.buildNewImage(tt.currentImage, tt.newTag)
			assert.Equal(t, tt.want, result)
		})
	}
}

// newComponentUpgradePhase builds an EnsureComponentUpgrade wired to a fake
// controller-runtime client (for ConfigMap reads) and an injectable mockClient.
func newComponentUpgradePhase(t *testing.T, bkeCluster *bkev1beta1.BKECluster, objs ...client.Object) *EnsureComponentUpgrade {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	if bkeCluster != nil {
		objs = append([]client.Object{bkeCluster}, objs...)
	}
	c := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureComponentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func componentUpgradeCluster(version string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: version},
			},
		},
	}
}

// ---- loadLocalKubeConfig ----

func TestComponentUpgradeLoadLocalKubeConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	t.Run("success", func(t *testing.T) {
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig, func(ctx context.Context, c client.Client) ([]byte, error) {
			return []byte("kubeconfig-data"), nil
		})
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		require.NoError(t, e.loadLocalKubeConfig())
		assert.Equal(t, []byte("kubeconfig-data"), e.localKubeConfig)
	})

	t.Run("secret not found", func(t *testing.T) {
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig, func(ctx context.Context, c client.Client) ([]byte, error) {
			return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "secrets"}, "local-kubeconfig")
		})
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		err := e.loadLocalKubeConfig()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "local kubeconfig secret not found")
	})

	t.Run("other error", func(t *testing.T) {
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig, func(ctx context.Context, c client.Client) ([]byte, error) {
			return nil, errors.New("boom")
		})
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		err := e.loadLocalKubeConfig()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get local kubeconfig secret")
	})
}

// ---- getRemoteClient ----

type stubRemoteKubeClient struct {
	kube.RemoteKubeClient
	cs *kubernetes.Clientset
}

func (s stubRemoteKubeClient) KubeClient() (*kubernetes.Clientset, dynamic.Interface) {
	return s.cs, nil
}

func TestComponentUpgradeGetRemoteClient(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	t.Run("success", func(t *testing.T) {
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster, func(ctx context.Context, c client.Client, bc *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
			return stubRemoteKubeClient{cs: &kubernetes.Clientset{}}, nil
		})
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		require.NoError(t, e.getRemoteClient())
		assert.NotNil(t, e.remoteClient)
	})

	t.Run("new client error", func(t *testing.T) {
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster, func(ctx context.Context, c client.Client, bc *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
			return nil, errors.New("dial failed")
		})
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		err := e.getRemoteClient()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "dial failed")
	})

	t.Run("nil kubeclient", func(t *testing.T) {
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster, func(ctx context.Context, c client.Client, bc *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
			return stubRemoteKubeClient{}, nil // cs == nil
		})
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		err := e.getRemoteClient()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get remote client")
	})
}

// ---- getPatchConfig ----

func TestComponentUpgradeGetPatchConfig(t *testing.T) {
	version := "v1.2.3"
	patchYAML := "openfuyaoVersion: v1.2.3\nrepos:\n- isKubernetes: false\n  subImages:\n  - images:\n    - name: myimage\n      tag: [\"v1.2.3\"]\n"

	localCM := func(data map[string]string) *corev1.ConfigMap {
		key := constant.GetLocalConfigMapObjectKey()
		return &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace}, Data: data}
	}
	patchCM := func(data map[string]string) *corev1.ConfigMap {
		return &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: "openfuyao-patch", Name: "cm." + version},
			Data:       data,
		}
	}

	t.Run("success", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster(version),
			localCM(map[string]string{"patch." + version: "ok"}),
			patchCM(map[string]string{version: patchYAML}),
		)
		cfg, err := e.getPatchConfig()
		require.NoError(t, err)
		require.NotNil(t, cfg)
		assert.Equal(t, version, cfg.OpenFuyaoVersion)
	})

	t.Run("local cm not found", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster(version))
		_, err := e.getPatchConfig()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "get cm failed")
	})

	t.Run("local cm missing patch key", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster(version),
			localCM(map[string]string{"other": "x"}),
		)
		_, err := e.getPatchConfig()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not found in local config")
	})

	t.Run("patch cm not found", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster(version),
			localCM(map[string]string{"patch." + version: "ok"}),
		)
		_, err := e.getPatchConfig()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "get cm failed")
	})

	t.Run("patch cm missing version key", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster(version),
			localCM(map[string]string{"patch." + version: "ok"}),
			patchCM(map[string]string{"other": "x"}),
		)
		_, err := e.getPatchConfig()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not found in patch config")
	})

	t.Run("invalid patch yaml", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster(version),
			localCM(map[string]string{"patch." + version: "ok"}),
			patchCM(map[string]string{version: "- a\n- b"}),
		)
		_, err := e.getPatchConfig()
		require.Error(t, err)
	})
}

// ---- rolloutOpenfuyaoComponent ----

func TestComponentUpgradeRolloutOpenfuyaoComponent(t *testing.T) {
	// processImageUpdates is small enough to be inlined, so we drive its real loop
	// body through the processRepoImages seam (which is not inlined).
	repoCfg := func() *phaseutil.PatchConfig {
		return &phaseutil.PatchConfig{OpenFuyaoVersion: "v1.2.3", Repos: []phaseutil.Repo{{IsKubernetes: false}}}
	}

	t.Run("success sets openfuyao version", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getPatchConfig", func(_ *EnsureComponentUpgrade) (*phaseutil.PatchConfig, error) {
			return repoCfg(), nil
		})
		patches.ApplyPrivateMethod(e, "processRepoImages", func(_ *EnsureComponentUpgrade, _ phaseutil.Repo) error {
			return nil
		})
		_, err := e.rolloutOpenfuyaoComponent()
		require.NoError(t, err)
		assert.Equal(t, "v1.2.3", e.Ctx.BKECluster.Status.OpenFuyaoVersion)
	})

	t.Run("get patch config error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getPatchConfig", func(_ *EnsureComponentUpgrade) (*phaseutil.PatchConfig, error) {
			return nil, errors.New("no patch")
		})
		_, err := e.rolloutOpenfuyaoComponent()
		require.Error(t, err)
	})

	t.Run("process image updates error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getPatchConfig", func(_ *EnsureComponentUpgrade) (*phaseutil.PatchConfig, error) {
			return repoCfg(), nil
		})
		patches.ApplyPrivateMethod(e, "processRepoImages", func(_ *EnsureComponentUpgrade, _ phaseutil.Repo) error {
			return errors.New("update failed")
		})
		_, err := e.rolloutOpenfuyaoComponent()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "update failed")
	})
}

// ---- Execute end-to-end ----

func TestComponentUpgradeExecute(t *testing.T) {
	t.Run("happy path", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getRemoteClient", func(_ *EnsureComponentUpgrade) error { return nil })
		patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(_ *EnsureComponentUpgrade) error { return nil })
		patches.ApplyPrivateMethod(e, "rolloutOpenfuyaoComponent", func(_ *EnsureComponentUpgrade) (ctrl.Result, error) {
			return ctrl.Result{}, nil
		})
		_, err := e.Execute()
		require.NoError(t, err)
	})

	t.Run("get remote client error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getRemoteClient", func(_ *EnsureComponentUpgrade) error { return errors.New("no client") })
		_, err := e.Execute()
		require.Error(t, err)
	})

	t.Run("load local kubeconfig error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getRemoteClient", func(_ *EnsureComponentUpgrade) error { return nil })
		patches.ApplyPrivateMethod(e, "loadLocalKubeConfig", func(_ *EnsureComponentUpgrade) error { return errors.New("no kubeconfig") })
		_, err := e.Execute()
		require.Error(t, err)
	})
}

// ---- findMatchingPods (real body) ----

func TestComponentUpgradeFindMatchingPods(t *testing.T) {
	e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
	e.mockClient = k8sfake.NewSimpleClientset(
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "mypod-abc", Namespace: "default"}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "other-xyz", Namespace: "default"}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "mypod-def", Namespace: "kube-system"}},
	)

	pods, err := e.findMatchingPods("default", "mypod")
	require.NoError(t, err)
	assert.Len(t, pods, 1)
	assert.Equal(t, "mypod-abc", pods[0].Name)

	// no match in namespace
	pods, err = e.findMatchingPods("default", "nonexistent")
	require.NoError(t, err)
	assert.Empty(t, pods)
}

// ---- updatePodImageTag (real body) ----

func TestComponentUpgradeUpdatePodImageTag(t *testing.T) {
	t.Run("no pods found returns nil", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset() // empty
		err := e.updatePodImageTag(&phaseutil.ImageUpdate{ImageName: "myimage", PodPrefix: "mypod", NameSpace: "default", NewTag: "v2.0"})
		require.NoError(t, err)
	})

	t.Run("with pod delegates to upgradePodImage", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset(
			&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "mypod-1", Namespace: "default"}},
		)
		called := false
		patches.ApplyPrivateMethod(e, "upgradePodImage", func(_ *EnsureComponentUpgrade, _ corev1.Pod, _ *phaseutil.ImageUpdate) error {
			called = true
			return nil
		})
		err := e.updatePodImageTag(&phaseutil.ImageUpdate{ImageName: "myimage", PodPrefix: "mypod", NameSpace: "default", NewTag: "v2.0"})
		require.NoError(t, err)
		assert.True(t, called)
	})
}

// ---- getPodController (real body) ----

func TestComponentUpgradeGetPodController(t *testing.T) {
	t.Run("no owner refs returns pod itself", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset()
		pod := corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "standalone", Namespace: "default"}}
		obj, kind, err := e.getPodController(pod)
		require.NoError(t, err)
		assert.Equal(t, "Pod", kind)
		assert.NotNil(t, obj)
	})

	t.Run("replicaset owner with deployment", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset(
			&appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "rs1", Namespace: "default", OwnerReferences: []metav1.OwnerReference{{Kind: "Deployment", Name: "dep1"}}},
			},
			&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "dep1", Namespace: "default"}},
		)
		pod := corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default", OwnerReferences: []metav1.OwnerReference{{Kind: "ReplicaSet", Name: "rs1"}}}}
		obj, kind, err := e.getPodController(pod)
		require.NoError(t, err)
		assert.Equal(t, "Deployment", kind)
		assert.Equal(t, "dep1", obj.GetName())
	})

	t.Run("replicaset owner without deployment", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset(
			&appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{Name: "rs1", Namespace: "default"}},
		)
		pod := corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default", OwnerReferences: []metav1.OwnerReference{{Kind: "ReplicaSet", Name: "rs1"}}}}
		obj, kind, err := e.getPodController(pod)
		require.NoError(t, err)
		assert.Equal(t, "ReplicaSet", kind)
		assert.Equal(t, "rs1", obj.GetName())
	})

	t.Run("statefulset owner", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset(
			&appsv1.StatefulSet{ObjectMeta: metav1.ObjectMeta{Name: "sts1", Namespace: "default"}},
		)
		pod := corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default", OwnerReferences: []metav1.OwnerReference{{Kind: "StatefulSet", Name: "sts1"}}}}
		obj, kind, err := e.getPodController(pod)
		require.NoError(t, err)
		assert.Equal(t, "StatefulSet", kind)
		assert.Equal(t, "sts1", obj.GetName())
	})

	t.Run("daemonset owner", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset(
			&appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: "ds1", Namespace: "default"}},
		)
		pod := corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default", OwnerReferences: []metav1.OwnerReference{{Kind: "DaemonSet", Name: "ds1"}}}}
		obj, kind, err := e.getPodController(pod)
		require.NoError(t, err)
		assert.Equal(t, "DaemonSet", kind)
		assert.Equal(t, "ds1", obj.GetName())
	})
}

// ---- handleReplicaSet / handleStatefulSet / handleDaemonSet (real body) ----

func TestComponentUpgradeHandleControllers(t *testing.T) {
	t.Run("handleReplicaSet with deployment owner", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset(
			&appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{Name: "rs1", Namespace: "default", OwnerReferences: []metav1.OwnerReference{{Kind: "Deployment", Name: "dep1"}}}},
			&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "dep1", Namespace: "default"}},
		)
		obj, kind, err := e.handleReplicaSet(context.Background(), e.mockClient, "default", metav1.OwnerReference{Name: "rs1"})
		require.NoError(t, err)
		assert.Equal(t, "Deployment", kind)
		assert.Equal(t, "dep1", obj.GetName())
	})

	t.Run("handleReplicaSet get error", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset() // no rs
		_, _, err := e.handleReplicaSet(context.Background(), e.mockClient, "default", metav1.OwnerReference{Name: "missing"})
		require.Error(t, err)
	})

	t.Run("handleStatefulSet success", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset(&appsv1.StatefulSet{ObjectMeta: metav1.ObjectMeta{Name: "sts1", Namespace: "default"}})
		obj, kind, err := e.handleStatefulSet(context.Background(), e.mockClient, "default", metav1.OwnerReference{Name: "sts1"})
		require.NoError(t, err)
		assert.Equal(t, "StatefulSet", kind)
		assert.Equal(t, "sts1", obj.GetName())
	})

	t.Run("handleStatefulSet get error", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset()
		_, _, err := e.handleStatefulSet(context.Background(), e.mockClient, "default", metav1.OwnerReference{Name: "missing"})
		require.Error(t, err)
	})

	t.Run("handleDaemonSet success", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset(&appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: "ds1", Namespace: "default"}})
		obj, kind, err := e.handleDaemonSet(context.Background(), e.mockClient, "default", metav1.OwnerReference{Name: "ds1"})
		require.NoError(t, err)
		assert.Equal(t, "DaemonSet", kind)
		assert.Equal(t, "ds1", obj.GetName())
	})

	t.Run("handleDaemonSet get error", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset()
		_, _, err := e.handleDaemonSet(context.Background(), e.mockClient, "default", metav1.OwnerReference{Name: "missing"})
		require.Error(t, err)
	})
}

// ---- upgradePodImage (switch on controller type) ----

func TestComponentUpgradeUpgradePodImage(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	t.Run("unsupported controller type", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getPodController", func(_ *EnsureComponentUpgrade, _ corev1.Pod) (metav1.Object, string, error) {
			return &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "x"}}, "CronJob", nil
		})
		err := e.upgradePodImage(corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p"}}, &phaseutil.ImageUpdate{ImageName: "img", NewTag: "v2"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported controller type")
	})

	t.Run("get controller error", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getPodController", func(_ *EnsureComponentUpgrade, _ corev1.Pod) (metav1.Object, string, error) {
			return nil, "", errors.New("ctrl err")
		})
		err := e.upgradePodImage(corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p"}}, &phaseutil.ImageUpdate{ImageName: "img", NewTag: "v2"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get controller")
	})

	t.Run("deployment type but wrong concrete type", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		patches.ApplyPrivateMethod(e, "getPodController", func(_ *EnsureComponentUpgrade, _ corev1.Pod) (metav1.Object, string, error) {
			return &appsv1.StatefulSet{ObjectMeta: metav1.ObjectMeta{Name: "x"}}, "Deployment", nil
		})
		err := e.upgradePodImage(corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p"}}, &phaseutil.ImageUpdate{ImageName: "img", NewTag: "v2"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not a Deployment")
	})

	t.Run("deployment upgrade success", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		dep := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "dep1", Namespace: "default"},
			Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "c1", Image: "reg.io/myimage:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(dep)
		patches.ApplyPrivateMethod(e, "getPodController", func(_ *EnsureComponentUpgrade, _ corev1.Pod) (metav1.Object, string, error) {
			return dep, "Deployment", nil
		})
		err := e.upgradePodImage(corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p"}}, &phaseutil.ImageUpdate{ImageName: "myimage", NewTag: "v2.0"})
		require.NoError(t, err)
		got, err := e.mockClient.AppsV1().Deployments("default").Get(context.Background(), "dep1", metav1.GetOptions{})
		require.NoError(t, err)
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.Containers[0].Image)
	})
}

// ---- upgrade{Deployment,StatefulSet,DaemonSet,ReplicaSet}Image (real body) ----

func TestComponentUpgradeUpgradeWorkloadImages(t *testing.T) {
	update := &phaseutil.ImageUpdate{ImageName: "myimage", NewTag: "v2.0"}

	t.Run("deployment container updated", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		dep := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "dep1", Namespace: "default"},
			Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers:     []corev1.Container{{Name: "c1", Image: "reg.io/myimage:v1.0"}},
				InitContainers: []corev1.Container{{Name: "ic1", Image: "reg.io/myimage:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(dep)
		require.NoError(t, e.upgradeDeploymentImage(&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "dep1", Namespace: "default"}}, update))
		got, err := e.mockClient.AppsV1().Deployments("default").Get(context.Background(), "dep1", metav1.GetOptions{})
		require.NoError(t, err)
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.Containers[0].Image)
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.InitContainers[0].Image)
	})

	t.Run("deployment no matching image", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		dep := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "dep1", Namespace: "default"},
			Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "c1", Image: "reg.io/other:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(dep)
		require.NoError(t, e.upgradeDeploymentImage(&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "dep1", Namespace: "default"}}, update))
		got, _ := e.mockClient.AppsV1().Deployments("default").Get(context.Background(), "dep1", metav1.GetOptions{})
		assert.Equal(t, "reg.io/other:v1.0", got.Spec.Template.Spec.Containers[0].Image)
	})

	t.Run("deployment get error", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		e.mockClient = k8sfake.NewSimpleClientset() // empty -> Get NotFound
		err := e.upgradeDeploymentImage(&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "missing", Namespace: "default"}}, update)
		require.Error(t, err)
	})

	t.Run("statefulset updated", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		sts := &appsv1.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "sts1", Namespace: "default"},
			Spec: appsv1.StatefulSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers:     []corev1.Container{{Name: "c1", Image: "reg.io/myimage:v1.0"}},
				InitContainers: []corev1.Container{{Name: "ic1", Image: "reg.io/myimage:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(sts)
		require.NoError(t, e.upgradeStatefulSetImage(&appsv1.StatefulSet{ObjectMeta: metav1.ObjectMeta{Name: "sts1", Namespace: "default"}}, update))
		got, _ := e.mockClient.AppsV1().StatefulSets("default").Get(context.Background(), "sts1", metav1.GetOptions{})
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.Containers[0].Image)
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.InitContainers[0].Image)
	})

	t.Run("statefulset no match", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		sts := &appsv1.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "sts1", Namespace: "default"},
			Spec: appsv1.StatefulSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "c1", Image: "reg.io/other:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(sts)
		require.NoError(t, e.upgradeStatefulSetImage(&appsv1.StatefulSet{ObjectMeta: metav1.ObjectMeta{Name: "sts1", Namespace: "default"}}, update))
	})

	t.Run("daemonset updated", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: "ds1", Namespace: "default"},
			Spec: appsv1.DaemonSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers:     []corev1.Container{{Name: "c1", Image: "reg.io/myimage:v1.0"}},
				InitContainers: []corev1.Container{{Name: "ic1", Image: "reg.io/myimage:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(ds)
		require.NoError(t, e.upgradeDaemonSetImage(&appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: "ds1", Namespace: "default"}}, update))
		got, _ := e.mockClient.AppsV1().DaemonSets("default").Get(context.Background(), "ds1", metav1.GetOptions{})
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.Containers[0].Image)
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.InitContainers[0].Image)
	})

	t.Run("daemonset no match", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: "ds1", Namespace: "default"},
			Spec: appsv1.DaemonSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "c1", Image: "reg.io/other:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(ds)
		require.NoError(t, e.upgradeDaemonSetImage(&appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: "ds1", Namespace: "default"}}, update))
	})

	t.Run("replicaset updated", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		rs := &appsv1.ReplicaSet{
			ObjectMeta: metav1.ObjectMeta{Name: "rs1", Namespace: "default"},
			Spec: appsv1.ReplicaSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers:     []corev1.Container{{Name: "c1", Image: "reg.io/myimage:v1.0"}},
				InitContainers: []corev1.Container{{Name: "ic1", Image: "reg.io/myimage:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(rs)
		require.NoError(t, e.upgradeReplicaSetImage(&appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{Name: "rs1", Namespace: "default"}}, update))
		got, _ := e.mockClient.AppsV1().ReplicaSets("default").Get(context.Background(), "rs1", metav1.GetOptions{})
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.Containers[0].Image)
		assert.Equal(t, "reg.io/myimage:v2.0", got.Spec.Template.Spec.InitContainers[0].Image)
	})

	t.Run("replicaset no match", func(t *testing.T) {
		e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
		rs := &appsv1.ReplicaSet{
			ObjectMeta: metav1.ObjectMeta{Name: "rs1", Namespace: "default"},
			Spec: appsv1.ReplicaSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "c1", Image: "reg.io/other:v1.0"}},
			}}},
		}
		e.mockClient = k8sfake.NewSimpleClientset(rs)
		require.NoError(t, e.upgradeReplicaSetImage(&appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{Name: "rs1", Namespace: "default"}}, update))
	})
}

// ---- isComponentNeedUpgrade non-initial branch ----

func TestComponentUpgradeIsComponentNeedUpgradeNonInitial(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := newComponentUpgradePhase(t, componentUpgradeCluster("v1.2.3"))
	e.Ctx.BKECluster.Status.OpenFuyaoVersion = "v1.2.0" // non-initial

	t.Run("returns false on node fetch error", func(t *testing.T) {
		// NodeFetcher().GetBKENodesWrapperForCluster returns error (no nodes in fake client)
		assert.False(t, e.isComponentNeedUpgrade(&bkev1beta1.BKECluster{}, e.Ctx.BKECluster))
	})
}
