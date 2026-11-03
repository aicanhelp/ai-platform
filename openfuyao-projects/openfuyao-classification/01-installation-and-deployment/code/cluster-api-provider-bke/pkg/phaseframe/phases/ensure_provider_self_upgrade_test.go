package phases

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/testutils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testTargetVersion = "v1.2.1"
	testOldImage      = "docker.io/dadaozbzy/provider-bke:old-version"
	testTargetImage   = "docker.io/dadaozbzy/provider-bke:1117-update"
)

type selfUpgradeTestCase struct {
	name         string
	newCluster   *bkev1beta1.BKECluster
	existingObjs []client.Object
	want         bool
}

func TestEnsureProviderSelfUpgradeIsProviderNeedUpgrade(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	_ = appsv1.AddToScheme(scheme)

	tests := getSelfUpgradeTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(tt.existingObjs...).Build()
			ctx := createPhaseContext(fakeClient, scheme, tt.newCluster)

			phase := NewEnsureProviderSelfUpgrade(ctx)
			p, ok := phase.(*EnsureProviderSelfUpgrade)
			assert.True(t, ok, "NewEnsureProviderSelfUpgrade should return *EnsureProviderSelfUpgrade")

			got := p.isProviderNeedUpgrade(nil, tt.newCluster)
			assert.Equal(t, tt.want, got)
		})
	}
}

func getSelfUpgradeTestCases() []selfUpgradeTestCase {
	patchKey := fmt.Sprintf("patch.%s", testTargetVersion)
	patchCMName := fmt.Sprintf("cm.%s", testTargetVersion)

	return []selfUpgradeTestCase{
		{
			name:       "Fresh install, non-patch version",
			newCluster: createBKECluster("v1.2.0", ""),
			want:       false,
		},
		{
			name:       "Fresh install, patch version, no deployment",
			newCluster: createBKECluster("v1.2.1", ""),
			want:       false,
		},
		{
			name:       "Version not changed",
			newCluster: createBKECluster("v1.2.0", "v1.2.0"),
			want:       false,
		},
		{
			name:         "Deployment missing",
			newCluster:   createBKECluster(testTargetVersion, "v1.2.0"),
			existingObjs: []client.Object{},
			want:         false,
		},
		getTestCaseLocalCMMissing(),
		getTestCaseImagesMatch(patchKey, patchCMName),
		getTestCaseImagesDiffer(patchKey, patchCMName),
	}
}

// 获取本地 CM 缺失的测试用例
func getTestCaseLocalCMMissing() selfUpgradeTestCase {
	return selfUpgradeTestCase{
		name:       "Local CM missing",
		newCluster: createBKECluster(testTargetVersion, "v1.2.0"),
		existingObjs: []client.Object{
			createDeployment(testOldImage),
		},
		want: false,
	}
}

// 获取镜像匹配的测试用例
func getTestCaseImagesMatch(patchKey, patchCMName string) selfUpgradeTestCase {
	return selfUpgradeTestCase{
		name:       "Images match",
		newCluster: createBKECluster(testTargetVersion, "v1.2.0"),
		existingObjs: []client.Object{
			createDeployment(testTargetImage),
			createLocalCM(patchKey),
			createPatchCM(patchCMName, testTargetVersion, getValidPatchYaml()),
		},
		want: false,
	}
}

// 获取镜像不匹配的测试用例
func getTestCaseImagesDiffer(patchKey, patchCMName string) selfUpgradeTestCase {
	return selfUpgradeTestCase{
		name:       "Images differ, need upgrade",
		newCluster: createBKECluster(testTargetVersion, "v1.2.0"),
		existingObjs: []client.Object{
			createDeployment(testOldImage),
			createLocalCM(patchKey),
			createPatchCM(patchCMName, testTargetVersion, getValidPatchYaml()),
		},
		want: true,
	}
}

// 获取有效的 Patch YAML
func getValidPatchYaml() string {
	return `
repos:
  - subImages:
      - sourceRepo: "docker.io/dadaozbzy"
        targetRepo: "kubernetes/provider-bke"
        images:
          - name: "provider-bke"
            usedPodInfo:
              - podPrefix: "bke-controller-manager"
                namespace: "cluster-system"
            tag: ["1117-update"]
`
}

// createPhaseContext 创建 PhaseContext
func createPhaseContext(c client.Client, scheme *runtime.Scheme, cluster *bkev1beta1.BKECluster) *phaseframe.PhaseContext {
	return &phaseframe.PhaseContext{
		BKECluster: cluster,
		Scheme:     scheme,
		Context:    context.Background(),
		Client:     c,
		Log: &bkev1beta1.BKELogger{
			Recorder:     record.NewBroadcaster().NewRecorder(scheme, corev1.EventSource{Component: "test"}),
			NormalLogger: testutils.NewLog(),
			EventBinder:  cluster,
		},
	}
}

// createBKECluster 创建测试用的 BKECluster
func createBKECluster(specVersion, statusVersion string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test-cluster", Namespace: "kube-system"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: specVersion},
			},
		},
		Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: statusVersion},
	}
}

// createDeployment 创建 Deployment
func createDeployment(image string) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: providerDeploymentName, Namespace: providerNamespace},
		Spec: appsv1.DeploymentSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: providerContainerName, Image: image}},
				},
			},
		},
	}
}

// createLocalCM 创建本地 ConfigMap
func createLocalCM(patchKey string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.GetLocalConfigMapObjectKey().Name,
			Namespace: constant.GetLocalConfigMapObjectKey().Namespace,
		},
		Data: map[string]string{patchKey: "true"},
	}
}

// createPatchCM 创建 Patch ConfigMap
func createPatchCM(name, version, yamlContent string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "openfuyao-patch"},
		Data:       map[string]string{version: yamlContent},
	}
}

// TestEnsureProviderSelfUpgradeIsPatchVersion 测试补丁版本判断
func TestEnsureProviderSelfUpgradeIsPatchVersion(t *testing.T) {
	tests := []struct {
		name    string
		version string
		want    bool
	}{
		{"Valid patch version", "v1.2.3", true},
		{"Non-patch version", "v1.2.0", false},
		{"Pre-release version", "v1.2.3-alpha", false},
		{"Invalid version", "invalid", false},
		{"Without v prefix", "1.2.3", true},
	}

	p := createTestPhase(t)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := p.isPatchVersion(tt.version)
			assert.Equal(t, tt.want, got)
		})
	}
}

// TestEnsureProviderSelfUpgradeIsProviderImage 测试镜像匹配
func TestEnsureProviderSelfUpgradeIsProviderImage(t *testing.T) {
	p := createTestPhase(t)

	tests := []struct {
		name  string
		image phaseutil.Image
		want  bool
	}{
		{
			name:  "Match by image name",
			image: phaseutil.Image{Name: "cluster-api-provider-bke"},
			want:  true,
		},
		{
			name: "Match by PodInfo",
			image: phaseutil.Image{
				Name:        "some-image",
				UsedPodInfo: []phaseutil.PodInfo{{PodPrefix: providerDeploymentName, NameSpace: providerNamespace}},
			},
			want: true,
		},
		{
			name:  "No match",
			image: phaseutil.Image{Name: "unrelated-image"},
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := p.isProviderImage(tt.image)
			assert.Equal(t, tt.want, got)
		})
	}
}

// createTestPhase 创建测试用的 Phase 实例
func createTestPhase(t *testing.T) *EnsureProviderSelfUpgrade {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	cluster := createBKECluster("v1.0.0", "")
	ctx := createPhaseContext(fakeClient, scheme, cluster)

	phase := NewEnsureProviderSelfUpgrade(ctx)
	p, ok := phase.(*EnsureProviderSelfUpgrade)
	assert.True(t, ok)
	return p
}

// TestEnsureProviderSelfUpgradeNeedExecute tests NeedExecute method
func TestEnsureProviderSelfUpgradeNeedExecute(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	_ = appsv1.AddToScheme(scheme)

	patchKey := fmt.Sprintf("patch.%s", testTargetVersion)
	patchCMName := fmt.Sprintf("cm.%s", testTargetVersion)

	tests := []struct {
		name         string
		oldCluster   *bkev1beta1.BKECluster
		newCluster   *bkev1beta1.BKECluster
		existingObjs []client.Object
		want         bool
	}{
		{
			name:       "Need upgrade - images differ",
			oldCluster: createBKECluster("v1.2.0", "v1.2.0"),
			newCluster: createBKECluster(testTargetVersion, "v1.2.0"),
			existingObjs: []client.Object{
				createDeployment(testOldImage),
				createLocalCM(patchKey),
				createPatchCM(patchCMName, testTargetVersion, getValidPatchYaml()),
			},
			want: true,
		},
		{
			name:       "No upgrade - images match",
			oldCluster: createBKECluster("v1.2.0", "v1.2.0"),
			newCluster: createBKECluster(testTargetVersion, "v1.2.0"),
			existingObjs: []client.Object{
				createDeployment(testTargetImage),
				createLocalCM(patchKey),
				createPatchCM(patchCMName, testTargetVersion, getValidPatchYaml()),
			},
			want: false,
		},
		{
			name:       "No upgrade - version unchanged",
			oldCluster: createBKECluster("v1.2.0", "v1.2.0"),
			newCluster: createBKECluster("v1.2.0", "v1.2.0"),
			want:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(tt.existingObjs...).Build()
			ctx := createPhaseContext(fakeClient, scheme, tt.newCluster)

			phase := NewEnsureProviderSelfUpgrade(ctx).(*EnsureProviderSelfUpgrade)
			got := phase.NeedExecute(tt.oldCluster, tt.newCluster)
			assert.Equal(t, tt.want, got)
		})
	}
}

// TestEnsureProviderSelfUpgradeExecute tests Execute method
func TestEnsureProviderSelfUpgradeExecute(t *testing.T) {
	t.Skip("Skipping - requires complex mocking of WaitDeploymentReady")
}

// TestEnsureProviderSelfUpgradeRolloutProviderPatchError tests rolloutProvider with patch error
func TestEnsureProviderSelfUpgradeRolloutProviderPatchError(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

// TestEnsureProviderSelfUpgradeRolloutProviderWaitError tests rolloutProvider with wait error
func TestEnsureProviderSelfUpgradeRolloutProviderWaitError(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

// TestEnsureProviderSelfUpgradePostHook tests PostHook method
func TestEnsureProviderSelfUpgradePostHook(t *testing.T) {
	t.Skip("Skipping - requires complex metrics setup")
}

// TestEnsureProviderSelfUpgradeFindProviderImageInSubImage tests findProviderImageInSubImage
func TestEnsureProviderSelfUpgradeFindProviderImageInSubImage(t *testing.T) {
	p := createTestPhase(t)

	tests := []struct {
		name      string
		subImage  phaseutil.SubImage
		wantImage string
		wantFound bool
	}{
		{
			name: "Found with tag",
			subImage: phaseutil.SubImage{
				SourceRepo: "docker.io/test",
				Images: []phaseutil.Image{
					{Name: "cluster-api-provider-bke", Tag: []string{"v1.0.0"}},
				},
			},
			wantImage: "docker.io/test/cluster-api-provider-bke:v1.0.0",
			wantFound: true,
		},
		{
			name: "Not found - no matching image",
			subImage: phaseutil.SubImage{
				SourceRepo: "docker.io/test",
				Images: []phaseutil.Image{
					{Name: "other-image", Tag: []string{"v1.0.0"}},
				},
			},
			wantImage: "",
			wantFound: false,
		},
		{
			name: "Not found - no tag",
			subImage: phaseutil.SubImage{
				SourceRepo: "docker.io/test",
				Images: []phaseutil.Image{
					{Name: "cluster-api-provider-bke", Tag: []string{}},
				},
			},
			wantImage: "",
			wantFound: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotImage, gotFound := p.findProviderImageInSubImage(tt.subImage)
			assert.Equal(t, tt.wantImage, gotImage)
			assert.Equal(t, tt.wantFound, gotFound)
		})
	}
}

// TestEnsureProviderSelfUpgradeFindProviderImageInPatchConfig tests findProviderImageInPatchConfig
func TestEnsureProviderSelfUpgradeFindProviderImageInPatchConfig(t *testing.T) {
	p := createTestPhase(t)

	tests := []struct {
		name        string
		patchConfig *phaseutil.PatchConfig
		wantImage   string
		wantErr     bool
	}{
		{
			name: "Found in patch config",
			patchConfig: &phaseutil.PatchConfig{
				Repos: []phaseutil.Repo{
					{
						SubImages: []phaseutil.SubImage{
							{
								SourceRepo: "docker.io/test",
								Images: []phaseutil.Image{
									{Name: "cluster-api-provider-bke", Tag: []string{"v1.0.0"}},
								},
							},
						},
					},
				},
			},
			wantImage: "docker.io/test/cluster-api-provider-bke:v1.0.0",
			wantErr:   false,
		},
		{
			name: "Not found in patch config",
			patchConfig: &phaseutil.PatchConfig{
				Repos: []phaseutil.Repo{
					{
						SubImages: []phaseutil.SubImage{
							{
								SourceRepo: "docker.io/test",
								Images: []phaseutil.Image{
									{Name: "other-image", Tag: []string{"v1.0.0"}},
								},
							},
						},
					},
				},
			},
			wantImage: "",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotImage, err := p.findProviderImageInPatchConfig(tt.patchConfig)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			assert.Equal(t, tt.wantImage, gotImage)
		})
	}
}

// TestEnsureProviderSelfUpgradeGetPatchConfigSuccess tests getPatchConfig success case
func TestEnsureProviderSelfUpgradeGetPatchConfigSuccess(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	patchKey := fmt.Sprintf("patch.%s", testTargetVersion)
	patchCMName := fmt.Sprintf("cm.%s", testTargetVersion)

	cluster := createBKECluster(testTargetVersion, "")
	existingObjs := []client.Object{
		createLocalCM(patchKey),
		createPatchCM(patchCMName, testTargetVersion, getValidPatchYaml()),
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(existingObjs...).Build()
	ctx := createPhaseContext(fakeClient, scheme, cluster)
	phase := NewEnsureProviderSelfUpgrade(ctx).(*EnsureProviderSelfUpgrade)

	patchConfig, err := phase.getPatchConfig(cluster)
	assert.NoError(t, err)
	assert.NotNil(t, patchConfig)
	assert.NotEmpty(t, patchConfig.Repos)
}

// TestEnsureProviderSelfUpgradeGetPatchConfigLocalCMMissing tests getPatchConfig with missing local CM
func TestEnsureProviderSelfUpgradeGetPatchConfigLocalCMMissing(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	cluster := createBKECluster(testTargetVersion, "")
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	ctx := createPhaseContext(fakeClient, scheme, cluster)
	phase := NewEnsureProviderSelfUpgrade(ctx).(*EnsureProviderSelfUpgrade)

	patchConfig, err := phase.getPatchConfig(cluster)
	assert.Error(t, err)
	assert.Nil(t, patchConfig)
}

// TestEnsureProviderSelfUpgradeGetPatchConfigNoPatchKey tests getPatchConfig with no patch key
func TestEnsureProviderSelfUpgradeGetPatchConfigNoPatchKey(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	cluster := createBKECluster(testTargetVersion, "")
	existingObjs := []client.Object{
		createLocalCM("other-key"),
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(existingObjs...).Build()
	ctx := createPhaseContext(fakeClient, scheme, cluster)
	phase := NewEnsureProviderSelfUpgrade(ctx).(*EnsureProviderSelfUpgrade)

	patchConfig, err := phase.getPatchConfig(cluster)
	assert.Error(t, err)
	assert.Nil(t, patchConfig)
	assert.Contains(t, err.Error(), "non-patch version")
}

// TestEnsureProviderSelfUpgradeGetPatchConfigPatchCMMissing tests getPatchConfig with missing patch CM
func TestEnsureProviderSelfUpgradeGetPatchConfigPatchCMMissing(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	patchKey := fmt.Sprintf("patch.%s", testTargetVersion)
	cluster := createBKECluster(testTargetVersion, "")
	existingObjs := []client.Object{
		createLocalCM(patchKey),
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(existingObjs...).Build()
	ctx := createPhaseContext(fakeClient, scheme, cluster)
	phase := NewEnsureProviderSelfUpgrade(ctx).(*EnsureProviderSelfUpgrade)

	patchConfig, err := phase.getPatchConfig(cluster)
	assert.Error(t, err)
	assert.Nil(t, patchConfig)
}

// TestEnsureProviderSelfUpgradeGetProviderTargetImageSuccess tests getProviderTargetImage success
func TestEnsureProviderSelfUpgradeGetProviderTargetImageSuccess(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	patchKey := fmt.Sprintf("patch.%s", testTargetVersion)
	patchCMName := fmt.Sprintf("cm.%s", testTargetVersion)

	cluster := createBKECluster(testTargetVersion, "")
	existingObjs := []client.Object{
		createLocalCM(patchKey),
		createPatchCM(patchCMName, testTargetVersion, getValidPatchYaml()),
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(existingObjs...).Build()
	ctx := createPhaseContext(fakeClient, scheme, cluster)
	phase := NewEnsureProviderSelfUpgrade(ctx).(*EnsureProviderSelfUpgrade)

	image, err := phase.getProviderTargetImage(cluster)
	assert.NoError(t, err)
	assert.Equal(t, testTargetImage, image)
}

// TestEnsureProviderSelfUpgradeGetProviderTargetImageError tests getProviderTargetImage error
func TestEnsureProviderSelfUpgradeGetProviderTargetImageError(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	cluster := createBKECluster(testTargetVersion, "")
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	ctx := createPhaseContext(fakeClient, scheme, cluster)
	phase := NewEnsureProviderSelfUpgrade(ctx).(*EnsureProviderSelfUpgrade)

	image, err := phase.getProviderTargetImage(cluster)
	assert.Error(t, err)
	assert.Empty(t, image)
}

// TestEnsureProviderSelfUpgradeRolloutProviderContextCanceled tests context canceled scenario
func TestEnsureProviderSelfUpgradeRolloutProviderContextCanceled(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

func newProviderSelfUpgradeCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster, objs ...client.Object) *EnsureProviderSelfUpgrade {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, appsv1.AddToScheme(scheme))
	builder := ctrlfake.NewClientBuilder().WithScheme(scheme)
	if bkeCluster != nil {
		builder = builder.WithObjects(bkeCluster)
	}
	for _, o := range objs {
		builder = builder.WithObjects(o)
	}
	c := builder.Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log: &bkev1beta1.BKELogger{
			Recorder:     record.NewBroadcaster().NewRecorder(scheme, corev1.EventSource{Component: "test"}),
			NormalLogger: testutils.NewLog(),
			EventBinder:  bkeCluster,
		},
	}
	return &EnsureProviderSelfUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

// providerClusterWithCMs builds a BKECluster + local CM + patch CM so getProviderTargetImage succeeds.
func providerClusterWithCMs(t *testing.T) (*bkev1beta1.BKECluster, []client.Object) {
	t.Helper()
	version := testTargetVersion
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test-cluster", Namespace: "kube-system"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: version},
			},
		},
	}
	patchKey := "patch." + version
	patchCMName := "cm." + version
	objs := []client.Object{
		createLocalCM(patchKey),
		createPatchCM(patchCMName, version, getValidPatchYaml()),
	}
	return cluster, objs
}

// ---- Execute (0% -> cover via real rolloutProvider with patched deps) ----

func TestProviderSelfUpgradeExecute(t *testing.T) {
	t.Run("rolloutProvider error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster, objs := providerClusterWithCMs(t)
		p := newProviderSelfUpgradeCov(t, cluster, objs...)
		patches.ApplyFunc(phaseutil.PatchDeploymentImage, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget, _ string) error {
			return assertErr("patch failed")
		})
		_, err := p.Execute()
		require.Error(t, err)
	})
}

// ---- rolloutProvider (0% -> cover all branches) ----

func TestProviderSelfUpgradeRolloutProvider(t *testing.T) {
	t.Run("target image empty returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v1.0.0"},
				},
			},
		}
		p := newProviderSelfUpgradeCov(t, cluster)
		result, err := p.rolloutProvider()
		require.Error(t, err)
		assert.False(t, result.Requeue)
		assert.Contains(t, err.Error(), "unable to parse target image")
	})

	t.Run("patch deployment error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster, objs := providerClusterWithCMs(t)
		p := newProviderSelfUpgradeCov(t, cluster, objs...)
		patches.ApplyFunc(phaseutil.PatchDeploymentImage, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget, _ string) error {
			return assertErr("patch failed")
		})
		result, err := p.rolloutProvider()
		require.Error(t, err)
		assert.False(t, result.Requeue)
		assert.Contains(t, err.Error(), "patch Deployment failed")
	})

	t.Run("wait ready error normal", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster, objs := providerClusterWithCMs(t)
		p := newProviderSelfUpgradeCov(t, cluster, objs...)
		patches.ApplyFunc(phaseutil.PatchDeploymentImage, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget, _ string) error {
			return nil
		})
		patches.ApplyFunc(phaseutil.WaitDeploymentReady, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget, _ string, _ time.Duration) error {
			return assertErr("deployment not ready")
		})
		result, err := p.rolloutProvider()
		require.Error(t, err)
		assert.False(t, result.Requeue)
		assert.Contains(t, err.Error(), "wait for Deployment ready failed")
	})

	t.Run("wait ready context canceled image mismatch error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster, objs := providerClusterWithCMs(t)
		p := newProviderSelfUpgradeCov(t, cluster, objs...)
		patches.ApplyFunc(phaseutil.PatchDeploymentImage, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget, _ string) error {
			return nil
		})
		patches.ApplyFunc(phaseutil.WaitDeploymentReady, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget, _ string, _ time.Duration) error {
			return assertErr("context canceled while waiting")
		})
		patches.ApplyFunc(phaseutil.GetDeploymentImage, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget) (string, error) {
			return "some-other-image:v1", nil
		})
		result, err := p.rolloutProvider()
		require.Error(t, err)
		assert.False(t, result.Requeue)
		assert.Contains(t, err.Error(), "wait for Deployment ready failed")
	})

	t.Run("wait ready context canceled get image error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster, objs := providerClusterWithCMs(t)
		p := newProviderSelfUpgradeCov(t, cluster, objs...)
		patches.ApplyFunc(phaseutil.PatchDeploymentImage, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget, _ string) error {
			return nil
		})
		patches.ApplyFunc(phaseutil.WaitDeploymentReady, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget, _ string, _ time.Duration) error {
			return assertErr("context canceled while waiting")
		})
		patches.ApplyFunc(phaseutil.GetDeploymentImage, func(_ context.Context, _ client.Client, _ phaseutil.DeploymentTarget) (string, error) {
			return "", assertErr("get image failed")
		})
		result, err := p.rolloutProvider()
		require.Error(t, err)
		assert.False(t, result.Requeue)
	})
}

// ---- PostHook (0% -> cover sleep + default hook) ----

func TestProviderSelfUpgradePostHook(t *testing.T) {
	t.Run("nil err sleeps and returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster, objs := providerClusterWithCMs(t)
		p := newProviderSelfUpgradeCov(t, cluster, objs...)
		slept := false
		patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultPostHook", func(_ *phaseframe.BasePhase, _ error) error { return nil })
		patches.ApplyFunc(time.Sleep, func(_ time.Duration) { slept = true })
		err := p.PostHook(nil)
		require.NoError(t, err)
		assert.True(t, slept)
	})

	t.Run("non-nil err no sleep", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster, objs := providerClusterWithCMs(t)
		p := newProviderSelfUpgradeCov(t, cluster, objs...)
		slept := false
		patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultPostHook", func(_ *phaseframe.BasePhase, _ error) error { return nil })
		patches.ApplyFunc(time.Sleep, func(_ time.Duration) { slept = true })
		err := p.PostHook(assertErr("upgrade failed"))
		require.NoError(t, err)
		assert.False(t, slept)
	})

	t.Run("default posthook error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster, objs := providerClusterWithCMs(t)
		p := newProviderSelfUpgradeCov(t, cluster, objs...)
		patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultPostHook", func(_ *phaseframe.BasePhase, _ error) error {
			return assertErr("posthook failed")
		})
		err := p.PostHook(assertErr("upgrade failed"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "posthook failed")
	})
}

// ---- getPatchConfig additional branch (non-patch version key missing) ----

func TestProviderSelfUpgradeGetPatchConfigVersionKeyMissing(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	cluster := createBKECluster(testTargetVersion, "")
	// local CM has the patch key but patch CM lacks the version data key
	patchKey := "patch." + testTargetVersion
	patchCMName := "cm." + testTargetVersion
	objs := []client.Object{
		createLocalCM(patchKey),
		&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: patchCMName, Namespace: "openfuyao-patch"},
			Data:       map[string]string{"other-version": "data"},
		},
	}

	fakeClient := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: cluster,
		Scheme:     scheme,
		Context:    context.Background(),
		Client:     fakeClient,
		Log: &bkev1beta1.BKELogger{
			Recorder:     record.NewBroadcaster().NewRecorder(scheme, corev1.EventSource{Component: "test"}),
			NormalLogger: testutils.NewLog(),
			EventBinder:  cluster,
		},
	}
	p := NewEnsureProviderSelfUpgrade(ctx).(*EnsureProviderSelfUpgrade)

	patchConfig, err := p.getPatchConfig(cluster)
	require.Error(t, err)
	assert.Nil(t, patchConfig)
	assert.Contains(t, err.Error(), "not found in patch config")
}
