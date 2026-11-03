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

package helminstaller

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/storage"
	"helm.sh/helm/v3/pkg/storage/driver"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/healthcheck"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type fakeComponentFileStore struct {
	files map[string][]byte
	err   error
}

func (f fakeComponentFileStore) GetComponentFile(
	_ context.Context,
	name, version, file string,
) ([]byte, error) {
	if f.err != nil {
		return nil, f.err
	}
	data, ok := f.files[name+"@"+version+"/"+file]
	if !ok {
		return nil, os.ErrNotExist
	}
	out := make([]byte, len(data))
	copy(out, data)
	return out, nil
}

func helmTestCluster() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "bke-system"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
	}
}

func helmTestClient() client.Client {
	return ctrlfake.NewClientBuilder().Build()
}

func helmCV(name, version string, helm *cvv1alpha1.HelmSpec) *cvv1alpha1.ComponentVersion {
	return &cvv1alpha1.ComponentVersion{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: cvv1alpha1.ComponentVersionSpec{
			Name:    name,
			Type:    cvv1alpha1.ComponentTypeHelm,
			Version: version,
			Helm:    helm,
		},
	}
}

func TestApplySelectsInstallWhenReleaseMissing(t *testing.T) {
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store:       fakeComponentFileStore{},
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})
	cv := helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{})
	var installed bool

	patches := gomonkey.ApplyPrivateMethod(reflect.TypeOf(inst), "initActionConfig",
		func(_ *HelmInstaller, namespace string, execCtx *ApplyContext) (*action.Configuration, error) {
			assert.Equal(t, defaultNamespace, namespace)
			require.NotNil(t, execCtx)
			return helmMemoryActionConfig(), nil
		})
	defer patches.Reset()
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "loadValues",
		func(_ *HelmInstaller, _ context.Context, gotCV *cvv1alpha1.ComponentVersion, _ *ApplyContext) (chartutil.Values, error) {
			assert.Same(t, cv, gotCV)
			return chartutil.Values{"replicas": 1}, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "fetchChart",
		func(_ *HelmInstaller, _ context.Context, gotCV *cvv1alpha1.ComponentVersion, _ *ApplyContext) (*chart.Chart, error) {
			assert.Same(t, cv, gotCV)
			return &chart.Chart{}, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "releaseExists",
		func(_ *HelmInstaller, _ *action.Configuration, releaseName string) (bool, error) {
			assert.Equal(t, "coredns", releaseName)
			return false, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "install",
		func(_ *HelmInstaller, _ context.Context, opts helmRunOptions) error {
			installed = true
			assert.Equal(t, "coredns", opts.releaseName)
			assert.Equal(t, defaultNamespace, opts.namespace)
			return nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "maybeHealthCheck",
		func(_ *HelmInstaller, _ context.Context, helmSpec *cvv1alpha1.HelmSpec, _ *ApplyContext) error {
			assert.Same(t, cv.Spec.Helm, helmSpec)
			return nil
		})

	require.NoError(t, inst.Apply(context.Background(), cv, &ApplyContext{}, ""))
	assert.True(t, installed)
}

func TestApplySelectsUpgradeWhenReleaseExists(t *testing.T) {
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store:       fakeComponentFileStore{},
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})
	cv := helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{})
	var upgraded bool

	patches := gomonkey.ApplyPrivateMethod(reflect.TypeOf(inst), "initActionConfig",
		func(_ *HelmInstaller, _ string, _ *ApplyContext) (*action.Configuration, error) {
			return helmMemoryActionConfig(), nil
		})
	defer patches.Reset()
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "loadValues",
		func(_ *HelmInstaller, _ context.Context, _ *cvv1alpha1.ComponentVersion, _ *ApplyContext) (chartutil.Values, error) {
			return chartutil.Values{}, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "fetchChart",
		func(_ *HelmInstaller, _ context.Context, _ *cvv1alpha1.ComponentVersion, _ *ApplyContext) (*chart.Chart, error) {
			return &chart.Chart{}, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "releaseExists",
		func(_ *HelmInstaller, _ *action.Configuration, _ string) (bool, error) {
			return true, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "upgrade",
		func(_ *HelmInstaller, opts helmRunOptions) error {
			upgraded = true
			assert.Equal(t, "coredns", opts.releaseName)
			return nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "maybeHealthCheck",
		func(_ *HelmInstaller, _ context.Context, _ *cvv1alpha1.HelmSpec, _ *ApplyContext) error {
			return nil
		})

	require.NoError(t, inst.Apply(context.Background(), cv, &ApplyContext{}, ""))
	assert.True(t, upgraded)
}

func TestApplyPropagatesStageErrors(t *testing.T) {
	tests := []struct {
		name       string
		patchStage func(*gomonkey.Patches, *HelmInstaller)
		action     Action
		want       string
	}{
		{
			name: "init action config",
			patchStage: func(p *gomonkey.Patches, inst *HelmInstaller) {
				p.ApplyPrivateMethod(reflect.TypeOf(inst), "initActionConfig",
					func(_ *HelmInstaller, _ string, _ *ApplyContext) (*action.Configuration, error) {
						return nil, errors.New("rest config invalid")
					})
			},
			want: "init helm action config",
		},
		{
			name: "load values",
			patchStage: func(p *gomonkey.Patches, inst *HelmInstaller) {
				patchApplyHappyPathBeforeRun(t, p, inst)
				p.ApplyPrivateMethod(reflect.TypeOf(inst), "loadValues",
					func(_ *HelmInstaller, _ context.Context, _ *cvv1alpha1.ComponentVersion, _ *ApplyContext) (chartutil.Values, error) {
						return nil, errors.New("values invalid")
					})
			},
			want: "load helm values",
		},
		{
			name: "fetch chart",
			patchStage: func(p *gomonkey.Patches, inst *HelmInstaller) {
				patchApplyHappyPathBeforeRun(t, p, inst)
				p.ApplyPrivateMethod(reflect.TypeOf(inst), "fetchChart",
					func(_ *HelmInstaller, _ context.Context, _ *cvv1alpha1.ComponentVersion, _ *ApplyContext) (*chart.Chart, error) {
						return nil, errors.New("fetch failed")
					})
			},
			want: "fetch helm chart",
		},
		{
			name: "release exists",
			patchStage: func(p *gomonkey.Patches, inst *HelmInstaller) {
				patchApplyHappyPathBeforeRun(t, p, inst)
				p.ApplyPrivateMethod(reflect.TypeOf(inst), "releaseExists",
					func(_ *HelmInstaller, _ *action.Configuration, _ string) (bool, error) {
						return false, errors.New("status failed")
					})
			},
			action: "",
			want:   "check helm release",
		},
		{
			name: "unsupported action",
			patchStage: func(p *gomonkey.Patches, inst *HelmInstaller) {
				patchApplyHappyPathBeforeRun(t, p, inst)
			},
			action: Action("Delete"),
			want:   "unsupported helm action",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inst := NewHelmInstaller(HelmInstallerConfig{
				Store:       fakeComponentFileStore{},
				LocalClient: helmTestClient(),
				BKECluster:  helmTestCluster(),
			})
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			tt.patchStage(patches, inst)
			err := inst.Apply(context.Background(), helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{}), &ApplyContext{}, tt.action)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}

func patchApplyHappyPathBeforeRun(t *testing.T, patches *gomonkey.Patches, inst *HelmInstaller) {
	t.Helper()
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "initActionConfig",
		func(_ *HelmInstaller, _ string, _ *ApplyContext) (*action.Configuration, error) {
			return helmMemoryActionConfig(), nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "loadValues",
		func(_ *HelmInstaller, _ context.Context, _ *cvv1alpha1.ComponentVersion, _ *ApplyContext) (chartutil.Values, error) {
			return chartutil.Values{}, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "fetchChart",
		func(_ *HelmInstaller, _ context.Context, _ *cvv1alpha1.ComponentVersion, _ *ApplyContext) (*chart.Chart, error) {
			return &chart.Chart{}, nil
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(inst), "releaseExists",
		func(_ *HelmInstaller, _ *action.Configuration, _ string) (bool, error) {
			return false, nil
		})
}

func TestUninstallPropagatesConfigAndTimeoutErrors(t *testing.T) {
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store:       fakeComponentFileStore{},
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})
	patches := gomonkey.ApplyPrivateMethod(reflect.TypeOf(inst), "initActionConfig",
		func(_ *HelmInstaller, _ string, _ *ApplyContext) (*action.Configuration, error) {
			return nil, errors.New("init failed")
		})
	defer patches.Reset()

	cv := helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{})
	err := inst.Uninstall(context.Background(), cv, &ApplyContext{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "init helm action config")

	patches.Reset()
	patches = gomonkey.ApplyPrivateMethod(reflect.TypeOf(inst), "initActionConfig",
		func(_ *HelmInstaller, _ string, _ *ApplyContext) (*action.Configuration, error) {
			return helmMemoryActionConfig(), nil
		})
	defer patches.Reset()
	cv.Spec.Helm.Strategy.Timeout = "bad"
	err = inst.Uninstall(context.Background(), cv, &ApplyContext{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "parse helm timeout")
}

func TestFetchChartRendersNamesAndPropagatesErrors(t *testing.T) {
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store:       fakeComponentFileStore{},
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})
	cv := helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{
		Chart: cvv1alpha1.ChartSpec{
			Name:    "{{ .ClusterName }}-chart",
			Version: "{{ .KubernetesVersion }}",
		},
	})

	patches := gomonkey.ApplyFunc(kube.ChartRepositoryConfigFromBKECluster,
		func(context.Context, client.Client, *bkev1beta1.BKECluster) (string, *kube.AuthConfig, error) {
			return "oci://repo.example.com/charts", kube.NewAuthConfig(), nil
		})
	defer patches.Reset()
	patches.ApplyFunc(kube.FetchChartUniversal,
		func(repoURL, chartName, version string, _ *kube.AuthConfig, _ *log.Logger) (*chart.Chart, error) {
			assert.Equal(t, "oci://repo.example.com/charts", repoURL)
			assert.Equal(t, "demo-chart", chartName)
			assert.Equal(t, "v1.34.3", version)
			return &chart.Chart{}, nil
		})

	got, err := inst.fetchChart(context.Background(), cv, &ApplyContext{
		TemplateContext: manifest.TemplateContext{ClusterName: "demo", KubernetesVersion: "v1.34.3"},
	})
	require.NoError(t, err)
	assert.NotNil(t, got)

	patches.Reset()
	patches = gomonkey.ApplyFunc(kube.ChartRepositoryConfigFromBKECluster,
		func(context.Context, client.Client, *bkev1beta1.BKECluster) (string, *kube.AuthConfig, error) {
			return "", kube.NewAuthConfig(), nil
		})
	defer patches.Reset()
	_, err = inst.fetchChart(context.Background(), helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{}), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "chart repository is empty")
}

func TestFetchChartRenderAndRepositoryErrors(t *testing.T) {
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store:       fakeComponentFileStore{},
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})

	_, err := inst.fetchChart(context.Background(), helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{
		Chart: cvv1alpha1.ChartSpec{Name: "{{ .Missing }}"},
	}), &ApplyContext{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "render chart.name")

	patches := gomonkey.ApplyFunc(kube.ChartRepositoryConfigFromBKECluster,
		func(context.Context, client.Client, *bkev1beta1.BKECluster) (string, *kube.AuthConfig, error) {
			return "", nil, errors.New("repo config failed")
		})
	defer patches.Reset()
	_, err = inst.fetchChart(context.Background(), helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{}), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "repo config failed")

	patches.Reset()
	patches = gomonkey.ApplyFunc(kube.ChartRepositoryConfigFromBKECluster,
		func(context.Context, client.Client, *bkev1beta1.BKECluster) (string, *kube.AuthConfig, error) {
			return "oci://repo.example.com", kube.NewAuthConfig(), nil
		})
	defer patches.Reset()
	patches.ApplyFunc(kube.FetchChartUniversal,
		func(string, string, string, *kube.AuthConfig, *log.Logger) (*chart.Chart, error) {
			return nil, errors.New("fetch failed")
		})
	_, err = inst.fetchChart(context.Background(), helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{}), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fetch failed")
}

func TestLoggerHelpers(t *testing.T) {
	logger := log.With("module", "helm-test")
	bkeLogger := &bkev1beta1.BKELogger{NormalLogger: logger}
	assert.Same(t, logger, normalLogger(bkeLogger))
	assert.NotNil(t, normalLogger(nil))

	warnf(nil, "test warning %s", "without logger")
}

func helmMemoryActionConfig() *action.Configuration {
	return &action.Configuration{
		Releases: storage.Init(driver.NewMemory()),
		Log:      func(string, ...interface{}) {},
	}
}
func TestNewHelmInstaller(t *testing.T) {
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store:       fakeComponentFileStore{},
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})
	require.NotNil(t, inst)
}

func TestValidateComponent(t *testing.T) {
	validHelm := &cvv1alpha1.HelmSpec{}
	tests := []struct {
		name    string
		inst    *HelmInstaller
		cv      *cvv1alpha1.ComponentVersion
		wantErr string
	}{
		{
			name:    "nil installer",
			inst:    nil,
			cv:      helmCV("coredns", "v1.0.0", validHelm),
			wantErr: "helm installer is nil",
		},
		{
			name: "nil component version",
			inst: NewHelmInstaller(HelmInstallerConfig{
				Store:       fakeComponentFileStore{},
				LocalClient: helmTestClient(),
				BKECluster:  helmTestCluster(),
			}),
			wantErr: "component version is nil",
		},
		{
			name: "missing helm spec",
			inst: NewHelmInstaller(HelmInstallerConfig{
				Store:       fakeComponentFileStore{},
				LocalClient: helmTestClient(),
				BKECluster:  helmTestCluster(),
			}),
			cv:      helmCV("coredns", "v1.0.0", nil),
			wantErr: "spec.helm is required",
		},
		{
			name: "missing store",
			inst: NewHelmInstaller(HelmInstallerConfig{
				LocalClient: helmTestClient(),
				BKECluster:  helmTestCluster(),
			}),
			cv:      helmCV("coredns", "v1.0.0", validHelm),
			wantErr: "component file store is nil",
		},
		{
			name: "missing local client",
			inst: NewHelmInstaller(HelmInstallerConfig{
				Store:      fakeComponentFileStore{},
				BKECluster: helmTestCluster(),
			}),
			cv:      helmCV("coredns", "v1.0.0", validHelm),
			wantErr: "management client is nil",
		},
		{
			name: "missing cluster",
			inst: NewHelmInstaller(HelmInstallerConfig{
				Store:       fakeComponentFileStore{},
				LocalClient: helmTestClient(),
			}),
			cv:      helmCV("coredns", "v1.0.0", validHelm),
			wantErr: "bke cluster is nil",
		},
		{
			name: "uses object name fallback",
			inst: NewHelmInstaller(HelmInstallerConfig{
				Store:       fakeComponentFileStore{},
				LocalClient: helmTestClient(),
				BKECluster:  helmTestCluster(),
			}),
			cv: &cvv1alpha1.ComponentVersion{
				ObjectMeta: metav1.ObjectMeta{Name: "fallback"},
				Spec:       cvv1alpha1.ComponentVersionSpec{Helm: validHelm},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotName, gotHelm, err := tt.inst.validateComponent(tt.cv)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, "fallback", gotName)
			assert.Same(t, validHelm, gotHelm)
		})
	}
}

func TestReleaseDefaultsAndChartFallbacks(t *testing.T) {
	cv := helmCV("coredns", "v1.12.3", &cvv1alpha1.HelmSpec{})
	assert.Equal(t, defaultNamespace, releaseNamespace(cv.Spec.Helm))
	assert.Equal(t, "coredns", releaseName("coredns", cv.Spec.Helm))
	assert.Equal(t, "coredns", chartName(cv))
	assert.Equal(t, "v1.12.3", chartVersion(cv))

	cv.Spec.Helm.Namespace = " kube-system "
	cv.Spec.Helm.ReleaseName = " kube-dns "
	cv.Spec.Helm.Chart.Name = " coredns-chart "
	cv.Spec.Helm.Chart.Version = " 1.2.3 "
	assert.Equal(t, "kube-system", releaseNamespace(cv.Spec.Helm))
	assert.Equal(t, "kube-dns", releaseName("coredns", cv.Spec.Helm))
	assert.Equal(t, "coredns-chart", chartName(cv))
	assert.Equal(t, "1.2.3", chartVersion(cv))
}

func TestParseTimeout(t *testing.T) {
	got, err := parseTimeout("")
	require.NoError(t, err)
	assert.Zero(t, got)

	got, err = parseTimeout("2m30s")
	require.NoError(t, err)
	assert.Equal(t, "2m30s", got.String())

	_, err = parseTimeout("bad")
	require.Error(t, err)
}

func TestRenderValueRecursively(t *testing.T) {
	tmpl := manifest.TemplateContext{
		ClusterName:       "demo",
		Namespace:         "bke-system",
		KubernetesVersion: "v1.34.3",
	}
	in := map[string]interface{}{
		"cluster": "{{ .ClusterName }}",
		"nested": map[string]interface{}{
			"{{ .Namespace }}": []interface{}{"{{ .KubernetesVersion }}", true},
		},
	}

	got, err := renderValue(in, tmpl)
	require.NoError(t, err)
	out := got.(map[string]interface{})
	assert.Equal(t, "demo", out["cluster"])
	nested := out["nested"].(map[string]interface{})
	assert.Equal(t, []interface{}{"v1.34.3", true}, nested["bke-system"])
}

func TestRenderValueMissingKeyFails(t *testing.T) {
	_, err := renderValue("{{ .Missing }}", manifest.TemplateContext{})
	require.Error(t, err)
}

func TestMergeValuesMergesNestedMapsAndOverridesScalars(t *testing.T) {
	dst := map[string]interface{}{
		"image":    map[string]interface{}{"repo": "old", "tag": "v1"},
		"replicas": 1,
	}
	src := map[string]interface{}{
		"image":    map[string]interface{}{"tag": "v2"},
		"replicas": 2,
	}

	mergeValues(dst, src)
	assert.Equal(t, "old", dst["image"].(map[string]interface{})["repo"])
	assert.Equal(t, "v2", dst["image"].(map[string]interface{})["tag"])
	assert.Equal(t, 2, dst["replicas"])
}

func TestLoadValuesMergesFilesThenInlineValues(t *testing.T) {
	store := fakeComponentFileStore{files: map[string][]byte{
		"coredns@v1.0.0/base.yaml": []byte(`apiVersion: v1
kind: ConfigMap
data:
  values.yaml: |
    image:
      repository: {{ .ImageRegistry }}coredns
      tag: old
    replicas: 1
`),
		"coredns@v1.0.0/override.yaml": []byte(`apiVersion: v1
kind: ConfigMap
data:
  values.yaml: |
    image:
      tag: {{ .KubernetesVersion }}
    service:
      clusterIP: None
`),
	}}
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store:       store,
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})
	cv := helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{
		ValuesFiles: []string{"base.yaml", "missing.yaml", "override.yaml"},
		Values: map[string]interface{}{
			"image": map[string]interface{}{"tag": "inline"},
			"env":   "{{ .ClusterName }}",
		},
	})

	values, err := inst.loadValues(context.Background(), cv, &ApplyContext{
		TemplateContext: manifest.TemplateContext{
			ClusterName:       "demo",
			KubernetesVersion: "v1.34.3",
			ImageRegistry:     "registry.example.com/openfuyao/",
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "registry.example.com/openfuyao/coredns", values["image"].(map[string]interface{})["repository"])
	assert.Equal(t, "inline", values["image"].(map[string]interface{})["tag"])
	assert.Equal(t, "None", values["service"].(map[string]interface{})["clusterIP"])
	assert.Equal(t, "demo", values["env"])
}

func TestLoadValuesRendersNestedListsAndFileOverride(t *testing.T) {
	store := fakeComponentFileStore{files: map[string][]byte{
		"coredns@v1.0.0/base.yaml": []byte(`apiVersion: v1
kind: ConfigMap
data:
  values.yaml: |
    tolerations:
    - key: old
      value: old
    servers:
    - host: {{ .ClusterName }}.old
`),
		"coredns@v1.0.0/override.yaml": []byte(`apiVersion: v1
kind: ConfigMap
data:
  values.yaml: |
    tolerations:
    - key: dedicated
      value: {{ .Namespace }}
    servers:
    - host: {{ .ClusterName }}.svc
      ports:
      - 53
      - 9153
`),
	}}
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store:       store,
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})
	cv := helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{
		ValuesFiles: []string{"base.yaml", "override.yaml"},
		Values: map[string]interface{}{
			"extraArgs": []interface{}{
				"--cluster={{ .ClusterName }}",
				map[string]interface{}{"namespace": "{{ .Namespace }}"},
			},
		},
	})

	values, err := inst.loadValues(context.Background(), cv, &ApplyContext{
		TemplateContext: manifest.TemplateContext{ClusterName: "demo", Namespace: "kube-system"},
	})
	require.NoError(t, err)

	tolerations := values["tolerations"].([]interface{})
	require.Len(t, tolerations, 1)
	assert.Equal(t, "dedicated", tolerations[0].(map[string]interface{})["key"])
	assert.Equal(t, "kube-system", tolerations[0].(map[string]interface{})["value"])

	servers := values["servers"].([]interface{})
	require.Len(t, servers, 1)
	assert.Equal(t, "demo.svc", servers[0].(map[string]interface{})["host"])
	assert.Equal(t, []interface{}{float64(53), float64(9153)}, servers[0].(map[string]interface{})["ports"])

	extraArgs := values["extraArgs"].([]interface{})
	require.Len(t, extraArgs, 2)
	assert.Equal(t, "--cluster=demo", extraArgs[0])
	assert.Equal(t, "kube-system", extraArgs[1].(map[string]interface{})["namespace"])
}
func TestLoadValueFileSkipsMissingOrEmptyValues(t *testing.T) {
	inst := NewHelmInstaller(HelmInstallerConfig{
		Store: fakeComponentFileStore{files: map[string][]byte{
			"coredns@v1.0.0/empty.yaml": []byte(`apiVersion: v1
kind: ConfigMap
data:
  values.yaml: ""
`),
		}},
		LocalClient: helmTestClient(),
		BKECluster:  helmTestCluster(),
	})
	cv := helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{})

	got, err := inst.loadValueFile(context.Background(), cv, "missing.yaml", manifest.TemplateContext{})
	require.NoError(t, err)
	assert.Nil(t, got)

	got, err = inst.loadValueFile(context.Background(), cv, "empty.yaml", manifest.TemplateContext{})
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestLoadValueFileReturnsParseAndRenderErrors(t *testing.T) {
	tests := []struct {
		name string
		file string
		data []byte
		want string
	}{
		{
			name: "invalid configmap yaml",
			file: "invalid-cm.yaml",
			data: []byte("apiVersion: v1\nkind: ConfigMap\ndata: ["),
			want: "parse values ConfigMap",
		},
		{
			name: "missing template key",
			file: "missing-key.yaml",
			data: []byte("apiVersion: v1\nkind: ConfigMap\ndata:\n  values.yaml: '{{ .Missing }}'\n"),
			want: "render values ConfigMap",
		},
		{
			name: "invalid values yaml",
			file: "invalid-values.yaml",
			data: []byte("apiVersion: v1\nkind: ConfigMap\ndata:\n  values.yaml: 'a: ['\n"),
			want: "parse values.yaml",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inst := NewHelmInstaller(HelmInstallerConfig{
				Store: fakeComponentFileStore{files: map[string][]byte{
					"coredns@v1.0.0/" + tt.file: tt.data,
				}},
				LocalClient: helmTestClient(),
				BKECluster:  helmTestCluster(),
			})
			_, err := inst.loadValueFile(
				context.Background(),
				helmCV("coredns", "v1.0.0", &cvv1alpha1.HelmSpec{}),
				tt.file,
				manifest.TemplateContext{},
			)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}

func TestEnsureNamespace(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	inst := NewHelmInstaller(HelmInstallerConfig{TargetClient: clientset})
	helmSpec := &cvv1alpha1.HelmSpec{
		Strategy: cvv1alpha1.HelmStrategySpec{CreateNamespace: true},
	}

	require.NoError(t, inst.ensureNamespace(context.Background(), helmSpec, "addon-system"))
	require.NoError(t, inst.ensureNamespace(context.Background(), helmSpec, "addon-system"))

	_, err := clientset.CoreV1().Namespaces().Get(context.Background(), "addon-system", metav1.GetOptions{})
	require.NoError(t, err)
}

func TestEnsureNamespaceRequiresTargetClientOnlyWhenCreateNamespace(t *testing.T) {
	inst := NewHelmInstaller(HelmInstallerConfig{})
	require.NoError(t, inst.ensureNamespace(context.Background(), &cvv1alpha1.HelmSpec{}, "ns"))

	err := inst.ensureNamespace(context.Background(), &cvv1alpha1.HelmSpec{
		Strategy: cvv1alpha1.HelmStrategySpec{CreateNamespace: true},
	}, "ns")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "target client is nil")
}

func TestMaybeHealthCheck(t *testing.T) {
	orig := healthCheckRun
	defer func() { healthCheckRun = orig }()

	calls := 0
	healthCheckRun = func(_ context.Context, clientset kubernetes.Interface, spec healthcheck.Spec) error {
		calls++
		require.NotNil(t, clientset)
		assert.True(t, spec.Enabled)
		return nil
	}

	inst := NewHelmInstaller(HelmInstallerConfig{TargetClient: fake.NewSimpleClientset()})
	require.NoError(t, inst.maybeHealthCheck(context.Background(), nil, nil))
	require.NoError(t, inst.maybeHealthCheck(context.Background(), &cvv1alpha1.HelmSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{Enabled: false},
	}, nil))
	assert.Zero(t, calls)

	require.NoError(t, inst.maybeHealthCheck(context.Background(), &cvv1alpha1.HelmSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{
			Enabled: true,
			Checks: []cvv1alpha1.HealthCheckItemSpec{{
				Type:   string(healthcheck.CheckTypeCustom),
				Custom: &cvv1alpha1.CustomCheckSpec{Command: "true"},
			}},
		},
	}, nil))
	assert.Equal(t, 1, calls)
}

func TestMaybeHealthCheckPropagatesErrors(t *testing.T) {
	orig := healthCheckRun
	defer func() { healthCheckRun = orig }()
	healthCheckRun = func(context.Context, kubernetes.Interface, healthcheck.Spec) error {
		return errors.New("not ready")
	}

	inst := NewHelmInstaller(HelmInstallerConfig{})
	err := inst.maybeHealthCheck(context.Background(), &cvv1alpha1.HelmSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{Enabled: true},
	}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "target client")

	inst.targetClient = fake.NewSimpleClientset()
	err = inst.maybeHealthCheck(context.Background(), &cvv1alpha1.HelmSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{
			Enabled: true,
			Checks: []cvv1alpha1.HealthCheckItemSpec{{
				Type:   string(healthcheck.CheckTypeCustom),
				Custom: &cvv1alpha1.CustomCheckSpec{Command: "true"},
			}},
		},
	}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not ready")
}

func containsAll(s string, parts ...string) bool {
	for _, part := range parts {
		if !strings.Contains(s, part) {
			return false
		}
	}
	return true
}
