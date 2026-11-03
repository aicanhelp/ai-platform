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
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/dagexec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

type helmMapCVStore map[string]*cvv1alpha1.ComponentVersion

func (m helmMapCVStore) GetComponentVersion(
	_ context.Context,
	name, version string,
) (*cvv1alpha1.ComponentVersion, error) {
	cv, ok := m[name+"@"+version]
	if !ok {
		return nil, errors.New("not found")
	}
	return cv, nil
}

func helmExecutorCV(name, version string, mode string) *cvv1alpha1.ComponentVersion {
	return &cvv1alpha1.ComponentVersion{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: cvv1alpha1.ComponentVersionSpec{
			Name:    name,
			Version: version,
			Type:    cvv1alpha1.ComponentTypeHelm,
			Helm: &cvv1alpha1.HelmSpec{
				Strategy: cvv1alpha1.HelmStrategySpec{Mode: mode},
			},
		},
	}
}

func TestHelmComponentExecutor_GetComponentType(t *testing.T) {
	exec := &HelmComponentExecutor{}
	assert.Equal(t, dagexec.ComponentTypeHelm, exec.GetComponentType())
}

func TestActionFromMode(t *testing.T) {
	tests := []struct {
		name    string
		mode    string
		want    Action
		wantErr bool
	}{
		{name: "auto", want: ""},
		{name: "install", mode: cvv1alpha1.HelmModeInstall, want: ActionInstall},
		{name: "upgrade", mode: cvv1alpha1.HelmModeUpgrade, want: ActionUpgrade},
		{name: "unsupported", mode: "Delete", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := actionFromMode(tt.mode)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestToApplyContext(t *testing.T) {
	assert.NotNil(t, toApplyContext(nil))

	clientset := fake.NewSimpleClientset()
	restConfig := &rest.Config{Host: "https://example.com"}
	execCtx := &dagexec.ExecutionContext{
		TemplateContext:  manifest.TemplateContext{ClusterName: "demo"},
		TargetClient:     clientset,
		TargetRestConfig: restConfig,
	}
	got := toApplyContext(execCtx)
	assert.Equal(t, "demo", got.TemplateContext.ClusterName)
	assert.Same(t, clientset, got.TargetClient)
	assert.Same(t, restConfig, got.TargetRestConfig)
}

func TestHelmComponentExecutor_SkipWhenVersionsMatch(t *testing.T) {
	calls := 0
	patches := gomonkey.ApplyMethod(reflect.TypeOf(&HelmInstaller{}), "Apply",
		func(_ *HelmInstaller, _ context.Context, _ *cvv1alpha1.ComponentVersion, _ *ApplyContext, _ Action) error {
			calls++
			return nil
		})
	defer patches.Reset()

	exec := &HelmComponentExecutor{
		Installer: &HelmInstaller{},
		CVStore: helmMapCVStore{
			"coredns@v1.0.0": helmExecutorCV("coredns", "v1.0.0", ""),
		},
	}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("coredns", "v1.0.0")
	vc.SetTarget("coredns", "v1.0.0")
	err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{
		Name: "coredns", Version: "v1.0.0",
	}, dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{VersionContext: vc}))
	require.NoError(t, err)
	assert.Zero(t, calls)
}

func TestHelmComponentExecutor_ExecuteComponentPassesForcedActionAndContext(t *testing.T) {
	var gotCV *cvv1alpha1.ComponentVersion
	var gotAction Action
	patches := gomonkey.ApplyMethod(reflect.TypeOf(&HelmInstaller{}), "Apply",
		func(_ *HelmInstaller, _ context.Context, cv *cvv1alpha1.ComponentVersion, execCtx *ApplyContext, action Action) error {
			gotCV = cv
			gotAction = action
			require.NotNil(t, execCtx)
			return nil
		})
	defer patches.Reset()

	exec := &HelmComponentExecutor{
		Installer: &HelmInstaller{},
		CVStore: helmMapCVStore{
			"coredns@v1.1.0": helmExecutorCV("coredns", "v1.1.0", cvv1alpha1.HelmModeUpgrade),
		},
	}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("coredns", "v1.0.0")
	vc.SetTarget("coredns", "v1.1.0")
	restConfig := &rest.Config{Host: "https://example.com"}
	clientset := fake.NewSimpleClientset()
	execCtx := dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{
		VersionContext:   vc,
		TargetClient:     clientset,
		TargetRestConfig: restConfig,
	})
	execCtx.TemplateContext.ClusterName = "demo"

	err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{
		Name: "coredns", Version: "v1.1.0",
	}, execCtx)
	require.NoError(t, err)
	require.NotNil(t, gotCV)
	assert.Equal(t, "coredns", gotCV.Spec.Name)
	assert.Equal(t, ActionUpgrade, gotAction)
}

func TestHelmComponentExecutor_UsesDefaultComponentVersion(t *testing.T) {
	calls := 0
	patches := gomonkey.ApplyMethod(reflect.TypeOf(&HelmInstaller{}), "Apply",
		func(_ *HelmInstaller, _ context.Context, cv *cvv1alpha1.ComponentVersion, _ *ApplyContext, action Action) error {
			calls++
			assert.Equal(t, "v1.0.0", cv.Spec.Version)
			assert.Equal(t, ActionInstall, action)
			return nil
		})
	defer patches.Reset()

	exec := &HelmComponentExecutor{
		Installer: &HelmInstaller{},
		CVStore: helmMapCVStore{
			"provider@v1.0.0": helmExecutorCV("provider", "v1.0.0", cvv1alpha1.HelmModeInstall),
		},
	}
	err := exec.ExecuteComponent(context.Background(), &topology.ComponentNode{Name: "provider"}, nil)
	require.NoError(t, err)
	assert.Equal(t, 1, calls)
}

func TestHelmComponentExecutor_ValidationAndSpecErrors(t *testing.T) {
	tests := []struct {
		name    string
		exec    *HelmComponentExecutor
		node    *topology.ComponentNode
		wantErr string
	}{
		{
			name:    "nil installer",
			exec:    &HelmComponentExecutor{CVStore: helmMapCVStore{}},
			node:    &topology.ComponentNode{Name: "x"},
			wantErr: "helm installer is nil",
		},
		{
			name:    "nil store",
			exec:    &HelmComponentExecutor{Installer: &HelmInstaller{}},
			node:    &topology.ComponentNode{Name: "x"},
			wantErr: "component version store is nil",
		},
		{
			name:    "nil node",
			exec:    &HelmComponentExecutor{Installer: &HelmInstaller{}, CVStore: helmMapCVStore{}},
			wantErr: "component node is nil",
		},
		{
			name: "missing component version",
			exec: &HelmComponentExecutor{
				Installer: &HelmInstaller{},
				CVStore:   helmMapCVStore{},
			},
			node:    &topology.ComponentNode{Name: "missing", Version: "v1.0.0"},
			wantErr: "get component version",
		},
		{
			name: "rejects non helm type",
			exec: &HelmComponentExecutor{
				Installer: &HelmInstaller{},
				CVStore: helmMapCVStore{
					"coredns@v1.0.0": {
						Spec: cvv1alpha1.ComponentVersionSpec{
							Name: "coredns", Version: "v1.0.0", Type: cvv1alpha1.ComponentTypeYAML,
						},
					},
				},
			},
			node:    &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"},
			wantErr: "is not helm",
		},
		{
			name: "requires helm spec",
			exec: &HelmComponentExecutor{
				Installer: &HelmInstaller{},
				CVStore: helmMapCVStore{
					"coredns@v1.0.0": {
						Spec: cvv1alpha1.ComponentVersionSpec{
							Name: "coredns", Version: "v1.0.0", Type: cvv1alpha1.ComponentTypeHelm,
						},
					},
				},
			},
			node:    &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"},
			wantErr: "spec.helm is required",
		},
		{
			name: "rejects unsupported mode",
			exec: &HelmComponentExecutor{
				Installer: &HelmInstaller{},
				CVStore: helmMapCVStore{
					"coredns@v1.0.0": helmExecutorCV("coredns", "v1.0.0", "Delete"),
				},
			},
			node:    &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"},
			wantErr: "unsupported helm strategy mode",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.exec.ExecuteComponent(context.Background(), tt.node, nil)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}
