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

package componentfactory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phases"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestRegisterInlinePhasesFromBundle(t *testing.T) {
	bundle := testReleaseBundle(t)
	f, err := NewFactoryFromBundle(bundle)
	if err != nil {
		t.Fatal(err)
	}

	phase, err := f.Resolve(upgrade.InlineHandlerEtcdUpgrade, "v2.0.0", &phaseframe.PhaseContext{})
	if err != nil {
		t.Fatal(err)
	}
	if phase == nil {
		t.Fatal("expected phase instance")
	}

	_, err = f.Resolve(upgrade.InlineHandlerEtcdUpgrade, "v9.9.9", &phaseframe.PhaseContext{})
	if err == nil {
		t.Fatal("expected resolve failure for unregistered handler version")
	}
}

func TestRegisterInlinePhasesFromBundle_UnknownHandler(t *testing.T) {
	bundle := &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{{
						Name:    "custom",
						Version: "v1.0.0",
						Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
							Handler: "UnknownHandler",
							Version: "v1.0.0",
						},
					}},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{},
	}
	_, err := NewFactoryFromBundle(bundle)
	if err == nil {
		t.Fatal("expected error for unknown handler")
	}
}

func TestRegisterInlinePhasesFromBundle_PreUpgradeResourcesInjectsComponentVersion(t *testing.T) {
	bundle := &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{{
						Name:    upgrade.ComponentPreUpgradeResources,
						Version: "v1.0.0",
						Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
							Handler: upgrade.InlineHandlerPreUpgradeResources,
							Version: "v1.0.0",
						},
					}},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey(upgrade.ComponentPreUpgradeResources, "v1.0.0"): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:    upgrade.ComponentPreUpgradeResources,
					Version: "v1.0.0",
					Inline: &cvv1alpha1.InlineSpec{
						Handler: upgrade.InlineHandlerPreUpgradeResources,
						Version: "v1.0.0",
					},
					Resources: []cvv1alpha1.ResourceSpec{{
						Kind:       "ConfigMap",
						APIVersion: "v1",
						Namespace:  "kube-system",
						Name:       "bootstrap-config",
					}},
				},
			},
		},
	}

	f, err := NewFactoryFromBundle(bundle)
	if err != nil {
		t.Fatal(err)
	}

	phase, err := f.Resolve(upgrade.InlineHandlerPreUpgradeResources, "v1.0.0", phaseframe.NewReconcilePhaseCtx(context.Background()).SetBKECluster(&bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "cluster", Namespace: "default"},
	}))
	if err != nil {
		t.Fatal(err)
	}

	prePhase, ok := phase.(*phases.EnsurePreUpgradeResources)
	if !ok {
		t.Fatalf("expected EnsurePreUpgradeResources, got %T", phase)
	}
	if prePhase.GetComponentVersion() == nil || len(prePhase.GetComponentVersion().Spec.Resources) != 1 {
		t.Fatal("expected injected ComponentVersion resources on pre-upgrade phase")
	}
}

func TestRegisterInlinePhasesFromBundle_PreUpgradeResourcesRequiresComponentVersion(t *testing.T) {
	bundle := &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{{
						Name:    upgrade.ComponentPreUpgradeResources,
						Version: "v1.0.0",
						Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
							Handler: upgrade.InlineHandlerPreUpgradeResources,
							Version: "v1.0.0",
						},
					}},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{},
	}

	_, err := NewFactoryFromBundle(bundle)
	if err == nil {
		t.Fatal("expected error when pre-upgrade resources component version is missing")
	}
}

func TestRegisterInlinePhasesFromBundle_NilFactory(t *testing.T) {
	err := RegisterInlinePhasesFromBundle(nil, testReleaseBundle(t))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "component factory is nil")
}

func TestRegisterInlinePhasesFromBundle_DeduplicatesAndDefaultsInlineVersion(t *testing.T) {
	bundle := &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{
						{
							Name:    upgrade.ComponentEtcd,
							Version: "v1.2.0",
							Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
								Handler: upgrade.InlineHandlerEtcdUpgrade,
							},
						},
						{
							Name:    upgrade.ComponentEtcd,
							Version: "v1.2.0",
							Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
								Handler: upgrade.InlineHandlerEtcdUpgrade,
							},
						},
					},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey(upgrade.ComponentEtcd, "v1.2.0"): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:    upgrade.ComponentEtcd,
					Version: "v1.2.0",
					Inline:  &cvv1alpha1.InlineSpec{Handler: upgrade.InlineHandlerEtcdUpgrade},
				},
			},
		},
	}

	f := NewComponentFactory()
	require.NoError(t, RegisterInlinePhasesFromBundle(f, bundle))

	phase, err := f.Resolve(upgrade.InlineHandlerEtcdUpgrade, upgrade.ComponentManifestVersion, &phaseframe.PhaseContext{})
	require.NoError(t, err)
	assert.NotNil(t, phase)
}

func TestRegisterInlineHandlerAndResolveInlineUpgradeErrors(t *testing.T) {
	f := NewComponentFactory()
	require.NoError(t, registerInlineHandler(f, upgrade.InlineHandlerEtcdUpgrade, upgrade.ComponentManifestVersion))

	phase, err := ResolveInlineUpgrade(f, upgrade.InlineHandlerEtcdUpgrade, upgrade.ComponentManifestVersion, phaseframe.NewReconcilePhaseCtx(context.Background()).SetBKECluster(&bkev1beta1.BKECluster{}))
	require.NoError(t, err)
	assert.NotNil(t, phase)

	_, err = ResolveInlineUpgrade(nil, upgrade.InlineHandlerEtcdUpgrade, upgrade.ComponentManifestVersion, &phaseframe.PhaseContext{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "component factory is nil")
}

func TestRegisterInlineComponentErrors(t *testing.T) {
	f := NewComponentFactory()

	err := registerInlineComponent(f, upgrade.InlineHandlerPreUpgradeResources, upgrade.ComponentManifestVersion, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "component version is required")

	err = registerInlineComponent(f, upgrade.InlineHandlerAgentUpgrade, upgrade.ComponentManifestVersion, nil)
	require.NoError(t, err)

	err = registerInlineComponent(f, "UnknownHandler", upgrade.ComponentManifestVersion, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown inline handler")
}

func TestRegisterInlineComponentSuccessPaths(t *testing.T) {
	f := NewComponentFactory()
	componentVersion := &cvv1alpha1.ComponentVersion{
		Spec: cvv1alpha1.ComponentVersionSpec{
			Name:    upgrade.ComponentPreUpgradeResources,
			Version: "v1.0.0",
			Resources: []cvv1alpha1.ResourceSpec{{
				Kind:      "ConfigMap",
				Name:      "pre-upgrade-config",
				Namespace: "kube-system",
			}},
		},
	}

	require.NoError(t, registerInlineComponent(f, upgrade.InlineHandlerPreUpgradeResources, upgrade.ComponentManifestVersion, componentVersion))
	require.NoError(t, registerInlineHandler(f, upgrade.InlineHandlerAgentUpgrade, upgrade.ComponentManifestVersion))

	phase, err := f.Resolve(upgrade.InlineHandlerPreUpgradeResources, upgrade.ComponentManifestVersion, phaseframe.NewReconcilePhaseCtx(context.Background()).SetBKECluster(&bkev1beta1.BKECluster{}))
	require.NoError(t, err)
	prePhase, ok := phase.(*phases.EnsurePreUpgradeResources)
	require.True(t, ok)
	require.NotNil(t, prePhase.GetComponentVersion())
	assert.Equal(t, "v1.0.0", prePhase.GetComponentVersion().Spec.Version)

	agentPhase, err := f.Resolve(upgrade.InlineHandlerAgentUpgrade, upgrade.ComponentManifestVersion, &phaseframe.PhaseContext{})
	require.NoError(t, err)
	assert.NotNil(t, agentPhase)
}

func TestRegisterInlineComponentSuccessPaths_AllInlineHandlers(t *testing.T) {
	f := NewComponentFactory()
	handlers := []string{
		upgrade.InlineHandlerEtcdUpgrade,
		upgrade.InlineHandlerMasterUpgrade,
		upgrade.InlineHandlerWorkerUpgrade,
		upgrade.InlineHandlerContainerdUpgrade,
	}
	ctx := phaseframe.NewReconcilePhaseCtx(context.Background()).SetBKECluster(&bkev1beta1.BKECluster{})

	for _, handler := range handlers {
		require.NoError(t, registerInlineHandler(f, handler, upgrade.ComponentManifestVersion), handler)
		phase, err := ResolveInlineUpgrade(f, handler, upgrade.ComponentManifestVersion, ctx)
		require.NoError(t, err, handler)
		assert.NotNil(t, phase, handler)
	}
}

func TestResolveInlineUpgrade_ResolveError(t *testing.T) {
	f := NewComponentFactory()
	_, err := ResolveInlineUpgrade(f, upgrade.InlineHandlerEtcdUpgrade, upgrade.ComponentManifestVersion, &phaseframe.PhaseContext{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func testReleaseBundle(t *testing.T) *releasemanifest.Bundle {
	t.Helper()
	return &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{
						{Name: upgrade.ComponentEtcd, Version: "v1.2.0"},
						{Name: upgrade.ComponentContainerd, Version: "v1.7.0"},
					},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey(upgrade.ComponentEtcd, "v1.2.0"): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:    upgrade.ComponentEtcd,
					Version: "v1.2.0",
					Inline:  &cvv1alpha1.InlineSpec{Handler: upgrade.InlineHandlerEtcdUpgrade, Version: "v2.0.0"},
				},
			},
			releasemanifest.ComponentKey(upgrade.ComponentContainerd, "v1.7.0"): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:    upgrade.ComponentContainerd,
					Version: "v1.7.0",
					Inline:  &cvv1alpha1.InlineSpec{Handler: upgrade.InlineHandlerContainerdUpgrade, Version: "v1.0.0"},
				},
			},
		},
	}
}
