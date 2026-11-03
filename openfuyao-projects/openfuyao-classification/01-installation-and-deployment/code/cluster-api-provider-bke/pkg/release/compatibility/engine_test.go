/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package compatibility

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	apiv1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
)

func TestCompatibilityEngineAcceptsValidKubernetesEtcdPair(t *testing.T) {
	bundle := compatibilityBundle("v1.29.1-of.1", "v3.5.21-of.1")

	report := NewEngine().Check(context.Background(), bundle)

	require.True(t, report.Allowed, report.Detail())
	assert.Len(t, report.Conflicts, 0)
}

func TestCompatibilityEngineRejectsInvalidKubernetesEtcdPair(t *testing.T) {
	bundle := compatibilityBundle("v1.29.1-of.1", "v3.4.0")

	report := NewEngine().Check(context.Background(), bundle)

	require.False(t, report.Allowed)
	assert.Contains(t, report.Detail(), "etcd")
	assert.Contains(t, report.Detail(), "kubernetes")
	assert.Contains(t, report.Detail(), ">=3.5.10 <3.6.0")
}

func TestFlattenExpandsSubComponents(t *testing.T) {
	bundle := compatibilityBundle("v1.29.1-of.1", "v3.5.21-of.1")
	bundle.Release.Spec.Upgrade.Components = []apiv1.ReleaseImageUpgradeComponent{{
		Name: "openfuyao-core", Version: "v26.03",
	}}
	bundle.Components[manifest.ComponentKey("openfuyao-core", "v26.03")] = component("openfuyao-core", "v26.03", apiv1.ComponentTypeYAML, nil,
		apiv1.SubComponent{Name: "kubernetes", Version: "v1.29.1-of.1"},
		apiv1.SubComponent{Name: "etcd", Version: "v3.5.21-of.1"},
	)

	components, err := Flatten(bundle)

	require.NoError(t, err)
	require.Len(t, components, 2)
	assert.Equal(t, "openfuyao-core", components[0].Parent)
	assert.Equal(t, "openfuyao-core", components[1].Parent)
}

func TestCompatibilityEngineReportsMissingAndInvalidRules(t *testing.T) {
	bundle := compatibilityBundle("v1.29.1-of.1", "v3.5.21-of.1")
	k8s := bundle.Components[manifest.ComponentKey("kubernetes", "v1.29.1-of.1")]
	k8s.Spec.Compatibility.Constraints = []apiv1.Constraint{
		{Component: "containerd", Rule: ">=1.7.0"},
		{Component: "etcd", Rule: "not-a-rule"},
	}
	bundle.Components[manifest.ComponentKey("kubernetes", "v1.29.1-of.1")] = k8s

	report := NewEngine().Check(context.Background(), bundle)

	require.False(t, report.Allowed)
	assert.Contains(t, report.Detail(), "containerd")
	assert.Contains(t, report.Detail(), "invalid compatibility rule")
}

func TestCompatibilityEngineReportsFlattenErrors(t *testing.T) {
	report := NewEngine().Check(context.Background(), nil)

	require.False(t, report.Allowed)
	assert.Contains(t, report.Detail(), "release bundle is nil")
}

func TestFlattenReadsInstallComponentsAndMissingComponent(t *testing.T) {
	bundle := compatibilityBundle("v1.29.1-of.1", "v3.5.21-of.1")
	bundle.Release.Spec.Upgrade = nil
	bundle.Release.Spec.Install = &apiv1.ReleaseImageInstallSpec{Components: []apiv1.ReleaseImageInstallComponent{{
		Name: "kubernetes", Version: "v1.29.1-of.1",
	}}}

	components, err := Flatten(bundle)
	require.NoError(t, err)
	require.Len(t, components, 1)
	assert.Equal(t, "kubernetes", components[0].Name)

	bundle.Release.Spec.Install.Components[0].Name = "missing"
	_, err = Flatten(bundle)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "missing")
}

func compatibilityBundle(k8sVersion, etcdVersion string) *manifest.Bundle {
	ri := apiv1.ReleaseImage{}
	ri.Spec.Version = "v26.03"
	ri.Spec.Upgrade = &apiv1.ReleaseImageUpgradeSpec{Components: []apiv1.ReleaseImageUpgradeComponent{
		{Name: "kubernetes", Version: k8sVersion},
		{Name: "etcd", Version: etcdVersion},
	}}
	components := map[string]apiv1.ComponentVersion{}
	k8s := component("kubernetes", k8sVersion, apiv1.ComponentTypeBinary, []apiv1.Constraint{
		{Component: "etcd", Rule: ">=3.5.10 <3.6.0"},
	})
	etcd := component("etcd", etcdVersion, apiv1.ComponentTypeBinary, nil)
	components[manifest.ComponentKey(k8s.Spec.Name, k8s.Spec.Version)] = k8s
	components[manifest.ComponentKey(etcd.Spec.Name, etcd.Spec.Version)] = etcd
	return &manifest.Bundle{Release: ri, Components: components}
}

func component(name, version string, componentType apiv1.ComponentType,
	constraints []apiv1.Constraint, subComponents ...apiv1.SubComponent) apiv1.ComponentVersion {
	cv := apiv1.ComponentVersion{}
	cv.Spec.Name = name
	cv.Spec.Version = version
	cv.Spec.Type = componentType
	cv.Spec.SubComponents = subComponents
	cv.Spec.Compatibility = apiv1.CompatibilitySpec{Constraints: constraints}
	return cv
}
