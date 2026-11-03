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

package upgrade

import (
	"testing"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
)

func TestEnrichUpgradeComponentFromBundleInline(t *testing.T) {
	bundle := &releasemanifest.Bundle{
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey("etcd", "v1.0.0"): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:    "etcd",
					Version: "v1.0.0",
					Type:    cvv1alpha1.ComponentTypeInline,
					Inline:  &cvv1alpha1.InlineSpec{Handler: InlineHandlerEtcdUpgrade, Version: "v1.0.0"},
				},
			},
		},
	}
	comp := enrichUpgradeComponent(cvv1alpha1.ReleaseImageUpgradeComponent{
		Name: "etcd", Version: "v1.0.0",
	}, bundle)
	if comp.Inline == nil || comp.Inline.Handler != InlineHandlerEtcdUpgrade {
		t.Fatalf("inline not enriched: %+v", comp.Inline)
	}
}

func TestEnrichUpgradeComponentPreservesExistingInline(t *testing.T) {
	bundle := &releasemanifest.Bundle{}
	inline := &cvv1alpha1.ReleaseImageUpgradeInline{Handler: InlineHandlerEtcdUpgrade, Version: "v1.0.0"}
	comp := enrichUpgradeComponent(cvv1alpha1.ReleaseImageUpgradeComponent{
		Name:    ComponentEtcd,
		Version: "v1.0.0",
		Inline:  inline,
	}, bundle)
	if comp.Inline != inline {
		t.Fatal("expected existing inline configuration to be preserved")
	}
}

func TestEnrichUpgradeComponentYAMLHasNoInline(t *testing.T) {
	bundle := &releasemanifest.Bundle{
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey("coredns", "v1.0.0"): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:    "coredns",
					Version: "v1.0.0",
					Type:    cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	}
	comp := enrichUpgradeComponent(cvv1alpha1.ReleaseImageUpgradeComponent{
		Name: "coredns", Version: "v1.0.0",
	}, bundle)
	if comp.Inline != nil {
		t.Fatalf("yaml component should not get inline: %+v", comp.Inline)
	}
}

func TestBuildDAGFromBundle_EnrichesInlineWithoutNodeType(t *testing.T) {
	const ver = "v1.0.0"
	bundle := &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{
						{Name: "coredns", Version: ver},
						{Name: ComponentEtcd, Version: ver},
					},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey("coredns", ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "coredns", Version: ver, Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
			releasemanifest.ComponentKey(ComponentEtcd, ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: ComponentEtcd, Version: ver, Type: cvv1alpha1.ComponentTypeInline,
					Inline: &cvv1alpha1.InlineSpec{Handler: InlineHandlerEtcdUpgrade, Version: ver},
				},
			},
		},
	}

	dag, err := BuildDAGFromBundle(bundle, nil)
	if err != nil {
		t.Fatal(err)
	}
	coredns, ok := dag.GetNode("coredns")
	if !ok || coredns.Inline != nil {
		t.Fatalf("coredns should have no inline: ok=%v node=%+v", ok, coredns)
	}
	etcd, ok := dag.GetNode(ComponentEtcd)
	if !ok || etcd.Inline == nil || etcd.Inline.Handler != InlineHandlerEtcdUpgrade {
		t.Fatalf("etcd inline should be enriched: ok=%v node=%+v", ok, etcd)
	}
}

func TestUpgradeComponentsFromBundleErrors(t *testing.T) {
	if _, err := UpgradeComponentsFromBundle(nil); err == nil {
		t.Fatal("expected nil bundle error")
	}
	_, err := UpgradeComponentsFromBundle(&releasemanifest.Bundle{Release: cvv1alpha1.ReleaseImage{}})
	if err == nil {
		t.Fatal("expected empty upgrade component error")
	}
}

func TestBuildDAGFromBundle(t *testing.T) {
	const ver = "v1.0.0"
	const (
		compBase  = "test-base"
		compMid   = "test-mid"
		compLeaf  = "test-leaf"
		compExtra = "test-extra"
	)

	bundle := &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{
						{Name: compBase, Version: ver},
						{Name: compMid, Version: ver},
						{Name: compLeaf, Version: ver},
						{Name: compExtra, Version: ver},
					},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey(compBase, ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{Name: compBase, Version: ver},
			},
			releasemanifest.ComponentKey(compMid, ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:         compMid,
					Version:      ver,
					Dependencies: []cvv1alpha1.Dependency{{Name: compBase}},
				},
			},
			releasemanifest.ComponentKey(compLeaf, ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:         compLeaf,
					Version:      ver,
					Dependencies: []cvv1alpha1.Dependency{{Name: compMid}},
				},
			},
			releasemanifest.ComponentKey(compExtra, ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:         compExtra,
					Version:      ver,
					Dependencies: []cvv1alpha1.Dependency{{Name: compBase}},
				},
			},
		},
	}

	dag, err := BuildDAGFromBundle(bundle, BundleDependencyResolver(bundle))
	if err != nil {
		t.Fatal(err)
	}

	batches, err := dag.TopologicalBatches()
	if err != nil {
		t.Fatal(err)
	}

	// Linear chain: base → mid → leaf; extra parallels mid (also depends on base).
	wantBatches := [][]string{
		{compBase},
		{compMid, compExtra},
		{compLeaf},
	}
	if len(batches) != len(wantBatches) {
		t.Fatalf("batch count: got %d want %d, batches=%v", len(batches), len(wantBatches), batches)
	}
	for i, want := range wantBatches {
		if !sameStringSet(batches[i], want) {
			t.Fatalf("batch[%d]: got %v want %v (all batches=%v)", i, batches[i], want, batches)
		}
	}

	execOrder := flattenBatches(batches)
	if indexOf(execOrder, compBase) >= indexOf(execOrder, compMid) ||
		indexOf(execOrder, compMid) >= indexOf(execOrder, compLeaf) ||
		indexOf(execOrder, compBase) >= indexOf(execOrder, compExtra) ||
		indexOf(execOrder, compExtra) >= indexOf(execOrder, compLeaf) {
		t.Fatalf("dependency order violated, execOrder=%v", execOrder)
	}
}

func TestBuildDAGFromBundle_ImplicitPreUpgradeDependency(t *testing.T) {
	const ver = "v1.0.0"
	bundle := &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{
						{Name: ComponentPreUpgradeResources, Version: ver, Inline: &cvv1alpha1.ReleaseImageUpgradeInline{Handler: InlineHandlerPreUpgradeResources, Version: ver}},
						{Name: ComponentProvider, Version: ver},
						{Name: ComponentEtcd, Version: ver, Inline: &cvv1alpha1.ReleaseImageUpgradeInline{Handler: InlineHandlerEtcdUpgrade, Version: ver}},
					},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey(ComponentPreUpgradeResources, ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{Name: ComponentPreUpgradeResources, Version: ver},
			},
			releasemanifest.ComponentKey(ComponentProvider, ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{Name: ComponentProvider, Version: ver},
			},
			releasemanifest.ComponentKey(ComponentEtcd, ver): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:         ComponentEtcd,
					Version:      ver,
					Dependencies: []cvv1alpha1.Dependency{{Name: ComponentProvider}},
				},
			},
		},
	}

	dag, err := BuildDAGFromBundle(bundle, BundleDependencyResolver(bundle))
	if err != nil {
		t.Fatal(err)
	}

	batches, err := dag.TopologicalBatches()
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) < 2 {
		t.Fatalf("expected at least 2 batches, got %v", batches)
	}
	if !sameStringSet(batches[0], []string{ComponentPreUpgradeResources}) {
		t.Fatalf("expected pre-upgrade resources to run first, got %v", batches)
	}
	if indexOf(flattenBatches(batches), ComponentPreUpgradeResources) > indexOf(flattenBatches(batches), ComponentProvider) {
		t.Fatalf("expected pre-upgrade resources before provider, got %v", batches)
	}
}

func TestBundleDependencyResolverWithoutComponentVersionStillAddsImplicitPreUpgradeDependency(t *testing.T) {
	bundle := &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{{
						Name:    ComponentPreUpgradeResources,
						Version: "v1.0.0",
					}, {
						Name:    ComponentProvider,
						Version: "v1.0.0",
					}},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{},
	}

	deps, err := BundleDependencyResolver(bundle)(ComponentProvider, "v1.0.0")
	if err != nil {
		t.Fatal(err)
	}
	if len(deps) != 1 || deps[0] != ComponentPreUpgradeResources {
		t.Fatalf("unexpected deps: %v", deps)
	}
}

func sameStringSet(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	seen := make(map[string]struct{}, len(got))
	for _, s := range got {
		seen[s] = struct{}{}
	}
	for _, s := range want {
		if _, ok := seen[s]; !ok {
			return false
		}
	}
	return true
}

func flattenBatches(batches [][]string) []string {
	var out []string
	for _, b := range batches {
		out = append(out, b...)
	}
	return out
}

func indexOf(order []string, name string) int {
	for i, n := range order {
		if n == name {
			return i
		}
	}
	return -1
}
