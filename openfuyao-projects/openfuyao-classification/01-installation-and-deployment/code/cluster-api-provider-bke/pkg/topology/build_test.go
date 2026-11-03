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

package topology

import (
	"testing"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

func TestBuildUpgradeDAG_NoDefaultDependencies(t *testing.T) {
	components := fullDeclarativeComponents()

	dag, err := BuildUpgradeDAG(components, DefaultDependencyResolver())
	if err != nil {
		t.Fatal(err)
	}
	if len(dag.NodeNames()) != len(components) {
		t.Fatalf("node count=%d want %d", len(dag.NodeNames()), len(components))
	}
	batches, err := dag.TopologicalBatches()
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) != 1 {
		t.Fatalf("expected single batch without default edges, got %d batches: %v", len(batches), batches)
	}
	if len(batches[0]) != len(components) {
		t.Fatalf("batch size=%d want %d", len(batches[0]), len(components))
	}
}

func TestDefaultDependenciesForReturnsEmpty(t *testing.T) {
	if deps := DefaultDependenciesFor(defaultContainerd); len(deps) != 0 {
		t.Fatalf("expected no default deps for containerd, got %v", deps)
	}
}

func TestBuildUpgradeDAG_MissingDependency(t *testing.T) {
	components := []cvv1alpha1.ReleaseImageUpgradeComponent{
		{Name: "orphan", Version: "v1"},
	}
	_, err := BuildUpgradeDAG(components, func(_, _ string) ([]string, error) {
		return []string{"missing-parent"}, nil
	})
	if err == nil {
		t.Fatal("expected missing dependency error")
	}
}

func fullDeclarativeComponents() []cvv1alpha1.ReleaseImageUpgradeComponent {
	return []cvv1alpha1.ReleaseImageUpgradeComponent{
		{Name: defaultPreUpgradeResources, Version: "v1.0.0", Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
			Handler: "EnsurePreUpgradeResources", Version: "v1.0.0",
		}},
		{Name: defaultProvider, Version: "v1.0.0"},
		{Name: defaultBKEAgentUpgrade, Version: "v1.0.0"},
		{Name: defaultContainerd, Version: "v1.0.0", Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
			Handler: "EnsureContainerdUpgrade", Version: "v1.0.0",
		}},
		{Name: defaultEtcd, Version: "v1.0.0", Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
			Handler: "EnsureEtcdUpgrade", Version: "v1.0.0",
		}},
		{Name: defaultK8sMaster, Version: "v1.0.0", Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
			Handler: "EnsureMasterUpgrade", Version: "v1.0.0",
		}},
		{Name: defaultK8sWorker, Version: "v1.0.0", Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
			Handler: "EnsureWorkerUpgrade", Version: "v1.0.0",
		}},
		{Name: defaultKubeProxy, Version: "v1.0.0"},
		{Name: defaultCoreDNS, Version: "v1.0.0"},
	}
}
