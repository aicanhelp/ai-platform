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

package yamlinstaller

import (
	"context"
	"testing"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
)

// Light integration: Apply then Uninstall both honor prune=true with the same selector.
func TestApplyAndUninstall_PruneBothSides(t *testing.T) {
	applier := &fakePruningApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store: fakeStore{pkg: &manifest.ComponentPackage{
			Name:      "coredns",
			Manifests: [][]byte{[]byte("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: keep\n")},
		}},
		Applier: applier,
	})
	sel := map[string]string{"app.kubernetes.io/name": "coredns"}
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		Namespace:          "kube-system",
		Prune:              true,
		PruneLabelSelector: sel,
	})

	if err := inst.Apply(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if applier.applyCalls != 1 || applier.pruneCalls != 1 {
		t.Fatalf("after apply: apply=%d prune=%d", applier.applyCalls, applier.pruneCalls)
	}

	if err := inst.Uninstall(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("uninstall: %v", err)
	}
	if applier.deleteCalls != 1 || applier.pruneCalls != 2 {
		t.Fatalf("after uninstall: delete=%d prune=%d", applier.deleteCalls, applier.pruneCalls)
	}
	if applier.lastNS != "kube-system" || applier.lastSel["app.kubernetes.io/name"] != "coredns" {
		t.Fatalf("unexpected prune args ns=%q sel=%#v", applier.lastNS, applier.lastSel)
	}
}

func TestApply_HealthCheckInvalidSpec(t *testing.T) {
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   fakeStore{pkg: &manifest.ComponentPackage{Name: "coredns"}},
		Applier: &fakeApplier{},
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		HealthCheck: &cvv1alpha1.HealthCheckSpec{
			Enabled: true,
			Timeout: "not-a-duration",
		},
	})
	err := inst.Apply(context.Background(), cv, &ApplyContext{})
	if err == nil || !containsAll(err.Error(), "coredns", "health check", "timeout") {
		t.Fatalf("expected convert healthCheck timeout error, got %v", err)
	}
}
