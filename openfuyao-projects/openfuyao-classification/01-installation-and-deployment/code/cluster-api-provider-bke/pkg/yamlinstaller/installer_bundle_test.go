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
	"strings"
	"testing"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
)

func TestApply_BundleFileAndInlineResources(t *testing.T) {
	bundle := &releasemanifest.Bundle{
		Files: map[string][]byte{
			"components/coredns/v1.0.0/01-cm.yaml": []byte("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: from-file\n"),
		},
		Components: map[string]cvv1alpha1.ComponentVersion{
			releasemanifest.ComponentKey("coredns", "v1.0.0"): {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name:    "coredns",
					Version: "v1.0.0",
					Type:    cvv1alpha1.ComponentTypeYAML,
					Resources: []cvv1alpha1.ResourceSpec{{
						Manifest: "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: from-inline\n",
					}},
					YAML: &cvv1alpha1.YAMLSpec{
						ApplyStrategy: cvv1alpha1.ApplyStrategyServerSideApply,
					},
				},
			},
		},
	}
	applier := &fakeApplier{}
	inst := NewYamlInstaller(YamlInstallerConfig{
		Store:   manifest.NewBundleStore(bundle),
		Applier: applier,
	})
	cv := sampleCV("coredns", &cvv1alpha1.YAMLSpec{
		ApplyStrategy: cvv1alpha1.ApplyStrategyServerSideApply,
	})
	cv.Spec.Version = "v1.0.0"

	if err := inst.Apply(context.Background(), cv, &ApplyContext{}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if applier.applyCalls != 1 {
		t.Fatalf("expected 1 apply, got %d", applier.applyCalls)
	}
	if len(applier.lastManifests) != 2 {
		t.Fatalf("expected bundle file + inline resources, got %d manifests", len(applier.lastManifests))
	}
	joined := string(applier.lastManifests[0]) + "\n" + string(applier.lastManifests[1])
	if !strings.Contains(joined, "from-file") || !strings.Contains(joined, "from-inline") {
		t.Fatalf("expected both file and inline manifests, got %#v", applier.lastManifests)
	}
	if applier.lastStrategy != cvv1alpha1.ApplyStrategyServerSideApply {
		t.Fatalf("unexpected strategy %q", applier.lastStrategy)
	}
}
