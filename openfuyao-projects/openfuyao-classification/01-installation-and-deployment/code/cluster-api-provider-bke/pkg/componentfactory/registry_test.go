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
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestResolveInlineUpgrade(t *testing.T) {
	f, err := NewFactoryFromBundle(catalogTestBundle())
	if err != nil {
		t.Fatal(err)
	}
	ctx := phaseframe.NewReconcilePhaseCtx(t.Context())

	phase, err := ResolveInlineUpgrade(f, upgrade.InlineHandlerEtcdUpgrade, upgrade.ComponentManifestVersion, ctx)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if phase.Name() != confv1beta1.BKEClusterPhase(upgrade.InlineHandlerEtcdUpgrade) {
		t.Fatalf("unexpected phase: %s", phase.Name())
	}
}

func catalogTestBundle() *releasemanifest.Bundle {
	return &releasemanifest.Bundle{
		Release: cvv1alpha1.ReleaseImage{
			Spec: cvv1alpha1.ReleaseImageSpec{
				Upgrade: &cvv1alpha1.ReleaseImageUpgradeSpec{
					Components: []cvv1alpha1.ReleaseImageUpgradeComponent{
						{
							Name:    upgrade.ComponentEtcd,
							Version: upgrade.ComponentManifestVersion,
							Inline: &cvv1alpha1.ReleaseImageUpgradeInline{
								Handler: upgrade.InlineHandlerEtcdUpgrade,
								Version: upgrade.ComponentManifestVersion,
							},
						},
					},
				},
			},
		},
		Components: map[string]cvv1alpha1.ComponentVersion{},
	}
}
