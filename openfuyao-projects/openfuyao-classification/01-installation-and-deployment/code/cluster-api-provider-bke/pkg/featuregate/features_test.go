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

package featuregate

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
)

func TestUpgradeReady(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
			UpgradeReadyAnnotationKey: "v2.6.0",
		}},
	}
	v, ok := UpgradeReady(bc)
	if !ok || v != "v2.6.0" {
		t.Fatalf("got %q ok=%v", v, ok)
	}
	if _, ok := UpgradeReady(&bkev1beta1.BKECluster{}); ok {
		t.Fatal("expected not ready")
	}
}

func TestHelmComponentEnabled(t *testing.T) {
	prev := config.HelmComponentSupport
	defer func() { config.HelmComponentSupport = prev }()

	config.HelmComponentSupport = false
	if HelmComponentEnabled(nil) {
		t.Fatal("nil object with flag off should be disabled")
	}
	if HelmComponentEnabled(&bkev1beta1.BKECluster{}) {
		t.Fatal("no annotation with flag off should be disabled")
	}

	config.HelmComponentSupport = true
	if !HelmComponentEnabled(&bkev1beta1.BKECluster{}) {
		t.Fatal("flag on without annotation should enable")
	}

	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
			HelmComponentAnnotationKey: "false",
		}},
	}
	if HelmComponentEnabled(bc) {
		t.Fatal("annotation false must override flag true")
	}

	bc.Annotations[HelmComponentAnnotationKey] = "true"
	config.HelmComponentSupport = false
	if !HelmComponentEnabled(bc) {
		t.Fatal("annotation true must override flag false")
	}
}
