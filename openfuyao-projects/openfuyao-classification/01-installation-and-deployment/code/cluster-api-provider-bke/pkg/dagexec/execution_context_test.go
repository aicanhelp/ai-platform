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

package dagexec

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestNewExecutionContext_FillsTemplateFromCluster(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "bke-system"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					KubernetesVersion: "v1.28.0",
					OpenFuyaoVersion:  "v26.06",
				},
			},
		},
	}
	old := cluster.DeepCopy()
	vc := upgrade.NewVersionContext()
	vc.SetTarget("coredns", "v1.0.0")

	execCtx := NewExecutionContext(NewExecutionContextOptions{
		OldCluster:     old,
		Cluster:        cluster,
		VersionContext: vc,
	})
	if execCtx == nil {
		t.Fatal("expected non-nil ExecutionContext")
	}
	if execCtx.Cluster != cluster || execCtx.OldCluster != old {
		t.Fatalf("cluster pointers not preserved")
	}
	if execCtx.VersionContext != vc {
		t.Fatal("VersionContext not preserved")
	}
	tmpl := execCtx.TemplateContext
	if tmpl.ClusterName != "demo" || tmpl.Namespace != "bke-system" {
		t.Fatalf("unexpected name/ns: %+v", tmpl)
	}
	if tmpl.KubernetesVersion != "v1.28.0" || tmpl.OpenFuyaoVersion != "v26.06" {
		t.Fatalf("unexpected versions: %+v", tmpl)
	}
}

func TestNewExecutionContext_NilClusterConfigNoPanic(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns"},
	}
	execCtx := NewExecutionContext(NewExecutionContextOptions{Cluster: cluster})
	if execCtx.TemplateContext.ClusterName != "demo" {
		t.Fatalf("expected cluster name, got %q", execCtx.TemplateContext.ClusterName)
	}
	if execCtx.TemplateContext.KubernetesVersion != "" || execCtx.TemplateContext.OpenFuyaoVersion != "" {
		t.Fatalf("expected empty versions with nil ClusterConfig, got %+v", execCtx.TemplateContext)
	}
}

func TestNewExecutionContext_NilClusterNoPanic(t *testing.T) {
	execCtx := NewExecutionContext(NewExecutionContextOptions{})
	if execCtx == nil {
		t.Fatal("expected non-nil ExecutionContext")
	}
	if execCtx.TemplateContext.ClusterName != "" {
		t.Fatalf("expected empty template, got %+v", execCtx.TemplateContext)
	}
}
