/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package clusterutil

import (
	"testing"

	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

type mockClusterObject struct {
	client.Object
	annotations map[string]string
}

func (m *mockClusterObject) GetAnnotations() map[string]string {
	return m.annotations
}

func (m *mockClusterObject) SetAnnotations(annotations map[string]string) {
	m.annotations = annotations
}

func TestFullyControlled_BKECluster(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBKE,
		},
	}
	if !FullyControlled(obj) {
		t.Error("expected true, got false")
	}
}

func TestFullyControlled_OtherCluster(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueOther,
		},
	}
	if FullyControlled(obj) {
		t.Error("expected false, got true")
	}
}

func TestFullyControlled_BocloudCluster_FullManagement(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey:                common.BKEClusterFromAnnotationValueBocloud,
			annotation.KONKFullManagementClusterAnnotationKey: "true",
		},
	}
	if !FullyControlled(obj) {
		t.Error("expected true, got false")
	}
}

func TestFullyControlled_BocloudCluster_NotFullManagement(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey:                common.BKEClusterFromAnnotationValueBocloud,
			annotation.KONKFullManagementClusterAnnotationKey: "false",
		},
	}
	if FullyControlled(obj) {
		t.Error("expected false, got true")
	}
}

func TestIsBKECluster_BKE(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBKE,
		},
	}
	if !IsBKECluster(obj) {
		t.Error("expected true, got false")
	}
}

func TestIsBKECluster_Empty(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: "",
		},
	}
	if !IsBKECluster(obj) {
		t.Error("expected true, got false")
	}
}

func TestIsBKECluster_NoAnnotation(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{},
	}
	if !IsBKECluster(obj) {
		t.Error("expected true, got false")
	}
}

func TestIsBKECluster_Bocloud(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBocloud,
		},
	}
	if IsBKECluster(obj) {
		t.Error("expected false, got true")
	}
}

func TestIsBocloudCluster(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBocloud,
		},
	}
	if !IsBocloudCluster(obj) {
		t.Error("expected true, got false")
	}
}

func TestIsBocloudCluster_NotBocloud(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBKE,
		},
	}
	if IsBocloudCluster(obj) {
		t.Error("expected false, got true")
	}
}

func TestIsOtherCluster(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueOther,
		},
	}
	if !IsOtherCluster(obj) {
		t.Error("expected true, got false")
	}
}

func TestIsOtherCluster_NotOther(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBKE,
		},
	}
	if IsOtherCluster(obj) {
		t.Error("expected false, got true")
	}
}

func TestGetClusterType_BKE(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBKE,
		},
	}
	if GetClusterType(obj) != common.BKEClusterFromAnnotationValueBKE {
		t.Error("expected bke, got " + GetClusterType(obj))
	}
}

func TestGetClusterType_Empty(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			common.BKEClusterFromAnnotationKey: "",
		},
	}
	if GetClusterType(obj) != "bke" {
		t.Error("expected bke, got " + GetClusterType(obj))
	}
}

func TestGetClusterType_NoAnnotation(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{},
	}
	if GetClusterType(obj) != "bke" {
		t.Error("expected bke, got " + GetClusterType(obj))
	}
}

func TestClusterInfoHasCollected_Base(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "base",
		},
	}
	if !ClusterInfoHasCollected(obj) {
		t.Error("expected true, got false")
	}
}

func TestClusterInfoHasCollected_Agent(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "agent",
		},
	}
	if !ClusterInfoHasCollected(obj) {
		t.Error("expected true, got false")
	}
}

func TestClusterInfoHasCollected_Both(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "base,agent",
		},
	}
	if !ClusterInfoHasCollected(obj) {
		t.Error("expected true, got false")
	}
}

func TestClusterInfoHasCollected_None(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{},
	}
	if ClusterInfoHasCollected(obj) {
		t.Error("expected false, got true")
	}
}

func TestClusterBaseInfoHasCollected(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "base",
		},
	}
	if !ClusterBaseInfoHasCollected(obj) {
		t.Error("expected true, got false")
	}
}

func TestClusterBaseInfoHasCollected_AgentOnly(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "agent",
		},
	}
	if ClusterBaseInfoHasCollected(obj) {
		t.Error("expected false, got true")
	}
}

func TestClusterAgentInfoHasCollected(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "agent",
		},
	}
	if !ClusterAgentInfoHasCollected(obj) {
		t.Error("expected true, got false")
	}
}

func TestClusterAgentInfoHasCollected_BaseOnly(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "base",
		},
	}
	if ClusterAgentInfoHasCollected(obj) {
		t.Error("expected false, got true")
	}
}

func TestMarkClusterBaseInfoCollected_New(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{},
	}
	MarkClusterBaseInfoCollected(obj)
	if obj.annotations[annotation.ClusterCollectdAnnotationKey] != "base" {
		t.Error("expected base, got " + obj.annotations[annotation.ClusterCollectdAnnotationKey])
	}
}

func TestMarkClusterBaseInfoCollected_Existing(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "agent",
		},
	}
	MarkClusterBaseInfoCollected(obj)
	if obj.annotations[annotation.ClusterCollectdAnnotationKey] != "agent,base" {
		t.Error("expected agent,base, got " + obj.annotations[annotation.ClusterCollectdAnnotationKey])
	}
}

func TestMarkClusterBaseInfoCollected_AlreadyMarked(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "base",
		},
	}
	MarkClusterBaseInfoCollected(obj)
	if obj.annotations[annotation.ClusterCollectdAnnotationKey] != "base" {
		t.Error("expected base, got " + obj.annotations[annotation.ClusterCollectdAnnotationKey])
	}
}

func TestMarkClusterAgentInfoCollected_New(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{},
	}
	MarkClusterAgentInfoCollected(obj)
	if obj.annotations[annotation.ClusterCollectdAnnotationKey] != "agent" {
		t.Error("expected agent, got " + obj.annotations[annotation.ClusterCollectdAnnotationKey])
	}
}

func TestMarkClusterAgentInfoCollected_Existing(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{
			annotation.ClusterCollectdAnnotationKey: "base",
		},
	}
	MarkClusterAgentInfoCollected(obj)
	if obj.annotations[annotation.ClusterCollectdAnnotationKey] != "base,agent" {
		t.Error("expected base,agent, got " + obj.annotations[annotation.ClusterCollectdAnnotationKey])
	}
}

func TestMarkClusterFullyControlled(t *testing.T) {
	obj := &mockClusterObject{
		annotations: map[string]string{},
	}
	MarkClusterFullyControlled(obj)
	if obj.annotations[annotation.KONKFullManagementClusterAnnotationKey] != "true" {
		t.Error("expected true, got " + obj.annotations[annotation.KONKFullManagementClusterAnnotationKey])
	}
}
