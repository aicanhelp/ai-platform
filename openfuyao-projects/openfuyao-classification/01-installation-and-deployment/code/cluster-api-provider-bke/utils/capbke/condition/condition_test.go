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

package condition

import (
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestConditionMark(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	ConditionMark(cluster, bkev1beta1.ClusterAPIObjCondition, confv1beta1.ConditionTrue, "test-reason", "test-message")

	if len(cluster.Status.Conditions) != 1 {
		t.Errorf("expected 1 condition, got %d", len(cluster.Status.Conditions))
	}
	if cluster.Status.Conditions[0].Type != bkev1beta1.ClusterAPIObjCondition {
		t.Errorf("expected ClusterAPIObjCondition, got %s", cluster.Status.Conditions[0].Type)
	}
	if cluster.Status.Conditions[0].Status != confv1beta1.ConditionTrue {
		t.Errorf("expected ConditionTrue, got %s", cluster.Status.Conditions[0].Status)
	}
	if cluster.Status.Conditions[0].Reason != "test-reason" {
		t.Errorf("expected test-reason, got %s", cluster.Status.Conditions[0].Reason)
	}
	if cluster.Status.Conditions[0].Message != "test-message" {
		t.Errorf("expected test-message, got %s", cluster.Status.Conditions[0].Message)
	}
}

func TestAddonConditionMark_NewAddonCondition(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	AddonConditionMark(MarkOptions{
		BKECluster: cluster,
		Type:       bkev1beta1.ClusterAddonCondition,
		Status:     confv1beta1.ConditionTrue,
		Reason:     "test-reason",
		Message:    "test-message",
		AddonName:  "test-addon",
	})

	if len(cluster.Status.Conditions) != 1 {
		t.Errorf("expected 1 condition, got %d", len(cluster.Status.Conditions))
	}
	if cluster.Status.Conditions[0].AddonName != "test-addon" {
		t.Errorf("expected test-addon, got %s", cluster.Status.Conditions[0].AddonName)
	}
}

func TestAddonConditionMark_UpdateExistingAddonCondition(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	cluster.Status.Conditions = []confv1beta1.ClusterCondition{
		{
			Type:      bkev1beta1.ClusterAddonCondition,
			AddonName: "test-addon",
			Status:    confv1beta1.ConditionFalse,
		},
	}
	AddonConditionMark(MarkOptions{
		BKECluster: cluster,
		Type:       bkev1beta1.ClusterAddonCondition,
		Status:     confv1beta1.ConditionTrue,
		Reason:     "new-reason",
		Message:    "new-message",
		AddonName:  "test-addon",
	})

	if len(cluster.Status.Conditions) != 1 {
		t.Errorf("expected 1 condition, got %d", len(cluster.Status.Conditions))
	}
	if cluster.Status.Conditions[0].Status != confv1beta1.ConditionTrue {
		t.Errorf("expected ConditionTrue, got %s", cluster.Status.Conditions[0].Status)
	}
	if cluster.Status.Conditions[0].Reason != "new-reason" {
		t.Errorf("expected new-reason, got %s", cluster.Status.Conditions[0].Reason)
	}
}

func TestAddonConditionMark_UpdateExistingCondition(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	cluster.Status.Conditions = []confv1beta1.ClusterCondition{
		{
			Type:   bkev1beta1.ClusterAPIObjCondition,
			Status: confv1beta1.ConditionFalse,
		},
	}
	AddonConditionMark(MarkOptions{
		BKECluster: cluster,
		Type:       bkev1beta1.ClusterAPIObjCondition,
		Status:     confv1beta1.ConditionTrue,
		Reason:     "new-reason",
		Message:    "new-message",
	})

	if len(cluster.Status.Conditions) != 1 {
		t.Errorf("expected 1 condition, got %d", len(cluster.Status.Conditions))
	}
	if cluster.Status.Conditions[0].Status != confv1beta1.ConditionTrue {
		t.Errorf("expected ConditionTrue, got %s", cluster.Status.Conditions[0].Status)
	}
}

func TestHasCondition(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	cluster.Status.Conditions = []confv1beta1.ClusterCondition{
		{
			Type:   bkev1beta1.ClusterAPIObjCondition,
			Status: confv1beta1.ConditionTrue,
		},
	}
	condition, ok := HasCondition(bkev1beta1.ClusterAPIObjCondition, cluster)
	if !ok {
		t.Error("expected true, got false")
	}
	if condition.Status != confv1beta1.ConditionTrue {
		t.Errorf("expected ConditionTrue, got %s", condition.Status)
	}
}

func TestHasCondition_NotFound(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	_, ok := HasCondition(bkev1beta1.ClusterAPIObjCondition, cluster)
	if ok {
		t.Error("expected false, got true")
	}
}

func TestHasCondition_NilConditions(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	_, ok := HasCondition(bkev1beta1.ClusterAPIObjCondition, cluster)
	if ok {
		t.Error("expected false, got true")
	}
}

func TestHasAddonCondition(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	cluster.Status.Conditions = []confv1beta1.ClusterCondition{
		{
			Type:      bkev1beta1.ClusterAddonCondition,
			AddonName: "test-addon",
			Status:    confv1beta1.ConditionTrue,
		},
	}
	condition, ok := HasAddonCondition(cluster, "test-addon")
	if !ok {
		t.Error("expected true, got false")
	}
	if condition.AddonName != "test-addon" {
		t.Errorf("expected test-addon, got %s", condition.AddonName)
	}
}

func TestHasAddonCondition_NotFound(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	_, ok := HasAddonCondition(cluster, "test-addon")
	if ok {
		t.Error("expected false, got true")
	}
}

func TestHasAddonCondition_NilConditions(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	_, ok := HasAddonCondition(cluster, "test-addon")
	if ok {
		t.Error("expected false, got true")
	}
}

func TestHasConditionStatus(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	cluster.Status.Conditions = []confv1beta1.ClusterCondition{
		{
			Type:   bkev1beta1.ClusterAPIObjCondition,
			Status: confv1beta1.ConditionTrue,
		},
	}
	result := HasConditionStatus(bkev1beta1.ClusterAPIObjCondition, cluster, confv1beta1.ConditionTrue)
	if !result {
		t.Error("expected true, got false")
	}
}

func TestHasConditionStatus_NotMatch(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	cluster.Status.Conditions = []confv1beta1.ClusterCondition{
		{
			Type:   bkev1beta1.ClusterAPIObjCondition,
			Status: confv1beta1.ConditionFalse,
		},
	}
	result := HasConditionStatus(bkev1beta1.ClusterAPIObjCondition, cluster, confv1beta1.ConditionTrue)
	if result {
		t.Error("expected false, got true")
	}
}

func TestHasConditionStatus_ConditionNotFound(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	result := HasConditionStatus(bkev1beta1.ClusterAPIObjCondition, cluster, confv1beta1.ConditionTrue)
	if result {
		t.Error("expected false, got true")
	}
}

func TestRemoveCondition(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	cluster.Status.Conditions = []confv1beta1.ClusterCondition{
		{Type: bkev1beta1.ClusterAPIObjCondition},
		{Type: bkev1beta1.BKEAgentCondition},
	}
	RemoveCondition(bkev1beta1.ClusterAPIObjCondition, cluster)
	if len(cluster.Status.Conditions) != 1 {
		t.Errorf("expected 1 condition, got %d", len(cluster.Status.Conditions))
	}
	if cluster.Status.Conditions[0].Type != bkev1beta1.BKEAgentCondition {
		t.Errorf("expected BKEAgentCondition, got %s", cluster.Status.Conditions[0].Type)
	}
}

func TestRemoveCondition_NotFound(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	cluster.Status.Conditions = []confv1beta1.ClusterCondition{
		{Type: bkev1beta1.ClusterAPIObjCondition},
	}
	RemoveCondition(bkev1beta1.BKEAgentCondition, cluster)
	if len(cluster.Status.Conditions) != 1 {
		t.Errorf("expected 1 condition, got %d", len(cluster.Status.Conditions))
	}
}

func TestRemoveCondition_NilConditions(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	RemoveCondition(bkev1beta1.ClusterAPIObjCondition, cluster)
}
