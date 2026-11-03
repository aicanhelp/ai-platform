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

package v1

import (
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	applyv1 "k8s.io/client-go/applyconfigurations/meta/v1"

	apiscolocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
)

func TestColocationWorkloadMatchRuleConstructor(t *testing.T) {
	b := ColocationWorkloadMatchRule("test-name", "test-ns")
	assert.NotNil(t, b)
	assert.Equal(t, "test-name", *b.Name)
	assert.Equal(t, "test-ns", *b.Namespace)
	assert.Equal(t, "ColocationWorkloadMatchRule", *b.Kind)
	assert.Equal(t, "colocation.openfuyao.com/v1", *b.APIVersion)
}

func TestColocationWorkloadMatchRuleApply_WithKind(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	result := b.WithKind("TestKind")
	assert.Equal(t, "TestKind", *b.Kind)
	assert.Equal(t, b, result)
}

func TestColocationWorkloadMatchRuleApply_WithAPIVersion(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	result := b.WithAPIVersion("v1")
	assert.Equal(t, "v1", *b.APIVersion)
	assert.Equal(t, b, result)
}

func TestColocationWorkloadMatchRuleApply_WithName(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithName("myname")
	assert.Equal(t, "myname", *b.Name)
}

func TestColocationWorkloadMatchRuleApply_WithGenerateName(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithGenerateName("gen-")
	assert.Equal(t, "gen-", *b.GenerateName)
}

func TestColocationWorkloadMatchRuleApply_WithNamespace(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithNamespace("ns1")
	assert.Equal(t, "ns1", *b.Namespace)
}

func TestColocationWorkloadMatchRuleApply_WithUID(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	uid := types.UID("abc")
	b.WithUID(uid)
	assert.Equal(t, uid, *b.UID)
}

func TestColocationWorkloadMatchRuleApply_WithResourceVersion(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithResourceVersion("rv1")
	assert.Equal(t, "rv1", *b.ResourceVersion)
}

func TestColocationWorkloadMatchRuleApply_WithGeneration(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithGeneration(5)
	assert.Equal(t, int64(5), *b.Generation)
}

func TestColocationWorkloadMatchRuleApply_WithCreationTimestamp(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	ts := metav1.Time{}
	b.WithCreationTimestamp(ts)
	assert.Equal(t, ts, *b.CreationTimestamp)
}

func TestColocationWorkloadMatchRuleApply_WithDeletionTimestamp(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	ts := metav1.Time{}
	b.WithDeletionTimestamp(ts)
	assert.Equal(t, ts, *b.DeletionTimestamp)
}

func TestColocationWorkloadMatchRuleApply_WithDeletionGracePeriodSeconds(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithDeletionGracePeriodSeconds(10)
	assert.Equal(t, int64(10), *b.DeletionGracePeriodSeconds)
}

func TestColocationWorkloadMatchRuleApply_WithLabels(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithLabels(map[string]string{"key": "value"})
	assert.Equal(t, "value", b.Labels["key"])
}

func TestColocationWorkloadMatchRuleApply_WithAnnotations(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithAnnotations(map[string]string{"ann": "val"})
	assert.Equal(t, "val", b.Annotations["ann"])
}

func TestColocationWorkloadMatchRuleApply_WithOwnerReferences(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	or := &applyv1.OwnerReferenceApplyConfiguration{}
	b.WithOwnerReferences(or)
	assert.Equal(t, 1, len(b.OwnerReferences))
}

func TestColocationWorkloadMatchRuleApply_WithFinalizers(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	b.WithFinalizers("f1", "f2")
	assert.Equal(t, 2, len(b.Finalizers))
}

func TestColocationWorkloadMatchRuleApply_WithSpec(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	spec := &ColocationWorkloadMatchRuleSpecApplyConfiguration{}
	b.WithSpec(spec)
	assert.Equal(t, spec, b.Spec)
}

func TestColocationWorkloadMatchRuleApply_WithStatus(t *testing.T) {
	b := &ColocationWorkloadMatchRuleApplyConfiguration{}
	status := apiscolocationv1.ColocationWorkloadMatchRuleStatus{}
	b.WithStatus(status)
	assert.NotNil(t, b.Status)
}

func TestColocationWorkloadMatchRuleSpecApplyConstructor(t *testing.T) {
	s := ColocationWorkloadMatchRuleSpec()
	assert.NotNil(t, s)
}

func TestColocationWorkloadMatchRuleSpecApply_WithSelectors(t *testing.T) {
	s := ColocationWorkloadMatchRuleSpec()
	sel := Selector().WithName("test")
	s.WithSelectors(sel)
	assert.Equal(t, 1, len(s.Selectors))
}

func TestColocationWorkloadMatchRuleSpecApply_WithColocation(t *testing.T) {
	s := ColocationWorkloadMatchRuleSpec()
	ct := apiscolocationv1.ColocationOnline
	s.WithColocation(ct)
	assert.Equal(t, ct, *s.Colocation)
}

func TestSelectorApplyConstructor(t *testing.T) {
	s := Selector()
	assert.NotNil(t, s)
}

func TestSelectorApply_WithNamespace(t *testing.T) {
	s := Selector().WithNamespace("ns")
	assert.Equal(t, "ns", *s.Namespace)
}

func TestSelectorApply_WithWorkloadTypes(t *testing.T) {
	s := Selector().WithWorkloadTypes("Deployment", "StatefulSet")
	assert.Equal(t, 2, len(s.WorkloadTypes))
}

func TestSelectorApply_WithName(t *testing.T) {
	s := Selector().WithName("app")
	assert.Equal(t, "app", *s.Name)
}

func TestSelectorApply_WitSelector(t *testing.T) {
	ls := &applyv1.LabelSelectorApplyConfiguration{}
	s := Selector().WitSelector(ls)
	assert.Equal(t, ls, s.Selector)
}
