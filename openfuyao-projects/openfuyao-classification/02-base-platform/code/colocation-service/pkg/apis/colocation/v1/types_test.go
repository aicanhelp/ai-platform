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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestResource(t *testing.T) {
	gr := Resource("colocationworkloadmatchrules")
	expected := schema.GroupResource{Group: "colocation.openfuyao.com", Resource: "colocationworkloadmatchrules"}
	assert.Equal(t, expected, gr)
}

func TestAddKnownTypes(t *testing.T) {
	scheme := runtime.NewScheme()
	err := addKnownTypes(scheme)
	assert.NoError(t, err)

	gvk := SchemeGroupVersion.WithKind("ColocationWorkloadMatchRule")
	obj, err := scheme.New(gvk)
	assert.NoError(t, err)
	_, ok := obj.(*ColocationWorkloadMatchRule)
	assert.True(t, ok)

	gvkList := SchemeGroupVersion.WithKind("ColocationWorkloadMatchRuleList")
	listObj, err := scheme.New(gvkList)
	assert.NoError(t, err)
	_, ok = listObj.(*ColocationWorkloadMatchRuleList)
	assert.True(t, ok)
}

func TestColocationWorkloadMatchRuleDeepCopy(t *testing.T) {
	original := &ColocationWorkloadMatchRule{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ColocationWorkloadMatchRule",
			APIVersion: "colocation.openfuyao.com/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-rule",
			Namespace: "default",
		},
		Spec: ColocationWorkloadMatchRuleSpec{
			Colocation: ColocationOnline,
			Selectors: []Selector{
				{
					Namespace:    "default",
					WorkloadTypes: []string{"Deployment"},
					Name:         "test-app",
				},
			},
		},
	}

	copied := original.DeepCopy()
	assert.True(t, reflect.DeepEqual(original, copied))

	original.Name = "modified"
	assert.NotEqual(t, original.Name, copied.Name)
}

func TestColocationWorkloadMatchRuleDeepCopy_Nil(t *testing.T) {
	var nilRule *ColocationWorkloadMatchRule
	assert.Nil(t, nilRule.DeepCopy())
}

func TestColocationWorkloadMatchRuleDeepCopyObject(t *testing.T) {
	original := &ColocationWorkloadMatchRule{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
	}
	obj := original.DeepCopyObject()
	copied, ok := obj.(*ColocationWorkloadMatchRule)
	assert.True(t, ok)
	assert.Equal(t, "test", copied.Name)
}

func TestColocationWorkloadMatchRuleDeepCopyObject_Nil(t *testing.T) {
	var nilRule *ColocationWorkloadMatchRule
	assert.Nil(t, nilRule.DeepCopyObject())
}

func TestColocationWorkloadMatchRuleListDeepCopy(t *testing.T) {
	original := &ColocationWorkloadMatchRuleList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ColocationWorkloadMatchRuleList",
			APIVersion: "colocation.openfuyao.com/v1",
		},
		Items: []ColocationWorkloadMatchRule{
			{ObjectMeta: metav1.ObjectMeta{Name: "item1"}},
		},
	}

	copied := original.DeepCopy()
	assert.True(t, reflect.DeepEqual(original, copied))

	original.Items[0].Name = "modified"
	assert.NotEqual(t, original.Items[0].Name, copied.Items[0].Name)
}

func TestColocationWorkloadMatchRuleListDeepCopy_Nil(t *testing.T) {
	var nilList *ColocationWorkloadMatchRuleList
	assert.Nil(t, nilList.DeepCopy())
}

func TestColocationWorkloadMatchRuleListDeepCopyObject(t *testing.T) {
	original := &ColocationWorkloadMatchRuleList{
		Items: []ColocationWorkloadMatchRule{
			{ObjectMeta: metav1.ObjectMeta{Name: "item1"}},
		},
	}
	obj := original.DeepCopyObject()
	copied, ok := obj.(*ColocationWorkloadMatchRuleList)
	assert.True(t, ok)
	assert.Equal(t, 1, len(copied.Items))
}

func TestColocationWorkloadMatchRuleListDeepCopyObject_Nil(t *testing.T) {
	var nilList *ColocationWorkloadMatchRuleList
	assert.Nil(t, nilList.DeepCopyObject())
}

func TestColocationWorkloadMatchRuleSpecDeepCopy(t *testing.T) {
	original := &ColocationWorkloadMatchRuleSpec{
		Colocation: ColocationOnline,
	}
	copied := original.DeepCopy()
	assert.True(t, reflect.DeepEqual(*original, *copied))

	var nilSpec *ColocationWorkloadMatchRuleSpec
	assert.Nil(t, nilSpec.DeepCopy())
}

func TestColocationWorkloadMatchRuleStatusDeepCopy(t *testing.T) {
	original := &ColocationWorkloadMatchRuleStatus{}
	copied := original.DeepCopy()
	assert.True(t, reflect.DeepEqual(*original, *copied))

	var nilStatus *ColocationWorkloadMatchRuleStatus
	assert.Nil(t, nilStatus.DeepCopy())
}
