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

package externalversions

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"

	apisv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
	"openfuyao.com/colocation-service/pkg/client/clientset/versioned/fake"
)

func TestNewSharedInformerFactory(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactory(cs, time.Minute)
	assert.NotNil(t, factory)
}

func TestNewSharedInformerFactoryWithOptions(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactoryWithOptions(cs, time.Minute, WithNamespace("default"))
	assert.NotNil(t, factory)
}

func TestNewFilteredSharedInformerFactory(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewFilteredSharedInformerFactory(cs, time.Minute, "default", nil)
	assert.NotNil(t, factory)
}

func TestWithCustomResyncConfig(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactoryWithOptions(cs, time.Minute,
		WithCustomResyncConfig(map[metav1.Object]time.Duration{}))
	assert.NotNil(t, factory)
}

func TestWithTweakListOptions(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactoryWithOptions(cs, time.Minute,
		WithTweakListOptions(func(*metav1.ListOptions) {}))
	assert.NotNil(t, factory)
}

func TestWithTransform(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactoryWithOptions(cs, time.Minute,
		WithTransform(nil))
	assert.NotNil(t, factory)
}

func TestSharedInformerFactory_Colocation(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactory(cs, time.Minute)
	c := factory.Colocation()
	assert.NotNil(t, c)
	v1if := c.V1()
	assert.NotNil(t, v1if)
	informer := v1if.ColocationWorkloadMatchRules()
	assert.NotNil(t, informer)
}

func TestSharedInformerFactory_ForResource_Unknown(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactory(cs, time.Minute)
	_, err := factory.ForResource(schema.GroupVersionResource{Group: "unknown", Version: "v1", Resource: "unknown"})
	assert.Error(t, err)
}

func TestSharedInformerFactory_ForResource_Valid(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactory(cs, time.Minute)
	gi, err := factory.ForResource(apisv1.SchemeGroupVersion.WithResource("colocationworkloadmatchrules"))
	assert.NoError(t, err)
	assert.NotNil(t, gi)
	assert.NotNil(t, gi.Informer())
	assert.NotNil(t, gi.Lister())
}

func TestSharedInformerFactory_Start_Empty(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactory(cs, time.Minute)
	stopCh := make(chan struct{})
	factory.Start(stopCh)
	close(stopCh)
}

func TestSharedInformerFactory_Shutdown(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactory(cs, time.Minute)
	factory.Shutdown()
}

func TestSharedInformerFactory_WaitForCacheSync(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := NewSharedInformerFactory(cs, time.Minute)
	stopCh := make(chan struct{})
	result := factory.WaitForCacheSync(stopCh)
	close(stopCh)
	assert.NotNil(t, result)
}
