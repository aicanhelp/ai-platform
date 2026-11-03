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

package v1_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	cache "k8s.io/client-go/tools/cache"

	"openfuyao.com/colocation-service/pkg/client/clientset/versioned/fake"
	"openfuyao.com/colocation-service/pkg/client/informers/externalversions"
	"openfuyao.com/colocation-service/pkg/client/informers/externalversions/colocation"
	informerv1 "openfuyao.com/colocation-service/pkg/client/informers/externalversions/colocation/v1"
)

func TestNew(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := externalversions.NewSharedInformerFactory(cs, time.Minute)
	g := colocation.New(factory, "default", nil)
	v := g.V1()
	assert.NotNil(t, v)
}

func TestVersion_ColocationWorkloadMatchRules(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := externalversions.NewSharedInformerFactory(cs, time.Minute)
	g := colocation.New(factory, "default", nil)
	v := g.V1()
	informer := v.ColocationWorkloadMatchRules()
	assert.NotNil(t, informer)

	i := informer.Informer()
	assert.NotNil(t, i)

	l := informer.Lister()
	assert.NotNil(t, l)
}

func TestNewColocationWorkloadMatchRuleInformer(t *testing.T) {
	cs := fake.NewSimpleClientset()
	indexers := cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}
	i := informerv1.NewColocationWorkloadMatchRuleInformer(cs, "default", time.Minute, indexers)
	assert.NotNil(t, i)
}

func TestNewFilteredColocationWorkloadMatchRuleInformer(t *testing.T) {
	cs := fake.NewSimpleClientset()
	indexers := cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}
	i := informerv1.NewFilteredColocationWorkloadMatchRuleInformer(cs, "default", time.Minute, indexers, nil)
	assert.NotNil(t, i)
}
