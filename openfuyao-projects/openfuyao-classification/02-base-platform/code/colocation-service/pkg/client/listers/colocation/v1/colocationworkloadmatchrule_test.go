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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/tools/cache"

	apisv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
)

func TestNewColocationWorkloadMatchRuleLister(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	lister := NewColocationWorkloadMatchRuleLister(indexer)
	assert.NotNil(t, lister)
}

func TestColocationWorkloadMatchRuleLister_List(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	obj := &apisv1.ColocationWorkloadMatchRule{}
	_ = indexer.Add(obj)

	lister := NewColocationWorkloadMatchRuleLister(indexer)
	result, err := lister.List(labels.Everything())
	assert.NoError(t, err)
	assert.Equal(t, 1, len(result))
}

func TestColocationWorkloadMatchRuleLister_ColocationWorkloadMatchRules(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	lister := NewColocationWorkloadMatchRuleLister(indexer)
	nsLister := lister.ColocationWorkloadMatchRules("default")
	assert.NotNil(t, nsLister)
}

func TestColocationWorkloadMatchRuleNamespaceLister_Get(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	obj := &apisv1.ColocationWorkloadMatchRule{}
	obj.Name = "test-rule"
	obj.Namespace = "default"
	_ = indexer.Add(obj)

	lister := NewColocationWorkloadMatchRuleLister(indexer)
	nsLister := lister.ColocationWorkloadMatchRules("default")
	result, err := nsLister.Get("test-rule")
	assert.NoError(t, err)
	assert.Equal(t, "test-rule", result.Name)
}

func TestColocationWorkloadMatchRuleNamespaceLister_Get_NotFound(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	lister := NewColocationWorkloadMatchRuleLister(indexer)
	nsLister := lister.ColocationWorkloadMatchRules("default")
	_, err := nsLister.Get("nonexistent")
	assert.Error(t, err)
}

func TestColocationWorkloadMatchRuleNamespaceLister_List(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	obj := &apisv1.ColocationWorkloadMatchRule{}
	obj.Name = "test-rule"
	obj.Namespace = "default"
	_ = indexer.Add(obj)

	lister := NewColocationWorkloadMatchRuleLister(indexer)
	nsLister := lister.ColocationWorkloadMatchRules("default")
	result, err := nsLister.List(labels.Everything())
	assert.NoError(t, err)
	assert.Equal(t, 1, len(result))
}
