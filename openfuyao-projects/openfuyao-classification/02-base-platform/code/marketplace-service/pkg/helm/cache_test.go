/*
 *
 *  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package helm

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/repo"
)

// TestNewChartCache 测试缓存初始化
func TestNewChartCache(t *testing.T) {
	cache := NewChartCache()
	assert.NotNil(t, cache)
	assert.IsType(t, &cachedChart{}, cache)
}

// TestSetChartCache 测试缓存设置
func TestSetChartCache(t *testing.T) {
	cache := &cachedChart{repoChartCache: make(map[string]map[string]repo.ChartVersions)}
	index := &repo.IndexFile{Entries: map[string]repo.ChartVersions{"test": {}}}

	cache.SetChartCache("test-repo", index)
	assert.Equal(t, index.Entries, cache.repoChartCache["test-repo"])
}

// TestUpdateChartCache 测试缓存更新
func TestUpdateChartCache(t *testing.T) {
	// Mock LoadRepoIndex 返回测试数据
	patches := gomonkey.ApplyFunc(LoadRepoIndex, func(_ *repo.Entry) (*repo.IndexFile, error) {
		return &repo.IndexFile{Entries: map[string]repo.ChartVersions{"test": {}}}, nil
	})
	defer patches.Reset()

	cache := &cachedChart{repoChartCache: make(map[string]map[string]repo.ChartVersions)}
	entry := &repo.Entry{Name: "test-repo"}
	err := cache.UpdateChartCache(entry)

	assert.NoError(t, err)
	assert.Equal(t, map[string]repo.ChartVersions{"test": {}}, cache.repoChartCache["test-repo"])
}

// TestUpdateChartCache_Error 测试缓存更新失败
func TestUpdateChartCache_Error(t *testing.T) {
	patches := gomonkey.ApplyFunc(LoadRepoIndex, func(_ *repo.Entry) (*repo.IndexFile, error) {
		return nil, assert.AnError
	})
	defer patches.Reset()

	cache := &cachedChart{}
	err := cache.UpdateChartCache(&repo.Entry{})
	assert.Error(t, err)
}

// TestGetChartCacheFromAllRepo 测试全仓库缓存查询
func TestGetChartCacheFromAllRepo(t *testing.T) {
	cache := &cachedChart{
		repoChartCache: map[string]map[string]repo.ChartVersions{"test-repo": {"test": {}}},
	}
	res := cache.GetChartCacheFromAllRepo()
	assert.Equal(t, cache.repoChartCache, res)
}

// TestGetChartCacheByRepo 测试指定仓库缓存查询
func TestGetChartCacheByRepo(t *testing.T) {
	cache := &cachedChart{
		repoChartCache: map[string]map[string]repo.ChartVersions{"test-repo": {"test": {}}},
	}
	// 存在的仓库
	entries, exists := cache.GetChartCacheByRepo("test-repo")
	assert.True(t, exists)
	assert.Equal(t, map[string]repo.ChartVersions{"test": {}}, entries)

	// 不存在的仓库
	entries2, exists2 := cache.GetChartCacheByRepo("no-repo")
	assert.False(t, exists2)
	assert.Nil(t, entries2)
}

// TestDeleteChartCache 测试缓存删除
func TestDeleteChartCache(t *testing.T) {
	cache := &cachedChart{
		repoChartCache: map[string]map[string]repo.ChartVersions{"test-repo": {"test": {}}},
	}
	cache.DeleteChartCache("test-repo")
	_, exists := cache.GetChartCacheByRepo("test-repo")
	assert.False(t, exists)
}
