/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

/*
Package helm 实现了对helm组件的所有操作
添加仓库，删除仓库
添加helm chart，删除helm chart
创建helm release，删除helm release
*/
package helm

import (
	"sync"

	"helm.sh/helm/v3/pkg/repo"
)

var cachedData = NewChartCache()

// NewChartCache create new map for storing charts in every repository
func NewChartCache() ChartCache {
	return &cachedChart{
		repoChartCache: map[string]map[string]repo.ChartVersions{},
	}
}

// ChartCache operations for manipulate cached charts
type ChartCache interface {
	SetChartCache(repoName string, indexFile *repo.IndexFile)
	UpdateChartCache(repoEntry *repo.Entry) error
	GetChartCacheFromAllRepo() map[string]map[string]repo.ChartVersions
	GetChartCacheByRepo(repoName string) (indexEntries map[string]repo.ChartVersions, exists bool)
	DeleteChartCache(repoName string)
}

type cachedChart struct {
	sync.RWMutex
	repoChartCache map[string]map[string]repo.ChartVersions
}

func (c *cachedChart) UpdateChartCache(repoEntry *repo.Entry) error {
	index, err := LoadRepoIndex(repoEntry)
	if err != nil {
		return err
	}
	c.Lock()
	defer c.Unlock()
	c.repoChartCache[repoEntry.Name] = index.Entries
	return nil
}

func (c *cachedChart) SetChartCache(repoName string, indexFile *repo.IndexFile) {
	c.Lock()
	defer c.Unlock()
	c.repoChartCache[repoName] = indexFile.Entries
}

func (c *cachedChart) GetChartCacheFromAllRepo() map[string]map[string]repo.ChartVersions {
	c.RLock()
	defer c.RUnlock()
	return c.repoChartCache
}

func (c *cachedChart) GetChartCacheByRepo(repoName string) (map[string]repo.ChartVersions, bool) {
	c.RLock()
	defer c.RUnlock()
	if indexEntries, exists := c.repoChartCache[repoName]; exists {
		return indexEntries, true
	}
	return nil, false
}

func (c *cachedChart) DeleteChartCache(repoName string) {
	c.RLock()
	defer c.RUnlock()
	delete(c.repoChartCache, repoName)
}
