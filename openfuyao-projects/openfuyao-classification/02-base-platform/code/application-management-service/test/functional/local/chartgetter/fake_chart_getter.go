/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package chartgetter

import (
	"fmt"
	"sync"

	"helm.sh/helm/v3/pkg/chart"
)

// FakeChartGetter is a reusable fake for helm.ChartGetter.
// It is used by integration tests to avoid real marketplace RPC calls.
type FakeChartGetter struct {
	mu sync.Mutex

	Charts map[string]*chart.Chart
	Err    error

	Last struct {
		Repo    string
		Chart   string
		Version string
	}
}

func New() *FakeChartGetter {
	return &FakeChartGetter{Charts: map[string]*chart.Chart{}}
}

func Key(repo, chartName, version string) string {
	return fmt.Sprintf("%s/%s:%s", repo, chartName, version)
}

func (f *FakeChartGetter) WithChart(repo, chartName, version string, c *chart.Chart) *FakeChartGetter {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Charts[Key(repo, chartName, version)] = c
	return f
}

func (f *FakeChartGetter) WithError(err error) *FakeChartGetter {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Err = err
	return f
}

func (f *FakeChartGetter) Get(repoName, chartName, version string) (*chart.Chart, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.Last.Repo = repoName
	f.Last.Chart = chartName
	f.Last.Version = version

	if f.Err != nil {
		return nil, f.Err
	}
	if c, ok := f.Charts[Key(repoName, chartName, version)]; ok {
		return c, nil
	}
	return nil, fmt.Errorf("chart not found: %s", Key(repoName, chartName, version))
}
