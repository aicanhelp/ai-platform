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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/repo"
)

// TestLoadRepoIndex 测试索引文件加载
func TestLoadRepoIndex(t *testing.T) {
	// 模拟测试服务器
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte(`apiVersion: v1
entries:
  test-chart:
  - version: 1.0.0
    urls:
    - http://test/test-chart-1.0.0.tgz
`))
		assert.NoError(t, err)
	}))
	defer ts.Close()

	repoEntry := &repo.Entry{URL: ts.URL}
	index, err := LoadRepoIndex(repoEntry)
	assert.NoError(t, err)
	assert.Equal(t, "v1", index.APIVersion)
	assert.Contains(t, index.Entries, "test-chart")
}

// TestLoadChartBytes 测试Chart字节加载
func TestLoadChartBytes(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("test chart data"))
		assert.NoError(t, err)
	}))
	defer ts.Close()

	repoEntry := &repo.Entry{URL: ts.URL}
	buf, err := LoadChartBytes("test-chart.tgz", repoEntry)
	assert.NoError(t, err)
	assert.Equal(t, "test chart data", buf.String())
}

// TestLoadData_StatusCode 测试非200状态码
func TestLoadData_StatusCode(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer ts.Close()

	repoEntry := &repo.Entry{URL: ts.URL}
	_, err := LoadData(ts.URL, repoEntry)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "received non-200 status code: 404")
}
