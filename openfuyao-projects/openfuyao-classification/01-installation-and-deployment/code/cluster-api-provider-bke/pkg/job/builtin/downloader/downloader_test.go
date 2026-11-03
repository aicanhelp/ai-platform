/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package downloader

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDownloaderName(t *testing.T) {
	plugin := &DownloaderPlugin{}
	assert.Equal(t, Name, plugin.Name())
}

func TestDownloaderParam(t *testing.T) {
	plugin := &DownloaderPlugin{}
	params := plugin.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "url")
	assert.Contains(t, params, "rename")
	assert.Contains(t, params, "chmod")
	assert.Contains(t, params, "saveto")
}

func TestDownloaderParamDefaults(t *testing.T) {
	plugin := &DownloaderPlugin{}
	params := plugin.Param()
	assert.Equal(t, "0644", params["chmod"].Default)
	assert.Equal(t, os.TempDir(), params["saveto"].Default)
}

func TestNewDownloader(t *testing.T) {
	plugin := New()
	assert.NotNil(t, plugin)
	assert.Equal(t, Name, plugin.Name())
}

func TestDownloaderExecuteMissingUrl(t *testing.T) {
	plugin := &DownloaderPlugin{}
	commands := []string{Name}
	result, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Missing required parameters")
	assert.Empty(t, result)
}

func TestDownloaderExecuteInvalidUrl(t *testing.T) {
	plugin := &DownloaderPlugin{}
	commands := []string{Name, "url=invalid-url"}
	result, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Empty(t, result)
}
