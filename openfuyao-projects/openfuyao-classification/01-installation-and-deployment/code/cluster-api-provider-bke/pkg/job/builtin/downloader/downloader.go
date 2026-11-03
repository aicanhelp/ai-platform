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

package downloader

import (
	"context"
	"os"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/download"
)

const Name = "Downloader"

type DownloaderPlugin struct{}

func New() plugin.Plugin {
	return &DownloaderPlugin{}
}

func (d *DownloaderPlugin) Name() string {
	return Name
}

func (d *DownloaderPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"url": {
			Key:         "url",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "download url",
		},
		"rename": {
			Key:         "rename",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "rename downloaded file",
		},
		"chmod": {
			Key:         "perm",
			Value:       "",
			Required:    false,
			Default:     "0644",
			Description: "file permission",
		},
		"saveto": {
			Key:         "saveto",
			Value:       "",
			Required:    true,
			Default:     os.TempDir(),
			Description: "save to directory",
		},
	}
}

func (d *DownloaderPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	commandMap, err := plugin.ParseCommands(d, commands)
	if err != nil {
		return nil, err
	}
	url := commandMap["url"]
	rename := commandMap["rename"]
	saveto := commandMap["saveto"]
	chmod := commandMap["chmod"]

	if err := download.ExecDownload(url, saveto, rename, chmod); err != nil {
		return nil, err
	}
	return nil, nil
}
