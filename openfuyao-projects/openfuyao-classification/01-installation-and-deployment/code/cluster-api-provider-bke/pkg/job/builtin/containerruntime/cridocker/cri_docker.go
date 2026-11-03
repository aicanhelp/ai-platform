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

package cridocker

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"text/template"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/downloader"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/httprepo"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	Name = "InstallCRIDocker"
	// RwRR is the permission of the file
	RwRR = 0644
)

var (
	//go:embed cri-dockerd.service
	criDockerdService string
	//go:embed cri-dockerd.socket
	criDockerdSocket string

	defaultRepo    = fmt.Sprintf("%s:%s", utils.DefaultImageRepo, utils.DefaultImageRepoPort)
	defaultSandbox = fmt.Sprintf("%s/kubernetes/%s", defaultRepo, "pause:3.8")
)

type CRIDockerPlugin struct {
	exec exec.Executor
}

func New(exec exec.Executor) plugin.Plugin {
	return &CRIDockerPlugin{
		exec: exec,
	}
}

func (cdp CRIDockerPlugin) Name() string {
	return Name
}

func (cdp CRIDockerPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"criDockerdUrl": {
			Key:         "criDockerdUrl",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "cri-dockerd url for download",
		},
		"sandbox": {
			Key:         "sandbox",
			Value:       "",
			Required:    true,
			Default:     defaultSandbox,
			Description: "Pod sandbox",
		},
	}
}

func (cdp CRIDockerPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	cdp.exec = exec.Bound(cdp.exec, ctx)

	runtimeParam, err := plugin.ParseCommands(cdp, commands)
	if err != nil {
		return nil, err
	}
	// 检查cri-dockerd是否存在
	if utils.Exists("/usr/bin/cri-dockerd") {
		if err = writeCriDockerdConfigToDisk(runtimeParam); err != nil {
			return nil, err
		}
		if err = cdp.startCriDockerd(); err != nil {
			return nil, err
		}
	}

	_ = cdp.exec.ExecuteCommand("sh", "-c", "systemctl stop cri-dockerd")
	_ = cdp.exec.ExecuteCommand("sh", "-c", "systemctl stop cri-dockerd.socket")

	url := fmt.Sprintf("url=%s", runtimeParam["criDockerdUrl"])
	cs := []string{
		downloader.Name,
		url,
		"chmod=755",
		"rename=cri-dockerd",
		"saveto=/usr/bin",
	}

	downloaderPlugin := downloader.New()
	if _, err = downloaderPlugin.Execute(context.Background(), cs); err != nil {
		return nil, fmt.Errorf("download cri-dockerd %s failed, err: %w", url, err)
	}

	if err = writeCriDockerdConfigToDisk(runtimeParam); err != nil {
		return nil, err
	}

	if err = httprepo.RepoInstall("socat"); err != nil {
		return nil, err
	}

	if err = cdp.startCriDockerd(); err != nil {
		return nil, err
	}
	return nil, nil
}

func (cdp CRIDockerPlugin) startCriDockerd() error {
	out, err := cdp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl daemon-reload")
	if err != nil {
		log.Warnf("daemon-reload failed, err: %v, out: %s", err, out)
	}
	out, err = cdp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl enable cri-dockerd")
	if err != nil {
		log.Warnf("enable cri-dockerd failed, err: %v, out: %s", err, out)
	}
	// start cri-dockerd
	out, err = cdp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl restart cri-dockerd")
	if err != nil {
		return fmt.Errorf("start cri-dockerd failed, err: %w, out: %s", err, out)
	}
	return nil
}

func writeCriDockerdConfigToDisk(runtimeParam map[string]string) error {
	// Render configuration file
	f, err := os.OpenFile("/etc/systemd/system/cri-dockerd.service", os.O_WRONLY|os.O_CREATE, RwRR)
	if err != nil {
		return err
	}
	defer f.Close()
	tpl, err := template.New("cri-dockerd.service").Parse(criDockerdService)
	if err := tpl.Execute(f, runtimeParam); err != nil {
		return err
	}

	err = os.WriteFile("/etc/systemd/system/cri-dockerd.socket", []byte(criDockerdSocket), RwRR)
	if err != nil {
		return err
	}
	return nil
}
