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

package docker

import (
	"context"
	_ "embed"
	"fmt"
	"strings"

	"github.com/pkg/errors"

	edocker "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/docker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/downloader"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/httprepo"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const Name = "InstallDocker"

var (
	defaultCgroupDriver = "systemd"
	defaultRuntime      = "runc"
	defaultDataRoot     = "/var/lib/docker"
)

type DockerPlugin struct {
	exec exec.Executor
}

func New(exec exec.Executor) plugin.Plugin {
	return &DockerPlugin{
		exec: exec,
	}
}

func (dp DockerPlugin) Name() string {
	return Name
}

func (dp DockerPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"insecureRegistries": {Key: "insecureRegistries", Value: "", Required: false, Default: "", Description: "insecure insecureRegistries, eg:a.com,b.com split by ',' "},
		"registryMirror":     {Key: "registryMirror", Value: "", Required: false, Default: "", Description: "registry mirror, eg:a.com,b.com split by ',' "},
		"cgroupDriver":       {Key: "cgroupDriver", Value: "", Required: false, Default: defaultCgroupDriver, Description: "cgroup driver, eg:systemd,cgroupfs"},
		"runtime":            {Key: "runtime", Value: "", Required: false, Default: defaultRuntime, Description: "runtime, eg:runc,richrunc,kata"},
		"dataRoot":           {Key: "dataRoot", Value: "", Required: false, Default: defaultDataRoot, Description: "Specify the data directory"},
		"enableDockerTls":    {Key: "enableDockerTls", Value: "", Required: false, Default: "false", Description: "Enable docker tls, eg:true,false"},
		"tlsHost":            {Key: "tlsHost", Value: "", Required: false, Default: "", Description: "tls host ip"},
		"runtimeUrl":         {Key: "runtimeUrl", Value: "", Required: false, Default: "", Description: "runtime url for download"},
	}
}

func (dp DockerPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	dp.exec = exec.Bound(dp.exec, ctx)

	runtimeParam, err := plugin.ParseCommands(dp, commands)
	if err != nil {
		return nil, err
	}
	if err = dp.installAndConfigureDocker(runtimeParam); err != nil {
		return nil, err
	}
	out, err := dp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl enable docker")
	if err != nil {
		log.Warnf("enable docker failed, err: %v, out: %s", err, out)
	}
	out, err = dp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl restart docker")
	if err != nil {
		errorMsg := fmt.Sprintf("start docker failed, err: %v, out: %s", err, out)
		return []string{errorMsg}, errors.New(errorMsg)
	}
	if err = edocker.WaitDockerReady(); err != nil {
		return nil, err
	}
	return nil, nil
}

func (dp DockerPlugin) installAndConfigureDocker(runtimeParam map[string]string) error {
	if runtimeParam == nil {
		return errors.New("runtime is required")
	}
	_ = dp.exec.ExecuteCommand("sh", "-c", "systemctl stop docker")
	_ = dp.exec.ExecuteCommand("sh", "-c", "systemctl stop docker.socket")
	if runtimeParam["runtime"] == "richrunc" {
		url := fmt.Sprintf("url=%s", runtimeParam["runtimeUrl"])
		cs := []string{downloader.Name, url, "chmod=755", "rename=runc", "saveto=/usr/local/beyondvm"}
		downloaderPlugin := downloader.New()
		if _, err := downloaderPlugin.Execute(context.Background(), cs); err != nil {
			return fmt.Errorf("download richrunc %s failed, err: %w", url, err)
		}
	}
	docker := "docker-ce"
	if err := httprepo.RepoSearch("docker-ce"); err != nil {
		log.Warnf("search docker-ce from repo failed, use docker-engine instead, err: %v", err)
		docker = "docker"
		runtimeParam["cgroupDriver"] = "cgroupfs"
	}
	if err := httprepo.RepoInstall(docker); err != nil {
		return fmt.Errorf("install docker failed, err: %w", err)
	}
	enableTls := runtimeParam["enableDockerTls"] == "true"
	registries := strings.Split(runtimeParam["insecureRegistries"], ",")
	return edocker.ConfigDockerDaemon(edocker.DockerDaemonConfig{
		CgroupDriver:       runtimeParam["cgroupDriver"],
		LowLevelRuntime:    runtimeParam["runtime"],
		DataRoot:           runtimeParam["dataRoot"],
		EnableTls:          enableTls,
		TlsHost:            runtimeParam["tlsHost"],
		InsecureRegistries: registries,
	})
}
