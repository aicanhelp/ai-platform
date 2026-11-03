/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package infrastructure

import (
	"context"
	"fmt"
	"time"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	cond "gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/dockerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

// RuntimeConfig 容器运行时安装配置
type RuntimeConfig struct {
	Runtime        string // 运行时类型：docker 或 containerd
	RuntimeStorage string // 运行时存储路径
	Domain         string // 镜像仓库域名
	ImageRepoPort  string // 镜像仓库端口
	ContainerdFile string // containerd 安装包路径
	CniPluginFile  string // CNI 插件安装包路径
	DockerdFile    string // docker 安装包路径
	HostIP         string // 主机 IP
	CAFile         string // CA 证书文件路径
}

// IsDocker 判断是否安装docker
func IsDocker() bool {
	if global.Docker == nil {
		global.Docker, _ = docker.NewDockerClient()
	}
	if global.Docker != nil {
		// 变更为timout请求
		ctx, cancel := context.WithTimeout(context.Background(), utils.DefaultTimeoutSeconds*time.Second)
		defer cancel()
		_, err := global.Docker.GetClient().Ping(ctx)
		if err == nil {
			log.Info("The docker client is ready.")
			return true
		}
	}
	return false
}

// IsContainerd 判断是否安装containerd
func IsContainerd() bool {
	if global.Containerd == nil {
		global.Containerd, _ = containerd.NewContainedClient()
	}
	if global.Containerd != nil {
		ctx, cancel := context.WithTimeout(context.Background(), utils.DefaultTimeoutSeconds*time.Second)
		defer cancel()
		flag, err := global.Containerd.GetClient().IsServing(ctx)
		if flag && err == nil {
			if !utils.Exists(utils.NerdCtl) {
				log.Debug("The /usr/bin/nerdctl tool was not found.")
				return false
			}
			return true
		}
	}
	return false
}

// RuntimeInstall
// 优先安装containerd
// 当没有containerd包时，安装docker
func RuntimeInstall(cfg RuntimeConfig) error {
	if IsDocker() || IsContainerd() {
		return nil
	}
	if cfg.Runtime == "docker" {
		if err := dockerd.EnsureDockerServer(cfg.Domain, cfg.RuntimeStorage, cfg.DockerdFile, cfg.HostIP); err != nil {
			return fmt.Errorf("install docker: %w", err)
		}
		return nil
	}

	err := cond.Install(cfg.Domain, cfg.ImageRepoPort, cfg.RuntimeStorage, cfg.ContainerdFile, cfg.CAFile)
	if err != nil {
		return fmt.Errorf("install containerd: %w", err)
	}

	if len(cfg.CniPluginFile) > 0 {
		if err = cond.CniPluginInstall(cfg.CniPluginFile); err != nil {
			return fmt.Errorf("install CNI plugin: %w", err)
		}
	}
	return nil
}

// StartLocalKubernetes
// kind docker 集群废弃的
// k3s containerd 当前支持
func StartLocalKubernetes(cfg k3s.Config, localImage string) error {
	if IsDocker() {
		if err := k3s.StartK3sWithDocker(cfg, localImage); err != nil {
			return fmt.Errorf("start k3s with docker: %w", err)
		}
	}
	if IsContainerd() {
		if err := k3s.EnsureDirExists(utils.DefaultExtendManifestsDir); err != nil {
			return fmt.Errorf("prepare k3s manifests directory: %w", err)
		}
		if err := k3s.StartK3sWithContainerd(cfg, localImage); err != nil {
			return fmt.Errorf("start k3s with containerd: %w", err)
		}
	}
	return nil
}
