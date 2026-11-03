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

package server

import (
	_ "embed"
	"fmt"
	"os"
	"path"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/go-connections/nat"

	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	//go:embed nginx.conf
	nginxConf string
)

// ensureDockerImageAndContainer 确保镜像存在并检查容器是否运行
func ensureDockerImageAndContainer(name, image, serviceName string) (bool, error) {
	if err := global.Docker.EnsureImageExists(docker.ImageRef{Image: image},
		utils.NewRetryOptions(utils.MaxRetryCount, utils.DelayTime)); err != nil {
		return false, err
	}
	serverRunFlag, err := global.Docker.EnsureContainerRun(name)
	if err != nil {
		return false, err
	}
	if serverRunFlag {
		log.Infof("The %s service is already running. ", serviceName)
		return true, nil
	}
	return false, nil
}

func StartYumRegistry(name, image, yumRegistryPort, yumDataDirectory string) error {
	if infrastructure.IsContainerd() && !infrastructure.IsDocker() {
		return startYumRegistryWithContainerd(name, image, yumRegistryPort, yumDataDirectory)
	}

	if !infrastructure.IsDocker() {
		return fmt.Errorf("docker or containerd runtime not found")
	}
	return startYumRegistryWithDocker(name, image, yumRegistryPort, yumDataDirectory)
}

// startYumRegistryWithDocker 使用 Docker 启动 Yum 仓库服务
func startYumRegistryWithDocker(name, image, yumRegistryPort, yumDataDirectory string) error {
	isRunning, err := ensureDockerImageAndContainer(name, image, "yum warehouse")
	if err != nil {
		return err
	}
	if isRunning {
		return nil
	}

	if err := runDockerYumRegistry(name, image, yumRegistryPort, yumDataDirectory); err != nil {
		return err
	}

	waitForDockerContainerRunning(name, "yum")
	return nil
}

// runDockerYumRegistry 运行 Docker Yum 仓库容器
func runDockerYumRegistry(name, image, yumRegistryPort, yumDataDirectory string) error {
	err := global.Docker.Run(
		&container.Config{
			Image:        image,
			ExposedPorts: map[nat.Port]struct{}{"80/tcp": {}},
		},
		&container.HostConfig{
			Mounts: []mount.Mount{{Type: mount.TypeBind, Source: yumDataDirectory, Target: "/repo"}},
			PortBindings: map[nat.Port][]nat.PortBinding{
				nat.Port("80/tcp"): {{HostIP: "0.0.0.0", HostPort: yumRegistryPort}},
			},
			RestartPolicy: container.RestartPolicy{Name: "always", MaximumRetryCount: 0},
		}, nil, nil, name)
	if err != nil {
		return fmt.Errorf("deploy yum registry container: %w", err)
	}
	return nil
}

func startYumRegistryWithContainerd(name, image, yumRegistryPort, yumDataDirectory string) error {
	err := econd.EnsureImageExists(image)
	if err != nil {
		return err
	}

	confPath := fmt.Sprintf("%s/%s", k3s.DefaultK3sDataDir, name)
	if !utils.FileExists(confPath) {
		err = os.MkdirAll(confPath, utils.DefaultDirPermission)
		if err != nil {
			return err
		}
	}
	conf := path.Join(confPath, "nginx.conf")
	if !utils.FileExists(conf) {
		err = utils.WriteCommon(conf, nginxConf)
		if err != nil {
			return err
		}
	}

	serverRunFlag, err := econd.EnsureContainerRun(name)
	if err != nil {
		return err
	}
	// 服务已经运行
	if serverRunFlag {
		log.Info("The yum warehouse service is already running. ")
		return nil
	}
	script := []string{
		"run", "-d", fmt.Sprintf("--name=%s", name),
		"-p", fmt.Sprintf("%s:80", yumRegistryPort),
		"--restart=always",
		"-v", fmt.Sprintf("%s:/etc/nginx/conf.d/default.conf", conf),
		"-v", fmt.Sprintf("%s:/repo", yumDataDirectory),
		image,
	}
	err = econd.Run(script)
	if err != nil {
		return fmt.Errorf("deploy yum registry container: %w", err)
	}

	for {
		log.Info("Wait for the container yum service to start...")
		time.Sleep(utils.ContainerStartWaitSeconds * time.Second)
		info, err := econd.ContainerInspect(name)
		if err != nil {
			continue
		}
		if info.State.Running {
			break
		}
	}
	log.Info("The container yum service is started. ")
	return nil
}

func RemoveYumRegistry(name string) error {
	log.Info("Remove the yum repository")
	return removeContainerWithRetry(name, nil)
}
