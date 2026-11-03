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
	"context"
	"fmt"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/go-connections/nat"

	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const containerRemoveMaxRetry = 2

// removeContainerWithRetry 移除容器，支持重试和额外清理操作
func removeContainerWithRetry(name string, extraCleanup func()) error {
	if infrastructure.IsDocker() {
		return removeExistingContainerWithRetry(name, extraCleanup,
			func(n string) bool {
				_, ok := global.Docker.ContainerExists(n)
				return ok
			},
			global.Docker.ContainerRemove,
		)
	}
	if infrastructure.IsContainerd() {
		return removeExistingContainerWithRetry(name, extraCleanup,
			func(n string) bool {
				_, ok := econd.ContainerExists(n)
				return ok
			},
			econd.ContainerRemove,
		)
	}
	return nil
}

func removeExistingContainerWithRetry(
	name string,
	extraCleanup func(),
	exists func(string) bool,
	remove func(string) error,
) error {
	if !exists(name) {
		return nil
	}
	for i := 0; i < containerRemoveMaxRetry; i++ {
		if err := remove(name); err != nil {
			log.Warnf("Remove container %s failed (attempt %d/%d): %v", name, i+1, containerRemoveMaxRetry, err)
		}
		if extraCleanup != nil {
			extraCleanup()
		}
		if !exists(name) {
			return nil
		}
		if i < containerRemoveMaxRetry-1 {
			time.Sleep(utils.ContainerRemoveWaitSeconds * time.Second)
		}
	}
	if exists(name) {
		log.Warnf("Remove container %s still exists after %d attempts", name, containerRemoveMaxRetry)
		return fmt.Errorf("container %s still exists after remove retries", name)
	}
	return nil
}

// StartChartRegistry starts the chart repository service.
func StartChartRegistry(name, image, chartRegistryPort, chartDataDirectory string) error {
	if infrastructure.IsContainerd() && !infrastructure.IsDocker() {
		return startChartRegistryWithContainerd(name, image, chartRegistryPort, chartDataDirectory)
	}

	if !infrastructure.IsDocker() {
		return fmt.Errorf("docker or containerd runtime not found")
	}
	return startChartRegistryWithDocker(name, image, chartRegistryPort, chartDataDirectory)
}

// startChartRegistryWithDocker 使用 Docker 启动 Chart 仓库服务
func startChartRegistryWithDocker(name, image, chartRegistryPort, chartDataDirectory string) error {
	isRunning, err := ensureDockerImageAndContainer(name, image, "chart warehouse")
	if err != nil {
		return err
	}
	if isRunning {
		return nil
	}

	err = runDockerChartRegistry(name, image, chartRegistryPort, chartDataDirectory)
	if err != nil {
		return err
	}

	waitForDockerContainerRunning(name, "chart")
	return nil
}

// runDockerChartRegistry 运行 Docker Chart 仓库容器
func runDockerChartRegistry(name, image, chartRegistryPort, chartDataDirectory string) error {
	err := global.Docker.Run(
		&container.Config{
			Image: image,
			ExposedPorts: map[nat.Port]struct{}{
				"8080/tcp": {},
			},
			Env: []string{"DEBUG=true", "STORAGE=local", "STORAGE_LOCAL_ROOTDIR=/charts"},
		},
		&container.HostConfig{
			Mounts: []mount.Mount{
				{
					Type:   mount.TypeBind,
					Source: chartDataDirectory,
					Target: "/charts",
				},
			},
			PortBindings: map[nat.Port][]nat.PortBinding{
				nat.Port("8080/tcp"): {
					{HostIP: "0.0.0.0", HostPort: chartRegistryPort},
				},
			},
			RestartPolicy: container.RestartPolicy{Name: "always", MaximumRetryCount: 0},
		}, nil, nil, name)
	if err != nil {
		return fmt.Errorf("deploy chart registry container: %w", err)
	}
	return nil
}

// waitForDockerContainerRunning 等待 Docker 容器启动运行
func waitForDockerContainerRunning(name, serviceName string) {
	client := global.Docker.GetClient()
	for {
		log.Infof("Wait for the %s mirroring service to start...", serviceName)
		time.Sleep(utils.ContainerStartWaitSeconds * time.Second)
		containerInfo, err := client.ContainerInspect(context.Background(), name)
		if err != nil {
			continue
		}
		if containerInfo.ContainerJSONBase == nil {
			continue
		}
		if containerInfo.State.Running {
			break
		}
	}
	log.Infof("The %s mirroring service is started. ", serviceName)
}

func startChartRegistryWithContainerd(name, image, chartRegistryPort, chartDataDirectory string) error {
	err := econd.EnsureImageExists(image)
	if err != nil {
		return err
	}

	serverRunFlag, err := econd.EnsureContainerRun(name)
	if err != nil {
		return err
	}
	// 服务已经运行
	if serverRunFlag {
		log.Info("The chart warehouse service is already running. ")
		return nil
	}
	script := []string{
		"run", "-d", fmt.Sprintf("--name=%s", name),
		"-p", fmt.Sprintf("%s:8080", chartRegistryPort),
		"-e", "DEBUG=true", "-e", "STORAGE=local", "-e", "STORAGE_LOCAL_ROOTDIR=/charts",
		"--restart=always",
		"-v", fmt.Sprintf("%s:/charts", chartDataDirectory),
		image,
	}
	err = econd.Run(script)
	if err != nil {
		return fmt.Errorf("deploy chart registry container: %w", err)
	}

	for {
		log.Info("Wait for the chart mirroring service to start...")
		time.Sleep(utils.ContainerStartWaitSeconds * time.Second)
		info, err := econd.ContainerInspect(name)
		if err != nil {
			continue
		}
		if info.State.Running {
			break
		}
	}
	log.Info("The chart mirroring service is started. ")
	return nil
}

func RemoveChartRegistry(name string) error {
	log.Info("Remove the chart repository")
	return removeContainerWithRetry(name, nil)
}
