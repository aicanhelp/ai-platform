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
	"fmt"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/strslice"
	"github.com/docker/go-connections/nat"
	"github.com/docker/go-units"

	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

func StartNFSServer(name, image, nfsDataDirectory string) error {
	if infrastructure.IsContainerd() && !infrastructure.IsDocker() {
		return startNFSServerWithContainerd(name, image, nfsDataDirectory)
	}

	if !infrastructure.IsDocker() {
		return fmt.Errorf("docker or containerd runtime not found")
	}
	return startNFSServerWithDocker(name, image, nfsDataDirectory)
}

// startNFSServerWithDocker 使用 Docker 启动 NFS 服务
func startNFSServerWithDocker(name, image, nfsDataDirectory string) error {
	isRunning, err := ensureDockerImageAndContainer(name, image, "nfs")
	if err != nil {
		return err
	}
	if isRunning {
		return nil
	}

	if err := runDockerNFSServer(name, image, nfsDataDirectory); err != nil {
		return err
	}

	waitForDockerContainerRunning(name, "nfs")
	return nil
}

// runDockerNFSServer 运行 Docker NFS 容器
func runDockerNFSServer(name, image, nfsDataDirectory string) error {
	err := global.Docker.Run(
		&container.Config{
			Image:        image,
			ExposedPorts: map[nat.Port]struct{}{"2049/tcp": {}},
			Env: []string{
				"SHARED_DIRECTORY=/nfsshare",
				"FILEPERMISSIONS_UID=0",
				"FILEPERMISSIONS_GID=0",
				"FILEPERMISSIONS_MODE=0755",
			},
		},
		&container.HostConfig{
			Mounts:     []mount.Mount{{Type: mount.TypeBind, Source: nfsDataDirectory, Target: "/nfsshare"}},
			Privileged: true,
			PortBindings: map[nat.Port][]nat.PortBinding{
				nat.Port("2049/tcp"): {{HostIP: "0.0.0.0", HostPort: "2049"}},
			},
			RestartPolicy: container.RestartPolicy{Name: "always", MaximumRetryCount: 0},
			CapAdd:        strslice.StrSlice{"SYS_ADMIN", "SETPCAP"},
			Resources: container.Resources{
				Ulimits: []*units.Ulimit{{Name: "nofile", Hard: 65536, Soft: 65536}},
			},
		}, nil, nil, name)
	if err != nil {
		return fmt.Errorf("deploy nfs server container: %w", err)
	}
	return nil
}

func startNFSServerWithContainerd(name, image, nfsDataDirectory string) error {
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
		log.Info("The nfs service is already running. ")
		return nil
	}
	script := []string{
		"run", "-d", fmt.Sprintf("--name=%s", name), "--privileged",
		"-p", "2049:2049", "--restart=always", "-e", "SHARED_DIRECTORY=/nfsshare",
		"-e", "FILEPERMISSIONS_UID=0", "-e", "FILEPERMISSIONS_GID=0", "-e", "FILEPERMISSIONS_MODE=0755",
		"-v", fmt.Sprintf("%s:/nfsshare", nfsDataDirectory),
		image,
	}
	err = econd.Run(script)
	if err != nil {
		return fmt.Errorf("deploy nfs server container: %w", err)
	}

	for {
		log.Info("Wait for the nfs mirroring service to start...")
		time.Sleep(utils.ContainerStartWaitSeconds * time.Second)
		info, err := econd.ContainerInspect(name)
		if err != nil {
			continue
		}
		if info.State.Running {
			break
		}
	}
	log.Info("The nfs mirroring service is started. ")
	return nil
}

func RemoveNFSServer(name string) error {
	log.Info("Remove the nfs repository")
	nfsdCleanup := func() {
		if err := global.Command.ExecuteCommand("sh", "-c", "ps aux | grep nfsd | awk '{print $2}' | xargs kill -9"); err != nil {
			log.Warnf("Kill residual nfsd processes failed: %v", err)
		}
	}
	return removeContainerWithRetry(name, nfsdCleanup)
}
