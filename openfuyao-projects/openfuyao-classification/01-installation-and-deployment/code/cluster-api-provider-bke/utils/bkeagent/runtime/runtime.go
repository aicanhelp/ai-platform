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

package runtime

import (
	"os"

	econd "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/containerd"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/docker"
)

const (
	ContainerRuntimeDocker     = "docker"
	ContainerRuntimeContainerd = "containerd"
	dockerSockLinux            = "/var/run/docker.sock"
	containerdSockLinux        = "/var/run/containerd/containerd.sock"
)

func DetectRuntime() string {
	var err error

	switch os.Getenv("RUNTIME") {
	case ContainerRuntimeDocker:
		return ContainerRuntimeDocker
	case ContainerRuntimeContainerd:
		return ContainerRuntimeContainerd
	default:
	}

	//docker
	if _, err = os.Stat(dockerSockLinux); err == nil {
		if _, err := docker.NewDockerClient(); err == nil {
			return ContainerRuntimeDocker
		}
	}

	// containerd
	if _, err = os.Stat(containerdSockLinux); err == nil {
		if _, err := econd.NewContainedClient(); err == nil {
			return ContainerRuntimeContainerd
		}
	}

	return ""
}
