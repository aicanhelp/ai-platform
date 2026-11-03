package runtime

import (
	"os"

	econd "gopkg.openfuyao.cn/bkeagent/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeagent/pkg/executor/docker"
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
