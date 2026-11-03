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

package kubelet

import (
	_ "embed"
	"fmt"
	"strings"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"

	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/docker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/httprepo"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

//go:embed tmpl/docker.tmpl
var dockerctl string

const dockerCleanKubelet = "docker ps -a | grep %s | awk '{print $1}' | xargs docker rm -f"

func (kp *kubeletPlugin) runWithDocker(config map[string]string) error {
	log.Infof("run kubelet with docker")

	if err := kp.exec.ExecuteCommand("/bin/sh", "-c", fmt.Sprintf("rm -f %s/cpu_manager_state", config["dataRootDir"])); err != nil {
		log.Warnf("failed to remove cpu_manager_state, err: %v", err)
	}

	if err := kp.ensureImages(config); err != nil {
		return errors.Wrap(err, "failed to ensure images")
	}

	_ = mountList()

	// 如果是麒麟需要重新生成kubelet config，cgroupDriver需要设置为cgroupfs
	h, _, _, err := host.PlatformInformation()
	if err != nil {
		return fmt.Errorf("get host platform information failed, err: %w", err)
	}
	if h == "kylin" {
		if err := httprepo.RepoSearch("docker-ce"); err != nil {
			log.Warnf("[kylin] search docker-ce from repo failed, use docker-engine will set cgroupDriver to cgroupfs, err: %v", err)
			if config != nil {
				config["cgroupDriver"] = "cgroupfs"
			}
			if err = kp.generateKubeletConfig(config); err != nil {
				return err
			}
		}
	}

	log.Infof("start kubelet container %q", config["containerName"])
	if err := newKubeletScript(config); err != nil {
		return fmt.Errorf("failed to generate kubelet script, err: %w", err)
	}
	runKubeletCommand := fmt.Sprintf("%s -a start -r %s", utils.GetKubeletScriptPath(), "docker")
	if output, err := kp.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", runKubeletCommand); err != nil {
		return errors.Wrapf(err, "failed to run kubelet container %q, output: %s", config["containerName"], output)
	}
	return nil
}

func newDockerCommand(config map[string]string) (string, []string, error) {
	return newKubeletCommand(config, "docker")
}

func dockerRunKubeletCommand(config map[string]string) (string, []string, error) {
	return runKubeletCommandFromTemplate(config, "dockerctl", dockerctl)
}

func getDockerRunKubeletConfig(config map[string]string) docker.ContainerSpec {
	var mounts []mount.Mount
	for _, v := range mountList() {
		mounts = append(mounts, mount.Mount{
			Type:   mount.TypeBind,
			Source: v[0],
			Target: v[1],
		})
	}
	return docker.ContainerSpec{
		ContainerConfig: &container.Config{
			Image:        config["kubeletImage"],
			Cmd:          cmdList(config),
			User:         "root",
			AttachStdin:  false,
			AttachStdout: false,
			AttachStderr: false,
			Tty:          true,
			StdinOnce:    false,
		},
		HostConfig: &container.HostConfig{
			Mounts:      mounts,
			Privileged:  true,
			NetworkMode: "host",
			PidMode:     "host",
			RestartPolicy: container.RestartPolicy{
				Name: "always",
			},
		},
		ContainerName: config["containerName"],

		NetworkingConfig: nil,
		Platform:         nil,
	}
}

func cmdList(config map[string]string) []string {
	nodeIP, _ := bkenet.GetExternalIP()
	command := []string{
		"kubelet",
		fmt.Sprintf("--pod-infra-container-image=%s", config["pauseImage"]),
		fmt.Sprintf("--kubeconfig=%s", config["kubeconfigPath"]),
		fmt.Sprintf("--node-ip=%s", nodeIP),
		"--config=/var/lib/kubelet/config.yaml",
		fmt.Sprintf("--hostname-override=%s", utils.HostName()),
		"--network-plugin=cni",
		"--cni-conf-dir=/etc/cni/net.d",
		"--cni-bin-dir=/opt/cni/bin",
		"--v=0",
	}
	if v, ok := config["extraArgs"]; ok {
		extraArgs := strings.Split(v, " ")
		for _, arg := range extraArgs {
			if arg != "" {
				command = append(command, arg)
			}
		}
	}

	return command
}

func (kp *kubeletPlugin) ensureDockerConfig(config map[string]string) error {
	repo := strings.Split(config["imageRepo"], "/")[0]
	err := docker.ConfigInsecureRegistries([]string{repo})
	if err != nil {
		return errors.Wrap(err, "failed to ensure insecure registries")
	}
	err = docker.ConfigCgroupDriver("systemd")
	if err != nil {
		return errors.Wrap(err, "failed to ensure cgroup driver")
	}

	log.Infof("restart docker")
	output, err := kp.exec.ExecuteCommandWithOutput("systemctl", "daemon-reload")
	if err != nil {
		return errors.Wrapf(err, "failed to reload docker daemon.conf: %s", output)
	}
	output, err = kp.exec.ExecuteCommandWithOutput("systemctl", "restart", "docker")
	if err != nil {
		return errors.Wrapf(err, "failed to restart docker: %s", output)
	}
	return docker.WaitDockerReady()
}
