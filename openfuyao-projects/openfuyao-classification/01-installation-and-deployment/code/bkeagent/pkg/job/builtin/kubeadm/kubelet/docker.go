package kubelet

import (
	"bytes"
	_ "embed"
	"fmt"
	"strings"
	"text/template"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"
	bkenet "gopkg.openfuyao.cn/bkecommon/utils/net"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/httprepo"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

//go:embed tmpl/docker.tmpl
var dockerctl string

const dockerCleanKubelet = "docker ps -a | grep %s | awk '{print $1}' | xargs docker rm -f"

func (kp *kubeletPlugin) runWithDocker(config map[string]string) error {
	log.Infof("run kubelet with docker")
	//if err := kp.ensureDockerConfig(config); err != nil {
	//	return errors.Wrap(err, "failed to ensure docker config")
	//}

	//if out, err := kp.exec.ExecuteCommandWithOutput("/bin/sh", "-c", fmt.Sprintf(dockerCleanKubelet, config["containerName"])); err != nil {
	//	log.Warnf("failed to remove container %s, %s, err: %v", config["containerName"], out, err)
	//}
	// rm dataRootDir/cpu_manager_state
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
		log.Errorf("get host platform information failed, err: %v", err)
		return errors.Errorf("get host platform information failed, err: %v", err)
	}
	if h == "kylin" {
		if err := httprepo.RepoSearch("docker-ce"); err != nil {
			log.Warnf("[kylin] search docker-ce from repo failed, use docker-engine will set cgroupDriver to cgroupfs, err: %v", err)
			config["cgroupDriver"] = "cgroupfs"
			if err = kp.generateKubeletConfig(config); err != nil {
				return err
			}
		}
	}

	log.Infof("start kubelet container %q", config["containerName"])
	if err := newKubeletScript(config); err != nil {
		return errors.Errorf("failed to generate kubelet script, err: %v", err)
	}
	runKubeletCommand := fmt.Sprintf("%s -a start -r %s", utils.GetKubeletScriptPath(), "docker")
	if output, err := kp.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", runKubeletCommand); err != nil {
		return errors.Wrapf(err, "failed to run kubelet container %q, output: %s", config["containerName"], output)
	}
	//command, args, err := newDockerCommand(config)
	//if err != nil {
	//	return err
	//}
	//if output, err := kp.exec.ExecuteCommandWithCombinedOutput(command, args...); err != nil {
	//	return errors.Wrapf(err, "failed to run kubelet container %q, output: %s", config["containerName"], output)
	//}
	return nil
}

func newDockerCommand(config map[string]string) (string, []string, error) {
	return NewRunKubeletCommand().
		SetContainerRuntime("docker").
		SetKubeletImage(config["kubeletImage"]).
		SetPauseImage(config["pauseImage"]).
		SetKubeConfigPath(config["kubeconfigPath"]).
		SetExtraVolumes(strings.Split(config["extraVolumes"], ";")).
		SetExtraKubeletArgs(strings.Split(config["extraArgs"], ";")).
		SetK8sVersion(config["kubernetesVersion"]).
		SetHostIP(config["hostIP"]).
		SetRootDirPath(config["dataRootDir"]).
		Command()
}

func dockerRunKubeletCommand(config map[string]string) (string, []string, error) {
	externalHost, err := bkenet.GetExternalIP()
	if err != nil {
		return "", nil, err
	}
	config["externalHost"] = externalHost
	config["hostName"] = utils.HostName()

	tpl, err := template.New("dockerctl").Parse(dockerctl)
	if err != nil {
		return "", nil, err
	}
	var buf bytes.Buffer
	if err := tpl.Execute(&buf, config); err != nil {
		return "", nil, err
	}
	t := strings.ReplaceAll(buf.String(), "\n", " ")
	t = strings.ReplaceAll(t, "\\", "")
	t = strings.ReplaceAll(t, "\r", "")
	r := strings.Split(t, " ")
	var res []string
	for _, v := range r {
		if v != "" {
			res = append(res, v)
		}
	}
	return res[0], res[1:], nil
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

	return nil
}
