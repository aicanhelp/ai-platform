package kubelet

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"strings"
	"text/template"

	"github.com/containerd/containerd/namespaces"
	"github.com/pkg/errors"
	"gopkg.openfuyao.cn/bkecommon/cluster/imagehelper"
	bkenet "gopkg.openfuyao.cn/bkecommon/utils/net"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

//go:embed tmpl/nerdctl.tmpl
var nerdctl string

const nerdctlCleanKubelet = "nerdctl -n k8s.io ps -a | grep %s | awk '{print $1}' | xargs nerdctl -n k8s.io rm -f"

func (kp *kubeletPlugin) runWithContainerd(config map[string]string) error {
	log.Infof("run kubelet with containerd")
	// remove kubelet containers
	//if out, err := kp.exec.ExecuteCommandWithOutput("/bin/sh", "-c", fmt.Sprintf(nerdctlCleanKubelet, config["containerName"])); err != nil {
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

	log.Infof("start kubelet container %q", config["containerName"])
	if err := newKubeletScript(config); err != nil {
		return errors.Errorf("failed to generate kubelet script, err: %v", err)
	}
	runKubeletCommand := fmt.Sprintf("%s -a start -r %s", utils.GetKubeletScriptPath(), "containerd")
	if output, err := kp.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", runKubeletCommand); err != nil {
		return errors.Wrapf(err, "failed to run kubelet container %q, output: %s", config["containerName"], output)
	}

	//command, args, err := newNerdctlCommand(config)
	//if err != nil {
	//	return err
	//}
	//if output, err := kp.exec.ExecuteCommandWithCombinedOutput(command, args...); err != nil {
	//	return errors.Wrapf(err, "failed to run kubelet container %q, output: %s", config["containerName"], output)
	//}

	return nil
}

func (kp *kubeletPlugin) removeKubeletContainers(config map[string]string) error {
	ctx := context.Background()
	ctx = namespaces.WithNamespace(ctx, "k8s.io")
	containerList, err := kp.containerd.ContainerList()
	if err != nil {
		return err
	}
	repo := config["imageRepo"]
	clusterVersion := config["kubernetesVersion"]

	exporter := imagehelper.NewImageExporter(repo, clusterVersion, "")
	images, err := exporter.ExportImageList()
	if err != nil {
		return err
	}
	imagesNameList := make([]string, len(images))
	for i, image := range images {
		t := strings.Split(image, "/")
		tn := strings.Split(t[len(t)-1], ":")[0]
		imagesNameList[i] = tn
	}

	errs := []error{}
	for _, container := range containerList {
		deleted := false
		// 如果容器在运行中，先停止
		if ok, _ := kp.containerd.EnsureContainerRun(container.ID()); ok {
			err := kp.containerd.Stop(container.ID())
			if err != nil {
				errs = append(errs, err)
			}
		}
		image, err := container.Image(ctx)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if utils.ContainsString(images, image.Name()) {
			if err := kp.containerd.Delete(container.ID()); err != nil {
				errs = append(errs, err)
			}
			deleted = true
			continue
		}

		if !deleted {
			labels, err := container.Labels(ctx)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			if v, ok := labels["io.kubernetes.container.name"]; ok {
				if utils.ContainsString(imagesNameList, v) {
					if err := kp.containerd.Delete(container.ID()); err != nil {
						errs = append(errs, err)
					}
					deleted = true
					continue
				}
			}
		}

	}
	//return kerrors.NewAggregate(errs)

	// ignore erros ,pause existing containers as much as possible
	return nil
}

// nerdctlRunKubeletCommand returns the command and arguments to run the kubelet container.
// Deprecated: use newNerdctlCommand instead.
func nerdctlRunKubeletCommand(config map[string]string) (string, []string, error) {
	externalHost, err := bkenet.GetExternalIP()
	if err != nil {
		return "", nil, err
	}
	config["externalHost"] = externalHost
	config["hostName"] = utils.HostName()

	tpl, err := template.New("nerdctl").Parse(nerdctl)
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

func newNerdctlCommand(config map[string]string) (string, []string, error) {
	return NewRunKubeletCommand().
		SetContainerRuntime("containerd").
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
