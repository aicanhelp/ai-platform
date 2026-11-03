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
	"context"
	_ "embed"
	"fmt"
	"strings"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/imagehelper"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

//go:embed tmpl/nerdctl.tmpl
var nerdctl string

const nerdctlCleanKubelet = "nerdctl -n k8s.io ps -a | grep %s | awk '{print $1}' | xargs nerdctl -n k8s.io rm -f"

func (kp *kubeletPlugin) runWithContainerd(config map[string]string) error {
	log.Infof("run kubelet with containerd")

	if err := kp.exec.ExecuteCommand("/bin/sh", "-c", fmt.Sprintf("rm -f %s/cpu_manager_state", config["dataRootDir"])); err != nil {
		log.Warnf("failed to remove cpu_manager_state, err: %v", err)
	}

	if err := kp.ensureImages(config); err != nil {
		return errors.Wrap(err, "failed to ensure images")
	}

	_ = mountList()

	log.Infof("start kubelet container %q", config["containerName"])
	if err := newKubeletScript(config); err != nil {
		return fmt.Errorf("failed to generate kubelet script, err: %w", err)
	}
	runKubeletCommand := fmt.Sprintf("%s -a start -r %s", utils.GetKubeletScriptPath(), "containerd")
	if output, err := kp.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", runKubeletCommand); err != nil {
		return errors.Wrapf(err, "failed to run kubelet container %q, output: %s", config["containerName"], output)
	}

	return nil
}

// processContainerForRemoval processes a single container for removal
func (kp *kubeletPlugin) processContainerForRemoval(ctx context.Context, container containerd.Container, images []string, imagesNameList []string, errs *[]error) {
	containerID := container.ID()

	if ok, _ := kp.containerd.EnsureContainerRun(containerID); ok {
		if err := kp.containerd.Stop(containerID); err != nil {
			*errs = append(*errs, err)
		}
	}

	image, err := container.Image(ctx)
	if err != nil {
		*errs = append(*errs, err)
		return
	}

	if utils.ContainsString(images, image.Name()) {
		if err := kp.containerd.Delete(containerID); err != nil {
			*errs = append(*errs, err)
		}
		return
	}

	labels, err := container.Labels(ctx)
	if err != nil {
		*errs = append(*errs, err)
		return
	}

	v, ok := labels["io.kubernetes.container.name"]
	if !ok {
		return
	}

	if !utils.ContainsString(imagesNameList, v) {
		return
	}

	if err := kp.containerd.Delete(containerID); err != nil {
		*errs = append(*errs, err)
	}
}

func (kp *kubeletPlugin) removeKubeletContainers(config map[string]string) error {
	ctx := context.Background()
	ctx = namespaces.WithNamespace(ctx, "k8s.io")
	containerList, err := kp.containerd.ContainerList()
	if err != nil {
		return err
	}

	repo, clusterVersion := config["imageRepo"], config["kubernetesVersion"]
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

	var errs []error
	for _, container := range containerList {
		kp.processContainerForRemoval(ctx, container, images, imagesNameList, &errs)
	}

	return nil
}

// nerdctlRunKubeletCommand returns the command and arguments to run the kubelet container.
// Deprecated: use newNerdctlCommand instead.
func nerdctlRunKubeletCommand(config map[string]string) (string, []string, error) {
	return runKubeletCommandFromTemplate(config, "nerdctl", nerdctl)
}

func newNerdctlCommand(config map[string]string) (string, []string, error) {
	return newKubeletCommand(config, "containerd")
}
