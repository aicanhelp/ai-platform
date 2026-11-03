// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"context"
	"fmt"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func trainerLogRotateConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.LogRotate {
	return &spec.Trainer.LogRotate
}

func devicePluginLogRotateConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.LogRotate {
	return &spec.DevicePlugin.LogRotate
}

func nodeDLogRotateConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.LogRotate {
	return &spec.NodeD.LogRotate
}

func exporterLogRotateConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.LogRotate {
	return &spec.Exporter.LogRotate
}

func driverLogRotateConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.LogRotate {
	return &spec.Driver.LogRotate
}

func clusterdLogRotateConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.LogRotate {
	return &spec.ClusterD.LogRotate
}

func rsControllerLogRotateConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.LogRotate {
	return &spec.RSController.LogRotate
}

// component resource
func trainerResourceRequirementsConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.Trainer.Resources
}

func vcSchedulerResourceRequirementsConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.VCScheduler.Resources
}

func vcControllerResourceRequirementsConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.VCController.Resources
}

func devicePluginResourceRequirementsConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.DevicePlugin.Resources
}

func nodeDResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.NodeD.Resources
}

func exporterResourceRequirementsConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.Exporter.Resources
}

func clusterdResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.ClusterD.Resources
}

func rsControllerResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.RSController.Resources
}

func vnpuClientUpdateResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.VNPU.ClientUpdate.Resources
}

func vnpuDevicePluginResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.VNPU.DevicePlugin.Resources
}

func vnpuExporterResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.VNPU.Exporter.Resources
}

func draKubeletPluginResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.DRA.KubeletPlugin.Resources
}

func draVCANNRTInstallerResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.DRA.SoftVNPU.VCANNRTInstaller.Resources
}

func draWebhookResourceConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
	return &spec.DRA.Webhook.Resources
}

// command configuration
func schedulerCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	return &spec.VCScheduler.Command
}

func vcControllerCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	return &spec.VCController.Command
}

func trainerCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	return &spec.Trainer.Command
}

func devicePluginCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	return &spec.DevicePlugin.Command
}

func nodeDCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	if isCommandSpecEmpty(&spec.NodeD.Command) {
		return &v1.CommandSpec{Args: []string{
			"/usr/local/bin/noded -logFile=/var/log/mindx-dl/noded/noded.log -logLevel=0",
		}}
	}
	return &spec.NodeD.Command
}

func exporterCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	return &spec.Exporter.Command
}

func clusterDCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	return &spec.ClusterD.Command
}

func rsControllerCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	return &spec.RSController.Command
}

func vnpuClientUpdateCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	return &spec.VNPU.ClientUpdate.Command
}

func vnpuDevicePluginCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	if !isCommandSpecEmpty(&spec.VNPU.DevicePlugin.Command) {
		return &spec.VNPU.DevicePlugin.Command
	}
	args := []string{}
	args = append(args, fmt.Sprintf("--device-split-count=%d",
		int32OrDefault(spec.VNPU.DevicePlugin.DeviceSplitCount, defaultVNPUDeviceSplitCount)))
	args = append(args, fmt.Sprintf("--logging-console=%t", spec.VNPU.DevicePlugin.LoggingConsole))
	return &v1.CommandSpec{Args: args}
}

func vnpuExporterCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	if !isCommandSpecEmpty(&spec.VNPU.Exporter.Command) {
		return &spec.VNPU.Exporter.Command
	}
	return &v1.CommandSpec{Args: []string{
		"--type=npu",
		"--logging-console=true",
		fmt.Sprintf("--port=%d", int32OrDefault(spec.VNPU.Exporter.Port, defaultVNPUExporterPort)),
	}}
}

func volcanoFlavor(spec *v1.NPUClusterPolicySpec) v1.VolcanoFlavor {
	if spec.Volcano.Flavor == "" {
		return v1.VolcanoFlavorMindCluster
	}
	return spec.Volcano.Flavor
}

func volcanoAdmissionEnabled(spec *v1.NPUClusterPolicySpec) bool {
	if spec.Volcano.Admission.Enabled != nil {
		return *spec.Volcano.Admission.Enabled
	}
	return volcanoFlavor(spec) == v1.VolcanoFlavorVNPU
}

func volcanoAdmissionImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.Volcano.Admission.Image, volcanoAdmissionInitImageConfig(spec)}
}

func volcanoAdmissionMainImageConfigGetter(spec *v1.NPUClusterPolicySpec) *v1.ImageSpec {
	return &spec.Volcano.Admission.Image
}

func volcanoAdmissionInitImageConfig(spec *v1.NPUClusterPolicySpec) *v1.ImageSpec {
	if spec.Volcano.Admission.InitImage.Name() != "" {
		return &spec.Volcano.Admission.InitImage
	}
	if volcanoFlavor(spec) != v1.VolcanoFlavorVNPU {
		return &spec.Volcano.Admission.InitImage
	}
	return &v1.ImageSpec{
		Repository:      stringEnvOrDefault(operatorBusyboxImageRepositoryEnvName, defaultBusyboxImageRepository),
		Tag:             stringEnvOrDefault(operatorBusyboxImageTagEnvName, defaultBusyboxImageTag),
		ImagePullPolicy: stringEnvOrDefault(operatorBusyboxImagePullPolicyEnvName, defaultBusyboxImagePullPolicy),
	}
}

func draDriverName(spec *v1.NPUClusterPolicySpec) string {
	return stringOrDefault(spec.DRA.DriverName, defaultDRADriverName)
}

func draKubeletPluginCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	if !isCommandSpecEmpty(&spec.DRA.KubeletPlugin.Command) {
		return &spec.DRA.KubeletPlugin.Command
	}
	args := []string{
		"--node-name=$(NODE_NAME)",
		fmt.Sprintf("--device-profile=%s", stringOrDefault(spec.DRA.DeviceProfile, defaultDRADeviceProfile)),
		fmt.Sprintf("--driver-name=%s", draDriverName(spec)),
		fmt.Sprintf("--kubelet-registrar-directory-path=%s",
			stringOrDefault(spec.DRA.KubeletRegistrarDirectoryPath, defaultDRAKubeletRegistrarDir)),
		fmt.Sprintf("--kubelet-plugins-directory-path=%s",
			stringOrDefault(spec.DRA.KubeletPluginsDirectoryPath, defaultDRAKubeletPluginsDir)),
		fmt.Sprintf("--cdi-root=%s", stringOrDefault(spec.DRA.CDIRoot, defaultDRACDIRoot)),
		fmt.Sprintf("--healthcheck-port=%d", int32OrDefault(spec.DRA.HealthcheckPort, defaultDRAHealthcheckPort)),
	}
	if spec.DRA.NumDevices != 0 {
		args = append(args, fmt.Sprintf("--num-devices=%d", spec.DRA.NumDevices))
	}
	if spec.DRA.Kubeconfig != "" {
		args = append(args, fmt.Sprintf("--kubeconfig=%s", spec.DRA.Kubeconfig))
	}
	if spec.DRA.KubeAPIQPS != "" {
		args = append(args, fmt.Sprintf("--kube-api-qps=%s", spec.DRA.KubeAPIQPS))
	}
	if spec.DRA.KubeAPIBurst != 0 {
		args = append(args, fmt.Sprintf("--kube-api-burst=%d", spec.DRA.KubeAPIBurst))
	}
	if spec.DRA.FeatureGates != "" {
		args = append(args, fmt.Sprintf("--feature-gates=%s", spec.DRA.FeatureGates))
	}
	if spec.DRA.SoftVNPU.Managed {
		args = append(args, draSoftVNPUArgs(spec)...)
	}
	return &v1.CommandSpec{Args: args}
}

func draSoftVNPUArgs(spec *v1.NPUClusterPolicySpec) []string {
	softVNPU := &spec.DRA.SoftVNPU
	args := []string{
		fmt.Sprintf("--share-count=%d", int32OrDefault(softVNPU.ShareCount, defaultDRASoftVNPUShareCount)),
		fmt.Sprintf("--npu-profile-config=%s",
			stringOrDefault(softVNPU.NPUProfileConfigPath, defaultDRASoftVNPUNPUProfileConfigPath)),
		fmt.Sprintf("--soft-share-dev-config-dir=%s",
			stringOrDefault(softVNPU.SoftShareDevConfigDir, defaultDRASoftVNPUShareDevConfigDir)),
		fmt.Sprintf("--soft-share-shm-dir=%s",
			stringOrDefault(softVNPU.SoftShareShmDir, defaultDRASoftVNPUShareShmDir)),
		fmt.Sprintf("--chip-capabilities-configmap=%s",
			stringOrDefault(softVNPU.ChipCapabilitiesConfigMap, defaultDRASoftVNPUChipCapabilitiesCM)),
	}
	if softVNPU.ChipCapabilitiesNamespace != "" {
		args = append(args, fmt.Sprintf("--chip-capabilities-namespace=%s", softVNPU.ChipCapabilitiesNamespace))
	}
	return args
}

func draVCANNRTInstallerCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	if !isCommandSpecEmpty(&spec.DRA.SoftVNPU.VCANNRTInstaller.Command) {
		return &spec.DRA.SoftVNPU.VCANNRTInstaller.Command
	}
	return &v1.CommandSpec{Command: []string{"/root/acl-client-update.sh"}}
}

func draWebhookCommandParametersGetter(spec *v1.NPUClusterPolicySpec) *v1.CommandSpec {
	if !isCommandSpecEmpty(&spec.DRA.Webhook.Command) {
		return &spec.DRA.Webhook.Command
	}
	webhook := spec.DRA.Webhook
	return &v1.CommandSpec{
		Command: []string{"/ascend-npu-dra-webhook"},
		Args: []string{
			fmt.Sprintf("--tls-cert-file=%s", stringOrDefault(webhook.TLSCertFile, defaultDRAWebhookTLSCertFile)),
			fmt.Sprintf("--tls-private-key-file=%s",
				stringOrDefault(webhook.TLSPrivateKeyFile, defaultDRAWebhookTLSPrivateKeyFile)),
			fmt.Sprintf("--port=%d", int32OrDefault(webhook.Port, defaultDRAWebhookPort)),
			fmt.Sprintf("--device-profile=%s", defaultDRAWebhookDeviceProfile),
			fmt.Sprintf("--driver-name=%s", draDriverName(spec)),
		},
	}
}

func isCommandSpecEmpty(command *v1.CommandSpec) bool {
	return command == nil || (len(command.Command) == 0 && len(command.Args) == 0)
}

// volumes obtain
func devicePluginVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.DevicePlugin.VolumeSpec
}

func nodeDVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.NodeD.VolumeSpec
}

func vcSchedulerVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.VCScheduler.VolumeSpec
}

func vcControllerVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.VCController.VolumeSpec
}

func trainerVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.Trainer.VolumeSpec
}

func exporterVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.Exporter.VolumeSpec
}

func clusterDVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.ClusterD.VolumeSpec
}

func vnpuClientUpdateVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.VNPU.ClientUpdate.VolumeSpec
}

func vnpuDevicePluginVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.VNPU.DevicePlugin.VolumeSpec
}

func vnpuExporterVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.VNPU.Exporter.VolumeSpec
}

func draKubeletPluginVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.DRA.KubeletPlugin.VolumeSpec
}

func draVCANNRTInstallerVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.DRA.SoftVNPU.VCANNRTInstaller.VolumeSpec
}

func draWebhookVolumesGetter(spec *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
	return &spec.DRA.Webhook.VolumeSpec
}

// env obtain
func driverEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.Driver.Env
}

func tftEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.MindIOTFT.Env
}

func acpEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.MindIOACP.Env
}

func vcSchedulerEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.VCScheduler.Env
}

func vcControllerEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.VCController.Env
}

func trainerEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.Trainer.Env
}

func nodeDEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.NodeD.Env
}

func clusterDEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.ClusterD.Env
}

func exporterEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.Exporter.Env
}

func devicePluginEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.DevicePlugin.Env
}

func vnpuClientUpdateEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.VNPU.ClientUpdate.Env
}

func vnpuDevicePluginEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.VNPU.DevicePlugin.Env
}

func vnpuExporterEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.VNPU.Exporter.Env
}

func draKubeletPluginEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.DRA.KubeletPlugin.Env
}

func draVCANNRTInstallerEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.DRA.SoftVNPU.VCANNRTInstaller.Env
}

func draWebhookEnvConfigGetter(spec *v1.NPUClusterPolicySpec) []corev1.EnvVar {
	return spec.DRA.Webhook.Env
}

func newWorkloadVolumesTransformer[T appsv1.DaemonSet | appsv1.Deployment](
	volConfigGetter func(*v1.NPUClusterPolicySpec) *v1.VolumesSpec,
) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]()

	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		volConfig := volConfigGetter(&r.instance.Spec)
		if volConfig == nil || (len(volConfig.Volumes) == 0 && len(volConfig.VolumeMounts) == 0) {
			return nil
		}

		podTmpl := podTemplateSpec(obj)
		if len(volConfig.Volumes) > 0 {
			if podTmpl.Spec.Volumes == nil {
				podTmpl.Spec.Volumes = make([]corev1.Volume, 0)
			}

			existingVolMap := make(map[string]int, len(podTmpl.Spec.Volumes))
			for idx, vol := range podTmpl.Spec.Volumes {
				existingVolMap[vol.Name] = idx
			}

			for _, newVol := range volConfig.Volumes {
				if newVol.Name == "" {
					continue
				}
				if idx, exists := existingVolMap[newVol.Name]; exists {
					podTmpl.Spec.Volumes[idx] = newVol
				} else {
					podTmpl.Spec.Volumes = append(podTmpl.Spec.Volumes, newVol)
				}
			}
		}

		if len(volConfig.VolumeMounts) > 0 {
			for i := range podTmpl.Spec.Containers {
				container := &podTmpl.Spec.Containers[i]

				if container.VolumeMounts == nil {
					container.VolumeMounts = make([]corev1.VolumeMount, 0)
				}

				existingMountMap := make(map[string]int, len(container.VolumeMounts))
				for idx, mount := range container.VolumeMounts {
					existingMountMap[mount.Name] = idx
				}

				for _, newMount := range volConfig.VolumeMounts {
					if newMount.Name == "" {
						continue
					}
					if idx, exists := existingMountMap[newMount.Name]; exists {
						container.VolumeMounts[idx] = newMount
					} else {
						container.VolumeMounts = append(container.VolumeMounts, newMount)
					}
				}
			}
		}

		return nil
	}
}

func newWorkloadCommandParameterTransformer[T appsv1.DaemonSet | appsv1.Deployment](
	command func(*v1.NPUClusterPolicySpec) *v1.CommandSpec) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]()
	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		conf := command(&r.instance.Spec)
		podTmpl := podTemplateSpec(obj)
		containers := podTmpl.Spec.Containers
		for i := range containers {
			c := &containers[i]
			if len(c.Command) == 1 && c.Command[0] == placeholderValue {
				c.Command = conf.Command
			}
			if len(c.Args) == 1 && c.Args[0] == placeholderValue {
				c.Args = conf.Args
			}
		}
		return nil
	}
}

func newVolcanoWorkloadCommandParameterTransformer[T appsv1.Deployment](
	command func(*v1.NPUClusterPolicySpec) *v1.CommandSpec) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]()
	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		conf := command(&r.instance.Spec)
		if isCommandSpecEmpty(conf) {
			return nil
		}
		podTmpl := podTemplateSpec(obj)
		for i := range podTmpl.Spec.Containers {
			if len(conf.Command) > 0 {
				podTmpl.Spec.Containers[i].Command = conf.Command
			}
			if len(conf.Args) > 0 {
				podTmpl.Spec.Containers[i].Args = conf.Args
			}
		}
		return nil
	}
}

func newWorkloadResourceRequirementsTransformer[T appsv1.DaemonSet | appsv1.Deployment](
	resources func(*v1.NPUClusterPolicySpec) *v1.ResourceRequirements) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]()
	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		conf := resources(&r.instance.Spec)
		if conf == nil {
			return nil
		}
		podTmpl := podTemplateSpec(obj)
		containers := podTmpl.Spec.Containers
		for i := range containers {
			c := &containers[i]
			if c.Resources.Limits == nil {
				c.Resources.Limits = corev1.ResourceList{}
			}
			if c.Resources.Requests == nil {
				c.Resources.Requests = corev1.ResourceList{}
			}
			if len(conf.Limits) > 0 {
				for key, value := range conf.Limits {
					c.Resources.Limits[key] = value
				}
			}
			if len(conf.Requests) > 0 {
				for key, value := range conf.Requests {
					c.Resources.Requests[key] = value
				}
			}
		}
		return nil
	}
}

func newVolcanoWorkloadImageTransformer[T appsv1.Deployment](
	image func(*v1.NPUClusterPolicySpec) []*v1.ImageSpec,
) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]()
	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		conf := image(&r.instance.Spec)
		if len(conf) == 0 {
			return nil
		}
		podTmpl := podTemplateSpec(obj)
		applyReleaseTagToPodSpec(&podTmpl.Spec)
		imagePullSecrets := podTmpl.Spec.ImagePullSecrets
		deployName := any(obj).(*appsv1.Deployment).Name
		for i, c := range conf {
			if c == nil {
				continue
			}
			if i == 0 && isInactiveVolcanoFlavorDefaultImage(deployName, volcanoFlavor(&r.instance.Spec), c) {
				continue
			}
			for _, secret := range c.ImagePullSecrets {
				imagePullSecrets = append(imagePullSecrets, corev1.LocalObjectReference{Name: secret})
			}
		}
		podTmpl.Spec.ImagePullSecrets = imagePullSecrets
		mainImage := conf[0]
		if !isInactiveVolcanoFlavorDefaultImage(deployName, volcanoFlavor(&r.instance.Spec), mainImage) {
			for i := range podTmpl.Spec.Containers {
				applyImageSpecToContainer(&podTmpl.Spec.Containers[i], mainImage, true)
			}
		}
		initImages := conf[1:]
		for i, initImage := range initImages {
			if initImage == nil || initImage.Name() == "" {
				continue
			}
			if i >= len(podTmpl.Spec.InitContainers) {
				return fmt.Errorf("init image count %d exceeds init container count %d",
					len(initImages), len(podTmpl.Spec.InitContainers))
			}
			applyImageSpecToContainer(&podTmpl.Spec.InitContainers[i], initImage, true)
		}
		return nil
	}
}

func isInactiveVolcanoFlavorDefaultImage(deploymentName string, activeFlavor v1.VolcanoFlavor, image *v1.ImageSpec) bool {
	repo := imageNameWithoutTag(image.Name())
	if repo == "" {
		return false
	}
	switch activeFlavor {
	case v1.VolcanoFlavorVNPU:
		return isAscendVolcanoDefaultImage(deploymentName, repo)
	default:
		return isVNPUVolcanoDefaultImage(deploymentName, repo)
	}
}

func isAscendVolcanoDefaultImage(deploymentName, repo string) bool {
	switch deploymentName {
	case vcControllerDeploymentName:
		return repo == ascendVCControllerRepo
	case vcSchedulerDeploymentName:
		return repo == ascendVCSchedulerRepo
	default:
		return false
	}
}

func isVNPUVolcanoDefaultImage(deploymentName, repo string) bool {
	switch deploymentName {
	case vcControllerDeploymentName:
		return repo == vnpuVCControllerRepo
	case vcSchedulerDeploymentName:
		return repo == vnpuVCSchedulerRepo
	case volcanoAdmissionDeploymentName:
		return repo == vnpuVolcanoAdmissionRepo
	default:
		return false
	}
}

func imageNameWithoutTag(image string) string {
	lastSlash := strings.LastIndex(image, "/")
	lastColon := strings.LastIndex(image, ":")
	if lastColon > lastSlash {
		return image[:lastColon]
	}
	return image
}

func newWorkloadImageTransformer[T appsv1.DaemonSet | appsv1.Deployment](
	image func(*v1.NPUClusterPolicySpec) []*v1.ImageSpec,
) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]() // 返回每个daemonset或者deployment的spec里的template.spec
	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		conf := image(&r.instance.Spec) // 返回 ImageSpec 这个是CR实例里定义的 镜像地址
		podTmpl := podTemplateSpec(obj) // 这个是 daemonset的yaml里定义的 pod 的 template
		mainImage := conf[0]
		imagePullSecrets := podTmpl.Spec.ImagePullSecrets
		containers := podTmpl.Spec.Containers // containers的参数定义
		for _, c := range conf {
			for _, secret := range c.ImagePullSecrets {
				imagePullSecrets = append(imagePullSecrets, corev1.LocalObjectReference{Name: secret})
			}
		}
		podTmpl.Spec.ImagePullSecrets = imagePullSecrets
		mainImagename := mainImage.Name() // 把CR里的镜像地址拼成一个完整的地址 然后传入到要部署的daemonset中
		for i := range containers {       // 遍历 daemonset 的yaml里的containers的列表
			c := &containers[i]
			if c.Image == placeholderValue {
				if len(mainImagename) == 0 {
					return fmt.Errorf("invalid image name: %v", *mainImage)
				}
				c.Image = mainImagename
				c.ImagePullPolicy = corev1.PullPolicy(mainImage.ImagePullPolicy)
			}
		}
		initImages := conf[1:]
		if len(initImages) > 0 {
			if len(podTmpl.Spec.InitContainers) < len(initImages) {
				return fmt.Errorf("init image count %d exceeds init container count %d",
					len(initImages), len(podTmpl.Spec.InitContainers))
			}
			for i, initImage := range initImages {
				initContainer := &podTmpl.Spec.InitContainers[i]
				if initContainer.Image == placeholderValue {
					name := initImage.Name()
					if len(name) == 0 {
						return fmt.Errorf("invalid init image name at index %d: %v", i, *initImage)
					}
					initContainer.Image = name
					initContainer.ImagePullPolicy = corev1.PullPolicy(initImage.ImagePullPolicy)
				}
			}
		}
		return nil
	}
}

func runtimeImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{
		&spec.OCIRuntime.Image, &spec.OCIRuntime.InitRuntimeImage, &spec.OCIRuntime.InitConfigImage}
}

func vcSchedulerImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.VCScheduler.Image, &spec.VCScheduler.InitImage}
}

func vcControllerImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.VCController.Image, &spec.VCController.InitImage}
}

func trainerImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.Trainer.Image, &spec.Trainer.InitImage}
}

func devicePluginImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{
		&spec.DevicePlugin.Image, &spec.DevicePlugin.InitImage}
}

func nodeDImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.NodeD.Image}
}

func exporterImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.Exporter.Image}
}

func driverImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.Driver.Image, &spec.Driver.InitImage}
}

func clusterdImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.ClusterD.Image}
}

func rscontrollerImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.RSController.Image}
}

func mindioTFTImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.MindIOTFT.Image}
}

func mindioACPImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.MindIOACP.Image}
}

func vnpuClientUpdateImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.VNPU.ClientUpdate.Image}
}

func vnpuDevicePluginImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.VNPU.DevicePlugin.Image}
}

func vnpuExporterImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.VNPU.Exporter.Image}
}

func draKubeletPluginImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.DRA.KubeletPlugin.Image}
}

func draVCANNRTInstallerImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.DRA.SoftVNPU.VCANNRTInstaller.Image}
}

func draWebhookImageConfigGetter(spec *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
	return []*v1.ImageSpec{&spec.DRA.Webhook.Image}
}

func mergePlacement(parent, child *v1.PlacementSpec) *v1.PlacementSpec {
	if parent == nil && child == nil {
		return nil
	}
	merged := &v1.PlacementSpec{}
	if parent != nil {
		merged.NodeSelector = parent.NodeSelector
		merged.Tolerations = parent.Tolerations
	}
	if child != nil {
		if len(child.NodeSelector) > 0 {
			merged.NodeSelector = child.NodeSelector
		}
		if len(child.Tolerations) > 0 {
			merged.Tolerations = child.Tolerations
		}
	}
	return merged
}

func driverPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.Driver.PlacementSpec
}

func runtimePlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.OCIRuntime.PlacementSpec
}

func devicePluginPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.DevicePlugin.PlacementSpec
}

func nodeDPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.NodeD.PlacementSpec
}

func exporterPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.Exporter.PlacementSpec
}

func mindIOTFTPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.MindIOTFT.PlacementSpec
}

func mindIOACPPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.MindIOACP.PlacementSpec
}

func trainerPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.Trainer.PlacementSpec
}

func clusterdPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.ClusterD.PlacementSpec
}

func rsControllerPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.RSController.PlacementSpec
}

func vcControllerPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.VCController.PlacementSpec
}

func vcSchedulerPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return &spec.VCScheduler.PlacementSpec
}

func vnpuClientUpdatePlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return mergePlacement(&spec.VNPU.PlacementSpec, &spec.VNPU.ClientUpdate.PlacementSpec)
}

func vnpuDevicePluginPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return mergePlacement(&spec.VNPU.PlacementSpec, &spec.VNPU.DevicePlugin.PlacementSpec)
}

func vnpuExporterPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return mergePlacement(&spec.VNPU.PlacementSpec, &spec.VNPU.Exporter.PlacementSpec)
}

func draKubeletPluginPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return mergePlacement(&spec.DRA.PlacementSpec, &spec.DRA.KubeletPlugin.PlacementSpec)
}

func draVCANNRTInstallerPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	placement := mergePlacement(&spec.DRA.PlacementSpec, &spec.DRA.SoftVNPU.PlacementSpec)
	return mergePlacement(placement, &spec.DRA.SoftVNPU.VCANNRTInstaller.PlacementSpec)
}

func draWebhookPlacementGetter(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
	return mergePlacement(&spec.DRA.PlacementSpec, &spec.DRA.Webhook.PlacementSpec)
}

func schedulerConfigMapGetter(spec *v1.NPUClusterPolicySpec) string {
	return spec.VCScheduler.SchedulerConfigMap
}
