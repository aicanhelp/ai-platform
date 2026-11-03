// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package v1

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func buildTestPolicy() *NPUClusterPolicy {
	commonVolume := []corev1.Volume{
		{Name: "config", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
	}
	commonMount := []corev1.VolumeMount{
		{Name: "config", MountPath: "/etc/config"},
	}
	commonResources := ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("1"),
			corev1.ResourceMemory: resource.MustParse("1Gi"),
		},
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("500m"),
			corev1.ResourceMemory: resource.MustParse("512Mi"),
		},
	}
	commonLogRotate := LogRotate{
		LogLevel: LogInfo,
		LogFile:  "/var/log/component.log",
		Rotate:   5,
		MaxAge:   7,
		Compress: true,
	}
	commonCommand := CommandSpec{
		Command: []string{"run"},
		Args:    []string{"--debug"},
	}
	commonImage := ImageSpec{
		Registry:         "example.io",
		Repository:       "npu/component",
		Tag:              "v1.0.0",
		ImagePullPolicy:  string(corev1.PullIfNotPresent),
		ImagePullSecrets: []string{"pull-secret"},
	}
	commonEnv := []corev1.EnvVar{{Name: "A", Value: "B"}}
	commonVolumes := VolumesSpec{
		Volumes:      commonVolume,
		VolumeMounts: commonMount,
	}
	commonPlacement := PlacementSpec{
		NodeSelector: map[string]string{"accelerator": "ascend"},
		Tolerations:  []corev1.Toleration{{Key: "npu", Operator: corev1.TolerationOpExists}},
	}
	commonVNPUComponent := testVNPUComponent(commonImage, commonEnv, commonCommand, commonVolumes, commonResources, commonPlacement)
	commonDRAComponent := testDRAComponent(commonImage, commonEnv, commonCommand, commonVolumes, commonResources, commonPlacement)

	return &NPUClusterPolicy{
		TypeMeta: metav1.TypeMeta{
			Kind:       "NPUClusterPolicy",
			APIVersion: GroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        "cluster",
			Labels:      map[string]string{"k1": "v1"},
			Annotations: map[string]string{"a1": "v1"},
			Finalizers:  []string{"protect.npu.openfuyao.com"},
		},
		Spec: NPUClusterPolicySpec{
			Operator: OperatorSpec{
				RuntimeClass: "ascend",
				Image:        commonImage,
			},
			Daemonsets: DaemonsetsSpec{
				Labels:       map[string]string{"daemon": "yes"},
				Annotations:  map[string]string{"anno": "yes"},
				NodeSelector: map[string]string{"daemon": "npu"},
				Tolerations:  []corev1.Toleration{{Key: "k", Operator: corev1.TolerationOpExists}},
				Image:        commonImage,
			},
			Driver: DriverSpec{
				Managed:       true,
				LogRotate:     commonLogRotate,
				Image:         commonImage,
				InitImage:     commonImage,
				Env:           commonEnv,
				PlacementSpec: commonPlacement,
			},
			OCIRuntime: OCIRuntimeSpec{
				Managed:          true,
				Interval:         300,
				Image:            commonImage,
				InitRuntimeImage: commonImage,
				InitConfigImage:  commonImage,
				PlacementSpec:    commonPlacement,
			},
			VCScheduler: SchedulerSpec{
				Managed:            true,
				InitImage:          commonImage,
				Image:              commonImage,
				Env:                commonEnv,
				Command:            commonCommand,
				VolumeSpec:         commonVolumes,
				Resources:          commonResources,
				SchedulerConfigMap: "actions: enqueue",
				PlacementSpec:      commonPlacement,
			},
			VCController: ControllerSpec{
				Managed:       true,
				InitImage:     commonImage,
				Image:         commonImage,
				Env:           commonEnv,
				Command:       commonCommand,
				VolumeSpec:    commonVolumes,
				Resources:     commonResources,
				PlacementSpec: commonPlacement,
			},
			Trainer: TrainerSpec{
				Managed:       true,
				InitImage:     commonImage,
				Image:         commonImage,
				Env:           commonEnv,
				Command:       commonCommand,
				VolumeSpec:    commonVolumes,
				Resources:     commonResources,
				LogRotate:     commonLogRotate,
				PlacementSpec: commonPlacement,
			},
			DevicePlugin: DevicePluginSpec{
				Managed:       true,
				PlacementSpec: commonPlacement,
				InitImage:     commonImage,
				Image:         commonImage,
				Env:           commonEnv,
				Command:       commonCommand,
				VolumeSpec:    commonVolumes,
				Resources:     commonResources,
				LogRotate:     commonLogRotate,
			},
			NodeD: NodeDSpec{
				Managed:           true,
				HeartbeatInterval: 10,
				Image:             commonImage,
				Env:               commonEnv,
				Command:           commonCommand,
				VolumeSpec:        commonVolumes,
				Resources:         commonResources,
				PollInterval:      60,
				LogRotate:         commonLogRotate,
				PlacementSpec:     commonPlacement,
			},
			Exporter: ExporterSpec{
				Managed:       true,
				LogRotate:     commonLogRotate,
				Image:         commonImage,
				Env:           commonEnv,
				Command:       commonCommand,
				VolumeSpec:    commonVolumes,
				Resources:     commonResources,
				PlacementSpec: commonPlacement,
			},
			ClusterD: ClusterDSpec{
				Managed:       true,
				LogRotate:     commonLogRotate,
				Image:         commonImage,
				Env:           commonEnv,
				Command:       commonCommand,
				VolumeSpec:    commonVolumes,
				Resources:     commonResources,
				PlacementSpec: commonPlacement,
			},
			RSController: RSControllerSpec{
				Managed:       true,
				LogRotate:     commonLogRotate,
				Resources:     commonResources,
				Image:         commonImage,
				Command:       commonCommand,
				PlacementSpec: commonPlacement,
			},
			MindIOTFT: MindIOTFTSpec{
				Managed:       true,
				Image:         commonImage,
				Env:           commonEnv,
				PlacementSpec: commonPlacement,
			},
			MindIOACP: MindIOACPSpec{
				Managed:       true,
				Version:       "2.0.0",
				Image:         commonImage,
				Env:           commonEnv,
				PlacementSpec: commonPlacement,
			},
			VNPU: VNPUSpec{
				Managed:               true,
				PlacementSpec:         commonPlacement,
				UpdateStrategy:        "RollingUpdate",
				PriorityClassName:     "system-node-critical",
				NodeModeConfig:        "nodeConfigs: []",
				VirtualTemplateConfig: "vnpus: []",
				DevicePlugin: VNPUDevicePluginSpec{
					Managed:          true,
					Image:            commonImage,
					Env:              commonEnv,
					Command:          commonCommand,
					VolumeSpec:       commonVolumes,
					Resources:        commonResources,
					PlacementSpec:    commonPlacement,
					DeviceSplitCount: 20,
					LoggingConsole:   true,
				},
				ClientUpdate: commonVNPUComponent,
				Exporter: VNPUExporterSpec{
					Managed:       true,
					Image:         commonImage,
					Env:           commonEnv,
					Command:       commonCommand,
					VolumeSpec:    commonVolumes,
					Resources:     commonResources,
					PlacementSpec: commonPlacement,
					Port:          8082,
					HTTPS:         true,
					ServiceMonitor: ServiceMonitorSpec{
						Enabled: true,
					},
				},
			},
			DRA: DRASpec{
				Managed:                       true,
				DriverName:                    "npu.huawei.com",
				DeviceProfile:                 "npu",
				CDIRoot:                       "/etc/cdi",
				HealthcheckPort:               -1,
				EnableHardVNPU:                true,
				KubeletPluginsDirectoryPath:   "/var/lib/kubelet/plugins",
				KubeletRegistrarDirectoryPath: "/var/lib/kubelet/plugins_registry",
				PlacementSpec:                 commonPlacement,
				KubeletPlugin:                 commonDRAComponent,
				DeviceClasses: DRADeviceClassesSpec{
					Managed: true,
					Templates: []DRADeviceClassTemplate{{
						Name:        "npu.huawei.com",
						SelectorCEL: `device.driver == "npu.huawei.com"`,
					}},
				},
				SoftVNPU: DRASoftVNPUSpec{
					Managed:               true,
					PlacementSpec:         commonPlacement,
					ShareCount:            16,
					SoftShareShmDir:       "/shared",
					SoftShareDevConfigDir: "/etc/enpu",
					NPUProfileConfigPath:  "/etc/dra/dra.config",
					ProfileConfig:         "hardIds: []",
					VCANNRTInstaller:      commonDRAComponent,
				},
				Webhook: DRAWebhookSpec{
					DRAComponentSpec:  commonDRAComponent,
					Port:              9443,
					TLSCertFile:       "/tls/tls.crt",
					TLSPrivateKeyFile: "/tls/tls.key",
					CABundle:          "LS0t",
					FailurePolicy:     "Fail",
				},
			},
		},
		Status: NPUClusterPolicyStatus{
			Phase:     PolicyNotReady,
			Namespace: "npu-system",
			Conditions: []metav1.Condition{
				{
					Type:               "Ready",
					Status:             metav1.ConditionFalse,
					Reason:             "Reconciling",
					Message:            "waiting",
					LastTransitionTime: metav1.NewTime(time.Now()),
				},
			},
			ComponentStatuses: []ComponentStatus{
				{
					Name: "driver",
					State: ComponentState{
						Phase:   ComponentDeploying,
						Reason:  "Reconciling",
						Message: "creating",
					},
				},
			},
		},
	}
}

func testVNPUComponent(
	image ImageSpec,
	env []corev1.EnvVar,
	command CommandSpec,
	volumes VolumesSpec,
	resources ResourceRequirements,
	placement PlacementSpec,
) VNPUComponentSpec {
	return VNPUComponentSpec{
		Managed:       true,
		Image:         image,
		Env:           env,
		Command:       command,
		VolumeSpec:    volumes,
		Resources:     resources,
		PlacementSpec: placement,
	}
}

func testDRAComponent(
	image ImageSpec,
	env []corev1.EnvVar,
	command CommandSpec,
	volumes VolumesSpec,
	resources ResourceRequirements,
	placement PlacementSpec,
) DRAComponentSpec {
	return DRAComponentSpec{
		Managed:       true,
		Image:         image,
		Env:           env,
		Command:       command,
		VolumeSpec:    volumes,
		Resources:     resources,
		PlacementSpec: placement,
	}
}

func TestImageSpecName(t *testing.T) {
	tests := []struct {
		name string
		spec ImageSpec
		want string
	}{
		{
			name: "empty repository returns empty",
			spec: ImageSpec{},
			want: "",
		},
		{
			name: "repo only",
			spec: ImageSpec{
				Repository: "npu/operator",
			},
			want: "npu/operator",
		},
		{
			name: "repo and tag",
			spec: ImageSpec{
				Repository: "npu/operator",
				Tag:        "v1",
			},
			want: "npu/operator:v1",
		},
		{
			name: "registry repo tag",
			spec: ImageSpec{
				Registry:   "example.io",
				Repository: "npu/operator",
				Tag:        "v1",
			},
			want: "example.io/npu/operator:v1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.spec.Name())
		})
	}
}

func TestLogLevelValue(t *testing.T) {
	assert.Equal(t, LogDebugValue, LogDebug.Value())
	assert.Equal(t, LogInfoValue, LogInfo.Value())
	assert.Equal(t, LogWarningValue, LogWarning.Value())
	assert.Equal(t, LogErrorValue, LogError.Value())
	assert.Equal(t, LogCriticalValue, LogCritical.Value())
	assert.Panics(t, func() {
		LogLevel("unexpected").Value()
	})
}

func TestNPUClusterPolicyStatusMethods(t *testing.T) {
	status := &NPUClusterPolicyStatus{
		Conditions: []metav1.Condition{{
			Type:    "Ready",
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciled",
			Message: "ok",
		}},
	}

	changed := status.UpdateConditions("Ready", "Error", "ReconcileFailed", "broken")
	require.True(t, changed)
	assert.Equal(t, metav1.ConditionFalse, status.Conditions[0].Status)

	found := false
	for _, c := range status.Conditions {
		if c.Type == "Error" {
			found = true
			assert.Equal(t, metav1.ConditionTrue, c.Status)
			assert.Equal(t, "ReconcileFailed", c.Reason)
			assert.Equal(t, "broken", c.Message)
		}
	}
	assert.True(t, found)

	state := ComponentState{Phase: ComponentDeploying, Reason: "Reconciling", Message: "creating"}
	assert.True(t, status.UpdateComponentStatus("driver", state))
	assert.False(t, status.UpdateComponentStatus("driver", state))

	updatedState := ComponentState{Phase: ComponentRunning, Reason: "Ready", Message: "running"}
	assert.True(t, status.UpdateComponentStatus("driver", updatedState))

	driverStatus := status.GetComponentStatus("driver")
	require.NotNil(t, driverStatus)
	assert.Equal(t, ComponentRunning, driverStatus.State.Phase)
	assert.Equal(t, ComponentDeploying, driverStatus.PrevState.Phase)
	assert.Nil(t, status.GetComponentStatus("unknown"))
}

func TestDeepCopyMethods(t *testing.T) {
	policy := buildTestPolicy()

	copied := policy.DeepCopy()
	require.NotNil(t, copied)
	assert.NotSame(t, policy, copied)

	copied.Spec.Daemonsets.Labels["daemon"] = "changed"
	copied.Spec.Daemonsets.NodeSelector["daemon"] = "changed"
	copied.Spec.Driver.Env[0].Value = "changed"
	copied.Spec.Driver.NodeSelector["accelerator"] = "changed"
	copied.Spec.VCController.NodeSelector["accelerator"] = "changed"
	copied.Spec.VCScheduler.Resources.Limits[corev1.ResourceCPU] = resource.MustParse("2")
	copied.Spec.VNPU.NodeSelector["accelerator"] = "changed"
	copied.Spec.VNPU.DevicePlugin.Resources.Limits[corev1.ResourceCPU] = resource.MustParse("3")
	copied.Spec.DRA.DeviceClasses.Templates[0].SelectorCEL = "changed"
	copied.Spec.DRA.SoftVNPU.NodeSelector["accelerator"] = "changed"
	copied.Spec.DRA.Webhook.NodeSelector["accelerator"] = "changed"
	copied.Status.ComponentStatuses[0].State.Reason = "changed"

	assert.Equal(t, "yes", policy.Spec.Daemonsets.Labels["daemon"])
	assert.Equal(t, "npu", policy.Spec.Daemonsets.NodeSelector["daemon"])
	assert.Equal(t, "B", policy.Spec.Driver.Env[0].Value)
	assert.Equal(t, "ascend", policy.Spec.Driver.NodeSelector["accelerator"])
	assert.Equal(t, "ascend", policy.Spec.VCController.NodeSelector["accelerator"])
	originCPU := policy.Spec.VCScheduler.Resources.Limits[corev1.ResourceCPU]
	assert.Equal(t, "1", originCPU.String())
	assert.Equal(t, "ascend", policy.Spec.VNPU.NodeSelector["accelerator"])
	vnpuCPU := policy.Spec.VNPU.DevicePlugin.Resources.Limits[corev1.ResourceCPU]
	assert.Equal(t, "1", vnpuCPU.String())
	assert.Equal(t, `device.driver == "npu.huawei.com"`, policy.Spec.DRA.DeviceClasses.Templates[0].SelectorCEL)
	assert.Equal(t, "ascend", policy.Spec.DRA.SoftVNPU.NodeSelector["accelerator"])
	assert.Equal(t, "ascend", policy.Spec.DRA.Webhook.NodeSelector["accelerator"])
	assert.Equal(t, "Reconciling", policy.Status.ComponentStatuses[0].State.Reason)

	list := &NPUClusterPolicyList{Items: []NPUClusterPolicy{*policy}}
	listCopied := list.DeepCopy()
	require.NotNil(t, listCopied)
	assert.Equal(t, "cluster", list.Items[0].Name)
	listCopied.Items[0].Name = "cluster-copy"
	assert.Equal(t, "cluster", list.Items[0].Name)

	require.IsType(t, &NPUClusterPolicy{}, policy.DeepCopyObject())
	require.IsType(t, &NPUClusterPolicyList{}, list.DeepCopyObject())

	spec := policy.Spec
	status := policy.Status
	clusterD := policy.Spec.ClusterD
	command := CommandSpec{Command: []string{"a"}, Args: []string{"b"}}
	componentState := ComponentState{Phase: ComponentPending}
	componentStatus := ComponentStatus{Name: "driver", State: componentState}
	controllerSpec := policy.Spec.VCController
	daemonsets := policy.Spec.Daemonsets
	devicePlugin := policy.Spec.DevicePlugin
	driver := policy.Spec.Driver
	exporter := policy.Spec.Exporter
	image := policy.Spec.Driver.Image
	logRotate := policy.Spec.Driver.LogRotate
	mindioACP := policy.Spec.MindIOACP
	mindioTFT := policy.Spec.MindIOTFT
	nodeD := policy.Spec.NodeD
	ociRuntime := policy.Spec.OCIRuntime
	operator := policy.Spec.Operator
	rsController := policy.Spec.RSController
	resources := policy.Spec.Trainer.Resources
	scheduler := policy.Spec.VCScheduler
	trainer := policy.Spec.Trainer
	volumes := policy.Spec.Trainer.VolumeSpec
	vnpu := policy.Spec.VNPU
	vnpuComponent := policy.Spec.VNPU.ClientUpdate
	vnpuDevicePlugin := policy.Spec.VNPU.DevicePlugin
	vnpuExporter := policy.Spec.VNPU.Exporter
	dra := policy.Spec.DRA
	draComponent := policy.Spec.DRA.KubeletPlugin
	draDeviceClasses := policy.Spec.DRA.DeviceClasses
	draSoftVNPU := policy.Spec.DRA.SoftVNPU
	draWebhook := policy.Spec.DRA.Webhook
	placement := policy.Spec.VNPU.PlacementSpec
	serviceMonitor := policy.Spec.VNPU.Exporter.ServiceMonitor

	require.NotNil(t, clusterD.DeepCopy())
	require.NotNil(t, command.DeepCopy())
	require.NotNil(t, componentState.DeepCopy())
	require.NotNil(t, componentStatus.DeepCopy())
	require.NotNil(t, controllerSpec.DeepCopy())
	require.NotNil(t, daemonsets.DeepCopy())
	require.NotNil(t, devicePlugin.DeepCopy())
	require.NotNil(t, driver.DeepCopy())
	require.NotNil(t, exporter.DeepCopy())
	require.NotNil(t, image.DeepCopy())
	require.NotNil(t, logRotate.DeepCopy())
	require.NotNil(t, mindioACP.DeepCopy())
	require.NotNil(t, mindioTFT.DeepCopy())
	require.NotNil(t, spec.DeepCopy())
	require.NotNil(t, status.DeepCopy())
	require.NotNil(t, nodeD.DeepCopy())
	require.NotNil(t, ociRuntime.DeepCopy())
	require.NotNil(t, operator.DeepCopy())
	require.NotNil(t, rsController.DeepCopy())
	require.NotNil(t, resources.DeepCopy())
	require.NotNil(t, scheduler.DeepCopy())
	require.NotNil(t, trainer.DeepCopy())
	require.NotNil(t, volumes.DeepCopy())
	require.NotNil(t, vnpu.DeepCopy())
	require.NotNil(t, vnpuComponent.DeepCopy())
	require.NotNil(t, vnpuDevicePlugin.DeepCopy())
	require.NotNil(t, vnpuExporter.DeepCopy())
	require.NotNil(t, dra.DeepCopy())
	require.NotNil(t, draComponent.DeepCopy())
	require.NotNil(t, draDeviceClasses.DeepCopy())
	require.NotNil(t, draSoftVNPU.DeepCopy())
	require.NotNil(t, draWebhook.DeepCopy())
	require.NotNil(t, placement.DeepCopy())
	require.NotNil(t, serviceMonitor.DeepCopy())

	var nilClusterD *ClusterDSpec
	var nilCommand *CommandSpec
	var nilComponentState *ComponentState
	var nilComponentStatus *ComponentStatus
	var nilController *ControllerSpec
	var nilDaemonsets *DaemonsetsSpec
	var nilDevicePlugin *DevicePluginSpec
	var nilDriver *DriverSpec
	var nilExporter *ExporterSpec
	var nilImage *ImageSpec
	var nilLogRotate *LogRotate
	var nilMindioACP *MindIOACPSpec
	var nilMindioTFT *MindIOTFTSpec
	var nilPolicy *NPUClusterPolicy
	var nilPolicyList *NPUClusterPolicyList
	var nilSpec *NPUClusterPolicySpec
	var nilStatus *NPUClusterPolicyStatus
	var nilNodeD *NodeDSpec
	var nilOCIRuntime *OCIRuntimeSpec
	var nilOperator *OperatorSpec
	var nilRSController *RSControllerSpec
	var nilResources *ResourceRequirements
	var nilScheduler *SchedulerSpec
	var nilTrainer *TrainerSpec
	var nilVolumes *VolumesSpec
	var nilVNPU *VNPUSpec
	var nilVNPUComponent *VNPUComponentSpec
	var nilVNPUDevicePlugin *VNPUDevicePluginSpec
	var nilVNPUExporter *VNPUExporterSpec
	var nilDRA *DRASpec
	var nilDRAComponent *DRAComponentSpec
	var nilDRADeviceClasses *DRADeviceClassesSpec
	var nilDRASoftVNPU *DRASoftVNPUSpec
	var nilDRAWebhook *DRAWebhookSpec
	var nilPlacement *PlacementSpec
	var nilServiceMonitor *ServiceMonitorSpec

	assertNilDeepCopy(t, nilClusterD, (*ClusterDSpec).DeepCopy)
	assertNilDeepCopy(t, nilCommand, (*CommandSpec).DeepCopy)
	assertNilDeepCopy(t, nilComponentState, (*ComponentState).DeepCopy)
	assertNilDeepCopy(t, nilComponentStatus, (*ComponentStatus).DeepCopy)
	assertNilDeepCopy(t, nilController, (*ControllerSpec).DeepCopy)
	assertNilDeepCopy(t, nilDaemonsets, (*DaemonsetsSpec).DeepCopy)
	assertNilDeepCopy(t, nilDevicePlugin, (*DevicePluginSpec).DeepCopy)
	assertNilDeepCopy(t, nilDriver, (*DriverSpec).DeepCopy)
	assertNilDeepCopy(t, nilExporter, (*ExporterSpec).DeepCopy)
	assertNilDeepCopy(t, nilImage, (*ImageSpec).DeepCopy)
	assertNilDeepCopy(t, nilLogRotate, (*LogRotate).DeepCopy)
	assertNilDeepCopy(t, nilMindioACP, (*MindIOACPSpec).DeepCopy)
	assertNilDeepCopy(t, nilMindioTFT, (*MindIOTFTSpec).DeepCopy)
	assertNilDeepCopy(t, nilPolicy, (*NPUClusterPolicy).DeepCopy)
	assertNilDeepCopy(t, nilPolicyList, (*NPUClusterPolicyList).DeepCopy)
	assertNilDeepCopy(t, nilSpec, (*NPUClusterPolicySpec).DeepCopy)
	assertNilDeepCopy(t, nilStatus, (*NPUClusterPolicyStatus).DeepCopy)
	assertNilDeepCopy(t, nilNodeD, (*NodeDSpec).DeepCopy)
	assertNilDeepCopy(t, nilOCIRuntime, (*OCIRuntimeSpec).DeepCopy)
	assertNilDeepCopy(t, nilOperator, (*OperatorSpec).DeepCopy)
	assertNilDeepCopy(t, nilRSController, (*RSControllerSpec).DeepCopy)
	assertNilDeepCopy(t, nilResources, (*ResourceRequirements).DeepCopy)
	assertNilDeepCopy(t, nilScheduler, (*SchedulerSpec).DeepCopy)
	assertNilDeepCopy(t, nilTrainer, (*TrainerSpec).DeepCopy)
	assertNilDeepCopy(t, nilVolumes, (*VolumesSpec).DeepCopy)
	assertNilDeepCopy(t, nilVNPU, (*VNPUSpec).DeepCopy)
	assertNilDeepCopy(t, nilVNPUComponent, (*VNPUComponentSpec).DeepCopy)
	assertNilDeepCopy(t, nilVNPUDevicePlugin, (*VNPUDevicePluginSpec).DeepCopy)
	assertNilDeepCopy(t, nilVNPUExporter, (*VNPUExporterSpec).DeepCopy)
	assertNilDeepCopy(t, nilDRA, (*DRASpec).DeepCopy)
	assertNilDeepCopy(t, nilDRAComponent, (*DRAComponentSpec).DeepCopy)
	assertNilDeepCopy(t, nilDRADeviceClasses, (*DRADeviceClassesSpec).DeepCopy)
	assertNilDeepCopy(t, nilDRASoftVNPU, (*DRASoftVNPUSpec).DeepCopy)
	assertNilDeepCopy(t, nilDRAWebhook, (*DRAWebhookSpec).DeepCopy)
	assertNilDeepCopy(t, nilPlacement, (*PlacementSpec).DeepCopy)
	assertNilDeepCopy(t, nilServiceMonitor, (*ServiceMonitorSpec).DeepCopy)
}

func assertNilDeepCopy[T any](t *testing.T, in *T, deepCopy func(*T) *T) {
	t.Helper()
	assert.Nil(t, deepCopy(in))
}
