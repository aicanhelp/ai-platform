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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Component differences stay in this registry; parsing and reconciliation are shared.
// Timings match the actual workload probes in the MindCluster 26.1.1 package.
// Volcano uses a different health endpoint and is intentionally not in this registry.
type healthProbeProfile struct {
	container              string
	binary                 string
	boolFlags              string
	valueFlags             string
	delay, period, timeout int32
}

var healthProbeProfiles = map[string]healthProbeProfile{
	clusterdDeploymentName: {
		container: "clusterd", binary: "clusterd", boolFlags: "useProxy",
		delay: 10, period: 10, timeout: 3,
	},
	trainerDeploymentName: {
		container: "manager", binary: "ascend-operator",
		boolFlags: "isCompress enableGangScheduling", valueFlags: "kubeApiQps kubeApiBurst",
		delay: 10, period: 10, timeout: 3,
	},
	inferOperatorDeploymentName: {
		container: "manager", binary: "infer-operator",
		boolFlags: "isCompress", valueFlags: "snapshotTimeout",
		delay: 10, period: 10, timeout: 3,
	},
	devicePluginDaemonSetName: {
		container: "device-plugin-01", binary: "device-plugin",
		boolFlags: "fdFlag volcanoType autoStowing presetVirtualDevice use310PMixedInsert " +
			"dealWatchHandler checkCachedPods enableSlowNode useSingleDieMode getPodFromKubelet useAscendDocker",
		valueFlags: "edgeLogFile listWatchPeriod hotReset shareDevCount linkdownTimeout " +
			"thirdPartyScanDelay deviceResetTimeout softShareDevConfigDir",
		delay: 30, period: 15, timeout: 5,
	},
	nodedDaemonsetName: {
		container: "noded", binary: "noded",
		valueFlags: "reportInterval monitorPeriod resultMaxAge deviceResetTimeout",
		delay:      20, period: 15, timeout: 5,
	},
	exporterDaemonsetName: {
		container: "npu-exporter", binary: "npu-exporter", boolFlags: "enableLegacyMetrics",
		valueFlags: "port ip updateTime containerMode containerd endpoint concurrency cacheSize " +
			"limitIPConn limitTotalConn limitIPReq platform textMetricsFilePath poll_interval " +
			"profilingTime hccsBWProfilingTime deviceResetTimeout",
		delay: 20, period: 15, timeout: 5,
	},
	rdmaDevicePluginDaemonSetName: {
		container: "k8s-rdma-shared-dp-ds", binary: "k8s-rdma-shared-dp",
		boolFlags: "use-cdi", valueFlags: "config-file",
		delay: 30, period: 15, timeout: 5,
	},
}

// Run after all component transforms, before workload hashing. Unknown workloads
// and other containers retain their existing probes, including readiness/startup.
func workloadHealthProbeTransformer[T appsv1.Deployment | appsv1.DaemonSet](
	ctx context.Context, _ *NPUClusterPolicyReconciler, obj *T) error {
	profile, supported := healthProbeProfiles[any(obj).(client.Object).GetName()]
	if !supported {
		return nil
	}
	pod := newWorkloadPodTemplateSpecGetter[T]()(obj)
	return applyWorkloadHealthProbe(ctx, &pod.Spec, profile)
}

func applyWorkloadHealthProbe(ctx context.Context, pod *corev1.PodSpec, profile healthProbeProfile) error {
	for i := range pod.Containers {
		container := &pod.Containers[i]
		if container.Name != profile.container {
			continue
		}
		probe, reason, err := resolveHealthProbe(profile, container.Command, container.Args)
		if err != nil {
			return fmt.Errorf("configure %s liveness probe: %w", profile.binary, err)
		}
		if reason != "" {
			// Never log command contents: they can contain credentials.
			log.FromContext(ctx).Info("Skipping automatic liveness probe", "component", profile.binary, "reason", reason)
		}
		// Assign nil too, removing static/stale probes when healthz is disabled or the image is downgraded.
		container.LivenessProbe = probe
		return nil
	}
	return fmt.Errorf("%s container not found", profile.container)
}
