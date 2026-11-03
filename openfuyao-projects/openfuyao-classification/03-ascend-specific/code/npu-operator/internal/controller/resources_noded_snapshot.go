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
)

// nodeDSnapshotTransformer augments a fresh base template before user overrides.
// Keeping the base asset immutable makes disabling snapshots reversible.
func nodeDSnapshotTransformer(_ context.Context, r *NPUClusterPolicyReconciler, obj *appsv1.DaemonSet) error {
	if !r.instance.Spec.NodeD.ContainerSnapshot.Enabled {
		return nil
	}
	pod := &obj.Spec.Template.Spec
	for i := range pod.Containers {
		container := &pod.Containers[i]
		if container.Name != nodedDaemonsetName {
			continue
		}
		pod.HostPID = true
		container.Env = append(container.Env, corev1.EnvVar{
			Name: "PATH", Value: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
		})
		addNodeDSnapshotVolumes(pod, container)
		return nil
	}
	return fmt.Errorf("NodeD snapshot template is missing container %q", nodedDaemonsetName)
}

func addNodeDSnapshotVolumes(pod *corev1.PodSpec, container *corev1.Container) {
	// Paths and types follow noded-container-snapshot-v26.1.1.yaml.
	for _, mount := range []struct {
		name, path string
		pathType   corev1.HostPathType
	}{
		{"dev-shm", "/dev/shm", corev1.HostPathDirectory},
		{"image-path", "/user/snapshot", corev1.HostPathDirectory},
		{"ascend-docker-runtime", "/usr/local/Ascend/Ascend-Docker-Runtime", corev1.HostPathDirectory},
		{"runtime-log", "/var/log/ascend-docker-runtime", corev1.HostPathDirectory},
		{"containerd", "/run/containerd", corev1.HostPathDirectory},
		{"containerd-lib", "/var/lib/containerd", corev1.HostPathDirectory},
		{"runc", "/usr/local/bin/runc", corev1.HostPathUnset},
		{"criu", "/usr/sbin/criu", corev1.HostPathUnset},
		{"npu-plugin", "/usr/lib/criu", corev1.HostPathDirectory},
		{"cgroup", "/sys/fs/cgroup", corev1.HostPathDirectory},
		{"systemd", "/run/systemd", corev1.HostPathDirectory},
		{"tmp", "/tmp", corev1.HostPathDirectory},
		{"run", "/run", corev1.HostPathDirectory},
		{"dcmi", "/usr/local/dcmi", corev1.HostPathUnset},
		{"ascend-driver-include", "/usr/local/Ascend/driver/include", corev1.HostPathUnset},
		{"var-kubelet", "/var/lib/kubelet", corev1.HostPathUnset},
	} {
		pathType := mount.pathType
		pod.Volumes = append(pod.Volumes, corev1.Volume{
			Name: mount.name,
			VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
				Path: mount.path, Type: &pathType,
			}},
		})
		volumeMount := corev1.VolumeMount{Name: mount.name, MountPath: mount.path}
		if mount.name == "containerd" {
			propagation := corev1.MountPropagationHostToContainer
			volumeMount.MountPropagation = &propagation
		}
		volumeMount.ReadOnly = mount.name == "containerd-lib"
		container.VolumeMounts = append(container.VolumeMounts, volumeMount)
	}
}
