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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"
)

func TestNodeDSnapshotLifecycle(t *testing.T) {
	ctx := context.Background()
	p := snapshotTestPolicy(t, true)
	userVolume := corev1.Volume{
		Name:         "user-data",
		VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	}
	userMount := corev1.VolumeMount{Name: userVolume.Name, MountPath: "/user-data"}
	userEnv := corev1.EnvVar{Name: "CUSTOM", Value: "preserved"}
	p.Spec.NodeD.VolumeSpec.Volumes = []corev1.Volume{userVolume}
	p.Spec.NodeD.VolumeSpec.VolumeMounts = []corev1.VolumeMount{userMount}
	p.Spec.NodeD.Env = []corev1.EnvVar{userEnv}
	_, r := getManagerAndReconciler(nil, p)
	r.hasNPUNodes = true

	baseDS := nodeDTestTemplate(t)
	data, err := os.ReadFile(filepath.Join("..", "..", "assets", "noded", "0200_clusterrrole.yaml"))
	require.NoError(t, err)
	baseRole := &rbacv1.ClusterRole{}
	require.NoError(t, yaml.Unmarshal(data, baseRole))
	resources := []componentResource{
		resourceCreators[gvkDaemonSet](baseDS),
		resourceCreators[gvkClusterRole](baseRole),
	}

	for _, enabled := range []bool{false, true, false, true, false} {
		stage := "enabled"
		if !enabled {
			stage = "disabled"
		}
		t.Run(stage, func(t *testing.T) {
			r.instance.Spec.NodeD.ContainerSnapshot.Enabled = enabled
			for _, res := range resources {
				_, err := res.clone().reconcile(ctx, &r, true)
				require.NoError(t, err)
			}

			ds := &appsv1.DaemonSet{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(baseDS), ds))
			role := &rbacv1.ClusterRole{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(baseRole), role))
			pod := ds.Spec.Template.Spec
			container := pod.Containers[0]
			require.Contains(t, pod.Volumes, userVolume)
			require.Contains(t, container.VolumeMounts, userMount)
			require.Contains(t, container.Env, userEnv)
			require.NotNil(t, container.SecurityContext)
			require.Equal(t, baseDS.Spec.Template.Spec.Containers[0].SecurityContext, container.SecurityContext)
			require.True(t, pod.HostNetwork)
			require.Equal(t, baseRole.Rules, role.Rules)

			assertNodeDSnapshotState(t, enabled, baseDS.Spec.Template.Spec, pod, userVolume, userMount)
		})
	}
}

func assertNodeDSnapshotState(t *testing.T, enabled bool, base, pod corev1.PodSpec,
	userVolume corev1.Volume, userMount corev1.VolumeMount) {
	t.Helper()
	container := pod.Containers[0]
	if enabled {
		require.True(t, pod.HostPID)
		require.True(t, pod.HostNetwork)
		require.Contains(t, container.Env, corev1.EnvVar{
			Name: "PATH", Value: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
		})
		require.Len(t, pod.Volumes, len(base.Volumes)+17)
		require.Len(t, container.VolumeMounts, len(base.Containers[0].VolumeMounts)+17)
		return
	}

	require.Equal(t, base.HostPID, pod.HostPID)
	require.Equal(t, base.HostNetwork, pod.HostNetwork)
	for _, env := range container.Env {
		require.NotEqual(t, "PATH", env.Name)
	}
	wantVolumes := append(append([]corev1.Volume(nil), base.Volumes...), userVolume)
	wantMounts := append(append([]corev1.VolumeMount(nil), base.Containers[0].VolumeMounts...), userMount)
	require.Equal(t, wantVolumes, pod.Volumes)
	require.Equal(t, wantMounts, container.VolumeMounts)
}
