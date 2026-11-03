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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	serializerjson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func snapshotTestPolicy(t *testing.T, enabled bool) *v1.NPUClusterPolicy {
	t.Helper()
	p := getInstance()
	data, err := json.Marshal(map[string]any{"containerSnapshot": map[string]bool{"enabled": enabled}})
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(data, &p.Spec.NodeD))
	p.Spec.NodeD.Image = v1.ImageSpec{Repository: "noded", Tag: "v26.1.0"}
	return p
}

func nodeDTestTemplate(t *testing.T) *appsv1.DaemonSet {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "assets", "noded", "0400_daemonset.yaml"))
	require.NoError(t, err)
	ds := &appsv1.DaemonSet{}
	require.NoError(t, yaml.Unmarshal(data, ds))
	return ds
}

func TestNodeDSnapshotConfiguration(t *testing.T) {
	p := snapshotTestPolicy(t, true)
	_, r := getManagerAndReconciler(nil, p)
	r.hasNPUNodes = true
	base := nodeDTestTemplate(t)
	ds := base.DeepCopy()
	require.NoError(t, newDaemonSetReconcileHooks(ds).transform(context.Background(), &r, ds))
	require.True(t, ds.Spec.Template.Spec.HostPID)
	require.True(t, ds.Spec.Template.Spec.HostNetwork)
	c := ds.Spec.Template.Spec.Containers[0]
	require.NotEmpty(t, c.Args)
	require.NotNil(t, c.SecurityContext)
	require.True(t, *c.SecurityContext.Privileged)
	require.True(t, c.SecurityContext.ReadOnlyRootFilesystem == nil || !*c.SecurityContext.ReadOnlyRootFilesystem)
	require.Equal(t, base.Spec.Template.Spec.Containers[0].SecurityContext, c.SecurityContext)
	require.Contains(t, c.Env, corev1.EnvVar{Name: "PATH", Value: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
	require.Len(t, ds.Spec.Template.Spec.Volumes, len(base.Spec.Template.Spec.Volumes)+16)
	expectedPaths := map[string]string{
		"dev-shm": "/dev/shm", "image-path": "/user/snapshot",
		"ascend-docker-runtime": "/usr/local/Ascend/Ascend-Docker-Runtime",
		"runtime-log":           "/var/log/ascend-docker-runtime", "containerd": "/run/containerd",
		"containerd-lib": "/var/lib/containerd", "runc": "/usr/local/bin/runc",
		"criu": "/usr/sbin/criu", "npu-plugin": "/usr/lib/criu", "cgroup": "/sys/fs/cgroup",
		"systemd": "/run/systemd", "tmp": "/tmp", "run": "/run", "dcmi": "/usr/local/dcmi",
		"ascend-driver-include": "/usr/local/Ascend/driver/include", "var-kubelet": "/var/lib/kubelet",
	}
	for name, path := range expectedPaths {
		found := false
		for _, volume := range ds.Spec.Template.Spec.Volumes {
			if volume.Name == name {
				require.NotNil(t, volume.HostPath)
				require.Equal(t, path, volume.HostPath.Path)
				found = true
			}
		}
		require.True(t, found, "missing volume %s", name)
		found = false
		for _, mount := range c.VolumeMounts {
			if mount.Name == name {
				require.Equal(t, path, mount.MountPath)
				found = true
			}
		}
		require.True(t, found, "missing mount %s", name)
	}
	for _, mount := range c.VolumeMounts {
		if mount.Name == "containerd" {
			require.NotNil(t, mount.MountPropagation)
			require.Equal(t, corev1.MountPropagationHostToContainer, *mount.MountPropagation)
		}
		if mount.Name == "containerd-lib" {
			require.True(t, mount.ReadOnly)
		}
	}
	// Each reconcile starts from an unmodified asset, including after disabling.
	r.instance = snapshotTestPolicy(t, false)
	disabled := base.DeepCopy()
	require.NoError(t, newDaemonSetReconcileHooks(disabled).transform(context.Background(), &r, disabled))
	require.False(t, disabled.Spec.Template.Spec.HostPID)
	require.Len(t, disabled.Spec.Template.Spec.Volumes, len(base.Spec.Template.Spec.Volumes))
	require.False(t, base.Spec.Template.Spec.HostPID)
}

func TestNodeDSnapshotUserOverrides(t *testing.T) {
	p := snapshotTestPolicy(t, true)
	p.Spec.NodeD.Env = []corev1.EnvVar{{Name: "PATH", Value: "/custom/bin:/usr/bin"}}
	p.Spec.NodeD.VolumeSpec.Volumes = []corev1.Volume{{Name: "runc", VolumeSource: corev1.VolumeSource{
		HostPath: &corev1.HostPathVolumeSource{Path: "/usr/bin/runc"},
	}}}
	_, r := getManagerAndReconciler(nil, p)
	r.hasNPUNodes = true
	ds := nodeDTestTemplate(t)
	require.NoError(t, newDaemonSetReconcileHooks(ds).transform(context.Background(), &r, ds))
	count := 0
	for _, v := range ds.Spec.Template.Spec.Volumes {
		if v.Name == "runc" {
			count++
			require.Equal(t, "/usr/bin/runc", v.HostPath.Path)
		}
	}
	require.Equal(t, 1, count)
	require.Contains(t, ds.Spec.Template.Spec.Containers[0].Env, p.Spec.NodeD.Env[0])
}

func TestComponentRegistration(t *testing.T) {
	p := getInstance()
	require.NoError(t, json.Unmarshal([]byte(`{"inferOperator":{"managed":true},"rdmaDevicePlugin":{"managed":true}}`), &p.Spec))
	r := &NPUClusterPolicyReconciler{instance: p}
	for _, name := range []string{"infer-operator", "rdma-device-plugin"} {
		path := componentPathPrefix + "/" + name
		require.Contains(t, resolveComponentPaths(&p.Spec), path)
		managed, ok := isComponentManaged[path]
		require.True(t, ok)
		require.True(t, managed(r))
		r.instance = getInstance()
		require.False(t, managed(r))
		r.instance = p
	}
	paths := resolveComponentPaths(&p.Spec)
	require.Equal(t, []string{inferOperatorComponent, rdmaDevicePluginComponent}, paths[len(paths)-2:])
}

func TestComponentConfigMapOverrides(t *testing.T) {
	p := getInstance()
	require.NoError(t, json.Unmarshal([]byte(`{"inferOperator":{"config":{"fault_handling":"false"}},"rdmaDevicePlugin":{"config":{"config.json":"{\"periodicUpdateInterval\":60,\"configList\":[]}"}}}`), &p.Spec))
	_, r := getManagerAndReconciler(nil, p)
	for _, tt := range []struct{ name, key, want string }{
		{"infer-operator-config", "fault_handling", "false"},
		{"rdma-devices", "config.json", `{"periodicUpdateInterval":60,"configList":[]}`},
	} {
		cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: tt.name}, Data: map[string]string{tt.key: "default"}}
		require.NoError(t, (configMapReconcileHooks{}).transform(context.Background(), &r, cm))
		require.Equal(t, tt.want, cm.Data[tt.key])
	}
}

func TestNodeDSnapshotRBAC(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "assets", "noded", "0200_clusterrrole.yaml"))
	require.NoError(t, err)
	base := &rbacv1.ClusterRole{}
	require.NoError(t, yaml.Unmarshal(data, base))
	require.Contains(t, base.Rules, rbacv1.PolicyRule{
		APIGroups: []string{""}, Resources: []string{"nodes"}, Verbs: []string{"get", "list", "patch"},
	})
	require.Contains(t, base.Rules, rbacv1.PolicyRule{
		APIGroups: []string{""}, Resources: []string{"pods"}, Verbs: []string{"get", "list", "watch", "patch"},
	})
	require.Contains(t, base.Rules, rbacv1.PolicyRule{
		APIGroups: []string{"apps"}, Resources: []string{"daemonsets"}, Verbs: []string{"get"},
	})
	for _, enabled := range []bool{true, false} {
		_, r := getManagerAndReconciler(nil, snapshotTestPolicy(t, enabled))
		role := base.DeepCopy()
		require.NoError(t, (clusterRoleReconcileHooks{}).transform(context.Background(), &r, role))
		require.Equal(t, base.Rules, role.Rules)
	}
}

func loadTestComponent(t *testing.T, r *NPUClusterPolicyReconciler, name string) component {
	t.Helper()
	comp := component{name: componentPathPrefix + "/" + name}
	serializer := serializerjson.NewSerializerWithOptions(serializerjson.DefaultMetaFactory, r.Scheme, r.Scheme,
		serializerjson.SerializerOptions{Yaml: true})
	files, err := filepath.Glob(filepath.Join("..", "..", "assets", name, "*.yaml"))
	require.NoError(t, err)
	require.NotEmpty(t, files)
	for _, file := range files {
		data, err := os.ReadFile(file)
		require.NoError(t, err)
		require.NoError(t, comp.loadResource(context.Background(), data, serializer))
	}
	return comp
}

func TestComponentWorkloadOverrides(t *testing.T) {
	for _, name := range []string{"infer-operator", "rdma-device-plugin"} {
		t.Run(name, func(t *testing.T) {
			p := getInstance()
			conf := &p.Spec.InferOperator.ComponentWorkloadSpec
			if name == "rdma-device-plugin" {
				conf = &p.Spec.RDMADevicePlugin.ComponentWorkloadSpec
			}
			conf.Managed = true
			conf.Image = v1.ImageSpec{Repository: "registry.example/custom", Tag: "v26.1.0", ImagePullPolicy: "Always", ImagePullSecrets: []string{"registry-auth"}}
			conf.Command = v1.CommandSpec{Command: []string{"custom-binary"}, Args: []string{"--custom"}}
			conf.Resources.Requests = corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("250m")}
			conf.Env = []corev1.EnvVar{{Name: "CUSTOM", Value: "value"}}
			conf.NodeSelector = map[string]string{"pool": "inference"}
			conf.Tolerations = []corev1.Toleration{{Key: "dedicated", Operator: corev1.TolerationOpExists}}
			_, r := getManagerAndReconciler(nil, p)
			r.hasNPUNodes = true
			comp := loadTestComponent(t, &r, name)
			for _, res := range comp.resources {
				var pod *corev1.PodSpec
				switch res := res.(type) {
				case typedComponentResource[appsv1.Deployment, *appsv1.Deployment]:
					require.NoError(t, res.transform(context.Background(), &r, res.obj))
					pod = &res.obj.Spec.Template.Spec
				case typedComponentResource[appsv1.DaemonSet, *appsv1.DaemonSet]:
					require.NoError(t, res.transform(context.Background(), &r, res.obj))
					pod = &res.obj.Spec.Template.Spec
				default:
					continue
				}
				require.Equal(t, "registry.example/custom:v26.1.0", pod.Containers[0].Image)
				require.Equal(t, pod.Containers[0].Image, pod.InitContainers[0].Image)
				require.Equal(t, corev1.PullAlways, pod.InitContainers[0].ImagePullPolicy)
				require.Equal(t, conf.Command.Command, pod.Containers[0].Command)
				require.Equal(t, conf.Command.Args, pod.Containers[0].Args)
				require.Equal(t, []string{"sh", "-c", pod.InitContainers[0].Command[2]}, pod.InitContainers[0].Command)
				require.Contains(t, pod.InitContainers[0].Command[2], "mkdir -p")
				require.Equal(t, "250m", pod.Containers[0].Resources.Requests.Cpu().String())
				require.Contains(t, pod.Containers[0].Env, conf.Env[0])
				require.Equal(t, "inference", pod.NodeSelector["pool"])
				require.Contains(t, pod.Tolerations, conf.Tolerations[0])
				require.Contains(t, pod.ImagePullSecrets, corev1.LocalObjectReference{Name: "registry-auth"})
				require.Len(t, pod.ImagePullSecrets, 1)
			}
		})
	}
}

func TestComponentLifecycle(t *testing.T) {
	ctx := context.Background()
	p := getInstance()
	p.Spec.InferOperator = v1.InferOperatorSpec{ComponentWorkloadSpec: v1.ComponentWorkloadSpec{Managed: true, Image: v1.ImageSpec{Repository: "infer-operator:v26.1.0"}}}
	p.Spec.RDMADevicePlugin = v1.RDMADevicePluginSpec{ComponentWorkloadSpec: v1.ComponentWorkloadSpec{Managed: true, Image: v1.ImageSpec{Repository: "k8s-rdma-shared-dp:v26.1.0"}}}
	_, r := getManagerAndReconciler(nil, p)
	r.hasNPUNodes = true
	r.components = []component{loadTestComponent(t, &r, "infer-operator"), loadTestComponent(t, &r, "rdma-device-plugin")}
	for i := range r.components {
		_, err := r.components[i].reconcile(ctx, &r)
		require.NoError(t, err)
	}
	cm := &corev1.ConfigMap{}
	key := client.ObjectKey{Name: "infer-operator-config", Namespace: "mindx-dl"}
	require.NoError(t, r.Get(ctx, key, cm))
	require.Equal(t, "true", cm.Data["fault_handling"])
	// CR updates reach existing objects, including the rollout annotation.
	r.instance.Spec.InferOperator.Config = map[string]string{"fault_handling": "false"}
	require.NoError(t, r.Update(ctx, r.instance))
	_, err := r.components[0].reconcile(ctx, &r)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, key, cm))
	require.Equal(t, "false", cm.Data["fault_handling"])
	dep := &appsv1.Deployment{}
	depKey := client.ObjectKey{Name: "infer-operator-manager", Namespace: "mindx-dl"}
	require.NoError(t, r.Get(ctx, depKey, dep))
	previousHash := dep.Spec.Template.Annotations[componentConfigHashAnnotation]
	// Removing an override restores upstream defaults, without mutating cached templates.
	r.instance.Spec.InferOperator.Config = nil
	require.NoError(t, r.Update(ctx, r.instance))
	_, err = r.components[0].reconcile(ctx, &r)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, key, cm))
	require.Equal(t, "true", cm.Data["fault_handling"])
	require.NoError(t, r.Get(ctx, depKey, dep))
	require.NotEqual(t, previousHash, dep.Spec.Template.Annotations[componentConfigHashAnnotation])
	r.instance.Spec.InferOperator.Managed, r.instance.Spec.RDMADevicePlugin.Managed = false, false
	require.NoError(t, r.Update(ctx, r.instance))
	for i := range r.components {
		state, err := r.components[i].reconcile(ctx, &r)
		require.NoError(t, err)
		require.Equal(t, v1.ComponentUnmanaged, state.Phase)
	}
	require.True(t, apierrors.IsNotFound(r.Get(ctx, key, &corev1.ConfigMap{})))
	require.True(t, apierrors.IsNotFound(r.Get(ctx, depKey, &appsv1.Deployment{})))
	require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKey{Name: "rdma-shared-dp-ds", Namespace: "kube-system"}, &appsv1.DaemonSet{})))
	for _, name := range []string{"inferservicesets", "inferservices", "instancesets"} {
		crd := &apiextensionsv1.CustomResourceDefinition{}
		require.NoError(t, r.Get(ctx, client.ObjectKey{Name: name + ".mindcluster.huawei.com"}, crd))
		require.Empty(t, crd.OwnerReferences)
	}
}

func TestComponentRejectsInvalidConfiguration(t *testing.T) {
	for _, tt := range []struct{ binary, key, value string }{
		{"infer-operator", "fault_handling", "yes"},
		{"k8s-rdma-shared-dp", "config.json", "{broken"},
		{"k8s-rdma-shared-dp", "config.json", "null"},
		{"k8s-rdma-shared-dp", "config.json", "[]"},
	} {
		require.Error(t, validateComponentConfig(tt.binary, map[string]string{tt.key: tt.value}))
	}
	_, err := componentDefaultArgs("k8s-rdma-shared-dp", rdmaDevicePluginDefaultHealthPort, v1.LogRotate{Compress: true})
	require.Error(t, err)
	args, err := componentDefaultArgs("infer-operator", inferOperatorDefaultHealthPort,
		v1.LogRotate{LogLevel: v1.LogDebug, Rotate: 12, MaxAge: 14, Compress: true})
	require.NoError(t, err)
	for _, flag := range []string{"-logLevel=-1", "-maxBackups=12", "-maxAge=14", "-isCompress=true", "--healthz-address=11254"} {
		require.Contains(t, args, flag)
	}
}

func TestComponentCommandOverridesAreIndependent(t *testing.T) {
	for _, component := range []struct {
		binary string
		port   int
		get    componentWorkloadConfigGetter
	}{
		{"infer-operator", inferOperatorDefaultHealthPort, inferOperatorWorkloadConfig},
		{"k8s-rdma-shared-dp", rdmaDevicePluginDefaultHealthPort, rdmaDevicePluginWorkloadConfig},
	} {
		for _, command := range []v1.CommandSpec{
			{}, {Command: []string{"custom-binary"}}, {Args: []string{"--custom"}},
		} {
			p := getInstance()
			component.get(&p.Spec).Command = command
			r := &NPUClusterPolicyReconciler{instance: p}
			dep := &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers:     []corev1.Container{{Name: "main"}},
				InitContainers: []corev1.Container{{Name: "init", Command: []string{"sh", "-c", "mkdir -p /var/log"}}},
			}}}}
			require.NoError(t, componentCommandTransformer[appsv1.Deployment](component.get, component.binary, component.port)(context.Background(), r, dep))
			main := dep.Spec.Template.Spec.Containers[0]
			if len(command.Command) > 0 {
				require.Equal(t, command.Command, main.Command)
			} else {
				require.Equal(t, []string{component.binary}, main.Command)
			}
			if len(command.Args) > 0 {
				require.Equal(t, command.Args, main.Args)
			} else {
				require.Contains(t, main.Args, "--enable-healthz=true")
				require.Contains(t, main.Args, "-logLevel=0")
			}
			require.Equal(t, []string{"sh", "-c", "mkdir -p /var/log"}, dep.Spec.Template.Spec.InitContainers[0].Command)
		}
	}
}

func TestComponentRequeuesForSupportingResourceDrift(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		p := getInstance()
		p.Spec.InferOperator.Managed = enabled
		_, r := getManagerAndReconciler(nil, p)
		r.hasNFDLabels, r.hasNPUNodes = true, true
		result, err := r.reconcileComponents(context.Background())
		require.NoError(t, err)
		if enabled {
			require.Equal(t, 3*time.Minute, result.RequeueAfter)
		} else {
			require.Zero(t, result.RequeueAfter)
		}
	}
}
