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
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	v1 "openfuyao.com/npu-operator/api/v1"
)

type healthProbeTestCase struct {
	component, asset, binary, options string
	port, delay, period, timeout      int32
}

// Values are taken from the workload containers (not CRD schemas) in the
// 26.1.1 installation package. Keep the assertions independent of the registry.
var healthProbeTestCases = []healthProbeTestCase{
	{"clusterd", "clusterd/0500_deployment.yaml", "clusterd", "-useProxy=false", 11253, 10, 10, 3},
	{"trainer", "trainer/0400_deployment.yaml", "ascend-operator",
		"--kubeApiQps=800 --kubeApiBurst=800 --enableGangScheduling=true", 11252, 10, 10, 3},
	{"devicePlugin", "device-plugin/0400_daemonset.yaml", "device-plugin",
		"-volcanoType=true -presetVirtualDevice=true -useAscendDocker=true", 11251, 30, 15, 5},
	{"nodeD", "noded/0400_daemonset.yaml", "noded", "-reportInterval=5 -monitorPeriod=5", 11255, 20, 15, 5},
	{"exporter", "npu-exporter/0200_daemonset.yaml", "npu-exporter",
		"-port=8082 -ip=0.0.0.0 -containerMode=containerd -updateTime=5 -poll_interval=1s", 11256, 20, 15, 5},
	{"inferOperator", "infer-operator/0800_deployment_infer-operator-manager.yaml", "infer-operator",
		"-isCompress=false -snapshotTimeout=60", 11254, 10, 10, 3},
	{"rdmaDevicePlugin", "rdma-device-plugin/0500_daemonset_rdma-shared-dp-ds.yaml", "k8s-rdma-shared-dp",
		"-config-file=/k8s-rdma-shared-dev-plugin/config.json -use-cdi=false", 11257, 30, 15, 5},
}

func healthProbeTestResource(t *testing.T, asset string) (client.Object, componentResource) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "assets", filepath.FromSlash(asset)))
	require.NoError(t, err)
	var kind struct{ Kind string }
	require.NoError(t, yaml.Unmarshal(data, &kind))
	if kind.Kind == "Deployment" {
		obj := &appsv1.Deployment{}
		require.NoError(t, yaml.Unmarshal(data, obj))
		return obj, resourceCreators[gvkDeployment](obj)
	}
	obj := &appsv1.DaemonSet{}
	require.NoError(t, yaml.Unmarshal(data, obj))
	return obj, resourceCreators[gvkDaemonSet](obj)
}

func healthProbeTestPod(obj client.Object) *corev1.PodSpec {
	switch obj := obj.(type) {
	case *appsv1.Deployment:
		return &obj.Spec.Template.Spec
	case *appsv1.DaemonSet:
		return &obj.Spec.Template.Spec
	default:
		panic("unsupported test workload")
	}
}

func setHealthProbeTestCommand(t *testing.T, spec *v1.NPUClusterPolicySpec, component, script string) {
	t.Helper()
	data, err := json.Marshal(map[string]any{component: map[string]any{
		"imageSpec":   map[string]string{"repository": "test-image", "tag": "v26.1.1"},
		"commandSpec": v1.CommandSpec{Command: []string{"/bin/bash", "-c", "--"}, Args: []string{script}},
	}})
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(data, spec))
	spec.Trainer.InitImage = v1.ImageSpec{Repository: "init-image", Tag: "test"}
	spec.DevicePlugin.InitImage = spec.Trainer.InitImage
}

func TestHealthProbeAllComponentTransitions(t *testing.T) {
	for _, tc := range healthProbeTestCases {
		t.Run(tc.component, func(t *testing.T) {
			runHealthProbeTransitions(t, tc)
		})
	}
}

func runHealthProbeTransitions(t *testing.T, tc healthProbeTestCase) {
	t.Helper()
	ctx := context.Background()
	p := getInstance()
	_, r := getManagerAndReconciler(nil, p)
	r.hasNPUNodes = true
	base, res := healthProbeTestResource(t, tc.asset)
	baseCommand := fmt.Sprintf("%s %s -logFile=/tmp/test.log -logLevel=0", tc.binary, tc.options)
	for _, step := range []struct {
		name, suffix string
		port         int32
		scheme       corev1.URIScheme
	}{
		{"legacy", "", 0, ""},
		{"enable", fmt.Sprintf("--enable-healthz=true --healthz-address=%d", tc.port), tc.port, corev1.URISchemeHTTP},
		{"port change", "--enable-healthz=true --healthz-address=12000", 12000, corev1.URISchemeHTTP},
		{"tls", "--enable-healthz --healthz-address=12001 --tls-cert-file=/tls/cert --tls-private-key-file=/tls/key", 12001, corev1.URISchemeHTTPS},
		{"disable", "--enable-healthz=false --healthz-address=12001", 0, ""},
		{"re-enable", "--enable-healthz --healthz-address=12002", 12002, corev1.URISchemeHTTP},
		{"downgrade", "", 0, ""},
	} {
		t.Run(step.name, func(t *testing.T) {
			setHealthProbeTestCommand(t, &r.instance.Spec, tc.component, baseCommand+" "+step.suffix)
			_, err := res.clone().reconcile(ctx, &r, true)
			require.NoError(t, err)
			actual, ok := base.DeepCopyObject().(client.Object)
			require.True(t, ok, "copied workload must implement client.Object")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), actual))
			pod := healthProbeTestPod(actual)
			probe := pod.Containers[0].LivenessProbe
			require.Equal(t, healthProbeTestPod(base).Containers[0].SecurityContext, pod.Containers[0].SecurityContext)
			if step.port == 0 {
				require.Nil(t, probe)
				return
			}
			require.NotNil(t, probe)
			require.Equal(t, step.port, probe.HTTPGet.Port.IntVal)
			require.Equal(t, step.scheme, probe.HTTPGet.Scheme)
			require.Equal(t, "/", probe.HTTPGet.Path)
			require.Equal(t, tc.delay, probe.InitialDelaySeconds)
			require.Equal(t, tc.period, probe.PeriodSeconds)
			require.Equal(t, tc.timeout, probe.TimeoutSeconds)
			require.EqualValues(t, 3, probe.FailureThreshold)
		})
	}
}

func TestHealthProbeDefaultGeneratedCommands(t *testing.T) {
	for _, tc := range healthProbeTestCases {
		if tc.component != "inferOperator" && tc.component != "rdmaDevicePlugin" {
			continue
		}
		t.Run(tc.component, func(t *testing.T) {
			p := getInstance()
			image := v1.ImageSpec{Repository: "test-image", Tag: "v26.1.1"}
			p.Spec.InferOperator.Image = image
			p.Spec.RDMADevicePlugin.Image = image
			_, r := getManagerAndReconciler(nil, p)
			r.hasNPUNodes = true
			base, res := healthProbeTestResource(t, tc.asset)
			ctx := context.Background()
			_, err := res.clone().reconcile(ctx, &r, true)
			require.NoError(t, err)
			actual, ok := base.DeepCopyObject().(client.Object)
			require.True(t, ok, "copied workload must implement client.Object")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), actual))
			probe := healthProbeTestPod(actual).Containers[0].LivenessProbe
			require.NotNil(t, probe)
			require.Equal(t, tc.port, probe.HTTPGet.Port.IntVal)
		})
	}
}

func TestHealthProbeIsolation(t *testing.T) {
	for _, tc := range healthProbeTestCases {
		t.Run(tc.component, func(t *testing.T) {
			base, _ := healthProbeTestResource(t, tc.asset)
			pod := healthProbeTestPod(base)
			main := &pod.Containers[0]
			main.Command = []string{tc.binary}
			main.Args = []string{"--enable-healthz", "--healthz-address=12000"}
			otherProbe := &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}}}
			main.ReadinessProbe = otherProbe.DeepCopy()
			main.StartupProbe = otherProbe.DeepCopy()
			pod.Containers = append(pod.Containers, corev1.Container{
				Name: "sidecar", LivenessProbe: otherProbe.DeepCopy(),
			})
			before := pod.DeepCopy()
			require.NoError(t, applyWorkloadHealthProbe(context.Background(), pod, healthProbeProfiles[base.GetName()]))
			require.NotNil(t, pod.Containers[0].LivenessProbe)
			// Every field other than the intended main-container liveness probe is preserved.
			pod.Containers[0].LivenessProbe = before.Containers[0].LivenessProbe
			require.Equal(t, before, pod)
			base.SetName("unrelated-workload")
			switch obj := base.(type) {
			case *appsv1.Deployment:
				require.NoError(t, workloadHealthProbeTransformer[appsv1.Deployment](context.Background(), nil, obj))
			case *appsv1.DaemonSet:
				require.NoError(t, workloadHealthProbeTransformer[appsv1.DaemonSet](context.Background(), nil, obj))
			default:
				t.Fatalf("unsupported test workload %T", base)
			}
			require.Equal(t, before, pod)
		})
	}
}
