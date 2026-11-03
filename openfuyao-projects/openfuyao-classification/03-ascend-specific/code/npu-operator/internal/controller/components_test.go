// Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/davecgh/go-spew/spew"
	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	k8sbatchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	v1 "openfuyao.com/npu-operator/api/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

type mockResource struct {
	kind string
}

func (m *mockResource) clone() componentResource {
	return &mockResource{kind: m.kind}
}

func (m *mockResource) gvk() string {
	return m.kind
}

func (m *mockResource) ref(string) componentResourceRef {
	return componentResourceRef{name: m.kind}
}

func (m *mockResource) reconcile(ctx context.Context, r *NPUClusterPolicyReconciler, isManaged bool) (*v1.ComponentState, error) {
	return &v1.ComponentState{Phase: v1.ComponentRunning}, nil
}

func TestReconcilerLoadComponent(t *testing.T) {
	r := &NPUClusterPolicyReconciler{instance: &v1.NPUClusterPolicy{
		Spec: v1.NPUClusterPolicySpec{
			Driver: v1.DriverSpec{Managed: true},
		},
		Status: v1.NPUClusterPolicyStatus{
			ComponentStatuses: []v1.ComponentStatus{{Name: "/var/lib/npu-operator/components/driver", State: v1.ComponentState{Phase: v1.ComponentPending}}},
		},
	},
	}
	test := []struct {
		name      string
		resource  componentResource
		phase     *v1.ComponentState
		stateErr  error
		updateErr error
	}{
		{
			name: "success",
			resource: &mockResource{
				kind: "NameSpace"},
			phase: &v1.ComponentState{Phase: v1.ComponentRunning},
		},
		{
			name: "pending",
			resource: &mockResource{
				kind: "Pod",
			},
			phase:    nil,
			stateErr: fmt.Errorf("component not ready"),
		},
		{
			name: "deploying",
			resource: &mockResource{
				kind: "Pod",
			},
			phase:     &v1.ComponentState{Phase: v1.ComponentDeploying},
			updateErr: fmt.Errorf("uodate state err"),
		},
	}

	for _, tt := range test {
		t.Run(tt.name, func(t *testing.T) {
			comp := component{
				name:      "/var/lib/npu-operator/components/driver",
				resources: []componentResource{tt.resource},
			}
			g := gomonkey.ApplyPrivateMethod(r, "updateStatus", func(_ *NPUClusterPolicyReconciler, _ context.Context, _ updateStatusCallback) error {
				return tt.updateErr
			})
			h := gomonkey.ApplyPrivateMethod(tt.resource, "reconcile", func(_ *mockResource, _ context.Context, _ *NPUClusterPolicyReconciler, _ bool) (*v1.ComponentState, error) {
				return tt.phase, tt.stateErr
			})
			defer g.Reset()
			defer h.Reset()
			state, _ := comp.reconcile(context.TODO(), r)
			if tt.phase == nil {
				assert.Equal(t, v1.ComponentPending, state.Phase)
			} else {
				assert.Equal(t, tt.phase.Phase, state.Phase)
			}

		})
	}
}

func mockConponent() []component {
	comp := make([]component, 0, len(componentPaths))
	for _, path := range componentPaths {
		comp = append(comp, component{name: path})
	}
	return comp
}
func mockReconciler() *NPUClusterPolicyReconciler {
	return &NPUClusterPolicyReconciler{instance: &v1.NPUClusterPolicy{
		Spec: v1.NPUClusterPolicySpec{
			Driver:       v1.DriverSpec{Managed: true},
			OCIRuntime:   v1.OCIRuntimeSpec{Managed: true},
			VCScheduler:  v1.SchedulerSpec{Managed: true},
			VCController: v1.ControllerSpec{Managed: true},
			Volcano:      v1.VolcanoSpec{Flavor: v1.VolcanoFlavorVNPU},
			DevicePlugin: v1.DevicePluginSpec{Managed: true},
			Trainer:      v1.TrainerSpec{Managed: false},
			NodeD:        v1.NodeDSpec{Managed: true},
			ClusterD:     v1.ClusterDSpec{Managed: true},
			RSController: v1.RSControllerSpec{Managed: true},
			Exporter:     v1.ExporterSpec{Managed: true},
			MindIOTFT:    v1.MindIOTFTSpec{Managed: true},
			MindIOACP:    v1.MindIOACPSpec{Managed: true},
			VNPU: v1.VNPUSpec{
				Managed:      true,
				ClientUpdate: v1.VNPUComponentSpec{Managed: true},
				DevicePlugin: v1.VNPUDevicePluginSpec{Managed: true},
				Exporter: v1.VNPUExporterSpec{
					Managed:        true,
					ServiceMonitor: v1.ServiceMonitorSpec{Enabled: true},
				},
			},
			DRA: v1.DRASpec{
				Managed:       true,
				KubeletPlugin: v1.DRAComponentSpec{Managed: true},
				DeviceClasses: v1.DRADeviceClassesSpec{Managed: true},
				SoftVNPU: v1.DRASoftVNPUSpec{
					Managed:          true,
					VCANNRTInstaller: v1.DRAComponentSpec{Managed: true},
				},
				Webhook: v1.DRAWebhookSpec{DRAComponentSpec: v1.DRAComponentSpec{Managed: true}},
			},
		},
	},
		components: mockConponent(),
		namespace:  "default",
		client:     &kubernetes.Clientset{},
		Scheme:     runtime.NewScheme(),
	}
}
func TestGetComponentManagedState(t *testing.T) {
	comp := mockConponent()
	r := mockReconciler()
	for _, c := range comp {
		isManaged := isComponentManaged[c.name](r)
		switch c.name {
		case trainerComponent, vccontrollerComponent, vcschedulerComponent, inferOperatorComponent, rdmaDevicePluginComponent:
			assert.Equal(t, isManaged, false)
		default:
			assert.Equal(t, isManaged, true)
		}
	}

}

func TestGetNewComponentManagedStateHonorsParentAndChild(t *testing.T) {
	r := mockReconciler()
	assert.True(t, isComponentManaged[vnpuNamespaceComponent](r))
	assert.True(t, isComponentManaged[vnpuDevicePluginComponent](r))
	assert.True(t, isComponentManaged[vnpuExporterSMComponent](r))
	assert.True(t, isComponentManaged[volcanoVNPUCRDsComponent](r))
	assert.True(t, isComponentManaged[volcanoVNPUNamespaceComponent](r))
	assert.True(t, isComponentManaged[volcanoVNPUAdmissionComponent](r))
	assert.True(t, isComponentManaged[draKubeletPluginComponent](r))

	r.instance.Spec.DRA.SoftVNPU.Managed = false
	r.instance.Spec.VNPU.Managed = false
	assert.False(t, isComponentManaged[vnpuNamespaceComponent](r))
	assert.False(t, isComponentManaged[vnpuDevicePluginComponent](r))

	r.instance.Spec.DRA.SoftVNPU.Managed = true
	r.instance.Spec.VNPU.Managed = true
	r.instance.Spec.VNPU.DevicePlugin.Managed = false
	assert.False(t, isComponentManaged[vnpuDevicePluginComponent](r))

	r.instance.Spec.VNPU.DevicePlugin.Managed = true
	r.instance.Spec.VNPU.Exporter.ServiceMonitor.Enabled = false
	assert.False(t, isComponentManaged[vnpuExporterSMComponent](r))

	r.instance.Spec.Volcano.Flavor = v1.VolcanoFlavorMindCluster
	assert.False(t, isComponentManaged[volcanoVNPUCRDsComponent](r))
	assert.False(t, isComponentManaged[volcanoVNPUNamespaceComponent](r))
	assert.False(t, isComponentManaged[volcanoVNPUAdmissionComponent](r))

	r.instance.Spec.Volcano.Flavor = v1.VolcanoFlavorVNPU
	disabled := false
	r.instance.Spec.Volcano.Admission.Enabled = &disabled
	assert.True(t, isComponentManaged[volcanoVNPUCRDsComponent](r))
	assert.True(t, isComponentManaged[volcanoVNPUNamespaceComponent](r))
	assert.False(t, isComponentManaged[volcanoVNPUAdmissionComponent](r))

	r.instance.Spec.DRA.Managed = false
	assert.False(t, isComponentManaged[draKubeletPluginComponent](r))

	r.instance.Spec.DRA.Managed = true
	r.instance.Spec.DRA.KubeletPlugin.Managed = false
	assert.False(t, isComponentManaged[draKubeletPluginComponent](r))
}

func TestResolveComponentPathsSelectsVolcanoFlavorAssets(t *testing.T) {
	defaultComponents := resolveComponentPaths(&v1.NPUClusterPolicySpec{})
	assert.Contains(t, defaultComponents, vccontrollerComponent)
	assert.Contains(t, defaultComponents, vcschedulerComponent)
	assert.Contains(t, defaultComponents, volcanoVNPUAdmissionComponent)
	assert.Contains(t, defaultComponents, volcanoVNPUControllerComponent)
	assert.Contains(t, defaultComponents, volcanoVNPUSchedulerComponent)
	assert.NotContains(t, defaultComponents, volcanoVNPUCRDsComponent)
	assert.True(t, slices.Index(defaultComponents, volcanoVNPUControllerComponent) < slices.Index(defaultComponents, vccontrollerComponent))
	assert.True(t, slices.Index(defaultComponents, volcanoVNPUSchedulerComponent) < slices.Index(defaultComponents, vcschedulerComponent))

	vnpu := resolveComponentPaths(&v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorVNPU},
	})
	assert.Contains(t, vnpu, vccontrollerComponent)
	assert.Contains(t, vnpu, vcschedulerComponent)
	assert.Contains(t, vnpu, volcanoVNPUControllerComponent)
	assert.Contains(t, vnpu, volcanoVNPUSchedulerComponent)
	assert.Contains(t, vnpu, volcanoVNPUCRDsComponent)
	assert.True(t, slices.Index(vnpu, vccontrollerComponent) < slices.Index(vnpu, volcanoVNPUControllerComponent))
	assert.True(t, slices.Index(vnpu, vcschedulerComponent) < slices.Index(vnpu, volcanoVNPUSchedulerComponent))
	assert.True(t, slices.Index(vnpu, volcanoVNPUCRDsComponent) < slices.Index(vnpu, volcanoVNPUControllerComponent))
	assert.True(t, slices.Index(vnpu, volcanoVNPUCRDsComponent) < slices.Index(vnpu, volcanoVNPUSchedulerComponent))
	assert.True(t, slices.Index(vnpu, volcanoVNPUControllerComponent) < slices.Index(vnpu, volcanoVNPUAdmissionComponent))
	assert.True(t, slices.Index(vnpu, volcanoVNPUSchedulerComponent) < slices.Index(vnpu, volcanoVNPUAdmissionComponent))

	external := resolveComponentPaths(&v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorExternal},
	})
	assert.Contains(t, external, vccontrollerComponent)
	assert.Contains(t, external, vcschedulerComponent)
	assert.Contains(t, external, volcanoVNPUCRDsComponent)
	assert.Contains(t, external, volcanoVNPUControllerComponent)
	assert.Contains(t, external, volcanoVNPUSchedulerComponent)
	assert.Contains(t, external, volcanoVNPUAdmissionComponent)
}

func TestInactiveVolcanoFlavorDoesNotDeleteActiveFlavorResourceWithSameName(t *testing.T) {
	instance := volcanoFlavorSwitchPolicy(v1.NPUClusterPolicySpec{
		Volcano:      v1.VolcanoSpec{Flavor: v1.VolcanoFlavorMindCluster},
		VCController: v1.ControllerSpec{Managed: true},
	})
	existing := volcanoControllerDeployment(placeholderValue, placeholderValue)
	setVolcanoFlavorOwner(existing, instance)
	r := volcanoFlavorSwitchReconciler(t, instance, existing)
	r.components = volcanoFlavorComponents(
		deploymentResource(volcanoControllerDeployment(placeholderValue, placeholderValue)),
		deploymentResource(volcanoControllerDeployment(placeholderValue, placeholderValue)),
	)
	inactive := r.components[0]

	_, err := inactive.reconcile(context.TODO(), r)

	require.NoError(t, err)
	got := &appsv1.Deployment{}
	assert.NoError(t, r.Get(context.TODO(), client.ObjectKey{
		Namespace: "volcano-system",
		Name:      vcControllerDeploymentName,
	}, got))
}

func TestInactiveVolcanoFlavorDeletesResourceUnusedByActiveFlavor(t *testing.T) {
	instance := volcanoFlavorSwitchPolicy(v1.NPUClusterPolicySpec{
		Volcano:      v1.VolcanoSpec{Flavor: v1.VolcanoFlavorMindCluster},
		VCController: v1.ControllerSpec{Managed: true},
	})
	existing := volcanoControllerService()
	setVolcanoFlavorOwner(existing, instance)
	r := volcanoFlavorSwitchReconciler(t, instance, existing)
	r.components = volcanoFlavorComponents(
		serviceResource(volcanoControllerService()),
		deploymentResource(volcanoControllerDeployment(placeholderValue, placeholderValue)),
	)
	inactive := r.components[0]

	_, err := inactive.reconcile(context.TODO(), r)

	require.NoError(t, err)
	assert.True(t, apierrors.IsNotFound(r.Get(context.TODO(), client.ObjectKey{
		Namespace: "volcano-system",
		Name:      "volcano-controllers-service",
	}, &corev1.Service{})))
}

func TestVolcanoFlavorSwitchUpdatesSharedDeploymentImage(t *testing.T) {
	const (
		oldVNPUImage            = "cr.openfuyao.cn/openfuyao/vnpu-volcano:v1"
		controllerImageRepo     = "docker.io/library/vc_controller"
		controllerImageTag      = "1.15.0"
		controllerInitImageRepo = "docker.io/library/busybox"
	)
	instance := volcanoFlavorSwitchPolicy(ascendVolcanoControllerSpec(
		controllerImageRepo,
		controllerImageTag,
		controllerInitImageRepo,
	))
	existing := volcanoControllerDeployment(oldVNPUImage, oldVNPUImage)
	setVolcanoFlavorOwner(existing, instance)
	r := volcanoFlavorSwitchReconciler(t, instance, existing)
	r.components = volcanoFlavorComponents(
		deploymentResource(volcanoControllerDeployment(placeholderValue, placeholderValue)),
		deploymentResource(volcanoControllerDeployment(placeholderValue, placeholderValue)),
	)
	inactive := r.components[0]
	active := r.components[1]

	_, err := inactive.reconcile(context.TODO(), r)
	require.NoError(t, err)
	_, err = active.reconcile(context.TODO(), r)
	require.NoError(t, err)

	got := &appsv1.Deployment{}
	require.NoError(t, r.Get(context.TODO(), client.ObjectKey{
		Namespace: "volcano-system",
		Name:      vcControllerDeploymentName,
	}, got))
	assert.Equal(t, controllerImageRepo+":"+controllerImageTag, got.Spec.Template.Spec.Containers[0].Image)
	assert.Equal(t, controllerInitImageRepo, got.Spec.Template.Spec.InitContainers[0].Image)
}

func volcanoFlavorSwitchPolicy(spec v1.NPUClusterPolicySpec) *v1.NPUClusterPolicy {
	return &v1.NPUClusterPolicy{
		TypeMeta:   metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: "policy-uid"},
		Spec:       spec,
		Status: v1.NPUClusterPolicyStatus{ComponentStatuses: []v1.ComponentStatus{{
			Name:  volcanoVNPUControllerStatus,
			State: v1.ComponentState{Phase: v1.ComponentRunning},
		}}},
	}
}

func ascendVolcanoControllerSpec(repository, tag, initRepository string) v1.NPUClusterPolicySpec {
	return v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorMindCluster},
		VCController: v1.ControllerSpec{
			Managed: true,
			Image: v1.ImageSpec{
				Repository:      repository,
				Tag:             tag,
				ImagePullPolicy: "IfNotPresent",
			},
			InitImage: v1.ImageSpec{
				Repository:      initRepository,
				ImagePullPolicy: "IfNotPresent",
			},
		},
	}
}

func setVolcanoFlavorOwner(obj client.Object, instance *v1.NPUClusterPolicy) {
	obj.SetOwnerReferences([]metav1.OwnerReference{*metav1.NewControllerRef(
		instance,
		v1.GroupVersion.WithKind("NPUClusterPolicy"),
	)})
}

func volcanoFlavorSwitchReconciler(
	t *testing.T,
	instance *v1.NPUClusterPolicy,
	existing client.Object,
) *NPUClusterPolicyReconciler {
	s := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(s))
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	return &NPUClusterPolicyReconciler{
		Client:    fake.NewClientBuilder().WithScheme(s).WithObjects(existing, instance).WithStatusSubresource(instance).Build(),
		Scheme:    s,
		namespace: "npu-system",
		instance:  instance,
	}
}

func volcanoFlavorComponents(inactiveResource, activeResource componentResource) []component {
	return []component{
		{name: volcanoVNPUControllerComponent, resources: []componentResource{inactiveResource}},
		{name: vccontrollerComponent, resources: []componentResource{activeResource}},
	}
}

func volcanoControllerDeployment(containerImage, initContainerImage string) *appsv1.Deployment {
	labels := map[string]string{"app": vcControllerDeploymentName}
	deploy := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{
		Name:      vcControllerDeploymentName,
		Namespace: "volcano-system",
	}, Spec: appsv1.DeploymentSpec{
		Selector: &metav1.LabelSelector{MatchLabels: labels},
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{Labels: labels},
			Spec: corev1.PodSpec{
				Containers:     []corev1.Container{{Name: vcControllerDeploymentName, Image: containerImage}},
				InitContainers: []corev1.Container{{Name: "init-permission", Image: initContainerImage}},
			},
		},
	}}
	deploy.GetObjectKind().SetGroupVersionKind(gvkDeployment)
	return deploy
}

func deploymentResource(deploy *appsv1.Deployment) componentResource {
	return resourceCreators[gvkDeployment](deploy)
}

func volcanoControllerService() *corev1.Service {
	service := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
		Name:      "volcano-controllers-service",
		Namespace: "volcano-system",
	}}
	service.GetObjectKind().SetGroupVersionKind(gvkService)
	return service
}

func serviceResource(service *corev1.Service) componentResource {
	return resourceCreators[gvkService](service)
}

func TestAddonComponentPathsCreateDRAConfigBeforeKubeletPlugin(t *testing.T) {
	paths := addonComponentPaths()

	assert.True(t, slices.Index(paths, draSoftVNPUComponent) < slices.Index(paths, draKubeletPluginComponent))
	assert.True(t, slices.Index(paths, draVCANNRTInstallerComponent) < slices.Index(paths, draKubeletPluginComponent))
	assert.True(t, slices.Index(paths, draKubeletPluginComponent) < slices.Index(paths, draDeviceClassesComponent))
}

func TestComponentStatusNameUsesUserFacingNewComponentNames(t *testing.T) {
	assert.Equal(t, driverComponent, componentStatusName(driverComponent))
	assert.Empty(t, componentStatusName(vnpuNamespaceComponent))
	assert.Equal(t, componentPathPrefix+"/vnpu-client-update", componentStatusName(vnpuClientUpdateComponent))
	assert.Equal(t, componentPathPrefix+"/vnpu-device-plugin", componentStatusName(vnpuDevicePluginComponent))
	assert.Equal(t, componentPathPrefix+"/vnpu-exporter", componentStatusName(vnpuExporterComponent))
	assert.Equal(t, componentPathPrefix+"/vnpu-exporter", componentStatusName(vnpuExporterSMComponent))

	assert.Empty(t, componentStatusName(volcanoVNPUCRDsComponent))
	assert.Empty(t, componentStatusName(volcanoVNPUNamespaceComponent))
	assert.Equal(t, componentPathPrefix+"/volcano-vnpu-controller", componentStatusName(volcanoVNPUControllerComponent))
	assert.Equal(t, componentPathPrefix+"/volcano-vnpu-scheduler", componentStatusName(volcanoVNPUSchedulerComponent))
	assert.Equal(t, componentPathPrefix+"/volcano-vnpu-admission", componentStatusName(volcanoVNPUAdmissionComponent))

	assert.Equal(t, componentPathPrefix+"/dra-kubelet-plugin", componentStatusName(draKubeletPluginComponent))
	assert.Empty(t, componentStatusName(draDeviceClassesComponent))
	assert.Empty(t, componentStatusName(draSoftVNPUComponent))
	assert.Equal(t, componentPathPrefix+"/dra-vcannrt-installer", componentStatusName(draVCANNRTInstallerComponent))
	assert.Equal(t, componentPathPrefix+"/dra-webhook", componentStatusName(draWebhookComponent))
}

func TestMergeComponentStatesKeepsEnabledComponentVisibleWhenSupportIsUnmanaged(t *testing.T) {
	running := v1.ComponentState{Phase: v1.ComponentRunning, Reason: reconciledReason}
	unmanaged := v1.ComponentState{Phase: v1.ComponentUnmanaged, Reason: componentUnmanagedReason}
	pending := v1.ComponentState{Phase: v1.ComponentPending, Reason: reconcileFailedReason, Message: "daemonset not ready"}

	assert.Equal(t, running, mergeComponentStates(running, unmanaged))
	assert.Equal(t, running, mergeComponentStates(unmanaged, running))
	assert.Equal(t, pending, mergeComponentStates(running, pending))
	assert.Equal(t, pending, mergeComponentStates(pending, unmanaged))
}

func TestSyncComponentStatusesPrunesFineGrainedSupportStatuses(t *testing.T) {
	status := &v1.NPUClusterPolicyStatus{
		ComponentStatuses: []v1.ComponentStatus{
			{Name: vnpuNamespaceComponent, State: v1.ComponentState{Phase: v1.ComponentRunning}},
			{Name: vnpuExporterSMComponent, State: v1.ComponentState{Phase: v1.ComponentRunning}},
			{Name: draKubeletPluginComponent, State: v1.ComponentState{Phase: v1.ComponentRunning}},
		},
	}
	components := []component{
		{name: vnpuNamespaceComponent},
		{name: vnpuExporterComponent},
		{name: vnpuExporterSMComponent},
		{name: draKubeletPluginComponent},
	}

	changed := syncComponentStatuses(status, components, map[string]v1.ComponentState{
		componentPathPrefix + "/vnpu-exporter":      {Phase: v1.ComponentRunning, Reason: reconciledReason},
		componentPathPrefix + "/dra-kubelet-plugin": {Phase: v1.ComponentRunning, Reason: reconciledReason},
	})

	require.True(t, changed)
	assert.Nil(t, status.GetComponentStatus(vnpuNamespaceComponent))
	assert.Nil(t, status.GetComponentStatus(vnpuExporterSMComponent))
	assert.Nil(t, status.GetComponentStatus(draKubeletPluginComponent))
	assert.NotNil(t, status.GetComponentStatus(componentPathPrefix+"/vnpu-exporter"))
	assert.NotNil(t, status.GetComponentStatus(componentPathPrefix+"/dra-kubelet-plugin"))
}

func mockScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(appsv1.AddToScheme(scheme))
	utilruntime.Must(rbacv1.AddToScheme(scheme))
	utilruntime.Must(apiextensionsv1.AddToScheme(scheme))
	utilruntime.Must(monitoringv1.AddToScheme(scheme))
	utilruntime.Must(admissionregistrationv1.AddToScheme(scheme))
	utilruntime.Must(k8sbatchv1.AddToScheme(scheme))
	utilruntime.Must(v1.AddToScheme(scheme))
	return scheme
}

func TestComponentLoadResource(t *testing.T) {
	manifest := []byte(`
apiVersion: v1
kind: Namespace
metadata:
  name: test-namespace
  labels:
    env: dev
`)
	comp, serializer := componentLoadTestFixture()
	err := comp.loadResource(context.TODO(), manifest, serializer)
	assert.NoError(t, err, "loadResource should not return error")
	var expected corev1.Namespace
	err = yaml.Unmarshal(manifest, &expected)
	assert.NoError(t, err, "yaml.Unmarshal should not fail")
	for _, res := range comp.resources {
		spew.Dump(res)
		typed, ok := res.(typedComponentResource[corev1.Namespace, *corev1.Namespace])
		if !ok {
			t.Errorf("expected typedComponentResource[Namespace], got %T", res)
			continue
		}
		actual := typed.obj
		assert.Equal(t, expected.Name, actual.Name, "name should match")
		assert.Equal(t, expected.Labels, actual.Labels, "labels should match")
		assert.Equal(t, expected.Spec, actual.Spec, "spec should match")
	}
}

func TestDRASoftVNPUDoesNotManageVNPUClientUpdate(t *testing.T) {
	r := &NPUClusterPolicyReconciler{
		instance: &v1.NPUClusterPolicy{
			Spec: v1.NPUClusterPolicySpec{
				VNPU: v1.VNPUSpec{Managed: false},
				DRA: v1.DRASpec{
					Managed:  true,
					SoftVNPU: v1.DRASoftVNPUSpec{Managed: true},
				},
			},
		},
	}

	assert.False(t, isComponentManaged[vnpuNamespaceComponent](r))
	assert.False(t, isComponentManaged[vnpuClientUpdateComponent](r))
	assert.True(t, isComponentManaged[draSoftVNPUComponent](r))
	assert.False(t, isComponentManaged[draVCANNRTInstallerComponent](r))
	r.instance.Spec.DRA.SoftVNPU.VCANNRTInstaller.Managed = true
	assert.True(t, isComponentManaged[draVCANNRTInstallerComponent](r))
	assert.False(t, isComponentManaged[vnpuDevicePluginComponent](r))
	assert.False(t, isComponentManaged[vnpuExporterComponent](r))
}

func TestComponentLoadResourceDeviceClassAsUnstructured(t *testing.T) {
	manifest := []byte(`
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: npu.huawei.com
spec:
  selectors:
  - cel:
      expression: device.driver == "npu.huawei.com"
`)
	comp, serializer := componentLoadTestFixture()
	err := comp.loadResource(context.TODO(), manifest, serializer)
	assert.NoError(t, err)
	assert.Len(t, comp.resources, 1)
	assert.Equal(t, "DeviceClass.resource.k8s.io/v1", comp.resources[0].gvk())
}

func componentLoadTestFixture() (component, *json.Serializer) {
	scheme := mockScheme()
	serializer := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, json.SerializerOptions{
		Yaml:   true,
		Pretty: false,
		Strict: false,
	})
	comp := component{
		name:      "test-component",
		resources: []componentResource{},
	}
	return comp, serializer
}

func TestNewComponentAssetsLoad(t *testing.T) {
	scheme := mockScheme()
	serializer := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, json.SerializerOptions{
		Yaml:   true,
		Pretty: false,
		Strict: false,
	})
	assetRoot := filepath.Clean(filepath.Join("..", "..", "assets"))
	tests := []struct {
		name string
		path string
	}{
		{name: "vnpu-client-update", path: filepath.Join(assetRoot, "vnpu", "client-update")},
		{name: "vnpu-namespace", path: filepath.Join(assetRoot, "vnpu", "namespace")},
		{name: "vnpu-device-plugin", path: filepath.Join(assetRoot, "vnpu", "device-plugin")},
		{name: "vnpu-exporter", path: filepath.Join(assetRoot, "vnpu", "exporter")},
		{name: "vnpu-exporter-servicemonitor", path: filepath.Join(assetRoot, "vnpu", "exporter-servicemonitor")},
		{name: "volcano-vnpu-crds", path: filepath.Join(assetRoot, "volcano", "vnpu-crds")},
		{name: "volcano-vnpu-controller", path: filepath.Join(assetRoot, "volcano", "vnpu-controller")},
		{name: "volcano-vnpu-scheduler", path: filepath.Join(assetRoot, "volcano", "vnpu-scheduler")},
		{name: "volcano-vnpu-namespaces", path: filepath.Join(assetRoot, "volcano", "vnpu-namespaces")},
		{name: "volcano-vnpu-admission", path: filepath.Join(assetRoot, "volcano", "vnpu-admission")},
		{name: "dra-kubelet-plugin", path: filepath.Join(assetRoot, "dra", "kubelet-plugin")},
		{name: "dra-deviceclasses", path: filepath.Join(assetRoot, "dra", "deviceclasses")},
		{name: "dra-soft-vnpu", path: filepath.Join(assetRoot, "dra", "soft-vnpu")},
		{name: "dra-vcannrt-installer", path: filepath.Join(assetRoot, "dra", "vcannrt-installer")},
		{name: "dra-webhook", path: filepath.Join(assetRoot, "dra", "webhook")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manifests := readManifests(context.TODO(), tt.path)
			assert.NotEmpty(t, manifests)
			comp := component{name: tt.name}
			for _, manifest := range manifests {
				require.NoError(t, comp.loadResource(context.TODO(), manifest, serializer))
			}
			assert.Len(t, comp.resources, len(manifests))
		})
	}
}

func TestVNPUAssetsMatchRequiredUpstreamShape(t *testing.T) {
	assetRoot := filepath.Clean(filepath.Join("..", "..", "assets"))
	clientUpdate, err := os.ReadFile(filepath.Join(assetRoot, "vnpu", "client-update", "0100_daemonset.yaml"))
	require.NoError(t, err)
	assert.NotContains(t, string(clientUpdate), "name: ascend")
	assert.NotContains(t, string(clientUpdate), "/usr/local/Ascend")

	crdManifests := readManifests(context.TODO(), filepath.Join(assetRoot, "volcano", "vnpu-crds"))
	joinedCRDs := ""
	for _, manifest := range crdManifests {
		joinedCRDs += string(manifest)
	}
	for _, name := range []string{
		"cronjobs.batch.volcano.sh",
		"hypernodes.topology.volcano.sh",
		"nodeshards.shard.volcano.sh",
		"colocationconfigurations.config.volcano.sh",
	} {
		assert.Contains(t, joinedCRDs, name)
	}

	jobsMutate, err := os.ReadFile(filepath.Join(assetRoot, "volcano", "vnpu-admission", "1110_jobs_mutating_webhook.yaml"))
	require.NoError(t, err)
	assert.Contains(t, string(jobsMutate), "namespaceSelector:")
	assert.Contains(t, string(jobsMutate), "objectSelector: {}")
	assert.Contains(t, string(jobsMutate), "reinvocationPolicy: Never")
	assert.Contains(t, string(jobsMutate), "sideEffects: NoneOnDryRun")
	assert.Contains(t, string(jobsMutate), "timeoutSeconds: 10")
	assert.Contains(t, string(jobsMutate), "- CREATE")
	assert.NotContains(t, string(jobsMutate), "- UPDATE")
}

func TestVolcanoVNPUAdmissionAssetsOrderInitBeforeConsumers(t *testing.T) {
	assetRoot := filepath.Clean(filepath.Join("..", "..", "assets"))
	resources := manifestResourceRefs(t, filepath.Join(assetRoot, "volcano", "vnpu-admission"))
	seen := map[string]string{}
	positions := map[string]int{}
	for i, resource := range resources {
		key := resource.key()
		if previousFile, ok := seen[key]; ok {
			t.Fatalf("duplicate resource %s in %s and %s", key, previousFile, resource.file)
		}
		seen[key] = resource.file
		positions[key] = i
	}

	initJobKey := manifestResourceRef{
		APIVersion: "batch/v1",
		Kind:       "Job",
	}
	initJobKey.Metadata.Namespace = volcanoSystemNamespace
	initJobKey.Metadata.Name = volcanoAdmissionInitJobName
	deploymentKey := manifestResourceRef{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
	}
	deploymentKey.Metadata.Namespace = volcanoSystemNamespace
	deploymentKey.Metadata.Name = volcanoAdmissionDeploymentName
	jobsMutatingWebhookKey := manifestResourceRef{
		APIVersion: "admissionregistration.k8s.io/v1",
		Kind:       "MutatingWebhookConfiguration",
	}
	jobsMutatingWebhookKey.Metadata.Name = "volcano-admission-service-jobs-mutate"

	require.Contains(t, positions, initJobKey.key())
	require.Contains(t, positions, deploymentKey.key())
	require.Contains(t, positions, jobsMutatingWebhookKey.key())
	assert.Less(t, positions[initJobKey.key()], positions[deploymentKey.key()])
	assert.Less(t, positions[initJobKey.key()], positions[jobsMutatingWebhookKey.key()])
}

type manifestResourceRef struct {
	file       string
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Metadata   struct {
		Namespace string `json:"namespace"`
		Name      string `json:"name"`
	} `json:"metadata"`
}

func (r manifestResourceRef) key() string {
	return fmt.Sprintf("%s/%s/%s/%s", r.APIVersion, r.Kind, r.Metadata.Namespace, r.Metadata.Name)
}

func manifestResourceRefs(t *testing.T, dir string) []manifestResourceRef {
	t.Helper()
	files, err := os.ReadDir(dir)
	require.NoError(t, err)
	resources := []manifestResourceRef{}
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		content, err := os.ReadFile(filepath.Join(dir, file.Name()))
		require.NoError(t, err)
		resource := manifestResourceRef{file: file.Name()}
		require.NoError(t, yaml.Unmarshal(content, &resource))
		resources = append(resources, resource)
	}
	return resources
}

func TestReadManifests(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-manifests")
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	files := []struct {
		name    string
		content []byte
	}{
		{"file1.txt", []byte("content1")},
		{"file2.txt", []byte("content2")},
		{"subdir/file3.txt", []byte("content3")},
	}
	for _, file := range files {
		filePath := filepath.Join(tempDir, file.name)
		if err := os.MkdirAll(filepath.Dir(filePath), os.ModePerm); err != nil {
			t.Fatalf("Failed to create directory: %v", err)
		}
		if err := ioutil.WriteFile(filePath, file.content, os.ModePerm); err != nil {
			t.Fatalf("Failed to write file: %v", err)
		}
	}
	ctx := context.TODO()
	manifests := readManifests(ctx, tempDir)
	expected := [][]byte{
		[]byte("content1"),
		[]byte("content2"),
		[]byte("content3"),
	}
	if len(manifests) != len(expected) {
		t.Errorf("Expected %d manifests, got %d", len(expected), len(manifests))
	}
	for i, manifest := range manifests {
		if !equal(manifest, expected[i]) {
			t.Errorf("Manifest content mismatch for file %d: expected %q, got %q", i, expected[i], manifest)
		}
	}
}

// 辅助函数，用于比较两个字节切片是否相等
func equal(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestNPUClusterPolicyReconcilerLoadComponents(t *testing.T) {
	r := mockReconciler()
	r.components = nil
	scheme := mockScheme()
	fakeCli := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(&v1.NPUClusterPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-policy",
			},
		}).Build()
	r.Client = fakeCli
	statusWriter := fakeCli.Status()
	realType := reflect.TypeOf(statusWriter)

	method, ok := realType.MethodByName("Update")
	if !ok {
		panic("cannot find method 'Update' on statusWriter")
	}
	patchFunc := reflect.MakeFunc(method.Type, func(args []reflect.Value) []reflect.Value {
		return []reflect.Value{reflect.Zero(reflect.TypeOf((*error)(nil)).Elem())}
	})
	patch := gomonkey.ApplyMethod(realType, "Update", patchFunc.Interface())
	g := gomonkey.ApplyPrivateMethod(r, "getInstance", func(_ *NPUClusterPolicyReconciler, _ context.Context, _ string) error {
		return nil
	})
	defer g.Reset()
	defer patch.Reset()
	assert.NoError(t, r.loadComponents(context.TODO()))
}

type mockDriver struct {
	name string
}

func (m *mockDriver) loadResource(ctx context.Context, manifest []byte, serializer *json.Serializer) error {
	return nil
}
func (m *mockDriver) reconcile(ctx context.Context, r *NPUClusterPolicyReconciler) (v1.ComponentState, error) {
	return v1.ComponentState{Phase: v1.ComponentDeploying}, nil
}

func TestReconcileComponents(t *testing.T) {
	r := mockReconciler()
	comp := &component{
		name:      "driver",
		resources: nil,
	}
	r.components = []component{*comp}
	patch := gomonkey.ApplyPrivateMethod(
		comp,
		"reconcile",
		func(_ *component, _ context.Context, _ *NPUClusterPolicyReconciler) (v1.ComponentState, error) {
			return v1.ComponentState{Phase: v1.ComponentRunning}, nil
		},
	)
	defer patch.Reset()

	patch2 := gomonkey.ApplyPrivateMethod(
		r,
		"updateStatus",
		func(_ *NPUClusterPolicyReconciler, _ context.Context, _ updateStatusCallback) error {
			return nil
		},
	)
	defer patch2.Reset()

	_, err := r.reconcileComponents(context.TODO())
	assert.NoError(t, err)
}
