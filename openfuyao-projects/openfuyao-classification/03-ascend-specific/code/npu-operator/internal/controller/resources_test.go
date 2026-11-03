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
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/kubernetes/scheme"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"volcano.sh/apis/pkg/apis/batch/v1alpha1"
	busv1alpha1 "volcano.sh/apis/pkg/apis/bus/v1alpha1"
	flowv1alpha1 "volcano.sh/apis/pkg/apis/flow/v1alpha1"
	nodeinfov1alpha1 "volcano.sh/apis/pkg/apis/nodeinfo/v1alpha1"
	"volcano.sh/apis/pkg/apis/scheduling"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func getInstance() *v1.NPUClusterPolicy {
	return &v1.NPUClusterPolicy{
		TypeMeta: metav1.TypeMeta{
			Kind:       "NPUClusterPolicy",
			APIVersion: v1.GroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "cluster",
		},
		Spec: v1.NPUClusterPolicySpec{
			Driver:       v1.DriverSpec{Managed: true},
			OCIRuntime:   v1.OCIRuntimeSpec{Managed: true},
			VCScheduler:  v1.SchedulerSpec{Managed: true},
			VCController: v1.ControllerSpec{Managed: true},
			DevicePlugin: v1.DevicePluginSpec{Managed: true},
			Trainer:      v1.TrainerSpec{Managed: false},
			NodeD:        v1.NodeDSpec{Managed: true},
			ClusterD:     v1.ClusterDSpec{Managed: true},
			RSController: v1.RSControllerSpec{Managed: true},
			Exporter:     v1.ExporterSpec{Managed: true},
			MindIOTFT:    v1.MindIOTFTSpec{Managed: true},
			MindIOACP:    v1.MindIOACPSpec{Managed: true},
		},
	}
}

func getManagerAndReconciler(
	f func(*fake.ClientBuilder) *fake.ClientBuilder,
	instance *v1.NPUClusterPolicy,
) (ctrl.Manager, NPUClusterPolicyReconciler) {
	scheme := runtime.NewScheme()
	utilruntime.Must(v1.AddToScheme(scheme))
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(apiextensionsv1.AddToScheme(scheme))
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(monitoringv1.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	utilruntime.Must(busv1alpha1.AddToScheme(scheme))
	utilruntime.Must(scheduling.AddToScheme(scheme))
	utilruntime.Must(nodeinfov1alpha1.AddToScheme(scheme))
	utilruntime.Must(flowv1alpha1.AddToScheme(scheme))
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if f != nil {
		builder = f(builder)
	}
	if instance != nil {
		builder = builder.WithObjects(instance).WithStatusSubresource(instance)
	}
	client := builder.Build()

	mgr, err := ctrl.NewManager(&rest.Config{}, ctrl.Options{
		Scheme: client.Scheme(),
	})
	if err != nil {
		return nil, NPUClusterPolicyReconciler{}
	}
	r := NPUClusterPolicyReconciler{
		Client:   client,
		Scheme:   client.Scheme(),
		instance: instance,
	}
	return mgr, r
}

func TestClone(t *testing.T) {
	obj := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
	}
	res := typedComponentResource[corev1.Namespace, *corev1.Namespace]{
		obj:                             obj,
		componentResourceReconcileHooks: namespaceReconcileHooks{},
	}
	tests := []struct {
		name string
	}{
		{
			name: "ok",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			re := res.clone()
			if !reflect.DeepEqual(re, res) {
				t.Errorf("clone() = %v, want %v", re, res)
			}
		})
	}
}

func TestGVK(t *testing.T) {
	test := []struct {
		name string
		obj  client.Object
		gvk  schema.GroupVersionKind
	}{
		{
			name: "deployment",
			obj: &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
			},
			gvk: gvkDeployment,
		},
		{
			name: "daemonset",
			obj:  &appsv1.DaemonSet{},
			gvk:  gvkDaemonSet,
		},
		{
			name: "deviceclass",
			obj: &unstructured.Unstructured{
				Object: map[string]any{
					"apiVersion": "resource.k8s.io/v1",
					"kind":       "DeviceClass",
					"metadata": map[string]any{
						"name": "npu.huawei.com",
					},
				},
			},
			gvk: gvkDeviceClass,
		},
	}
	for _, tt := range test {
		t.Run(tt.name, func(t *testing.T) {
			tt.obj.GetObjectKind().SetGroupVersionKind(tt.gvk)
			rc := resourceCreators[tt.gvk](tt.obj)
			got := rc.gvk()
			parts := strings.Split(got, "/")
			gk := parts[0]
			v := parts[1]
			gkparts := strings.SplitN(gk, ".", 2)
			k := gkparts[0]
			g := gkparts[1]
			if g != tt.gvk.Group || k != tt.gvk.Kind || v != tt.gvk.Version {
				t.Errorf("no equal got %v, want %v", got, tt.gvk)
			}
		})
	}
}

func TestDRAResourceNamesDoNotUseAscendPrefix(t *testing.T) {
	assert.Equal(t, "npu-dra-kubeletplugin", draKubeletPluginDaemonSetName)
	assert.Equal(t, "vcannrt-installer", draVCANNRTInstallerDaemonSetName)
	assert.Equal(t, "dra-profile-config", draProfileConfigMapName)
	assert.Equal(t, "vnpu-template-config", draChipCapabilitiesConfigMapName)
}

func TestNewWorkloadPlacementTransformerDaemonSet(t *testing.T) {
	transformer := newWorkloadPlacementTransformer[appsv1.DaemonSet](func(spec *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
		return &spec.VNPU.DevicePlugin.PlacementSpec
	})
	ds := &appsv1.DaemonSet{
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					NodeSelector: map[string]string{"existing": "keep", "accelerator": "old"},
					Tolerations: []corev1.Toleration{{
						Key:      "existing",
						Operator: corev1.TolerationOpExists,
					}},
				},
			},
		},
	}
	r := &NPUClusterPolicyReconciler{
		instance: &v1.NPUClusterPolicy{
			Spec: v1.NPUClusterPolicySpec{
				VNPU: v1.VNPUSpec{
					DevicePlugin: v1.VNPUDevicePluginSpec{
						PlacementSpec: v1.PlacementSpec{
							NodeSelector: map[string]string{"accelerator": "vnpu"},
							Tolerations: []corev1.Toleration{{
								Key:      "vnpu",
								Operator: corev1.TolerationOpExists,
							}},
						},
					},
				},
			},
		},
	}

	err := transformer(context.TODO(), r, ds)
	require.NoError(t, err)
	assert.Equal(t, "keep", ds.Spec.Template.Spec.NodeSelector["existing"])
	assert.Equal(t, "vnpu", ds.Spec.Template.Spec.NodeSelector["accelerator"])
	assert.Len(t, ds.Spec.Template.Spec.Tolerations, 2)
	assert.Equal(t, "vnpu", ds.Spec.Template.Spec.Tolerations[1].Key)
}

func TestPlacementGettersForExistingComponents(t *testing.T) {
	spec := placementSpecForGetters()
	for _, tt := range placementGetterCases {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.get(spec).NodeSelector["component"])
		})
	}
}

func placementSpecForGetters() *v1.NPUClusterPolicySpec {
	return &v1.NPUClusterPolicySpec{
		Driver:       placementDriverSpec("driver"),
		OCIRuntime:   v1.OCIRuntimeSpec{PlacementSpec: namedPlacement("runtime")},
		DevicePlugin: v1.DevicePluginSpec{PlacementSpec: namedPlacement("device-plugin")},
		NodeD:        v1.NodeDSpec{PlacementSpec: namedPlacement("noded")},
		Exporter:     v1.ExporterSpec{PlacementSpec: namedPlacement("exporter")},
		MindIOTFT:    v1.MindIOTFTSpec{PlacementSpec: namedPlacement("mindiotft")},
		MindIOACP:    v1.MindIOACPSpec{PlacementSpec: namedPlacement("mindioacp")},
		Trainer:      v1.TrainerSpec{PlacementSpec: namedPlacement("trainer")},
		ClusterD:     v1.ClusterDSpec{PlacementSpec: namedPlacement("clusterd")},
		RSController: v1.RSControllerSpec{PlacementSpec: namedPlacement("rscontroller")},
		VCController: v1.ControllerSpec{PlacementSpec: namedPlacement("vccontroller")},
		VCScheduler:  v1.SchedulerSpec{PlacementSpec: namedPlacement("vcscheduler")},
	}
}

func placementDriverSpec(component string) v1.DriverSpec {
	return v1.DriverSpec{PlacementSpec: namedPlacement(component)}
}

func namedPlacement(component string) v1.PlacementSpec {
	return v1.PlacementSpec{NodeSelector: map[string]string{"component": component}}
}

type placementGetterCase struct {
	name string
	get  func(*v1.NPUClusterPolicySpec) *v1.PlacementSpec
	want string
}

var placementGetterCases = []placementGetterCase{
	{name: "driver", get: driverPlacementGetter, want: "driver"},
	{name: "runtime", get: runtimePlacementGetter, want: "runtime"},
	{name: "device-plugin", get: devicePluginPlacementGetter, want: "device-plugin"},
	{name: "noded", get: nodeDPlacementGetter, want: "noded"},
	{name: "exporter", get: exporterPlacementGetter, want: "exporter"},
	{name: "mindiotft", get: mindIOTFTPlacementGetter, want: "mindiotft"},
	{name: "mindioacp", get: mindIOACPPlacementGetter, want: "mindioacp"},
	{name: "trainer", get: trainerPlacementGetter, want: "trainer"},
	{name: "clusterd", get: clusterdPlacementGetter, want: "clusterd"},
	{name: "rscontroller", get: rsControllerPlacementGetter, want: "rscontroller"},
	{name: "vccontroller", get: vcControllerPlacementGetter, want: "vccontroller"},
	{name: "vcscheduler", get: vcSchedulerPlacementGetter, want: "vcscheduler"},
}

func TestResourceReconcile(t *testing.T) {
	tests := []struct {
		name        string
		gvk         schema.GroupVersionKind
		obj         client.Object
		managed     bool
		transforErr error
		inspectErr  error
		want        *v1.ComponentState
		wantErr     error
	}{
		{
			name: "reconcile namespace",
			obj: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
			},
			gvk:     gvkNamespace,
			managed: true,
		},
		{
			name: "reconcile daemonset",
			obj: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{
				Name:      nodedDaemonsetName,
				Namespace: placeholderValue,
			}, Spec: appsv1.DaemonSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: nodedDaemonsetName, Image: "noded:test"}},
			}}}},
			gvk:     gvkDaemonSet,
			managed: true,
		},
		{
			name: "reconcile driver daemonset",
			obj: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{
				Name:      driverDaemonSetName,
				Namespace: placeholderValue,
			}},
			gvk:     gvkDaemonSet,
			managed: false,
		},
		{
			name: "reconcile device plugin daemonset",
			obj: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{
				Name:      devicePluginDaemonSetName,
				Namespace: placeholderValue,
			}},
			gvk:     gvkDaemonSet,
			managed: false,
		},
		{
			name: "reconcile exporter daemonset",
			obj: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{
				Name:      exporterDaemonsetName,
				Namespace: placeholderValue,
			}},
			gvk:     gvkDaemonSet,
			managed: false,
		},
		{
			name: "reconcile driver daemonset",
			obj: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{
				Name:      mindIOACPDaemonsetName,
				Namespace: placeholderValue,
			}},
			gvk:     gvkDaemonSet,
			managed: false,
		},
		{
			name: "reconcile runtime daemonset",
			obj: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{
				Name:      runtimeDaemonSetNamePrefix,
				Namespace: placeholderValue,
			}},
			gvk:     gvkDaemonSet,
			managed: false,
		},
		{
			name: "reconcile controller deployment",
			obj: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{
				Name:      vcControllerDeploymentName,
				Namespace: placeholderValue,
			},
			},
			gvk:     gvkDeployment,
			managed: false,
		},
		{
			name:    "reconcile runtime daemonset",
			obj:     readyDaemonSetForTest("ascend-runtime-", "def", map[string]string{"openfuyao.com/container.runtime": "containerd"}),
			gvk:     gvkDaemonSet,
			managed: true,
		},
		{
			name:    "reconcile CRD",
			obj:     &apiextensionsv1.CustomResourceDefinition{},
			gvk:     gvkCRD,
			managed: false,
		},
		{
			name:    "reconcile runtimeclass",
			obj:     &nodev1.RuntimeClass{},
			gvk:     gvkRuntimeClass,
			managed: false,
		},
		{
			name:    "reconcile serviceaccount",
			obj:     &corev1.ServiceAccount{},
			gvk:     gvkServiceAccount,
			managed: false,
		},
		{
			name:    "reconcile clusterrole",
			obj:     &rbacv1.ClusterRole{},
			gvk:     gvkClusterRole,
			managed: false,
		},
		{
			name:    "reconcile clusterrolebinding",
			obj:     &rbacv1.ClusterRoleBinding{},
			gvk:     gvkClusterRoleBinding,
			managed: false,
		},
		{
			name:    "reconcile servicemonitor",
			obj:     &monitoringv1.ServiceMonitor{},
			gvk:     gvkServiceMonitor,
			managed: false,
		},
		{
			name:    "reconcile service",
			obj:     &corev1.Service{},
			gvk:     gvkService,
			managed: false,
		},
		{
			name:    "reconcile lease",
			obj:     &coordinationv1.Lease{},
			gvk:     gvkLease,
			managed: false,
		},
		{
			name:    "reconcile configmap",
			obj:     &corev1.ConfigMap{},
			gvk:     gvkConfigMap,
			managed: false,
		},
		{
			name: "reconcile unmanaged unstructured deviceclass",
			obj: &unstructured.Unstructured{Object: map[string]any{
				"apiVersion": gvkDeviceClass.GroupVersion().String(),
				"kind":       gvkDeviceClass.Kind,
				"metadata": map[string]any{
					"name": "fixed.npu.huawei.com",
				},
			}},
			gvk:     gvkDeviceClass,
			managed: false,
		},
		{
			name:    "reconcile volcano.batch.job",
			obj:     &v1alpha1.Job{},
			gvk:     gvkVolcanoBatchJob,
			managed: false,
		},
		{
			name:    "reconcile volcano.bus.command",
			obj:     &busv1alpha1.Command{},
			gvk:     gvkVolcanoBusCommand,
			managed: false,
		},
		{
			name:    "reconcile pog group",
			obj:     &scheduling.PodGroup{},
			gvk:     gvkSchedulingPodgroup,
			managed: false,
		},
		{
			name:    "reconcile queue",
			gvk:     gvkSchedulerQueue,
			obj:     &scheduling.Queue{},
			managed: false,
		},
		{
			gvk:     gvkNodeInfo,
			obj:     &nodeinfov1alpha1.Numatopology{},
			name:    "reconcile numatopology",
			managed: false,
		},
		{
			name:    "reconcile jobtemplate",
			gvk:     gvkJobTemplate,
			obj:     &flowv1alpha1.JobTemplate{},
			managed: false,
		},
		{
			name:    "reconcile jobflow",
			gvk:     gvkJobFlow,
			obj:     &flowv1alpha1.JobFlow{},
			managed: false,
		},
	}
	instance := getInstance()
	_, r := getManagerAndReconciler(nil, instance)
	r.hasNPUNodes = true
	r.hasNFDLabels = true
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.obj.GetObjectKind().SetGroupVersionKind(tt.gvk)
			res := resourceCreators[tt.gvk](tt.obj)
			state, err := res.reconcile(context.Background(), &r, tt.managed)
			assert.Equal(t, tt.want, state)
			assert.Equal(t, tt.wantErr, err)
		})
	}
}

func TestDaemonsetInspect(t *testing.T) {
	test := []struct {
		name    string
		obj     client.Object
		gvk     schema.GroupVersionKind
		want    *v1.ComponentState
		wantErr error
	}{
		{
			name: "deployment",
			obj: &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: appsv1.DeploymentSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{},
					},
				},
			},
			gvk:  gvkDeployment,
			want: &v1.ComponentState{Phase: "pending", Reason: "PodNotRunning", Message: "Deployment Pod pod1 ,state "},
		},
		{
			name: "daemonset",
			obj:  readyDaemonSetForTest("ok", "def", nil),
			gvk:  gvkDaemonSet,
			want: &v1.ComponentState{Phase: "pending", Reason: "PodNotRunning", Message: "Daemonset Pod pod1 ,state "},
		},
	}
	for _, tt := range test {
		t.Run(tt.name, func(t *testing.T) {
			tt.obj.GetObjectKind().SetGroupVersionKind(tt.gvk)
			pod1 := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
				},
			}
			scheme := runtime.NewScheme()
			_ = corev1.AddToScheme(scheme)
			_ = clientgoscheme.AddToScheme(scheme)
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(pod1, tt.obj).
				Build()
			r := mockReconciler()
			r.Client = fakeClient
			a := gomonkey.ApplyMethod(r, "Get", func(_ *NPUClusterPolicyReconciler, _ context.Context, _ client.ObjectKey, _ client.Object, _ ...client.GetOption) error {
				return nil
			})
			b := gomonkey.ApplyMethod(r, "List", func(_ *NPUClusterPolicyReconciler, _ context.Context, _ client.ObjectList, _ ...client.ListOption) error {
				return nil
			})
			defer a.Reset()
			defer b.Reset()
			switch obj := tt.obj.(type) {
			case *appsv1.DaemonSet:
				h := &daemonSetReconcileHooks{}
				state, err := h.inspect(context.Background(), r, obj)
				assert.Equal(t, tt.want, state)
				assert.Equal(t, tt.wantErr, err)
			case *appsv1.Deployment:
				h := &deploymentReconcileHooks{}
				state, err := h.inspect(context.Background(), r, obj)
				assert.Equal(t, tt.want, state)
				assert.Equal(t, tt.wantErr, err)
			case *corev1.Namespace:
				h := &namespaceReconcileHooks{}
				state, err := h.inspect(context.Background(), r, obj)
				assert.Equal(t, tt.want, state)
				assert.Equal(t, tt.wantErr, err)
			default:
				t.Fatalf("unsupported object type: %T", tt.obj)
			}
		})
	}
}

func readyDaemonSetForTest(name, namespace string, nodeSelector map[string]string) *appsv1.DaemonSet {
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Status: appsv1.DaemonSetStatus{
			DesiredNumberScheduled: 1,
			NumberUnavailable:      0,
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{},
			},
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					NodeSelector: nodeSelector,
				},
			},
		},
	}
}

func TestDelete(t *testing.T) {
	_ = corev1.AddToScheme(scheme.Scheme)

	owner := &v1.NPUClusterPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "cluster",
			UID:  uuid.NewUUID(),
		},
		TypeMeta: metav1.TypeMeta{
			Kind: "NPU ClusterPolicy",
		},
	}

	ctx := context.TODO()

	tests := []struct {
		name       string
		obj        *corev1.Namespace
		existing   client.Object
		ownerRef   bool
		shouldPass bool
	}{
		{
			name: "OwnerReference matches",
			obj: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			existing: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
					OwnerReferences: []metav1.OwnerReference{
						{
							Kind: "NPU ClusterPolicy",
							Name: "cluster",
							UID:  owner.UID,
						},
					},
				},
			},
			shouldPass: true,
		},
		{
			name: "OwnerReference does not match",
			obj: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			existing: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
					OwnerReferences: []metav1.OwnerReference{
						{
							Kind: "OtherKind",
							Name: "other",
							UID:  "xxxx",
						},
					},
				},
			},
			shouldPass: false,
		},
		{
			name: "No OwnerReference",
			obj: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			existing: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			shouldPass: false,
		},
		{
			name: "NotFound error",
			obj: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "not-exist"},
			},
			existing:   nil,
			shouldPass: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := fake.NewClientBuilder().WithScheme(scheme.Scheme)
			if tt.existing != nil {
				builder = builder.WithObjects(tt.existing)
			}
			cl := builder.Build()

			r := &NPUClusterPolicyReconciler{
				Client:    cl,
				Scheme:    scheme.Scheme,
				namespace: "default",
				instance:  owner,
			}
			hook := defaultResourceReconcileHooks[*corev1.Namespace]{}
			obj, ok, err := hook.canDelete(ctx, r, tt.obj)
			require.NoError(t, err)

			if tt.shouldPass {
				assert.True(t, ok)
				assert.NotNil(t, obj)
			} else {
				assert.False(t, ok)
				assert.Nil(t, obj)
			}
		})
	}
}

func TestCanDeleteUnstructuredDeviceClassPreservesGVK(t *testing.T) {
	ctx := context.Background()
	owner := getInstance()
	existing := draDeviceClassObject("fixed.npu.huawei.com", `device.driver == "npu.huawei.com"`)
	existing.SetOwnerReferences([]metav1.OwnerReference{*metav1.NewControllerRef(
		owner,
		v1.GroupVersion.WithKind("NPUClusterPolicy"),
	)})
	s := runtime.NewScheme()
	s.AddKnownTypeWithName(gvkDeviceClass, &unstructured.Unstructured{})
	r := &NPUClusterPolicyReconciler{
		Client:   fake.NewClientBuilder().WithScheme(s).WithObjects(existing).Build(),
		Scheme:   s,
		instance: owner,
	}
	desired := draDeviceClassObject("fixed.npu.huawei.com", `device.driver == "npu.huawei.com"`)
	hook := defaultResourceReconcileHooks[*unstructured.Unstructured]{}

	obj, ok, err := hook.canDelete(ctx, r, desired)
	require.NoError(t, err)

	require.True(t, ok)
	require.NotNil(t, obj)
	assert.Equal(t, "fixed.npu.huawei.com", obj.GetName())
}

func TestCanDeleteMatchesVNPUControllerOwnerWhenInstanceKindIsEmpty(t *testing.T) {
	ctx := context.Background()
	owner := getInstance()
	owner.Kind = ""
	owner.APIVersion = ""
	ownerRef := *metav1.NewControllerRef(owner, v1.GroupVersion.WithKind("NPUClusterPolicy"))
	s := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(s))
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	for _, tt := range vnpuControllerOwnerCanDeleteCases(ownerRef) {
		t.Run(tt.name, func(t *testing.T) {
			r := &NPUClusterPolicyReconciler{
				Client:   fake.NewClientBuilder().WithScheme(s).WithObjects(tt.existing).Build(),
				Scheme:   s,
				instance: owner,
			}

			obj, ok, err := tt.canDelete(ctx, r)
			require.NoError(t, err)

			require.True(t, ok)
			require.NotNil(t, obj)
			assert.Equal(t, tt.wantName, obj.GetName())
		})
	}
}

type vnpuControllerOwnerCanDeleteCase struct {
	name      string
	wantName  string
	existing  client.Object
	canDelete func(context.Context, *NPUClusterPolicyReconciler) (client.Object, bool, error)
}

func vnpuControllerOwnerCanDeleteCases(ownerRef metav1.OwnerReference) []vnpuControllerOwnerCanDeleteCase {
	return []vnpuControllerOwnerCanDeleteCase{
		vnpuControllerDaemonSetCanDeleteCase("client update daemonset", vnpuClientUpdateDaemonSetName, ownerRef),
		vnpuControllerDaemonSetCanDeleteCase("device plugin daemonset", vnpuDevicePluginDaemonSetName, ownerRef),
		vnpuControllerDaemonSetCanDeleteCase("exporter daemonset", vnpuExporterDaemonSetName, ownerRef),
		vnpuControllerServiceCanDeleteCase("exporter service", vnpuExporterServiceName, ownerRef),
		vnpuControllerConfigMapCanDeleteCase("device plugin configmap", vnpuNodeModeConfigMapName, ownerRef),
	}
}

func vnpuControllerDaemonSetCanDeleteCase(
	name, resourceName string,
	ownerRef metav1.OwnerReference,
) vnpuControllerOwnerCanDeleteCase {
	return vnpuControllerOwnerCanDeleteCase{
		name:     name,
		wantName: resourceName,
		existing: &appsv1.DaemonSet{ObjectMeta: vnpuControllerObjectMeta(resourceName, ownerRef)},
		canDelete: func(ctx context.Context, r *NPUClusterPolicyReconciler) (client.Object, bool, error) {
			desired := &appsv1.DaemonSet{ObjectMeta: vnpuControllerDesiredObjectMeta(resourceName)}
			return defaultResourceReconcileHooks[*appsv1.DaemonSet]{}.canDelete(ctx, r, desired)
		},
	}
}

func vnpuControllerServiceCanDeleteCase(
	name, resourceName string,
	ownerRef metav1.OwnerReference,
) vnpuControllerOwnerCanDeleteCase {
	return vnpuControllerOwnerCanDeleteCase{
		name:     name,
		wantName: resourceName,
		existing: &corev1.Service{ObjectMeta: vnpuControllerObjectMeta(resourceName, ownerRef)},
		canDelete: func(ctx context.Context, r *NPUClusterPolicyReconciler) (client.Object, bool, error) {
			desired := &corev1.Service{ObjectMeta: vnpuControllerDesiredObjectMeta(resourceName)}
			return defaultResourceReconcileHooks[*corev1.Service]{}.canDelete(ctx, r, desired)
		},
	}
}

func vnpuControllerConfigMapCanDeleteCase(
	name, resourceName string,
	ownerRef metav1.OwnerReference,
) vnpuControllerOwnerCanDeleteCase {
	return vnpuControllerOwnerCanDeleteCase{
		name:     name,
		wantName: resourceName,
		existing: &corev1.ConfigMap{ObjectMeta: vnpuControllerObjectMeta(resourceName, ownerRef)},
		canDelete: func(ctx context.Context, r *NPUClusterPolicyReconciler) (client.Object, bool, error) {
			desired := &corev1.ConfigMap{ObjectMeta: vnpuControllerDesiredObjectMeta(resourceName)}
			return defaultResourceReconcileHooks[*corev1.ConfigMap]{}.canDelete(ctx, r, desired)
		},
	}
}

func vnpuControllerObjectMeta(resourceName string, ownerRef metav1.OwnerReference) metav1.ObjectMeta {
	meta := vnpuControllerDesiredObjectMeta(resourceName)
	meta.OwnerReferences = []metav1.OwnerReference{ownerRef}
	return meta
}

func vnpuControllerDesiredObjectMeta(resourceName string) metav1.ObjectMeta {
	return metav1.ObjectMeta{Name: resourceName, Namespace: "xpu"}
}

func Test_runtimeDaemonSetTransformer(t *testing.T) {
	tests := []struct {
		name       string
		dsName     string
		runtimes   map[string]struct{}
		expectSkip bool
	}{
		{
			name:     "runtime exists",
			dsName:   "ascend-runtime-containerd",
			runtimes: map[string]struct{}{"containerd": {}},
		},
		{
			name:       "runtime not found",
			dsName:     "ascend-runtime-docker",
			runtimes:   map[string]struct{}{"containerd": {}},
			expectSkip: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name: tt.dsName,
				},
			}

			r := &NPUClusterPolicyReconciler{
				runtimes: tt.runtimes,
			}

			err := runtimeDaemonSetTransformer(context.TODO(), r, ds)

			if tt.expectSkip {
				assert.Equal(t, errSkipReconciliation, err)
			} else {
				assert.NoError(t, err)
				runtime := strings.TrimPrefix(tt.dsName, runtimeDaemonSetNamePrefix)
				assert.Equal(t, runtime, ds.Spec.Template.Spec.NodeSelector[runtimeLabelKey])
				assert.Equal(t, "", ds.Spec.Template.Annotations[runtimeEndpointAnnotationKey])
			}
		})
	}
}

func Test_runtimeDaemonSetInspector(t *testing.T) {
	const (
		nodeName   = "npu-node"
		podName    = "npu-pod"
		endpoint   = "unix:///var/run/containerd.sock"
		ns         = "test-ns"
		labelKey   = "app"
		labelValue = "npu"
	)

	selector := map[string]string{labelKey: labelValue}

	tests := []struct {
		name          string
		pods          []corev1.Pod
		nodeEndpoints map[string]string
		expectUpdated bool
		expectErr     bool
	}{
		{
			name: "pod matches node endpoint and gets annotation",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: ns,
						Labels:    selector,
					},
					Spec: corev1.PodSpec{
						NodeName: nodeName,
					},
				},
			},
			nodeEndpoints: map[string]string{
				nodeName: endpoint,
			},
			expectUpdated: true,
		},
		{
			name: "pod node not in endpoint map, no update",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: ns,
						Labels:    selector,
					},
					Spec: corev1.PodSpec{
						NodeName: "unknown-node",
					},
				},
			},
			nodeEndpoints: map[string]string{
				nodeName: endpoint,
			},
			expectUpdated: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_ = corev1.AddToScheme(scheme.Scheme)

			podObjs := make([]client.Object, len(tt.pods))
			for i := range tt.pods {
				podObjs[i] = &tt.pods[i]
			}

			cl := fake.NewClientBuilder().
				WithScheme(scheme.Scheme).
				WithObjects(podObjs...).
				Build()

			r := &NPUClusterPolicyReconciler{
				Client:               cl,
				Scheme:               scheme.Scheme,
				nodeRuntimeEndpoints: tt.nodeEndpoints,
				namespace:            ns,
			}

			ds := &appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: selector,
					},
				},
			}

			err := runtimeDaemonSetInspector(context.TODO(), r, ds)
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			if tt.expectUpdated {
				var updatedPod corev1.Pod
				err := cl.Get(context.TODO(), client.ObjectKey{Namespace: ns, Name: podName}, &updatedPod)
				assert.NoError(t, err)
				assert.Equal(t, endpoint, updatedPod.Annotations[runtimeEndpointAnnotationKey])
			}
		})
	}
}

func mockLog(*v1.NPUClusterPolicySpec) *v1.LogRotate {
	return &v1.LogRotate{
		LogLevel: v1.LogInfo,
		LogFile:  "/var/log/npu.log",
		Rotate:   5,
		MaxAge:   7,
		Compress: true,
	}
}

func Test_LogRotateTransformer_DaemonSet(t *testing.T) {
	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "npu",
							Env: []corev1.EnvVar{
								{Name: logRotateLogLevelEnvName, Value: "old"},
								{Name: logRotateLogFileEnvName, Value: "old"},
								{Name: logRotateRotateEnvName, Value: "old"},
								{Name: logRotateMaxAgeEnvName, Value: "old"},
								{Name: logRotateCompressEnvName, Value: "old"},
							},
						},
					},
				},
			},
		},
	}

	r := mockReconciler()
	r.instance = getInstance()
	transformer := newWorkloadLogRotateTransformer[appsv1.DaemonSet](mockLog)
	err := transformer(context.TODO(), r, ds)
	assert.NoError(t, err)
	envMap := map[string]string{}
	for _, env := range ds.Spec.Template.Spec.Containers[0].Env {
		envMap[env.Name] = env.Value
	}
	assert.Equal(t, fmt.Sprintf("%d", v1.LogInfoValue), envMap[logRotateLogLevelEnvName])
	assert.Equal(t, "/var/log/npu.log", envMap[logRotateLogFileEnvName])
	assert.Equal(t, "5", envMap[logRotateRotateEnvName])
	assert.Equal(t, "7", envMap[logRotateMaxAgeEnvName])
	assert.Equal(t, "true", envMap[logRotateCompressEnvName])
}

func Test_configMapReconcileHooks_transform_volcanoConfig(t *testing.T) {
	const (
		configMapName          = "volcano-scheduler-configmap"
		namespace              = "test-ns"
		nodeName               = "node01"
		arch                   = "amd64"
		volcanoVersion         = "7.1.RC1"
		expectedLine           = "  - name: volcano-npu_7.1.RC1_linux-x86_64"
		initialSchedulerConfig = `actions: "enqueue, allocate"
tiers:
- plugins:
  - name: volcano-npu_OLD_linux-aarch64`
	)
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: namespace,
		},
		Data: map[string]string{},
	}
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName},
		Status: corev1.NodeStatus{
			NodeInfo: corev1.NodeSystemInfo{Architecture: arch},
		},
	}
	s := scheme.Scheme
	_ = corev1.AddToScheme(s)
	_ = clientgoscheme.AddToScheme(s)
	_ = v1.AddToScheme(s)
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(configMap, node).Build()
	r := &NPUClusterPolicyReconciler{
		Client:    cl,
		Scheme:    s,
		namespace: namespace,
		instance: &v1.NPUClusterPolicy{
			Spec: v1.NPUClusterPolicySpec{
				VCScheduler: v1.SchedulerSpec{
					SchedulerConfigMap: initialSchedulerConfig,
				},
			},
		},
	}
	t.Setenv("OPERATOR_NODE_NAME", nodeName)
	patchVolcanoVer := gomonkey.ApplyFuncReturn(ObtainVolcanoVersion, volcanoVersion, nil)
	defer patchVolcanoVer.Reset()
	h := configMapReconcileHooks{}
	err := h.transform(context.TODO(), r, configMap)
	assert.NoError(t, err)
	result := configMap.Data["volcano-scheduler.conf"]
	t.Logf("configmap:\n%s", result)
	assert.NotEmpty(t, result, "ConfigMap should not be empty")
	assert.Contains(t, result, expectedLine)
	assert.NotContains(t, result, "volcano-npu_OLD_linux-aarch64")
}

func TestConfigMapReconcileHooksUpdatesDefaultVolcanoConfigArch(t *testing.T) {
	const schedulerConfig = `actions: "enqueue, allocate"
tiers:
- plugins:
  - name: volcano-npu_v7.3.0_linux-aarch64`
	configMap, r := newVolcanoSchedulerConfigMapArchTestReconciler(t, schedulerConfig)

	err := configMapReconcileHooks{}.transform(context.TODO(), r, configMap)

	require.NoError(t, err)
	assert.Contains(t, configMap.Data["volcano-scheduler.conf"], "volcano-npu_v7.3.0_linux-x86_64")
	assert.NotContains(t, configMap.Data["volcano-scheduler.conf"], "volcano-npu_v7.3.0_linux-aarch64")
}

func TestConfigMapReconcileHooksFallbackOnlyUpdatesVolcanoConfigArch(t *testing.T) {
	const schedulerConfig = `actions: "enqueue, allocate"
tiers:
- plugins:
    - name: "volcano-npu_v7.3.0_linux-aarch64" # built-in plugin`
	configMap, r := newVolcanoSchedulerConfigMapArchTestReconciler(t, schedulerConfig)

	err := configMapReconcileHooks{}.transform(context.TODO(), r, configMap)

	require.NoError(t, err)
	assert.Contains(t, configMap.Data["volcano-scheduler.conf"],
		`    - name: "volcano-npu_v7.3.0_linux-x86_64" # built-in plugin`)
}

func newVolcanoSchedulerConfigMapArchTestReconciler(t *testing.T,
	schedulerConfig string) (*corev1.ConfigMap, *NPUClusterPolicyReconciler) {
	t.Helper()
	const nodeName = "node01"
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      volcanoSchedulerConfigMapName,
			Namespace: "test-ns",
		},
		Data: map[string]string{
			"volcano-scheduler.conf": schedulerConfig,
		},
	}
	s := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	cr := &v1.NPUClusterPolicy{
		TypeMeta: metav1.TypeMeta{
			Kind:       "NPUClusterPolicy",
			APIVersion: v1.GroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
		Spec: v1.NPUClusterPolicySpec{
			Volcano: v1.VolcanoSpec{
				Flavor: v1.VolcanoFlavorMindCluster,
			},
		},
	}
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName},
		Status: corev1.NodeStatus{
			NodeInfo: corev1.NodeSystemInfo{Architecture: "amd64"},
		},
	}
	r := &NPUClusterPolicyReconciler{
		Client:    fake.NewClientBuilder().WithScheme(s).WithObjects(cr, node).Build(),
		Scheme:    s,
		namespace: "test-ns",
		instance:  cr,
	}
	t.Setenv("OPERATOR_NODE_NAME", nodeName)

	return configMap, r
}

func TestConfigMapReconcileHooksTransformVNPUAndDRAConfigs(t *testing.T) {
	s := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	r := &NPUClusterPolicyReconciler{
		Scheme:      s,
		namespace:   "npu-system",
		hasNPUNodes: true,
		instance: &v1.NPUClusterPolicy{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NPUClusterPolicy",
				APIVersion: v1.GroupVersion.String(),
			},
			ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
			Spec: v1.NPUClusterPolicySpec{
				VNPU: v1.VNPUSpec{
					VirtualTemplateConfig: "vnpus:\n- chipName: 910B4",
					NodeModeConfig:        "nodeConfigs:\n- name: worker1\n  vnpu-mode: hard",
				},
				DRA: v1.DRASpec{
					SoftVNPU: v1.DRASoftVNPUSpec{
						ProfileConfig:             "nodes:\n  worker1:\n  - physicalId: 0\n    vnpuMode: soft",
						ChipCapabilities:          "chips:\n- chipName: 910B4",
						ChipCapabilitiesConfigMap: "custom-template",
					},
				},
			},
		},
	}
	h := configMapReconcileHooks{}

	virtualTemplate := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: vnpuVirtualTemplateConfigMapName},
		Data:       map[string]string{"virtual-template-config.yaml": "old"},
	}
	require.NoError(t, h.transform(context.TODO(), r, virtualTemplate))
	assert.Equal(t, "vnpus:\n- chipName: 910B4", virtualTemplate.Data["virtual-template-config.yaml"])

	nodeMode := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: vnpuNodeModeConfigMapName},
		Data:       map[string]string{"node-mode.config.yaml": "old"},
	}
	require.NoError(t, h.transform(context.TODO(), r, nodeMode))
	assert.Equal(t, "nodeConfigs:\n- name: worker1\n  vnpu-mode: hard", nodeMode.Data["node-mode.config.yaml"])

	draProfileConfig := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: draProfileConfigMapName},
		Data:       map[string]string{"dra.config": "old"},
	}
	require.NoError(t, h.transform(context.TODO(), r, draProfileConfig))
	assert.Equal(t, "nodes:\n  worker1:\n  - physicalId: 0\n    vnpuMode: soft", draProfileConfig.Data["dra.config"])

	draChipCapabilities := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: draChipCapabilitiesConfigMapName},
		Data:       map[string]string{"capabilities.yaml": "old"},
	}
	require.NoError(t, h.transform(context.TODO(), r, draChipCapabilities))
	assert.Equal(t, "chips:\n- chipName: 910B4", draChipCapabilities.Data["capabilities.yaml"])
}

func TestDaemonSetReconcileHooksTransformVNPUSettings(t *testing.T) {
	r := vnpuDaemonSetTestReconciler(t)
	tests := []struct {
		name string
		ds   *appsv1.DaemonSet
	}{
		{
			name: "client update",
			ds:   daemonSetWithContainers(vnpuClientUpdateDaemonSetName, corev1.Container{Name: "client"}),
		},
		{
			name: "device plugin",
			ds:   vnpuDevicePluginDaemonSet(),
		},
		{
			name: "exporter",
			ds:   vnpuExporterDaemonSet(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newDaemonSetReconcileHooks(tt.ds)
			require.NoError(t, h.transform(context.TODO(), r, tt.ds))
			assert.Equal(t, appsv1.DaemonSetUpdateStrategyType("OnDelete"), tt.ds.Spec.UpdateStrategy.Type)
			assert.Equal(t, "system-cluster-critical", tt.ds.Spec.Template.Spec.PriorityClassName)
		})
	}

	deviceArgs := tests[1].ds.Spec.Template.Spec.Containers[0].Args
	assert.Contains(t, deviceArgs, "--device-split-count=8")
	assert.Contains(t, deviceArgs, "--logging-console=false")
	assert.NotContains(t, deviceArgs, "--runtime-type=containerd")
	assert.Equal(t, "/custom/Ascend", tests[1].ds.Spec.Template.Spec.Volumes[0].HostPath.Path)
	assert.Equal(t, "/custom/bin/npu-smi", tests[1].ds.Spec.Template.Spec.Volumes[1].HostPath.Path)
}

func vnpuDaemonSetTestReconciler(t *testing.T) *NPUClusterPolicyReconciler {
	s := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	return &NPUClusterPolicyReconciler{
		Scheme:      s,
		namespace:   "npu-system",
		hasNPUNodes: true,
		instance: &v1.NPUClusterPolicy{
			TypeMeta:   metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
			Spec: v1.NPUClusterPolicySpec{VNPU: v1.VNPUSpec{
				UpdateStrategy:    "OnDelete",
				PriorityClassName: "system-cluster-critical",
				DevicePlugin: v1.VNPUDevicePluginSpec{
					DeviceSplitCount: 8,
					LoggingConsole:   false,
					VolumeSpec: v1.VolumesSpec{Volumes: []corev1.Volume{
						hostPathVolume("ascend", "/custom/Ascend"),
						hostPathVolume("npu-smi", "/custom/bin/npu-smi"),
					}},
				},
			}},
		},
	}
}

func daemonSetWithContainers(name string, containers ...corev1.Container) *appsv1.DaemonSet {
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: containers}},
		},
	}
}

func vnpuDevicePluginDaemonSet() *appsv1.DaemonSet {
	ds := daemonSetWithContainers(vnpuDevicePluginDaemonSetName, corev1.Container{
		Name:    "device-plugin",
		Command: []string{placeholderValue},
		Args:    []string{placeholderValue},
	})
	ds.Spec.Template.Spec.Volumes = []corev1.Volume{
		hostPathVolume("ascend", "/usr/local/Ascend"),
		hostPathVolume("npu-smi", "/usr/local/bin/npu-smi"),
	}
	return ds
}

func vnpuExporterDaemonSet() *appsv1.DaemonSet {
	return daemonSetWithContainers(vnpuExporterDaemonSetName, corev1.Container{
		Name: "exporter",
		Env:  []corev1.EnvVar{{Name: "HTTPS_ENABLE", Value: "off"}},
		VolumeMounts: []corev1.VolumeMount{{
			Name:      "tls-secret",
			MountPath: "/opt/xpu/certs",
		}},
	})
}

func hostPathVolume(name, path string) corev1.Volume {
	return corev1.Volume{
		Name: name,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{Path: path},
		},
	}
}

func TestDaemonSetReconcileHooksTransformDRAPaths(t *testing.T) {
	r := draPathTestReconciler(t)
	kubeletPlugin := draKubeletPluginDaemonSet()
	require.NoError(t, newDaemonSetReconcileHooks(kubeletPlugin).transform(context.TODO(), r, kubeletPlugin))

	assertDRAKubeletPluginPaths(t, kubeletPlugin)
	assertDRAKubeletPluginArgs(t, kubeletPlugin)
}

func draPathTestReconciler(t *testing.T) *NPUClusterPolicyReconciler {
	s := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	return &NPUClusterPolicyReconciler{
		Scheme:      s,
		namespace:   "npu-system",
		hasNPUNodes: true,
		instance: &v1.NPUClusterPolicy{
			TypeMeta: metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{
				Name: "cluster",
				UID:  uuid.NewUUID(),
			},
			Spec: v1.NPUClusterPolicySpec{
				DRA: v1.DRASpec{
					CDIRoot:                       "/custom/cdi",
					KubeletRegistrarDirectoryPath: "/custom/registry",
					KubeletPluginsDirectoryPath:   "/custom/plugins",
					DriverName:                    "custom.npu.io",
					DeviceProfile:                 "custom-profile",
					HealthcheckPort:               9999,
					EnableHardVNPU:                true,
					SoftVNPU: v1.DRASoftVNPUSpec{
						Managed:                   true,
						ShareCount:                32,
						SoftShareShmDir:           "/custom/shared",
						SoftShareDevConfigDir:     "/custom/enpu",
						NPUProfileConfigPath:      "/custom/dra.config",
						ChipCapabilitiesConfigMap: "custom-template",
						ChipCapabilitiesNamespace: "custom-ns",
					},
				},
			},
		},
	}
}

func draKubeletPluginDaemonSet() *appsv1.DaemonSet {
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: draKubeletPluginDaemonSetName},
		Spec: appsv1.DaemonSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    "kubeletplugin",
				Command: []string{placeholderValue},
				Args:    []string{placeholderValue},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "kubelet-plugins-registry"},
					{Name: "kubelet-plugins"},
					{Name: "cdi"},
					{Name: "enpu-path"},
					{Name: "shared"},
					{Name: "dra-config"},
				},
			}},
			Volumes: []corev1.Volume{
				{Name: "kubelet-plugins-registry", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{}}},
				{Name: "kubelet-plugins", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{}}},
				{Name: "cdi", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{}}},
				{Name: "enpu-path", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{}}},
				{Name: "shared", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{}}},
				{Name: "dra-config", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{}}},
			},
		}}},
	}
}

func assertDRAKubeletPluginPaths(t *testing.T, kubeletPlugin *appsv1.DaemonSet) {
	assert.Equal(t, "/custom/registry", kubeletPlugin.Spec.Template.Spec.Containers[0].VolumeMounts[0].MountPath)
	assert.Equal(t, "/custom/plugins", kubeletPlugin.Spec.Template.Spec.Containers[0].VolumeMounts[1].MountPath)
	assert.Equal(t, "/custom/cdi", kubeletPlugin.Spec.Template.Spec.Containers[0].VolumeMounts[2].MountPath)
	assert.Equal(t, "/custom/registry", kubeletPlugin.Spec.Template.Spec.Volumes[0].HostPath.Path)
	assert.Equal(t, "/custom/plugins", kubeletPlugin.Spec.Template.Spec.Volumes[1].HostPath.Path)
	assert.Equal(t, "/custom/cdi", kubeletPlugin.Spec.Template.Spec.Volumes[2].HostPath.Path)
	assert.Equal(t, "/custom/enpu", kubeletPlugin.Spec.Template.Spec.Volumes[3].HostPath.Path)
	assert.Equal(t, "/custom/shared", kubeletPlugin.Spec.Template.Spec.Volumes[4].HostPath.Path)
	assert.Equal(t, draProfileConfigMapName, kubeletPlugin.Spec.Template.Spec.Volumes[5].ConfigMap.Name)
}

func assertDRAKubeletPluginArgs(t *testing.T, kubeletPlugin *appsv1.DaemonSet) {
	assert.Contains(t, kubeletPlugin.Spec.Template.Spec.Containers[0].Args, "--driver-name=custom.npu.io")
	assert.Contains(t, kubeletPlugin.Spec.Template.Spec.Containers[0].Args, "--cdi-root=/custom/cdi")
	assert.Contains(t, kubeletPlugin.Spec.Template.Spec.Containers[0].Args, "--share-count=32")
	assert.Contains(t, kubeletPlugin.Spec.Template.Spec.Containers[0].Args, "--npu-profile-config=/custom/dra.config")
	assert.Contains(t, kubeletPlugin.Spec.Template.Spec.Containers[0].Args, "--soft-share-dev-config-dir=/custom/enpu")
	assert.Contains(t, kubeletPlugin.Spec.Template.Spec.Containers[0].Args, "--soft-share-shm-dir=/custom/shared")
	assert.Contains(t, kubeletPlugin.Spec.Template.Spec.Containers[0].Args, "--chip-capabilities-configmap=custom-template")
	assert.Contains(t, kubeletPlugin.Spec.Template.Spec.Containers[0].Args, "--chip-capabilities-namespace=custom-ns")
}

func TestVNPUClientUpdatePlacementGetterIgnoresDRASoftVNPUPlacement(t *testing.T) {
	spec := &v1.NPUClusterPolicySpec{
		VNPU: v1.VNPUSpec{
			Managed: false,
			PlacementSpec: v1.PlacementSpec{NodeSelector: map[string]string{
				"huawei.com/vnpu": "ready",
			}},
			ClientUpdate: v1.VNPUComponentSpec{PlacementSpec: v1.PlacementSpec{NodeSelector: map[string]string{
				"client-update": "true",
			}}},
		},
		DRA: v1.DRASpec{
			Managed: true,
			PlacementSpec: v1.PlacementSpec{NodeSelector: map[string]string{
				"dra": "true",
			}},
			SoftVNPU: v1.DRASoftVNPUSpec{
				Managed: true,
				PlacementSpec: v1.PlacementSpec{NodeSelector: map[string]string{
					"dra-soft-vnpu": "true",
				}},
			},
		},
	}

	placement := vnpuClientUpdatePlacementGetter(spec)

	assert.Equal(t, map[string]string{"client-update": "true"}, placement.NodeSelector)

	spec.VNPU.ClientUpdate.NodeSelector = nil
	placement = vnpuClientUpdatePlacementGetter(spec)

	assert.Equal(t, map[string]string{"huawei.com/vnpu": "ready"}, placement.NodeSelector)
}

func TestDRACommandGettersUseDefaultsWhenFieldsAreEmpty(t *testing.T) {
	spec := &v1.NPUClusterPolicySpec{}

	kubeletCommand := draKubeletPluginCommandParametersGetter(spec)
	assert.Contains(t, kubeletCommand.Args, "--device-profile=npu")
	assert.Contains(t, kubeletCommand.Args, "--driver-name=npu.huawei.com")
	assert.Contains(t, kubeletCommand.Args, "--cdi-root=/var/run/cdi")
	assert.Contains(t, kubeletCommand.Args, "--healthcheck-port=-1")
	assert.Contains(t, kubeletCommand.Args, "--kubelet-registrar-directory-path=/var/lib/kubelet/plugins_registry")
	assert.Contains(t, kubeletCommand.Args, "--kubelet-plugins-directory-path=/var/lib/kubelet/plugins")
	assert.NotContains(t, kubeletCommand.Args, "--share-count=16")

	spec.DRA.SoftVNPU.Managed = true
	kubeletCommand = draKubeletPluginCommandParametersGetter(spec)
	assert.Contains(t, kubeletCommand.Args, "--share-count=16")
	assert.Contains(t, kubeletCommand.Args, "--soft-share-shm-dir=/shared")
	assert.Contains(t, kubeletCommand.Args, "--soft-share-dev-config-dir=/etc/enpu")
	assert.Contains(t, kubeletCommand.Args, "--npu-profile-config=/etc/dra/dra.config")
	assert.Contains(t, kubeletCommand.Args, "--chip-capabilities-configmap=vnpu-template-config")

	installerCommand := draVCANNRTInstallerCommandParametersGetter(spec)
	assert.Equal(t, []string{"/root/acl-client-update.sh"}, installerCommand.Command)

	webhookCommand := draWebhookCommandParametersGetter(spec)
	assert.Contains(t, webhookCommand.Args, "--device-profile=gpu")
	assert.Contains(t, webhookCommand.Args, "--driver-name=npu.huawei.com")
}

func TestDRAKubeletCommandGetterUsesOptionalFlags(t *testing.T) {
	spec := &v1.NPUClusterPolicySpec{DRA: v1.DRASpec{
		NumDevices:     4,
		Kubeconfig:     "/root/.kube/config",
		KubeAPIQPS:     "50",
		KubeAPIBurst:   100,
		FeatureGates:   "ContextualLogging=true",
		DriverName:     "custom.npu.io",
		DeviceProfile:  "gpu",
		EnableHardVNPU: true,
	}}

	command := draKubeletPluginCommandParametersGetter(spec)

	assert.Contains(t, command.Args, "--num-devices=4")
	assert.Contains(t, command.Args, "--kubeconfig=/root/.kube/config")
	assert.Contains(t, command.Args, "--kube-api-qps=50")
	assert.Contains(t, command.Args, "--kube-api-burst=100")
	assert.Contains(t, command.Args, "--feature-gates=ContextualLogging=true")
	assert.NotContains(t, command.Args, "--enable-hard-vnpu")
}

func TestDesiredDRADeviceClassNamesIsEmptyWhenUnmanaged(t *testing.T) {
	spec := &v1.NPUClusterPolicySpec{
		DRA: v1.DRASpec{
			Managed: false,
			DeviceClasses: v1.DRADeviceClassesSpec{
				Managed: true,
				Templates: []v1.DRADeviceClassTemplate{{
					Name:        "custom.npu.io",
					SelectorCEL: `device.driver == "npu.huawei.com"`,
				}},
			},
		},
	}

	assert.Empty(t, desiredDRADeviceClassNames(spec))

	spec.DRA.Managed = true
	spec.DRA.DeviceClasses.Managed = false

	assert.Empty(t, desiredDRADeviceClassNames(spec))
}

func TestDesiredDRADeviceClassNamesMatchUpstreamDefaults(t *testing.T) {
	spec := &v1.NPUClusterPolicySpec{
		DRA: v1.DRASpec{
			Managed:       true,
			DeviceClasses: v1.DRADeviceClassesSpec{Managed: true},
		},
	}

	names := desiredDRADeviceClassNames(spec)

	assert.Equal(t, map[string]bool{
		"full.npu.huawei.com":        true,
		"elastic.npu.huawei.com":     true,
		"fixed.npu.huawei.com":       true,
		"best-effort.npu.huawei.com": true,
		"hard.npu.huawei.com":        true,
	}, names)
}

func TestDRADeviceClassTransformUsesConfiguredDriverName(t *testing.T) {
	s := runtime.NewScheme()
	require.NoError(t, v1.AddToScheme(s))
	r := &NPUClusterPolicyReconciler{
		Scheme: s,
		instance: &v1.NPUClusterPolicy{
			TypeMeta: metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{
				Name: "cluster",
				UID:  uuid.NewUUID(),
			},
			Spec: v1.NPUClusterPolicySpec{
				DRA: v1.DRASpec{DriverName: "custom.npu.io"},
			},
		},
	}
	obj := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "resource.k8s.io/v1",
		"kind":       "DeviceClass",
		"metadata": map[string]any{
			"name": "hard.npu.huawei.com",
		},
		"spec": map[string]any{
			"selectors": []any{map[string]any{"cel": map[string]any{
				"expression": `device.driver == "npu.huawei.com" && device.attributes["npu.huawei.com"].vnpuMode == "hard"`,
			}}},
		},
	}}

	require.NoError(t, unstructuredReconcileHooks{}.transform(context.TODO(), r, obj))
	assert.Equal(t, "hard.custom.npu.io", obj.GetName())
	expression := deviceClassExpression(t, obj)
	assert.Contains(t, expression, `device.driver == "custom.npu.io"`)
	assert.Contains(t, expression, `device.attributes["custom.npu.io"].vnpuMode`)
	assert.NotContains(t, expression, "npu.huawei.com")
}

func TestReconcileDRADeviceClassGarbageDeletesMatchingResourceSlices(t *testing.T) {
	for _, managed := range []bool{true, false} {
		t.Run(fmt.Sprintf("managed=%t", managed), func(t *testing.T) {
			s := runtime.NewScheme()
			s.AddKnownTypeWithName(gvkDeviceClass, &unstructured.Unstructured{})
			s.AddKnownTypeWithName(schema.GroupVersionKind{
				Group:   gvkDeviceClass.Group,
				Version: gvkDeviceClass.Version,
				Kind:    "DeviceClassList",
			}, &unstructured.UnstructuredList{})
			s.AddKnownTypeWithName(gvkResourceSlice, &unstructured.Unstructured{})
			s.AddKnownTypeWithName(schema.GroupVersionKind{
				Group:   gvkResourceSlice.Group,
				Version: gvkResourceSlice.Version,
				Kind:    "ResourceSliceList",
			}, &unstructured.UnstructuredList{})

			instance := &v1.NPUClusterPolicy{
				TypeMeta: metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
				ObjectMeta: metav1.ObjectMeta{
					Name: "cluster",
					UID:  uuid.NewUUID(),
				},
				Spec: v1.NPUClusterPolicySpec{
					DRA: v1.DRASpec{
						Managed: managed,
						DeviceClasses: v1.DRADeviceClassesSpec{
							Managed: true,
						},
					},
				},
			}
			staleDeviceClass := draDeviceClassObject("custom.npu.io", `device.driver == "custom.npu.io"`)
			staleDeviceClass.SetOwnerReferences([]metav1.OwnerReference{*metav1.NewControllerRef(
				instance,
				v1.GroupVersion.WithKind("NPUClusterPolicy"),
			)})
			currentResourceSlice := draResourceSliceObject("m0-npu.huawei.com", "npu.huawei.com")
			staleResourceSlice := draResourceSliceObject("m0-custom.npu.io", "custom.npu.io")
			r := &NPUClusterPolicyReconciler{
				Client: fake.NewClientBuilder().
					WithScheme(s).
					WithObjects(staleDeviceClass, currentResourceSlice, staleResourceSlice).
					Build(),
				instance: instance,
			}

			require.NoError(t, reconcileDRADeviceClassGarbage(context.TODO(), r))

			gotDeviceClass := &unstructured.Unstructured{}
			gotDeviceClass.SetGroupVersionKind(gvkDeviceClass)
			assert.True(t, apierrors.IsNotFound(r.Get(context.TODO(), client.ObjectKey{Name: "custom.npu.io"}, gotDeviceClass)))
			gotStaleResourceSlice := &unstructured.Unstructured{}
			gotStaleResourceSlice.SetGroupVersionKind(gvkResourceSlice)
			assert.True(t, apierrors.IsNotFound(r.Get(context.TODO(), client.ObjectKey{Name: "m0-custom.npu.io"}, gotStaleResourceSlice)))
			gotCurrentResourceSlice := &unstructured.Unstructured{}
			gotCurrentResourceSlice.SetGroupVersionKind(gvkResourceSlice)
			require.NoError(t, r.Get(context.TODO(), client.ObjectKey{Name: "m0-npu.huawei.com"}, gotCurrentResourceSlice))

		})
	}
}

func deviceClassExpression(t *testing.T, obj *unstructured.Unstructured) string {
	t.Helper()
	selectors, found, err := unstructured.NestedSlice(obj.Object, "spec", "selectors")
	require.NoError(t, err)
	require.True(t, found)
	require.NotEmpty(t, selectors)
	selector, ok := selectors[0].(map[string]any)
	require.True(t, ok)
	cel, ok := selector["cel"].(map[string]any)
	require.True(t, ok)
	expression, ok := cel["expression"].(string)
	require.True(t, ok)
	return expression
}

func draDeviceClassObject(name string, expression string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "resource.k8s.io/v1",
		"kind":       "DeviceClass",
		"metadata": map[string]any{
			"name": name,
			"annotations": map[string]any{
				draDeviceClassManagedAnnotation: "true",
			},
		},
		"spec": map[string]any{
			"selectors": []any{map[string]any{"cel": map[string]any{
				"expression": expression,
			}}},
		},
	}}
	obj.SetGroupVersionKind(gvkDeviceClass)
	return obj
}

func draResourceSliceObject(name string, driver string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "resource.k8s.io/v1",
		"kind":       "ResourceSlice",
		"metadata": map[string]any{
			"name": name,
		},
		"spec": map[string]any{
			"driver": driver,
		},
	}}
	obj.SetGroupVersionKind(gvkResourceSlice)
	return obj
}

func TestBuildDRADeviceClassTemplateObjects(t *testing.T) {
	objects, err := buildDRADeviceClassTemplateObjects([]v1.DRADeviceClassTemplate{{
		Name:        "custom.npu.io",
		SelectorCEL: `device.driver == "custom.npu.io"`,
	}})

	require.NoError(t, err)
	require.Len(t, objects, 1)
	assert.Equal(t, "custom.npu.io", objects[0].GetName())
	assert.Equal(t, `device.driver == "custom.npu.io"`, deviceClassExpression(t, objects[0]))
}
