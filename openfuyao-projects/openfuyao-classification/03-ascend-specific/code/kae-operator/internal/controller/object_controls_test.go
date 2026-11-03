/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	promv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/stretchr/testify/assert"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/constants"
)

var testKAEPolicy = &kaev1.KAEPolicy{
	ObjectMeta: metav1.ObjectMeta{Name: "test-kaepolicy"},
	Spec: kaev1.KAEPolicySpec{
		Operator: kaev1.OperatorSpec{EnabledHPRE: BoolPtr(true)},
		DaemonSets: kaev1.DaemonSetsSpec{
			Labels:            map[string]string{"testLabel": "true"},
			Annotations:       map[string]string{"testAnnotations": "true"},
			Tolerations:       []corev1.Toleration{{Key: "test", Value: "", Operator: "Equal", Effect: "NoSchedule"}},
			PriorityClassName: "system-node-critical",
			UpdateStrategy:    "RollingUpdate",
			RollingUpdate:     &kaev1.RollingUpdateSpec{MaxUnavailable: "1"},
		},
		Driver: kaev1.DriverSpec{
			Enabled:          BoolPtr(true),
			EnabledCleanup:   BoolPtr(true),
			Repository:       "cr.openfuyao.cn/openfuyao",
			Image:            "kae-driver-installer",
			Version:          "latest",
			ImagePullPolicy:  "Always",
			ImagePullSecrets: []string{"pull-secret-1", "pull-secret-2"},
			Resources: &kaev1.ResourceRequirements{
				Limits:   map[corev1.ResourceName]resource.Quantity{"cpu": resource.MustParse("500m")},
				Requests: map[corev1.ResourceName]resource.Quantity{"cpu": resource.MustParse("200m")},
			},
			Args: []string{"test"},
			Env:  []kaev1.EnvVar{{Name: "INSTALL_POLICY", Value: "IfNotPresent"}},
		},
		DevicePlugin: kaev1.DevicePluginSpec{
			EnabledHPRE:     BoolPtr(true),
			EnabledZIP:      BoolPtr(true),
			EnabledSEC:      BoolPtr(false),
			Repository:      "cr.openfuyao.cn/openfuyao",
			Image:           "kae-device-plugin",
			Version:         "latest",
			ImagePullPolicy: "Always",
		},
		PSA: kaev1.PSASpec{Enabled: BoolPtr(true)},
	},
}

var testNs = &corev1.Namespace{
	TypeMeta: metav1.TypeMeta{
		APIVersion: "v1",
		Kind:       "Namespace",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "default",
		Labels: map[string]string{
			"name": "default",
		},
	},
}

var (
	testControllerNodes = []*corev1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node3", Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present":   "true",
					"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node4", Labels: map[string]string{
					"openfuyao.com/kae.present":                               "true",
					"openfuyao.com/kae.deploy.driver":                         "true",
					"openfuyao.com/kae.sec.deploy.device-plugin":              "true",
					"openfuyao.com/kae.zip.deploy.device-plugin":              "true",
					"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
	}
)

func TestObjectController(t *testing.T) {
	ctx := context.TODO()
	testScheme := scheme.Scheme
	initScheme(testScheme)

	kaePolicyManager = KAEPolicyManager{}
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Scheme: testScheme,
	}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	err := os.Setenv("OPERATOR_NAMESPACE", "default")
	assert.Nil(t, err)
	// 确保测试结束后清理环境变量
	defer os.Unsetenv("OPERATOR_NAMESPACE")

	for _, node := range testControllerNodes {
		err = r.Client.Create(ctx, node)
		assert.Nil(t, err)
	}
	_ = r.Client.Create(ctx, testNs)

	patches.ApplyGlobalVar(&componentPaths, []string{
		"../../config/rbac",
		"../../config/default/metrics_service.yaml",
	})
	err = kaePolicyManager.init(ctx, r, testKAEPolicy)
	assert.Nil(t, err)
	req := reconcile.Request{
		NamespacedName: client.ObjectKey{
			Namespace: "default",
			Name:      "test-kaepolicy",
		},
	}
	overallStatus, statesNotReady, err := r.deployAllComponents(ctx, req, testKAEPolicy)
	assert.Nil(t, err)
	assert.NotNil(t, overallStatus)
	assert.Nil(t, statesNotReady)
}

func initScheme(testScheme *runtime.Scheme) {
	utilruntime.Must(scheme.AddToScheme(testScheme))
	utilruntime.Must(kaev1.AddToScheme(testScheme))
	utilruntime.Must(promv1.AddToScheme(testScheme))
	utilruntime.Must(apiextensionsv1.AddToScheme(testScheme))
}

func TestCreateOrUpdateOrDeleteObject(t *testing.T) {
	newScheme := runtime.NewScheme()
	_ = corev1.AddToScheme(newScheme)
	fakeClient := fake.NewClientBuilder().WithScheme(newScheme).Build()
	mgr := KAEPolicyManager{
		client:      fakeClient,
		scheme:      newScheme,
		kaeInstance: testKAEPolicy,
		namespace:   "default",
	}

	obj := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-configmap",
		},
		Data: map[string]string{
			"foo": "bar",
		},
	}

	// Scenario 1: setObjReferenceToManager returns an error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(controllerutil.SetControllerReference, func(owner, obj metav1.Object, scheme *runtime.Scheme) error {
		return fmt.Errorf("mock setObjReferenceToManager error")
	})
	stat, err := createOrUpdateObject(mgr, obj)
	assert.NotNil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)
	patches.Reset()

	patches.ApplyFunc(controllerutil.SetControllerReference, func(owner, obj metav1.Object, scheme *runtime.Scheme) error {
		return nil
	})
	// // Scenario 2: create successful
	stat, err = createOrUpdateObject(mgr, obj)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.Ready, stat)

	// Scenario 3: create Role failed due to AlreadyExists, and update failed
	stat, err = createOrUpdateObject(mgr, obj)
	assert.NotNil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)

	// Scenario 4: create Role failed due to AlreadyExists, and update successful
	obj.ObjectMeta.ResourceVersion = ""
	stat, err = createOrUpdateObject(mgr, obj)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.Ready, stat)
}

func TestDeleteObject(t *testing.T) {
	newScheme := runtime.NewScheme()
	_ = corev1.AddToScheme(newScheme)
	obj := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-configmap",
		},
		Data: map[string]string{
			"foo": "bar",
		},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(newScheme).WithRuntimeObjects(obj).Build()
	mgr := KAEPolicyManager{
		client:      fakeClient,
		scheme:      newScheme,
		kaeInstance: testKAEPolicy,
		namespace:   "default",
	}

	// Scenario 1: Delete successful
	stat, err := deleteObject(mgr, obj)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.Disabled, stat)

	// Scenario 2: Delete failed
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(errors.IsNotFound, func(err error) bool {
		return false
	})
	stat, err = deleteObject(mgr, obj)
	assert.NotNil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)
}

func TestGetAndUpdateServiceOrCreate(t *testing.T) {
	newScheme := runtime.NewScheme()
	_ = corev1.AddToScheme(newScheme)
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-service",
			Namespace: "default",
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": "test"},
			Ports: []corev1.ServicePort{
				{Port: 80},
			},
		},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(newScheme).WithRuntimeObjects(svc).Build()
	mgr := KAEPolicyManager{
		client:      fakeClient,
		scheme:      newScheme,
		kaeInstance: testKAEPolicy,
		namespace:   "default",
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	// Scenario 1: setObjReferenceToManager returns an error
	patches.ApplyFunc(controllerutil.SetControllerReference, func(owner, obj metav1.Object, scheme *runtime.Scheme) error {
		return fmt.Errorf("mock setObjReferenceToManager error")
	})
	stat, err := getAndUpdateServiceOrCreate(mgr, svc)
	assert.NotNil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)
	patches.Reset()

	patches.ApplyFunc(controllerutil.SetControllerReference, func(owner, obj metav1.Object, scheme *runtime.Scheme) error {
		return nil
	})

	// Scenario 2: Service NotFound but Create failed
	svcTest := svc.DeepCopy()
	svcTest.ObjectMeta.Name = "test-service-new"
	svcTest.ResourceVersion = "1"
	stat, err = getAndUpdateServiceOrCreate(mgr, svcTest)
	assert.NotNil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)

	// Scenario 3: Service NotFound and Create sucess
	svcTest = svc.DeepCopy()
	svcTest.ObjectMeta.Name = "test-service-new"
	svcTest.ResourceVersion = ""
	stat, err = getAndUpdateServiceOrCreate(mgr, svcTest)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.Ready, stat)

	// Scenario 4: Service found and Update success
	svc.Spec.Ports[0].Port = 8080
	stat, err = getAndUpdateServiceOrCreate(mgr, svc)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.Ready, stat)
}

func TestDeploymentHandlers(t *testing.T) {
	mgr := KAEPolicyManager{
		client:      fake.NewClientBuilder().Build(),
		scheme:      runtime.NewScheme(),
		kaeInstance: testKAEPolicy,
		namespace:   "default",
		resources: []Resources{
			{
				Deployment: appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test-deployment",
					},
				},
			},
		},
		componentNames: []string{"test-deployment"},
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyPrivateMethod(&mgr, "isComponentEnabled", func(componentName string) bool {
		return false
	})

	stat, err := HandlerDeployment(mgr)
	assert.NotNil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)

}

func TestIsDeploymentPodReady(t *testing.T) {
	newScheme := runtime.NewScheme()
	_ = corev1.AddToScheme(newScheme)
	deploy := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-deployment",
		},
	}
	podPending := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-pod-pending",
			Labels: map[string]string{"app": "pending"},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodPending,
		},
	}
	runningPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-pod-running",
			Labels: map[string]string{"app": "running"},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
		},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(newScheme).WithRuntimeObjects(podPending, runningPod).Build()
	mgr := KAEPolicyManager{
		client:      fakeClient,
		scheme:      newScheme,
		kaeInstance: testKAEPolicy,
		namespace:   "default",
	}

	stat, err := isDeploymentPodReady(mgr, deploy)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)

	deploy.Spec.Template.ObjectMeta.Labels = map[string]string{"app": "non-existing"}
	stat, err = isDeploymentPodReady(mgr, deploy)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)

	deploy.Spec.Template.ObjectMeta.Labels = map[string]string{"app": "pending"}
	stat, err = isDeploymentPodReady(mgr, deploy)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.NotReady, stat)

	deploy.Spec.Template.ObjectMeta.Labels = map[string]string{"app": "running"}
	stat, err = isDeploymentPodReady(mgr, deploy)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.Ready, stat)
}

func TestAddDaemonsetLabelsAndAnnotations(t *testing.T) {
	obj := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{},
	}
	addDaemonsetLabels(obj, map[string]string{
		"test1": "test1",
	})
	assert.Equal(t, "test1", obj.Labels["test1"])
	addDaemonsetAnnotations(obj, map[string]string{
		"test2": "test2",
	})
	assert.Equal(t, "test2", obj.Annotations["test2"])
}

func TestNoNeedDeploy(t *testing.T) {
	mgr := KAEPolicyManager{
		hasKAENodes: false,
		kaeInstance: &kaev1.KAEPolicy{
			ObjectMeta: metav1.ObjectMeta{Name: "test-kaepolicy"},
			Spec: kaev1.KAEPolicySpec{
				Operator: kaev1.OperatorSpec{
					EnabledHPRE: BoolPtr(false),
					EnabledSEC:  BoolPtr(false),
					EnabledZIP:  BoolPtr(false),
				},
			},
		},
		resources: []Resources{
			{
				DaemonSet: appsv1.DaemonSet{
					ObjectMeta: metav1.ObjectMeta{
						Name: "",
					},
				},
			},
		},
	}

	assert.True(t, noNeedDeploy(mgr))

	mgr.hasKAENodes = true
	mgr.resources[0].DaemonSet.ObjectMeta.Name = constants.HpreDevicePluginDaemonSetName
	assert.True(t, noNeedDeploy(mgr))

	mgr.hasKAENodes = true
	mgr.resources[0].DaemonSet.ObjectMeta.Name = constants.SecDevicePluginDaemonSetName
	assert.True(t, noNeedDeploy(mgr))

	mgr.hasKAENodes = true
	mgr.resources[0].DaemonSet.ObjectMeta.Name = constants.ZipDevicePluginDaemonSetName
	assert.True(t, noNeedDeploy(mgr))

	mgr.resources[0].DaemonSet.ObjectMeta.Name = "test-deployment"
	assert.False(t, noNeedDeploy(mgr))
}

func TestPreProcessDaemonSet(t *testing.T) {
	mgr := KAEPolicyManager{
		hasKAENodes: false,
		kaeInstance: testKAEPolicy,
	}

	tempDaemonSet := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-deployment",
		},
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
						},
					},
					InitContainers: []corev1.Container{
						{
							Name: "test-init-container",
						},
					},
				},
			},
		},
	}

	testDaemonSet := tempDaemonSet.DeepCopy()
	err := preProcessDaemonSet(testDaemonSet, mgr)
	assert.NotNil(t, err)

	testDaemonSet = tempDaemonSet.DeepCopy()
	testDaemonSet.Name = constants.DriverDaemonSetName
	err = preProcessDaemonSet(testDaemonSet, mgr)
	assert.Nil(t, err)

	testDaemonSet = tempDaemonSet.DeepCopy()
	testDaemonSet.Name = constants.HpreDevicePluginDaemonSetName
	err = preProcessDaemonSet(testDaemonSet, mgr)
	assert.Nil(t, err)
}

func TestGetDaemonSetHash(t *testing.T) {
	daemonSet := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-daemonset",
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test"},
			},
		},
	}
	hash := getDaemonSetHash(daemonSet)
	assert.NotEmpty(t, hash)
	assert.NotEqual(t, "", hash)
}

func TestIsAllContainersReady(t *testing.T) {
	// Test with all containers ready
	pod := corev1.Pod{
		Status: corev1.PodStatus{
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "c1", Ready: true},
				{Name: "c2", Ready: true},
			},
		},
	}
	assert.True(t, isAllContainersReady(pod))

	// Test with one container not ready
	pod = corev1.Pod{
		Status: corev1.PodStatus{
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "c1", Ready: true},
				{Name: "c2", Ready: false},
			},
		},
	}
	assert.False(t, isAllContainersReady(pod))

	// Test with empty container statuses
	pod = corev1.Pod{
		Status: corev1.PodStatus{
			ContainerStatuses: []corev1.ContainerStatus{},
		},
	}
	assert.True(t, isAllContainersReady(pod))
}

func TestFilterDaemonSetOwnedPods(t *testing.T) {
	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-daemonset",
			UID:  "ds-uid-123",
		},
	}
	pods := []corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod1",
				OwnerReferences: []metav1.OwnerReference{
					{UID: "ds-uid-123"},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod2",
				OwnerReferences: []metav1.OwnerReference{
					{UID: "other-uid"},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod3",
			},
		},
	}

	filtered := filterDaemonSetOwnedPods(ds, pods)
	assert.Equal(t, 1, len(filtered))
	assert.Equal(t, "pod1", filtered[0].Name)
}

func TestFilterDaemonSetOwnedPodsIgnoresEmptyOwnerReferences(t *testing.T) {
	daemonSet := &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{UID: "daemonset-uid"}}
	pods := []corev1.Pod{
		{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{}}},
		{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{
			{UID: "another-owner"},
			{UID: daemonSet.UID},
		}}},
	}

	filteredPods := filterDaemonSetOwnedPods(daemonSet, pods)
	assert.Len(t, filteredPods, 1)
}

func TestIsDaemonSetSpecChanged(t *testing.T) {
	// Test with old nil, new not nil
	result := isDaemonSetSpecChanged(nil, &appsv1.DaemonSet{})
	assert.True(t, result)

	// Test with both nil
	result = isDaemonSetSpecChanged(nil, nil)
	assert.False(t, result)

	// Test with old not nil, new nil
	result = isDaemonSetSpecChanged(&appsv1.DaemonSet{}, nil)
	assert.False(t, result)

	// Test with both non-nil but annotations nil (should return false per logic)
	oldDS := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				"test": "value",
			},
		},
	}
	// Add hash annotation to make them equivalent
	oldDS.ObjectMeta.Annotations[annotationKeyLastAppliedHash] = "hash1"
	newDS := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				"test": "value",
			},
		},
	}
	// Add same hash to new DS
	newDS.ObjectMeta.Annotations[annotationKeyLastAppliedHash] = "hash1"
	result = isDaemonSetSpecChanged(oldDS, newDS)
	assert.True(t, result)

	// Test with different hashes
	newDS.ObjectMeta.Annotations = map[string]string{
		"test": "different",
	}
	result = isDaemonSetSpecChanged(oldDS, newDS)
	assert.True(t, result)
}
