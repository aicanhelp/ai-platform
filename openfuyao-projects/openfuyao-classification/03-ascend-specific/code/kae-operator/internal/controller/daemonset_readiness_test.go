/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of the Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/constants"
)

func newDaemonSetManager(objects ...client.Object) KAEPolicyManager {
	scheme := runtime.NewScheme()
	if err := appsv1.AddToScheme(scheme); err != nil {
		panic(err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		panic(err)
	}
	if err := kaev1.AddToScheme(scheme); err != nil {
		panic(err)
	}

	return KAEPolicyManager{
		ctx:       context.Background(),
		namespace: "default",
		client:    fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build(),
		scheme:    scheme,
	}
}

func TestIsDaemonSetReadyReportsObservedStatus(t *testing.T) {
	testCases := []struct {
		name      string
		daemonSet *appsv1.DaemonSet
		wantState kaev1.State
		wantError bool
	}{
		{
			name:      "missing daemon set",
			daemonSet: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: "missing", Namespace: "default"}},
			wantState: kaev1.NotReady,
			wantError: true,
		},
		{
			name: "no desired pods",
			daemonSet: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "no-pods", Namespace: "default"},
			},
			wantState: kaev1.NotReady,
		},
		{
			name: "unavailable pods",
			daemonSet: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "unavailable", Namespace: "default"},
				Status:     appsv1.DaemonSetStatus{DesiredNumberScheduled: 1, NumberUnavailable: 1},
			},
			wantState: kaev1.NotReady,
		},
		{
			name: "rolling update is ready",
			daemonSet: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "ready", Namespace: "default"},
				Status:     appsv1.DaemonSetStatus{DesiredNumberScheduled: 1},
				Spec: appsv1.DaemonSetSpec{
					UpdateStrategy: appsv1.DaemonSetUpdateStrategy{Type: appsv1.RollingUpdateDaemonSetStrategyType},
				},
			},
			wantState: kaev1.Ready,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			manager := newDaemonSetManager()
			if !testCase.wantError {
				manager = newDaemonSetManager(testCase.daemonSet)
			}
			state, err := isDaemonSetReady(testCase.daemonSet.Name, manager)

			assert.Equal(t, testCase.wantState, state)
			assert.Equal(t, testCase.wantError, err != nil)
		})
	}
}

func TestIsDaemonSetReadyChecksOnDeletePodRevision(t *testing.T) {
	const (
		daemonSetName = "kae-device-plugin"
		revision      = "latest"
	)
	labels := map[string]string{"app": daemonSetName}
	daemonSet := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: daemonSetName, Namespace: "default", UID: types.UID("daemonset-uid")},
		Spec: appsv1.DaemonSetSpec{
			Selector:       &metav1.LabelSelector{MatchLabels: labels},
			Template:       corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: labels}},
			UpdateStrategy: appsv1.DaemonSetUpdateStrategy{Type: appsv1.OnDeleteDaemonSetStrategyType},
		},
		Status: appsv1.DaemonSetStatus{DesiredNumberScheduled: 1},
	}
	controllerRevision := &appsv1.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{Name: daemonSetName + "-" + revision, Namespace: "default", Labels: labels},
		Revision:   2,
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kae-pod",
			Namespace: "default",
			Labels: map[string]string{
				"app":               daemonSetName,
				labelKeyCtrlRevHash: revision,
			},
			OwnerReferences: []metav1.OwnerReference{{UID: daemonSet.UID}},
		},
		Status: corev1.PodStatus{
			Phase:             corev1.PodRunning,
			ContainerStatuses: []corev1.ContainerStatus{{Name: "plugin", Ready: true}},
		},
	}

	state, err := isDaemonSetReady(daemonSetName, newDaemonSetManager(daemonSet, controllerRevision, pod))
	assert.NoError(t, err)
	assert.Equal(t, kaev1.Ready, state)
}

func TestIsDaemonSetReadyRejectsOnDeleteDaemonSetWithoutPods(t *testing.T) {
	daemonSet := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: "on-delete", Namespace: "default"},
		Spec: appsv1.DaemonSetSpec{
			UpdateStrategy: appsv1.DaemonSetUpdateStrategy{Type: appsv1.OnDeleteDaemonSetStrategyType},
		},
		Status: appsv1.DaemonSetStatus{DesiredNumberScheduled: 1},
	}

	state, err := isDaemonSetReady(daemonSet.Name, newDaemonSetManager(daemonSet))
	assert.NoError(t, err)
	assert.Equal(t, kaev1.NotReady, state)
}

func TestHandlerConfigMapsCreatesEachConfiguredMap(t *testing.T) {
	manager := newDaemonSetManager()
	policy := testKAEPolicy.DeepCopy()
	policy.Namespace = "default"
	manager.kaeInstance = policy
	manager.componentNames = []string{constants.ComponentDriverPluginHpre}
	manager.resources = []Resources{{
		ConfigMaps: []corev1.ConfigMap{
			{ObjectMeta: metav1.ObjectMeta{Name: "first-config"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "second-config"}},
		},
	}}

	state, err := HandlerConfigMaps(manager)
	assert.NoError(t, err)
	assert.Equal(t, kaev1.Ready, state)

	for _, name := range []string{"first-config", "second-config"} {
		configMap := &corev1.ConfigMap{}
		err := manager.client.Get(manager.ctx, client.ObjectKey{Namespace: "default", Name: name}, configMap)
		assert.NoError(t, err)
	}
}

func TestHandlerConfigMapsReportsDisabledWhenComponentIsNotEnabled(t *testing.T) {
	manager := newDaemonSetManager()
	manager.kaeInstance = testKAEPolicy.DeepCopy()
	manager.componentNames = []string{"unsupported-component"}
	manager.resources = []Resources{{
		ConfigMaps: []corev1.ConfigMap{{ObjectMeta: metav1.ObjectMeta{Name: "disabled-config"}}},
	}}

	state, err := HandlerConfigMaps(manager)
	assert.NoError(t, err)
	assert.Equal(t, kaev1.NotReady, state)
}

func TestIsAllDaemonSetPodReadyRejectsPendingPod(t *testing.T) {
	labels := map[string]string{"app": "kae"}
	daemonSet := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: "kae", Namespace: "default", UID: types.UID("daemonset-uid")},
		Spec:       appsv1.DaemonSetSpec{Selector: &metav1.LabelSelector{MatchLabels: labels}},
	}
	revision := &appsv1.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{Name: "kae-latest", Namespace: "default", Labels: labels},
		Revision:   1,
	}
	pod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{UID: daemonSet.UID}}},
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	state, err := isAllDaemonSetPodReady(daemonSet, &corev1.PodList{Items: []corev1.Pod{pod}},
		newDaemonSetManager(revision))
	assert.NoError(t, err)
	assert.Equal(t, kaev1.NotReady, state)
}

func TestGetCtrlRevisionHashReturnsErrorWithoutMatchingRevision(t *testing.T) {
	daemonSet := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: "kae", Namespace: "default"},
		Spec:       appsv1.DaemonSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "kae"}}},
	}

	_, err := getCtrlRevisionHash(context.Background(), daemonSet, newDaemonSetManager())
	assert.Error(t, err)
}

func TestCreateDaemonSetStoresAppliedHash(t *testing.T) {
	manager := newDaemonSetManager()
	daemonSet := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "kae-device-plugin",
			Namespace:   "default",
			Annotations: map[string]string{},
		},
	}

	state, err := createDaemonSet(manager, daemonSet)
	assert.NoError(t, err)
	assert.Equal(t, kaev1.NotReady, state)
	assert.NotEmpty(t, daemonSet.Annotations[annotationKeyLastAppliedHash])

	storedDaemonSet := &appsv1.DaemonSet{}
	err = manager.client.Get(manager.ctx, client.ObjectKey{Namespace: "default", Name: daemonSet.Name}, storedDaemonSet)
	assert.NoError(t, err)
}

func TestHandlerDaemonSetSkipsDeploymentWithoutKAENodes(t *testing.T) {
	manager := newDaemonSetManager()
	manager.kaeInstance = testKAEPolicy.DeepCopy()
	manager.componentNames = []string{constants.ComponentDriverPluginHpre}
	manager.resources = []Resources{{
		DaemonSet: appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: constants.HpreDevicePluginDaemonSetName}},
	}}

	state, err := HandlerDaemonSet(manager)
	assert.NoError(t, err)
	assert.Equal(t, kaev1.Ready, state)
}

func TestIsDeploymentReadyReportsMissingAndUnavailableDeployments(t *testing.T) {
	missingDeployment := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "missing", Namespace: "default"}}
	state, err := isDeploymentReady(newDaemonSetManager(), missingDeployment)
	assert.Error(t, err)
	assert.Equal(t, kaev1.NotReady, state)

	unavailableDeployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "unavailable", Namespace: "default"},
		Status:     appsv1.DeploymentStatus{UnavailableReplicas: 1},
	}
	state, err = isDeploymentReady(newDaemonSetManager(unavailableDeployment), unavailableDeployment)
	assert.NoError(t, err)
	assert.Equal(t, kaev1.NotReady, state)
}

func TestIsDeploymentReadyReportsRunningPods(t *testing.T) {
	labels := map[string]string{"app": "kae"}
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "ready", Namespace: "default"},
		Spec:       appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: labels}}},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "ready-pod", Namespace: "default", Labels: labels},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning},
	}

	state, err := isDeploymentReady(newDaemonSetManager(deployment, pod), deployment)
	assert.NoError(t, err)
	assert.Equal(t, kaev1.Ready, state)
}

func TestHandlerDaemonSetDisablesUnsupportedComponent(t *testing.T) {
	manager := newDaemonSetManager()
	manager.kaeInstance = testKAEPolicy.DeepCopy()
	manager.componentNames = []string{"unsupported-component"}
	manager.resources = []Resources{{
		DaemonSet: appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: "disabled-daemon-set"}},
	}}

	state, err := HandlerDaemonSet(manager)
	assert.NoError(t, err)
	assert.Equal(t, kaev1.Disabled, state)
}

func TestApplyUpdateStrategyConfig(t *testing.T) {
	testCases := []struct {
		name          string
		daemonSet     *appsv1.DaemonSet
		configuration *kaev1.DaemonSetsSpec
		wantError     bool
		wantType      appsv1.DaemonSetUpdateStrategyType
	}{
		{
			name:          "on delete",
			daemonSet:     &appsv1.DaemonSet{},
			configuration: &kaev1.DaemonSetsSpec{UpdateStrategy: "OnDelete"},
			wantType:      appsv1.OnDeleteDaemonSetStrategyType,
		},
		{
			name:          "rolling update without maximum",
			daemonSet:     &appsv1.DaemonSet{},
			configuration: &kaev1.DaemonSetsSpec{UpdateStrategy: "RollingUpdate"},
		},
		{
			name:      "rolling update percentage",
			daemonSet: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: "device-plugin"}},
			configuration: &kaev1.DaemonSetsSpec{
				UpdateStrategy: "RollingUpdate",
				RollingUpdate:  &kaev1.RollingUpdateSpec{MaxUnavailable: "25%"},
			},
			wantType: appsv1.RollingUpdateDaemonSetStrategyType,
		},
		{
			name:      "invalid maximum unavailable",
			daemonSet: &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: "device-plugin"}},
			configuration: &kaev1.DaemonSetsSpec{
				UpdateStrategy: "RollingUpdate",
				RollingUpdate:  &kaev1.RollingUpdateSpec{MaxUnavailable: "invalid"},
			},
			wantError: true,
		},
		{
			name:          "unsupported strategy",
			daemonSet:     &appsv1.DaemonSet{},
			configuration: &kaev1.DaemonSetsSpec{UpdateStrategy: "unsupported"},
			wantError:     true,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			err := applyUpdateStrategyConfig(testCase.daemonSet, testCase.configuration)
			assert.Equal(t, testCase.wantError, err != nil)
			assert.Equal(t, testCase.wantType, testCase.daemonSet.Spec.UpdateStrategy.Type)
		})
	}
}

func TestCreateIntOrStringFromUnavailable(t *testing.T) {
	percentage, err := createIntOrStringFromUnavailable("10%")
	assert.NoError(t, err)
	assert.Equal(t, intstr.String, percentage.Type)

	count, err := createIntOrStringFromUnavailable("2")
	assert.NoError(t, err)
	assert.Equal(t, int32(2), count.IntVal)

	_, err = createIntOrStringFromUnavailable("invalid")
	assert.Error(t, err)
}

func TestFilterAndSortControllerRevisions(t *testing.T) {
	revisions := []appsv1.ControllerRevision{
		{ObjectMeta: metav1.ObjectMeta{Name: "other-1"}, Revision: 3},
		{ObjectMeta: metav1.ObjectMeta{Name: "kae-1"}, Revision: 1},
		{ObjectMeta: metav1.ObjectMeta{Name: "kae-2"}, Revision: 2},
	}

	filtered, err := filterAndSortControllerRevisions(revisions, "kae")
	assert.NoError(t, err)
	assert.Len(t, filtered, 2)
	assert.Equal(t, "kae-2", filtered[0].Name)

	_, err = filterAndSortControllerRevisions(revisions, "missing")
	assert.Error(t, err)
}
