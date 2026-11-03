/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package upgrade

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureBKENodeWebhooks(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, admissionregistrationv1.AddToScheme(scheme))
	require.NoError(t, appsv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	replicas := int32(1)
	mutateCA := []byte("mutate-live-ca")
	validateCA := []byte("validate-live-ca")
	mutateClusterPath := "/mutate-bke-bocloud-com-v1beta1-bkecluster"
	validateClusterPath := "/validate-bke-bocloud-com-v1beta1-bkecluster"
	fail := admissionregistrationv1.Fail
	none := admissionregistrationv1.SideEffectClassNone

	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		readyProviderDeploy(replicas, "img:v26.09"),
		readyProviderPod("img:v26.09"),
		&admissionregistrationv1.MutatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: bkeMutatingWebhookConfigurationName},
			Webhooks: []admissionregistrationv1.MutatingWebhook{{
				Name:                    bkeMutatingWebhookClusterName,
				AdmissionReviewVersions: []string{"v1"},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					CABundle: mutateCA,
					Service: &admissionregistrationv1.ServiceReference{
						Name:      bkeWebhookServiceName,
						Namespace: bkeWebhookServiceNamespace,
						Path:      &mutateClusterPath,
					},
				},
				FailurePolicy: &fail,
				SideEffects:   &none,
			}},
		},
		&admissionregistrationv1.ValidatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: bkeValidatingWebhookConfigurationName},
			Webhooks: []admissionregistrationv1.ValidatingWebhook{{
				Name:                    bkeValidatingWebhookClusterName,
				AdmissionReviewVersions: []string{"v1"},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					CABundle: validateCA,
					Service: &admissionregistrationv1.ServiceReference{
						Name:      bkeWebhookServiceName,
						Namespace: bkeWebhookServiceNamespace,
						Path:      &validateClusterPath,
					},
				},
				FailurePolicy: &fail,
				SideEffects:   &none,
			}},
		},
	).Build()

	require.NoError(t, EnsureBKENodeWebhooks(context.Background(), c))

	mwc := &admissionregistrationv1.MutatingWebhookConfiguration{}
	require.NoError(t, c.Get(context.Background(), types.NamespacedName{Name: bkeMutatingWebhookConfigurationName}, mwc))
	require.True(t, mutatingWebhookHasName(mwc.Webhooks, bkeMutatingWebhookNodeName))
	var mutateNode *admissionregistrationv1.MutatingWebhook
	for i := range mwc.Webhooks {
		if mwc.Webhooks[i].Name == bkeMutatingWebhookNodeName {
			mutateNode = &mwc.Webhooks[i]
			break
		}
	}
	require.NotNil(t, mutateNode)
	assert.Equal(t, mutateCA, mutateNode.ClientConfig.CABundle)
	require.NotNil(t, mutateNode.ClientConfig.Service)
	assert.Equal(t, bkeMutateBKENodePath, *mutateNode.ClientConfig.Service.Path)

	vwc := &admissionregistrationv1.ValidatingWebhookConfiguration{}
	require.NoError(t, c.Get(context.Background(), types.NamespacedName{Name: bkeValidatingWebhookConfigurationName}, vwc))
	require.True(t, validatingWebhookHasName(vwc.Webhooks, bkeValidatingWebhookNodeName))
	var validateNode *admissionregistrationv1.ValidatingWebhook
	for i := range vwc.Webhooks {
		if vwc.Webhooks[i].Name == bkeValidatingWebhookNodeName {
			validateNode = &vwc.Webhooks[i]
			break
		}
	}
	require.NotNil(t, validateNode)
	assert.Equal(t, validateCA, validateNode.ClientConfig.CABundle)
	require.NotNil(t, validateNode.ClientConfig.Service)
	assert.Equal(t, bkeValidateBKENodePath, *validateNode.ClientConfig.Service.Path)
	require.Len(t, validateNode.Rules, 1)
	assert.Contains(t, validateNode.Rules[0].Operations, admissionregistrationv1.Delete)

	// Idempotent.
	require.NoError(t, EnsureBKENodeWebhooks(context.Background(), c))
}

func TestEnsureBKENodeWebhooks_RequiresMutatingClusterWebhook(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, admissionregistrationv1.AddToScheme(scheme))
	require.NoError(t, appsv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	replicas := int32(1)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		readyProviderDeploy(replicas, "img:v1"),
		readyProviderPod("img:v1"),
		&admissionregistrationv1.MutatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: bkeMutatingWebhookConfigurationName},
		},
		&admissionregistrationv1.ValidatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: bkeValidatingWebhookConfigurationName},
		},
	).Build()

	err := EnsureBKENodeWebhooks(context.Background(), c)
	require.Error(t, err)
	assert.Contains(t, err.Error(), bkeMutatingWebhookClusterName)
}

func TestEnsureBKENodeWebhooks_RequiresValidatingClusterWebhook(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, admissionregistrationv1.AddToScheme(scheme))
	require.NoError(t, appsv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	replicas := int32(1)
	caBundle := []byte("mutate-ca")
	mutateClusterPath := "/mutate-bke-bocloud-com-v1beta1-bkecluster"
	fail := admissionregistrationv1.Fail
	none := admissionregistrationv1.SideEffectClassNone

	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		readyProviderDeploy(replicas, "img:v1"),
		readyProviderPod("img:v1"),
		&admissionregistrationv1.MutatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: bkeMutatingWebhookConfigurationName},
			Webhooks: []admissionregistrationv1.MutatingWebhook{{
				Name:                    bkeMutatingWebhookClusterName,
				AdmissionReviewVersions: []string{"v1"},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					CABundle: caBundle,
					Service: &admissionregistrationv1.ServiceReference{
						Name: bkeWebhookServiceName, Namespace: bkeWebhookServiceNamespace, Path: &mutateClusterPath,
					},
				},
				FailurePolicy: &fail,
				SideEffects:   &none,
			}},
		},
		&admissionregistrationv1.ValidatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: bkeValidatingWebhookConfigurationName},
		},
	).Build()

	err := EnsureBKENodeWebhooks(context.Background(), c)
	require.Error(t, err)
	assert.Contains(t, err.Error(), bkeValidatingWebhookClusterName)
}

func TestEnsureBKENodeWebhooks_NilClient(t *testing.T) {
	err := EnsureBKENodeWebhooks(context.Background(), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "client is nil")
}

func readyProviderDeploy(replicas int32, image string) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: providerWebhookDeployment, Namespace: providerWebhookNamespace},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "manager"}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "manager"}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: providerWebhookContainer, Image: image}},
				},
			},
		},
		Status: appsv1.DeploymentStatus{
			Replicas: replicas, UpdatedReplicas: replicas, AvailableReplicas: replicas, ReadyReplicas: replicas,
		},
	}
}

func readyProviderPod(image string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bke-controller-manager-0", Namespace: providerWebhookNamespace,
			Labels: map[string]string{"app": "manager"},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: providerWebhookContainer, Image: image}},
		},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
}
