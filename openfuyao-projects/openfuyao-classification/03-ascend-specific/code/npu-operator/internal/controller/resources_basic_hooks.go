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
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"hash/fnv"
	"reflect"
	"strings"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	k8sbatchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"volcano.sh/apis/pkg/apis/batch/v1alpha1"
	busv1alpha1 "volcano.sh/apis/pkg/apis/bus/v1alpha1"
	flowv1alpha1 "volcano.sh/apis/pkg/apis/flow/v1alpha1"
	nodeinfov1alpha1 "volcano.sh/apis/pkg/apis/nodeinfo/v1alpha1"
	"volcano.sh/apis/pkg/apis/scheduling"

	v1 "openfuyao.com/npu-operator/api/v1"
)

type serviceAccountReconcileHooks struct {
	defaultResourceReconcileHooks[*corev1.ServiceAccount]
}

func (h serviceAccountReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *corev1.ServiceAccount) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type clusterRoleReconcileHooks struct {
	defaultResourceReconcileHooks[*rbacv1.ClusterRole]
}

func (h clusterRoleReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *rbacv1.ClusterRole,
) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type clusterRoleBindingReconcileHooks struct {
	defaultResourceReconcileHooks[*rbacv1.ClusterRoleBinding]
}

func (h clusterRoleBindingReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *rbacv1.ClusterRoleBinding,
) error {
	for i := range obj.Subjects {
		if obj.Subjects[i].Namespace == placeholderValue {
			obj.Subjects[i].Namespace = r.namespace
		}
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type roleReconcileHooks struct {
	defaultResourceReconcileHooks[*rbacv1.Role]
}

func (h roleReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *rbacv1.Role,
) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type roleBindingReconcileHooks struct {
	defaultResourceReconcileHooks[*rbacv1.RoleBinding]
}

func (h roleBindingReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *rbacv1.RoleBinding,
) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	for i := range obj.Subjects {
		if obj.Subjects[i].Namespace == placeholderValue {
			obj.Subjects[i].Namespace = r.namespace
		}
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type serviceMoniterReconcileHooks struct {
	defaultResourceReconcileHooks[*monitoringv1.ServiceMonitor]
}

func (h serviceMoniterReconcileHooks) transform(ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *monitoringv1.ServiceMonitor,
) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	for i := range obj.Spec.NamespaceSelector.MatchNames {
		if obj.Spec.NamespaceSelector.MatchNames[i] == placeholderValue {
			obj.Spec.NamespaceSelector.MatchNames[i] = r.namespace
		}
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type serviceReconcileHooks struct {
	defaultResourceReconcileHooks[*corev1.Service]
}

func (h serviceReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *corev1.Service) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	if obj.Name == vnpuExporterServiceName && r.instance.Spec.VNPU.Exporter.Port > 0 {
		for i := range obj.Spec.Ports {
			obj.Spec.Ports[i].Port = r.instance.Spec.VNPU.Exporter.Port
			obj.Spec.Ports[i].TargetPort = intstr.FromInt32(r.instance.Spec.VNPU.Exporter.Port)
		}
	}
	if obj.Name == draWebhookServiceName {
		port := int32OrDefault(r.instance.Spec.DRA.Webhook.Port, defaultDRAWebhookPort)
		for i := range obj.Spec.Ports {
			obj.Spec.Ports[i].Port = port
			obj.Spec.Ports[i].TargetPort = intstr.FromInt32(port)
		}
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type leaseReconcileHooks struct {
	defaultResourceReconcileHooks[*coordinationv1.Lease]
}

func (h leaseReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *coordinationv1.Lease) error {
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type secretReconcileHooks struct {
	defaultResourceReconcileHooks[*corev1.Secret]
}

func (h secretReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler, obj *corev1.Secret) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type batchJobReconcileHooks struct {
	defaultResourceReconcileHooks[*k8sbatchv1.Job]
}

func (h batchJobReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler, obj *k8sbatchv1.Job) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	if obj.Name == volcanoAdmissionInitJobName {
		ensureBatchJobSelector(obj)
		applyReleaseTagToPodSpec(&obj.Spec.Template.Spec)
		applyVolcanoRunAs(&obj.Spec.Template.Spec, &r.instance.Spec)
		applyImageSpecToPodSpecWithOverride(&obj.Spec.Template.Spec,
			volcanoAdmissionMainImageConfigGetter(&r.instance.Spec), true)
		existing := &k8sbatchv1.Job{}
		if err := r.Get(ctx, client.ObjectKeyFromObject(obj), existing); err == nil {
			if isBatchJobComplete(existing) {
				return errSkipReconciliation
			}
			if batchJobImmutableSpecChanged(existing, obj) {
				if err := r.Delete(ctx, existing); client.IgnoreNotFound(err) != nil {
					return err
				}
				return errSkipCreation
			}
		} else if !apierrors.IsNotFound(err) {
			return err
		}
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

func ensureBatchJobSelector(job *k8sbatchv1.Job) {
	manualSelector := true
	job.Spec.ManualSelector = &manualSelector
	if job.Spec.Template.Labels == nil {
		job.Spec.Template.Labels = map[string]string{}
	}
	if job.Spec.Template.Labels["app"] == "" {
		job.Spec.Template.Labels["app"] = job.Name
	}
	if job.Spec.Selector == nil {
		job.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{}}
	}
	if job.Spec.Selector.MatchLabels == nil {
		job.Spec.Selector.MatchLabels = map[string]string{}
	}
	if len(job.Spec.Selector.MatchLabels) == 0 {
		job.Spec.Selector.MatchLabels["app"] = job.Spec.Template.Labels["app"]
	}
}

func batchJobImmutableSpecChanged(existing, desired *k8sbatchv1.Job) bool {
	return !reflect.DeepEqual(existing.Spec.ManualSelector, desired.Spec.ManualSelector) ||
		!reflect.DeepEqual(existing.Spec.Selector, desired.Spec.Selector) ||
		!reflect.DeepEqual(existing.Spec.Template, desired.Spec.Template)
}

func isBatchJobComplete(job *k8sbatchv1.Job) bool {
	for _, condition := range job.Status.Conditions {
		if condition.Type == k8sbatchv1.JobComplete && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

type mutatingWebhookConfigurationReconcileHooks struct {
	defaultResourceReconcileHooks[*admissionregistrationv1.MutatingWebhookConfiguration]
}

func (h mutatingWebhookConfigurationReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *admissionregistrationv1.MutatingWebhookConfiguration,
) error {
	if caBundle, ok, err := volcanoAdmissionCABundleForWebhook(ctx, r, obj.Name); err != nil {
		return err
	} else if ok {
		applyMutatingWebhookCABundle(obj.Webhooks, caBundle)
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type validatingWebhookConfigurationReconcileHooks struct {
	defaultResourceReconcileHooks[*admissionregistrationv1.ValidatingWebhookConfiguration]
}

func (h validatingWebhookConfigurationReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *admissionregistrationv1.ValidatingWebhookConfiguration,
) error {
	if caBundle, ok, err := volcanoAdmissionCABundleForWebhook(ctx, r, obj.Name); err != nil {
		return err
	} else if ok {
		applyValidatingWebhookCABundle(obj.Webhooks, caBundle)
	}
	if obj.Name == draWebhookConfigurationName {
		applyDRAWebhookConfiguration(&r.instance.Spec, r.namespace, obj)
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

func volcanoAdmissionCABundleForWebhook(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	name string,
) ([]byte, bool, error) {
	if !isVolcanoAdmissionWebhook(name) || !volcanoAdmissionEnabled(&r.instance.Spec) {
		return nil, false, nil
	}
	caBundle, err := volcanoAdmissionCABundle(ctx, r)
	return caBundle, true, err
}

func applyMutatingWebhookCABundle(webhooks []admissionregistrationv1.MutatingWebhook, caBundle []byte) {
	for i := range webhooks {
		webhooks[i].ClientConfig.CABundle = caBundle
	}
}

func applyValidatingWebhookCABundle(webhooks []admissionregistrationv1.ValidatingWebhook, caBundle []byte) {
	for i := range webhooks {
		webhooks[i].ClientConfig.CABundle = caBundle
	}
}

func isVolcanoAdmissionWebhook(name string) bool {
	return strings.HasPrefix(name, "volcano-admission-service-")
}

func volcanoAdmissionCABundle(ctx context.Context, r *NPUClusterPolicyReconciler) ([]byte, error) {
	secret, err := volcanoAdmissionSecret(ctx, r)
	if err != nil {
		return nil, err
	}
	caBundle := secret.Data[volcanoAdmissionCAKey]
	if len(caBundle) == 0 {
		return nil, fmt.Errorf("Volcano admission secret %s/%s missing %s",
			volcanoSystemNamespace, volcanoAdmissionSecretName, volcanoAdmissionCAKey)
	}
	return caBundle, nil
}

func volcanoAdmissionSecret(ctx context.Context, r *NPUClusterPolicyReconciler) (*corev1.Secret, error) {
	secret := &corev1.Secret{}
	key := client.ObjectKey{Namespace: volcanoSystemNamespace, Name: volcanoAdmissionSecretName}
	if err := r.Get(ctx, key, secret); err == nil {
		return secret, nil
	} else if !apierrors.IsNotFound(err) {
		return nil, fmt.Errorf("get Volcano admission secret: %w", err)
	}
	if err := ensureVolcanoAdmissionInitComplete(ctx, r); err != nil {
		return nil, err
	}
	if err := r.Get(ctx, key, secret); err != nil {
		return nil, fmt.Errorf("get Volcano admission secret: %w", err)
	}
	return secret, nil
}

func ensureVolcanoAdmissionInitComplete(ctx context.Context, r *NPUClusterPolicyReconciler) error {
	job := &k8sbatchv1.Job{}
	key := client.ObjectKey{Namespace: volcanoSystemNamespace, Name: volcanoAdmissionInitJobName}
	if err := r.Get(ctx, key, job); err != nil {
		return fmt.Errorf("get Volcano admission init job: %w", err)
	}
	if !isBatchJobComplete(job) {
		return fmt.Errorf("wait for Volcano admission init job to complete")
	}
	return nil
}

func volcanoAdmissionSecretHash(secret *corev1.Secret) string {
	hasher := fnv.New32a()
	for _, key := range []string{volcanoAdmissionCAKey, "tls.crt", "tls.key"} {
		_, _ = hasher.Write([]byte(key))
		_, _ = hasher.Write(secret.Data[key])
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

func applyDRAWebhookConfiguration(
	spec *v1.NPUClusterPolicySpec,
	namespace string,
	obj *admissionregistrationv1.ValidatingWebhookConfiguration,
) {
	webhookSpec := spec.DRA.Webhook
	port := int32OrDefault(webhookSpec.Port, defaultDRAWebhookPort)
	failurePolicy := admissionregistrationv1.FailurePolicyType(
		stringOrDefault(webhookSpec.FailurePolicy, defaultDRAWebhookFailurePolicy))
	caBundle := decodeCABundle(webhookSpec.CABundle)
	for i := range obj.Webhooks {
		webhook := &obj.Webhooks[i]
		webhook.FailurePolicy = &failurePolicy
		if webhook.ClientConfig.Service != nil {
			webhook.ClientConfig.Service.Namespace = namespace
			webhook.ClientConfig.Service.Name = draWebhookServiceName
			webhook.ClientConfig.Service.Port = &port
		}
		if len(caBundle) > 0 {
			webhook.ClientConfig.CABundle = caBundle
		}
	}
}

func decodeCABundle(value string) []byte {
	if value == "" {
		return nil
	}
	decoded, err := base64.StdEncoding.DecodeString(value)
	if err == nil {
		return decoded
	}
	return []byte(value)
}

type configMapReconcileHooks struct {
	defaultResourceReconcileHooks[*corev1.ConfigMap]
}

func ObtainVolcanoVersion(ctx context.Context, r *NPUClusterPolicyReconciler) (version string, err error) {
	logger := log.FromContext(ctx)
	instance := &v1.NPUClusterPolicy{}
	if err := r.Get(ctx, client.ObjectKey{Name: r.instance.Name, Namespace: r.instance.Namespace}, instance); err != nil {
		return "", fmt.Errorf("get CR instance: %w", err)
	}
	tag := instance.Spec.VCScheduler.Image.Tag
	version = volcanoPluginVersion(tag)
	if tag != "" && version == "" {
		return "", fmt.Errorf("%w: %q", errUnrecognizedVolcanoPluginVersion, tag)
	}
	logger.Info("volcano version", "version", version)
	return version, err
}

func (h configMapReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *corev1.ConfigMap) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}
	switch obj.Name {
	case inferOperatorConfigMapName:
		if err := applyComponentConfig(obj, "infer-operator", r.instance.Spec.InferOperator.Config); err != nil {
			return err
		}
	case rdmaDevicePluginConfigMapName:
		if err := applyComponentConfig(obj, "k8s-rdma-shared-dp", r.instance.Spec.RDMADevicePlugin.Config); err != nil {
			return err
		}
	case vnpuVirtualTemplateConfigMapName:
		updateConfigMapData(obj, "virtual-template-config.yaml", r.instance.Spec.VNPU.VirtualTemplateConfig)
	case vnpuNodeModeConfigMapName:
		updateConfigMapData(obj, "node-mode.config.yaml", r.instance.Spec.VNPU.NodeModeConfig)
	case draProfileConfigMapName:
		updateConfigMapData(obj, "dra.config", r.instance.Spec.DRA.SoftVNPU.ProfileConfig)
	case draChipCapabilitiesConfigMapName:
		updateConfigMapData(obj, "capabilities.yaml", r.instance.Spec.DRA.SoftVNPU.ChipCapabilities)
	case volcanoSchedulerConfigMapName:
		if err := updateVolcanoSchedulerConfigMap(ctx, r, obj); err != nil {
			return err
		}
	}

	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

func updateConfigMapData(obj *corev1.ConfigMap, key, value string) {
	if value == "" {
		return
	}
	if obj.Data == nil {
		obj.Data = map[string]string{}
	}
	obj.Data[key] = value
}

func updateConfig(r *NPUClusterPolicyReconciler, obj *corev1.ConfigMap, ctx context.Context) bool {
	logger := log.FromContext(ctx)
	conf := schedulerConfigMapGetter(&r.instance.Spec)
	if conf == "" {
		return false
	}
	obj.Data["volcano-scheduler.conf"] = conf
	logger.Info("updating volcano scheduler configmap", "volcano-scheduler.conf", conf)
	return true
}

// //////////////////////////////////////////
type jobReconcileHooks struct {
	defaultResourceReconcileHooks[*v1alpha1.Job]
}

func (h jobReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *v1alpha1.Job) error {
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type commandReconcileHooks struct {
	defaultResourceReconcileHooks[*busv1alpha1.Command]
}

func (h commandReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *busv1alpha1.Command) error {
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type podGroupReconcileHooks struct {
	defaultResourceReconcileHooks[*scheduling.PodGroup]
}

func (h podGroupReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *scheduling.PodGroup) error {
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type queueReconcileHooks struct {
	defaultResourceReconcileHooks[*scheduling.Queue]
}

func (h queueReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *scheduling.Queue) error {
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type numatopologyReconcileHooks struct {
	defaultResourceReconcileHooks[*nodeinfov1alpha1.Numatopology]
}

func (h numatopologyReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *nodeinfov1alpha1.Numatopology) error {
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type jobTemplateReconcileHooks struct {
	defaultResourceReconcileHooks[*flowv1alpha1.JobTemplate]
}

func (h jobTemplateReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *flowv1alpha1.JobTemplate) error {
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type jobFlowReconcileHooks struct {
	defaultResourceReconcileHooks[*flowv1alpha1.JobFlow]
}

func (h jobFlowReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *flowv1alpha1.JobFlow) error {
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

type CRDReconcileHooks struct {
	defaultResourceReconcileHooks[*apiextensionsv1.CustomResourceDefinition]
}

func (h CRDReconcileHooks) transform(ctx context.Context, r *NPUClusterPolicyReconciler, obj *apiextensionsv1.CustomResourceDefinition) error {
	if isInferOperatorCRD(obj) {
		return nil
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}
