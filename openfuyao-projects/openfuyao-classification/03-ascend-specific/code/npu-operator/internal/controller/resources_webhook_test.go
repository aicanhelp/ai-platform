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
	"testing"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	k8sbatchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"
	"volcano.sh/apis/pkg/apis/batch/v1alpha1"
	busv1alpha1 "volcano.sh/apis/pkg/apis/bus/v1alpha1"
	"volcano.sh/apis/pkg/apis/scheduling"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func TestDRAWebhookDeploymentTransform(t *testing.T) {
	r := draWebhookTestReconciler(t)
	deploy := draWebhookDeployment()

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))
	assertDRAWebhookDeployment(t, deploy)
}

func draWebhookTestReconciler(t *testing.T) *NPUClusterPolicyReconciler {
	s := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	return &NPUClusterPolicyReconciler{
		Scheme:    s,
		namespace: "npu-system",
		instance: &v1.NPUClusterPolicy{
			TypeMeta: metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{
				Name: "cluster",
				UID:  uuid.NewUUID(),
			},
			Spec: v1.NPUClusterPolicySpec{
				DRA: v1.DRASpec{
					DriverName:    "custom.npu.io",
					DeviceProfile: "npu",
					Webhook: v1.DRAWebhookSpec{
						DRAComponentSpec: v1.DRAComponentSpec{
							Image: v1.ImageSpec{
								Repository:      "repo/dra-webhook",
								Tag:             "v1",
								ImagePullPolicy: string(corev1.PullIfNotPresent),
							},
							VolumeSpec: v1.VolumesSpec{
								Volumes: []corev1.Volume{{
									Name: "tls-secret",
									VolumeSource: corev1.VolumeSource{
										Secret: &corev1.SecretVolumeSource{SecretName: "custom-dra-webhook-tls"},
									},
								}},
							},
						},
						Port:              9443,
						TLSCertFile:       "/certs/tls.crt",
						TLSPrivateKeyFile: "/certs/tls.key",
					},
				},
			},
		},
	}
}

func draWebhookDeployment() *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: draWebhookDeploymentName, Namespace: placeholderValue},
		Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    "webhook",
				Image:   placeholderValue,
				Command: []string{placeholderValue},
				Args:    []string{placeholderValue},
				Ports:   []corev1.ContainerPort{{ContainerPort: 443}},
			}},
			Volumes: []corev1.Volume{{Name: "tls-secret", VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{SecretName: "dra-webhook-tls"},
			}}},
		}}},
	}
}

func assertDRAWebhookDeployment(t *testing.T, deploy *appsv1.Deployment) {
	container := deploy.Spec.Template.Spec.Containers[0]
	assert.Equal(t, "repo/dra-webhook:v1", container.Image)
	assert.Equal(t, []string{"/ascend-npu-dra-webhook"}, container.Command)
	assert.Contains(t, container.Args, "--tls-cert-file=/certs/tls.crt")
	assert.Contains(t, container.Args, "--tls-private-key-file=/certs/tls.key")
	assert.Contains(t, container.Args, "--port=9443")
	assert.Contains(t, container.Args, "--device-profile=gpu")
	assert.Contains(t, container.Args, "--driver-name=custom.npu.io")
	assert.Equal(t, int32(9443), container.Ports[0].ContainerPort)
	assert.Equal(t, "custom-dra-webhook-tls", deploy.Spec.Template.Spec.Volumes[0].Secret.SecretName)
}

func TestDRAWebhookConfigurationTransform(t *testing.T) {
	policySpec := &v1.NPUClusterPolicySpec{
		DRA: v1.DRASpec{Webhook: v1.DRAWebhookSpec{
			Port:          9443,
			CABundle:      "Y2E=",
			FailurePolicy: "Ignore",
		}},
	}
	vwc := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: draWebhookConfigurationName},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				Service: &admissionregistrationv1.ServiceReference{
					Name:      "old",
					Namespace: "old",
				},
			},
		}},
	}

	applyDRAWebhookConfiguration(policySpec, "npu-system", vwc)

	policy := *vwc.Webhooks[0].FailurePolicy
	assert.Equal(t, admissionregistrationv1.Ignore, policy)
	assert.Equal(t, []byte("ca"), vwc.Webhooks[0].ClientConfig.CABundle)
	assert.Equal(t, "dra-webhook-service", vwc.Webhooks[0].ClientConfig.Service.Name)
	assert.Equal(t, "npu-system", vwc.Webhooks[0].ClientConfig.Service.Namespace)
	assert.Equal(t, int32(9443), *vwc.Webhooks[0].ClientConfig.Service.Port)
}

func TestVolcanoAdmissionWebhookConfigurationUsesSecretCABundle(t *testing.T) {
	r := volcanoAdmissionTestReconciler(t, completeVolcanoAdmissionInitJob(), volcanoAdmissionCASecret())
	mwc := &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "volcano-admission-service-jobs-mutate"},
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			ClientConfig: admissionregistrationv1.WebhookClientConfig{},
		}},
	}
	vwc := volcanoValidatingWebhook("volcano-admission-service-podgroups-validate")

	require.NoError(t, mutatingWebhookConfigurationReconcileHooks{}.transform(context.TODO(), r, mwc))
	require.NoError(t, validatingWebhookConfigurationReconcileHooks{}.transform(context.TODO(), r, vwc))

	assert.Equal(t, []byte("ca"), mwc.Webhooks[0].ClientConfig.CABundle)
	assert.Equal(t, []byte("ca"), vwc.Webhooks[0].ClientConfig.CABundle)
}

func TestVolcanoAdmissionWebhookConfigurationUsesExistingSecretWithoutInitJob(t *testing.T) {
	r := volcanoAdmissionTestReconciler(t, volcanoAdmissionCASecret())
	vwc := volcanoValidatingWebhook("volcano-admission-service-podgroups-validate")

	require.NoError(t, validatingWebhookConfigurationReconcileHooks{}.transform(context.TODO(), r, vwc))

	assert.Equal(t, []byte("ca"), vwc.Webhooks[0].ClientConfig.CABundle)
}

func TestVolcanoAdmissionWebhookConfigurationWaitsForSecret(t *testing.T) {
	r := volcanoAdmissionTestReconciler(t, completeVolcanoAdmissionInitJob())
	vwc := volcanoValidatingWebhook("volcano-admission-service-podgroups-validate")

	err := validatingWebhookConfigurationReconcileHooks{}.transform(context.TODO(), r, vwc)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "get Volcano admission secret")
}

func volcanoAdmissionTestReconciler(t *testing.T, objects ...client.Object) *NPUClusterPolicyReconciler {
	enabled := true
	s := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, k8sbatchv1.AddToScheme(s))
	require.NoError(t, admissionregistrationv1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	return &NPUClusterPolicyReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithObjects(objects...).Build(),
		Scheme: s,
		instance: &v1.NPUClusterPolicy{
			TypeMeta:   metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
			Spec: v1.NPUClusterPolicySpec{Volcano: v1.VolcanoSpec{
				Flavor:    v1.VolcanoFlavorVNPU,
				Admission: v1.VolcanoAdmissionSpec{Enabled: &enabled},
			}},
		},
	}
}

func volcanoAdmissionCASecret() *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoAdmissionSecretName, Namespace: volcanoSystemNamespace},
		Data:       map[string][]byte{volcanoAdmissionCAKey: []byte("ca")},
	}
}

func volcanoValidatingWebhook(name string) *admissionregistrationv1.ValidatingWebhookConfiguration {
	return &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			ClientConfig: admissionregistrationv1.WebhookClientConfig{},
		}},
	}
}

func completeVolcanoAdmissionInitJob() *k8sbatchv1.Job {
	return &k8sbatchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoAdmissionInitJobName, Namespace: volcanoSystemNamespace},
		Status: k8sbatchv1.JobStatus{Conditions: []k8sbatchv1.JobCondition{{
			Type:   k8sbatchv1.JobComplete,
			Status: corev1.ConditionTrue,
		}}},
	}
}

func TestEnsureBatchJobSelectorUsesTemplateAppLabel(t *testing.T) {
	job := &k8sbatchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoAdmissionInitJobName, Namespace: volcanoSystemNamespace},
		Spec: k8sbatchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "volcano-admission-init"}},
			},
		},
	}

	ensureBatchJobSelector(job)

	require.NotNil(t, job.Spec.Selector)
	require.NotNil(t, job.Spec.ManualSelector)
	assert.True(t, *job.Spec.ManualSelector)
	assert.Equal(t, map[string]string{"app": "volcano-admission-init"}, job.Spec.Selector.MatchLabels)
	assert.Equal(t, "volcano-admission-init", job.Spec.Template.Labels["app"])
}

func TestBatchJobTransformDeletesJobWithStaleSelector(t *testing.T) {
	existing := &k8sbatchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoAdmissionInitJobName, Namespace: volcanoSystemNamespace},
		Spec: k8sbatchv1.JobSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"controller-uid": "old"}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"controller-uid": "old"}},
			},
		},
	}
	r := batchJobTestReconciler(t, existing)
	desired := &k8sbatchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoAdmissionInitJobName, Namespace: volcanoSystemNamespace},
		Spec: k8sbatchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "volcano-admission-init"}},
			},
		},
	}

	err := batchJobReconcileHooks{}.transform(context.TODO(), r, desired)

	require.ErrorIs(t, err, errSkipCreation)
	current := &k8sbatchv1.Job{}
	err = r.Get(context.TODO(), client.ObjectKeyFromObject(existing), current)
	require.True(t, apierrors.IsNotFound(err))
}

func TestBatchJobTransformDeletesJobWithChangedTemplate(t *testing.T) {
	manualSelector := true
	selector := &metav1.LabelSelector{MatchLabels: map[string]string{"app": "volcano-admission-init"}}
	existing := &k8sbatchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoAdmissionInitJobName, Namespace: volcanoSystemNamespace},
		Spec: k8sbatchv1.JobSpec{
			ManualSelector: &manualSelector,
			Selector:       selector.DeepCopy(),
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "volcano-admission-init"}},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers:    []corev1.Container{{Name: "main", Image: "old-image"}},
				},
			},
		},
	}
	r := batchJobTestReconciler(t, existing)
	desired := existing.DeepCopy()
	desired.Spec.Template.Spec.Containers[0].Image = "new-image"

	err := batchJobReconcileHooks{}.transform(context.TODO(), r, desired)

	require.ErrorIs(t, err, errSkipCreation)
	current := &k8sbatchv1.Job{}
	err = r.Get(context.TODO(), client.ObjectKeyFromObject(existing), current)
	require.True(t, apierrors.IsNotFound(err))
}

func batchJobTestReconciler(t *testing.T, existing *k8sbatchv1.Job) *NPUClusterPolicyReconciler {
	s := runtime.NewScheme()
	require.NoError(t, k8sbatchv1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	return &NPUClusterPolicyReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithObjects(existing).Build(),
		Scheme: s,
		instance: &v1.NPUClusterPolicy{
			TypeMeta:   metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
		},
	}
}

func TestNamespacedResourceHooksReplacePlaceholderNamespace(t *testing.T) {
	s := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, monitoringv1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	r := &NPUClusterPolicyReconciler{
		Scheme:    s,
		namespace: "npu-system",
		instance: &v1.NPUClusterPolicy{
			TypeMeta:   metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
		},
	}

	svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "xpu-exporter-svc", Namespace: placeholderValue}}
	require.NoError(t, serviceReconcileHooks{}.transform(context.TODO(), r, svc))
	assert.Equal(t, "npu-system", svc.Namespace)

	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: vnpuNodeModeConfigMapName, Namespace: placeholderValue}}
	require.NoError(t, configMapReconcileHooks{}.transform(context.TODO(), r, cm))
	assert.Equal(t, "npu-system", cm.Namespace)

	sm := &monitoringv1.ServiceMonitor{
		ObjectMeta: metav1.ObjectMeta{Name: "xpu-exporter-servicemonitor", Namespace: placeholderValue},
		Spec: monitoringv1.ServiceMonitorSpec{
			NamespaceSelector: monitoringv1.NamespaceSelector{MatchNames: []string{placeholderValue}},
		},
	}
	require.NoError(t, serviceMoniterReconcileHooks{}.transform(context.TODO(), r, sm))
	assert.Equal(t, "npu-system", sm.Namespace)
	assert.Equal(t, []string{"npu-system"}, sm.Spec.NamespaceSelector.MatchNames)
}

func TestServiceReconcileHooksTransformVNPUExporterPort(t *testing.T) {
	spec := v1.NPUClusterPolicySpec{
		VNPU: v1.VNPUSpec{Exporter: v1.VNPUExporterSpec{Port: 19100}},
	}
	assertServicePortTransform(t, spec, vnpuExporterServiceName, "xpu-exporter-port", 8082, 19100)
}

func TestVNPUNamespaceAssetAllowsPrivilegedPods(t *testing.T) {
	content, err := os.ReadFile("../../assets/vnpu/namespace/0100_namespace.yaml")
	require.NoError(t, err)

	ns := &corev1.Namespace{}
	require.NoError(t, yaml.Unmarshal(content, ns))

	assert.Equal(t, "xpu", ns.Name)
	assert.Equal(t, "xpu", ns.Labels["kubernetes.io/metadata.name"])
	assert.Equal(t, "privileged", ns.Labels["pod-security.kubernetes.io/enforce"])
}

func TestServiceReconcileHooksTransformDRAWebhookPort(t *testing.T) {
	spec := v1.NPUClusterPolicySpec{
		DRA: v1.DRASpec{Webhook: v1.DRAWebhookSpec{Port: 9443}},
	}
	assertServicePortTransform(t, spec, draWebhookServiceName, "https", 443, 9443)
}

func assertServicePortTransform(
	t *testing.T,
	spec v1.NPUClusterPolicySpec,
	serviceName, portName string,
	oldPort, wantPort int32,
) {
	s := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	r := &NPUClusterPolicyReconciler{
		Scheme:    s,
		namespace: "npu-system",
		instance: &v1.NPUClusterPolicy{
			TypeMeta: metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{
				Name: "cluster",
				UID:  uuid.NewUUID(),
			},
			Spec: spec,
		},
	}
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: placeholderValue},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{{Name: portName, Port: oldPort, TargetPort: intstr.FromInt32(oldPort)}},
		},
	}
	require.NoError(t, serviceReconcileHooks{}.transform(context.TODO(), r, svc))
	assert.Equal(t, "npu-system", svc.Namespace)
	assert.Equal(t, wantPort, svc.Spec.Ports[0].Port)
	assert.Equal(t, intstr.FromInt32(wantPort), svc.Spec.Ports[0].TargetPort)
}

func TestVNPUExporterCommandGetterIncludesConfiguredPort(t *testing.T) {
	spec := &v1.NPUClusterPolicySpec{
		VNPU: v1.VNPUSpec{
			Exporter: v1.VNPUExporterSpec{Port: 19100},
		},
	}
	command := vnpuExporterCommandParametersGetter(spec)
	assert.Contains(t, command.Args, "--type=npu")
	assert.Contains(t, command.Args, "--logging-console=true")
	assert.Contains(t, command.Args, "--port=19100")
}

func TestConfigMapTransformSkipsVolcanoNPURewriteForVNPUFlavor(t *testing.T) {
	spec := v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorVNPU},
		VCScheduler: v1.SchedulerSpec{
			SchedulerConfigMap: "actions: enqueue\ntiers:\n- plugins:\n  - name: volcano-vxpu\n",
		},
	}
	assertSchedulerConfigKeepsVXPU(t, spec, true)
}

func TestConfigMapTransformSkipsVolcanoNPURewriteWhenPluginIsAbsent(t *testing.T) {
	spec := v1.NPUClusterPolicySpec{
		VCScheduler: v1.SchedulerSpec{
			SchedulerConfigMap: "actions: enqueue\ntiers:\n- plugins:\n  - name: volcano-vxpu\n",
		},
	}
	assertSchedulerConfigKeepsVXPU(t, spec, false)
}

func assertSchedulerConfigKeepsVXPU(t *testing.T, spec v1.NPUClusterPolicySpec, wantNamespace bool) {
	s := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	r := &NPUClusterPolicyReconciler{
		Scheme:    s,
		namespace: "volcano-system",
		instance: &v1.NPUClusterPolicy{
			TypeMeta:   metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
			Spec:       spec,
		},
	}
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoSchedulerConfigMapName, Namespace: placeholderValue},
		Data:       map[string]string{"volcano-scheduler.conf": "old"},
	}
	require.NoError(t, configMapReconcileHooks{}.transform(context.TODO(), r, cm))
	if wantNamespace {
		assert.Equal(t, "volcano-system", cm.Namespace)
	}
	assert.Contains(t, cm.Data["volcano-scheduler.conf"], "volcano-vxpu")
}

func TestVolcanoAdmissionDeploymentTransformer(t *testing.T) {
	runAsUser := int64(1001)
	runAsGroup := int64(1002)
	logLevel := int32(5)
	r := volcanoAdmissionDeploymentReconciler(t, runAsUser, runAsGroup, logLevel)
	deploy := volcanoAdmissionDeployment()

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))
	assertVolcanoAdmissionDeployment(t, deploy, runAsUser, runAsGroup)
}

func volcanoAdmissionDeploymentReconciler(
	t *testing.T,
	runAsUser, runAsGroup int64,
	logLevel int32,
) *NPUClusterPolicyReconciler {
	spec := v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{
			Flavor:     v1.VolcanoFlavorVNPU,
			RunAsUser:  &runAsUser,
			RunAsGroup: &runAsGroup,
			Admission: v1.VolcanoAdmissionSpec{
				Image: v1.ImageSpec{
					Repository:      "repo/vc_webhook_manager",
					Tag:             "2.0.0",
					ImagePullPolicy: "Always",
				},
				Log: v1.VolcanoLogSpec{
					Dir:   "/custom/admission",
					File:  "/custom/admission/admission.log",
					Level: &logLevel,
				},
			},
		},
	}
	return volcanoDeploymentReconcilerForSpec(t, spec, completeVolcanoAdmissionInitJob(), volcanoAdmissionTLSSecret())
}

func volcanoAdmissionDeploymentReconcilerWithoutOverrides(t *testing.T) *NPUClusterPolicyReconciler {
	return volcanoDeploymentReconcilerForSpec(t, vnpuVolcanoSpec(), completeVolcanoAdmissionInitJob(), volcanoAdmissionTLSSecret())
}

func volcanoAdmissionTLSSecret() *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoAdmissionSecretName, Namespace: volcanoSystemNamespace},
		Data: map[string][]byte{
			volcanoAdmissionCAKey: []byte("ca"),
			"tls.crt":             []byte("cert"),
			"tls.key":             []byte("key"),
		},
	}
}

func volcanoDeploymentReconcilerForSpec(
	t *testing.T,
	spec v1.NPUClusterPolicySpec,
	objects ...client.Object,
) *NPUClusterPolicyReconciler {
	s := schemeForVolcanoDeploymentTest(t)
	require.NoError(t, k8sbatchv1.AddToScheme(s))
	return &NPUClusterPolicyReconciler{
		Client:    fake.NewClientBuilder().WithScheme(s).WithObjects(objects...).Build(),
		Scheme:    s,
		namespace: "volcano-system",
		instance:  volcanoDeploymentPolicy(spec),
	}
}

func volcanoDeploymentPolicy(spec v1.NPUClusterPolicySpec) *v1.NPUClusterPolicy {
	return &v1.NPUClusterPolicy{
		TypeMeta:   metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
		Spec:       spec,
	}
}

func schemeForVolcanoDeploymentTest(t *testing.T) *runtime.Scheme {
	s := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(s))
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	return s
}

func volcanoSchedulerDeploymentWithAssetDefaults(runAsUser, runAsGroup int64) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: vcSchedulerDeploymentName, Namespace: "volcano-system"},
		Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			SecurityContext: &corev1.PodSecurityContext{
				RunAsUser:  &runAsUser,
				RunAsGroup: &runAsGroup,
			},
			InitContainers: []corev1.Container{{
				Name:            "init-permission",
				Image:           "asset/busybox:default",
				SecurityContext: &corev1.SecurityContext{},
			}},
			Containers: []corev1.Container{{
				Name:            "scheduler",
				Image:           "asset/scheduler:default",
				ImagePullPolicy: corev1.PullIfNotPresent,
				Command:         []string{"/asset-scheduler"},
				Args:            []string{"--asset-conf=/asset.conf"},
				SecurityContext: &corev1.SecurityContext{
					RunAsUser:  &runAsUser,
					RunAsGroup: &runAsGroup,
				},
			}},
		}}},
	}
}

func volcanoAdmissionDeployment() *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoAdmissionDeploymentName, Namespace: placeholderValue},
		Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			SecurityContext: &corev1.PodSecurityContext{},
			Containers: []corev1.Container{{
				Name:            "admission",
				Image:           placeholderValue,
				ImagePullPolicy: corev1.PullIfNotPresent,
				Args: []string{
					"--log_dir=/var/log/volcano-admission",
					"--log_file=/var/log/volcano-admission/volcano-admission.log",
					"-v=4",
				},
				SecurityContext: &corev1.SecurityContext{},
				VolumeMounts:    []corev1.VolumeMount{{Name: "admission-config-log", MountPath: "/var/log/volcano-admission"}},
			}},
			InitContainers: []corev1.Container{{
				Name:            "init-permission",
				Image:           placeholderValue,
				ImagePullPolicy: corev1.PullIfNotPresent,
				SecurityContext: &corev1.SecurityContext{},
				VolumeMounts:    []corev1.VolumeMount{{Name: "admission-config-log", MountPath: "/var/log/volcano-admission"}},
			}},
			Volumes: []corev1.Volume{{Name: "admission-config-log", VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{Path: "/var/log/volcano-admission"},
			}}},
		}}},
	}
}

func assertVolcanoAdmissionDeployment(t *testing.T, deploy *appsv1.Deployment, runAsUser, runAsGroup int64) {
	container := deploy.Spec.Template.Spec.Containers[0]
	assert.Equal(t, "repo/vc_webhook_manager:2.0.0", container.Image)
	assert.Equal(t, corev1.PullAlways, container.ImagePullPolicy)
	assert.Contains(t, container.Args, "--log_dir=/custom/admission")
	assert.Contains(t, container.Args, "--log_file=/custom/admission/admission.log")
	assert.Contains(t, container.Args, "-v=5")
	assert.Equal(t, "/custom/admission", container.VolumeMounts[0].MountPath)
	assert.Equal(t, "/custom/admission", deploy.Spec.Template.Spec.Volumes[0].HostPath.Path)
	assert.Equal(t, runAsUser, *deploy.Spec.Template.Spec.SecurityContext.RunAsUser)
	assert.Equal(t, runAsGroup, *deploy.Spec.Template.Spec.SecurityContext.RunAsGroup)
	assert.Equal(t, runAsUser, *container.SecurityContext.RunAsUser)
	assert.Equal(t, runAsGroup, *container.SecurityContext.RunAsGroup)
	assert.NotEmpty(t, deploy.Spec.Template.Annotations[volcanoAdmissionSecretHashAnnotation])
}

func TestVolcanoRunAsDoesNotOverrideAssetWhenUnsetForVNPUFlavor(t *testing.T) {
	s := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(s))
	require.NoError(t, v1.AddToScheme(s))
	r := &NPUClusterPolicyReconciler{
		Scheme:    s,
		namespace: "volcano-system",
		instance: &v1.NPUClusterPolicy{
			TypeMeta:   metav1.TypeMeta{Kind: "NPUClusterPolicy", APIVersion: v1.GroupVersion.String()},
			ObjectMeta: metav1.ObjectMeta{Name: "cluster", UID: uuid.NewUUID()},
			Spec: v1.NPUClusterPolicySpec{
				Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorVNPU},
				VCScheduler: v1.SchedulerSpec{
					Image:     v1.ImageSpec{Repository: "repo/scheduler", Tag: "1.15.0", ImagePullPolicy: "IfNotPresent"},
					InitImage: v1.ImageSpec{Repository: "repo/busybox", ImagePullPolicy: "IfNotPresent"},
				},
			},
		},
	}
	deploy := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: vcSchedulerDeploymentName, Namespace: "volcano-system"},
		Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			SecurityContext: &corev1.PodSecurityContext{},
			InitContainers: []corev1.Container{{
				Name:            "init-permission",
				Image:           placeholderValue,
				SecurityContext: &corev1.SecurityContext{},
			}},
			Containers: []corev1.Container{{
				Name:            "scheduler",
				Image:           placeholderValue,
				SecurityContext: &corev1.SecurityContext{},
			}},
		}}},
	}

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))
	require.NotNil(t, deploy.Spec.Template.Spec.SecurityContext)
	assert.Nil(t, deploy.Spec.Template.Spec.SecurityContext.RunAsUser)
	assert.Nil(t, deploy.Spec.Template.Spec.SecurityContext.RunAsGroup)
	assert.Nil(t, deploy.Spec.Template.Spec.SecurityContext.FSGroup)
	assert.Nil(t, deploy.Spec.Template.Spec.Containers[0].SecurityContext.RunAsUser)
	assert.Nil(t, deploy.Spec.Template.Spec.Containers[0].SecurityContext.RunAsGroup)
}

func TestVolcanoDeploymentPreservesAssetDefaultsWhenCRFieldsAreEmpty(t *testing.T) {
	assetRunAsUser := int64(2000)
	assetRunAsGroup := int64(3000)
	r := volcanoDeploymentReconcilerForSpec(t, vnpuVolcanoSpec())
	deploy := volcanoSchedulerDeploymentWithAssetDefaults(assetRunAsUser, assetRunAsGroup)

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	container := deploy.Spec.Template.Spec.Containers[0]
	assert.Equal(t, "asset/scheduler:default", container.Image)
	assert.Equal(t, []string{"/asset-scheduler"}, container.Command)
	assert.Equal(t, []string{"--asset-conf=/asset.conf"}, container.Args)
	assert.Equal(t, assetRunAsUser, *deploy.Spec.Template.Spec.SecurityContext.RunAsUser)
	assert.Equal(t, assetRunAsGroup, *deploy.Spec.Template.Spec.SecurityContext.RunAsGroup)
	assert.Equal(t, assetRunAsUser, *container.SecurityContext.RunAsUser)
	assert.Equal(t, assetRunAsGroup, *container.SecurityContext.RunAsGroup)
}

func TestVolcanoDeploymentOverridesAssetDefaultsWhenCRFieldsAreSet(t *testing.T) {
	runAsRoot := int64(0)
	r := volcanoDeploymentReconcilerForSpec(t, v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{
			Flavor:     v1.VolcanoFlavorVNPU,
			RunAsUser:  &runAsRoot,
			RunAsGroup: &runAsRoot,
		},
		VCScheduler: v1.SchedulerSpec{
			Image: v1.ImageSpec{
				Repository:      "override/scheduler",
				Tag:             "test",
				ImagePullPolicy: "Always",
			},
			Command: v1.CommandSpec{
				Command: []string{"/override-scheduler"},
				Args:    []string{"--override"},
			},
		},
	})
	deploy := volcanoSchedulerDeploymentWithAssetDefaults(2000, 3000)

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	container := deploy.Spec.Template.Spec.Containers[0]
	assert.Equal(t, "override/scheduler:test", container.Image)
	assert.Equal(t, corev1.PullAlways, container.ImagePullPolicy)
	assert.Equal(t, []string{"/override-scheduler"}, container.Command)
	assert.Equal(t, []string{"--override"}, container.Args)
	assert.Equal(t, int64(0), *deploy.Spec.Template.Spec.SecurityContext.RunAsUser)
	assert.Equal(t, int64(0), *deploy.Spec.Template.Spec.SecurityContext.RunAsGroup)
	assert.Equal(t, int64(0), *container.SecurityContext.RunAsUser)
	assert.Equal(t, int64(0), *container.SecurityContext.RunAsGroup)
}

func TestVolcanoDeploymentReplacesReleaseTagPlaceholder(t *testing.T) {
	t.Setenv(operatorReleaseTagEnvName, "26.9.0")
	r := volcanoDeploymentReconcilerForSpec(t, vnpuVolcanoSpec())
	deploy := volcanoSchedulerDeploymentWithAssetDefaults(2000, 3000)
	deploy.Spec.Template.Spec.Containers[0].Image = "cr.openfuyao.cn/openfuyao/vnpu/vc-scheduler:" + releaseTagPlaceholder

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	assert.Equal(t, "cr.openfuyao.cn/openfuyao/vnpu/vc-scheduler:26.9.0",
		deploy.Spec.Template.Spec.Containers[0].Image)
}

func TestVolcanoDeploymentImageOverrideWinsOverReleaseTagPlaceholder(t *testing.T) {
	t.Setenv(operatorReleaseTagEnvName, "26.9.0")
	r := volcanoDeploymentReconcilerForSpec(t, v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorVNPU},
		VCScheduler: v1.SchedulerSpec{
			Image: v1.ImageSpec{
				Repository:      "override/scheduler",
				Tag:             "custom",
				ImagePullPolicy: "Always",
			},
		},
	})
	deploy := volcanoSchedulerDeploymentWithAssetDefaults(2000, 3000)
	deploy.Spec.Template.Spec.Containers[0].Image = "cr.openfuyao.cn/openfuyao/vnpu/vc-scheduler:" + releaseTagPlaceholder

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	assert.Equal(t, "override/scheduler:custom", deploy.Spec.Template.Spec.Containers[0].Image)
}

func TestVolcanoVNPUFlavorIgnoresStaleAscendSchedulerImage(t *testing.T) {
	t.Setenv(operatorReleaseTagEnvName, "26.9.0")
	r := volcanoDeploymentReconcilerForSpec(t, v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorVNPU},
		VCScheduler: v1.SchedulerSpec{
			Image: v1.ImageSpec{
				Repository:      "hub.oepkgs.net/openfuyao/ascendhub/vc-scheduler",
				Tag:             "v1.9.0-v7.2.RC1",
				ImagePullPolicy: "Always",
			},
		},
	})
	deploy := volcanoSchedulerDeploymentWithAssetDefaults(2000, 3000)
	deploy.Spec.Template.Spec.Containers[0].Image = "cr.openfuyao.cn/openfuyao/vnpu/vc-scheduler:" + releaseTagPlaceholder

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	assert.Equal(t, "cr.openfuyao.cn/openfuyao/vnpu/vc-scheduler:26.9.0",
		deploy.Spec.Template.Spec.Containers[0].Image)
	assert.Equal(t, corev1.PullIfNotPresent, deploy.Spec.Template.Spec.Containers[0].ImagePullPolicy)
}

func TestVolcanoAscendFlavorIgnoresStaleVNPUSchedulerImage(t *testing.T) {
	r := volcanoDeploymentReconcilerForSpec(t, v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorMindCluster},
		VCScheduler: v1.SchedulerSpec{
			Image: v1.ImageSpec{
				Repository:      "cr.openfuyao.cn/openfuyao/vnpu/vc-scheduler",
				Tag:             "latest",
				ImagePullPolicy: "IfNotPresent",
			},
		},
	})
	deploy := volcanoSchedulerDeploymentWithAssetDefaults(2000, 3000)
	deploy.Spec.Template.Spec.Containers[0].Image = "hub.oepkgs.net/openfuyao/ascendhub/vc-scheduler:v1.9.0-v7.3.0"
	deploy.Spec.Template.Spec.Containers[0].ImagePullPolicy = corev1.PullAlways

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	assert.Equal(t, "hub.oepkgs.net/openfuyao/ascendhub/vc-scheduler:v1.9.0-v7.3.0",
		deploy.Spec.Template.Spec.Containers[0].Image)
	assert.Equal(t, corev1.PullAlways, deploy.Spec.Template.Spec.Containers[0].ImagePullPolicy)
}

func TestVolcanoSchedulerConfigMapPreservesAssetDefaultWhenCROverrideEmpty(t *testing.T) {
	r := volcanoDeploymentReconcilerForSpec(t, vnpuVolcanoSpec())
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: volcanoSchedulerConfigMapName, Namespace: "volcano-system"},
		Data: map[string]string{
			"volcano-scheduler.conf": "actions: enqueue\n- name: volcano-vxpu\n",
		},
	}

	require.NoError(t, configMapReconcileHooks{}.transform(context.TODO(), r, cm))

	assert.Equal(t, "actions: enqueue\n- name: volcano-vxpu\n", cm.Data["volcano-scheduler.conf"])
}

func vnpuVolcanoSpec() v1.NPUClusterPolicySpec {
	return v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{Flavor: v1.VolcanoFlavorVNPU},
	}
}

func TestVolcanoAdmissionDeploymentPreservesAssetLogAndImageWhenCROverrideEmpty(t *testing.T) {
	r := volcanoAdmissionDeploymentReconcilerWithoutOverrides(t)
	deploy := volcanoAdmissionDeployment()
	deploy.Spec.Template.Spec.Containers[0].Image = "asset/admission:default"
	deploy.Spec.Template.Spec.InitContainers = []corev1.Container{{
		Name:            "init-permission",
		Image:           placeholderValue,
		ImagePullPolicy: corev1.PullIfNotPresent,
	}}
	deploy.Spec.Template.Spec.Containers[0].Args = []string{
		"--log_dir=/asset/admission",
		"--log_file=/asset/admission/admission.log",
		"-v=2",
	}
	deploy.Spec.Template.Spec.Containers[0].VolumeMounts[0].MountPath = "/asset/admission"
	deploy.Spec.Template.Spec.Volumes[0].HostPath.Path = "/asset/admission"

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	container := deploy.Spec.Template.Spec.Containers[0]
	assert.Equal(t, "asset/admission:default", container.Image)
	assert.Equal(t, "hub.oepkgs.net/openfuyao/busybox:1.36.1", deploy.Spec.Template.Spec.InitContainers[0].Image)
	assert.Contains(t, container.Args, "--log_dir=/asset/admission")
	assert.Contains(t, container.Args, "--log_file=/asset/admission/admission.log")
	assert.Contains(t, container.Args, "-v=2")
	assert.Equal(t, "/asset/admission", container.VolumeMounts[0].MountPath)
	assert.Equal(t, "/asset/admission", deploy.Spec.Template.Spec.Volumes[0].HostPath.Path)
}

func TestVolcanoAdmissionDeploymentUsesBusyboxEnvForDefaultInitImage(t *testing.T) {
	t.Setenv("NPU_OPERATOR_BUSYBOX_IMAGE_REPOSITORY", "custom.registry/openfuyao/busybox")
	t.Setenv("NPU_OPERATOR_BUSYBOX_IMAGE_TAG", "custom-tag")
	t.Setenv("NPU_OPERATOR_BUSYBOX_IMAGE_PULL_POLICY", "IfNotPresent")
	r := volcanoAdmissionDeploymentReconcilerWithoutOverrides(t)
	deploy := volcanoAdmissionDeployment()
	deploy.Spec.Template.Spec.InitContainers = []corev1.Container{{
		Name:            "init-permission",
		Image:           placeholderValue,
		ImagePullPolicy: corev1.PullAlways,
	}}

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	initContainer := deploy.Spec.Template.Spec.InitContainers[0]
	assert.Equal(t, "custom.registry/openfuyao/busybox:custom-tag", initContainer.Image)
	assert.Equal(t, corev1.PullIfNotPresent, initContainer.ImagePullPolicy)
}

func TestVolcanoAdmissionDeploymentHasInitContainerForLogPermission(t *testing.T) {
	data, err := os.ReadFile("../../assets/volcano/vnpu-admission/1000_deployment.yaml")
	require.NoError(t, err)
	deploy := &appsv1.Deployment{}
	require.NoError(t, yaml.Unmarshal(data, deploy))
	require.Len(t, deploy.Spec.Template.Spec.InitContainers, 1)

	initContainer := deploy.Spec.Template.Spec.InitContainers[0]
	assert.Equal(t, "init-permission", initContainer.Name)
	assert.Equal(t, placeholderValue, initContainer.Image)
	require.NotNil(t, initContainer.SecurityContext)
	assert.Equal(t, int64(0), *initContainer.SecurityContext.RunAsUser)
	assert.Equal(t, int64(0), *initContainer.SecurityContext.RunAsGroup)
	assert.Contains(t, initContainer.Command[2], "chown 1000:1000 /var/log/volcano-admission")
	assert.Equal(t, "admission-config-log", initContainer.VolumeMounts[0].Name)
	assert.Equal(t, "/var/log/volcano-admission", initContainer.VolumeMounts[0].MountPath)
}

func TestVNPUVolcanoLogPermissionInitContainersArePrivileged(t *testing.T) {
	for _, path := range []string{
		"../../assets/volcano/vnpu-controller/0700_deployment.yaml",
		"../../assets/volcano/vnpu-scheduler/0600_deployment.yaml",
		"../../assets/volcano/vnpu-admission/1000_deployment.yaml",
	} {
		t.Run(path, func(t *testing.T) {
			deploy := readDeploymentAsset(t, path)
			require.Len(t, deploy.Spec.Template.Spec.InitContainers, 1)
			initContainer := deploy.Spec.Template.Spec.InitContainers[0]
			require.NotNil(t, initContainer.SecurityContext)
			require.NotNil(t, initContainer.SecurityContext.Privileged)
			assert.True(t, *initContainer.SecurityContext.Privileged)
			assert.Equal(t, int64(0), *initContainer.SecurityContext.RunAsUser)
			assert.Equal(t, int64(0), *initContainer.SecurityContext.RunAsGroup)

			container := deploy.Spec.Template.Spec.Containers[0]
			require.NotNil(t, container.SecurityContext)
			assert.Equal(t, int64(1000), *container.SecurityContext.RunAsUser)
		})
	}
}

func TestVNPUVolcanoDeploymentsOverrideRunAsRoot(t *testing.T) {
	runAsRoot := int64(0)
	r := volcanoAdmissionDeploymentReconcilerWithoutOverrides(t)
	r.instance.Spec.Volcano = v1.VolcanoSpec{
		Flavor:     v1.VolcanoFlavorVNPU,
		RunAsUser:  &runAsRoot,
		RunAsGroup: &runAsRoot,
	}

	for _, path := range []string{
		"../../assets/volcano/vnpu-controller/0700_deployment.yaml",
		"../../assets/volcano/vnpu-scheduler/0600_deployment.yaml",
	} {
		t.Run(path, func(t *testing.T) {
			deploy := readDeploymentAsset(t, path)

			require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

			container := deploy.Spec.Template.Spec.Containers[0]
			require.NotNil(t, container.SecurityContext)
			require.NotNil(t, container.SecurityContext.RunAsUser)
			require.NotNil(t, container.SecurityContext.RunAsGroup)
			require.NotNil(t, container.SecurityContext.RunAsNonRoot)
			assert.Equal(t, int64(0), *container.SecurityContext.RunAsUser)
			assert.Equal(t, int64(0), *container.SecurityContext.RunAsGroup)
			assert.False(t, *container.SecurityContext.RunAsNonRoot)
		})
	}
}

func readDeploymentAsset(t *testing.T, path string) *appsv1.Deployment {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	deploy := &appsv1.Deployment{}
	require.NoError(t, yaml.Unmarshal(data, deploy))
	return deploy
}

func TestVolcanoAdmissionDeploymentOverridesInitImage(t *testing.T) {
	r := volcanoAdmissionDeploymentReconcilerWithoutOverrides(t)
	r.instance.Spec.Volcano.Admission.InitImage = v1.ImageSpec{
		Repository:      "custom/busybox",
		Tag:             "1.36",
		ImagePullPolicy: "Always",
	}
	deploy := volcanoAdmissionDeployment()
	deploy.Spec.Template.Spec.InitContainers = []corev1.Container{{
		Name:            "init-permission",
		Image:           placeholderValue,
		ImagePullPolicy: corev1.PullIfNotPresent,
	}}

	require.NoError(t, newDeploymentReconcileHooks(deploy).transform(context.TODO(), r, deploy))

	initContainer := deploy.Spec.Template.Spec.InitContainers[0]
	assert.Equal(t, "custom/busybox:1.36", initContainer.Image)
	assert.Equal(t, corev1.PullAlways, initContainer.ImagePullPolicy)
}

func TestVolcanoRunAsRootClearsRunAsNonRoot(t *testing.T) {
	runAsRoot := int64(0)
	spec := &v1.NPUClusterPolicySpec{
		Volcano: v1.VolcanoSpec{
			Flavor:     v1.VolcanoFlavorVNPU,
			RunAsUser:  &runAsRoot,
			RunAsGroup: &runAsRoot,
		},
	}
	runAsNonRoot := true
	podSpec := &corev1.PodSpec{
		SecurityContext: &corev1.PodSecurityContext{},
		InitContainers: []corev1.Container{{
			Name: "init",
			SecurityContext: &corev1.SecurityContext{
				RunAsNonRoot: &runAsNonRoot,
			},
		}},
		Containers: []corev1.Container{{
			Name: "main",
			SecurityContext: &corev1.SecurityContext{
				RunAsNonRoot: &runAsNonRoot,
			},
		}},
	}

	applyVolcanoRunAs(podSpec, spec)

	require.NotNil(t, podSpec.Containers[0].SecurityContext.RunAsNonRoot)
	require.NotNil(t, podSpec.InitContainers[0].SecurityContext.RunAsNonRoot)
	assert.False(t, *podSpec.Containers[0].SecurityContext.RunAsNonRoot)
	assert.False(t, *podSpec.InitContainers[0].SecurityContext.RunAsNonRoot)
	assert.Equal(t, int64(0), *podSpec.Containers[0].SecurityContext.RunAsUser)
	assert.Equal(t, int64(0), *podSpec.Containers[0].SecurityContext.RunAsGroup)
}

func TestNewWorkloadVolumesTransformerMergeExisting(t *testing.T) {
	spec := &v1.NPUClusterPolicySpec{
		DevicePlugin: v1.DevicePluginSpec{
			VolumeSpec: v1.VolumesSpec{
				Volumes: []corev1.Volume{
					{Name: "config", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
				},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "config", MountPath: "/etc/config"},
				},
			},
		},
	}
	transformer := newWorkloadVolumesTransformer[appsv1.DaemonSet](devicePluginVolumesGetter)
	ds := &appsv1.DaemonSet{
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{Name: "existing", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/tmp"}}},
						{Name: "config", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/old"}}},
					},
					Containers: []corev1.Container{
						{
							Name: "dp",
							VolumeMounts: []corev1.VolumeMount{
								{Name: "existing", MountPath: "/tmp"},
								{Name: "config", MountPath: "/old"},
							},
						},
					},
				},
			},
		},
	}
	r := &NPUClusterPolicyReconciler{
		instance: &v1.NPUClusterPolicy{Spec: *spec},
	}
	err := transformer(context.Background(), r, ds)
	require.NoError(t, err)
	assert.Len(t, ds.Spec.Template.Spec.Volumes, 2)
	volMap := make(map[string]corev1.Volume)
	for _, v := range ds.Spec.Template.Spec.Volumes {
		volMap[v.Name] = v
	}
	assert.Equal(t, "/tmp", volMap["existing"].HostPath.Path)
	assert.NotNil(t, volMap["config"].EmptyDir) // replaced by EmptyDir
	mountMap := make(map[string]corev1.VolumeMount)
	for _, m := range ds.Spec.Template.Spec.Containers[0].VolumeMounts {
		mountMap[m.Name] = m
	}
	assert.Equal(t, "/tmp", mountMap["existing"].MountPath)
	assert.Equal(t, "/etc/config", mountMap["config"].MountPath) // updated
}

type testCase struct {
	name     string
	src      client.Object
	dst      client.Object
	validate func(*testing.T, client.Object)
}

func runIntoTest[T any, PT clientObject[T]](t *testing.T, tc testCase) {
	t.Run(tc.name, func(t *testing.T) {

		res := typedComponentResource[T, PT]{
			obj: tc.src.(PT),
		}

		res.into(tc.dst)

		tc.validate(t, tc.dst)
	})
}

// TestInto_Metadata verifies that Labels, Annotations, OwnerRefs are merged/copied correctly.
func TestIntoMetadata(t *testing.T) {
	src := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Labels:          map[string]string{"app": "npu", "new": "val"},
			Annotations:     map[string]string{"anno": "yes"},
			OwnerReferences: []metav1.OwnerReference{{Name: "owner"}},
			Finalizers:      []string{"finalizer.npu"},
		},
	}
	dst := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      map[string]string{"existing": "old", "app": "old"},
			Annotations: map[string]string{"other": "no"},
		},
	}

	tc := testCase{
		name: "Metadata Merge",
		src:  src,
		dst:  dst,
		validate: func(t *testing.T, obj client.Object) {
			labels := obj.GetLabels()
			assert.Equal(t, "npu", labels["app"])
			assert.Equal(t, "old", labels["existing"])
			assert.Equal(t, "val", labels["new"])

			annos := obj.GetAnnotations()
			assert.Equal(t, "yes", annos["anno"])
			assert.Equal(t, "no", annos["other"])

			refs := obj.GetOwnerReferences()
			if assert.NotEmpty(t, refs) {
				assert.Equal(t, "owner", refs[0].Name)
			}

			finals := obj.GetFinalizers()
			if assert.NotEmpty(t, finals) {
				assert.Equal(t, "finalizer.npu", finals[0])
			}
		},
	}
	runIntoTest[appsv1.Deployment, *appsv1.Deployment](t, tc)
}

func TestIntoWorkloads(t *testing.T) {
	tests := []testCase{
		{
			name: "Deployment Spec Copy",
			src:  &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: int32Ptr(3)}},
			dst:  &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: int32Ptr(1)}},
			validate: func(t *testing.T, obj client.Object) {
				assert.Equal(t, int32(3), *obj.(*appsv1.Deployment).Spec.Replicas)
			},
		},
		{
			name: "DaemonSet Spec Copy",
			src:  &appsv1.DaemonSet{Spec: appsv1.DaemonSetSpec{MinReadySeconds: 10}},
			dst:  &appsv1.DaemonSet{Spec: appsv1.DaemonSetSpec{MinReadySeconds: 0}},
			validate: func(t *testing.T, obj client.Object) {
				assert.Equal(t, int32(10), obj.(*appsv1.DaemonSet).Spec.MinReadySeconds)
			},
		},
		{
			name: "Volcano Job Spec Copy",
			src:  &v1alpha1.Job{Spec: v1alpha1.JobSpec{MinAvailable: 5}},
			dst:  &v1alpha1.Job{Spec: v1alpha1.JobSpec{MinAvailable: 1}},
			validate: func(t *testing.T, obj client.Object) {
				assert.Equal(t, int32(5), obj.(*v1alpha1.Job).Spec.MinAvailable)
			},
		},
	}

	runIntoTest[appsv1.Deployment, *appsv1.Deployment](t, tests[0])
	runIntoTest[appsv1.DaemonSet, *appsv1.DaemonSet](t, tests[1])
	runIntoTest[v1alpha1.Job, *v1alpha1.Job](t, tests[2])
}

func TestIntoCoreResources(t *testing.T) {
	tests := []testCase{
		{
			name: "Service Preserve ClusterIP",
			src:  &corev1.Service{Spec: corev1.ServiceSpec{Ports: []corev1.ServicePort{{Port: 80}}}},
			dst:  &corev1.Service{Spec: corev1.ServiceSpec{ClusterIP: "1.2.3.4", Ports: []corev1.ServicePort{{Port: 90}}}},
			validate: func(t *testing.T, obj client.Object) {
				svc := obj.(*corev1.Service)
				assert.Equal(t, "1.2.3.4", svc.Spec.ClusterIP, "ClusterIP should be preserved")
				assert.Equal(t, int32(80), svc.Spec.Ports[0].Port, "Ports should be updated")
			},
		},
		{
			name: "ConfigMap Data Copy",
			src:  &corev1.ConfigMap{Data: map[string]string{"key": "new"}},
			dst:  &corev1.ConfigMap{Data: map[string]string{"key": "old"}},
			validate: func(t *testing.T, obj client.Object) {
				assert.Equal(t, "new", obj.(*corev1.ConfigMap).Data["key"])
			},
		},
		{
			name: "ServiceAccount Copy",
			src:  &corev1.ServiceAccount{AutomountServiceAccountToken: boolPtr(true)},
			dst:  &corev1.ServiceAccount{AutomountServiceAccountToken: boolPtr(false)},
			validate: func(t *testing.T, obj client.Object) {
				assert.True(t, *obj.(*corev1.ServiceAccount).AutomountServiceAccountToken)
			},
		},
		{
			name: "Namespace Spec Copy",
			src:  &corev1.Namespace{Spec: corev1.NamespaceSpec{Finalizers: []corev1.FinalizerName{"test"}}},
			dst:  &corev1.Namespace{},
			validate: func(t *testing.T, obj client.Object) {
				assert.Equal(t, corev1.FinalizerName("test"), obj.(*corev1.Namespace).Spec.Finalizers[0])
			},
		},
	}

	runIntoTest[corev1.Service, *corev1.Service](t, tests[0])
	runIntoTest[corev1.ConfigMap, *corev1.ConfigMap](t, tests[1])
	runIntoTest[corev1.ServiceAccount, *corev1.ServiceAccount](t, tests[2])
	runIntoTest[corev1.Namespace, *corev1.Namespace](t, tests[3])
}

func TestIntoVolcanoResources(t *testing.T) {
	cmdTest := testCase{
		name: "Volcano Command Fields Copy",
		src:  &busv1alpha1.Command{Action: "Restart", Message: "Updating"},
		dst:  &busv1alpha1.Command{Action: "Stop", Message: "Old"},
		validate: func(t *testing.T, obj client.Object) {
			cmd := obj.(*busv1alpha1.Command)
			assert.Equal(t, "Restart", cmd.Action)
			assert.Equal(t, "Updating", cmd.Message)
		},
	}
	runIntoTest[busv1alpha1.Command, *busv1alpha1.Command](t, cmdTest)

	pgTest := testCase{
		name: "PodGroup Spec Copy",
		src:  &scheduling.PodGroup{Spec: scheduling.PodGroupSpec{MinMember: 2}},
		dst:  &scheduling.PodGroup{Spec: scheduling.PodGroupSpec{MinMember: 1}},
		validate: func(t *testing.T, obj client.Object) {
			assert.Equal(t, int32(2), obj.(*scheduling.PodGroup).Spec.MinMember)
		},
	}
	runIntoTest[scheduling.PodGroup, *scheduling.PodGroup](t, pgTest)

}

func TestIntoExtendedResources(t *testing.T) {
	tests := []testCase{
		{
			name: "RuntimeClass Fields Copy",
			src:  &nodev1.RuntimeClass{Handler: "runv", Overhead: &nodev1.Overhead{}},
			dst:  &nodev1.RuntimeClass{Handler: "runc"},
			validate: func(t *testing.T, obj client.Object) {
				rc := obj.(*nodev1.RuntimeClass)
				assert.Equal(t, "runv", rc.Handler)
				assert.NotNil(t, rc.Overhead)
			},
		},
		{
			name: "ClusterRole Rules Copy",
			src:  &rbacv1.ClusterRole{Rules: []rbacv1.PolicyRule{{Verbs: []string{"get"}}}},
			dst:  &rbacv1.ClusterRole{Rules: []rbacv1.PolicyRule{}},
			validate: func(t *testing.T, obj client.Object) {
				assert.Equal(t, "get", obj.(*rbacv1.ClusterRole).Rules[0].Verbs[0])
			},
		},
	}

	runIntoTest[nodev1.RuntimeClass, *nodev1.RuntimeClass](t, tests[0])
	runIntoTest[rbacv1.ClusterRole, *rbacv1.ClusterRole](t, tests[1])

	deviceClassTest := testCase{
		name: "Unstructured DeviceClass Copy",
		src: &unstructured.Unstructured{Object: map[string]any{
			"apiVersion": "resource.k8s.io/v1",
			"kind":       "DeviceClass",
			"metadata": map[string]any{
				"name": "npu.huawei.com",
			},
			"spec": map[string]any{
				"selectors": []any{map[string]any{"cel": map[string]any{"expression": "device.driver == \"npu.huawei.com\""}}},
			},
		}},
		dst: &unstructured.Unstructured{Object: map[string]any{
			"apiVersion": "resource.k8s.io/v1",
			"kind":       "DeviceClass",
			"metadata": map[string]any{
				"name": "npu.huawei.com",
			},
			"spec": map[string]any{
				"selectors": []any{},
			},
		}},
		validate: func(t *testing.T, obj client.Object) {
			selectors, ok, err := unstructured.NestedSlice(obj.(*unstructured.Unstructured).Object, "spec", "selectors")
			require.NoError(t, err)
			require.True(t, ok)
			require.Len(t, selectors, 1)
		},
	}
	runIntoTest[unstructured.Unstructured, *unstructured.Unstructured](t, deviceClassTest)
}

func int32Ptr(i int32) *int32 { return &i }
func boolPtr(b bool) *bool    { return &b }
