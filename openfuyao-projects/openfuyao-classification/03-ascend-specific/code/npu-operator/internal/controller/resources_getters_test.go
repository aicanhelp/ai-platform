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
	"testing"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	k8sbatchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"volcano.sh/apis/pkg/apis/batch/v1alpha1"
	busv1alpha1 "volcano.sh/apis/pkg/apis/bus/v1alpha1"
	flowv1alpha1 "volcano.sh/apis/pkg/apis/flow/v1alpha1"
	nodeinfov1alpha1 "volcano.sh/apis/pkg/apis/nodeinfo/v1alpha1"
	"volcano.sh/apis/pkg/apis/scheduling"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func TestConfigGetterFunctions(t *testing.T) {
	instance := getInstance()
	spec := &instance.Spec

	spec.Trainer.LogRotate.LogFile = "trainer.log"
	spec.DevicePlugin.LogRotate.LogFile = "device.log"
	spec.NodeD.LogRotate.LogFile = "noded.log"
	spec.Exporter.LogRotate.LogFile = "exporter.log"
	spec.Driver.LogRotate.LogFile = "driver.log"
	spec.ClusterD.LogRotate.LogFile = "clusterd.log"
	spec.RSController.LogRotate.LogFile = "rs.log"
	spec.VCScheduler.SchedulerConfigMap = "scheduler-config"
	spec.Volcano.Flavor = v1.VolcanoFlavorVNPU

	spec.Trainer.Command.Command = []string{"trainer"}
	spec.DevicePlugin.Command.Command = []string{"dp"}
	spec.NodeD.Command.Command = []string{"noded"}
	spec.Exporter.Command.Command = []string{"exporter"}
	spec.ClusterD.Command.Command = []string{"clusterd"}
	spec.RSController.Command.Command = []string{"rs"}
	spec.VCScheduler.Command.Command = []string{"scheduler"}
	spec.VCController.Command.Command = []string{"controller"}

	spec.Driver.Env = []corev1.EnvVar{{Name: "DRIVER", Value: "1"}}
	spec.MindIOTFT.Env = []corev1.EnvVar{{Name: "TFT", Value: "1"}}
	spec.MindIOACP.Env = []corev1.EnvVar{{Name: "ACP", Value: "1"}}
	spec.VCScheduler.Env = []corev1.EnvVar{{Name: "SCHEDULER", Value: "1"}}
	spec.VCController.Env = []corev1.EnvVar{{Name: "VC_CONTROLLER", Value: "1"}}
	spec.Trainer.Env = []corev1.EnvVar{{Name: "TRAINER", Value: "1"}}
	spec.NodeD.Env = []corev1.EnvVar{{Name: "NODED", Value: "1"}}
	spec.ClusterD.Env = []corev1.EnvVar{{Name: "CLUSTERD", Value: "1"}}
	spec.Exporter.Env = []corev1.EnvVar{{Name: "EXPORTER", Value: "1"}}
	spec.DevicePlugin.Env = []corev1.EnvVar{{Name: "DEVICE_PLUGIN", Value: "1"}}

	assert.Same(t, &spec.Trainer.LogRotate, trainerLogRotateConfigGetter(spec))
	assert.Same(t, &spec.DevicePlugin.LogRotate, devicePluginLogRotateConfigGetter(spec))
	assert.Same(t, &spec.NodeD.LogRotate, nodeDLogRotateConfigGetter(spec))
	assert.Same(t, &spec.Exporter.LogRotate, exporterLogRotateConfigGetter(spec))
	assert.Same(t, &spec.Driver.LogRotate, driverLogRotateConfigGetter(spec))
	assert.Same(t, &spec.ClusterD.LogRotate, clusterdLogRotateConfigGetter(spec))
	assert.Same(t, &spec.RSController.LogRotate, rsControllerLogRotateConfigGetter(spec))

	assert.Same(t, &spec.Trainer.Resources, trainerResourceRequirementsConfigGetter(spec))
	assert.Same(t, &spec.VCScheduler.Resources, vcSchedulerResourceRequirementsConfigGetter(spec))
	assert.Same(t, &spec.VCController.Resources, vcControllerResourceRequirementsConfigGetter(spec))
	assert.Same(t, &spec.DevicePlugin.Resources, devicePluginResourceRequirementsConfigGetter(spec))
	assert.Same(t, &spec.NodeD.Resources, nodeDResourceConfigGetter(spec))
	assert.Same(t, &spec.Exporter.Resources, exporterResourceRequirementsConfigGetter(spec))
	assert.Same(t, &spec.ClusterD.Resources, clusterdResourceConfigGetter(spec))
	assert.Same(t, &spec.RSController.Resources, rsControllerResourceConfigGetter(spec))

	assert.Same(t, &spec.VCScheduler.Command, schedulerCommandParametersGetter(spec))
	assert.Same(t, &spec.VCController.Command, vcControllerCommandParametersGetter(spec))
	assert.Same(t, &spec.Trainer.Command, trainerCommandParametersGetter(spec))
	assert.Same(t, &spec.DevicePlugin.Command, devicePluginCommandParametersGetter(spec))
	assert.Same(t, &spec.NodeD.Command, nodeDCommandParametersGetter(spec))
	assert.Same(t, &spec.Exporter.Command, exporterCommandParametersGetter(spec))
	assert.Same(t, &spec.ClusterD.Command, clusterDCommandParametersGetter(spec))
	assert.Same(t, &spec.RSController.Command, rsControllerCommandParametersGetter(spec))

	assert.Same(t, &spec.DevicePlugin.VolumeSpec, devicePluginVolumesGetter(spec))
	assert.Same(t, &spec.NodeD.VolumeSpec, nodeDVolumesGetter(spec))
	assert.Same(t, &spec.VCScheduler.VolumeSpec, vcSchedulerVolumesGetter(spec))
	assert.Same(t, &spec.VCController.VolumeSpec, vcControllerVolumesGetter(spec))
	assert.Same(t, &spec.Trainer.VolumeSpec, trainerVolumesGetter(spec))
	assert.Same(t, &spec.Exporter.VolumeSpec, exporterVolumesGetter(spec))
	assert.Same(t, &spec.ClusterD.VolumeSpec, clusterDVolumesGetter(spec))

	assert.Equal(t, spec.Driver.Env, driverEnvConfigGetter(spec))
	assert.Equal(t, spec.MindIOTFT.Env, tftEnvConfigGetter(spec))
	assert.Equal(t, spec.MindIOACP.Env, acpEnvConfigGetter(spec))
	assert.Equal(t, spec.VCScheduler.Env, vcSchedulerEnvConfigGetter(spec))
	assert.Equal(t, spec.VCController.Env, vcControllerEnvConfigGetter(spec))
	assert.Equal(t, spec.Trainer.Env, trainerEnvConfigGetter(spec))
	assert.Equal(t, spec.NodeD.Env, nodeDEnvConfigGetter(spec))
	assert.Equal(t, spec.ClusterD.Env, clusterDEnvConfigGetter(spec))
	assert.Equal(t, spec.Exporter.Env, exporterEnvConfigGetter(spec))
	assert.Equal(t, spec.DevicePlugin.Env, devicePluginEnvConfigGetter(spec))

	assert.Equal(t, "scheduler-config", schedulerConfigMapGetter(spec))
	assert.Equal(t, v1.VolcanoFlavorVNPU, volcanoFlavor(spec))
	assert.True(t, volcanoAdmissionEnabled(spec))
}

func TestImageConfigGetterFunctions(t *testing.T) {
	instance := getInstance()
	spec := &instance.Spec

	assert.Len(t, runtimeImageConfigGetter(spec), 3)
	assert.Same(t, &spec.OCIRuntime.Image, runtimeImageConfigGetter(spec)[0])
	assert.Same(t, &spec.OCIRuntime.InitRuntimeImage, runtimeImageConfigGetter(spec)[1])
	assert.Same(t, &spec.OCIRuntime.InitConfigImage, runtimeImageConfigGetter(spec)[2])

	assert.Len(t, vcSchedulerImageConfigGetter(spec), 2)
	assert.Same(t, &spec.VCScheduler.Image, vcSchedulerImageConfigGetter(spec)[0])
	assert.Same(t, &spec.VCScheduler.InitImage, vcSchedulerImageConfigGetter(spec)[1])

	assert.Len(t, vcControllerImageConfigGetter(spec), 2)
	assert.Same(t, &spec.VCController.Image, vcControllerImageConfigGetter(spec)[0])
	assert.Same(t, &spec.VCController.InitImage, vcControllerImageConfigGetter(spec)[1])

	assert.Len(t, trainerImageConfigGetter(spec), 2)
	assert.Same(t, &spec.Trainer.Image, trainerImageConfigGetter(spec)[0])
	assert.Same(t, &spec.Trainer.InitImage, trainerImageConfigGetter(spec)[1])

	assert.Len(t, devicePluginImageConfigGetter(spec), 2)
	assert.Same(t, &spec.DevicePlugin.Image, devicePluginImageConfigGetter(spec)[0])
	assert.Same(t, &spec.DevicePlugin.InitImage, devicePluginImageConfigGetter(spec)[1])

	assert.Len(t, nodeDImageConfigGetter(spec), 1)
	assert.Same(t, &spec.NodeD.Image, nodeDImageConfigGetter(spec)[0])
	assert.Same(t, &spec.Exporter.Image, exporterImageConfigGetter(spec)[0])
	assert.Same(t, &spec.Driver.Image, driverImageConfigGetter(spec)[0])
	assert.Same(t, &spec.Driver.InitImage, driverImageConfigGetter(spec)[1])
	assert.Same(t, &spec.ClusterD.Image, clusterdImageConfigGetter(spec)[0])
	assert.Same(t, &spec.RSController.Image, rscontrollerImageConfigGetter(spec)[0])
	assert.Same(t, &spec.MindIOTFT.Image, mindioTFTImageConfigGetter(spec)[0])
	assert.Same(t, &spec.MindIOACP.Image, mindioACPImageConfigGetter(spec)[0])
	assert.Same(t, &spec.Volcano.Admission.Image, volcanoAdmissionImageConfigGetter(spec)[0])
}

func TestObtainVolcanoVersion(t *testing.T) {
	scheme := runtime.NewScheme()
	utilruntime.Must(v1.AddToScheme(scheme))

	cr := getInstance()
	cr.Spec.VCScheduler.Image.Tag = "volcano-7.1.RC1"
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cr).Build()

	r := &NPUClusterPolicyReconciler{
		Client:   fakeClient,
		Scheme:   scheme,
		instance: &v1.NPUClusterPolicy{ObjectMeta: metav1.ObjectMeta{Name: cr.Name}},
	}

	version, err := ObtainVolcanoVersion(context.Background(), r)
	require.NoError(t, err)
	assert.Equal(t, "7.1.RC1", version)

	cr.Spec.VCScheduler.Image.Tag = ""
	require.NoError(t, fakeClient.Update(context.Background(), cr))
	version, err = ObtainVolcanoVersion(context.Background(), r)
	require.NoError(t, err)
	assert.Empty(t, version)

	r.instance.Name = "not-found"
	version, err = ObtainVolcanoVersion(context.Background(), r)
	require.Error(t, err)
	assert.Empty(t, version)
}

func TestTransformWrapperFunctions(t *testing.T) {
	r := wrapperTestReconciler()
	ctx := context.Background()

	assertNamespacedWrapperTransforms(t, ctx, r)
	assertGenericWrapperTransforms(t, ctx, r)
}

func wrapperTestReconciler() *NPUClusterPolicyReconciler {
	s := runtime.NewScheme()
	utilruntime.Must(v1.AddToScheme(s))
	utilruntime.Must(corev1.AddToScheme(s))
	utilruntime.Must(nodev1.AddToScheme(s))
	utilruntime.Must(rbacv1.AddToScheme(s))
	utilruntime.Must(coordinationv1.AddToScheme(s))
	utilruntime.Must(admissionregistrationv1.AddToScheme(s))
	utilruntime.Must(k8sbatchv1.AddToScheme(s))
	utilruntime.Must(apiextensionsv1.AddToScheme(s))
	utilruntime.Must(monitoringv1.AddToScheme(s))
	utilruntime.Must(v1alpha1.AddToScheme(s))
	utilruntime.Must(busv1alpha1.AddToScheme(s))
	utilruntime.Must(scheduling.AddToScheme(s))
	utilruntime.Must(nodeinfov1alpha1.AddToScheme(s))
	utilruntime.Must(flowv1alpha1.AddToScheme(s))

	return &NPUClusterPolicyReconciler{
		Scheme:    s,
		namespace: "npu-system",
		instance: &v1.NPUClusterPolicy{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NPUClusterPolicy",
				APIVersion: v1.GroupVersion.String(),
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "cluster",
				UID:  "uid-1",
			},
		},
	}
}

func assertNamespacedWrapperTransforms(t *testing.T, ctx context.Context, r *NPUClusterPolicyReconciler) {
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "sa", Namespace: placeholderValue}}
	require.NoError(t, serviceAccountReconcileHooks{}.transform(ctx, r, sa))
	assert.Equal(t, "npu-system", sa.Namespace)

	cr := &rbacv1.ClusterRole{ObjectMeta: metav1.ObjectMeta{Name: "cr", Namespace: placeholderValue}}
	require.NoError(t, clusterRoleReconcileHooks{}.transform(ctx, r, cr))
	assert.Equal(t, "npu-system", cr.Namespace)

	crb := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "crb"},
		Subjects: []rbacv1.Subject{
			{Kind: "ServiceAccount", Name: "sa", Namespace: placeholderValue},
		},
	}
	require.NoError(t, clusterRoleBindingReconcileHooks{}.transform(ctx, r, crb))
	assert.Equal(t, "npu-system", crb.Subjects[0].Namespace)

	role := &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: "role", Namespace: placeholderValue}}
	require.NoError(t, roleReconcileHooks{}.transform(ctx, r, role))
	assert.Equal(t, "npu-system", role.Namespace)

	rb := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "rb", Namespace: placeholderValue},
		Subjects: []rbacv1.Subject{
			{Kind: "ServiceAccount", Name: "sa", Namespace: placeholderValue},
		},
	}
	require.NoError(t, roleBindingReconcileHooks{}.transform(ctx, r, rb))
	assert.Equal(t, "npu-system", rb.Namespace)
	assert.Equal(t, "npu-system", rb.Subjects[0].Namespace)

	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "secret", Namespace: placeholderValue}}
	require.NoError(t, secretReconcileHooks{}.transform(ctx, r, secret))
	assert.Equal(t, "npu-system", secret.Namespace)

	job := &k8sbatchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "job", Namespace: placeholderValue}}
	require.NoError(t, batchJobReconcileHooks{}.transform(ctx, r, job))
	assert.Equal(t, "npu-system", job.Namespace)
}

func assertGenericWrapperTransforms(t *testing.T, ctx context.Context, r *NPUClusterPolicyReconciler) {
	require.NoError(t, serviceMoniterReconcileHooks{}.transform(ctx, r, &monitoringv1.ServiceMonitor{ObjectMeta: metav1.ObjectMeta{Name: "sm"}}))
	require.NoError(t, serviceReconcileHooks{}.transform(ctx, r, &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "svc"}}))
	require.NoError(t, leaseReconcileHooks{}.transform(ctx, r, &coordinationv1.Lease{ObjectMeta: metav1.ObjectMeta{Name: "lease"}}))
	require.NoError(t, jobReconcileHooks{}.transform(ctx, r, &v1alpha1.Job{ObjectMeta: metav1.ObjectMeta{Name: "job"}}))
	require.NoError(t, commandReconcileHooks{}.transform(ctx, r, &busv1alpha1.Command{ObjectMeta: metav1.ObjectMeta{Name: "cmd"}}))
	require.NoError(t, podGroupReconcileHooks{}.transform(ctx, r, &scheduling.PodGroup{ObjectMeta: metav1.ObjectMeta{Name: "pg"}}))
	require.NoError(t, queueReconcileHooks{}.transform(ctx, r, &scheduling.Queue{ObjectMeta: metav1.ObjectMeta{Name: "queue"}}))
	require.NoError(t, numatopologyReconcileHooks{}.transform(ctx, r, &nodeinfov1alpha1.Numatopology{ObjectMeta: metav1.ObjectMeta{Name: "numa"}}))
	require.NoError(t, jobTemplateReconcileHooks{}.transform(ctx, r, &flowv1alpha1.JobTemplate{ObjectMeta: metav1.ObjectMeta{Name: "jt"}}))
	require.NoError(t, jobFlowReconcileHooks{}.transform(ctx, r, &flowv1alpha1.JobFlow{ObjectMeta: metav1.ObjectMeta{Name: "jf"}}))
	require.NoError(t, CRDReconcileHooks{}.transform(ctx, r, &apiextensionsv1.CustomResourceDefinition{ObjectMeta: metav1.ObjectMeta{Name: "crd"}}))
	require.NoError(t, mutatingWebhookConfigurationReconcileHooks{}.transform(ctx, r,
		&admissionregistrationv1.MutatingWebhookConfiguration{ObjectMeta: metav1.ObjectMeta{Name: "mwh"}}))
	require.NoError(t, validatingWebhookConfigurationReconcileHooks{}.transform(ctx, r,
		&admissionregistrationv1.ValidatingWebhookConfiguration{ObjectMeta: metav1.ObjectMeta{Name: "vwh"}}))
}
