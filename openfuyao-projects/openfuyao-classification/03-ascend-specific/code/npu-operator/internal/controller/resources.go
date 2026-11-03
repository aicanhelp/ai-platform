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
	"errors"
	"fmt"
	"maps"
	"os"
	"strings"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	k8sbatchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"volcano.sh/apis/pkg/apis/batch/v1alpha1"
	busv1alpha1 "volcano.sh/apis/pkg/apis/bus/v1alpha1"
	flowv1alpha1 "volcano.sh/apis/pkg/apis/flow/v1alpha1"
	nodeinfov1alpha1 "volcano.sh/apis/pkg/apis/nodeinfo/v1alpha1"
	"volcano.sh/apis/pkg/apis/scheduling"

	v1 "openfuyao.com/npu-operator/api/v1"
)

const (
	hashAnnotationKey                     = "openfuyao.com/last-applied-hash"
	placeholderValue                      = "FILLED_BY_OPERATOR"
	releaseTagPlaceholder                 = "__NPU_OPERATOR_RELEASE_TAG__"
	operatorReleaseTagEnvName             = "NPU_OPERATOR_RELEASE_TAG"
	operatorBusyboxImageRepositoryEnvName = "NPU_OPERATOR_BUSYBOX_IMAGE_REPOSITORY"
	operatorBusyboxImageTagEnvName        = "NPU_OPERATOR_BUSYBOX_IMAGE_TAG"
	operatorBusyboxImagePullPolicyEnvName = "NPU_OPERATOR_BUSYBOX_IMAGE_PULL_POLICY"
	volcanoAdmissionSecretHashAnnotation  = "openfuyao.com/volcano-admission-secret-hash"

	operatorName                     = "npu-operator"
	runtimeDaemonSetNamePrefix       = "ascend-runtime-"
	driverDaemonSetName              = "npu-driver"
	devicePluginDaemonSetName        = "ascend-device-plugin"
	trainerDeploymentName            = "ascend-operator-manager"
	clusterdDeploymentName           = "clusterd"
	nodedDaemonsetName               = "noded"
	exporterDaemonsetName            = "npu-exporter"
	rsControllerDeploymentName       = "resilience-controller"
	vcControllerDeploymentName       = "volcano-controllers"
	vcSchedulerDeploymentName        = "volcano-scheduler"
	volcanoAdmissionDeploymentName   = "volcano-admission"
	volcanoAdmissionInitJobName      = "volcano-admission-init"
	volcanoAdmissionSecretName       = "volcano-admission-secret"
	volcanoAdmissionCAKey            = "ca.crt"
	volcanoSystemNamespace           = "volcano-system"
	ascendVCControllerRepo           = "hub.oepkgs.net/openfuyao/ascendhub/vc-controller-manager"
	ascendVCSchedulerRepo            = "hub.oepkgs.net/openfuyao/ascendhub/vc-scheduler"
	vnpuVCControllerRepo             = "cr.openfuyao.cn/openfuyao/vnpu/vc-controller-manager"
	vnpuVCSchedulerRepo              = "cr.openfuyao.cn/openfuyao/vnpu/vc-scheduler"
	vnpuVolcanoAdmissionRepo         = "cr.openfuyao.cn/openfuyao/vnpu/vc-webhook-manager"
	mindIOACPDaemonsetName           = "mindio-acp"
	mindIOTFTDaemonsetName           = "mindio-tft"
	volcanoSchedulerConfigMapName    = "volcano-scheduler-configmap"
	vnpuClientUpdateDaemonSetName    = "npu-client-update-daemonset"
	vnpuDevicePluginDaemonSetName    = "npu-device-plugin-daemonset"
	vnpuExporterDaemonSetName        = "xpu-exporter-daemonset"
	vnpuExporterServiceName          = "xpu-exporter-svc"
	draKubeletPluginDaemonSetName    = "npu-dra-kubeletplugin"
	draVCANNRTInstallerDaemonSetName = "vcannrt-installer"
	draWebhookDeploymentName         = "dra-webhook"
	draWebhookServiceName            = "dra-webhook-service"
	draWebhookConfigurationName      = "dra-webhook-resource-claim-parameters"
	vnpuVirtualTemplateConfigMapName = "vnpu.virtual-template.config"
	vnpuNodeModeConfigMapName        = "vnpu.node-mode.config"
	draProfileConfigMapName          = "dra-profile-config"
	draChipCapabilitiesConfigMapName = "vnpu-template-config"
	draDeviceClassManagedAnnotation  = "openfuyao.com/dra-deviceclass-managed"

	defaultVNPUDeviceSplitCount            = int32(20)
	defaultDRADriverName                   = "npu.huawei.com"
	defaultDRADeviceProfile                = "npu"
	defaultDRACDIRoot                      = "/var/run/cdi"
	defaultDRAHealthcheckPort              = int32(-1)
	defaultDRAKubeletRegistrarDir          = "/var/lib/kubelet/plugins_registry"
	defaultDRAKubeletPluginsDir            = "/var/lib/kubelet/plugins"
	defaultDRASoftVNPUShareCount           = int32(16)
	defaultDRASoftVNPUShareShmDir          = "/shared"
	defaultDRASoftVNPUShareDevConfigDir    = "/etc/enpu"
	defaultDRASoftVNPUNPUProfileConfigPath = "/etc/dra/dra.config"
	defaultDRASoftVNPUChipCapabilitiesCM   = "vnpu-template-config"
	defaultDRAWebhookPort                  = int32(443)
	defaultDRAWebhookDeviceProfile         = "gpu"
	defaultDRAWebhookTLSCertFile           = "/tls/tls.crt"
	defaultDRAWebhookTLSPrivateKeyFile     = "/tls/tls.key"
	defaultDRAWebhookFailurePolicy         = "Fail"
	defaultVNPUExporterPort                = int32(8082)
	defaultBusyboxImageRepository          = "hub.oepkgs.net/openfuyao/busybox"
	defaultBusyboxImageTag                 = "1.36.1"
	defaultBusyboxImagePullPolicy          = "Always"

	logRotateLogLevelEnvName = "NPU_OPERATOR_LOG_LEVEL"
	logRotateLogFileEnvName  = "NPU_OPERATOR_LOG_FILE"
	logRotateRotateEnvName   = "NPU_OPERATOR_ROTATE"
	logRotateMaxAgeEnvName   = "NPU_OPERATOR_MAX_AGE"
	logRotateCompressEnvName = "NPU_OPERATOR_COMPRESS"

	operatorNodeEnvName = "OPERATOR_NODE_NAME"
)

var (
	gvkNamespace = schema.GroupVersionKind{
		Version: "v1",
		Kind:    "Namespace",
	}
	gvkRuntimeClass = schema.GroupVersionKind{
		Group:   "node.k8s.io",
		Version: "v1",
		Kind:    "RuntimeClass",
	}
	gvkDaemonSet = schema.GroupVersionKind{
		Group:   "apps",
		Version: "v1",
		Kind:    "DaemonSet",
	}
	gvkDeployment = schema.GroupVersionKind{
		Group:   "apps",
		Version: "v1",
		Kind:    "Deployment",
	}
	gvkServiceAccount = schema.GroupVersionKind{
		Version: "v1",
		Kind:    "ServiceAccount",
	}
	gvkClusterRole = schema.GroupVersionKind{
		Group:   "rbac.authorization.k8s.io",
		Version: "v1",
		Kind:    "ClusterRole",
	}
	gvkClusterRoleBinding = schema.GroupVersionKind{
		Group:   "rbac.authorization.k8s.io",
		Version: "v1",
		Kind:    "ClusterRoleBinding",
	}
	gvkRole = schema.GroupVersionKind{
		Group:   "rbac.authorization.k8s.io",
		Version: "v1",
		Kind:    "Role",
	}
	gvkRoleBinding = schema.GroupVersionKind{
		Group:   "rbac.authorization.k8s.io",
		Version: "v1",
		Kind:    "RoleBinding",
	}
	gvkServiceMonitor = schema.GroupVersionKind{
		Group:   "monitoring.coreos.com",
		Version: "v1",
		Kind:    "ServiceMonitor",
	}
	gvkService = schema.GroupVersionKind{
		Version: "v1",
		Kind:    "Service",
	}
	gvkSecret = schema.GroupVersionKind{
		Version: "v1",
		Kind:    "Secret",
	}
	gvkBatchJob = schema.GroupVersionKind{
		Group:   "batch",
		Version: "v1",
		Kind:    "Job",
	}
	gvkLease = schema.GroupVersionKind{
		Group:   "coordination.k8s.io",
		Version: "v1",
		Kind:    "Lease",
	}
	gvkMutatingWebhookConfiguration = schema.GroupVersionKind{
		Group:   "admissionregistration.k8s.io",
		Version: "v1",
		Kind:    "MutatingWebhookConfiguration",
	}
	gvkValidatingWebhookConfiguration = schema.GroupVersionKind{
		Group:   "admissionregistration.k8s.io",
		Version: "v1",
		Kind:    "ValidatingWebhookConfiguration",
	}
	gvkVolcanoBatchJob = schema.GroupVersionKind{
		Group:   "batch.volcano.sh",
		Version: "v1alpha1",
		Kind:    "Job",
	}
	gvkVolcanoBusCommand = schema.GroupVersionKind{
		Group:   "bus.volcano.sh",
		Version: "v1alpha1",
		Kind:    "Command",
	}
	gvkSchedulingPodgroup = schema.GroupVersionKind{
		Group:   "scheduling.volcano.sh",
		Version: "v1beta1",
		Kind:    "PodGroup",
	}
	gvkSchedulerQueue = schema.GroupVersionKind{
		Group:   "scheduling.volcano.sh",
		Version: "v1beta1",
		Kind:    "Queue",
	}
	gvkNodeInfo = schema.GroupVersionKind{
		Group:   "nodeinfo.volcano.sh",
		Version: "v1alpha1",
		Kind:    "Numatopology",
	}
	gvkJobTemplate = schema.GroupVersionKind{
		Group:   "flow.volcano.sh",
		Version: "v1alpha1",
		Kind:    "JobTemplate",
	}
	gvkJobFlow = schema.GroupVersionKind{
		Group:   "flow.volcano.sh",
		Version: "v1alpha1",
		Kind:    "JobFlow",
	}
	gvkConfigMap = schema.GroupVersionKind{
		Version: "v1",
		Kind:    "ConfigMap",
	}
	gvkCRD = schema.GroupVersionKind{
		Group:   "apiextensions.k8s.io",
		Version: "v1",
		Kind:    "CustomResourceDefinition",
	}
	gvkDeviceClass = schema.GroupVersionKind{
		Group:   "resource.k8s.io",
		Version: "v1",
		Kind:    "DeviceClass",
	}
	gvkResourceSlice = schema.GroupVersionKind{
		Group:   "resource.k8s.io",
		Version: "v1",
		Kind:    "ResourceSlice",
	}

	resourceCreators = map[schema.GroupVersionKind]func(client.Object) componentResource{
		gvkCRD: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*apiextensionsv1.CustomResourceDefinition), CRDReconcileHooks{})
		},
		gvkDeviceClass: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*unstructured.Unstructured), unstructuredReconcileHooks{})
		},
		gvkNamespace: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*corev1.Namespace), namespaceReconcileHooks{})
		},
		gvkRuntimeClass: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*nodev1.RuntimeClass), runtimeClassReconcileHooks{})
		},
		gvkDaemonSet: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*appsv1.DaemonSet), newDaemonSetReconcileHooks(obj))
		},
		gvkDeployment: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*appsv1.Deployment), newDeploymentReconcileHooks(obj))
		},
		gvkServiceAccount: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*corev1.ServiceAccount), serviceAccountReconcileHooks{})
		},
		gvkClusterRole: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*rbacv1.ClusterRole), clusterRoleReconcileHooks{})
		},
		gvkClusterRoleBinding: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*rbacv1.ClusterRoleBinding), clusterRoleBindingReconcileHooks{})
		},
		gvkRole: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*rbacv1.Role), roleReconcileHooks{})
		},
		gvkRoleBinding: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*rbacv1.RoleBinding), roleBindingReconcileHooks{})
		},
		gvkServiceMonitor: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*monitoringv1.ServiceMonitor), serviceMoniterReconcileHooks{})
		},
		gvkService: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*corev1.Service), serviceReconcileHooks{})
		},
		gvkSecret: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*corev1.Secret), secretReconcileHooks{})
		},
		gvkBatchJob: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*k8sbatchv1.Job), batchJobReconcileHooks{})
		},
		gvkLease: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*coordinationv1.Lease), leaseReconcileHooks{})
		},
		gvkMutatingWebhookConfiguration: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*admissionregistrationv1.MutatingWebhookConfiguration),
				mutatingWebhookConfigurationReconcileHooks{})
		},
		gvkValidatingWebhookConfiguration: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*admissionregistrationv1.ValidatingWebhookConfiguration),
				validatingWebhookConfigurationReconcileHooks{})
		},
		gvkConfigMap: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*corev1.ConfigMap), configMapReconcileHooks{})
		},
		gvkVolcanoBatchJob: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*v1alpha1.Job), jobReconcileHooks{})
		},
		gvkVolcanoBusCommand: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*busv1alpha1.Command), commandReconcileHooks{})
		},
		gvkSchedulingPodgroup: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*scheduling.PodGroup), podGroupReconcileHooks{})
		},
		gvkSchedulerQueue: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*scheduling.Queue), queueReconcileHooks{})
		},
		gvkNodeInfo: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*nodeinfov1alpha1.Numatopology), numatopologyReconcileHooks{})
		},
		gvkJobTemplate: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*flowv1alpha1.JobTemplate), jobTemplateReconcileHooks{})
		},
		gvkJobFlow: func(obj client.Object) componentResource {
			return newComponentResource(obj.(*flowv1alpha1.JobFlow), jobFlowReconcileHooks{})
		},
	}

	errSkipCreation       = errors.New("skip resource creation")
	errSkipReconciliation = errors.New("skip resource reconciliation")
)

type componentResource interface {
	clone() componentResource
	gvk() string
	ref(defaultNamespace string) componentResourceRef
	reconcile(ctx context.Context, r *NPUClusterPolicyReconciler, isManaged bool) (*v1.ComponentState, error)
}

type componentResourceRef struct {
	gvk       schema.GroupVersionKind
	namespace string
	name      string
}

type componentResourceReconcileHooks[T any] interface {
	canDelete(context.Context, *NPUClusterPolicyReconciler, *T) (client.Object, bool, error) // 是否可以删除该资源
	transform(context.Context, *NPUClusterPolicyReconciler, *T) error                        // 为资源对象设置 OwnerReference，将控制器实例标记为这个资源的所有者控制者。
	inspect(context.Context, *NPUClusterPolicyReconciler, *T) (*v1.ComponentState, error)    // 检查组件
}

type (
	componentResourceTransformer[T any] func(context.Context, *NPUClusterPolicyReconciler, *T) error
	componentResourceInspector[T any]   func(context.Context, *NPUClusterPolicyReconciler, *T) error
)

type clientObject[T any] interface {
	*T
	client.Object
}

type typedComponentResource[T any, PT clientObject[T]] struct {
	componentResourceReconcileHooks[T]
	obj PT
}

func (res typedComponentResource[T, PT]) clone() componentResource {
	return typedComponentResource[T, PT]{
		componentResourceReconcileHooks: res.componentResourceReconcileHooks,
		obj:                             res.obj.DeepCopyObject().(PT),
	}
}

func (res typedComponentResource[T, PT]) gvk() string {
	gvk := res.obj.GetObjectKind().GroupVersionKind()
	return fmt.Sprintf("%s/%s", gvk.GroupKind(), gvk.Version)
}

func (res typedComponentResource[T, PT]) ref(defaultNamespace string) componentResourceRef {
	namespace := res.obj.GetNamespace()
	if namespace == placeholderValue {
		namespace = defaultNamespace
	}
	return componentResourceRef{
		gvk:       res.obj.GetObjectKind().GroupVersionKind(),
		namespace: namespace,
		name:      res.obj.GetName(),
	}
}

func (res typedComponentResource[T, PT]) reconcile(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	isManaged bool,
) (*v1.ComponentState, error) {
	gvk := res.gvk()
	logger := log.FromContext(ctx, gvk, res.obj.GetName())
	ctx = log.IntoContext(ctx, logger)
	mutated := true

	if !isManaged {
		// If component is not managed by operator and resource can be deleted
		realObj, ok, err := res.canDelete(ctx, r, res.obj)
		if err != nil {
			return nil, fmt.Errorf("check deletion of %s resource: %w", gvk, err)
		}
		if ok {
			if err := r.Delete(ctx, realObj); client.IgnoreNotFound(err) != nil {
				return nil, fmt.Errorf("delete %s resource: %w", gvk, err)
			}
		}
		return nil, nil
	}

	if err := res.transform(ctx, r, res.obj); err != nil {
		if errors.Is(err, errSkipReconciliation) {
			return nil, nil
		}
		if !errors.Is(err, errSkipCreation) {
			return nil, fmt.Errorf("transform %s resource: %w", gvk, err)
		}
		mutated = false
	}

	if mutated {
		// Create new object to receive transformed resource
		o, err := r.Scheme.New(res.obj.GetObjectKind().GroupVersionKind()) // 创建一个新的对象实例，返回新的空对象gvk
		if err != nil {
			if u, ok := any(res.obj).(*unstructured.Unstructured); ok {
				o = u.DeepCopyObject()
			} else {
				return nil, fmt.Errorf("new %s object: %w", gvk, err)
			}
		}
		obj := o.(client.Object)
		obj.SetNamespace(res.obj.GetNamespace())
		obj.SetName(res.obj.GetName())

		op, err := controllerutil.CreateOrUpdate(ctx, r.Client, obj, func() error {
			res.into(obj) // 使得obj的内容和res.obj一样
			return nil
		})
		if err != nil && !errors.Is(err, errSkipCreation) {
			return nil, fmt.Errorf("create or update %s resource: %w", gvk, err)
		}
		logger.V(1).Info(fmt.Sprintf("Resource %s", op))
	}

	state, err := res.inspect(ctx, r, res.obj)
	if err != nil && state == nil {
		state = &v1.ComponentState{
			Phase:   v1.ComponentTerminated,
			Reason:  reconcileFailedReason,
			Message: err.Error(),
		}
	}
	return state, err
}

func (res typedComponentResource[T, PT]) into(receiver client.Object) {
	currentLabels := receiver.GetLabels()
	if currentLabels == nil {
		currentLabels = make(map[string]string)
	}
	for k, v := range res.obj.GetLabels() {
		currentLabels[k] = v
	}
	receiver.SetLabels(currentLabels)
	currentAnno := receiver.GetAnnotations()
	if currentAnno == nil {
		currentAnno = make(map[string]string)
	}
	for k, v := range res.obj.GetAnnotations() {
		currentAnno[k] = v
	}
	receiver.SetAnnotations(currentAnno)

	receiver.SetOwnerReferences(res.obj.GetOwnerReferences())
	receiver.SetFinalizers(res.obj.GetFinalizers())
	if res.copyWorkloads(receiver) {
		return
	}
	if res.copyCoreResources(receiver) {
		return
	}
	if res.copyVolcanoResources(receiver) {
		return
	}
	if res.copyExtendedResources(receiver) {
		return
	}
	log.Log.WithName("npu-operator").Error(nil, "Unhandled type in into()", "type", fmt.Sprintf("%T", receiver))
}

func (res typedComponentResource[T, PT]) copyWorkloads(receiver client.Object) bool {
	switch target := receiver.(type) {
	case *appsv1.DaemonSet:
		target.Spec = any(res.obj).(*appsv1.DaemonSet).Spec
		return true
	case *appsv1.Deployment:
		target.Spec = any(res.obj).(*appsv1.Deployment).Spec
		return true
	case *v1alpha1.Job:
		target.Spec = any(res.obj).(*v1alpha1.Job).Spec
		return true
	case *k8sbatchv1.Job:
		target.Spec = any(res.obj).(*k8sbatchv1.Job).Spec
		return true
	}
	return false
}

func (res typedComponentResource[T, PT]) copyCoreResources(receiver client.Object) bool {
	switch target := receiver.(type) {
	case *corev1.Service:
		s := any(res.obj).(*corev1.Service)
		oldIP := target.Spec.ClusterIP
		target.Spec = s.Spec
		if oldIP != "" {
			target.Spec.ClusterIP = oldIP
		}
		return true
	case *corev1.ConfigMap:
		s := any(res.obj).(*corev1.ConfigMap)
		target.Data = s.Data
		target.BinaryData = s.BinaryData
		return true
	case *corev1.ServiceAccount:
		s := any(res.obj).(*corev1.ServiceAccount)
		target.Secrets = s.Secrets
		target.ImagePullSecrets = s.ImagePullSecrets
		target.AutomountServiceAccountToken = s.AutomountServiceAccountToken
		return true
	case *corev1.Secret:
		s := any(res.obj).(*corev1.Secret)
		target.Type = s.Type
		target.Data = s.Data
		target.StringData = s.StringData
		target.Immutable = s.Immutable
		return true
	case *corev1.Namespace:
		target.Spec = any(res.obj).(*corev1.Namespace).Spec
		return true
	}
	return false
}

func (res typedComponentResource[T, PT]) copyVolcanoResources(receiver client.Object) bool {
	switch target := receiver.(type) {
	case *scheduling.PodGroup:
		target.Spec = any(res.obj).(*scheduling.PodGroup).Spec
		return true
	case *scheduling.Queue:
		target.Spec = any(res.obj).(*scheduling.Queue).Spec
		return true
	case *nodeinfov1alpha1.Numatopology:
		target.Spec = any(res.obj).(*nodeinfov1alpha1.Numatopology).Spec
		return true
	case *busv1alpha1.Command:
		target.Action = any(res.obj).(*busv1alpha1.Command).Action
		target.TargetObject = any(res.obj).(*busv1alpha1.Command).TargetObject
		target.Message = any(res.obj).(*busv1alpha1.Command).Message
		target.Reason = any(res.obj).(*busv1alpha1.Command).Reason
		return true
	case *flowv1alpha1.JobTemplate:
		target.Spec = any(res.obj).(*flowv1alpha1.JobTemplate).Spec
		return true
	case *flowv1alpha1.JobFlow:
		target.Spec = any(res.obj).(*flowv1alpha1.JobFlow).Spec
		return true
	}
	return false
}

func (res typedComponentResource[T, PT]) copyExtendedResources(receiver client.Object) bool {
	switch target := receiver.(type) {
	case *nodev1.RuntimeClass:
		s := any(res.obj).(*nodev1.RuntimeClass)
		target.Handler = s.Handler
		target.Overhead = s.Overhead
		target.Scheduling = s.Scheduling
		return true
	case *rbacv1.ClusterRole:
		s := any(res.obj).(*rbacv1.ClusterRole)
		target.Rules = s.Rules
		target.AggregationRule = s.AggregationRule
		return true
	case *rbacv1.ClusterRoleBinding:
		s := any(res.obj).(*rbacv1.ClusterRoleBinding)
		target.Subjects = s.Subjects
		target.RoleRef = s.RoleRef
		return true
	case *rbacv1.Role:
		s := any(res.obj).(*rbacv1.Role)
		target.Rules = s.Rules
		return true
	case *rbacv1.RoleBinding:
		s := any(res.obj).(*rbacv1.RoleBinding)
		target.Subjects = s.Subjects
		target.RoleRef = s.RoleRef
		return true
	case *monitoringv1.ServiceMonitor:
		target.Spec = any(res.obj).(*monitoringv1.ServiceMonitor).Spec
		return true
	case *apiextensionsv1.CustomResourceDefinition:
		target.Spec = any(res.obj).(*apiextensionsv1.CustomResourceDefinition).Spec
		return true
	case *coordinationv1.Lease:
		target.Spec = any(res.obj).(*coordinationv1.Lease).Spec
		return true
	case *admissionregistrationv1.MutatingWebhookConfiguration:
		target.Webhooks = any(res.obj).(*admissionregistrationv1.MutatingWebhookConfiguration).Webhooks
		return true
	case *admissionregistrationv1.ValidatingWebhookConfiguration:
		target.Webhooks = any(res.obj).(*admissionregistrationv1.ValidatingWebhookConfiguration).Webhooks
		return true
	case *unstructured.Unstructured:
		target.Object = any(res.obj).(*unstructured.Unstructured).DeepCopy().Object
		return true
	}
	return false
}

func newComponentResource[T any, PT clientObject[T]](
	obj *T,
	hooks componentResourceReconcileHooks[T],
) componentResource {
	return typedComponentResource[T, PT]{hooks, obj}
}

type defaultResourceReconcileHooks[T client.Object] struct{}

func (defaultResourceReconcileHooks[T]) canDelete(ctx context.Context, r *NPUClusterPolicyReconciler, obj T) (client.Object, bool, error) {
	logger := log.FromContext(ctx)
	if obj.GetNamespace() == placeholderValue {
		obj.SetNamespace(r.namespace)
	}
	key := client.ObjectKeyFromObject(obj)
	currentResource := obj.DeepCopyObject().(T)
	currentResource.SetNamespace(key.Namespace)
	currentResource.SetName(key.Name)
	if err := r.Get(ctx, key, currentResource); err != nil {
		// Optional APIs may never have been installed in this cluster.
		if optionalAPIAbsent(err, obj.GetObjectKind().GroupVersionKind().GroupVersion()) {
			return nil, false, nil
		}
		return nil, false, err
	}
	references := currentResource.GetOwnerReferences()
	logger.Info("Reference", "reference", references)
	if references == nil {
		return nil, false, nil
	}
	for _, ref := range references {
		if ownerReferenceMatchesInstance(ref, r.instance) {
			return currentResource, true, nil
		}
	}
	return nil, false, nil
}

func ownerReferenceMatchesInstance(ref metav1.OwnerReference, instance *v1.NPUClusterPolicy) bool {
	kind := instance.Kind
	if kind == "" {
		kind = v1.GroupVersion.WithKind("NPUClusterPolicy").Kind
	}
	return ref.Kind == kind && ref.Name == instance.Name && ref.UID == instance.UID
}

func (defaultResourceReconcileHooks[T]) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj T,
) error {
	// set ownerReference.controller to true
	if err := controllerutil.SetControllerReference(r.instance, obj, r.Scheme); err != nil {
		return fmt.Errorf("set owner as controller: %w", err)
	}

	return nil
}

func (defaultResourceReconcileHooks[T]) inspect(
	context.Context,
	*NPUClusterPolicyReconciler,
	T,
) (*v1.ComponentState, error) {
	return nil, nil
}

type namespaceReconcileHooks struct {
	defaultResourceReconcileHooks[*corev1.Namespace]
}

func (h namespaceReconcileHooks) canDelete(
	context.Context,
	*NPUClusterPolicyReconciler,
	*corev1.Namespace,
) (client.Object, bool, error) {
	return nil, false, nil
}

func (h namespaceReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *corev1.Namespace,
) error {
	found := &corev1.Namespace{}
	if err := r.Get(ctx, client.ObjectKeyFromObject(obj), found); err != nil {
		if apierrors.IsNotFound(err) {
			return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
		}
		return err
	}
	return nil
}

type runtimeClassReconcileHooks struct {
	defaultResourceReconcileHooks[*nodev1.RuntimeClass]
}

type unstructuredReconcileHooks struct {
	defaultResourceReconcileHooks[*unstructured.Unstructured]
}

func (h unstructuredReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *unstructured.Unstructured,
) error {
	if obj.GroupVersionKind() == gvkDeviceClass {
		rewriteDRADeviceClass(obj, draDriverName(&r.instance.Spec))
		annotations := obj.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		annotations[draDeviceClassManagedAnnotation] = "true"
		obj.SetAnnotations(annotations)
	}
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

func (h runtimeClassReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *nodev1.RuntimeClass,
) error {
	if obj.Name == placeholderValue {
		obj.Name = r.instance.Spec.Operator.RuntimeClass
	}
	obj.Handler = r.instance.Spec.Operator.RuntimeClass
	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

func buildDRADeviceClassTemplateObjects(templates []v1.DRADeviceClassTemplate) ([]*unstructured.Unstructured, error) {
	objects := make([]*unstructured.Unstructured, 0, len(templates))
	for _, tmpl := range templates {
		if tmpl.Name == "" {
			return nil, fmt.Errorf("DRA DeviceClass template name is empty")
		}
		if tmpl.SelectorCEL == "" {
			return nil, fmt.Errorf("DRA DeviceClass template %s selectorCEL is empty", tmpl.Name)
		}
		obj := &unstructured.Unstructured{Object: map[string]any{
			"apiVersion": "resource.k8s.io/v1",
			"kind":       "DeviceClass",
			"metadata": map[string]any{
				"name": tmpl.Name,
				"annotations": map[string]any{
					draDeviceClassManagedAnnotation: "true",
				},
			},
			"spec": map[string]any{
				"selectors": []any{map[string]any{"cel": map[string]any{
					"expression": tmpl.SelectorCEL,
				}}},
			},
		}}
		objects = append(objects, obj)
	}
	return objects, nil
}

func rewriteDRADeviceClass(obj *unstructured.Unstructured, driverName string) {
	if driverName == defaultDRADriverName {
		return
	}
	obj.SetName(strings.ReplaceAll(obj.GetName(), defaultDRADriverName, driverName))
	selectors, found, err := unstructured.NestedSlice(obj.Object, "spec", "selectors")
	if err != nil || !found {
		return
	}
	for i := range selectors {
		selector, ok := selectors[i].(map[string]any)
		if !ok {
			continue
		}
		cel, ok := selector["cel"].(map[string]any)
		if !ok {
			continue
		}
		expression, ok := cel["expression"].(string)
		if !ok {
			continue
		}
		cel["expression"] = strings.ReplaceAll(expression, defaultDRADriverName, driverName)
	}
	_ = unstructured.SetNestedSlice(obj.Object, selectors, "spec", "selectors")
}

func reconcileDRADeviceClassGarbage(ctx context.Context, r *NPUClusterPolicyReconciler) error {
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   gvkDeviceClass.Group,
		Version: gvkDeviceClass.Version,
		Kind:    "DeviceClassList",
	})
	if err := r.List(ctx, list); err != nil {
		if (!r.instance.Spec.DRA.Managed || !r.instance.Spec.DRA.DeviceClasses.Managed) &&
			optionalAPIAbsent(err, gvkDeviceClass.GroupVersion()) {
			return nil
		}
		return fmt.Errorf("list DRA DeviceClasses: %w", err)
	}
	desired := desiredDRADeviceClassNames(&r.instance.Spec)
	for i := range list.Items {
		item := &list.Items[i]
		if item.GetAnnotations()[draDeviceClassManagedAnnotation] != "true" || desired[item.GetName()] {
			continue
		}
		if !hasOwnerReference(item.GetOwnerReferences(), r.instance) {
			continue
		}
		if err := deleteDRAResourceSlicesForDeviceClass(ctx, r, item); err != nil {
			return err
		}
		if err := r.Delete(ctx, item); client.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete stale DRA DeviceClass %s: %w", item.GetName(), err)
		}
	}
	return nil
}

func deleteDRAResourceSlicesForDeviceClass(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	deviceClass *unstructured.Unstructured,
) error {
	driverName := driverNameFromDRADeviceClass(deviceClass)
	if driverName == "" {
		return nil
	}
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   gvkResourceSlice.Group,
		Version: gvkResourceSlice.Version,
		Kind:    "ResourceSliceList",
	})
	if err := r.List(ctx, list); err != nil {
		if optionalAPIAbsent(err, gvkResourceSlice.GroupVersion()) {
			return nil
		}
		return fmt.Errorf("list DRA ResourceSlices for driver %s: %w", driverName, err)
	}
	for i := range list.Items {
		item := &list.Items[i]
		driver, _, _ := unstructured.NestedString(item.Object, "spec", "driver")
		if driver != driverName {
			continue
		}
		if err := r.Delete(ctx, item); client.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete stale DRA ResourceSlice %s for driver %s: %w", item.GetName(), driverName, err)
		}
	}
	return nil
}

func driverNameFromDRADeviceClass(obj *unstructured.Unstructured) string {
	selectors, found, err := unstructured.NestedSlice(obj.Object, "spec", "selectors")
	if err != nil || !found {
		return ""
	}
	for _, selector := range selectors {
		selectorMap, ok := selector.(map[string]any)
		if !ok {
			continue
		}
		cel, ok := selectorMap["cel"].(map[string]any)
		if !ok {
			continue
		}
		expression, _ := cel["expression"].(string)
		driverName := driverNameFromCELExpression(expression)
		if driverName != "" {
			return driverName
		}
	}
	return ""
}

func driverNameFromCELExpression(expression string) string {
	const marker = "device.driver"
	idx := strings.Index(expression, marker)
	if idx < 0 {
		return ""
	}
	expression = expression[idx+len(marker):]
	eqIdx := strings.Index(expression, "==")
	if eqIdx < 0 {
		return ""
	}
	expression = strings.TrimSpace(expression[eqIdx+len("=="):])
	if len(expression) < 2 || expression[0] != '"' {
		return ""
	}
	expression = expression[1:]
	endIdx := strings.Index(expression, "\"")
	if endIdx < 0 {
		return ""
	}
	return expression[:endIdx]
}

func desiredDRADeviceClassNames(spec *v1.NPUClusterPolicySpec) map[string]bool {
	driverName := draDriverName(spec)
	names := map[string]bool{}
	if !spec.DRA.Managed || !spec.DRA.DeviceClasses.Managed {
		return names
	}
	for _, name := range []string{
		"full." + defaultDRADriverName,
		"elastic." + defaultDRADriverName,
		"fixed." + defaultDRADriverName,
		"best-effort." + defaultDRADriverName,
		"hard." + defaultDRADriverName,
	} {
		names[strings.ReplaceAll(name, defaultDRADriverName, driverName)] = true
	}
	for _, tmpl := range spec.DRA.DeviceClasses.Templates {
		if tmpl.Name != "" {
			names[tmpl.Name] = true
		}
	}
	return names
}

func hasOwnerReference(refs []metav1.OwnerReference, owner *v1.NPUClusterPolicy) bool {
	for _, ref := range refs {
		if ref.Kind == owner.Kind && ref.Name == owner.Name && ref.UID == owner.UID {
			return true
		}
	}
	return false
}

type daemonSetReconcileHooks struct {
	defaultResourceReconcileHooks[*appsv1.DaemonSet]
	transformers []componentResourceTransformer[appsv1.DaemonSet]
	inspectors   []componentResourceInspector[appsv1.DaemonSet]
}

func (h daemonSetReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.DaemonSet,
) error {
	if !r.hasNPUNodes {
		logger := log.FromContext(ctx)
		logger.V(1).Info("No supported NPU node in cluster, skip DaemonSet creation")
		return errSkipReconciliation
	}
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}

	// Set common DaemonSet configurations
	dsSpec := &r.instance.Spec.Daemonsets
	podTmpl := &obj.Spec.Template
	setCommonDaemonSetMetadata(&obj.ObjectMeta, dsSpec)
	setCommonDaemonSetMetadata(&podTmpl.ObjectMeta, dsSpec)
	if podTmpl.Spec.Tolerations == nil {
		podTmpl.Spec.Tolerations = make([]corev1.Toleration, 0)
	}
	if len(dsSpec.Tolerations) > 0 {
		podTmpl.Spec.Tolerations = append(podTmpl.Spec.Tolerations, dsSpec.Tolerations...)
	}
	if len(dsSpec.NodeSelector) > 0 {
		if podTmpl.Spec.NodeSelector == nil {
			podTmpl.Spec.NodeSelector = map[string]string{}
		}
		maps.Copy(podTmpl.Spec.NodeSelector, dsSpec.NodeSelector)
	}

	for _, tf := range h.transformers {
		if err := tf(ctx, r, obj); err != nil {
			return err
		}
	}

	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

func (h daemonSetReconcileHooks) inspect(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.DaemonSet,
) (*v1.ComponentState, error) {
	logger := log.FromContext(ctx)
	if err := r.Get(ctx, client.ObjectKeyFromObject(obj), obj); err != nil {
		return nil, fmt.Errorf("get DaemonSet: %w", err)
	}

	for _, inspect := range h.inspectors {
		if err := inspect(ctx, r, obj); err != nil {
			return nil, err
		}
	}

	if obj.Status.DesiredNumberScheduled == 0 {
		logger.V(1).Info("DaemonSet selects 0 node")
		return nil, nil
	}
	if obj.Status.NumberUnavailable != 0 {
		logger.V(1).Info(
			fmt.Sprintf("DaemonSet %s/%s not ready", obj.Namespace, obj.Name),
			"NumberUnavailable",
			obj.Status.NumberUnavailable,
		)
		return nil, fmt.Errorf("DaemonSet %s/%s not ready", obj.Namespace, obj.Name)
	}
	podList := &corev1.PodList{}
	if err := r.Client.List(ctx, podList, client.MatchingLabels(obj.Spec.Selector.MatchLabels)); err != nil {
		return nil, fmt.Errorf("Get pod list from daemonset failed: %w", err)
	}
	for _, pod := range podList.Items {
		if pod.Status.Phase != corev1.PodRunning {
			return &v1.ComponentState{
				Phase:   v1.ComponentPending,
				Reason:  "PodNotRunning",
				Message: fmt.Sprintf("Daemonset Pod %s ,state %s", pod.Name, pod.Status.Phase),
			}, nil
		}
	}

	return nil, nil
}

func setCommonDaemonSetMetadata(metadata *metav1.ObjectMeta, spec *v1.DaemonsetsSpec) {
	if metadata.Labels == nil {
		metadata.Labels = make(map[string]string)
	}
	if len(spec.Labels) > 0 {
		maps.Copy(metadata.Labels, spec.Labels)
	}

	if metadata.Annotations == nil {
		metadata.Annotations = make(map[string]string)
	}
	if len(spec.Annotations) > 0 {
		maps.Copy(metadata.Annotations, spec.Annotations)
	}
}

func newDaemonSetReconcileHooks(obj client.Object) daemonSetReconcileHooks {
	hooks := daemonSetReconcileHooks{}
	name := obj.GetName()
	tfs := &hooks.transformers // 这个的hook的transformers里存储的是不同的transformer函数，利用装饰器设计模式，进行不同功能的添加
	ins := &hooks.inspectors

	if strings.HasPrefix(name, runtimeDaemonSetNamePrefix) {
		*tfs = append(
			*tfs,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](runtimePlacementGetter),
			runtimeDaemonSetTransformer,
			newWorkloadImageTransformer[appsv1.DaemonSet](runtimeImageConfigGetter),
		)
		*ins = append(*ins, runtimeDaemonSetInspector)
	}
	switch name {
	case driverDaemonSetName:
		*tfs = append(
			*tfs,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](driverPlacementGetter),
			newWorkloadLogRotateTransformer[appsv1.DaemonSet](driverLogRotateConfigGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](driverImageConfigGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](driverEnvConfigGetter),
		)
	case devicePluginDaemonSetName:
		*tfs = append(
			*tfs,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](devicePluginPlacementGetter),
			newWorkloadLogRotateTransformer[appsv1.DaemonSet](devicePluginLogRotateConfigGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](devicePluginImageConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.DaemonSet](devicePluginCommandParametersGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.DaemonSet](devicePluginResourceRequirementsConfigGetter),
			newWorkloadVolumesTransformer[appsv1.DaemonSet](devicePluginVolumesGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](devicePluginEnvConfigGetter),
		)
	case nodedDaemonsetName:
		*tfs = append(
			*tfs,
			nodeDSnapshotTransformer,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](nodeDPlacementGetter),
			newWorkloadLogRotateTransformer[appsv1.DaemonSet](nodeDLogRotateConfigGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](nodeDImageConfigGetter),
			nodeDCommandTransformer,
			newWorkloadResourceRequirementsTransformer[appsv1.DaemonSet](nodeDResourceConfigGetter),
			newWorkloadVolumesTransformer[appsv1.DaemonSet](nodeDVolumesGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](nodeDEnvConfigGetter),
		)
	case rdmaDevicePluginDaemonSetName:
		*tfs = append(*tfs, componentWorkloadTransformers[appsv1.DaemonSet](
			rdmaDevicePluginWorkloadConfig, "k8s-rdma-shared-dp", rdmaDevicePluginDefaultHealthPort,
		)...)
	case exporterDaemonsetName:
		*tfs = append(
			*tfs,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](exporterPlacementGetter),
			newWorkloadLogRotateTransformer[appsv1.DaemonSet](exporterLogRotateConfigGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](exporterImageConfigGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.DaemonSet](exporterResourceRequirementsConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.DaemonSet](exporterCommandParametersGetter),
			newWorkloadVolumesTransformer[appsv1.DaemonSet](exporterVolumesGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](exporterEnvConfigGetter),
		)
	case mindIOTFTDaemonsetName:
		*tfs = append(
			*tfs,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](mindIOTFTPlacementGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](mindioTFTImageConfigGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](tftEnvConfigGetter),
		)
	case mindIOACPDaemonsetName:
		*tfs = append(
			*tfs,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](mindIOACPPlacementGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](mindioACPImageConfigGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](acpEnvConfigGetter),
		)
	case vnpuClientUpdateDaemonSetName:
		*tfs = append(
			*tfs,
			vnpuDaemonSetTransformer,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](vnpuClientUpdatePlacementGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](vnpuClientUpdateImageConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.DaemonSet](vnpuClientUpdateCommandParametersGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.DaemonSet](vnpuClientUpdateResourceConfigGetter),
			newWorkloadVolumesTransformer[appsv1.DaemonSet](vnpuClientUpdateVolumesGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](vnpuClientUpdateEnvConfigGetter),
		)
	case vnpuDevicePluginDaemonSetName:
		*tfs = append(
			*tfs,
			vnpuDaemonSetTransformer,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](vnpuDevicePluginPlacementGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](vnpuDevicePluginImageConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.DaemonSet](vnpuDevicePluginCommandParametersGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.DaemonSet](vnpuDevicePluginResourceConfigGetter),
			newWorkloadVolumesTransformer[appsv1.DaemonSet](vnpuDevicePluginVolumesGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](vnpuDevicePluginEnvConfigGetter),
		)
	case vnpuExporterDaemonSetName:
		*tfs = append(
			*tfs,
			vnpuDaemonSetTransformer,
			vnpuExporterDaemonSetTransformer,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](vnpuExporterPlacementGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](vnpuExporterImageConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.DaemonSet](vnpuExporterCommandParametersGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.DaemonSet](vnpuExporterResourceConfigGetter),
			newWorkloadVolumesTransformer[appsv1.DaemonSet](vnpuExporterVolumesGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](vnpuExporterEnvConfigGetter),
		)
	case draKubeletPluginDaemonSetName:
		*tfs = append(
			*tfs,
			draKubeletPluginDaemonSetTransformer,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](draKubeletPluginPlacementGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](draKubeletPluginImageConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.DaemonSet](draKubeletPluginCommandParametersGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.DaemonSet](draKubeletPluginResourceConfigGetter),
			newWorkloadVolumesTransformer[appsv1.DaemonSet](draKubeletPluginVolumesGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](draKubeletPluginEnvConfigGetter),
		)
	case draVCANNRTInstallerDaemonSetName:
		*tfs = append(
			*tfs,
			newWorkloadPlacementTransformer[appsv1.DaemonSet](draVCANNRTInstallerPlacementGetter),
			newWorkloadImageTransformer[appsv1.DaemonSet](draVCANNRTInstallerImageConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.DaemonSet](draVCANNRTInstallerCommandParametersGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.DaemonSet](draVCANNRTInstallerResourceConfigGetter),
			newWorkloadVolumesTransformer[appsv1.DaemonSet](draVCANNRTInstallerVolumesGetter),
			newWorkloadEnvTransformer[appsv1.DaemonSet](draVCANNRTInstallerEnvConfigGetter),
		)
	default:
	}
	*tfs = append(*tfs, workloadHealthProbeTransformer[appsv1.DaemonSet], workloadHashTransformer[appsv1.DaemonSet])

	return hooks
}

func runtimeDaemonSetTransformer(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.DaemonSet,
) error {
	logger := log.FromContext(ctx)
	runtime := strings.TrimPrefix(obj.Name, runtimeDaemonSetNamePrefix)
	if _, ok := r.runtimes[runtime]; !ok {
		logger.V(1).Info(fmt.Sprintf("No node with CRI runtime %s found in clustre", runtime))
		return errSkipReconciliation
	}

	selector := obj.Spec.Template.Spec.NodeSelector
	if selector == nil {
		selector = map[string]string{}
	}
	selector[runtimeLabelKey] = runtime
	obj.Spec.Template.Spec.NodeSelector = selector

	anno := obj.Spec.Template.Annotations
	if anno == nil {
		anno = map[string]string{}
	}
	anno[runtimeEndpointAnnotationKey] = ""
	obj.Spec.Template.Annotations = anno

	return nil
}

func runtimeDaemonSetInspector(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.DaemonSet,
) error {
	logger := log.FromContext(ctx)

	podList := &corev1.PodList{}
	if err := r.List(ctx, podList, client.InNamespace(r.namespace), client.MatchingLabels(obj.Spec.Selector.MatchLabels)); err != nil {
		return fmt.Errorf("list pods: %w", err)
	}

	for _, pod := range podList.Items {
		logger := logger.WithValues("pod", pod.Name)
		anno := pod.Annotations
		if anno == nil {
			anno = map[string]string{}
		}
		update := false
		node := pod.Spec.NodeName
		if endpoint, ok := r.nodeRuntimeEndpoints[node]; ok {
			logger.Info(fmt.Sprintf("CRI container runtime endpoint: %s", endpoint))
			update = addAnnotation(anno, runtimeEndpointAnnotationKey, endpoint)
		} else {
			logger.Info(fmt.Sprintf("Pod running on unrecognized node %s", node))
		}
		pod.Annotations = anno
		if update {
			if err := r.Update(ctx, &pod); err != nil {
				return fmt.Errorf("inject endpoint to pod: %w", err)
			}
		}
	}

	return nil
}

func vnpuDaemonSetTransformer(
	_ context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.DaemonSet,
) error {
	if r.instance.Spec.VNPU.UpdateStrategy != "" {
		obj.Spec.UpdateStrategy.Type = appsv1.DaemonSetUpdateStrategyType(r.instance.Spec.VNPU.UpdateStrategy)
	}
	if r.instance.Spec.VNPU.PriorityClassName != "" {
		obj.Spec.Template.Spec.PriorityClassName = r.instance.Spec.VNPU.PriorityClassName
	}
	return nil
}

func vnpuExporterDaemonSetTransformer(
	_ context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.DaemonSet,
) error {
	httpsValue := "off"
	if r.instance.Spec.VNPU.Exporter.HTTPS {
		httpsValue = "on"
	}
	upsertContainerEnv(&obj.Spec.Template.Spec, "HTTPS_ENABLE", httpsValue)
	return nil
}

func volcanoAdmissionDeploymentTransformer(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.Deployment,
) error {
	secret, err := volcanoAdmissionSecret(ctx, r)
	if err != nil {
		return err
	}
	logSpec := r.instance.Spec.Volcano.Admission.Log
	if logSpec.Dir != "" {
		setPodVolumePath(&obj.Spec.Template.Spec, "admission-config-log", logSpec.Dir)
	}
	for i := range obj.Spec.Template.Spec.Containers {
		args := obj.Spec.Template.Spec.Containers[i].Args
		if logSpec.Dir != "" {
			args = upsertArg(args, "--log_dir", logSpec.Dir)
		}
		if logSpec.File != "" {
			args = upsertArg(args, "--log_file", logSpec.File)
		}
		if logSpec.Level != nil {
			args = upsertArg(args, "-v", fmt.Sprintf("%d", *logSpec.Level))
		}
		obj.Spec.Template.Spec.Containers[i].Args = args
	}
	annotations := obj.Spec.Template.GetAnnotations()
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations[volcanoAdmissionSecretHashAnnotation] = volcanoAdmissionSecretHash(secret)
	obj.Spec.Template.SetAnnotations(annotations)
	return nil
}

func volcanoRunAsDeploymentTransformer(
	_ context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.Deployment,
) error {
	applyVolcanoRunAs(&obj.Spec.Template.Spec, &r.instance.Spec)
	return nil
}

func applyVolcanoRunAs(podSpec *corev1.PodSpec, spec *v1.NPUClusterPolicySpec) {
	runAsUser, runAsGroup := volcanoRunAs(spec)
	if runAsUser == nil && runAsGroup == nil {
		return
	}
	if podSpec.SecurityContext == nil {
		podSpec.SecurityContext = &corev1.PodSecurityContext{}
	}
	if runAsUser != nil {
		podSpec.SecurityContext.RunAsUser = runAsUser
		podSpec.SecurityContext.FSGroup = runAsUser
	}
	if runAsGroup != nil {
		podSpec.SecurityContext.RunAsGroup = runAsGroup
		podSpec.SecurityContext.FSGroup = runAsGroup
	}
	applyContainerRunAs(podSpec.Containers, runAsUser, runAsGroup)
	applyContainerRunAs(podSpec.InitContainers, runAsUser, runAsGroup)
}

func volcanoRunAs(spec *v1.NPUClusterPolicySpec) (*int64, *int64) {
	return spec.Volcano.RunAsUser, spec.Volcano.RunAsGroup
}

func applyContainerRunAs(containers []corev1.Container, runAsUser, runAsGroup *int64) {
	for i := range containers {
		if containers[i].SecurityContext == nil {
			containers[i].SecurityContext = &corev1.SecurityContext{}
		}
		if runAsUser != nil {
			containers[i].SecurityContext.RunAsUser = runAsUser
			if *runAsUser == 0 {
				runAsNonRoot := false
				containers[i].SecurityContext.RunAsNonRoot = &runAsNonRoot
			}
		}
		if runAsGroup != nil {
			containers[i].SecurityContext.RunAsGroup = runAsGroup
		}
	}
}

func applyImageSpecToPodSpec(podSpec *corev1.PodSpec, image *v1.ImageSpec) {
	applyImageSpecToPodSpecWithOverride(podSpec, image, false)
}

func applyImageSpecToPodSpecWithOverride(podSpec *corev1.PodSpec, image *v1.ImageSpec, override bool) {
	if image == nil {
		return
	}
	for i := range podSpec.Containers {
		applyImageSpecToContainer(&podSpec.Containers[i], image, override)
	}
	for i := range podSpec.InitContainers {
		applyImageSpecToContainer(&podSpec.InitContainers[i], image, override)
	}
}

func applyImageSpecToContainer(container *corev1.Container, image *v1.ImageSpec, override bool) {
	if image == nil {
		return
	}
	imageName := image.Name()
	if imageName == "" {
		return
	}
	if override || container.Image == placeholderValue {
		container.Image = imageName
		if image.ImagePullPolicy != "" {
			container.ImagePullPolicy = corev1.PullPolicy(image.ImagePullPolicy)
		}
	}
}

func applyReleaseTagToPodSpec(podSpec *corev1.PodSpec) {
	for i := range podSpec.Containers {
		podSpec.Containers[i].Image = replaceReleaseTagPlaceholder(podSpec.Containers[i].Image)
	}
	for i := range podSpec.InitContainers {
		podSpec.InitContainers[i].Image = replaceReleaseTagPlaceholder(podSpec.InitContainers[i].Image)
	}
}

func replaceReleaseTagPlaceholder(image string) string {
	if !strings.Contains(image, releaseTagPlaceholder) {
		return image
	}
	return strings.ReplaceAll(image, releaseTagPlaceholder, operatorReleaseTag())
}

func operatorReleaseTag() string {
	tag := os.Getenv(operatorReleaseTagEnvName)
	if tag == "" {
		return "latest"
	}
	return tag
}

func upsertArg(args []string, key, value string) []string {
	next := fmt.Sprintf("%s=%s", key, value)
	for i := range args {
		if args[i] == key || strings.HasPrefix(args[i], key+"=") {
			args[i] = next
			return args
		}
	}
	return append(args, next)
}

func draKubeletPluginDaemonSetTransformer(
	_ context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.DaemonSet,
) error {
	spec := &r.instance.Spec.DRA
	setPodVolumePath(&obj.Spec.Template.Spec, "kubelet-plugins-registry",
		stringOrDefault(spec.KubeletRegistrarDirectoryPath, defaultDRAKubeletRegistrarDir))
	setPodVolumePath(&obj.Spec.Template.Spec, "kubelet-plugins",
		stringOrDefault(spec.KubeletPluginsDirectoryPath, defaultDRAKubeletPluginsDir))
	setPodVolumePath(&obj.Spec.Template.Spec, "cdi", stringOrDefault(spec.CDIRoot, defaultDRACDIRoot))
	if spec.SoftVNPU.Managed {
		ensureDRASoftVNPUVolumes(&obj.Spec.Template.Spec, spec)
	}
	return nil
}

func draWebhookDeploymentTransformer(
	_ context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.Deployment,
) error {
	webhook := r.instance.Spec.DRA.Webhook
	port := int32OrDefault(webhook.Port, defaultDRAWebhookPort)
	for i := range obj.Spec.Template.Spec.Containers {
		container := &obj.Spec.Template.Spec.Containers[i]
		for j := range container.Ports {
			container.Ports[j].ContainerPort = port
		}
	}
	return nil
}

func upsertContainerEnv(podSpec *corev1.PodSpec, name, value string) {
	for i := range podSpec.Containers {
		replaced := false
		for j := range podSpec.Containers[i].Env {
			if podSpec.Containers[i].Env[j].Name != name {
				continue
			}
			podSpec.Containers[i].Env[j].Value = value
			podSpec.Containers[i].Env[j].ValueFrom = nil
			replaced = true
			break
		}
		if !replaced {
			podSpec.Containers[i].Env = append(podSpec.Containers[i].Env, corev1.EnvVar{Name: name, Value: value})
		}
	}
}

func setPodVolumePath(podSpec *corev1.PodSpec, name, path string) {
	if path == "" {
		return
	}
	setVolumeMountPath(podSpec, name, path)
	setHostPathVolumePath(podSpec, name, path)
}

func ensureDRASoftVNPUVolumes(podSpec *corev1.PodSpec, spec *v1.DRASpec) {
	softVNPU := &spec.SoftVNPU
	enpuPath := stringOrDefault(softVNPU.SoftShareDevConfigDir, defaultDRASoftVNPUShareDevConfigDir)
	sharedPath := stringOrDefault(softVNPU.SoftShareShmDir, defaultDRASoftVNPUShareShmDir)
	profilePath := stringOrDefault(softVNPU.NPUProfileConfigPath, defaultDRASoftVNPUNPUProfileConfigPath)
	directoryOrCreate := corev1.HostPathDirectoryOrCreate

	upsertPodVolume(podSpec, corev1.Volume{
		Name: "enpu-path",
		VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
			Path: enpuPath,
			Type: &directoryOrCreate,
		}},
	})
	upsertPodVolume(podSpec, corev1.Volume{
		Name: "shared",
		VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
			Path: sharedPath,
			Type: &directoryOrCreate,
		}},
	})
	upsertPodVolume(podSpec, corev1.Volume{
		Name: "dra-config",
		VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
			LocalObjectReference: corev1.LocalObjectReference{Name: draProfileConfigMapName},
			Items: []corev1.KeyToPath{{
				Key:  "dra.config",
				Path: "dra.config",
			}},
		}},
	})
	upsertContainerVolumeMounts(podSpec,
		corev1.VolumeMount{Name: "enpu-path", MountPath: enpuPath},
		corev1.VolumeMount{Name: "shared", MountPath: sharedPath},
		corev1.VolumeMount{Name: "dra-config", MountPath: linuxDir(profilePath)},
	)
}

func upsertPodVolume(podSpec *corev1.PodSpec, volume corev1.Volume) {
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == volume.Name {
			podSpec.Volumes[i] = volume
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, volume)
}

func upsertContainerVolumeMounts(podSpec *corev1.PodSpec, mounts ...corev1.VolumeMount) {
	for i := range podSpec.Containers {
		for _, mount := range mounts {
			upsertContainerVolumeMount(&podSpec.Containers[i], mount)
		}
	}
}

func upsertContainerVolumeMount(container *corev1.Container, mount corev1.VolumeMount) {
	for i := range container.VolumeMounts {
		if container.VolumeMounts[i].Name == mount.Name {
			container.VolumeMounts[i] = mount
			return
		}
	}
	container.VolumeMounts = append(container.VolumeMounts, mount)
}

func setHostPathVolumePath(podSpec *corev1.PodSpec, name, path string) {
	if path == "" {
		return
	}
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name != name || podSpec.Volumes[i].HostPath == nil {
			continue
		}
		podSpec.Volumes[i].HostPath.Path = path
	}
}

func setVolumeMountPath(podSpec *corev1.PodSpec, name, path string) {
	if path == "" {
		return
	}
	for i := range podSpec.Containers {
		for j := range podSpec.Containers[i].VolumeMounts {
			if podSpec.Containers[i].VolumeMounts[j].Name == name {
				podSpec.Containers[i].VolumeMounts[j].MountPath = path
			}
		}
	}
}

func linuxDir(p string) string {
	idx := strings.LastIndex(p, "/")
	if idx <= 0 {
		return "/"
	}
	return p[:idx]
}

func stringOrDefault(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func stringEnvOrDefault(name, fallback string) string {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	return value
}

func int32OrDefault(value, fallback int32) int32 {
	if value == 0 {
		return fallback
	}
	return value
}
