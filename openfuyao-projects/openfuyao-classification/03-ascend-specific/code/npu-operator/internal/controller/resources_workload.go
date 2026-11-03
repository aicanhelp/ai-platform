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
	"encoding/hex"
	"fmt"
	"hash/fnv"
	"maps"

	"github.com/davecgh/go-spew/spew"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "openfuyao.com/npu-operator/api/v1"
)

type deploymentReconcileHooks struct {
	defaultResourceReconcileHooks[*appsv1.Deployment]
	transformers []componentResourceTransformer[appsv1.Deployment]
}

func (h deploymentReconcileHooks) transform(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.Deployment,
) error {
	if obj.Namespace == placeholderValue {
		obj.Namespace = r.namespace
	}

	for _, tf := range h.transformers {
		if err := tf(ctx, r, obj); err != nil {
			return err
		}
	}

	return h.defaultResourceReconcileHooks.transform(ctx, r, obj)
}

func (deploymentReconcileHooks) inspect(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj *appsv1.Deployment,
) (*v1.ComponentState, error) {
	logger := log.FromContext(ctx)
	if err := r.Get(ctx, client.ObjectKeyFromObject(obj), obj); err != nil {
		return nil, fmt.Errorf("get Deployment: %w", err)
	}
	if obj.Status.UnavailableReplicas != 0 {
		logger.V(1).Info(
			fmt.Sprintf("Deployment %s/%s not ready", obj.Namespace, obj.Name),
			"NumberUnavailable",
			obj.Status.UnavailableReplicas,
		)
		return nil, fmt.Errorf("Deployment %s/%s not ready", obj.Namespace, obj.Name)
	}
	podList := &corev1.PodList{}
	if err := r.Client.List(ctx, podList, client.MatchingLabels(obj.Spec.Selector.MatchLabels)); err != nil {
		return nil, fmt.Errorf("Get pod list from deployment failed: %w", err)
	}

	for _, pod := range podList.Items {
		if pod.Status.Phase != corev1.PodRunning {
			return &v1.ComponentState{
				Phase:   v1.ComponentPending,
				Reason:  "PodNotRunning",
				Message: fmt.Sprintf("Deployment Pod %s ,state %s", pod.Name, pod.Status.Phase),
			}, nil
		}
	}
	return nil, nil
}

func newDeploymentReconcileHooks(obj client.Object) deploymentReconcileHooks {
	hooks := deploymentReconcileHooks{}
	hooks.transformers = deploymentTransformers(obj.GetName())
	hooks.transformers = append(hooks.transformers, workloadHealthProbeTransformer[appsv1.Deployment], workloadHashTransformer[appsv1.Deployment])
	return hooks
}

func deploymentTransformers(name string) []componentResourceTransformer[appsv1.Deployment] {
	if transformers := coreDeploymentTransformers(name); transformers != nil {
		return transformers
	}
	return volcanoDeploymentTransformers(name)
}

func coreDeploymentTransformers(name string) []componentResourceTransformer[appsv1.Deployment] {
	switch name {
	case inferOperatorDeploymentName:
		return componentWorkloadTransformers[appsv1.Deployment](
			inferOperatorWorkloadConfig, "infer-operator", inferOperatorDefaultHealthPort,
		)
	case trainerDeploymentName:
		return []componentResourceTransformer[appsv1.Deployment]{
			newWorkloadPlacementTransformer[appsv1.Deployment](trainerPlacementGetter),
			newWorkloadLogRotateTransformer[appsv1.Deployment](trainerLogRotateConfigGetter),
			newWorkloadImageTransformer[appsv1.Deployment](trainerImageConfigGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.Deployment](trainerResourceRequirementsConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.Deployment](trainerCommandParametersGetter),
			newWorkloadVolumesTransformer[appsv1.Deployment](trainerVolumesGetter),
			newWorkloadEnvTransformer[appsv1.Deployment](trainerEnvConfigGetter),
		}
	case clusterdDeploymentName:
		return []componentResourceTransformer[appsv1.Deployment]{
			newWorkloadPlacementTransformer[appsv1.Deployment](clusterdPlacementGetter),
			newWorkloadLogRotateTransformer[appsv1.Deployment](clusterdLogRotateConfigGetter),
			newWorkloadImageTransformer[appsv1.Deployment](clusterdImageConfigGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.Deployment](clusterdResourceConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.Deployment](clusterDCommandParametersGetter),
			newWorkloadVolumesTransformer[appsv1.Deployment](clusterDVolumesGetter),
			newWorkloadEnvTransformer[appsv1.Deployment](clusterDEnvConfigGetter),
		}
	case rsControllerDeploymentName:
		return []componentResourceTransformer[appsv1.Deployment]{
			newWorkloadPlacementTransformer[appsv1.Deployment](rsControllerPlacementGetter),
			newWorkloadLogRotateTransformer[appsv1.Deployment](rsControllerLogRotateConfigGetter),
			newWorkloadImageTransformer[appsv1.Deployment](rscontrollerImageConfigGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.Deployment](rsControllerResourceConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.Deployment](rsControllerCommandParametersGetter),
		}
	default:
		return nil
	}
}

func volcanoDeploymentTransformers(name string) []componentResourceTransformer[appsv1.Deployment] {
	switch name {
	case vcControllerDeploymentName:
		return []componentResourceTransformer[appsv1.Deployment]{
			newWorkloadPlacementTransformer[appsv1.Deployment](vcControllerPlacementGetter),
			volcanoRunAsDeploymentTransformer,
			newVolcanoWorkloadImageTransformer[appsv1.Deployment](vcControllerImageConfigGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.Deployment](vcControllerResourceRequirementsConfigGetter),
			newVolcanoWorkloadCommandParameterTransformer[appsv1.Deployment](vcControllerCommandParametersGetter),
			newWorkloadVolumesTransformer[appsv1.Deployment](vcControllerVolumesGetter),
			newWorkloadEnvTransformer[appsv1.Deployment](vcControllerEnvConfigGetter),
		}
	case vcSchedulerDeploymentName:
		return []componentResourceTransformer[appsv1.Deployment]{
			newWorkloadPlacementTransformer[appsv1.Deployment](vcSchedulerPlacementGetter),
			volcanoRunAsDeploymentTransformer,
			newVolcanoWorkloadImageTransformer[appsv1.Deployment](vcSchedulerImageConfigGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.Deployment](vcSchedulerResourceRequirementsConfigGetter),
			newVolcanoWorkloadCommandParameterTransformer[appsv1.Deployment](schedulerCommandParametersGetter),
			newWorkloadVolumesTransformer[appsv1.Deployment](vcSchedulerVolumesGetter),
			newWorkloadEnvTransformer[appsv1.Deployment](vcSchedulerEnvConfigGetter),
		}
	case volcanoAdmissionDeploymentName:
		return []componentResourceTransformer[appsv1.Deployment]{
			volcanoAdmissionDeploymentTransformer,
			volcanoRunAsDeploymentTransformer,
			newVolcanoWorkloadImageTransformer[appsv1.Deployment](volcanoAdmissionImageConfigGetter),
		}
	case draWebhookDeploymentName:
		return []componentResourceTransformer[appsv1.Deployment]{
			newWorkloadPlacementTransformer[appsv1.Deployment](draWebhookPlacementGetter),
			newWorkloadImageTransformer[appsv1.Deployment](draWebhookImageConfigGetter),
			newWorkloadCommandParameterTransformer[appsv1.Deployment](draWebhookCommandParametersGetter),
			newWorkloadResourceRequirementsTransformer[appsv1.Deployment](draWebhookResourceConfigGetter),
			newWorkloadVolumesTransformer[appsv1.Deployment](draWebhookVolumesGetter),
			newWorkloadEnvTransformer[appsv1.Deployment](draWebhookEnvConfigGetter),
			draWebhookDeploymentTransformer,
		}
	default:
		return nil
	}
}

// NodeD command overrides apply equally to normal and snapshot mode.
func nodeDCommandTransformer(_ context.Context, r *NPUClusterPolicyReconciler, obj *appsv1.DaemonSet) error {
	conf := nodeDCommandParametersGetter(&r.instance.Spec)
	for i := range obj.Spec.Template.Spec.Containers {
		container := &obj.Spec.Template.Spec.Containers[i]
		if len(conf.Command) > 0 {
			container.Command = append([]string(nil), conf.Command...)
		}
		container.Args = append([]string(nil), conf.Args...)
	}
	return nil
}

func newWorkloadPodTemplateSpecGetter[T appsv1.DaemonSet | appsv1.Deployment]() func(*T) *corev1.PodTemplateSpec {
	var o T
	switch any(o).(type) {
	case appsv1.DaemonSet:
		return func(obj *T) *corev1.PodTemplateSpec { return &any(obj).(*appsv1.DaemonSet).Spec.Template }
	case appsv1.Deployment:
		return func(obj *T) *corev1.PodTemplateSpec { return &any(obj).(*appsv1.Deployment).Spec.Template }
	default:
		return nil
	}
}

func newWorkloadPlacementTransformer[T appsv1.DaemonSet | appsv1.Deployment](
	placementGetter func(*v1.NPUClusterPolicySpec) *v1.PlacementSpec,
) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]()
	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		placement := placementGetter(&r.instance.Spec)
		if placement == nil {
			return nil
		}
		podTmpl := podTemplateSpec(obj)
		if len(placement.NodeSelector) > 0 {
			if podTmpl.Spec.NodeSelector == nil {
				podTmpl.Spec.NodeSelector = map[string]string{}
			}
			maps.Copy(podTmpl.Spec.NodeSelector, placement.NodeSelector)
		}
		if len(placement.Tolerations) > 0 {
			podTmpl.Spec.Tolerations = append(podTmpl.Spec.Tolerations, placement.Tolerations...)
		}
		return nil
	}
}

// Hash workload resource to avoid unnecessary update
func workloadHashTransformer[T appsv1.DaemonSet | appsv1.Deployment, PT clientObject[T]](
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	obj PT,
) error {
	hasher := fnv.New32a()
	spewer := spew.ConfigState{
		DisableMethods:          true,
		DisablePointerAddresses: true,
		SortKeys:                true,
		SpewKeys:                true,
	}
	spewer.Fprintf(hasher, "%#v", obj)
	anno := obj.GetAnnotations()
	if anno == nil {
		anno = make(map[string]string)
		obj.SetAnnotations(anno)
	}
	obj.GetAnnotations()[hashAnnotationKey] = hex.EncodeToString(hasher.Sum(nil))
	return nil
}

func newWorkloadEnvTransformer[T appsv1.DaemonSet | appsv1.Deployment](
	obtainEnv func(*v1.NPUClusterPolicySpec) []corev1.EnvVar,
) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]()
	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		specEnv := obtainEnv(&r.instance.Spec)
		podTmpl := podTemplateSpec(obj)
		injectEnv := func(containers []corev1.Container) []corev1.Container {
			for i := range containers {
				env := containers[i].Env
				if env == nil {
					env = make([]corev1.EnvVar, 0)
				}
				for j := range specEnv {
					replaced := false
					for k := range env {
						if env[k].Name != specEnv[j].Name {
							continue
						}
						env[k] = specEnv[j]
						replaced = true
						break
					}
					if !replaced {
						env = append(env, specEnv[j])
					}
				}
				containers[i].Env = env
			}
			return containers
		}
		podTmpl.Spec.Containers = injectEnv(podTmpl.Spec.Containers)
		podTmpl.Spec.InitContainers = injectEnv(podTmpl.Spec.InitContainers)
		return nil
	}
}

func newWorkloadLogRotateTransformer[T appsv1.DaemonSet | appsv1.Deployment]( // 通过环境变量的方式修改log的相关参数
	logRotate func(*v1.NPUClusterPolicySpec) *v1.LogRotate,
) componentResourceTransformer[T] {
	podTemplateSpec := newWorkloadPodTemplateSpecGetter[T]()
	return func(ctx context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		conf := logRotate(&r.instance.Spec)
		podTmpl := podTemplateSpec(obj)
		containers := podTmpl.Spec.Containers
		for i := range containers {
			env := containers[i].Env
			for j := range env {
				switch env[j].Name {
				case logRotateLogLevelEnvName:
					env[j].Value = fmt.Sprintf("%d", conf.LogLevel.Value())
				case logRotateLogFileEnvName:
					env[j].Value = conf.LogFile
				case logRotateRotateEnvName:
					env[j].Value = fmt.Sprintf("%d", conf.Rotate)
				case logRotateMaxAgeEnvName:
					env[j].Value = fmt.Sprintf("%d", conf.MaxAge)
				case logRotateCompressEnvName:
					env[j].Value = fmt.Sprintf("%t", conf.Compress)
				}
			}
		}
		return nil
	}
}

// log rotate obtain
