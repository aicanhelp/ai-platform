/**
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
**/

package controller

import (
	"context"
	"fmt"
	"hash/fnv"
	"sort"
	"strconv"
	"strings"

	"github.com/davecgh/go-spew/spew"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/constants"
	"gopkg.openfuyao.cn/kae-operator/internal/utils"
	"gopkg.openfuyao.cn/kae-operator/internal/zlog"
)

type object interface {
	metav1.Object
	runtime.Object
	GetName() string
	GetNamespace() string
	SetName(string)
	SetNamespace(string)
}

func setObjReferenceToManager(manager KAEPolicyManager, obj client.Object) error {
	return controllerutil.SetControllerReference(
		manager.kaeInstance,
		obj,
		manager.scheme,
	)
}

func createOrUpdateObject(manager KAEPolicyManager, obj object) (kaev1.State, error) {
	obj.SetNamespace(manager.namespace)
	// 先尝试创建，若存在则更新。client 进行一或两次请求，效率更高
	if err := setObjReferenceToManager(manager, obj); err != nil {
		return kaev1.NotReady, err
	}

	err := manager.client.Create(manager.ctx, obj)
	if err == nil {
		return kaev1.Ready, nil
	}

	if errors.IsAlreadyExists(err) {
		zlog.Info("Found Resource and trying to update")
		if err = manager.client.Update(manager.ctx, obj); err == nil {
			zlog.Info("Update Resource successfully")
			return kaev1.Ready, err
		}
	}
	// 其他错误类型
	zlog.Errorf("Failed to create or update resource: %v", err)
	return kaev1.NotReady, err
}

func deleteObject(manager KAEPolicyManager, obj client.Object) (kaev1.State, error) {
	err := manager.client.Delete(manager.ctx, obj)
	if err != nil && !errors.IsNotFound(err) {
		zlog.Errorf("Failed to delete resource: %v", err)
		return kaev1.NotReady, err
	}
	return kaev1.Disabled, nil
}

func handleObject(manager KAEPolicyManager, obj client.Object) (kaev1.State, error) {
	component := manager.componentNames[manager.index]
	if !manager.isComponentEnabled(component) {
		return deleteObject(manager, obj)
	}
	return createOrUpdateObject(manager, obj)
}

const (
	parseBase    = 10
	parseBitSize = 32
)

type resourceHandler []func(manager KAEPolicyManager) (kaev1.State, error)

const (
	annotationKeyLastAppliedHash = "openfuyao.com/last-applied-hash"
	labelKeyCtrlRevHash          = "controller-revision-hash"
)

// HandlerServiceAccount 处理 ServiceAccount 资源变更
func HandlerServiceAccount(manager KAEPolicyManager) (kaev1.State, error) {
	obj := manager.resources[manager.index].ServiceAccount.DeepCopy()
	obj.Namespace = manager.namespace
	zlog.Infof("Handling Resource: ServiceAccount [%s]", obj.GetName())
	return handleObject(manager, obj)
}

// HandlerRole 处理 Role 资源变更
func HandlerRole(manager KAEPolicyManager) (kaev1.State, error) {
	obj := manager.resources[manager.index].Role.DeepCopy()
	obj.Namespace = manager.namespace
	zlog.Infof("Handling Resource: Role [%s]", obj.GetName())
	return handleObject(manager, obj)
}

// HandlerRoleBinding 处理 RoleBinding 资源变更
func HandlerRoleBinding(manager KAEPolicyManager) (kaev1.State, error) {
	obj := manager.resources[manager.index].RoleBinding.DeepCopy()
	obj.Namespace = manager.namespace
	for _, subject := range obj.Subjects {
		// 如果资源配置了命名空间，此处就不再修改了
		if subject.Namespace == "FILLED BY THE OPERATOR" {
			subject.Namespace = manager.namespace
		}
	}

	zlog.Infof("Handling Resource: RoleBinding [%s]", obj.GetName())
	return handleObject(manager, obj)
}

// HandlerClusterRole 处理 ClusterRole 资源变更
func HandlerClusterRole(manager KAEPolicyManager) (kaev1.State, error) {
	obj := manager.resources[manager.index].ClusterRole.DeepCopy()
	obj.Namespace = manager.namespace
	zlog.Infof("Handling Resource: ClusterRole [%s]", obj.GetName())
	return handleObject(manager, obj)
}

// HandlerClusterRoleBinding 处理 ClusterRoleBinding 资源变更
func HandlerClusterRoleBinding(manager KAEPolicyManager) (kaev1.State, error) {
	obj := manager.resources[manager.index].ClusterRoleBinding.DeepCopy()
	obj.Namespace = manager.namespace
	// 同时也设置 ClusterRoleBinding 的 Subject 的命名空间
	for idx := range obj.Subjects {
		obj.Subjects[idx].Namespace = manager.namespace
	}
	zlog.Infof("Handling Resource: ClusterRoleBinding [%s]", obj.GetName())
	return handleObject(manager, obj)
}

// HandlerConfigMaps 处理一系列 ConfigMap 资源变更
func HandlerConfigMaps(manager KAEPolicyManager) (kaev1.State, error) {
	status := kaev1.Ready
	for i := range manager.resources[manager.index].ConfigMaps {
		stat, err := handlerConfigMap(manager, i)
		if err != nil {
			return stat, err
		}
		if stat != kaev1.Ready {
			status = kaev1.NotReady
		}
	}
	return status, nil
}

// HandlerConfigMap 处理单一 ConfigMap 资源变更
func handlerConfigMap(manager KAEPolicyManager, configMapIdx int) (kaev1.State, error) {
	obj := manager.resources[manager.index].ConfigMaps[configMapIdx].DeepCopy()
	obj.Namespace = manager.namespace
	zlog.Infof("createConfigMap: %s, Namespace: %s", obj.GetName(), manager.namespace)
	return handleObject(manager, obj)
}

// HandlerService 处理 Service 资源变更
func HandlerService(manager KAEPolicyManager) (kaev1.State, error) {
	obj := manager.resources[manager.index].Service.DeepCopy()
	obj.Namespace = manager.namespace
	zlog.Infof("Handling Resource: %s, Service: %s", obj.GetName(), manager.namespace)
	if !manager.isComponentEnabled(manager.componentNames[manager.index]) {
		return deleteObject(manager, obj)
	}
	return getAndUpdateServiceOrCreate(manager, obj)
}

// getAndUpdateServiceOrCreate 专门用于 Service 的更新
// Service cluster IP 等字段不可变无法直接更新,需要先尝试获取，若存在则更新（不改变原有Cluster IP），不存在则创建
func getAndUpdateServiceOrCreate(manager KAEPolicyManager, svc *corev1.Service) (kaev1.State, error) {
	svc.SetNamespace(manager.namespace)

	if err := setObjReferenceToManager(manager, svc); err != nil {
		return kaev1.NotReady, err
	}

	// 检验 Service 是否存在，不存在则创建
	found := &corev1.Service{}
	err := manager.client.Get(
		manager.ctx,
		types.NamespacedName{Namespace: svc.Namespace, Name: svc.Name},
		found,
	)
	if err != nil {
		if !errors.IsNotFound(err) {
			return kaev1.NotReady, err
		}
		zlog.Info("Service not found, creating...")
		err = manager.client.Create(manager.ctx, svc)
		if err != nil {
			zlog.Errorf("failed to create: %v", err)
			return kaev1.NotReady, err
		}
		return kaev1.Ready, nil
	}

	// Service 存在则更新，更新时不能更改 ClusterIP
	svc.Spec.ClusterIP = found.Spec.ClusterIP
	svc.ResourceVersion = found.ResourceVersion
	zlog.Info("Found Service, updating...")
	err = manager.client.Update(manager.ctx, svc)
	if err != nil {
		zlog.Errorf("Failed to create or update resource: %v", err)
		return kaev1.NotReady, err
	}
	return kaev1.Ready, nil
}

// HandlerDeployment creates Deployment resource
func HandlerDeployment(manager KAEPolicyManager) (kaev1.State, error) {
	obj := manager.resources[manager.index].Deployment.DeepCopy()
	obj.Namespace = manager.namespace
	zlog.Infof("Handling Resource: Deployment: %s", obj.GetName())
	if state, err := handleObject(manager, obj); err != nil {
		return state, err
	}
	return isDeploymentReady(manager, obj)
}

func isDeploymentReady(manager KAEPolicyManager, obj *appsv1.Deployment) (kaev1.State, error) {
	found := &appsv1.Deployment{}
	err := manager.client.Get(
		manager.ctx,
		types.NamespacedName{Namespace: manager.namespace, Name: obj.GetName()},
		found,
	)
	if err != nil {
		zlog.Errorf("fail to get Deployment: %s", obj.GetName())
		return kaev1.NotReady, err
	}
	if found.Status.UnavailableReplicas != 0 {
		zlog.Warnf("deployment has unavailable replicas: %d != 0", found.Status.UnavailableReplicas)
		return kaev1.NotReady, nil
	}
	return isDeploymentPodReady(manager, found)
}

func isDeploymentPodReady(manager KAEPolicyManager, deployment *appsv1.Deployment) (kaev1.State, error) {
	found := &corev1.PodList{}
	opt := []client.ListOption{client.MatchingLabels(deployment.Spec.Template.ObjectMeta.Labels)}
	err := manager.client.List(manager.ctx, found, opt...)
	if err != nil {
		zlog.Errorf("Could not get PodList, err: %v", err)
		return kaev1.NotReady, err
	}
	if len(found.Items) == 0 {
		zlog.Warnf("No pod found")
		return kaev1.NotReady, nil
	}
	zlog.Infof("Found number of pods: %d", len(found.Items))
	state := kaev1.Ready
	for _, pod := range found.Items {
		if pod.Status.Phase != corev1.PodRunning {
			zlog.Warnf("Replica Pod %s is not running, Phase: %s", pod.Name, pod.Status.Phase)
			state = kaev1.NotReady
		}
	}
	return state, nil
}

// HandlerDaemonSet creates DaemonSet resource
func HandlerDaemonSet(manager KAEPolicyManager) (kaev1.State, error) {
	obj := manager.resources[manager.index].DaemonSet.DeepCopy()
	obj.Namespace = manager.namespace
	zlog.Infof("DaemonSet: %s, Namespace: %s", obj.GetName(), manager.namespace)
	if !manager.isComponentEnabled(manager.componentNames[manager.index]) {
		return deleteObject(manager, obj)
	}
	if noNeedDeploy(manager) {
		return kaev1.Ready, nil
	}
	if stat, err := createOrUpdateDaemonSet(manager, obj); err != nil {
		return stat, err
	}
	return isDaemonSetReady(obj.GetName(), manager)
}

func createOrUpdateDaemonSet(manager KAEPolicyManager, obj *appsv1.DaemonSet) (kaev1.State, error) {
	if err := preProcessDaemonSet(obj, manager); err != nil {
		zlog.Errorf("Could not pre-process, err: %v", err)
		return kaev1.NotReady, err
	}
	if err := setObjReferenceToManager(manager, obj); err != nil {
		zlog.Errorf("SetControllerReference failed, err: %v", err)
		return kaev1.NotReady, err
	}

	addDaemonsetAnnotations(obj, manager.kaeInstance.Spec.DaemonSets.Annotations)
	addDaemonsetLabels(obj, manager.kaeInstance.Spec.DaemonSets.Labels)

	found := &appsv1.DaemonSet{}
	err := manager.client.Get(
		manager.ctx,
		types.NamespacedName{Namespace: obj.GetNamespace(), Name: obj.GetName()},
		found,
	)
	// Daemonset 不存在，尝试创建
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.Info("DaemonSet not found, trying to create")
			return createDaemonSet(manager, obj)
		} else {
			zlog.Errorf("failed to get DaemonSet : %v", err)
			return kaev1.NotReady, err
		}
	}

	// Daemonset 存在，且 spec 发生变化，尝试更新
	if isDaemonSetSpecChanged(found, obj) {
		zlog.Info("DaemonSet spec changed, trying to update")
		if err = manager.client.Update(manager.ctx, obj); err != nil {
			zlog.Errorf("failed to update DaemonSet : %v", err)
			return kaev1.NotReady, err
		}
	} else {
		zlog.Info("DaemonSet did not change, skipping update")
	}
	return kaev1.Ready, nil
}

func addDaemonsetLabels(obj *appsv1.DaemonSet, labels map[string]string) {
	// 添加 CR 中 DaemonSet 公共 labels
	if obj.Labels == nil {
		obj.Labels = make(map[string]string)
	}
	for k, v := range labels {
		obj.Labels[k] = v
	}
}

// 添加 CR 中 DaemonSet 公共 Annotations
func addDaemonsetAnnotations(obj *appsv1.DaemonSet, annotations map[string]string) {
	if obj.Annotations == nil {
		obj.Annotations = make(map[string]string)
	}
	for k, v := range annotations {
		obj.Annotations[k] = v
	}
}

func createDaemonSet(manager KAEPolicyManager, obj *appsv1.DaemonSet) (kaev1.State, error) {
	// 生成 hash 值
	hashStr := getDaemonSetHash(obj)
	// 给 DaemonSet 添加注解，注解值为上面的 hash 值
	obj.Annotations[annotationKeyLastAppliedHash] = hashStr
	err := manager.client.Create(manager.ctx, obj)
	if err != nil {
		zlog.Errorf("Couldn't create DaemonSet, Name: %s, err: %v", obj.GetName(), err)
		return kaev1.NotReady, err
	}
	return isDaemonSetReady(obj.GetName(), manager)
}

func noNeedDeploy(manager KAEPolicyManager) bool {
	if !manager.hasKAENodes {
		// 在没有设备的情况下，无法部署多个daemonSet，因此现在跳过它们的部署。当新节点加入集群时，将通知操作符(watchKAENodeLabels)。
		zlog.Info("No KAE node in the cluster, do not create DaemonSets")
		return true
	}

	if !utils.IsOperatorEnabledHPRE(manager.kaeInstance.Spec.Operator) &&
		manager.resources[manager.index].DaemonSet.GetName() == constants.HpreDevicePluginDaemonSetName {
		return true
	}

	if !utils.IsOperatorEnabledSEC(manager.kaeInstance.Spec.Operator) &&
		manager.resources[manager.index].DaemonSet.GetName() == constants.SecDevicePluginDaemonSetName {
		return true
	}

	if !utils.IsOperatorEnabledZIP(manager.kaeInstance.Spec.Operator) &&
		manager.resources[manager.index].DaemonSet.GetName() == constants.ZipDevicePluginDaemonSetName {
		return true
	}
	return false
}

func preProcessDaemonSet(obj *appsv1.DaemonSet, manager KAEPolicyManager) error {
	var transform func(*appsv1.DaemonSet, *kaev1.KAEPolicySpec) error
	switch obj.GetName() {
	case constants.DriverDaemonSetName:
		transform = transformDriver
	case constants.HpreDevicePluginDaemonSetName,
		constants.SecDevicePluginDaemonSetName,
		constants.ZipDevicePluginDaemonSetName:
		transform = transformDevicePlugin
	default:
		err := fmt.Errorf("common DaemonSet transformation doesnot apply to resource: %s", obj.GetName())
		return err
	}

	err := applyCommonDaemonSetConfig(obj, &manager.kaeInstance.Spec.DaemonSets)
	if err != nil {
		zlog.Errorf("Failed to apply common DaemonSet transformation, resource: %s, err: %v", obj.GetName(), err)
		return err
	}
	// apply per operand DaemonSet config
	err = transform(obj, &manager.kaeInstance.Spec)
	if err != nil {
		zlog.Errorf("Failed to apply transformation, resource: %s, err: %v", obj.GetName(), err)
		return err
	}

	// apply custom Labels and Annotations to the podSpec if any
	applyCommonDaemonSetMetadata(obj, &manager.kaeInstance.Spec.DaemonSets)

	return nil
}

func transformDriver(obj *appsv1.DaemonSet, config *kaev1.KAEPolicySpec) error {
	// 更新普通容器
	if err := handlerDriverContainer(&obj.Spec.Template.Spec.Containers, config); err != nil {
		return err
	}
	if len(config.Driver.ImagePullSecrets) > 0 {
		addPullSecrets(&obj.Spec.Template.Spec, config.Driver.ImagePullSecrets)
	}
	// 设置普通容器清理钩子
	if utils.IsDriverEnabledCleanup(config.Driver) {
		obj.Spec.Template.Spec.Containers[0].Lifecycle = &corev1.Lifecycle{
			PreStop: &corev1.LifecycleHandler{
				Exec: &corev1.ExecAction{
					Command: []string{"/bin/sh", "-c", "sh /home/kae-driver/kae-driver-entrypoint.sh -o uninstall"},
				},
			},
		}
	} else {
		obj.Spec.Template.Spec.Containers[0].Lifecycle = &corev1.Lifecycle{}
	}
	// 更新初始化容器
	if err := handlerDriverContainer(&obj.Spec.Template.Spec.InitContainers, config); err != nil {
		return err
	}
	return nil
}

func handlerDriverContainer(containers *([]corev1.Container), config *kaev1.KAEPolicySpec) error {
	kaeDriver := config.Driver
	// 设置镜像
	image, err := utils.Image(&kaeDriver)
	if err != nil {
		return err
	}
	(*containers)[0].Image = image
	(*containers)[0].ImagePullPolicy = utils.ImagePullPolicy(kaeDriver.ImagePullPolicy)

	// 设置资源限制
	if kaeDriver.Resources != nil {
		for i := range *containers {
			(*containers)[i].Resources = corev1.ResourceRequirements{
				Requests: kaeDriver.Resources.Requests,
				Limits:   kaeDriver.Resources.Limits,
			}
		}
	}
	// 设置参数和环境变量
	if len(kaeDriver.Args) > 0 {
		(*containers)[0].Args = kaeDriver.Args
	}
	if len(kaeDriver.Env) > 0 {
		setContainerEnv(&(*containers)[0], kaeDriver.Env)
	}
	return nil
}

func transformDevicePlugin(obj *appsv1.DaemonSet, config *kaev1.KAEPolicySpec) error {
	if err := handlerDevicePluginContainer(&obj.Spec.Template.Spec.Containers, config); err != nil {
		return err
	}
	if len(config.DevicePlugin.ImagePullSecrets) > 0 {
		addPullSecrets(&obj.Spec.Template.Spec, config.DevicePlugin.ImagePullSecrets)
	}
	return nil
}

func handlerDevicePluginContainer(containers *([]corev1.Container), config *kaev1.KAEPolicySpec) error {
	kaeDevicePlugin := config.DevicePlugin
	// 设置镜像
	image, err := utils.Image(&kaeDevicePlugin)
	if err != nil {
		return err
	}
	(*containers)[0].Image = image
	(*containers)[0].ImagePullPolicy = utils.ImagePullPolicy(kaeDevicePlugin.ImagePullPolicy)

	// 设置资源限制
	if kaeDevicePlugin.Resources != nil {
		for i := range *containers {
			(*containers)[i].Resources = corev1.ResourceRequirements{
				Requests: kaeDevicePlugin.Resources.Requests,
				Limits:   kaeDevicePlugin.Resources.Limits,
			}
		}
	}
	// 设置参数和环境变量
	if len(kaeDevicePlugin.Args) > 0 {
		(*containers)[0].Args = kaeDevicePlugin.Args
	}
	if len(kaeDevicePlugin.Env) > 0 {
		setContainerEnv(&(*containers)[0], kaeDevicePlugin.Env)
	}
	return nil
}

func addPullSecrets(podSpec *corev1.PodSpec, secrets []string) {
	existingSecrets := make(map[string]string)
	for _, s := range podSpec.ImagePullSecrets {
		existingSecrets[s.Name] = ""
	}

	for _, secretName := range secrets {
		if _, found := existingSecrets[secretName]; !found {
			podSpec.ImagePullSecrets = append(podSpec.ImagePullSecrets, corev1.LocalObjectReference{Name: secretName})
			existingSecrets[secretName] = ""
		}
	}
}

func setContainerEnv(container *corev1.Container, newEnvs []kaev1.EnvVar) {
	for _, newEnv := range newEnvs {
		found := false
		for i, existingEnv := range container.Env {
			if existingEnv.Name == newEnv.Name {
				container.Env[i].Value = newEnv.Value
				found = true
				break
			}
		}
		if !found {
			container.Env = append(container.Env, corev1.EnvVar{Name: newEnv.Name, Value: newEnv.Value})
		}
	}
}

// Apply common config that is applicable for all DaemonSets
func applyCommonDaemonSetConfig(obj *appsv1.DaemonSet, daemonset *kaev1.DaemonSetsSpec) error {
	// 应用daemonSet更新策略
	err := applyUpdateStrategyConfig(obj, daemonset)
	if err != nil {
		return err
	}
	// 更新priorityclass
	if len(daemonset.PriorityClassName) > 0 {
		obj.Spec.Template.Spec.PriorityClassName = daemonset.PriorityClassName
	}
	// 更新容忍度
	obj.Spec.Template.Spec.Tolerations = append(obj.Spec.Template.Spec.Tolerations, daemonset.Tolerations...)
	return nil
}

func applyUpdateStrategyConfig(obj *appsv1.DaemonSet, daemonset *kaev1.DaemonSetsSpec) error {
	switch daemonset.UpdateStrategy {
	case "OnDelete":
		// 更新Update策略
		obj.Spec.UpdateStrategy = appsv1.DaemonSetUpdateStrategy{Type: appsv1.OnDeleteDaemonSetStrategyType}
	case "RollingUpdate":
		// 更新rollingupdate策略配置
		if daemonset.RollingUpdate == nil || daemonset.RollingUpdate.MaxUnavailable == "" {
			return nil
		}
		if strings.HasPrefix(obj.GetName(), constants.DriverDaemonSetName) {
			return nil
		}
		intOrString, err := createIntOrStringFromUnavailable(daemonset.RollingUpdate.MaxUnavailable)
		if err != nil {
			return fmt.Errorf("failed to apply rolling update config: %w", err)
		}
		obj.Spec.UpdateStrategy = appsv1.DaemonSetUpdateStrategy{
			Type: appsv1.RollingUpdateDaemonSetStrategyType,
			RollingUpdate: &appsv1.RollingUpdateDaemonSet{
				MaxUnavailable: &intOrString,
			},
		}
	default:
		return fmt.Errorf("unsupported update strategy: %s", daemonset.UpdateStrategy)
	}
	return nil
}

func createIntOrStringFromUnavailable(maxUnavailable string) (intstr.IntOrString, error) {
	if strings.HasSuffix(maxUnavailable, "%") {
		return intstr.IntOrString{
			Type:   intstr.String,
			StrVal: maxUnavailable,
		}, nil
	} else {
		valInt64, err := strconv.ParseInt(maxUnavailable, parseBase, parseBitSize)
		if err != nil {
			return intstr.IntOrString{}, fmt.Errorf("failed to parse MaxUnavailable '%s': %w", maxUnavailable, err)
		}
		return intstr.IntOrString{
			Type:   intstr.Int,
			IntVal: int32(valInt64),
		}, nil
	}
}

// applyCommonDaemonSetMetadata 在 daemonSet podSpec 中添加额外的标签和注释，如果用户在podSpec中指定了任何标签和注释
func applyCommonDaemonSetMetadata(obj *appsv1.DaemonSet, dsSpec *kaev1.DaemonSetsSpec) {
	if len(dsSpec.Labels) > 0 {
		if obj.Spec.Template.ObjectMeta.Labels == nil {
			obj.Spec.Template.ObjectMeta.Labels = make(map[string]string)
		}
		for labelKey, labelValue := range dsSpec.Labels {
			// 由于 Pod 标签不可变，这里跳过 DaemonSet "app"或"app.kubernetes.io/part-of" 标签的覆盖，使得 DaemonSet 创建的 od 仍然可以选择
			if labelKey == "app" || labelKey == "app.kubernetes.io/part-of" {
				continue
			}
			obj.Spec.Template.ObjectMeta.Labels[labelKey] = labelValue
		}
	}

	if len(dsSpec.Annotations) > 0 {
		if obj.Spec.Template.ObjectMeta.Annotations == nil {
			obj.Spec.Template.ObjectMeta.Annotations = make(map[string]string)
		}
		for annoKey, annoVal := range dsSpec.Annotations {
			obj.Spec.Template.ObjectMeta.Annotations[annoKey] = annoVal
		}
	}
}

func getDaemonSetHash(daemonSet *appsv1.DaemonSet) string {
	hashes := fnv.New32a()
	printer := spew.ConfigState{
		Indent:         " ",
		SortKeys:       true,
		DisableMethods: true,
		SpewKeys:       true,
	}
	_, err := printer.Fprintf(hashes, "%#v", daemonSet)
	if err != nil {
		return ""
	}
	return fmt.Sprint(hashes.Sum32())
}

func isDaemonSetReady(name string, manager KAEPolicyManager) (kaev1.State, error) {
	foundDS := &appsv1.DaemonSet{}
	err := manager.client.Get(manager.ctx, types.NamespacedName{Namespace: manager.namespace, Name: name}, foundDS)
	if err != nil {
		zlog.Errorf("fail to get daemonSet: %v", err)
		return kaev1.NotReady, err
	}
	if foundDS.Status.DesiredNumberScheduled == 0 {
		zlog.Warnf("DaemonSet %s has no desired pods", name)
		return kaev1.NotReady, nil
	}
	if foundDS.Status.NumberUnavailable != 0 {
		zlog.Warnf("Daemonset has %d unavailable pods", foundDS.Status.NumberUnavailable)
		return kaev1.NotReady, nil
	}
	// if ds is running with "OnDelete" strategy, check if the revision matches for all pods
	if foundDS.Spec.UpdateStrategy.Type != appsv1.OnDeleteDaemonSetStrategyType {
		return kaev1.Ready, nil
	}

	opts := []client.ListOption{client.MatchingLabels(foundDS.Spec.Template.ObjectMeta.Labels)}

	foundPodList := &corev1.PodList{}
	err = manager.client.List(manager.ctx, foundPodList, opts...)
	if err != nil {
		zlog.Errorf("fail to get PodList, err: %v", err)
		return kaev1.NotReady, err
	}
	zlog.Infof("Number Of Pods: %d", len(foundPodList.Items))
	if len(foundPodList.Items) == 0 {
		zlog.Warnf("number Of Pods: %d", len(foundPodList.Items))
		return kaev1.NotReady, nil
	}

	return isAllDaemonSetPodReady(foundDS, foundPodList, manager)
}

func isAllDaemonSetPodReady(ds *appsv1.DaemonSet, list *corev1.PodList, manager KAEPolicyManager) (kaev1.State, error) {
	daemonSetRevisionHash, err := getCtrlRevisionHash(manager.ctx, ds, manager)
	if err != nil {
		zlog.Errorf("Failed to get daemonSet template revision hash, err: %v", err)
		return kaev1.NotReady, err
	}

	dsPods := filterDaemonSetOwnedPods(ds, list.Items)
	for _, dsPod := range dsPods {
		if dsPod.Status.Phase != "Running" {
			zlog.Warn("pod is not Running")
			return kaev1.NotReady, nil
		}
		if !isAllContainersReady(dsPod) {
			zlog.Warnf("pod %s ContainerStatuses is not all ready", dsPod.Name)
			return kaev1.NotReady, nil
		}
		if podRevHash, ok := dsPod.Labels[labelKeyCtrlRevHash]; !ok {
			errMsg := fmt.Errorf("controller-revision-hash label not present for pod %s", dsPod.Name)
			zlog.Errorf("fail to get pod template revision hash, err: %v", errMsg)
			return kaev1.NotReady, errMsg
		} else if podRevHash != daemonSetRevisionHash {
			zlog.Warn("pod template revision does not match daemonSet template revision")
			return kaev1.NotReady, nil
		}
	}
	return kaev1.Ready, nil
}

func isAllContainersReady(pod corev1.Pod) bool {
	for _, status := range pod.Status.ContainerStatuses {
		if !status.Ready {
			return false
		}
	}
	return true
}

func filterDaemonSetOwnedPods(ds *appsv1.DaemonSet, pods []corev1.Pod) []corev1.Pod {
	var podsFiltered []corev1.Pod
	for _, pod := range pods {
		for _, ownerReference := range pod.OwnerReferences {
			if ownerReference.UID == ds.UID {
				podsFiltered = append(podsFiltered, pod)
				break
			}
		}
	}
	return podsFiltered
}

func getCtrlRevisionHash(ctx context.Context, daemonSet *appsv1.DaemonSet, n KAEPolicyManager) (string, error) {
	// get all revisions for the daemonSet
	ctrlRevList := &appsv1.ControllerRevisionList{}
	err := n.client.List(ctx, ctrlRevList, []client.ListOption{
		client.InNamespace(n.namespace),
		client.MatchingLabels(daemonSet.Spec.Selector.MatchLabels),
	}...)
	if err != nil {
		zlog.Errorf("fail to get controller revision list for daemonSet %s: %v", daemonSet.Name, err)
		return "", err
	}
	zlog.Infof("Obtained controller revisions, Daemonset: %s, len: %d", daemonSet.Name, len(ctrlRevList.Items))

	ctrlRevListFiltered, err := filterAndSortControllerRevisions(ctrlRevList.Items, daemonSet.Name)
	if err != nil {
		zlog.Warnf("no revision found for daemonSet %s: %v", daemonSet.Name, err)
		return "", err
	}

	currRevision := ctrlRevListFiltered[0].Name
	return strings.TrimPrefix(currRevision, fmt.Sprintf("%s-", daemonSet.Name)), nil
}

func filterAndSortControllerRevisions(revisions []appsv1.ControllerRevision, daemonSetName string) (
	[]appsv1.ControllerRevision, error) {
	var filteredRevisions []appsv1.ControllerRevision
	for _, controllerRevision := range revisions {
		if strings.HasPrefix(controllerRevision.Name, daemonSetName) {
			filteredRevisions = append(filteredRevisions, controllerRevision)
		}
	}
	if len(filteredRevisions) == 0 {
		return nil, fmt.Errorf("no revision found for daemonSet %s", daemonSetName)
	}

	// sort the revision list to make sure we obtain latest revision always
	sort.Slice(filteredRevisions, func(i, j int) bool {
		return filteredRevisions[i].Revision > filteredRevisions[j].Revision
	})
	return filteredRevisions, nil
}

// isDaemonSetSpecChanged returns true if the spec has changed between existing one
// and new DaemonSet spec compared by hash.
func isDaemonSetSpecChanged(old *appsv1.DaemonSet, new *appsv1.DaemonSet) bool {
	if old == nil && new != nil {
		return true
	}
	if old == nil || new == nil {
		return false
	}
	if old.Annotations == nil || new.Annotations == nil {
		zlog.Warn("appsv1.DaemonSet.Annotations must be allocated when comparing DaemonSetSpec")
	}

	return comparreAndUdateDaemonSetAnnotations(old, new)
}

func comparreAndUdateDaemonSetAnnotations(old *appsv1.DaemonSet, new *appsv1.DaemonSet) bool {
	hashStr := getDaemonSetHash(new)
	annotationUpdated := false
	if value, ok := old.Annotations[annotationKeyLastAppliedHash]; ok {
		if value != hashStr {
			new.Annotations[annotationKeyLastAppliedHash] = hashStr
			annotationUpdated = true
		}
	} else {
		new.Annotations[annotationKeyLastAppliedHash] = hashStr
		annotationUpdated = true
	}

	return annotationUpdated
}
