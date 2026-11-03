/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phases

import (
	"context"
	"fmt"
	"strings"

	"github.com/coreos/go-semver/semver"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureComponentUpgradeName confv1beta1.BKEClusterPhase = "EnsureComponentUpgrade"
)

type EnsureComponentUpgrade struct {
	phaseframe.BasePhase
	localKubeConfig []byte
	remoteClient    *kubernetes.Clientset
	mockClient      kubernetes.Interface
}

// kubeClient returns the in-cluster client used to upgrade workload images.
// In production this is the remote client obtained from the target cluster;
// tests may inject a fake clientset via mockClient.
func (e *EnsureComponentUpgrade) kubeClient() kubernetes.Interface {
	if e.mockClient != nil {
		return e.mockClient
	}
	return e.remoteClient
}

func NewEnsureComponentUpgrade(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureComponentUpgradeName)
	return &EnsureComponentUpgrade{BasePhase: base}
}

// Execute 执行具体的升级操作
func (e *EnsureComponentUpgrade) Execute() (ctrl.Result, error) {
	if err := e.getRemoteClient(); err != nil {
		return ctrl.Result{}, err
	}
	if err := e.loadLocalKubeConfig(); err != nil {
		return ctrl.Result{}, err
	}
	return e.rolloutOpenfuyaoComponent()
}

func (e *EnsureComponentUpgrade) loadLocalKubeConfig() error {
	phaseContextFields76 := e.Ctx.Untie()
	ctx, c, _, _, _ := phaseContextFields76.Context,
		phaseContextFields76.Client,
		phaseContextFields76.BKECluster,
		phaseContextFields76.Scheme,
		phaseContextFields76.Log
	localKubeConfig, err := phaseutil.GetLocalKubeConfig(ctx, c)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fmt.Errorf("local kubeconfig secret not found")
		}
		return fmt.Errorf("failed to get local kubeconfig secret: %w", err)
	}
	e.localKubeConfig = localKubeConfig
	return nil
}

// getRemoteClient get remote cluster client
func (e *EnsureComponentUpgrade) getRemoteClient() error {
	phaseContextFields90 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, _ := phaseContextFields90.Context,
		phaseContextFields90.Client,
		phaseContextFields90.BKECluster,
		phaseContextFields90.Scheme,
		phaseContextFields90.Log
	targetClusterClient, err := kube.NewRemoteClientByBKECluster(ctx, c, bkeCluster)
	if err != nil {
		return fmt.Errorf("failed to get BKECluster %q remote cluster client: %w", utils.ClientObjNS(bkeCluster), err)
	}
	e.remoteClient, _ = targetClusterClient.KubeClient()
	if e.remoteClient == nil {
		return fmt.Errorf("failed to get remote client")
	}
	return nil
}

func (e *EnsureComponentUpgrade) isPatchVersion(version string) bool {
	cleanVersion := strings.TrimPrefix(version, "v")

	v, err := semver.NewVersion(cleanVersion)
	if err != nil {
		return false
	}

	return v.Patch > 0 && v.PreRelease == ""
}

func (e *EnsureComponentUpgrade) isComponentNeedUpgrade(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	// 初次安装，如果安装的是补丁版本，那么就需要进行openFuyao核心组件的升级处理
	if new.Status.OpenFuyaoVersion == "" {
		return e.isPatchVersion(new.Spec.ClusterConfig.Cluster.OpenFuyaoVersion)
	}
	// 非初次安装，需要根据status和spec中的openFuyao版本判断是否需要进行核心组件的升级处理
	// Use NodeFetcher to get BKENodes from API server
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, new)
	if err != nil {
		return false
	}
	nodes := phaseutil.GetNeedUpgradeComponentNodesWithBKENodes(new, bkeNodes)
	if nodes == nil || nodes.Length() == 0 {
		return false
	}

	return true
}

// NeedExecute 这个阶段，只有在初始新建补丁版本时才需要执行，如何判断是初始新建补丁版本？old为空，new中openFuyao version带小版本
func (e *EnsureComponentUpgrade) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}
	if !e.isComponentNeedUpgrade(old, new) {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureComponentUpgrade) rolloutOpenfuyaoComponent() (ctrl.Result, error) {
	phaseContextFields145 := e.Ctx.Untie()
	_, _, bkeCluster, _, log := phaseContextFields145.Context,
		phaseContextFields145.Client,
		phaseContextFields145.BKECluster,
		phaseContextFields145.Scheme,
		phaseContextFields145.Log
	patchCfg, err := e.getPatchConfig()
	if err != nil {
		return ctrl.Result{}, err
	}

	if err = e.processImageUpdates(patchCfg); err != nil {
		return ctrl.Result{}, err
	}

	log.Info(constant.ComponentUpgradeSuccess, "upgrade all component success")
	bkeCluster.Status.OpenFuyaoVersion = bkeCluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion

	return ctrl.Result{}, nil
}

func (e *EnsureComponentUpgrade) getPatchConfig() (*phaseutil.PatchConfig, error) {
	phaseContextFields162 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields162.Context,
		phaseContextFields162.Client,
		phaseContextFields162.BKECluster,
		phaseContextFields162.Scheme,
		phaseContextFields162.Log
	openFuyaoVersion := bkeCluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion
	log.Info(constant.ComponentUpgradingReason, "openFuyaoVersion: %v", openFuyaoVersion)

	patchCfg, err := phaseutil.LoadPatchConfig(ctx, c, openFuyaoVersion)
	if err != nil {
		return nil, err
	}
	log.Info(constant.ComponentUpgradingReason, "get patch config for version %s successfully", openFuyaoVersion)
	return patchCfg, nil
}

func (e *EnsureComponentUpgrade) processImageUpdates(patchCfg *phaseutil.PatchConfig) error {
	for _, repo := range patchCfg.Repos {
		if err := e.processRepoImages(repo); err != nil {
			return err
		}
	}
	return nil
}

func (e *EnsureComponentUpgrade) processRepoImages(repo phaseutil.Repo) error {
	if repo.IsKubernetes {
		return nil // k8s组件在其他流程升级
	}

	for _, subImage := range repo.SubImages {
		if err := e.processSubImage(subImage); err != nil {
			return err
		}
	}
	return nil
}

func (e *EnsureComponentUpgrade) processSubImage(subImage phaseutil.SubImage) error {
	for _, image := range subImage.Images {
		if err := e.updateSingleImage(image); err != nil {
			return err
		}
	}
	return nil
}

func (e *EnsureComponentUpgrade) updateSingleImage(image phaseutil.Image) error {
	phaseContextFields206 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields206.Context,
		phaseContextFields206.Client,
		phaseContextFields206.BKECluster,
		phaseContextFields206.Scheme,
		phaseContextFields206.Log
	// 依次处理镜像，补丁升级操作确保只会有一个tag
	if len(image.Tag) == 0 {
		return fmt.Errorf("image %s has no tags", image.Name)
	}

	tag := image.Tag[0]
	for _, podInfo := range image.UsedPodInfo {
		updateInfo := &phaseutil.ImageUpdate{
			ImageName: image.Name,
			PodPrefix: podInfo.PodPrefix,
			NameSpace: podInfo.NameSpace,
			NewTag:    tag,
		}
		log.Info(constant.ComponentUpgradingReason, "update info is %+v", updateInfo)
		if err := e.updatePodImageTag(updateInfo); err != nil {
			return fmt.Errorf("update image %s tag failed: %w", image.Name, err)
		}
	}

	return nil
}

func (e *EnsureComponentUpgrade) updatePodImageTag(update *phaseutil.ImageUpdate) error {
	phaseContextFields230 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields230.Context,
		phaseContextFields230.Client,
		phaseContextFields230.BKECluster,
		phaseContextFields230.Scheme,
		phaseContextFields230.Log
	pods, err := e.findMatchingPods(update.NameSpace, update.PodPrefix)
	if err != nil {
		return fmt.Errorf("failed to find matching %s/%s pods: %w", update.NameSpace, update.PodPrefix, err)
	}

	if len(pods) == 0 {
		log.Info(constant.ComponentUpgradingReason, "no pods found in %s with prefix %s", update.NameSpace, update.PodPrefix)
		return nil
	}
	return e.upgradePodImage(pods[0], update)
}

func (e *EnsureComponentUpgrade) findMatchingPods(namespace, podPrefix string) ([]corev1.Pod, error) {
	phaseContextFields244 := e.Ctx.Untie()
	ctx, _, _, _, _ := phaseContextFields244.Context,
		phaseContextFields244.Client,
		phaseContextFields244.BKECluster,
		phaseContextFields244.Scheme,
		phaseContextFields244.Log
	pods, err := e.kubeClient().CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	var matchingPods []corev1.Pod
	for _, pod := range pods.Items {
		if strings.HasPrefix(pod.Name, podPrefix) {
			matchingPods = append(matchingPods, pod)
		}
	}

	return matchingPods, nil
}

func (e *EnsureComponentUpgrade) upgradePodImage(pod corev1.Pod, update *phaseutil.ImageUpdate) error {
	phaseContextFields260 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields260.Context,
		phaseContextFields260.Client,
		phaseContextFields260.BKECluster,
		phaseContextFields260.Scheme,
		phaseContextFields260.Log
	controller, controllerType, err := e.getPodController(pod)
	if err != nil {
		return fmt.Errorf("failed to get controller for pod %s: %w", pod.Name, err)
	}
	log.Info(constant.ComponentUpgradingReason, "Pod %s is managed by %s: %s", pod.Name, controllerType, controller.GetName())

	switch controllerType {
	case "Deployment":
		if deployment, ok := controller.(*appsv1.Deployment); ok {
			return e.upgradeDeploymentImage(deployment, update)
		}
		return fmt.Errorf("controller is not a Deployment as expected, got: %T", controller)
	case "StatefulSet":
		if statefulSet, ok := controller.(*appsv1.StatefulSet); ok {
			return e.upgradeStatefulSetImage(statefulSet, update)
		}
		return fmt.Errorf("controller is not a StatefulSet as expected, got: %T", controller)
	case "DaemonSet":
		if daemonSet, ok := controller.(*appsv1.DaemonSet); ok {
			return e.upgradeDaemonSetImage(daemonSet, update)
		}
		return fmt.Errorf("controller is not a DaemonSet as expected, got: %T", controller)
	case "ReplicaSet":
		if replicaSet, ok := controller.(*appsv1.ReplicaSet); ok {
			return e.upgradeReplicaSetImage(replicaSet, update)
		}
		return fmt.Errorf("controller is not a ReplicaSet as expected, got: %T", controller)
	default:
		return fmt.Errorf("unsupported controller type: %s", controllerType)
	}
}

func (e *EnsureComponentUpgrade) getPodController(pod corev1.Pod) (metav1.Object, string, error) {
	phaseContextFields294 := e.Ctx.Untie()
	ctx, _, _, _, _ := phaseContextFields294.Context,
		phaseContextFields294.Client,
		phaseContextFields294.BKECluster,
		phaseContextFields294.Scheme,
		phaseContextFields294.Log
	namespace := e.getNamespace(pod)

	for _, ownerRef := range pod.OwnerReferences {
		controller, kind, err := e.handleOwnerReference(ctx, e.kubeClient(), namespace, ownerRef)
		if err != nil {
			return nil, "", err
		}
		if controller != nil {
			return controller, kind, nil
		}
	}

	return &pod, "Pod", nil
}

func (e *EnsureComponentUpgrade) getNamespace(pod corev1.Pod) string {
	if pod.Namespace == "" {
		return metav1.NamespaceDefault
	}
	return pod.Namespace
}

func (e *EnsureComponentUpgrade) handleOwnerReference(ctx context.Context, clientSet kubernetes.Interface, namespace string, ownerRef metav1.OwnerReference) (metav1.Object, string, error) {
	switch ownerRef.Kind {
	case "ReplicaSet":
		return e.handleReplicaSet(ctx, clientSet, namespace, ownerRef)
	case "StatefulSet":
		return e.handleStatefulSet(ctx, clientSet, namespace, ownerRef)
	case "DaemonSet":
		return e.handleDaemonSet(ctx, clientSet, namespace, ownerRef)
	default:
		return nil, "", nil
	}
}

func (e *EnsureComponentUpgrade) handleReplicaSet(ctx context.Context, clientSet kubernetes.Interface, namespace string, ownerRef metav1.OwnerReference) (metav1.Object, string, error) {
	rs, err := clientSet.AppsV1().ReplicaSets(namespace).Get(ctx, ownerRef.Name, metav1.GetOptions{})
	if err != nil {
		return nil, "", err
	}

	for _, rsOwnerRef := range rs.OwnerReferences {
		if rsOwnerRef.Kind == "Deployment" {
			deployment, err := clientSet.AppsV1().Deployments(namespace).Get(ctx, rsOwnerRef.Name, metav1.GetOptions{})
			if err != nil {
				return nil, "", err
			}
			return deployment, "Deployment", nil
		}
	}

	return rs, "ReplicaSet", nil
}

func (e *EnsureComponentUpgrade) handleStatefulSet(ctx context.Context, clientSet kubernetes.Interface, namespace string, ownerRef metav1.OwnerReference) (metav1.Object, string, error) {
	statefulSet, err := clientSet.AppsV1().StatefulSets(namespace).Get(ctx, ownerRef.Name, metav1.GetOptions{})
	if err != nil {
		return nil, "", err
	}
	return statefulSet, "StatefulSet", nil
}

func (e *EnsureComponentUpgrade) handleDaemonSet(ctx context.Context, clientSet kubernetes.Interface, namespace string, ownerRef metav1.OwnerReference) (metav1.Object, string, error) {
	daemonSet, err := clientSet.AppsV1().DaemonSets(namespace).Get(ctx, ownerRef.Name, metav1.GetOptions{})
	if err != nil {
		return nil, "", err
	}
	return daemonSet, "DaemonSet", nil
}

func (e *EnsureComponentUpgrade) isMatchingImage(currentImage, targetImageName string) bool {
	var imageNameWithoutTag string

	lastColonIndex := strings.LastIndex(currentImage, ":")
	if lastColonIndex != -1 {
		slashIndex := strings.LastIndex(currentImage, "/")
		if slashIndex == -1 || lastColonIndex > slashIndex {
			imageNameWithoutTag = currentImage[:lastColonIndex]
		} else {
			imageNameWithoutTag = currentImage
		}
	} else {
		imageNameWithoutTag = currentImage
	}

	return strings.HasSuffix(imageNameWithoutTag, targetImageName)
}

func (e *EnsureComponentUpgrade) buildNewImage(currentImage, newTag string) string {
	lastColonIndex := strings.LastIndex(currentImage, ":")
	if lastColonIndex != -1 {
		slashIndex := strings.LastIndex(currentImage, "/")
		if slashIndex == -1 || lastColonIndex > slashIndex {
			return fmt.Sprintf("%s:%s", currentImage[:lastColonIndex], newTag)
		}
	}
	return fmt.Sprintf("%s:%s", currentImage, newTag)
}

func (e *EnsureComponentUpgrade) upgradeDeploymentImage(deployment *appsv1.Deployment, update *phaseutil.ImageUpdate) error {
	phaseContextFields395 := e.Ctx.Untie()
	ctx, _, _, _, log := phaseContextFields395.Context,
		phaseContextFields395.Client,
		phaseContextFields395.BKECluster,
		phaseContextFields395.Scheme,
		phaseContextFields395.Log
	namespace := deployment.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceDefault
	}
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		deploymentCfg, err := e.kubeClient().AppsV1().Deployments(namespace).Get(ctx, deployment.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("get deployment err: %s", err)
		}

		needUpdated := false
		for i, container := range deploymentCfg.Spec.Template.Spec.Containers {
			log.Info(constant.ComponentUpgradingReason, "Updating Deployment image info %s: %s -> %s", container.Image, update.ImageName, update.NewTag)
			if e.isMatchingImage(container.Image, update.ImageName) {
				newImage := e.buildNewImage(container.Image, update.NewTag)
				if container.Image != newImage {
					deploymentCfg.Spec.Template.Spec.Containers[i].Image = newImage
					needUpdated = true
					log.Info(constant.ComponentUpgradingReason, "Updating Deployment %s: %s -> %s", deployment.Name, container.Image, newImage)
				}
			}
		}
		for i, initContainer := range deploymentCfg.Spec.Template.Spec.InitContainers {
			log.Info(constant.ComponentUpgradingReason, "Updating Deployment initContainer image info %s: %s -> %s", initContainer.Image, update.ImageName, update.NewTag)
			if e.isMatchingImage(initContainer.Image, update.ImageName) {
				newImage := e.buildNewImage(initContainer.Image, update.NewTag)
				if initContainer.Image != newImage {
					deploymentCfg.Spec.Template.Spec.InitContainers[i].Image = newImage
					needUpdated = true
					log.Info(constant.ComponentUpgradingReason, "Updating Deployment initContainer %s: %s -> %s", deployment.Name, initContainer.Image, newImage)
				}
			}
		}

		if needUpdated {
			_, err = e.kubeClient().AppsV1().Deployments(namespace).Update(ctx, deploymentCfg, metav1.UpdateOptions{})
			return err
		}
		log.Info(constant.ComponentUpgradingReason, "No containers or initContainers with image '%s' found in Deployment %s", update.ImageName, deployment.Name)
		return nil
	})
}

func (e *EnsureComponentUpgrade) upgradeStatefulSetImage(statefulSet *appsv1.StatefulSet, update *phaseutil.ImageUpdate) error {
	phaseContextFields440 := e.Ctx.Untie()
	ctx, _, _, _, log := phaseContextFields440.Context,
		phaseContextFields440.Client,
		phaseContextFields440.BKECluster,
		phaseContextFields440.Scheme,
		phaseContextFields440.Log
	namespace := statefulSet.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceDefault
	}
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		stsCfg, err := e.kubeClient().AppsV1().StatefulSets(namespace).Get(ctx, statefulSet.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		needUpdated := false
		for i, container := range stsCfg.Spec.Template.Spec.Containers {
			log.Info(constant.ComponentUpgradingReason, "Updating StatefulSet image info %s: %s -> %s", container.Image, update.ImageName, update.NewTag)
			if e.isMatchingImage(container.Image, update.ImageName) {
				newImage := e.buildNewImage(container.Image, update.NewTag)
				if container.Image != newImage {
					stsCfg.Spec.Template.Spec.Containers[i].Image = newImage
					needUpdated = true
					log.Info(constant.ComponentUpgradingReason, "Updating StatefulSet %s: %s -> %s", statefulSet.Name, container.Image, newImage)
				}
			}
		}
		for i, initContainer := range stsCfg.Spec.Template.Spec.InitContainers {
			log.Info(constant.ComponentUpgradingReason, "Updating StatefulSet initContainer image info %s: %s -> %s", initContainer.Image, update.ImageName, update.NewTag)
			if e.isMatchingImage(initContainer.Image, update.ImageName) {
				newImage := e.buildNewImage(initContainer.Image, update.NewTag)
				if initContainer.Image != newImage {
					stsCfg.Spec.Template.Spec.InitContainers[i].Image = newImage
					needUpdated = true
					log.Info(constant.ComponentUpgradingReason, "Updating StatefulSet initContainer %s: %s -> %s", statefulSet.Name, initContainer.Image, newImage)
				}
			}
		}

		if needUpdated {
			_, err = e.kubeClient().AppsV1().StatefulSets(namespace).Update(ctx, stsCfg, metav1.UpdateOptions{})
			return err
		}

		log.Info(constant.ComponentUpgradingReason, "No containers or initContainers with image '%s' found in StatefulSet %s", update.ImageName, statefulSet.Name)
		return nil
	})
}

func (e *EnsureComponentUpgrade) upgradeDaemonSetImage(daemonSet *appsv1.DaemonSet, update *phaseutil.ImageUpdate) error {
	phaseContextFields486 := e.Ctx.Untie()
	ctx, _, _, _, log := phaseContextFields486.Context,
		phaseContextFields486.Client,
		phaseContextFields486.BKECluster,
		phaseContextFields486.Scheme,
		phaseContextFields486.Log
	namespace := daemonSet.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceDefault
	}
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		dsCfg, err := e.kubeClient().AppsV1().DaemonSets(namespace).Get(ctx, daemonSet.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		dsUpdated := false
		for i, container := range dsCfg.Spec.Template.Spec.Containers {
			log.Info(constant.ComponentUpgradingReason, "Updating DaemonSet image info %s: %s -> %s", container.Image, update.ImageName, update.NewTag)
			if e.isMatchingImage(container.Image, update.ImageName) {
				newImage := e.buildNewImage(container.Image, update.NewTag)
				if container.Image != newImage {
					dsCfg.Spec.Template.Spec.Containers[i].Image = newImage
					dsUpdated = true
					log.Info(constant.ComponentUpgradingReason, "Updating DaemonSet %s: %s -> %s", daemonSet.Name, container.Image, newImage)
				}
			}
		}
		for i, initContainer := range dsCfg.Spec.Template.Spec.InitContainers {
			log.Info(constant.ComponentUpgradingReason, "Updating DaemonSet initContainer image info %s: %s -> %s", initContainer.Image, update.ImageName, update.NewTag)
			if e.isMatchingImage(initContainer.Image, update.ImageName) {
				newImage := e.buildNewImage(initContainer.Image, update.NewTag)
				if initContainer.Image != newImage {
					dsCfg.Spec.Template.Spec.InitContainers[i].Image = newImage
					dsUpdated = true
					log.Info(constant.ComponentUpgradingReason, "Updating DaemonSet initContainer %s: %s -> %s", daemonSet.Name, initContainer.Image, newImage)
				}
			}
		}

		if dsUpdated {
			_, err = e.kubeClient().AppsV1().DaemonSets(namespace).Update(ctx, dsCfg, metav1.UpdateOptions{})
			return err
		}

		log.Info(constant.ComponentUpgradingReason, "No containers or initContainers with image '%s' found in DaemonSet %s", update.ImageName, daemonSet.Name)
		return nil
	})
}

func (e *EnsureComponentUpgrade) upgradeReplicaSetImage(replicaSet *appsv1.ReplicaSet, update *phaseutil.ImageUpdate) error {
	phaseContextFields532 := e.Ctx.Untie()
	ctx, _, _, _, log := phaseContextFields532.Context,
		phaseContextFields532.Client,
		phaseContextFields532.BKECluster,
		phaseContextFields532.Scheme,
		phaseContextFields532.Log
	namespace := replicaSet.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceDefault
	}
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		rsCfg, err := e.kubeClient().AppsV1().ReplicaSets(namespace).Get(ctx, replicaSet.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		rsUpdated := false
		for i, container := range rsCfg.Spec.Template.Spec.Containers {
			log.Info(constant.ComponentUpgradingReason, "Updating ReplicaSet image info %s: %s -> %s", container.Image, update.ImageName, update.NewTag)
			if e.isMatchingImage(container.Image, update.ImageName) {
				newImage := e.buildNewImage(container.Image, update.NewTag)
				if container.Image != newImage {
					rsCfg.Spec.Template.Spec.Containers[i].Image = newImage
					rsUpdated = true
					log.Info(constant.ComponentUpgradingReason, "Updating ReplicaSet %s: %s -> %s", replicaSet.Name, container.Image, newImage)
				}
			}
		}
		for i, initContainer := range rsCfg.Spec.Template.Spec.InitContainers {
			log.Info(constant.ComponentUpgradingReason, "Updating ReplicaSet initContainer image info %s: %s -> %s", initContainer.Image, update.ImageName, update.NewTag)
			if e.isMatchingImage(initContainer.Image, update.ImageName) {
				newImage := e.buildNewImage(initContainer.Image, update.NewTag)
				if initContainer.Image != newImage {
					rsCfg.Spec.Template.Spec.InitContainers[i].Image = newImage
					rsUpdated = true
					log.Info(constant.ComponentUpgradingReason, "Updating ReplicaSet initContainer %s: %s -> %s", replicaSet.Name, initContainer.Image, newImage)
				}
			}
		}

		if rsUpdated {
			_, err = e.kubeClient().AppsV1().ReplicaSets(namespace).Update(ctx, rsCfg, metav1.UpdateOptions{})
			return err
		}
		log.Info(constant.ComponentUpgradingReason, "No containers or initContainers with image '%s' found in ReplicaSet %s", update.ImageName, replicaSet.Name)
		return nil
	})
}
