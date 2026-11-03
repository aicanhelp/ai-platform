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

package phaseutil

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

const (
	gracefulShutdownDuration = 2 * time.Second
)

type DeploymentTarget struct {
	Namespace string
	Name      string
	Container string
}

// GetDeploymentImage 读取指定 Deployment 容器的当前镜像
func GetDeploymentImage(ctx context.Context, cli client.Client, target DeploymentTarget) (string, error) {
	var deploy appsv1.Deployment
	if err := cli.Get(ctx, types.NamespacedName{Namespace: target.Namespace, Name: target.Name}, &deploy); err != nil {
		return "", fmt.Errorf("get Deployment %s/%s: %w", target.Namespace, target.Name, err)
	}

	for _, c := range deploy.Spec.Template.Spec.Containers {
		if c.Name == target.Container {
			return c.Image, nil
		}
	}
	return "", fmt.Errorf("容器 %s 未找到", target.Container)
}

// PatchDeploymentImage 更新 Deployment 指定容器的镜像
func PatchDeploymentImage(ctx context.Context, cli client.Client, target DeploymentTarget, image string) error {
	return RetryOnConflict(func() error {
		var deploy appsv1.Deployment
		if err := cli.Get(ctx, types.NamespacedName{Namespace: target.Namespace, Name: target.Name}, &deploy); err != nil {
			return fmt.Errorf("get Deployment: %w", err)
		}

		updated := false
		for i := range deploy.Spec.Template.Spec.Containers {
			if deploy.Spec.Template.Spec.Containers[i].Name == target.Container {
				deploy.Spec.Template.Spec.Containers[i].Image = image
				updated = true
				break
			}
		}
		if !updated {
			return fmt.Errorf("容器 %s 未找到", target.Container)
		}

		if deploy.Spec.Template.Annotations == nil {
			deploy.Spec.Template.Annotations = make(map[string]string)
		}
		deploy.Spec.Template.Annotations[annotation.ProviderRestartedAtAnnotationKey] = time.Now().Format(time.RFC3339)

		if err := cli.Update(ctx, &deploy); err != nil {
			return fmt.Errorf("update Deployment: %w", err)
		}
		return nil
	})
}

// WaitDeploymentReady 等待 Deployment Available 且存在使用目标镜像的 Ready Pod
func WaitDeploymentReady(ctx context.Context, cli client.Client, target DeploymentTarget, targetImage string, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, gracefulShutdownDuration, timeout, true, func(ctx context.Context) (bool, error) {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		default:
		}

		var deploy appsv1.Deployment
		if err := cli.Get(ctx, types.NamespacedName{Namespace: target.Namespace, Name: target.Name}, &deploy); err != nil {
			return false, err
		}

		if deploy.Status.UpdatedReplicas != *deploy.Spec.Replicas ||
			deploy.Status.AvailableReplicas != *deploy.Spec.Replicas {
			return false, nil
		}

		selector := labels.SelectorFromSet(deploy.Spec.Selector.MatchLabels)
		var podList corev1.PodList
		if err := cli.List(ctx, &podList, &client.ListOptions{
			Namespace:     target.Namespace,
			LabelSelector: selector,
		}); err != nil {
			return false, err
		}

		for _, pod := range podList.Items {
			if podHasImage(pod, targetImage) && PodIsReady(pod) {
				return true, nil
			}
		}

		return false, nil
	})
}

func podHasImage(pod corev1.Pod, image string) bool {
	for _, c := range pod.Spec.Containers {
		if c.Image == image {
			return true
		}
	}
	return false
}

func PodIsReady(pod corev1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
