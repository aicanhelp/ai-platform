/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package npu

import (
	"context"
	"fmt"
	"net"
	"sort"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	metav1labels "k8s.io/apimachinery/pkg/labels"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

type targetResolver interface {
	resolveTargetURL(ctx context.Context, namespace, podName string) (string, error)
}

type kubeTargetResolver struct {
	client   ctrlclient.Client
	params   Parameters
	selector metav1labels.Selector
}

func newKubeTargetResolver(kubeClient ctrlclient.Client, params Parameters) (targetResolver, error) {
	if kubeClient == nil {
		return nil, fmt.Errorf("kubernetes client must not be nil")
	}
	selector, err := metav1labels.Parse(params.ExporterPodLabelSelector)
	if err != nil {
		return nil, fmt.Errorf("parse exporterPodLabelSelector: %w", err)
	}
	return &kubeTargetResolver{client: kubeClient, params: params, selector: selector}, nil
}

func (r *kubeTargetResolver) resolveTargetURL(ctx context.Context, namespace, podName string) (string, error) {
	workloadPod, err := r.getWorkloadPod(ctx, namespace, podName)
	if err != nil {
		return "", err
	}
	if workloadPod.Spec.NodeName == "" {
		return "", fmt.Errorf("workload pod %s/%s has empty spec.nodeName", namespace, podName)
	}

	exporterPod, err := r.findExporterPodForNode(ctx, workloadPod.Spec.NodeName)
	if err != nil {
		return "", err
	}
	host := net.JoinHostPort(exporterPod.Status.PodIP, strconv.Itoa(int(r.params.ExporterPort)))
	return buildTargetURL(host, r.params.Path), nil
}

func (r *kubeTargetResolver) getWorkloadPod(ctx context.Context, namespace, podName string) (*corev1.Pod, error) {
	var pod corev1.Pod
	key := ctrlclient.ObjectKey{Namespace: namespace, Name: podName}
	if err := r.client.Get(ctx, key, &pod); err != nil {
		return nil, fmt.Errorf("get workload pod %s/%s: %w", namespace, podName, err)
	}
	return &pod, nil
}

func (r *kubeTargetResolver) findExporterPodForNode(ctx context.Context, nodeName string) (*corev1.Pod, error) {
	var exporterPods corev1.PodList
	if err := r.client.List(
		ctx,
		&exporterPods,
		ctrlclient.InNamespace(r.params.ExporterNamespace),
		ctrlclient.MatchingLabelsSelector{Selector: r.selector},
	); err != nil {
		return nil, fmt.Errorf("list exporter pods in namespace %s: %w", r.params.ExporterNamespace, err)
	}

	sort.Slice(exporterPods.Items, func(i, j int) bool {
		return exporterPods.Items[i].Name < exporterPods.Items[j].Name
	})
	for i := range exporterPods.Items {
		pod := &exporterPods.Items[i]
		if pod.Spec.NodeName != nodeName {
			continue
		}
		if pod.Status.Phase != corev1.PodRunning {
			continue
		}
		if pod.Status.PodIP == "" {
			return nil, fmt.Errorf("exporter pod %s/%s on node %s has empty status.podIP", pod.Namespace, pod.Name, nodeName)
		}
		return pod, nil
	}
	return nil, fmt.Errorf("no running exporter pod found on node %s", nodeName)
}
