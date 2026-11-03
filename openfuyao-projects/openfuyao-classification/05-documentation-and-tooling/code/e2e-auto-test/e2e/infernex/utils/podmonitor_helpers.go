package utils

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

var (
	// PodMonitorGVR 是 PodMonitor 的 GroupVersionResource
	PodMonitorGVR = schema.GroupVersionResource{
		Group:    "monitoring.coreos.com",
		Version:  "v1",
		Resource: "podmonitors",
	}
)

// HasPodMonitorCRD 检查集群是否安装了 PodMonitor CRD
func HasPodMonitorCRD(clientset kubernetes.Interface) bool {
	InvalidateDiscovery(clientset)
	_, err := clientset.Discovery().ServerResourcesForGroupVersion("monitoring.coreos.com/v1")
	return err == nil
}

// ListPodMonitors 列出指定 namespace 的所有 PodMonitor 资源
func ListPodMonitors(dynamicClient dynamic.Interface, ctx context.Context, namespace string) ([]unstructured.Unstructured, error) {
	list, err := dynamicClient.Resource(PodMonitorGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("list PodMonitors: %w", err)
	}
	return list.Items, nil
}
