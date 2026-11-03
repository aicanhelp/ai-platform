package utils

import (
	"context"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

var (
	gvrInferNexService = schema.GroupVersionResource{
		Group: "infernex.infernex.io", Version: "v1alpha1", Resource: "infernexservices",
	}
	gvrInferNexServiceConfig = schema.GroupVersionResource{
		Group: "infernex.infernex.io", Version: "v1alpha1", Resource: "infernexserviceconfigs",
	}
)

func InferNexServiceGVR() schema.GroupVersionResource { return gvrInferNexService }

func GetInferNexService(ctx context.Context, dyn dynamic.Interface, ns, name string) (*unstructured.Unstructured, error) {
	return dyn.Resource(gvrInferNexService).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
}

func InferNexServiceExists(ctx context.Context, dyn dynamic.Interface, ns, name string) (bool, error) {
	_, err := GetInferNexService(ctx, dyn, ns, name)
	if err != nil {
		if isNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func DeleteInferNexService(ctx context.Context, dyn dynamic.Interface, ns, name string) error {
	return dyn.Resource(gvrInferNexService).Namespace(ns).Delete(ctx, name, metav1.DeleteOptions{})
}

func GetInferNexServiceConfig(ctx context.Context, dyn dynamic.Interface, ns, name string) (*unstructured.Unstructured, error) {
	return dyn.Resource(gvrInferNexServiceConfig).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
}

func DeleteInferNexServiceConfig(ctx context.Context, dyn dynamic.Interface, ns, name string) error {
	return dyn.Resource(gvrInferNexServiceConfig).Namespace(ns).Delete(ctx, name, metav1.DeleteOptions{})
}

func ConditionStatus(obj *unstructured.Unstructured, condType string) (status, reason, message string, ok bool) {
	conds, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if err != nil || !found {
		return "", "", "", false
	}
	for _, c := range conds {
		m, _ := c.(map[string]interface{})
		if m == nil {
			continue
		}
		t, _ := m["type"].(string)
		if t != condType {
			continue
		}
		status, _ = m["status"].(string)
		reason, _ = m["reason"].(string)
		message, _ = m["message"].(string)
		return status, reason, message, true
	}
	return "", "", "", false
}

func isNotFound(err error) bool {
	return apierrors.IsNotFound(err)
}
