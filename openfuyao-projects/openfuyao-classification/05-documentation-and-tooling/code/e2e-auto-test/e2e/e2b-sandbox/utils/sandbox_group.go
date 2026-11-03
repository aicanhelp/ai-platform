package utils

import (
	"context"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

var (
	sandboxGroupGVR = schema.GroupVersionResource{
		Group:    "sandbox.flux.io",
		Version:  "v1alpha1",
		Resource: "sandboxgroups",
	}
)

// EnsureSandboxGroup creates a SandboxGroup CR if it does not exist, waits for
// it to become Ready. If the SG already exists, it is reused as-is.
func EnsureSandboxGroup(ctx context.Context, k8sClient *k8s.K8SClient, sgName, namespace string) error {
	dc := k8sClient.DynamicClient.Resource(sandboxGroupGVR).Namespace(namespace)

	// If already exists, reuse it
	_, err := dc.Get(ctx, sgName, metav1.GetOptions{})
	if err == nil {
		return waitSandboxGroupReady(ctx, dc, sgName)
	}

	// Create SandboxGroup
	sg := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "sandbox.flux.io/v1alpha1",
		"kind":       "SandboxGroup",
		"metadata": map[string]interface{}{
			"name":      sgName,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"capacity": map[string]interface{}{
				"agentPodMin":        1,
				"agentPodMax":        1,
				"sandboxBufferMin":   1,
				"maxSandboxesPerPod": 200,
			},
			"sandboxResources": map[string]interface{}{
				"requests": map[string]interface{}{
					"cpu":    "100m",
					"memory": "256Mi",
				},
				"limits": map[string]interface{}{
					"cpu":    "500m",
					"memory": "512Mi",
				},
			},
			"agentEnv": map[string]interface{}{
				"RUNTIME_TYPE": "e2b",
			},
		},
	}}

	_, err = dc.Create(ctx, sg, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("create SandboxGroup %s: %w", sgName, err)
	}

	return waitSandboxGroupReady(ctx, dc, sgName)
}

// DeleteSandboxGroup deletes a SandboxGroup CR.
func DeleteSandboxGroup(ctx context.Context, k8sClient *k8s.K8SClient, sgName, namespace string) error {
	dc := k8sClient.DynamicClient.Resource(sandboxGroupGVR).Namespace(namespace)
	return dc.Delete(ctx, sgName, metav1.DeleteOptions{})
}

// waitSandboxGroupReady waits for the SandboxGroup Ready condition, then returns.
func waitSandboxGroupReady(ctx context.Context, dc dynamic.ResourceInterface, sgName string) error {
	// Wait for Ready condition (poll up to 2 minutes)
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		obj, err := dc.Get(ctx, sgName, metav1.GetOptions{})
		if err == nil {
			conditions, found, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
			if found {
				for _, c := range conditions {
					cond, ok := c.(map[string]interface{})
					if !ok {
						continue
					}
					if cond["type"] == "Ready" && cond["status"] == "True" {
						return nil
					}
				}
			}
		}
		time.Sleep(5 * time.Second)
	}
	return fmt.Errorf("SandboxGroup %s did not become Ready within 2 minutes", sgName)
}
