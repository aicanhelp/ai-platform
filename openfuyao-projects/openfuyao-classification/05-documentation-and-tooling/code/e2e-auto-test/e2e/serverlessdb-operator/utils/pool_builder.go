package utils

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

// BuildDBResourcePool constructs an unstructured DBResourcePool backed by a
// placeholder compute image. The pool is created in ns with the given replicas
// and PostgreSQL major version. The compute container is named "compute" with a
// "pg" port on 5432, CPU/memory requests+limits usable for pool matching, and
// resizePolicy NotRequired so in-place vertical scaling does not restart.
func BuildDBResourcePool(name, ns string, replicas, pgVersion int) *unstructured.Unstructured {
	computeImage := EnvOr("SERVERLESSDB_COMPUTE_IMAGE", DefaultComputeImage)
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": GroupName + "/" + GroupVer,
			"kind":       KindPool,
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": ns,
			},
			"spec": map[string]interface{}{
				"replicas":  replicas,
				"pgVersion": pgVersion,
				"podTemplate": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  ContainerCompute,
								"image": computeImage,
								"env": []interface{}{
									map[string]interface{}{"name": "POSTGRES_PASSWORD", "value": "postgres"},
								},
								"resizePolicy": []interface{}{
									map[string]interface{}{"resourceName": "cpu", "restartPolicy": "NotRequired"},
									map[string]interface{}{"resourceName": "memory", "restartPolicy": "NotRequired"},
								},
								"ports": []interface{}{
									map[string]interface{}{"name": PortNamePg, "containerPort": DefaultPgPort},
								},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{"cpu": ComputeCpuReq, "memory": ComputeMemReq},
									"limits":   map[string]interface{}{"cpu": ComputeCpuLimit, "memory": ComputeMemLimit},
								},
							},
						},
					},
				},
			},
		},
	}
}

// SetReplicasPatch returns a merge-patch body to update spec.replicas.
func SetReplicasPatch(n int) []byte {
	return []byte(fmt.Sprintf(`{"spec":{"replicas":%d}}`, n))
}

// CreatePool creates a DBResourcePool and returns it. Caller is responsible for cleanup.
func CreatePool(ctx context.Context, k8sClient *k8s.K8SClient, pool *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	return k8sClient.DynamicClient.Resource(PoolGVR).Namespace(pool.GetNamespace()).Create(ctx, pool, metav1.CreateOptions{})
}

// PatchPoolReplicas patches spec.replicas on the named pool.
func PatchPoolReplicas(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string, replicas int) (*unstructured.Unstructured, error) {
	return k8sClient.DynamicClient.Resource(PoolGVR).Namespace(ns).Patch(ctx, name, types.MergePatchType, SetReplicasPatch(replicas), metav1.PatchOptions{})
}

// GetPool returns the named DBResourcePool.
func GetPool(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string) (*unstructured.Unstructured, error) {
	return k8sClient.DynamicClient.Resource(PoolGVR).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
}

// DeletePool deletes the named DBResourcePool (cascade via ownerRef cleans up DBInstances/Pods).
func DeletePool(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string) error {
	return k8sClient.DynamicClient.Resource(PoolGVR).Namespace(ns).Delete(ctx, name, metav1.DeleteOptions{})
}

// ListDBInstances lists DBInstances in ns; optionally filtered by a label selector.
func ListDBInstances(ctx context.Context, k8sClient *k8s.K8SClient, ns, labelSelector string) (*unstructured.UnstructuredList, error) {
	opts := metav1.ListOptions{}
	if labelSelector != "" {
		opts.LabelSelector = labelSelector
	}
	return k8sClient.DynamicClient.Resource(DBIGVR).Namespace(ns).List(ctx, opts)
}

// GetDBInstance returns the named DBInstance.
func GetDBInstance(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string) (*unstructured.Unstructured, error) {
	return k8sClient.DynamicClient.Resource(DBIGVR).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
}

// DBInstancePhase extracts status.phase from a DBInstance unstructured object.
func DBInstancePhase(obj *unstructured.Unstructured) string {
	phase, _, _ := unstructured.NestedString(obj.Object, "status", "phase")
	return phase
}

// DBInstanceAppRef extracts the app-ref label from a DBInstance.
func DBInstanceAppRef(obj *unstructured.Unstructured) string {
	if labels := obj.GetLabels(); labels != nil {
		return labels[LabelAppRef]
	}
	return ""
}

// BuildDBResourcePoolWithResources constructs a DBResourcePool with custom resource requirements.
// This is useful for testing scenarios like insufficient resources or capacity limits.
func BuildDBResourcePoolWithResources(name, ns string, replicas, pgVersion int,
	cpuReq, memReq, cpuLimit, memLimit string) *unstructured.Unstructured {
	computeImage := EnvOr("SERVERLESSDB_COMPUTE_IMAGE", DefaultComputeImage)
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": GroupName + "/" + GroupVer,
			"kind":       KindPool,
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": ns,
			},
			"spec": map[string]interface{}{
				"replicas":  replicas,
				"pgVersion": pgVersion,
				"podTemplate": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  ContainerCompute,
								"image": computeImage,
								"env": []interface{}{
									map[string]interface{}{"name": "POSTGRES_PASSWORD", "value": "postgres"},
								},
								"resizePolicy": []interface{}{
									map[string]interface{}{"resourceName": "cpu", "restartPolicy": "NotRequired"},
									map[string]interface{}{"resourceName": "memory", "restartPolicy": "NotRequired"},
								},
								"ports": []interface{}{
									map[string]interface{}{"name": PortNamePg, "containerPort": DefaultPgPort},
								},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{"cpu": cpuReq, "memory": memReq},
									"limits":   map[string]interface{}{"cpu": cpuLimit, "memory": memLimit},
								},
							},
						},
					},
				},
			},
		},
	}
}
