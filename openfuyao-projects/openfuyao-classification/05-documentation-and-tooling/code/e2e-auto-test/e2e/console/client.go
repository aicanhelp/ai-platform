package console

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// ConsoleTestConfig 控制台测试配置
type ConsoleTestConfig struct {
	BaseURL string
	Client  *http.Client
}

// Global test configuration - will be set by test suite
var testConfig *ConsoleTestConfig

func makeRequest(method, path string, body interface{}) (*http.Response, error) {
	if testConfig == nil {
		panic("testConfig is not initialized. Make sure BeforeSuite has been called.")
	}

	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reqBody = bytes.NewBuffer(jsonData)
	}

	req, err := http.NewRequest(method, testConfig.BaseURL+path, reqBody)
	if err != nil {
		return nil, err
	}

	if reqBody != nil {
		// 对于 PATCH 请求，使用 JSON Merge Patch 的 Content-Type
		if method == "PATCH" {
			req.Header.Set("Content-Type", "application/merge-patch+json")
		} else {
			req.Header.Set("Content-Type", "application/json")
		}
	}

	return testConfig.Client.Do(req)
}

func createPatch(name, namespace string) map[string]interface{} {
	patch := map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
			"labels": map[string]interface{}{
				"app": "test",
			},
			"annotations": map[string]interface{}{
				"app": "test",
			},
		},
	}
	if namespace != "" {
		patch["metadata"].(map[string]interface{})["namespace"] = namespace
	}
	return patch
}

func createMalformedPatch(name, namespace string) map[string]interface{} {
	patch := map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":        name,
			"namespace":   namespace,
			"labels":      "app: test",
			"annotations": "app:test",
		},
	}
	if namespace != "" {
		patch["metadata"].(map[string]interface{})["namespace"] = namespace
	}
	return patch
}

func createPod(name, namespace string) (*http.Response, error) {
	podYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "Pod",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"containers": []map[string]interface{}{
				{
					"name":  "test-container",
					"image": "nginx:latest",
				},
			},
		},
	}
	return makeRequest("POST", "/api/v1/namespaces/"+namespace+"/pods", podYAML)
}

func createDeployment(name, namespace string) (*http.Response, error) {
	deploymentYAML := map[string]interface{}{
		"apiVersion": "apps/v1",
		"kind":       "Deployment",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"replicas": 1,
			"selector": map[string]interface{}{
				"matchLabels": map[string]interface{}{
					"app": "test",
				},
			},
			"template": map[string]interface{}{
				"metadata": map[string]interface{}{
					"labels": map[string]interface{}{
						"app": "test",
					},
				},
				"spec": map[string]interface{}{
					"containers": []map[string]interface{}{
						{
							"name":  "test-container",
							"image": "nginx:latest",
						},
					},
				},
			},
		},
	}
	return makeRequest("POST", "/apis/apps/v1/namespaces/"+namespace+"/deployments", deploymentYAML)
}

func createStatefulSet(name, namespace string) (*http.Response, error) {
	statefulSetYAML := map[string]interface{}{
		"apiVersion": "apps/v1",
		"kind":       "StatefulSet",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"serviceName": name,
			"replicas":    1,
			"selector": map[string]interface{}{
				"matchLabels": map[string]interface{}{
					"app": "test",
				},
			},
			"template": map[string]interface{}{
				"metadata": map[string]interface{}{
					"labels": map[string]interface{}{
						"app": "test",
					},
				},
				"spec": map[string]interface{}{
					"containers": []map[string]interface{}{
						{
							"name":  "test-container",
							"image": "nginx:latest",
						},
					},
				},
			},
		},
	}
	return makeRequest("POST", "/apis/apps/v1/namespaces/"+namespace+"/statefulsets", statefulSetYAML)
}

func createDaemonSet(name, namespace string) (*http.Response, error) {
	daemonSetYAML := map[string]interface{}{
		"apiVersion": "apps/v1",
		"kind":       "DaemonSet",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"selector": map[string]interface{}{
				"matchLabels": map[string]interface{}{
					"app": "test",
				},
			},
			"template": map[string]interface{}{
				"metadata": map[string]interface{}{
					"labels": map[string]interface{}{
						"app": "test",
					},
				},
				"spec": map[string]interface{}{
					"containers": []map[string]interface{}{
						{
							"name":  "test-container",
							"image": "nginx:latest",
						},
					},
				},
			},
		},
	}
	return makeRequest("POST", "/apis/apps/v1/namespaces/"+namespace+"/daemonsets", daemonSetYAML)
}

func createJob(name, namespace string) (*http.Response, error) {
	jobYAML := map[string]interface{}{
		"apiVersion": "batch/v1",
		"kind":       "Job",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"containers": []map[string]interface{}{
						{
							"name":    "test-container",
							"image":   "busybox:latest",
							"command": []string{"echo", "hello"},
						},
					},
					"restartPolicy": "Never",
				},
			},
		},
	}
	return makeRequest("POST", "/apis/batch/v1/namespaces/"+namespace+"/jobs", jobYAML)
}

func createCronJob(name, namespace string) (*http.Response, error) {
	cronJobYAML := map[string]interface{}{
		"apiVersion": "batch/v1",
		"kind":       "CronJob",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"schedule": "*/5 * * * *", // 每5分钟执行一次
			"jobTemplate": map[string]interface{}{
				"spec": map[string]interface{}{
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []map[string]interface{}{
								{
									"name":    "test-container",
									"image":   "busybox:latest",
									"command": []string{"echo", "hello"},
								},
							},
							"restartPolicy": "Never",
						},
					},
				},
			},
		},
	}
	return makeRequest("POST", "/apis/batch/v1/namespaces/"+namespace+"/cronjobs", cronJobYAML)
}

func createConfigMap(name, namespace string) (*http.Response, error) {
	configMapYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"data": map[string]interface{}{
			"key1": "value1",
			"key2": "value2",
		},
	}
	return makeRequest("POST", "/api/v1/namespaces/"+namespace+"/configmaps", configMapYAML)
}

func createSecret(name, namespace string) (*http.Response, error) {
	secretYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "Secret",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"type": "Opaque",
		"data": map[string]interface{}{
			"key1": "dmFsdWUx", // base64 encoded "value1"
			"key2": "dmFsdWUy", // base64 encoded "value2"
		},
	}
	return makeRequest("POST", "/api/v1/namespaces/"+namespace+"/secrets", secretYAML)
}

func createService(name, namespace string) (*http.Response, error) {
	serviceYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "Service",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"type": "ClusterIP",
			"ports": []map[string]interface{}{
				{
					"port":       80,
					"targetPort": 8080,
					"protocol":   "TCP",
				},
			},
			"selector": map[string]interface{}{
				"app": "test",
			},
		},
	}
	return makeRequest("POST", "/api/v1/namespaces/"+namespace+"/services", serviceYAML)
}

func createIngress(name, namespace string) (*http.Response, error) {
	ingressYAML := map[string]interface{}{
		"apiVersion": "networking.k8s.io/v1",
		"kind":       "Ingress",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"rules": []map[string]interface{}{
				{
					"host": "example.com",
					"http": map[string]interface{}{
						"paths": []map[string]interface{}{
							{
								"path":     "/",
								"pathType": "Prefix",
								"backend": map[string]interface{}{
									"service": map[string]interface{}{
										"name": "test-service",
										"port": map[string]interface{}{
											"number": 80,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	return makeRequest("POST", "/apis/networking.k8s.io/v1/namespaces/"+namespace+"/ingresses", ingressYAML)
}

func createRole(name, namespace string) (*http.Response, error) {
	roleYAML := map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "Role",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"rules": []map[string]interface{}{
			{
				"apiGroups": []string{""},
				"resources": []string{"pods"},
				"verbs":     []string{"get", "list", "watch"},
			},
		},
	}
	return makeRequest("POST", "/apis/rbac.authorization.k8s.io/v1/namespaces/"+namespace+"/roles", roleYAML)
}

func createRoleBinding(name, roleName, namespace string) (*http.Response, error) {
	roleBindingYAML := map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "RoleBinding",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"roleRef": map[string]interface{}{
			"kind":     "Role",
			"name":     roleName,
			"apiGroup": "rbac.authorization.k8s.io",
		},
		"subjects": []map[string]interface{}{
			{
				"kind":     "User",
				"name":     "test-user",
				"apiGroup": "rbac.authorization.k8s.io",
			},
		},
	}
	return makeRequest("POST", "/apis/rbac.authorization.k8s.io/v1/namespaces/"+namespace+"/rolebindings", roleBindingYAML)
}

func createClusterRole(name string) (*http.Response, error) {
	clusterRoleYAML := map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "ClusterRole",
		"metadata": map[string]interface{}{
			"name": name,
		},
		"rules": []map[string]interface{}{
			{
				"apiGroups": []string{""},
				"resources": []string{"pods"},
				"verbs":     []string{"get", "list", "watch"},
			},
		},
	}
	return makeRequest("POST", "/apis/rbac.authorization.k8s.io/v1/clusterroles", clusterRoleYAML)
}

func createClusterRoleBinding(name, clusterRoleName string) (*http.Response, error) {
	clusterRoleBindingYAML := map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "ClusterRoleBinding",
		"metadata": map[string]interface{}{
			"name": name,
		},
		"roleRef": map[string]interface{}{
			"kind":     "ClusterRole",
			"name":     clusterRoleName,
			"apiGroup": "rbac.authorization.k8s.io",
		},
		"subjects": []map[string]interface{}{
			{
				"kind":     "User",
				"name":     "test-user",
				"apiGroup": "rbac.authorization.k8s.io",
			},
		},
	}
	return makeRequest("POST", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings", clusterRoleBindingYAML)
}

func createServiceAccount(name, namespace string) (*http.Response, error) {
	serviceAccountYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ServiceAccount",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
	}
	return makeRequest("POST", "/api/v1/namespaces/"+namespace+"/serviceaccounts", serviceAccountYAML)
}

func createNamespace(name string) (*http.Response, error) {
	namespaceYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "Namespace",
		"metadata": map[string]interface{}{
			"name": name,
		},
	}
	return makeRequest("POST", "/api/v1/namespaces", namespaceYAML)
}

func createResourceQuota(name, namespace string) (*http.Response, error) {
	resourceQuotaYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ResourceQuota",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"hard": map[string]interface{}{
				"requests.cpu":    "2",
				"requests.memory": "2Gi",
				"limits.cpu":      "4",
				"limits.memory":   "4Gi",
				"pods":            "10",
			},
		},
	}
	return makeRequest("POST", "/api/v1/namespaces/"+namespace+"/resourcequotas", resourceQuotaYAML)
}

func createLimitRange(name, namespace string) (*http.Response, error) {
	limitRangeYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "LimitRange",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"limits": []map[string]interface{}{
				{
					"type": "Container",
					"max": map[string]interface{}{
						"cpu":    "2",
						"memory": "1Gi",
					},
					"min": map[string]interface{}{
						"cpu":    "100m",
						"memory": "4Mi",
					},
					"default": map[string]interface{}{
						"cpu":    "300m",
						"memory": "200Mi",
					},
				},
			},
		},
	}
	return makeRequest("POST", "/api/v1/namespaces/"+namespace+"/limitranges", limitRangeYAML)
}

func createPersistentVolume(name string) (*http.Response, error) {
	pvYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "PersistentVolume",
		"metadata": map[string]interface{}{
			"name": name,
		},
		"spec": map[string]interface{}{
			"capacity": map[string]interface{}{
				"storage": "10Gi",
			},
			"accessModes": []string{
				"ReadWriteOnce",
			},
			"persistentVolumeReclaimPolicy": "Retain",
			"storageClassName":              "manual",
			"hostPath": map[string]interface{}{
				"path": "/tmp/data",
			},
		},
	}
	return makeRequest("POST", "/api/v1/persistentvolumes", pvYAML)
}

func createPersistentVolumeClaim(name, namespace string) (*http.Response, error) {
	pvcYAML := map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "PersistentVolumeClaim",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"accessModes": []string{
				"ReadWriteOnce",
			},
			"resources": map[string]interface{}{
				"requests": map[string]interface{}{
					"storage": "5Gi",
				},
			},
		},
	}
	return makeRequest("POST", "/api/v1/namespaces/"+namespace+"/persistentvolumeclaims", pvcYAML)
}

func createStorageClass(name string) (*http.Response, error) {
	scYAML := map[string]interface{}{
		"apiVersion": "storage.k8s.io/v1",
		"kind":       "StorageClass",
		"metadata": map[string]interface{}{
			"name": name,
		},
		"provisioner":          "kubernetes.io/no-provisioner",
		"reclaimPolicy":        "Retain",
		"volumeBindingMode":    "Immediate",
		"allowVolumeExpansion": true,
	}
	return makeRequest("POST", "/apis/storage.k8s.io/v1/storageclasses", scYAML)
}

func createCustomResourceDefinition(name string) (*http.Response, error) {
	crdYAML := map[string]interface{}{
		"apiVersion": "apiextensions.k8s.io/v1",
		"kind":       "CustomResourceDefinition",
		"metadata": map[string]interface{}{
			"name": strings.ToLower(name) + "s.example.com",
		},
		"spec": map[string]interface{}{
			"group": "example.com",
			"versions": []map[string]interface{}{
				{
					"name":    "v1",
					"served":  true,
					"storage": true,
					"schema": map[string]interface{}{
						"openAPIV3Schema": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"spec": map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"targets": map[string]interface{}{
											"type":                         "array",
											"x-kubernetes-patch-strategy":  "merge",
											"x-kubernetes-patch-merge-key": "name",
											"items": map[string]interface{}{
												"type": "object",
												"properties": map[string]interface{}{
													"name": map[string]interface{}{
														"type": "string",
													},
													"port": map[string]interface{}{
														"type":    "integer",
														"minimum": 0,
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			"scope": "Namespaced",
			"names": map[string]interface{}{
				"plural":   strings.ToLower(name) + "s",
				"singular": strings.ToLower(name),
				"kind":     name,
			},
		},
	}
	return makeRequest("POST", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions", crdYAML)
}

func createCustomResource(name, namespace, crdName string) (*http.Response, error) {
	crYAML := map[string]interface{}{
		"apiVersion": "example.com/v1",
		"kind":       crdName,
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"targets": []map[string]interface{}{
				{
					"name": "test-target",
					"port": 8080,
				},
			},
		},
	}
	return makeRequest("POST", "/apis/example.com/v1/namespaces/"+namespace+"/"+strings.ToLower(crdName)+"s", crYAML)
}
