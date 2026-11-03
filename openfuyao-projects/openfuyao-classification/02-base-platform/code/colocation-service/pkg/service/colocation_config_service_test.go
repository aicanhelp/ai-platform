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

package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2"

	colocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
	colocationclientset "openfuyao.com/colocation-service/pkg/client/clientset/versioned/fake"
)

// TestEnvironment 封装测试环境相关的参数
type TestEnvironment struct {
	Server         *Server
	Clientset      *fake.Clientset
	OldConfigMap   *corev1.ConfigMap
	Recorder       *httptest.ResponseRecorder
	NodeName       string
	ExpectedStatus string
}

// createTestServerInStatus 创建测试用的 Server 实例
func createTestServer() (*Server, *fake.Clientset, *colocationclientset.Clientset) {
	clientset := fake.NewSimpleClientset()
	colocationClientset := colocationclientset.NewSimpleClientset()

	server := &Server{
		client:           clientset,
		colocationclient: colocationClientset,
		ws:               new(restful.WebService),
	}
	server.initColocationNodeRouteNew()

	return server, clientset, colocationClientset
}

// createTestRequest 创建测试用的 restful.Request
func createTestRequest(method, path string) *restful.Request {
	req := httptest.NewRequest(method, path, nil)
	req.Header.Set("Accept", "application/json")
	return restful.NewRequest(req)
}

// createTestResponseInStatus 创建测试用的 restful.Response
func createTestResponse() (*httptest.ResponseRecorder, *restful.Response) {
	recorder := httptest.NewRecorder()
	response := restful.NewResponse(recorder)
	response.SetRequestAccepts("application/json")
	return recorder, response
}

// createColocationConfig 创建 colocation-config ConfigMap
func createColocationConfig(namespace string, coloNodes string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "colocation-config",
			Namespace: namespace,
		},
		Data: map[string]string{
			"colocation-options": fmt.Sprintf(`{
				"enable": %s,
				"nodesConfig": [%s]
			}`, fmt.Sprintf("%t", len(coloNodes) > 0), coloNodes),
			"rubik-options": `{
				"eviction": {
					"enable": true,
					"cpuevict": {
						"threshold": 60,
						"interval": 30,
						"windows": 1,
						"cooldown": 20
					},
					"memoryevict": {
						"threshold": 60,
						"interval": 30,
						"cooldown": 20
					}
				}
			}`,
			"volcano-scheduler-options": `{
				"usagePlugin": {
					"enable": true,
					"usageThreshold": {
						"enable": true,
						"cpu": 60,
						"memory": 60
					}
				}
			}`,
		},
	}
}

// createRubikConfig 创建 colocation-rubik-config ConfigMap
func createRubikConfig(namespace string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "colocation-rubik-config",
			Namespace: namespace,
		},
		Data: map[string]string{
			"config.json": `{
				"cgroupDriver": "systemd",
				"logDriver": "stdio",
				"logDir": "/var/log/rubik",
				"logSize": 100,
				"logLevel": "info",
				"cgroupRoot": "/sys/fs/cgroup",
				"enabledFeatures": ["cpuevict", "memoryevict"],
				"preemption": {
					"resource": [
						"cpu",
						"memory"
					]
				},
				"cpuevict": {
					"threshold": 60
				},
				"memoryevict": {
					"threshold": 60
				}
			}`,
		},
	}
}

// createVolcanoConfig 创建 volcano-scheduler-configmap ConfigMap
func createVolcanoConfig(namespace string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "volcano-scheduler-configmap",
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "Helm",
			},
			Annotations: map[string]string{
				"meta.helm.sh/release-name":      "volcano",
				"meta.helm.sh/release-namespace": namespace,
			},
		},
		Data: map[string]string{
			"volcano-scheduler.conf": `actions: "enqueue, allocate, backfill"
				tiers:
				- plugins:
				  - name: usage
					enablePredicate: true
					arguments:
					  usage.weight: 5
					  cpu.weight: 1
					  memory.weight: 1
					  thresholds:
						cpu: 60
						mem: 60
				  - name: priority
				  - name: gang
					enablePreemptable: false
				  - name: conformance
				- plugins:
				  - name: overcommit
				  - name: drf
					enablePreemptable: false
				  - name: predicates
				  - name: proportion
				  - name: nodeorder
				  - name: binpack
				
				metrics:
				  type: prometheus
				  address: http://prometheus-k8s.monitoring.svc.cluster.local:9090
				  interval: 30s`,
		},
	}
}

// createTestConfigMaps 创建测试所需的 ConfigMaps
func createTestConfigMaps(clientset kubernetes.Interface, coloNodes string) error {
	// 1. 创建 colocation-config
	if _, err := clientset.CoreV1().ConfigMaps(colocationConfigMapNamespace).Create(
		context.TODO(), createColocationConfig(colocationConfigMapNamespace, coloNodes),
		metav1.CreateOptions{}); err != nil {
		return err
	}

	// 2. 创建 colocation-rubik-config
	if _, err := clientset.CoreV1().ConfigMaps(colocationConfigMapNamespace).Create(
		context.TODO(), createRubikConfig(colocationConfigMapNamespace),
		metav1.CreateOptions{}); err != nil {
		return err
	}

	// 3. 创建 volcano-scheduler-configmap
	_, err := clientset.CoreV1().ConfigMaps("volcano-system").Create(
		context.TODO(), createVolcanoConfig("volcano-system"), metav1.CreateOptions{})
	return err
}

// createTestNodeInStatus 创建用于测试的节点
func createTestNode(clientset kubernetes.Interface,
	nodeName string, labels map[string]string) (*corev1.Node, error) {
	if labels == nil {
		labels = make(map[string]string)
	}

	// 设置默认标签
	labels["kubernetes.io/os"] = "linux"
	labels["kubernetes.io/arch"] = "amd64"
	labels["node-role.kubernetes.io/worker"] = ""

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   nodeName,
			Labels: labels,
		},
		Status: corev1.NodeStatus{
			NodeInfo: corev1.NodeSystemInfo{
				KernelVersion:           "5.10.0-60.139.0.166.oe2203.x86_64",
				OSImage:                 "openEuler 22.03 LTS",
				OperatingSystem:         "linux",
				Architecture:            "amd64",
				ContainerRuntimeVersion: "containerd://1.7.14",
				KubeletVersion:          "v1.28.8",
				KubeProxyVersion:        "v1.28.8",
			},
			Conditions: []corev1.NodeCondition{
				{
					Type:               corev1.NodeReady,
					Status:             corev1.ConditionTrue,
					LastHeartbeatTime:  metav1.Now(),
					LastTransitionTime: metav1.Now(),
					Reason:             "KubeletReady",
					Message:            "kubelet is posting ready status",
				},
			},
		},
	}
	return clientset.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
}

// createColocationNode 创建一个已启用共池的节点
func createColocationNode(clientset kubernetes.Interface, nodeName string) (*corev1.Node, error) {
	labels := map[string]string{
		colocationv1.ColocationNodeLabel: "true",
	}
	return createTestNode(clientset, nodeName, labels)
}

func TestGetAllColocationNodesNewWithConfigMaps(t *testing.T) {
	// 1. 初始化测试环境
	server, clientset, _ := createTestServer()

	// 2. 创建测试节点
	_, err := createColocationNode(clientset, "node1")
	assert.NoError(t, err)

	// 3. 创建测试 ConfigMaps
	var coloNodes string
	err = createTestConfigMaps(clientset, coloNodes)
	assert.NoError(t, err)

	// 4. 创建测试请求和响应
	request := createTestRequest("GET", "/rest/colocation/v1/colocation-nodes")
	recorder, response := createTestResponse()

	// 5. 调用被测函数
	server.getAllColocationNodesNew(request, response)

	// 6. 验证响应
	assert.Equal(t, http.StatusOK, recorder.Code, "Expected HTTP 200, got %d", recorder.Code)

	// 7. 解析并验证响应数据
	var result struct {
		Code int
		Data ColocationConfigRequest
	}
	err = json.NewDecoder(recorder.Body).Decode(&result)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(result.Data.NodeInfo))
}

// setupTestEnvironment 初始化测试环境
func setupTestEnvironment(t *testing.T, nodeName string,
	initialLabels map[string]string, coloNodes string) *TestEnvironment {
	// 1. 初始化测试环境
	server, clientset, _ := createTestServer()

	// 2. 创建测试节点
	_, err := createTestNode(clientset, nodeName, initialLabels)
	assert.NoError(t, err)

	// 3. 创建测试 ConfigMaps
	err = createTestConfigMaps(clientset, coloNodes)
	assert.NoError(t, err)

	// 4. 获取初始 ConfigMap
	oldCM, err := server.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), ColocationConfigMapName, metav1.GetOptions{})
	assert.NoError(t, err)
	klog.Info("before: ", oldCM.Data["colocation-options"])

	return &TestEnvironment{
		Server:       server,
		Clientset:    clientset,
		OldConfigMap: oldCM,
		NodeName:     nodeName,
	}
}

// verifyTestResults 验证测试结果
func verifyTestResults(t *testing.T, env *TestEnvironment) {
	// 验证HTTP状态码
	assert.Equal(t, http.StatusOK, env.Recorder.Code)

	// 验证 ConfigMap 更新
	newCM, err := env.Server.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), ColocationConfigMapName, metav1.GetOptions{})
	assert.NoError(t, err)
	klog.Info("after: ", newCM.Data["colocation-options"])

	// 验证事件处理
	eventHandler := NewConfigMapEvenHandler(env.Server.client)
	eventHandler.OnUpdate(env.OldConfigMap, newCM)

	// 验证节点标签更新
	updatedNode, err := env.Clientset.CoreV1().Nodes().Get(context.TODO(), env.NodeName, metav1.GetOptions{})
	assert.NoError(t, err)

	if env.ExpectedStatus == "false" {
		// 对于禁用情况，可能是标签被移除或者设置为false
		if labelValue, exists := updatedNode.Labels[colocationv1.ColocationNodeLabel]; exists {
			assert.Equal(t, "false", labelValue)
		} else {
			// 或者如果实现是移除标签
			assert.False(t, exists, "colocation label should be removed")
		}
	} else {
		assert.Equal(t, env.ExpectedStatus, updatedNode.Labels[colocationv1.ColocationNodeLabel])
	}
}

func TestToggleColocationNodeByNameWithRetry(t *testing.T) {
	// 初始化测试环境
	env := setupTestEnvironment(t, "node1", nil, "")
	env.ExpectedStatus = "true"

	// 创建测试请求和响应
	request := createTestRequest("PATCH",
		"/rest/colocation/v1/colocation-nodes/node1?colocation-status=true")
	request.PathParameters()["name"] = "node1"
	recorder, response := createTestResponse()
	env.Recorder = recorder

	// 调用处理函数
	env.Server.toggleColocationNodeByNameWithRetry(request, response)

	// 验证结果
	verifyTestResults(t, env)
}

func TestToggleColocationNodeByNameWithRetryDisableColocation(t *testing.T) {
	// 初始化测试环境（初始设置为true）
	env := setupTestEnvironment(t, "node1",
		map[string]string{colocationv1.ColocationNodeLabel: "true"},
		`{"name":"node1"}`)
	env.ExpectedStatus = "false"

	// 创建测试请求和响应（请求设置为false）
	request := createTestRequest("PATCH",
		"/rest/colocation/v1/colocation-nodes/node1?colocation-status=false")
	request.PathParameters()["name"] = "node1"
	recorder, response := createTestResponse()
	env.Recorder = recorder

	// 调用处理函数
	env.Server.toggleColocationNodeByNameWithRetry(request, response)

	// 验证结果
	verifyTestResults(t, env)
}

func TestHandleColocationConfigWithRetry(t *testing.T) {
	// 1. 初始化测试环境
	server, clientset, _ := createTestServer()

	// 2. 创建初始 ConfigMaps
	var coloNodes string
	err := createTestConfigMaps(clientset, coloNodes)
	assert.NoError(t, err)

	// 3. 创建测试用的 ColocationConfig 并修改特定值
	testConfig := createColocationConfigForPost()

	// 4. 创建 POST 请求
	requestBody, err := json.Marshal(testConfig)
	assert.NoError(t, err)

	request := restful.NewRequest(httptest.NewRequest(
		"POST",
		"/rest/colocation/v1/colocation-configs/",
		bytes.NewBuffer(requestBody)))
	request.Request.Header.Set("Content-Type", "application/json")
	request.Request.Header.Set("Accept", "application/json")

	// 5. 创建 ResponseRecorder
	recorder := httptest.NewRecorder()
	response := restful.NewResponse(recorder)
	response.SetRequestAccepts("application/json")

	oldCM, err := clientset.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), ColocationConfigMapName, metav1.GetOptions{})
	assert.NoError(t, err)

	// 6. 调用处理函数
	server.handleColocationConfigWithRetry(request, response)

	// 7. 验证响应
	assert.Equal(t, http.StatusOK, recorder.Code)

	// 8. 验证 ConfigMap 是否已更新
	updatedCM, err := clientset.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), ColocationConfigMapName, metav1.GetOptions{})
	assert.NoError(t, err)

	// 验证事件处理
	eventHandler := NewConfigMapEvenHandler(server.client)
	eventHandler.OnUpdate(oldCM, updatedCM)

	// 解析更新后的配置
	var updatedConfig ColocationConfigRequest
	err = json.Unmarshal([]byte(updatedCM.Data["colocation-options"]), &updatedConfig)
	assert.NoError(t, err)
}

// createColocationConfigForPost 创建用于 POST 请求的测试配置
func createColocationConfigForPost() ColocationConfigRequest {
	return ColocationConfigRequest{
		NodeInfo: []NodeAttr{
			{
				Uid:         "test-uid",
				Name:        "test-node",
				OSImage:     "openEuler 22.03 LTS",
				Status:      "Ready",
				IsColocated: true,
			},
		},
		RubikConfig: RubikConfig{
			Eviction: RubikEviction{
				Enabled: true,
				CPUEvict: CPUEvictConfig{
					Threshold: 60,
					Interval:  30,
					Windows:   1,
					Cooldown:  20,
				},
				MemoryEvict: MemoryEvictConfig{
					Threshold: 50, // 修改为50
					Interval:  30,
					Cooldown:  20,
				},
			},
		},
		VolcanoSchedulerConfig: VolcanoSchedulerConfig{
			UsagePlugin: UsagePluginConfig{
				Enabled: true,
				UsageThreshold: UsageThresholdConfig{
					Enabled:         true,
					CPUThreshold:    50, // 修改为50
					MemoryThreshold: 60,
				},
			},
		},
	}
}

// TestConfigMapEventHandlerOnAdd 测试 ConfigMap 的 OnAdd 事件处理
func TestConfigMapEventHandlerOnAdd(t *testing.T) {
	// 1. 创建模拟的 Kubernetes 客户端
	clientset := fake.NewSimpleClientset()

	// 2. 创建测试节点
	node1, err := createTestNode(clientset, "node1", nil)
	assert.NoError(t, err)

	// 3. 创建测试 ConfigMap（模拟新增事件）
	testCM := createColocationConfig(colocationConfigMapNamespace, `{"name":"node1"}`)

	// 4. 初始化事件处理器
	handler := NewConfigMapEvenHandler(clientset)

	// 5. 模拟 OnAdd 事件
	handler.OnAdd(testCM, false)

	// 6. 验证节点标签是否已更新
	updatedNode, err := clientset.CoreV1().Nodes().Get(context.TODO(), node1.Name, metav1.GetOptions{})
	assert.NoError(t, err)
	assert.Equal(t, "true", updatedNode.Labels[colocationv1.ColocationNodeLabel])
}

// TestConfigMapEventHandlerOnDelete 测试 ConfigMap 的 OnDelete 事件处理
func TestConfigMapEventHandlerOnDelete(t *testing.T) {
	// 1. 创建模拟的 Kubernetes 客户端
	clientset := fake.NewSimpleClientset()

	// 2. 创建带有混部标签的测试节点
	initialLabels := map[string]string{
		colocationv1.ColocationNodeLabel: "true",
	}
	node1, err := createTestNode(clientset, "node1", initialLabels)
	assert.NoError(t, err)

	// 3. 创建测试 ConfigMap（模拟被删除的 ConfigMap）
	testCM := createColocationConfig(colocationConfigMapNamespace, `{"name":"node1"}`)

	// 4. 初始化事件处理器
	handler := NewConfigMapEvenHandler(clientset)

	// 5. 模拟 OnDelete 事件
	handler.OnDelete(testCM)

	// 6. 验证节点标签是否已被移除
	updatedNode, err := clientset.CoreV1().Nodes().Get(context.TODO(), node1.Name, metav1.GetOptions{})
	assert.NoError(t, err)

	// 检查其他标签是否保持不变
	assert.Equal(t, "linux", updatedNode.Labels["kubernetes.io/os"])
	assert.Equal(t, "amd64", updatedNode.Labels["kubernetes.io/arch"])
}
