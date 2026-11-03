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
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	kt "k8s.io/client-go/testing"
	"k8s.io/klog/v2"

	colocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
)

// 定义测试常量
const (
	colocationLabel = colocationv1.ColocationNodeLabel
)

// createTestClient 创建测试客户端
func createTestClient(initialConfig *v1.ConfigMap) *fake.Clientset {
	if initialConfig != nil {
		return fake.NewSimpleClientset(initialConfig)
	}
	return fake.NewSimpleClientset()
}

// createTestServerWithClient 创建使用指定客户端的测试服务器
func createTestServerWithClient(t *testing.T, client *fake.Clientset) *Server {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	fakeColocationClient := &fakeColocationClient{}
	server, err := New(ctx, client, fakeColocationClient)
	require.NoError(t, err)
	require.NotNil(t, server)

	return server
}

// createBaseConfigMap 创建基础测试数据
func createBaseConfigMap() *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ColocationConfigMapName,
			Namespace: colocationConfigMapNamespace,
		},
		Data: map[string]string{
			"colocation-options": `{"nodes": ["node1", "node2"]}`,
		},
	}
}

// modifyColocationConfigTestCase 测试用例结构体
type modifyColocationConfigTestCase struct {
	name          string
	coloNode      []string
	operation     string
	initialConfig *v1.ConfigMap
	expectedError bool
	expectedNodes []string
	setupClient   func(*fake.Clientset)
}

// createConfigMapNotExistsTestCase 创建 ConfigMap 不存在的测试用例
func createConfigMapNotExistsTestCase() modifyColocationConfigTestCase {
	return modifyColocationConfigTestCase{
		name:          "configmap does not exist",
		coloNode:      []string{"node3"},
		operation:     "add",
		initialConfig: nil,
		expectedError: true,
		expectedNodes: nil,
	}
}

// createInvalidJSONTestCase 创建无效 JSON 的测试用例
func createInvalidJSONTestCase() modifyColocationConfigTestCase {
	return modifyColocationConfigTestCase{
		name:      "invalid json in configmap",
		coloNode:  []string{"node3"},
		operation: "add",
		initialConfig: &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      ColocationConfigMapName,
				Namespace: colocationConfigMapNamespace,
			},
			Data: map[string]string{
				"colocation-options": `{"nodes": ["node1", "node2"`,
			},
		},
		expectedError: true,
		expectedNodes: nil,
	}
}

// createSuccessfulAddTestCase 创建成功添加操作的测试用例
func createSuccessfulAddTestCase(baseConfigMap *v1.ConfigMap) modifyColocationConfigTestCase {
	return modifyColocationConfigTestCase{
		name:          "successful add operation",
		coloNode:      []string{"node3"},
		operation:     "add",
		initialConfig: baseConfigMap,
		expectedError: false,
		expectedNodes: []string{"node1", "node2", "node3"},
	}
}

// createSuccessfulDeleteTestCase 创建成功删除操作的测试用例
func createSuccessfulDeleteTestCase(baseConfigMap *v1.ConfigMap) modifyColocationConfigTestCase {
	return modifyColocationConfigTestCase{
		name:          "successful delete operation",
		coloNode:      []string{"node1"},
		operation:     "delete",
		initialConfig: baseConfigMap,
		expectedError: false,
		expectedNodes: []string{"node2"},
	}
}

// createInvalidOperationTestCase 创建无效操作的测试用例
func createInvalidOperationTestCase(baseConfigMap *v1.ConfigMap) modifyColocationConfigTestCase {
	return modifyColocationConfigTestCase{
		name:          "invalid operation",
		coloNode:      []string{"node3"},
		operation:     "invalid",
		initialConfig: baseConfigMap,
		expectedError: true,
		expectedNodes: nil,
	}
}

// createUpdateFailureTestCase 创建更新失败的测试用例
func createUpdateFailureTestCase(baseConfigMap *v1.ConfigMap) modifyColocationConfigTestCase {
	return modifyColocationConfigTestCase{
		name:          "simulate update failure with custom reactor",
		coloNode:      []string{"node3"},
		operation:     "add",
		initialConfig: baseConfigMap,
		expectedError: true,
		expectedNodes: nil,
		setupClient:   createUpdateFailureReactor,
	}
}

// createUpdateFailureReactor 创建更新失败的反应器
func createUpdateFailureReactor(client *fake.Clientset) {
	client.PrependReactor("update", "configmaps",
		func(action kt.Action) (handled bool, ret runtime.Object, err error) {
			return true, nil, errors.New("simulated update error")
		})
}

// runModifyColocationConfigTestCase 执行单个测试用例
func runModifyColocationConfigTestCase(t *testing.T, tt modifyColocationConfigTestCase) {
	client := createTestClient(tt.initialConfig)

	if tt.setupClient != nil {
		tt.setupClient(client)
	}

	server := createTestServerWithClient(t, client)
	err := server.modifyColocationConfig(tt.coloNode, tt.operation)

	verifyModifyColocationConfigTestResult(t, tt, err)
}

// verifyModifyColocationConfigTestResult 验证测试结果
func verifyModifyColocationConfigTestResult(t *testing.T, tt modifyColocationConfigTestCase, err error) {
	if tt.expectedError {
		assert.Error(t, err, "Expected error but got none")
	} else {
		assert.NoError(t, err, "Unexpected error: %v", err)
	}
}

// TestModifyColocationConfig 主测试函数
func TestModifyColocationConfig(t *testing.T) {
	// 基础测试数据 - 初始 ConfigMap
	baseConfigMap := createBaseConfigMap()

	tests := []modifyColocationConfigTestCase{
		createConfigMapNotExistsTestCase(),
		createInvalidJSONTestCase(),
		createSuccessfulAddTestCase(baseConfigMap),
		createSuccessfulDeleteTestCase(baseConfigMap),
		createInvalidOperationTestCase(baseConfigMap),
		createUpdateFailureTestCase(baseConfigMap),
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runModifyColocationConfigTestCase(t, tt)
		})
	}
}

// TestHandleDeleteOperation 主测试函数
func TestHandleDeleteOperation(t *testing.T) {
	// 禁用 klog 输出以避免干扰测试输出
	klog.LogToStderr(false)

	// 定义测试节点配置
	node1 := NodeConfig{Name: "node1"}

	tests := []struct {
		name           string
		coloNode       []string
		initialOptions ColocationOptions
		expectedNodes  []NodeConfig
		expectedEnable bool
	}{
		// 删除不存在的节点
		{
			name:     "delete non-existent node",
			coloNode: []string{"node2"},
			initialOptions: ColocationOptions{
				Enable:      true,
				NodesConfig: []NodeConfig{node1},
			},
			expectedNodes:  []NodeConfig{node1},
			expectedEnable: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 执行被测试的函数
			result := handleDeleteOperation(tt.coloNode, tt.initialOptions)

			// 验证节点配置
			assert.ElementsMatch(t, tt.expectedNodes, result.NodesConfig, "Node list doesn't match expected")

			// 验证启用状态
			assert.Equal(t, tt.expectedEnable, result.Enable, "Enable flag doesn't match expected")
		})
	}
}

// updateNodeColocationStatus 测试用例结构体
type updateNodeTestCase struct {
	name          string
	nodeInfos     []NodeAttr
	initialNodes  []*v1.Node            // 初始节点状态
	setupClient   func(*fake.Clientset) // 客户端设置
	expectedError bool
	errContains   string
}

// runUpdateNodeColocationStatusTestCase 执行单个测试用例
func runUpdateNodeColocationStatusTestCase(t *testing.T, tt updateNodeTestCase) {
	client := fake.NewSimpleClientset()
	for _, node := range tt.initialNodes {
		_, err := client.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
		require.NoError(t, err)
	}

	if tt.setupClient != nil {
		tt.setupClient(client)
	}

	server := createTestServerWithClient(t, client)

	err := server.updateNodeColocationStatus(tt.nodeInfos)

	verifyUpdateNodeColocationStatusTestResult(t, tt, err)
	if !tt.expectedError {
		verifyNodeLabels(t, client, tt.nodeInfos)
	}
}

// verifyUpdateNodeColocationStatusTestResult 验证测试结果
func verifyUpdateNodeColocationStatusTestResult(t *testing.T, tt updateNodeTestCase, err error) {
	if tt.expectedError {
		assert.Error(t, err, "Expected error but got none")
		if tt.errContains != "" {
			assert.Contains(t, err.Error(), tt.errContains, "Error message should contain expected text")
		}
	} else {
		assert.NoError(t, err, "Unexpected error: %v", err)
	}
}

// verifyNodeLabels 辅助函数，验证节点标签是否正确更新
func verifyNodeLabels(t *testing.T, client *fake.Clientset, nodeInfos []NodeAttr) {
	for _, nodeInfo := range nodeInfos {
		node, err := client.CoreV1().Nodes().Get(context.TODO(), nodeInfo.Name, metav1.GetOptions{})
		if err != nil {
			continue
		}

		// 检查节点标签是否符合预期
		hasLabel := isColocationNode(node)

		if nodeInfo.IsColocated {
			assert.True(t, hasLabel, "Node %s should have colocation label", nodeInfo.Name)
		} else {
			assert.False(t, hasLabel, "Node %s should not have colocation label", nodeInfo.Name)
		}
	}
}

// TestUpdateNodeColocationStatus 主测试函数
func TestUpdateNodeColocationStatus(t *testing.T) {
	klog.LogToStderr(false)

	// 创建测试节点
	nodeWithLabel := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-with-label",
			Labels: map[string]string{
				colocationLabel: "true",
			},
		},
		Status: v1.NodeStatus{
			NodeInfo: v1.NodeSystemInfo{KernelVersion: "5.10.0"},
		},
	}
	nodeWithoutLabel := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-without-label",
			Labels: map[string]string{},
		},
		Status: v1.NodeStatus{
			NodeInfo: v1.NodeSystemInfo{KernelVersion: "5.10.0"},
		},
	}
	nodeNotSupport := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-not-support",
			Labels: map[string]string{},
		},
		Status: v1.NodeStatus{
			NodeInfo: v1.NodeSystemInfo{KernelVersion: "4.10.0"},
		},
	}

	// 创建测试 NodeAttr 对象
	nodeAttrWithColocationTrue := NodeAttr{
		Name:        "node-with-label",
		IsColocated: true,
	}
	nodeAttrWithColocationFalse := NodeAttr{
		Name:        "node-with-label",
		IsColocated: false,
	}
	nodeAttrWithoutColocationTrue := NodeAttr{
		Name:        "node-without-label",
		IsColocated: true,
	}

	tests := []updateNodeTestCase{
		// 成功场景
		{
			name: "no change",
			nodeInfos: []NodeAttr{
				nodeAttrWithColocationTrue,
			},
			initialNodes:  []*v1.Node{nodeWithLabel},
			expectedError: false,
		},
		{
			name: "successfully append label to node",
			nodeInfos: []NodeAttr{
				nodeAttrWithoutColocationTrue,
			},
			initialNodes:  []*v1.Node{nodeWithoutLabel},
			expectedError: false,
		},
		{
			name: "successfully delete label from node",
			nodeInfos: []NodeAttr{
				nodeAttrWithColocationFalse,
			},
			initialNodes:  []*v1.Node{nodeWithLabel},
			expectedError: false,
		},

		// 错误场景
		{
			name: "node not support",
			nodeInfos: []NodeAttr{
				{Name: "node-not-support", IsColocated: true},
			},
			initialNodes:  []*v1.Node{nodeNotSupport},
			expectedError: true,
			errContains:   "does not support colocation",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runUpdateNodeColocationStatusTestCase(t, tt)
		})
	}
}

// featureTestCase 特性测试用例结构体
type featureTestCase struct {
	name        string
	config      RubikConfig
	expectedErr bool
	errContains string
}

// prepareQuotaTurboTestCases 准备 QuotaTurbo 测试用例
func prepareQuotaTurboTestCases() []featureTestCase {
	return []featureTestCase{
		{
			name: "QuotaTurbo highWaterMark too low",
			config: RubikConfig{
				QuotaTurbo: &QuotaTurboConfig{
					Enable:         true,
					HighWaterMark:  QuotaTurboMin - 1,
					AlarmWaterMark: 80,
				},
			},
			expectedErr: true,
			errContains: "highWaterMark must be between",
		},
		{
			name: "QuotaTurbo alarmWaterMark too high",
			config: RubikConfig{
				QuotaTurbo: &QuotaTurboConfig{
					Enable:         true,
					HighWaterMark:  60,
					AlarmWaterMark: QuotaTurboMax + 1,
				},
			},
			expectedErr: true,
			errContains: "alarmWaterMark must be between",
		},
	}
}

// prepareDynCacheTestCases 准备 DynCache 测试用例
func prepareDynCacheTestCases() []featureTestCase {
	return []featureTestCase{
		{
			name: "valid DynCache configuration with L3Percent",
			config: RubikConfig{
				DynCache: &DynCacheConfig{
					Enable: true,
					L3Percent: &CachePercentConfig{
						Low:  10,
						Mid:  50,
						High: 90,
					},
				},
			},
			expectedErr: false,
		},
		{
			name: "valid DynCache configuration with MemBandPercent",
			config: RubikConfig{
				DynCache: &DynCacheConfig{
					Enable: true,
					MemBandPercent: &CachePercentConfig{
						Low:  10,
						Mid:  50,
						High: 90,
					},
				},
			},
			expectedErr: false,
		},
		{
			name: "DynCache with invalid L3Percent high value",
			config: RubikConfig{
				DynCache: &DynCacheConfig{
					Enable: true,
					L3Percent: &CachePercentConfig{
						Low:  10,
						Mid:  50,
						High: 190,
					},
				},
			},
			expectedErr: true,
			errContains: "l3Percent.high must be between",
		},
		{
			name: "DynCache with invalid MemBandPercent",
			config: RubikConfig{
				DynCache: &DynCacheConfig{
					Enable: true,
					MemBandPercent: &CachePercentConfig{
						Low:  10,
						Mid:  50,
						High: 190,
					},
				},
			},
			expectedErr: true,
			errContains: "memBandPercent.high must be between",
		},
	}
}

// preparePSITestCases 准备 PSI 测试用例
func preparePSITestCases() []featureTestCase {
	return []featureTestCase{
		{
			name: "PSI avg10Threshold too high",
			config: RubikConfig{
				PSI: &PSIConfig{
					Enable:         true,
					Avg10Threshold: PSIThresholdMax + 1.0,
					Resource:       []string{"cpu"},
				},
			},
			expectedErr: true,
			errContains: "avg10Threshold must be between",
		},
		{
			name: "PSI with invalid resource type",
			config: RubikConfig{
				PSI: &PSIConfig{
					Enable:         true,
					Avg10Threshold: 50.0,
					Resource:       []string{"cpu", "invalid", "memory"},
				},
			},
			expectedErr: true,
			errContains: "invalid psi resource type",
		},
		{
			name: "PSI with no resource field",
			config: RubikConfig{
				PSI: &PSIConfig{
					Enable:         true,
					Avg10Threshold: 50.0,
				},
			},
			expectedErr: true,
			errContains: "psi.resource cannot be empty",
		},
	}
}

// runAllFeatureTests 运行所有特性测试
func runAllFeatureTests(t *testing.T, testGroups ...[]featureTestCase) {
	featureNames := []string{"QuotaTurbo", "DynCache", "PSI"}

	for i, tests := range testGroups {
		if i < len(featureNames) {
			runFeatureTestSuite(t, featureNames[i], tests)
		}
	}
}

// runFeatureTestSuite 运行特定特性测试套件
func runFeatureTestSuite(t *testing.T, featureName string, tests []featureTestCase) {
	t.Run(featureName+" Features", func(t *testing.T) {
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				runSingleFeatureTest(t, tt)
			})
		}
	})
}

// runSingleFeatureTest 运行单个特性测试
func runSingleFeatureTest(t *testing.T, tt featureTestCase) {
	server := Server{}
	err := server.validateRubikHighFeatures(tt.config)
	verifyFeatureTestError(t, tt, err)
}

// verifyFeatureTestError 验证特性测试错误
func verifyFeatureTestError(t *testing.T, tt featureTestCase, err error) {
	if tt.expectedErr {
		assert.Error(t, err, "Expected error but got none")
		if tt.errContains != "" {
			assert.Contains(t, err.Error(), tt.errContains,
				"Error message should contain expected text")
		}
	} else {
		assert.NoError(t, err, "Unexpected error: %v", err)
	}
}

// TestValidateRubikHighFeatures 主测试函数
func TestValidateRubikHighFeatures(t *testing.T) {
	quotaTurboTests := prepareQuotaTurboTestCases()
	dynCacheTests := prepareDynCacheTestCases()
	psiTests := preparePSITestCases()

	runAllFeatureTests(t, quotaTurboTests, dynCacheTests, psiTests)
}

// cachePercentTestCase 测试用例结构体
type cachePercentTestCase struct {
	name        string
	config      CachePercentConfig
	fieldName   string
	expectedErr bool
	errContains string
}

func createCachePercentTestCase() []cachePercentTestCase {
	tests := []cachePercentTestCase{
		// Low 字段验证测试用例
		{
			name: "low value below minimum",
			config: CachePercentConfig{
				Low:  CachePercentMin - 1,
				Mid:  50,
				High: 90,
			},
			fieldName:   "testField",
			expectedErr: true,
			errContains: "testField.low must be between",
		},
		// Mid 字段验证测试用例
		{
			name: "mid value below low",
			config: CachePercentConfig{
				Low:  50,
				Mid:  49, // 小于 Low
				High: 90,
			},
			fieldName:   "testField",
			expectedErr: true,
			errContains: "testField.mid must be between",
		},
		// High 字段验证测试用例
		{
			name: "high value above maximum",
			config: CachePercentConfig{
				Low:  10,
				Mid:  50,
				High: CachePercentMax + 1,
			},
			fieldName:   "testField",
			expectedErr: true,
			errContains: "testField.high must be between",
		},
	}
	return tests
}

// TestValidateCachePercent 主测试函数
func TestValidateCachePercent(t *testing.T) {
	tests := createCachePercentTestCase()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 执行被测试的函数
			err := validateCachePercent(tt.config, tt.fieldName)

			// 验证错误预期
			if tt.expectedErr {
				assert.Error(t, err, "Expected error but got none")
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains, "Error message should contain expected text")
				}
			} else {
				assert.NoError(t, err, "Unexpected error: %v", err)
			}
		})
	}
}

type coverColoConfigTestCase struct {
	name           string
	newConfig      ColocationConfigRequest
	initialConfig  *v1.ConfigMap         // 初始 ConfigMap 状态
	setupClient    func(*fake.Clientset) // 客户端设置
	expectedError  bool
	expectedErrMsg string // 期望的错误消息内容
}

func createColoConfigRubikTestCase(invalidCPURubikConfig RubikConfig, baseConfigMap *v1.ConfigMap,
	validRubikConfig RubikConfig) []coverColoConfigTestCase {
	tests := []coverColoConfigTestCase{
		// 错误场景
		{
			name: "CPU threshold too high",
			newConfig: ColocationConfigRequest{
				RubikConfig: invalidCPURubikConfig,
			},
			initialConfig:  baseConfigMap,
			expectedError:  true,
			expectedErrMsg: "rubik CPU threshold must be between",
		},
		{
			name: "Memory threshold too high",
			newConfig: ColocationConfigRequest{
				RubikConfig: RubikConfig{
					Eviction: RubikEviction{
						Enabled: true,
						CPUEvict: CPUEvictConfig{
							Threshold: 80,
						},
						MemoryEvict: MemoryEvictConfig{
							Threshold: RubikThresholdMax + 1,
						},
					},
				},
			},
			initialConfig:  baseConfigMap,
			expectedError:  true,
			expectedErrMsg: "rubik Memory threshold must be between",
		},
		{
			name: "validateRubikHighFeatures fails",
			newConfig: ColocationConfigRequest{
				RubikConfig: RubikConfig{
					Eviction: RubikEviction{
						Enabled: true,
						CPUEvict: CPUEvictConfig{
							Threshold: 80,
						},
						MemoryEvict: MemoryEvictConfig{
							Threshold: 85,
						},
					},
					QuotaTurbo: &QuotaTurboConfig{
						Enable:         true,
						HighWaterMark:  60,
						AlarmWaterMark: 50,
					},
				},
			},
			initialConfig:  baseConfigMap,
			expectedError:  true,
			expectedErrMsg: "quotaTurbo.highWaterMark must be between",
		},
		{
			name: "configmap not found",
			newConfig: ColocationConfigRequest{
				RubikConfig: validRubikConfig,
			},
			initialConfig:  nil, // 不创建 ConfigMap
			expectedError:  true,
			expectedErrMsg: "not found", // ConfigMap 不存在
		},
		{
			name: "update configmap fails",
			newConfig: ColocationConfigRequest{
				RubikConfig: validRubikConfig,
			},
			initialConfig: baseConfigMap,
			setupClient: func(client *fake.Clientset) {
				// 模拟更新失败
				client.PrependReactor("update", "configmaps", func(action kt.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, errors.New("failed to update configmap")
				})
			},
			expectedError:  true,
			expectedErrMsg: "failed to update configmap",
		},
	}
	return tests
}

// runCoverColoConfigTestCase 执行单个测试用例
func runCoverColoConfigTestCase(t *testing.T, tt coverColoConfigTestCase) {
	client := createTestClient(tt.initialConfig)

	if tt.setupClient != nil {
		tt.setupClient(client)
	}

	server := createTestServerWithClient(t, client)
	err := server.coverColoConfigRubik(tt.newConfig)

	verifyCoverColoConfigTestResult(t, tt, err)

	// 对于成功场景，验证 ConfigMap 是否正确更新
	if !tt.expectedError && tt.initialConfig != nil {
		verifyConfigMapUpdated(t, client, tt.newConfig.RubikConfig)
	}
}

// verifyCoverColoConfigTestResult 验证测试结果
func verifyCoverColoConfigTestResult(t *testing.T, tt coverColoConfigTestCase, err error) {
	// 验证错误预期
	if tt.expectedError {
		assert.Error(t, err, "Expected error but got none")
		if tt.expectedErrMsg != "" {
			assert.Contains(t, err.Error(), tt.expectedErrMsg, "Error message should contain expected text")
		}
	} else {
		assert.NoError(t, err, "Unexpected error: %v", err)
	}
}

// TestCoverColoConfigRubik 主测试函数
func TestCoverColoConfigRubik(t *testing.T) {
	klog.LogToStderr(false)

	// 创建基础配置映射
	baseConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ColocationConfigMapName,
			Namespace: colocationConfigMapNamespace,
		},
		Data: map[string]string{
			"rubik-options": `{"eviction":{"enabled":false}}`,
		},
	}

	// 创建有效的 Rubik 配置
	validRubikConfig := RubikConfig{
		Eviction: RubikEviction{
			Enabled: true,
			CPUEvict: CPUEvictConfig{
				Threshold: 80,
			},
			MemoryEvict: MemoryEvictConfig{
				Threshold: 85,
			},
		},
	}

	// 创建无效的 Rubik 配置（CPU阈值超出范围）
	invalidCPURubikConfig := RubikConfig{
		Eviction: RubikEviction{
			Enabled: true,
			CPUEvict: CPUEvictConfig{
				Threshold: RubikThresholdMax + 1, // 超出最大值
			},
			MemoryEvict: MemoryEvictConfig{
				Threshold: 85,
			},
		},
	}

	tests := createColoConfigRubikTestCase(invalidCPURubikConfig, baseConfigMap, validRubikConfig)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runCoverColoConfigTestCase(t, tt)
		})
	}
}

// verifyConfigMapUpdated 辅助函数，验证 ConfigMap 是否正确更新
func verifyConfigMapUpdated(t *testing.T, client *fake.Clientset, expectedConfig RubikConfig) {
	// 获取更新后的 ConfigMap
	configMap, err := client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), ColocationConfigMapName, metav1.GetOptions{})
	require.NoError(t, err)

	// 解析配置
	var actualConfig RubikConfig
	err = json.Unmarshal([]byte(configMap.Data["rubik-options"]), &actualConfig)
	require.NoError(t, err, "Failed to parse updated config")

	// 验证配置是否正确更新
	assert.Equal(t, expectedConfig.Eviction.Enabled, actualConfig.Eviction.Enabled)
	if expectedConfig.Eviction.Enabled {
		assert.Equal(t, expectedConfig.Eviction.CPUEvict.Threshold, actualConfig.Eviction.CPUEvict.Threshold)
		assert.Equal(t, expectedConfig.Eviction.MemoryEvict.Threshold, actualConfig.Eviction.MemoryEvict.Threshold)
	}
}

func createColoConfigVolcanoTestCase(invalidCPUVolcanoConfig VolcanoSchedulerConfig, baseConfigMap *v1.ConfigMap,
	invalidMemoryVolcanoConfig VolcanoSchedulerConfig, validVolcanoConfig VolcanoSchedulerConfig) []coverColoConfigTestCase {
	tests := []coverColoConfigTestCase{
		// 错误场景
		{
			name: "CPU threshold too high",
			newConfig: ColocationConfigRequest{
				VolcanoSchedulerConfig: invalidCPUVolcanoConfig,
			},
			initialConfig:  baseConfigMap,
			expectedError:  true,
			expectedErrMsg: "volcano scheduler CPU threshold must be between",
		},
		{
			name: "Memory threshold too low",
			newConfig: ColocationConfigRequest{
				VolcanoSchedulerConfig: invalidMemoryVolcanoConfig,
			},
			initialConfig:  baseConfigMap,
			expectedError:  true,
			expectedErrMsg: "volcano scheduler memory threshold must be between",
		},
		{
			name: "configmap not found",
			newConfig: ColocationConfigRequest{
				VolcanoSchedulerConfig: validVolcanoConfig,
			},
			initialConfig:  nil, // 不创建 ConfigMap
			expectedError:  true,
			expectedErrMsg: "not found", // ConfigMap 不存在
		},
		{
			name: "update configmap fails",
			newConfig: ColocationConfigRequest{
				VolcanoSchedulerConfig: validVolcanoConfig,
			},
			initialConfig: baseConfigMap,
			setupClient: func(client *fake.Clientset) {
				// 模拟更新失败
				client.PrependReactor("update", "configmaps", func(action kt.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, errors.New("failed to update volcanoConfigmap")
				})
			},
			expectedError:  true,
			expectedErrMsg: "failed to update volcanoConfigmap",
		},
	}
	return tests
}

// runCoverColoConfigTestCase 执行单个测试用例
func runCoverColoConfigVolcanoTestCase(t *testing.T, tt coverColoConfigTestCase) {
	client := createTestClient(tt.initialConfig)

	if tt.setupClient != nil {
		tt.setupClient(client)
	}

	server := createTestServerWithClient(t, client)
	err := server.coverColoConfigVolcanoScheduler(tt.newConfig)

	verifyCoverColoConfigTestResult(t, tt, err)

	// 对于成功场景，验证 ConfigMap 是否正确更新
	if !tt.expectedError && tt.initialConfig != nil {
		verifyConfigMapUpdated(t, client, tt.newConfig.RubikConfig)
	}
}

// TestCoverColoConfigVolcanoScheduler 主测试函数
func TestCoverColoConfigVolcanoScheduler(t *testing.T) {
	klog.LogToStderr(false)

	// 创建基础配置映射
	baseConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ColocationConfigMapName,
			Namespace: colocationConfigMapNamespace,
		},
		Data: map[string]string{
			"volcano-scheduler-options": `{"usagePlugin":{"enabled":false}}`,
		},
	}
	// 创建有效的 Volcano 调度器配置
	validVolcanoConfig := VolcanoSchedulerConfig{
		UsagePlugin: UsagePluginConfig{
			Enabled: true,
			UsageThreshold: UsageThresholdConfig{
				Enabled:         true,
				CPUThreshold:    80.0,
				MemoryThreshold: 85.0,
			},
		},
	}
	// 创建无效的 Volcano 配置（CPU阈值超出范围）
	invalidCPUVolcanoConfig := VolcanoSchedulerConfig{
		UsagePlugin: UsagePluginConfig{
			Enabled: true,
			UsageThreshold: UsageThresholdConfig{
				Enabled:         true,
				CPUThreshold:    VolcanoThresholdMax + 1.0, // 超出最大值
				MemoryThreshold: 85.0,
			},
		},
	}

	// 创建无效的 Volcano 配置（内存阈值超出范围）
	invalidMemoryVolcanoConfig := VolcanoSchedulerConfig{
		UsagePlugin: UsagePluginConfig{
			Enabled: true,
			UsageThreshold: UsageThresholdConfig{
				Enabled:         true,
				CPUThreshold:    80.0,
				MemoryThreshold: PercentageMin - 1.0, // 低于最小值
			},
		},
	}

	tests := createColoConfigVolcanoTestCase(invalidCPUVolcanoConfig, baseConfigMap,
		invalidMemoryVolcanoConfig, validVolcanoConfig)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runCoverColoConfigVolcanoTestCase(t, tt)
		})
	}
}

func TestAddFeatureToEnabledList(t *testing.T) {
	handler := &ConfigMapEvenHandler{}
	features := []string{"a", "b"}
	handler.addFeatureToEnabledList(&features, "c")
	assert.Contains(t, features, "c")
	handler.addFeatureToEnabledList(&features, "a")
	assert.Equal(t, 3, len(features))
}

func TestRemoveFeatureFromEnabledList(t *testing.T) {
	handler := &ConfigMapEvenHandler{}
	features := []string{"a", "b", "c"}
	handler.removeFeatureFromEnabledList(&features, "b")
	assert.NotContains(t, features, "b")
	assert.Equal(t, 2, len(features))
}
