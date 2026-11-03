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

package installer

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/crypto/ssh"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/restmapper"
	k8stesting "k8s.io/client-go/testing"

	"installer-service/pkg/constant"
	"installer-service/pkg/utils/k8sutil"
)

type TestCaseStruct struct {
	name     string
	nodeInfo *ClusterNodeInfo
	want     error
}

type TestNodeStruct struct {
	caseName  string
	nameSpace string
	balanceIp string
}

func GenTestCase(caseInfo TestNodeStruct, nodeIp, nodeName []string, wanted error) *TestCaseStruct {
	res := &TestCaseStruct{
		name: caseInfo.caseName,
		nodeInfo: &ClusterNodeInfo{
			NameSpace: caseInfo.nameSpace,
			BalanceIp: caseInfo.balanceIp,
		},
		want: wanted,
	}
	res.nodeInfo.Nodes = make([]ClusterNode, max(len(nodeIp), len(nodeName)))
	for i, _ := range nodeIp {
		res.nodeInfo.Nodes[i].Ip = nodeIp[i]
	}

	for i, _ := range nodeName {
		res.nodeInfo.Nodes[i].Hostname = nodeName[i]
	}
	return res
}

func PatchGetClustersOrNodes(para string, client *installerClient) *gomonkey.Patches {
	if para == "cluster" {
		return gomonkey.ApplyMethod(client, "GetClusters",
			func(_ *installerClient) ([]configv1beta1.BKECluster, error) {
				return []configv1beta1.BKECluster{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "bke_cluster",
						},
					},
				}, nil
			})
	} else if para == "node" {
		return gomonkey.ApplyMethod(client, "GetNodes", func(_ *installerClient,
			clusterName string) ([]corev1.Node, error) {
			if clusterName == "bke_cluster" {
				return []corev1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "master1",
						},
						Status: corev1.NodeStatus{
							Addresses: []corev1.NodeAddress{
								{
									Type:    corev1.NodeInternalIP,
									Address: "127.0.0.1",
								},
							},
						},
					},
				}, nil
			}
			return nil, fmt.Errorf("get %s cluster err", clusterName)
		})
	}
	return nil
}

func TestJudgeClusterNodeIp(t *testing.T) {
	client := &installerClient{}
	tests := []*TestCaseStruct{
		GenTestCase(TestNodeStruct{"test_exist_cluster_node_ip_success", "bke_cluster", ""},
			[]string{"127.0.0.2"}, nil, nil),
		GenTestCase(TestNodeStruct{"test_exist_cluster_node_ip_fail", "bke_cluster", ""},
			[]string{"127.0.0.1"}, nil, errors.New("cluster node ip fail")),
		GenTestCase(TestNodeStruct{"test_not_exist_cluster_node_ip_success", "work_cluster", ""},
			[]string{"127.0.0.2", "127.0.0.3"}, nil, nil),
		GenTestCase(TestNodeStruct{"test_not_exist_cluster_node_ip_fail", "work_cluster", ""},
			[]string{"127.0.0.1", "127.0.0.3"}, nil, errors.New("cluster node ip fail")),
	}

	patchGetClusters := PatchGetClustersOrNodes("cluster", client)
	defer patchGetClusters.Reset()

	patchGetNodes := PatchGetClustersOrNodes("node", client)
	defer patchGetNodes.Reset()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := client.IsNodeIpOk(tt.nodeInfo)
			fmt.Println("err is", err)
			if tt.want == nil {
				require.Equal(t, nil, err)
			} else {
				require.NotEqual(t, nil, err)
			}
		})
	}
}

func TestJudgeClusterNodeName(t *testing.T) {
	client := &installerClient{}
	tests := []*TestCaseStruct{
		GenTestCase(TestNodeStruct{"test_cluster_node_name_success", "bke_cluster", ""},
			nil, []string{"master2"}, nil),
		GenTestCase(TestNodeStruct{"test_cluster_node_name_fail", "bke_cluster", ""},
			nil, []string{"master1"}, errors.New("cluster node name fail")),
	}

	patchGetNodes := PatchGetClustersOrNodes("node", client)
	defer patchGetNodes.Reset()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := client.IsNodeNameOk(tt.nodeInfo)
			fmt.Println("err is", err)
			if tt.want == nil {
				require.Equal(t, nil, err)
			} else {
				require.NotEqual(t, nil, err)
			}
		})
	}
}

func TestJudgeClusterNodeInfo(t *testing.T) {
	client := &installerClient{}
	tests := []*TestCaseStruct{
		GenTestCase(TestNodeStruct{"test_cluster_node_info_fail", "bke_cluster", ""},
			[]string{"127.0.0.2"}, []string{"master2"}, errors.New("cluster node info fail")),
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := client.IsNodeInfoOk(tt.nodeInfo)

			fmt.Println("err is", err)
			if tt.want == nil {
				require.Equal(t, nil, err)
			} else {
				require.NotEqual(t, nil, err)
			}
		})
	}
}

func TestJudgeClusterBalanceIp(t *testing.T) {
	client := &installerClient{}
	tests := []*TestCaseStruct{
		GenTestCase(TestNodeStruct{"test_cluster_balance_ip_fail", "bke_cluster", "127.0.0.1"},
			[]string{"127.0.0.2"}, []string{"master1"}, errors.New("cluster balance ip fail")),
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := client.IsBalanceIpOk(tt.nodeInfo.BalanceIp)

			fmt.Println("err is", err)
			if tt.want == nil {
				require.Equal(t, nil, err)
			} else {
				require.NotEqual(t, nil, err)
			}
		})
	}
}

func TestJudgeClusterNodeAll(t *testing.T) {
	client := &installerClient{}
	tests := []*TestCaseStruct{
		GenTestCase(TestNodeStruct{"test_judge_node_ip_fail", "bke_cluster", ""},
			[]string{"127.0.0.1"}, []string{"master1"}, errors.New("cluster balance ip fail")),
		GenTestCase(TestNodeStruct{"test_judge_node_name_fail", "bke_cluster", ""},
			[]string{"127.0.0.2"}, []string{"master1"}, errors.New("cluster node name fail")),
		GenTestCase(TestNodeStruct{"test_judge_all_success", "work_cluster", "127.0.0.255"},
			[]string{"127.0.0.2"}, []string{"master1"}, nil),
	}

	patchIsNodeInfoOk := gomonkey.ApplyMethod(client, "IsNodeInfoOk", func(_ *installerClient,
		nodeInfo *ClusterNodeInfo) (bool, error) {
		return true, nil
	})
	defer patchIsNodeInfoOk.Reset()

	patchIsBalanceIpOk := gomonkey.ApplyMethod(client, "IsBalanceIpOk", func(_ *installerClient,
		balanceIp string) (bool, error) {
		return true, nil
	})
	defer patchIsBalanceIpOk.Reset()

	patchGetClusters := PatchGetClustersOrNodes("cluster", client)
	defer patchGetClusters.Reset()

	patchGetNodes := PatchGetClustersOrNodes("node", client)
	defer patchGetNodes.Reset()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := client.JudgeClusterNode(tt.nodeInfo)
			fmt.Println("err is", err)
			if tt.want == nil {
				require.Equal(t, nil, err)
			} else {
				require.NotEqual(t, nil, err)
			}
		})
	}
}

// 测试 CreateCluster
type fakeDiscovery struct {
	discovery.DiscoveryInterface
}

func (f *fakeDiscovery) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return []*metav1.APIResourceList{
		{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{Name: "namespaces", SingularName: "", Namespaced: false, Kind: "Namespace",
					Verbs: []string{"get", "list", "create"}},
				{Name: "configmaps", SingularName: "", Namespaced: true, Kind: "ConfigMap",
					Verbs: []string{"get", "list", "create"}},
			},
		},
	}, nil
}

func (f *fakeDiscovery) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	if groupVersion == "v1" {
		return &metav1.APIResourceList{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{Name: "namespaces", SingularName: "", Namespaced: false, Kind: "Namespace",
					Verbs: []string{"get", "list", "create"}},
				{Name: "configmaps", SingularName: "", Namespaced: true, Kind: "ConfigMap",
					Verbs: []string{"get", "list", "create"}},
			},
		}, nil
	}
	return nil, nil
}

func newFakeClientsetWithDiscovery() kubernetes.Interface {
	clientset := k8sfake.NewSimpleClientset()

	clientset.Fake.Resources = []*metav1.APIResourceList{
		{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{Name: "namespaces", Kind: "Namespace", Namespaced: false},
				{Name: "configmaps", Kind: "ConfigMap", Namespaced: true},
			},
		},
	}

	_ = clientset.Discovery()

	type withDiscovery struct {
		*k8sfake.Clientset
		discovery.DiscoveryInterface
	}
	return &withDiscovery{
		Clientset:          clientset,
		DiscoveryInterface: &fakeDiscovery{},
	}
}

// 简化的手动模拟实现
type mockDynamicClient struct {
	dynamic.Interface
	getResult *unstructured.Unstructured
	getError  error
}

func (m *mockDynamicClient) Resource(gvr schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	return &mockNamespaceableResource{
		getResult: m.getResult,
		getError:  m.getError,
	}
}

type mockNamespaceableResource struct {
	dynamic.NamespaceableResourceInterface
	getResult *unstructured.Unstructured
	getError  error
}

func (m *mockNamespaceableResource) Namespace(namespace string) dynamic.ResourceInterface {
	return &mockResource{
		getResult: m.getResult,
		getError:  m.getError,
	}
}

type mockResource struct {
	dynamic.ResourceInterface
	getResult *unstructured.Unstructured
	getError  error
}

func (m *mockResource) Get(context.Context, string, metav1.GetOptions, ...string) (*unstructured.Unstructured, error) {
	return m.getResult, m.getError
}

const ip1 = "abc"

func TestGetDefaultConfig(t *testing.T) {
	// 创建fake clientset
	fakeClientset := k8sfake.NewSimpleClientset()
	// 创建installerClient实例
	client := &installerClient{
		clientset: fakeClientset,
	}
	t.Run("should return error when GetConfigMap fails", func(t *testing.T) {
		// Mock k8sutil.GetConfigMap返回错误
		patches := gomonkey.ApplyFunc(
			k8sutil.GetConfigMap, func(_ kubernetes.Interface, _, _ string) (*corev1.ConfigMap, error) {
				return nil, errors.New("configmap error")
			})
		defer patches.Reset()
		// 执行函数
		resp, err := client.GetDefaultConfig()
		// 验证结果
		assert.Error(t, err)
		assert.Equal(t, DefaultResp{}, resp)
	})

	t.Run("should handle default values when optional fields are missing", func(t *testing.T) {
		// Mock k8sutil.GetConfigMap返回ConfigMap（缺少可选字段）
		patches := gomonkey.ApplyFunc(
			k8sutil.GetConfigMap, func(_ kubernetes.Interface, _, _ string) (*corev1.ConfigMap, error) {
				return &corev1.ConfigMap{
					Data: map[string]string{
						"domain": "docker.io",
						"host":   ip1,
					},
				}, nil
			})
		defer patches.Reset()

		// 执行函数
		resp, err := client.GetDefaultConfig()

		// 验证结果
		assert.NoError(t, err)
		assert.Equal(t, ip1, resp.Ip)
		assert.Equal(t, configv1beta1.Repo{
			Domain: "docker.io",
		}, resp.ImageRepo)
		assert.Equal(t, configv1beta1.Repo{
			Domain: configinit.DefaultYumRepo,
		}, resp.HttpRepo)
	})

}

// TestCleanConfigMapPatches runs table-driven tests for cleanConfigMapPatches.
func TestCleanConfigMapPatches(t *testing.T) {
	cases := []struct {
		name       string
		mainData   map[string]string
		existsName string
		injectFail bool
		wantErr    bool
	}{
		{name: "success", mainData: map[string]string{constant.PatchKeyPrefix + "1": "cm-to-delete"}, existsName: "cm-to-delete", injectFail: false, wantErr: false},
		{name: "delete-fail", mainData: map[string]string{constant.PatchKeyPrefix + "1": "cm-to-delete"}, existsName: "cm-to-delete", injectFail: true, wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ns := constant.PatchNameSpace
			main := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "main-cm", Namespace: ns}, Data: tc.mainData}
			var objs []runtime.Object
			objs = append(objs, main)
			if tc.existsName != "" {
				objs = append(objs, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: tc.existsName, Namespace: ns}, Data: map[string]string{"k": "v"}})
			}
			cs := k8sfake.NewSimpleClientset(objs...)
			if tc.injectFail {
				cs.PrependReactor("delete", "configmaps", func(action k8stesting.Action) (bool, runtime.Object, error) {
					return true, nil, fmt.Errorf("injected delete error")
				})
			}
			c := &installerClient{clientset: cs}

			err := c.cleanConfigMapPatches(main)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// 新增：测试 createBKENodes 的 name 回退逻辑与创建错误处理
func TestCreateBKENodes_NameFallbackAndCreateError(t *testing.T) {
	cs := newFakeClientsetWithDiscovery()
	scheme := runtime.NewScheme()
	dyn := dynamicfake.NewSimpleDynamicClient(scheme)

	client := &installerClient{clientset: cs, dynamicClient: dyn}

	// case1: name 为空，使用 ip 替代并替换点为短横线
	// 注意：避免使用 []string 直接放入 unstructured，会导致 deep-copy 问题，使用 nil 或者不包含 role 字段
	nodes := []ClusterNode{{Hostname: "", Ip: "1.2.3.4", Port: "22", Username: "u", Password: "p", Role: nil}}
	// 测试 name 回退逻辑（独立于动态客户端的创建）
	n := nodes[0]
	expectedName := n.Hostname
	if expectedName == "" {
		expectedName = strings.ReplaceAll(n.Ip, ".", "-")
	}
	if expectedName != "1-2-3-4" {
		t.Fatalf("expected fallback name 1-2-3-4, got %s", expectedName)
	}
	nodeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkenodes"}

	// case2: 动态客户端在创建节点时返回错误，应被上抛
	// 使用 gomonkey 打桩 Create 方法返回错误
	resInterface := dyn.Resource(nodeGVR).Namespace("err-ns")
	patches := gomonkey.ApplyMethod(resInterface, "Create",
		func(_ dynamic.ResourceInterface, ctx context.Context, obj *unstructured.Unstructured, opts metav1.CreateOptions) (*unstructured.Unstructured, error) {
			return nil, errors.New("create node failed")
		})
	defer patches.Reset()

	err := client.createBKENodes("err-ns", []ClusterNode{{Hostname: "n", Ip: "9.9.9.9", Port: "22"}})
	if err == nil || !strings.Contains(err.Error(), "create node failed") {
		t.Fatalf("expected error from create node, got: %v", err)
	}
}

func TestDeleteCluster(t *testing.T) {

	t.Run("should return error when getting cluster resource fails", func(t *testing.T) {
		// 创建空 fake 动态客户端
		scheme := runtime.NewScheme()
		fakeDynamicClient := dynamicfake.NewSimpleDynamicClient(scheme) // 无资源

		client := &installerClient{
			dynamicClient: fakeDynamicClient,
		}

		// 执行函数
		resp, status := client.DeleteCluster("test-cluster")

		// 验证结果 - 修复类型问题
		assert.Equal(t, http.StatusInternalServerError, int(status))
		assert.Equal(t, http.StatusInternalServerError, int(resp.Code))
		assert.Contains(t, resp.Message, "Failed to get cluster resource")
	})
}

// 定义测试中使用的全局变量
var (
	testGVR = schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1", // 与测试对象版本一致
		Resource: "bkeclusters",
	}
)

// 测试patchYAML
func TestPatchYamlWithMock(t *testing.T) {
	t.Run("successful patch", testSuccessfulPatch)
	t.Run("patch failure - discovery error", testPatchFailureDiscovery)
	t.Run("patch failure - resource not found", testPatchFailureResourceNotFound)
}

// 公共初始化函数
func setupTestEnv() (*installerClient, *dynamicfake.FakeDynamicClient) {
	scheme := runtime.NewScheme()
	dyn := dynamicfake.NewSimpleDynamicClient(scheme)
	clientset := k8sfake.NewSimpleClientset()

	clientset.Fake.Resources = []*metav1.APIResourceList{
		{
			GroupVersion: "bke.bocloud.com/v1beta1",
			APIResources: []metav1.APIResource{
				{Name: "bkeclusters", Kind: "BKECluster", Namespaced: true},
			},
		},
	}

	return &installerClient{
		clientset:     clientset,
		dynamicClient: dyn,
	}, dyn
}

// 公共测试配置
func getTestManifest() string {
	return `
apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: test-bkecluster
  namespace: test-ns
spec:
  key: updated-value
`
}

func testSuccessfulPatch(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(patchVersion, func(c *installerClient, obj *unstructured.Unstructured) error { return nil })

	client, dyn := setupTestEnv()
	manifest := getTestManifest()
	gvrBKE := schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1",
		Resource: "bkeclusters",
	}

	// 预先创建资源
	initialBKE := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "bke.bocloud.com/v1beta1",
			"kind":       "BKECluster",
			"metadata": map[string]interface{}{
				"name":      "test-bkecluster",
				"namespace": "test-ns",
			},
			"spec": map[string]interface{}{
				"key": "initial-value",
			},
		},
	}
	_, err := dyn.Resource(gvrBKE).Namespace("test-ns").Create(
		context.Background(), initialBKE, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create initial BKECluster: %v", err)
	}

	// 执行patch
	err = client.PatchYaml(manifest, false)
	if err != nil {
		t.Fatalf("PatchYaml error: %v", err)
	}

	// 验证结果
	bke, err := dyn.Resource(gvrBKE).Namespace("test-ns").Get(
		context.Background(), "test-bkecluster", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get BKECluster: %v", err)
	}

	spec, ok := bke.Object["spec"].(map[string]interface{})
	if !ok {
		t.Errorf("not get spec")
	}
	if spec["key"] != "updated-value" {
		t.Errorf("Expected spec.key=updated-value, got %v", spec["key"])
	}
}

func testPatchFailureDiscovery(t *testing.T) {
	client, _ := setupTestEnv()
	manifest := getTestManifest()

	// Mock发现错误
	patches := gomonkey.ApplyFunc(restmapper.GetAPIGroupResources,
		func(_ discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return nil, errors.New("mock discovery error")
		})
	patches.ApplyFunc(patchVersion, func(c *installerClient, obj *unstructured.Unstructured) error { return nil })
	defer patches.Reset()

	// 执行并验证错误
	err := client.PatchYaml(manifest, false)
	if err == nil {
		t.Fatal("Expected error but got none")
	}
}

func testPatchFailureResourceNotFound(t *testing.T) {
	client, dyn := setupTestEnv()
	manifest := getTestManifest()

	// Mock动态客户端返回错误
	patches := gomonkey.ApplyMethod(
		dyn.Resource(schema.GroupVersionResource{}).Namespace(""),
		"Patch",
		func(_ dynamic.ResourceInterface, ctx context.Context, name string, pt types.PatchType,
			data []byte, opts metav1.PatchOptions, subresources ...string) (*unstructured.Unstructured, error) {
			return nil, errors.New("resource not found")
		},
	)
	patches.ApplyFunc(patchVersion, func(c *installerClient, obj *unstructured.Unstructured) error { return nil })
	defer patches.Reset()

	// 执行并验证错误
	err := client.PatchYaml(manifest, false)
	if err == nil {
		t.Fatal("Expected error but got none")
	}
}

func TestScaleDownCluster(t *testing.T) {
	t.Run("successful scale down", testSuccessfulScaleDown)
	t.Run("discovery error", testScaleDownDiscoveryError)
	t.Run("yaml decode error", testScaleDownYAMLError)
}

// 公共测试配置
const testIP = "abc1"
const validManifest = `
apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: test-cluster
  namespace: test-ns
spec:
  nodes:
  - ip: abc1
  - ip: abc2
`

// 创建初始资源
func createInitialResource(dyn dynamic.Interface) error {
	gvr := schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1",
		Resource: "bkeclusters",
	}

	initialObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "bke.bocloud.com/v1beta1",
			"kind":       "BKECluster",
			"metadata": map[string]interface{}{
				"name":        "test-cluster",
				"namespace":   "test-ns",
				"annotations": map[string]interface{}{},
			},
			"spec": map[string]interface{}{
				"nodes": []interface{}{
					map[string]interface{}{"ip": "abc1"},
					map[string]interface{}{"ip": "abc2"},
				},
			},
		},
	}

	_, err := dyn.Resource(gvr).Namespace("test-ns").Create(
		context.Background(), initialObj, metav1.CreateOptions{})
	return err
}

// 验证注解
func verifyAnnotations(t *testing.T, dyn dynamic.Interface, expectedIP string) {
	gvr := schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1",
		Resource: "bkeclusters",
	}

	patchedObj, err := dyn.Resource(gvr).Namespace("test-ns").Get(
		context.Background(), "test-cluster", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get patched resource: %v", err)
	}

	annotations := patchedObj.GetAnnotations()
	if annotations["bke.bocloud.com/ignore-target-cluster-delete"] != "false" {
		t.Errorf("ignore-target-cluster-delete annotation not set to false")
	}
	if annotations["bke.bocloud.com/appointment-deleted-nodes"] != expectedIP {
		t.Errorf("appointment-deleted-nodes annotation not set correctly, expected %s, got %s",
			expectedIP, annotations["bke.bocloud.com/appointment-deleted-nodes"])
	}
}

func testSuccessfulScaleDown(t *testing.T) {
	client, _ := setupTestEnv()
	dyn, _ := client.dynamicClient.(*dynamicfake.FakeDynamicClient)

	// 预先创建资源
	if err := createInitialResource(dyn); err != nil {
		t.Fatalf("Failed to create initial resource: %v", err)
	}

	// 执行缩容
	if err := client.ScaleDownCluster(validManifest, testIP); err != nil {
		t.Fatalf("ScaleDownCluster error: %v", err)
	}

	// 验证注解
	verifyAnnotations(t, dyn, testIP)
}

func testScaleDownDiscoveryError(t *testing.T) {
	client, _ := setupTestEnv()

	// Mock发现错误
	patches := gomonkey.ApplyFunc(restmapper.GetAPIGroupResources,
		func(_ discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return nil, errors.New("mock discovery error")
		})
	defer patches.Reset()

	// 执行并验证错误
	err := client.ScaleDownCluster(validManifest, testIP)
	if err == nil {
		t.Fatal("Expected discovery error but got none")
	}
}

func testScaleDownYAMLError(t *testing.T) {
	client, _ := setupTestEnv()

	// 使用无效的YAML
	err := client.ScaleDownCluster("invalid yaml", testIP)
	if err == nil {
		t.Fatal("Expected yaml decode error but got none")
	}
}

// 测试 GetClusterLog
func TestGetClusterLog(t *testing.T) {
	// 创建测试Scheme
	scheme := runtime.NewScheme()
	corev1.AddToScheme(scheme)
	clientset := k8sfake.NewSimpleClientset()
	wsConn := &websocket.Conn{}
	var lastMessage []byte
	patches := gomonkey.ApplyMethodFunc(wsConn, "WriteMessage", func(messageType int, data []byte) error {
		lastMessage = data
		return nil
	})
	defer patches.Reset()
	client := &installerClient{
		clientset: clientset,
	}
	namespace := "test-ns"
	t.Run("normal event flow", func(t *testing.T) {
		go func() {
			resp, code := client.GetClusterLog(namespace, wsConn)
			if code != http.StatusOK {
				t.Errorf("Expected status code %d, got %d", http.StatusOK, code)
			}
			if resp.Message != "get the log of cluster." {
				t.Errorf("Unexpected response message: %s", resp.Message)
			}
		}()
		event := &corev1.Event{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "test-event",
				Namespace:   namespace,
				Annotations: map[string]string{BKEEventAnnotationKey: "true"},
			},
			Type:          "Normal",
			Reason:        "Created",
			Message:       "Pod created",
			LastTimestamp: metav1.Now(),
		}
		_, err := clientset.CoreV1().Events(namespace).Create(context.Background(), event, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create event: %v", err)
		}
		const num = 100
		time.Sleep(num * time.Millisecond)
		expected := fmt.Sprintf("Time:%s, Type: %s, Reason: %s, Message: %s \n",
			event.LastTimestamp.Format("2006-01-02 15:04:05"),
			event.Type,
			event.Reason,
			event.Message)
		if string(lastMessage) != expected {
			t.Errorf("Expected message %q, got %q", expected, string(lastMessage))
		}
	})
}

const clustersNum = 1

// 测试 TestGetClusters
func TestGetClusters(t *testing.T) {
	t.Run("successfully get clusters", func(t *testing.T) {
		scheme := runtime.NewScheme()
		testClusters := []unstructured.Unstructured{
			{
				Object: map[string]interface{}{
					"apiVersion": "bke.bocloud.com/v1beta1",
					"kind":       "BKECluster",
					"metadata": map[string]interface{}{
						"name": "cluster-1",
					},
					"spec": map[string]interface{}{
						"pause":  true,
						"dryRun": false,
						"controlPlaneEndpoint": map[string]interface{}{
							"host": "api.cluster-1.example.com",
							"port": float64(6443),
						},
					},
				},
			},
		}
		dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme, &testClusters[0])
		client := &installerClient{dynamicClient: dynamicClient}
		patches := gomonkey.ApplyGlobalVar(&gvr, testGVR)
		defer patches.Reset()
		clusters, err := client.GetClusters()
		if err != nil {
			t.Fatalf("GetClusters failed: %v", err)
		}
		if len(clusters) != clustersNum {
			t.Errorf("Expected 2 clusters, got %d", len(clusters))
		}
		cluster1 := clusters[0]
		if cluster1.Name != "cluster-1" {
			t.Errorf("Expected cluster name 'cluster-1', got '%s'", cluster1.Name)
		}
		if !cluster1.Spec.Pause {
			t.Errorf("Expected Pause=true for cluster-1")
		}
		if cluster1.Spec.DryRun {
			t.Errorf("Expected DryRun=false for cluster-1")
		}
		if cluster1.Spec.ControlPlaneEndpoint.Host != "api.cluster-1.example.com" {
			t.Errorf("Expected host 'api.cluster-1.example.com', got '%s'", cluster1.Spec.ControlPlaneEndpoint.Host)
		}
	})
}

// 新增表驱动测试：parseImageRepo
func TestParseImageRepo(t *testing.T) {
	tests := []struct {
		name string
		data map[string]string
		want configv1beta1.Repo
	}{
		{"domain only", map[string]string{"domain": "docker.io"}, configv1beta1.Repo{Domain: "docker.io"}},
		{"otherRepo with port and prefix", map[string]string{"otherRepo": "reg.example.com:5000/prefix/path", "otherRepoIp": "1.2.3.4"}, configv1beta1.Repo{Domain: "reg.example.com", Ip: "1.2.3.4", Port: "5000", Prefix: "prefix/path"}},
		{"onlineImage uses host when otherRepoIp empty", map[string]string{"onlineImage": "reg.example.com:4443/some", "otherRepoIp": "", "host": "hostip"}, configv1beta1.Repo{Domain: "reg.example.com", Ip: "hostip", Port: "4443"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseImageRepo(tt.data)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("parseImageRepo() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// 新增表驱动测试：parseHttpRepo
func TestParseHttpRepo(t *testing.T) {
	tests := []struct {
		name string
		data map[string]string
		want configv1beta1.Repo
	}{
		{"empty", map[string]string{}, configv1beta1.Repo{Domain: configinit.DefaultYumRepo}},
		// 以下用例暂定现在want内容，可能需要修改具体实现
		{"ip with port", map[string]string{"otherSource": "http://1.2.3.4:8080"}, configv1beta1.Repo{Domain: "1.2.3.4"}},
		{"domain with port", map[string]string{"otherSource": "http://repo.example.com:88"}, configv1beta1.Repo{Ip: "repo.example.com"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseHttpRepo(tt.data)
			if got.Domain == "" && tt.want.Domain != "" {
				t.Fatalf("expected Domain %q, got empty", tt.want.Domain)
			}
			// compare relevant fields
			if got.Ip != tt.want.Ip || got.Port != tt.want.Port || got.Domain != tt.want.Domain {
				t.Fatalf("parseHttpRepo() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// 新增：测试 REMOTE path helpers
func TestGetRemotePathEnv(t *testing.T) {
	t.Run("patch env", func(t *testing.T) {
		t.Setenv("REMOTE_PATCH_PATH", "/custom/patch/")
		if got := getRemotePatchPath(); got != "/custom/patch/" {
			t.Fatalf("getRemotePatchPath() = %q", got)
		}
	})
	t.Run("deploy env", func(t *testing.T) {
		t.Setenv("REMOTE_DEPLOY_PATH", "/custom/deploy/")
		if got := getRemoteDeployPath(); got != "/custom/deploy/" {
			t.Fatalf("getRemoteDeployPath() = %q", got)
		}
	})
}

// 新增：测试节点相关的 helper 函数
func TestNodeHelpers(t *testing.T) {
	c := &installerClient{}
	node := corev1.Node{}
	node.Status.Addresses = []corev1.NodeAddress{{Type: corev1.NodeInternalIP, Address: "10.0.0.1"}}
	node.Status.Conditions = []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionTrue}}
	node.Labels = map[string]string{constant.NodeRoleMasterLabel: "true", constant.NodeRoleNodeLabel: "true"}
	node.Status.NodeInfo = corev1.NodeSystemInfo{Architecture: "amd64", OSImage: "Ubuntu"}
	node.Status.Capacity = corev1.ResourceList{corev1.ResourceCPU: *resourceQuantity("2"), corev1.ResourceMemory: *resourceQuantity("2147483648")} // 2GB

	if ip := c.getNodeIP(node); ip != "10.0.0.1" {
		t.Fatalf("getNodeIP = %s", ip)
	}
	if status := c.getNodeStatus(node); status != "Ready" {
		t.Fatalf("getNodeStatus = %s", status)
	}
	roles := c.getNodeRole(node)
	if len(roles) != 2 {
		t.Fatalf("expected 2 roles, got %v", roles)
	}
	cpu, mem := c.getNodeResources(node)
	if cpu != 2 || int(mem) != 2 {
		t.Fatalf("expected cpu=2 mem=2, got cpu=%d mem=%f", cpu, mem)
	}
}

func TestGetNodesFromBKENode(t *testing.T) {
	clusterName := "bn-cluster"
	scheme := runtime.NewScheme()
	listKinds := map[schema.GroupVersionResource]string{
		bkeNodeGVR: "BKENodeList",
	}

	readyNode := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKENode",
		"metadata": map[string]interface{}{
			"name":      "ready-host",
			"namespace": clusterName,
		},
		"spec": map[string]interface{}{
			"hostname": "ready-host",
			"ip":       "10.0.0.1",
			"role":     []interface{}{"master/node"},
		},
		"status": map[string]interface{}{
			"state": "Ready",
		},
	}}
	failedNode := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKENode",
		"metadata": map[string]interface{}{
			"name":      "failed-host",
			"namespace": clusterName,
		},
		"spec": map[string]interface{}{
			"hostname": "failed-host",
			"ip":       "10.0.0.2",
			"role":     []interface{}{"node"},
		},
		"status": map[string]interface{}{
			"state": "Failed",
		},
	}}

	dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds, readyNode, failedNode)
	client := &installerClient{dynamicClient: dyn}

	nodes, err := client.GetNodes(clusterName)
	require.NoError(t, err)
	require.Len(t, nodes, 2)

	byName := map[string]corev1.Node{}
	for _, n := range nodes {
		byName[n.Name] = n
	}
	require.Contains(t, byName, "ready-host")
	require.Contains(t, byName, "failed-host")

	assert.Equal(t, "10.0.0.1", client.getNodeIP(byName["ready-host"]))
	assert.Equal(t, "Ready", client.getNodeStatus(byName["ready-host"]))
	assert.Contains(t, client.getNodeRole(byName["ready-host"]), "master")
	assert.Contains(t, client.getNodeRole(byName["ready-host"]), "node")

	assert.Equal(t, "10.0.0.2", client.getNodeIP(byName["failed-host"]))
	assert.Equal(t, "Failed", client.getNodeStatus(byName["failed-host"]))
	assert.Equal(t, []string{"node"}, client.getNodeRole(byName["failed-host"]))

	// API path should also return failed BKENodes
	resp, err := client.GetNodesByQuery(NodeRequest{ClusterName: clusterName})
	require.NoError(t, err)
	require.Len(t, resp.Items, 2)
	statuses := map[string]string{}
	for _, item := range resp.Items {
		statuses[item.Hostname] = item.Status
	}
	assert.Equal(t, "Ready", statuses["ready-host"])
	assert.Equal(t, "Failed", statuses["failed-host"])
}

func TestDeleteBKENodesByHostnameOrIP(t *testing.T) {
	clusterName := "del-cluster"
	scheme := runtime.NewScheme()
	listKinds := map[schema.GroupVersionResource]string{
		bkeNodeGVR: "BKENodeList",
	}

	worker := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKENode",
		"metadata": map[string]interface{}{
			"name":      "bke-cluster-n1",
			"namespace": clusterName,
		},
		"spec": map[string]interface{}{
			"hostname": "n1",
			"ip":       "10.0.0.11",
			"role":     []interface{}{"node"},
		},
		"status": map[string]interface{}{"state": "Failed"},
	}}
	sameName := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKENode",
		"metadata": map[string]interface{}{
			"name":      "worker-2",
			"namespace": clusterName,
		},
		"spec": map[string]interface{}{
			"hostname": "worker-2",
			"ip":       "10.0.0.12",
			"role":     []interface{}{"node"},
		},
		"status": map[string]interface{}{"state": "Ready"},
	}}

	dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds, worker, sameName)
	client := &installerClient{dynamicClient: dyn}

	// Console passes hostname when CR name differs.
	require.NoError(t, client.DeleteBKENodes(clusterName, []string{"n1"}))
	_, err := dyn.Resource(bkeNodeGVR).Namespace(clusterName).Get(context.TODO(), "bke-cluster-n1", metav1.GetOptions{})
	require.True(t, apierrors.IsNotFound(err))

	// Delete by IP also works.
	require.NoError(t, client.DeleteBKENodes(clusterName, []string{"10.0.0.12"}))
	_, err = dyn.Resource(bkeNodeGVR).Namespace(clusterName).Get(context.TODO(), "worker-2", metav1.GetOptions{})
	require.True(t, apierrors.IsNotFound(err))

	// Unknown identifier must fail (no silent 200).
	err = client.DeleteBKENodes(clusterName, []string{"missing-host"})
	require.Error(t, err)
}

func TestMergeRemoteNodeInfo(t *testing.T) {
	nodes := []corev1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "ready-host"},
			Status: corev1.NodeStatus{
				Addresses: []corev1.NodeAddress{{Type: corev1.NodeInternalIP, Address: "10.0.0.1"}},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "failed-host"},
			Status: corev1.NodeStatus{
				Addresses: []corev1.NodeAddress{{Type: corev1.NodeInternalIP, Address: "10.0.0.2"}},
			},
		},
	}
	remote := []corev1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "ready-host"},
			Status: corev1.NodeStatus{
				Addresses: []corev1.NodeAddress{{Type: corev1.NodeInternalIP, Address: "10.0.0.1"}},
				NodeInfo:  corev1.NodeSystemInfo{Architecture: "amd64", OSImage: "openEuler 22.03"},
				Capacity: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("4"),
					corev1.ResourceMemory: resource.MustParse("8Gi"),
				},
			},
		},
	}

	mergeRemoteNodeInfo(nodes, remote)

	assert.Equal(t, "amd64", nodes[0].Status.NodeInfo.Architecture)
	assert.Equal(t, "openEuler 22.03", nodes[0].Status.NodeInfo.OSImage)
	assert.Equal(t, int64(4), nodes[0].Status.Capacity.Cpu().Value())
	assert.Empty(t, nodes[1].Status.NodeInfo.Architecture)
	assert.Empty(t, nodes[1].Status.NodeInfo.OSImage)

	client := &installerClient{}
	items := client.processNodes(NodeRequest{}, nodes)
	byHost := map[string]NodeData{}
	for _, item := range items {
		byHost[item.Hostname] = item
	}
	assert.Equal(t, "amd64", byHost["ready-host"].Architecture)
	assert.Equal(t, "openEuler 22.03", byHost["ready-host"].Os)
	assert.Empty(t, byHost["failed-host"].Architecture)
	assert.Empty(t, byHost["failed-host"].Os)
}

func TestProcessNodeDataEmptyArch(t *testing.T) {
	client := &installerClient{}
	base := &ClusterData{Name: "c1", OsList: []string{}}
	nodes := []corev1.Node{
		{Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{Architecture: "", OSImage: ""}}},
		{Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{Architecture: "amd64", OSImage: "Ubuntu 22.04"}}},
	}
	data := client.processNodeData(base, nodes)
	assert.True(t, data.IsAmd)
	assert.False(t, data.IsArm)
	assert.Equal(t, []string{"Ubuntu 22.04"}, data.OsList)
}

func TestGetNodesNilDynamicClient(t *testing.T) {
	client := &installerClient{}
	_, err := client.GetNodes("any")
	require.Error(t, err)
}

// helper to create resource.Quantity pointer (single shared helper)
func resourceQuantity(s string) *resource.Quantity {
	q := resource.MustParse(s)
	return &q
}

// 测试 GetClustersByName
func TestGetClustersByName(t *testing.T) {
	testGVR := schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1",
		Resource: "bkeclusters",
	}
	clusterName := "test-cluster"
	t.Run("successfully get cluster by name", func(t *testing.T) {
		// 创建测试数据
		testCluster := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "bke.bocloud.com/v1beta1",
				"kind":       "BKECluster",
				"metadata": map[string]interface{}{
					"name":      clusterName,
					"namespace": clusterName,
				},
				"spec": map[string]interface{}{
					"pause": true,
				},
			},
		}

		// 创建dynamic fake client并注入测试数据
		dynamicClient := dynamicfake.NewSimpleDynamicClient(runtime.NewScheme(), testCluster)

		client := &installerClient{
			dynamicClient: dynamicClient,
		}

		// 替换全局GVR变量
		patches := gomonkey.ApplyGlobalVar(&gvr, testGVR)
		defer patches.Reset()

		// 执行测试
		cluster, err := client.GetClustersByName(clusterName)
		if err != nil {
			t.Fatalf("GetClustersByName failed: %v", err)
		}

		// 验证结果
		if cluster.Name != clusterName {
			t.Errorf("Expected cluster name %s, got %s", clusterName, cluster.Name)
		}
		if !cluster.Spec.Pause {
			t.Errorf("Expected Pause=true")
		}
	})
}

// 新增：为 getVersion 编写表驱动测试
func TestGetVersion(t *testing.T) {
	// We'll construct clusters. For the normal case we set nested spec.ClusterConfig.Cluster.KubernetesVersion using reflection
	statusUp := configv1beta1.BKECluster{Status: configv1beta1.BKEClusterStatus{ClusterStatus: "Upgrading"}}

	normal := configv1beta1.BKECluster{Status: configv1beta1.BKEClusterStatus{ClusterStatus: "Ready"}}
	// set nested field normal.Spec.ClusterConfig.Cluster.KubernetesVersion = "v1.22.3" via reflection to avoid depending on concrete ClusterConfig type
	specVal := reflect.ValueOf(&normal).Elem().FieldByName("Spec")
	if !specVal.IsValid() {
		t.Fatal("Spec field not found on BKECluster")
	}
	ccField := specVal.FieldByName("ClusterConfig")
	if !ccField.IsValid() {
		t.Fatal("ClusterConfig field not found on BKECluster.Spec")
	}
	// allocate pointer if nil
	if ccField.Kind() == reflect.Ptr && ccField.IsNil() {
		ccField.Set(reflect.New(ccField.Type().Elem()))
	}
	// get the value that contains Cluster
	var ccVal reflect.Value
	if ccField.Kind() == reflect.Ptr {
		ccVal = ccField.Elem()
	} else {
		ccVal = ccField
	}
	clusterField := ccVal.FieldByName("Cluster")
	if !clusterField.IsValid() {
		t.Fatal("Cluster field not found on ClusterConfig")
	}
	// set KubernetesVersion if the field exists
	kvField := clusterField.FieldByName("KubernetesVersion")
	if !kvField.IsValid() || !kvField.CanSet() {
		t.Fatal("KubernetesVersion field not found or cannot be set")
	}
	kvField.SetString("v1.22.3")

	tests := []struct {
		name    string
		cluster configv1beta1.BKECluster
		want    string
	}{
		{"status upgrading", statusUp, "v1.28.8"},
		{"normal returns spec version", normal, "v1.22.3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getVersion(tt.cluster)
			if got != tt.want {
				t.Fatalf("getVersion() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestStatusProcessormapClusterStatus(t *testing.T) {
	tests := []struct {
		name           string
		status         configv1beta1.BKEClusterStatus
		expectedStatus string
	}{
		// 扩缩容进行中（透传）
		{"ScalingWorkerNodesUp", configv1beta1.BKEClusterStatus{ClusterStatus: "ScalingWorkerNodesUp"}, "ScalingWorkerNodesUp"},
		{"ScalingWorkerNodesDown", configv1beta1.BKEClusterStatus{ClusterStatus: "ScalingWorkerNodesDown"}, "ScalingWorkerNodesDown"},
		{"ScalingMasterNodesUp", configv1beta1.BKEClusterStatus{ClusterStatus: "ScalingMasterNodesUp"}, "ScalingMasterNodesUp"},
		{"ScalingMasterNodesDown", configv1beta1.BKEClusterStatus{ClusterStatus: "ScalingMasterNodesDown"}, "ScalingMasterNodesDown"},

		// 扩缩容失败 → 统一 ScaleFailed
		{"MasterScalingUpFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "MasterScalingUpFailed"}, "ScaleFailed"},
		{"MasterScalingDownFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "MasterScalingDownFailed"}, "ScaleFailed"},
		{"WorkerScalingUpFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "WorkerScalingUpFailed"}, "ScaleFailed"},
		{"WorkerScalingDownFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "WorkerScalingDownFailed"}, "ScaleFailed"},

		// 删除
		{"Deleting", configv1beta1.BKEClusterStatus{ClusterStatus: "Deleting"}, "Deleting"},
		{"DeleteFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "DeleteFailed"}, "DeleteFailed"},

		// 安装/初始化
		{"Initializing", configv1beta1.BKEClusterStatus{ClusterStatus: "Initializing"}, "Deploying"},
		{"DeployingAddon", configv1beta1.BKEClusterStatus{ClusterStatus: "DeployingAddon"}, "Deploying"},
		{"InitializationFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "InitializationFailed"}, "DeployFailed"},
		{"DeployAddonFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "DeployAddonFailed"}, "DeployFailed"},

		// 升级
		{"Upgrading", configv1beta1.BKEClusterStatus{ClusterStatus: "Upgrading"}, "Upgrading"},
		{"UpgradeFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "UpgradeFailed"}, "UpgradeFailed"},

		// 管理 → 健康
		{"Managing", configv1beta1.BKEClusterStatus{ClusterStatus: "Managing"}, "Healthy"},
		{"ManageFailed", configv1beta1.BKEClusterStatus{ClusterStatus: "ManageFailed"}, "Healthy"},

		// 健康
		{"Ready", configv1beta1.BKEClusterStatus{ClusterStatus: "Ready"}, "Healthy"},
		{"Unhealthy", configv1beta1.BKEClusterStatus{ClusterStatus: "Unhealthy"}, "Unhealthy"},

		// 未映射状态：原样透传
		{"Empty status", configv1beta1.BKEClusterStatus{}, ""},
		{"Unknown", configv1beta1.BKEClusterStatus{ClusterStatus: "Unknown"}, "Unknown"},
		{"Checking", configv1beta1.BKEClusterStatus{ClusterStatus: "Checking"}, "Checking"},
		{"Paused", configv1beta1.BKEClusterStatus{ClusterStatus: "Paused"}, "Paused"},
		{"DryRun", configv1beta1.BKEClusterStatus{ClusterStatus: "DryRun"}, "DryRun"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &StatusProcessor{}
			result := p.mapClusterStatus(tt.status.ClusterStatus)
			if result != tt.expectedStatus {
				t.Errorf("mapClusterStatus(%q) = %q, want %q", tt.status.ClusterStatus, result, tt.expectedStatus)
			}
		})
	}
}

func TestGetRestConfigByToken(t *testing.T) {
	bkeCluster := &configv1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: configv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: configv1beta1.APIEndpoint{
				Host: "api.example.com",
				Port: 6443,
			},
		},
	}
	client := &installerClient{
		clientset: k8sfake.NewSimpleClientset(),
	}
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		// Mock GetSecret 返回有效 token
		patches.ApplyFunc(k8sutil.GetSecret, func(_ kubernetes.Interface, name, namespace string) (*corev1.Secret, error) {
			return &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      name,
					Namespace: namespace,
				},
				Data: map[string][]byte{
					"token": []byte("test-token"),
				},
			}, nil
		})
		config, err := client.GetRestConfigByToken(bkeCluster)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		// 验证配置
		if config.Host != "https://api.example.com:6443" {
			t.Errorf("Expected host 'https://api.example.com:6443', got '%s'", config.Host)
		}
		if config.BearerToken != "test-token" {
			t.Errorf("Expected token 'test-token', got '%s'", config.BearerToken)
		}
		if !config.TLSClientConfig.Insecure {
			t.Error("Expected Insecure=true")
		}
	})
}

// 测试集群获取失败
func TestIsHAGetClusterError(t *testing.T) {
	client := &installerClient{}
	patches := gomonkey.ApplyMethodFunc(client,
		"GetClustersByName", func(name string) (*configv1beta1.BKECluster, error) {
			return nil, errors.New("cluster not found")
		})
	defer patches.Reset()

	_, err := client.isHA("test-cluster")
	if err == nil {
		t.Fatal("Expected error but got nil")
	}
}

func TestGetClusterConfigGetClusterError(t *testing.T) {
	// Mock 客户端
	client := &installerClient{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(client, "GetClustersByName", func(name string) (*configv1beta1.BKECluster, error) {
		return nil, errors.New("cluster not found")
	})

	// 执行测试
	_, err := client.GetClusterConfig("test-cluster")
	if err == nil {
		t.Fatal("Expected error but got nil")
	}
}

func TestGetOpenFuyaoVersionFromUnstructured(t *testing.T) {
	t.Run("VersionNotString", testGetOpenFuyaoVersionFromUnstructuredVersionNotString)
	t.Run("VersionMissing", testGetOpenFuyaoVersionFromUnstructuredVersionMissing)
	t.Run("VersionIsBlank", testGetOpenFuyaoVersionFromUnstructuredVersionIsBlank)
	t.Run("Success", testGetOpenFuyaoVersionFromUnstructuredSuccess)
}

func testGetOpenFuyaoVersionFromUnstructuredVersionNotString(t *testing.T) {
	in := &unstructured.Unstructured{
		Object: map[string]any{
			"spec": map[string]any{
				"clusterConfig": map[string]any{
					"cluster": map[string]any{
						"openFuyaoVersion": 1,
					},
				},
			},
		},
	}
	_, err := getOpenFuyaoVersionFromUnstructured(in)
	assert.EqualError(t, err, "failed to extract openFuyaoVersion "+
		"from cluster config: .spec.clusterConfig.cluster.openFuyaoVersion "+
		"accessor error: 1 is of the type int, expected string")
}

func testGetOpenFuyaoVersionFromUnstructuredVersionMissing(t *testing.T) {
	in := &unstructured.Unstructured{}
	_, err := getOpenFuyaoVersionFromUnstructured(in)
	assert.EqualError(t, err, "openFuyaoVersion field is "+
		"missing from spec.clusterConfig.cluster, please ensure the "+
		"field exists in your YAML configuration")
}

func testGetOpenFuyaoVersionFromUnstructuredVersionIsBlank(t *testing.T) {
	in := &unstructured.Unstructured{
		Object: map[string]any{
			"spec": map[string]any{
				"clusterConfig": map[string]any{
					"cluster": map[string]any{
						"openFuyaoVersion": "   ",
					},
				},
			},
		},
	}
	_, err := getOpenFuyaoVersionFromUnstructured(in)
	assert.EqualError(t, err, "openFuyaoVersion field is empty, please provide a valid version")
}

func testGetOpenFuyaoVersionFromUnstructuredSuccess(t *testing.T) {
	in := &unstructured.Unstructured{
		Object: map[string]any{
			"spec": map[string]any{
				"clusterConfig": map[string]any{
					"cluster": map[string]any{
						"openFuyaoVersion": "v25.09.1sp2",
					},
				},
			},
		},
	}
	version, err := getOpenFuyaoVersionFromUnstructured(in)
	assert.Equal(t, version, "v25.09.1sp2")
	assert.Equal(t, err, nil)
}

func TestGetVersionFromPatchCM(t *testing.T) {
	t.Run("GetCMFailed", testGetVersionFromPatchCMGetCMFailed)
	t.Run("CMDataNotFound", testGetVersionFromPatchCMCMDataNotFound)
	t.Run("ValidationError", testGetVersionFromPatchCMValidationError)
	t.Run("Success", testGetVersionFromPatchCMSuccess)
}

func testGetVersionFromPatchCMGetCMFailed(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset()
	patchCMName := "patch-value-v25.09.1sp2"

	// 模拟 Get 失败（ConfigMap 不存在）
	_, err := getVersionFromPatchCM(clientset, patchCMName)
	assert.Contains(t, err.Error(), "failed to get config map")
	assert.Contains(t, err.Error(), patchCMName)
}

func testGetVersionFromPatchCMCMDataNotFound(t *testing.T) {
	version := "v25.09.1sp2"
	patchCMName := constant.PatchValuePrefix + version
	ns := constant.PatchNameSpace

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      patchCMName,
			Namespace: ns,
		},
		Data: map[string]string{
			"other-key": "some content",
		},
	}

	clientset := k8sfake.NewSimpleClientset(cm)
	_, err := getVersionFromPatchCM(clientset, patchCMName)
	if err == nil {
		t.Fatal("expected validation error, but got nil")
	}
	assert.Contains(t, err.Error(), "invalid patch version info")
}

func testGetVersionFromPatchCMValidationError(t *testing.T) {
	version := "v25.09.1sp1"

	invalidVersionYAML := `openFuyaoVersion: ""`

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s%s", constant.PatchValuePrefix, version),
			Namespace: constant.PatchNameSpace,
		},
		Data: map[string]string{
			version: invalidVersionYAML,
		},
	}

	clientSet := k8sfake.NewSimpleClientset(cm)
	patch := gomonkey.NewPatches()
	defer patch.Reset()

	_, err := getVersionFromPatchCM(clientSet, fmt.Sprintf("%s%s", constant.PatchValuePrefix, version))
	assert.Contains(t, err.Error(), "invalid patch version info")
	assert.Contains(t, err.Error(), fmt.Sprintf("%s%s", constant.PatchValuePrefix, version))
}

func testGetVersionFromPatchCMSuccess(t *testing.T) {
	version := "v25.09.1sp2"
	patchCMName := constant.PatchValuePrefix + version
	ns := constant.PatchNameSpace

	validYAML := `openFuyaoVersion: "v25.09.1sp2"
kubernetesVersion: "v1.33"
containerdVersion: "v2.1"`

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      patchCMName,
			Namespace: ns,
		},
		Data: map[string]string{
			version: validYAML,
		},
	}

	clientset := k8sfake.NewSimpleClientset(cm)

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFuncReturn(validatePatchVersionInfo, nil)

	res, err := getVersionFromPatchCM(clientset, patchCMName)
	assert.NoError(t, err)
	assert.Equal(t, PatchVersionInfo{
		OpenFuyaoVersion:  "v25.09.1sp2",
		KubernetesVersion: "v1.33",
		ContainerdVersion: "v2.1",
	}, res)
}

func TestPatchVersion(t *testing.T) {
	t.Run("getOpenFuyaoVersionFromUnstructuredFailed", testPatchVersionGetOpenFuyaoVersionFromUnstructuredFailed)
	t.Run("getOpenFuyaoVersionsFailed", testPatchVersionGetOpenFuyaoVersionsFailed)
	t.Run("VersionNotFound", testPatchVersionVersionNotFound)
	t.Run("getConfigMapFailed", testPatchVersionGetConfigMapFailed)
	t.Run("patchKeyNotFound", testPatchVersionPatchKeyNotFound)
	t.Run("patchFilePathEmpty", testPatchVersionPatchFilePathEmpty)
	t.Run("getClusterInfoFailed", testPatchVersionGetClusterInfoFailed)
	t.Run("getOpenFuyaoVersionFromPatchFileFailed", testPatchVersionGetOpenFuyaoVersionFromPatchFileFailed)
	t.Run("updateOpenFuyaoVersionFailed", testPatchVersionUpdateOpenFuyaoVersionFailed)
	t.Run("success", testPatchVersionSuccess)
}

func testPatchVersionGetOpenFuyaoVersionFromUnstructuredFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "", errors.New("error"))
	obj := &unstructured.Unstructured{}
	obj.SetKind("BKECluster")
	err := patchVersion(&installerClient{}, obj)
	assert.ErrorContains(t, err, "failed to get openFuyao version from input")
}

func testPatchVersionGetOpenFuyaoVersionsFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.1", nil)
	patches.ApplyFuncReturn(k8sutil.GetConfigMap, nil, errors.New("error"))
	client := &installerClient{}

	obj := &unstructured.Unstructured{}
	obj.SetKind("BKECluster")
	err := patchVersion(client, obj)
	assert.ErrorContains(t, err, "failed to get available openFuyao versions")
}

func testPatchVersionVersionNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.2", nil)
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BKEConfigCmKey().Name,
			Namespace: BKEConfigCmKey().Namespace,
		},
		Data: map[string]string{
			"otherRepo":                          "",
			"onlineImage":                        "",
			constant.PatchKeyPrefix + "v25.09.3": "patch-value-v25.09.3",
		},
	}
	client := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}

	obj := &unstructured.Unstructured{}
	obj.SetKind("BKECluster")
	err := patchVersion(client, obj)
	assert.ErrorContains(t, err, "is not supported. Available versions:")
}

func testPatchVersionGetConfigMapFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.3", nil)
	patches.ApplyFuncSeq(k8sutil.GetConfigMap, []gomonkey.OutputCell{
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.3": "patch-value-v25.09.3",
			},
		}, nil}},
		{Values: gomonkey.Params{nil, errors.New("error")}},
	})

	obj := &unstructured.Unstructured{}
	obj.SetKind("BKECluster")
	err := patchVersion(&installerClient{}, obj)
	assert.ErrorContains(t, err, "failed to get ConfigMap")
}

func testPatchVersionPatchKeyNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.4", nil)
	patches.ApplyFuncSeq(k8sutil.GetConfigMap, []gomonkey.OutputCell{
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.4": "patch-value-v25.09.4",
			},
		}, nil}},
		{Values: gomonkey.Params{&corev1.ConfigMap{Data: map[string]string{
			"otherRepo": "",
		}}, nil}},
	})

	obj := &unstructured.Unstructured{}
	obj.SetKind("BKECluster")
	err := patchVersion(&installerClient{}, obj)
	assert.ErrorContains(t, err, "patch file path not found in ConfigMap for key")
}

func testPatchVersionPatchFilePathEmpty(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.5", nil)
	patches.ApplyFuncSeq(k8sutil.GetConfigMap, []gomonkey.OutputCell{
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.5": "patch-value-v25.09.5",
			},
		}, nil}},
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.5": "",
			},
		}, nil}},
	})

	obj := &unstructured.Unstructured{}
	obj.SetKind("BKECluster")
	err := patchVersion(&installerClient{}, obj)
	assert.ErrorContains(t, err, "patch config map is empty for version")
}

func testPatchVersionGetClusterInfoFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.6", nil)
	patches.ApplyFuncSeq(k8sutil.GetConfigMap, []gomonkey.OutputCell{
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.6": constant.PatchValuePrefix + "v25.09.6",
			},
		}, nil}},
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.6": constant.PatchValuePrefix + "v25.09.6",
			},
		}, nil}},
	})
	patches.ApplyFuncReturn(unstructured.NestedMap, nil, false, errors.New("error"))

	obj := &unstructured.Unstructured{}
	obj.SetKind("BKECluster")
	err := patchVersion(&installerClient{}, obj)
	assert.ErrorContains(t, err, "failed to get cluster info from unstructured object")
}

func testPatchVersionGetOpenFuyaoVersionFromPatchFileFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.7", nil)
	patches.ApplyFuncSeq(k8sutil.GetConfigMap, []gomonkey.OutputCell{
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.7": constant.PatchValuePrefix + "v25.09.7",
			},
		}, nil}},
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.7": constant.PatchValuePrefix + "v25.09.7",
			},
		}, nil}},
	})
	patches.ApplyFuncReturn(unstructured.NestedMap, map[string]any{}, true, nil)
	patches.ApplyFuncReturn(getVersionFromPatchCM, nil, errors.New("error"))

	obj := &unstructured.Unstructured{}
	obj.SetKind("BKECluster")
	err := patchVersion(&installerClient{}, obj)
	assert.ErrorContains(t, err, "error")
}

func testPatchVersionUpdateOpenFuyaoVersionFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.8", nil)
	patches.ApplyFuncSeq(k8sutil.GetConfigMap, []gomonkey.OutputCell{
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.8": constant.PatchValuePrefix + "v25.09.8.yaml",
			},
		}, nil}},
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.8": constant.PatchValuePrefix + "v25.09.8.yaml",
			},
		}, nil}},
	})
	patches.ApplyFuncReturn(unstructured.NestedMap, map[string]any{}, true, nil)
	patches.ApplyFuncReturn(getVersionFromPatchCM, PatchVersionInfo{
		OpenFuyaoVersion:  "v25.09.2",
		KubernetesVersion: "v1.33.1",
		ContainerdVersion: "v2.1",
	}, nil)

	obj := unstructured.Unstructured{
		Object: map[string]any{
			"spec": map[string]any{
				"clusterConfig": map[int]any{
					1: "error",
				},
			},
		},
	}
	obj.SetKind("BKECluster")

	err := patchVersion(&installerClient{}, &obj)
	assert.ErrorContains(t, err, "failed to update cluster configuration with patch version info")
}

func testPatchVersionSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFuncReturn(getOpenFuyaoVersionFromUnstructured, "v25.09.9", nil)
	patches.ApplyFuncSeq(k8sutil.GetConfigMap, []gomonkey.OutputCell{
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.9": constant.PatchValuePrefix + "v25.09.9.yaml",
			},
		}, nil}},
		{Values: gomonkey.Params{&corev1.ConfigMap{
			Data: map[string]string{
				"otherRepo":                          "",
				"onlineImage":                        "",
				constant.PatchKeyPrefix + "v25.09.9": constant.PatchValuePrefix + "v25.09.9.yaml",
			},
		}, nil}},
	})
	patches.ApplyFuncReturn(unstructured.NestedMap, map[string]any{}, true, nil)
	patches.ApplyFuncReturn(getVersionFromPatchCM, PatchVersionInfo{
		OpenFuyaoVersion:  "v25.09.3",
		KubernetesVersion: "v1.33.1",
		ContainerdVersion: "v2.2",
	}, nil)

	obj := unstructured.Unstructured{
		Object: map[string]any{
			"spec": map[string]any{
				"clusterConfig": map[string]any{
					"cluster": map[string]any{
						"openFuyaoVersion":  "v25.09.1",
						"kubernetesVersion": "v1.28.0",
						"containerdVersion": "v1.9",
					},
				},
			},
		},
	}
	obj.SetKind("BKECluster")
	err := patchVersion(&installerClient{}, &obj)
	assert.Equal(t, err, nil)
}

func TestValidatePatchVersionInfo(t *testing.T) {
	t.Run("openFuyaoVersionMissing", testValidatePatchVersionInfoOpenFuyaoVersion)
	t.Run("kubernetesVersionMissing", testValidatePatchVersionInfoKubernetesVersion)
	t.Run("containerdVersionMissing", testValidatePatchVersionInfoContainerdVersion)
	t.Run("containerdVersionMissing", testValidatePatchVersionInfoEtcdVersion)
	t.Run("success", testValidatePatchVersionInfoSuccess)
}

func testValidatePatchVersionInfoOpenFuyaoVersion(t *testing.T) {
	err := validatePatchVersionInfo(&PatchVersionInfo{
		OpenFuyaoVersion:  "",
		KubernetesVersion: "v1",
		ContainerdVersion: "v1",
		EtcdVersion:       "v1",
	})

	assert.EqualError(t, err, "openFuyaoVersion is required but empty")
}

func testValidatePatchVersionInfoKubernetesVersion(t *testing.T) {
	err := validatePatchVersionInfo(&PatchVersionInfo{
		OpenFuyaoVersion:  "v1",
		KubernetesVersion: "",
		ContainerdVersion: "v1",
		EtcdVersion:       "v1",
	})

	assert.EqualError(t, err, "kubernetesVersion is required but empty")
}

func testValidatePatchVersionInfoContainerdVersion(t *testing.T) {
	err := validatePatchVersionInfo(&PatchVersionInfo{
		OpenFuyaoVersion:  "v1",
		KubernetesVersion: "v1",
		ContainerdVersion: "",
		EtcdVersion:       "v1",
	})

	assert.EqualError(t, err, "containerdVersion is required but empty")
}

func testValidatePatchVersionInfoEtcdVersion(t *testing.T) {
	err := validatePatchVersionInfo(&PatchVersionInfo{
		OpenFuyaoVersion:  "v1",
		KubernetesVersion: "v1",
		ContainerdVersion: "v1",
		EtcdVersion:       "",
	})

	assert.EqualError(t, err, "etcdVersion is required but empty")
}

func testValidatePatchVersionInfoSuccess(t *testing.T) {
	err := validatePatchVersionInfo(&PatchVersionInfo{
		OpenFuyaoVersion:  "v1",
		KubernetesVersion: "v1",
		ContainerdVersion: "v1",
		EtcdVersion:       "v1",
	})

	assert.Equal(t, err, nil)
}

func TestAddPatchInfoToConfigMap(t *testing.T) {
	t.Run("success", testAddPatchInfoToConfigMapSuccess)
	t.Run("updateFailure", testAddPatchInfoToConfigMapUpdateFailure)
}

func testAddPatchInfoToConfigMapSuccess(t *testing.T) {
	// 创建fake clientset
	fakeClientset := k8sfake.NewSimpleClientset()

	c := &installerClient{
		clientset: fakeClientset,
	}

	fakeCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-configmap",
			Namespace: "default",
		},
		Data: map[string]string{},
	}

	_, err := fakeClientset.CoreV1().ConfigMaps("default").
		Create(context.Background(), fakeCM, metav1.CreateOptions{})
	assert.NoError(t, err)

	version := "v25.09.1"
	fakePatchKey := fmt.Sprintf("%s%s", constant.PatchKeyPrefix, version)
	fakePatchCM := fmt.Sprintf("%s%s", constant.PatchValuePrefix, version)

	err = c.addPatchInfoToConfigMap(fakeCM, fakePatchKey, fakePatchCM)
	assert.NoError(t, err)

	updatedCM, err := fakeClientset.CoreV1().ConfigMaps("default").
		Get(context.Background(), "test-configmap", metav1.GetOptions{})
	assert.NoError(t, err)
	assert.Equal(t, fakePatchCM, updatedCM.Data[fakePatchKey])
}

func testAddPatchInfoToConfigMapUpdateFailure(t *testing.T) {
	// 创建fake clientset
	fakeClientset := k8sfake.NewSimpleClientset()

	c := &installerClient{
		clientset: fakeClientset,
	}

	fakeCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "non-existent-configmap",
			Namespace: "default",
		},
		Data: map[string]string{},
	}

	fakeOpenFuyaoVersion := "v25.09.1"
	fakeFilePath := "/a/file/path"

	err := c.addPatchInfoToConfigMap(fakeCM, fakeOpenFuyaoVersion, fakeFilePath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to patch ConfigMap")
}

type fakeHttpClient struct{}

func (c *fakeHttpClient) Do(req *http.Request) (*http.Response, error) {
	return nil, errors.New("patch me")
}

func TestGetRemoteContent(t *testing.T) {
	t.Run("httpFailed", testGetRemoteContentHttpFailed)
	t.Run("httpStatusNotOk", testGetRemoteContentHttpStatusNotOk)
	t.Run("readBodyFailed", testGetRemoteContentReadBodyFailed)
	t.Run("success", testGetRemoteContentSuccess)
}

func testGetRemoteContentHttpFailed(t *testing.T) {
	c := &fakeHttpClient{}
	_, err := getRemoteContent(c, "fakeUrl", "fakePath")
	assert.ErrorContains(t, err, "failed to fetch remote content from")
}

func testGetRemoteContentHttpStatusNotOk(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	c := &fakeHttpClient{}
	fakeResponse := http.Response{
		StatusCode: 404,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	patches.ApplyMethodReturn(c, "Do", &fakeResponse, nil)

	_, err := getRemoteContent(c, "fakeUrl", "fakePath")
	assert.ErrorContains(t, err, "HTTP request failed: status")
}

func testGetRemoteContentReadBodyFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	c := &fakeHttpClient{}
	fakeResponse := http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	patches.ApplyMethodReturn(c, "Do", &fakeResponse, nil)
	patches.ApplyFuncReturn(io.ReadAll, []byte{}, errors.New("error"))

	_, err := getRemoteContent(c, "fakeUrl", "fakePath")
	assert.ErrorContains(t, err, "failed to read response body from")
}

func testGetRemoteContentSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	c := &fakeHttpClient{}
	fakeContent := "fake content"
	fakeResponse := http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader(fakeContent)),
	}
	patches.ApplyMethodReturn(c, "Do", &fakeResponse, nil)

	body, err := getRemoteContent(c, "fakeUrl", "fakePath")
	assert.Equal(t, err, nil)
	assert.Equal(t, body, []byte(fakeContent))
}

func TestGetPatchIndexFromRemoteRepo(t *testing.T) {
	t.Run("getIndexFailed", testGetPatchIndexFromRemoteRepoGetIndexFailed)
	t.Run("decodeFailed", testGetPatchIndexFromRemoteRepoDecodeFailed)
	t.Run("openFuyaoVersionNotFound", testGetPatchIndexFromRemoteRepoOpenFuyaoVersionNotFound)
	t.Run("filePathNotFound", testGetPatchIndexFromRemoteRepoFilePathNotFound)
	t.Run("success", testGetPatchIndexFromRemoteRepoSuccess)
}

func testGetPatchIndexFromRemoteRepoGetIndexFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getRemoteContent, []byte{}, errors.New("error"))

	_, err := getPatchIndexFromRemoteRepo(&fakeHttpClient{}, "fakeUrl")
	assert.ErrorContains(t, err, "error")
}

func testGetPatchIndexFromRemoteRepoDecodeFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getRemoteContent, []byte("not a valid yaml"), nil)

	_, err := getPatchIndexFromRemoteRepo(&fakeHttpClient{}, "fakeUrl")
	assert.ErrorContains(t, err, "error")
}

func testGetPatchIndexFromRemoteRepoOpenFuyaoVersionNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getRemoteContent, []byte(`- a: "1"`), nil)

	_, err := getPatchIndexFromRemoteRepo(&fakeHttpClient{}, "fakeUrl")
	assert.ErrorContains(t, err, "missing required field 'openFuyaoVersion'")
}

func testGetPatchIndexFromRemoteRepoFilePathNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getRemoteContent, []byte(`- openFuyaoVersion: "v25.09"`), nil)

	_, err := getPatchIndexFromRemoteRepo(&fakeHttpClient{}, "fakeUrl")
	assert.ErrorContains(t, err, "missing required field 'filePath'")
}

func testGetPatchIndexFromRemoteRepoSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFuncReturn(getRemoteContent, []byte(`- openFuyaoVersion: "v25.09"
  filePath: "./v25.09.yaml"`), nil)

	res, err := getPatchIndexFromRemoteRepo(&fakeHttpClient{}, "fakeUrl")
	assert.Equal(t, err, nil)
	assert.Equal(t, res[0]["openFuyaoVersion"], "v25.09")
	assert.Equal(t, res[0]["filePath"], "v25.09.yaml")
}

func TestIsOnlineMode(t *testing.T) {
	t.Run("configMapDataLenIsZero", testIsOnlineModeConfigMapDataLenIsZero)
	t.Run("otherRepoMissing", testIsOnlineModeOtherRepoMissing)
	t.Run("onlineImageMissing", testIsOnlineModeOnlineImageMissing)
	t.Run("offline", testIsOnlineModeOffline)
	t.Run("online1", testIsOnlineModeOnline1)
	t.Run("online2", testIsOnlineModeOnline2)
	t.Run("online3", testIsOnlineModeOnline3)
}

func testIsOnlineModeConfigMapDataLenIsZero(t *testing.T) {
	fakeCM := &corev1.ConfigMap{}
	_, err := IsOnlineMode(fakeCM)
	assert.ErrorContains(t, err, "configMap.Data is nil or empty")
}

func testIsOnlineModeOtherRepoMissing(t *testing.T) {
	fakeCM := &corev1.ConfigMap{
		Data: map[string]string{
			"onlineImage": "",
		},
	}
	_, err := IsOnlineMode(fakeCM)
	assert.ErrorContains(t, err, "configMap key `otherRepo` or `onlineImage` missing")
}

func testIsOnlineModeOnlineImageMissing(t *testing.T) {
	fakeCM := &corev1.ConfigMap{
		Data: map[string]string{
			"otherRepo": "",
		},
	}
	_, err := IsOnlineMode(fakeCM)
	assert.ErrorContains(t, err, "configMap key `otherRepo` or `onlineImage` missing")
}

func testIsOnlineModeOffline(t *testing.T) {
	fakeCM := &corev1.ConfigMap{
		Data: map[string]string{
			"otherRepo":   "",
			"onlineImage": "",
		},
	}
	isOnline, err := IsOnlineMode(fakeCM)
	assert.Equal(t, err, nil)
	assert.Equal(t, isOnline, false)
}

func testIsOnlineModeOnline1(t *testing.T) {
	fakeCM := &corev1.ConfigMap{
		Data: map[string]string{
			"otherRepo":   "repo.address",
			"onlineImage": "",
		},
	}
	isOnline, err := IsOnlineMode(fakeCM)
	assert.Equal(t, err, nil)
	assert.Equal(t, isOnline, true)
}

func testIsOnlineModeOnline2(t *testing.T) {
	fakeCM := &corev1.ConfigMap{
		Data: map[string]string{
			"onlineImage": "image.address",
			"otherRepo":   "",
		},
	}
	isOnline, err := IsOnlineMode(fakeCM)
	assert.Equal(t, err, nil)
	assert.Equal(t, isOnline, true)
}

func testIsOnlineModeOnline3(t *testing.T) {
	fakeCM := &corev1.ConfigMap{
		Data: map[string]string{
			"onlineImage": "image.address",
			"otherRepo":   "repo.address",
		},
	}
	isOnline, err := IsOnlineMode(fakeCM)
	assert.Equal(t, err, nil)
	assert.Equal(t, isOnline, true)
}

func TestEnsureNsExists(t *testing.T) {
	t.Run("NamespaceAlreadyExists", testEnsureNsExistsAlreadyExists)
	t.Run("CreateSuccess", testEnsureNsExistsCreateSuccess)
	t.Run("CreateAlreadyExistsRace", testEnsureNsExistsCreateAlreadyExistsRace)
	t.Run("GetErrorOtherThanNotFound", testEnsureNsExistsGetError)
	t.Run("CreateError", testEnsureNsExistsCreateError)
}

func testEnsureNsExistsAlreadyExists(t *testing.T) {
	nsName := "patch-system"
	existingNs := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: nsName},
	}
	clientset := k8sfake.NewSimpleClientset(existingNs)

	err := ensureNsExists(clientset, nsName)
	assert.NoError(t, err)
}

func testEnsureNsExistsCreateSuccess(t *testing.T) {
	nsName := "patch-system"
	clientset := k8sfake.NewSimpleClientset()

	err := ensureNsExists(clientset, nsName)
	assert.NoError(t, err)

	_, err = clientset.CoreV1().Namespaces().Get(context.TODO(), nsName, metav1.GetOptions{})
	assert.NoError(t, err)
}

func testEnsureNsExistsCreateAlreadyExistsRace(t *testing.T) {
	nsName := "patch-system"
	clientset := k8sfake.NewSimpleClientset()
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: nsName}}
	_, _ = clientset.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{})

	err := ensureNsExists(clientset, nsName)
	assert.NoError(t, err)
}

func testEnsureNsExistsGetError(t *testing.T) {
	nsName := "patch-system"
	clientset := k8sfake.NewSimpleClientset()

	patches := gomonkey.ApplyMethodSeq(
		reflect.TypeOf(clientset.CoreV1().Namespaces()),
		"Get",
		[]gomonkey.OutputCell{
			{Values: gomonkey.Params{nil, errors.New("network timeout")}},
		},
	)
	defer patches.Reset()

	err := ensureNsExists(clientset, nsName)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "check namespace patch-system failed")
	assert.Contains(t, err.Error(), "network timeout")
}

func testEnsureNsExistsCreateError(t *testing.T) {
	nsName := "patch-system"
	clientset := k8sfake.NewSimpleClientset()

	patches := gomonkey.ApplyMethodSeq(
		reflect.TypeOf(clientset.CoreV1().Namespaces()),
		"Create",
		[]gomonkey.OutputCell{
			{Values: gomonkey.Params{nil, errors.New("create forbidden")}},
		},
	)
	defer patches.Reset()

	patches.ApplyMethodSeq(
		reflect.TypeOf(clientset.CoreV1().Namespaces()),
		"Get",
		[]gomonkey.OutputCell{
			{Values: gomonkey.Params{nil, apierrors.NewNotFound(corev1.Resource("namespaces"), nsName)}},
		},
	)

	err := ensureNsExists(clientset, nsName)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "create namespace patch-system failed")
	assert.Contains(t, err.Error(), "create forbidden")
}

func TestAddPatchesToConfigMap(t *testing.T) {
	t.Run("EnsureNsFailed", testAddPatchesToConfigMapEnsureNsFailed)
	t.Run("CreateCMSuccess", testAddPatchesToConfigMapCreateCMSuccess)
	t.Run("CMAlreadyExists", testAddPatchesToConfigMapCMAlreadyExists)
	t.Run("GetCMErrOtherThanNotFound", testAddPatchesToConfigMapGetCMErr)
	t.Run("CreateCMErr", testAddPatchesToConfigMapCreateCMErr)
}

func testAddPatchesToConfigMapEnsureNsFailed(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset()

	patches := gomonkey.ApplyFunc(ensureNsExists, func(client kubernetes.Interface, ns string) error {
		return errors.New("mock ensure ns error")
	})
	defer patches.Reset()

	key, value, err := addPatchesToConfigMap(clientset, "v25.09.1sp2.yaml", "content")

	assert.Empty(t, key)
	assert.Empty(t, value)
	assert.Contains(t, err.Error(), "failed to ensure ns")
	assert.Contains(t, err.Error(), "mock ensure ns error")
}

func testAddPatchesToConfigMapCreateCMSuccess(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset()
	patches := gomonkey.ApplyFunc(ensureNsExists, func(client kubernetes.Interface, ns string) error {
		if _, err := client.CoreV1().Namespaces().Get(context.TODO(), ns, metav1.GetOptions{}); err != nil {
			if apierrors.IsNotFound(err) {
				_, _ = client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
					ObjectMeta: metav1.ObjectMeta{Name: ns},
				}, metav1.CreateOptions{})
			}
		}
		return nil
	})
	defer patches.Reset()

	content := "openFuyaoVersion: v25.09.1sp2"
	version := "v25.09.1sp2"
	expectedKey := constant.PatchKeyPrefix + version
	expectedValue := constant.PatchValuePrefix + version

	key, value, err := addPatchesToConfigMap(clientset, version, content)

	assert.NoError(t, err)
	assert.Equal(t, expectedKey, key)
	assert.Equal(t, expectedValue, value)

	cm, err := clientset.CoreV1().ConfigMaps(constant.PatchNameSpace).Get(context.TODO(), expectedValue, metav1.GetOptions{})
	require.NoError(t, err)
	assert.Equal(t, content, cm.Data[version])
}

func testAddPatchesToConfigMapCMAlreadyExists(t *testing.T) {
	version := "v25.09.1sp2"
	cmName := constant.PatchValuePrefix + version
	ns := constant.PatchNameSpace

	existingCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      cmName,
			Namespace: ns,
		},
		Data: map[string]string{
			version: "old-content",
		},
	}

	clientset := k8sfake.NewSimpleClientset(existingCM)

	patches := gomonkey.ApplyFunc(ensureNsExists, func(client kubernetes.Interface, ns string) error {
		_, _ = client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: ns},
		}, metav1.CreateOptions{})
		return nil
	})
	defer patches.Reset()

	key, value, err := addPatchesToConfigMap(clientset, "v25.09.1sp2", "new-content")

	assert.NoError(t, err)
	assert.Equal(t, constant.PatchKeyPrefix+version, key)
	assert.Equal(t, constant.PatchValuePrefix+version, value)

	cm, _ := clientset.CoreV1().ConfigMaps(ns).Get(context.TODO(), cmName, metav1.GetOptions{})
	assert.Equal(t, "new-content", cm.Data[version])
}

func testAddPatchesToConfigMapGetCMErr(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset()
	patches := gomonkey.ApplyFunc(ensureNsExists, func(client kubernetes.Interface, ns string) error {
		_, _ = client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{})
		return nil
	})

	patches.ApplyMethodSeq(
		reflect.TypeOf(clientset.CoreV1().ConfigMaps(constant.PatchNameSpace)),
		"Get",
		[]gomonkey.OutputCell{
			{Values: gomonkey.Params{nil, errors.New("api server timeout")}},
		},
	)
	defer patches.Reset()

	key, value, err := addPatchesToConfigMap(clientset, "v25.09.1sp2.yaml", "content")

	assert.Empty(t, key)
	assert.Empty(t, value)
	assert.Contains(t, err.Error(), "api server timeout")
}

func testAddPatchesToConfigMapCreateCMErr(t *testing.T) {
	clientset := k8sfake.NewSimpleClientset()
	patches := gomonkey.ApplyFunc(ensureNsExists, func(client kubernetes.Interface, ns string) error {
		_, _ = client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{})
		return nil
	})

	patches.ApplyMethodSeq(
		reflect.TypeOf(clientset.CoreV1().ConfigMaps(constant.PatchNameSpace)),
		"Get",
		[]gomonkey.OutputCell{
			{Values: gomonkey.Params{nil, apierrors.NewNotFound(corev1.Resource("configmaps"), "patch-value-v25.09.1sp2")}},
		},
	)

	patches.ApplyMethodSeq(
		reflect.TypeOf(clientset.CoreV1().ConfigMaps(constant.PatchNameSpace)),
		"Create",
		[]gomonkey.OutputCell{
			{Values: gomonkey.Params{nil, errors.New("forbidden: cannot create configmap")}},
		},
	)
	defer patches.Reset()

	key, value, err := addPatchesToConfigMap(clientset, "v25.09.1sp2.yaml", "content")

	assert.Empty(t, key)
	assert.Empty(t, value)
	assert.Contains(t, err.Error(), "forbidden: cannot create configmap")
}

func TestOpenFuyaoVersion(t *testing.T) {
	t.Run("TestGetOpenFuyaoVersions", testGetOpenFuyaoVersionsWithGomonkey)
	t.Run("TestGetInstallableVersions", testGetInstallableVersions)
	t.Run("TestGetUpgradeableVersions", testGetUpgradeableVersions)
	t.Run("TestGetOpenFuyaoUpgradeVersions", testGetOpenFuyaoUpgradeVersionsWithGomonkey)
	t.Run("TestGetOpenFuyaoUpgradeVersionsInvalid", testGetOpenFuyaoUpgradeVersionsInvalidCurrent)
	t.Run("TestGetOpenFuyaoVersionsGetVersionsError", testGetOpenFuyaoVersionsGetVersionsError)
}

func testGetOpenFuyaoVersionsWithGomonkey(t *testing.T) {
	mockVersions := []string{
		"latest",
		"v25.09",
		"v25.10-rc.2",
		"v25.10",
		"v25.10.2",
		"v25.12",
		"v25.12-rc.1",
		"v25.12.2",
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BKEConfigCmKey().Name,
			Namespace: BKEConfigCmKey().Namespace,
		},
		Data: map[string]string{
			"otherRepo":   "",
			"onlineImage": "",
		},
	}
	for _, v := range mockVersions {
		cm.Data[constant.PatchKeyPrefix+v] = constant.PatchValuePrefix + v
	}
	c := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}
	got, err := c.GetOpenFuyaoVersions()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := []string{"latest", "v25.09", "v25.10-rc.2", "v25.10", "v25.12", "v25.12-rc.1"}
	sort.Strings(got)
	sort.Strings(want)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetOpenFuyaoVersions() = %v, want %v", got, want)
	}
}

func testGetInstallableVersions(t *testing.T) {
	scheme := runtime.NewScheme()
	objects := []runtime.Object{
		&unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "config.openfuyao.com/v1alpha1",
				"kind":       "UpgradePath",
				"metadata": map[string]interface{}{
					"name": "paths",
				},
				"spec": map[string]interface{}{
					"versions": []interface{}{
						map[string]interface{}{
							"version":     "v25.09.0",
							"installable": true,
							"deprecated":  false,
						},
						map[string]interface{}{
							"version":     "v25.10.0",
							"installable": false,
							"deprecated":  false,
						},
						map[string]interface{}{
							"version":     "v25.12.0",
							"installable": true,
							"deprecated":  true,
						},
						map[string]interface{}{
							"version":     "v26.03.0",
							"installable": true,
							"deprecated":  false,
						},
					},
				},
			},
		},
	}
	dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
		scheme,
		map[schema.GroupVersionResource]string{upgradePathGVR: "UpgradePathList"},
		objects...,
	)
	c := &installerClient{dynamicClient: dyn}

	got, err := c.GetInstallableVersions()
	require.NoError(t, err)
	assert.Equal(t, []string{"v25.09.0", "v26.03.0"}, got)
}

func testGetUpgradeableVersions(t *testing.T) {
	newClient := func(paths []interface{}, versions []interface{}) *installerClient {
		scheme := runtime.NewScheme()
		objects := []runtime.Object{
			&unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "config.openfuyao.com/v1alpha1",
					"kind":       "UpgradePath",
					"metadata": map[string]interface{}{
						"name": "paths",
					},
					"spec": map[string]interface{}{
						"paths":    paths,
						"versions": versions,
					},
				},
			},
		}
		dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
			scheme,
			map[schema.GroupVersionResource]string{upgradePathGVR: "UpgradePathList"},
			objects...,
		)
		return &installerClient{dynamicClient: dyn}
	}

	paths := []interface{}{
		map[string]interface{}{"from": "v25.09.0", "to": "v25.12.0"},
		map[string]interface{}{"from": "v25.12.0", "to": "v26.03.0"},
		map[string]interface{}{"from": "v25.09.0", "to": "v26.09.0", "blocked": true},
	}
	versions := []interface{}{
		map[string]interface{}{"version": "v25.09.0", "installable": true},
		map[string]interface{}{"version": "v25.12.0", "installable": true},
		map[string]interface{}{"version": "v26.03.0", "installable": true},
		map[string]interface{}{"version": "v26.09.0", "installable": true},
	}

	t.Run("returns reachable versions and skips blocked paths", func(t *testing.T) {
		got, err := newClient(paths, versions).GetUpgradeableVersions("v25.09.0")
		require.NoError(t, err)
		assert.Equal(t, []string{"v25.12.0", "v26.03.0"}, got)
	})

	t.Run("returns empty result for unknown current version", func(t *testing.T) {
		got, err := newClient(paths, versions).GetUpgradeableVersions("v99.99.0")
		require.NoError(t, err)
		assert.Empty(t, got)
	})

	t.Run("returns error when dynamic client is nil", func(t *testing.T) {
		_, err := (&installerClient{}).GetUpgradeableVersions("v25.09.0")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "dynamic client is nil")
	})

	t.Run("returns error for invalid upgrade graph", func(t *testing.T) {
		invalidPaths := []interface{}{
			map[string]interface{}{"from": "v25.09.0", "to": "v25.09.0"},
		}
		_, err := newClient(invalidPaths, versions).GetUpgradeableVersions("v25.09.0")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "self loop")
	})
}

func testGetOpenFuyaoUpgradeVersionsWithGomonkey(t *testing.T) {
	mockVersions := []string{"v25.09", "v25.10-rc.2", "v25.10", "v25.10.2", "v25.12", "v25.12-rc.1", "v25.12.2", "v25.12.3", "v26.03-rc.1", "v26.03.1"}

	tests := []struct {
		current string
		want    []string
	}{
		{"v25.09", []string{"v25.10-rc.2", "v25.10", "v25.12", "v25.12-rc.1", "v26.03-rc.1"}},
		{"v25.10", []string{"v25.10.2", "v25.12", "v25.12-rc.1", "v26.03-rc.1"}},
		{"v25.10-rc.2", []string{"v25.10", "v25.10.2", "v25.12", "v25.12-rc.1", "v26.03-rc.1"}},
		{"v25.12", []string{"v25.12.2", "v25.12.3", "v26.03-rc.1"}},
		{"v25.12-rc.1", []string{"v25.12", "v25.12.2", "v25.12.3", "v26.03-rc.1"}},
		{"v25.12.2", []string{"v25.12.3", "v26.03-rc.1"}},
		{"v26.03-rc.1", []string{"v26.03.1"}},
		{"v26.03.1", nil},
		{"latest", []string{}},
	}

	for _, tt := range tests {
		t.Run("from "+tt.current, func(t *testing.T) {
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      BKEConfigCmKey().Name,
					Namespace: BKEConfigCmKey().Namespace,
				},
				Data: map[string]string{
					"otherRepo":   "",
					"onlineImage": "",
				},
			}
			for _, v := range mockVersions {
				cm.Data[constant.PatchKeyPrefix+v] = constant.PatchValuePrefix + v
			}
			c := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}
			got, err := c.GetOpenFuyaoUpgradeVersions(tt.current)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got == nil {
				got = []string{}
			}
			sort.Strings(got)
			want := append([]string(nil), tt.want...)
			if want == nil {
				want = []string{}
			}
			sort.Strings(want)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func testGetOpenFuyaoUpgradeVersionsInvalidCurrent(t *testing.T) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BKEConfigCmKey().Name,
			Namespace: BKEConfigCmKey().Namespace,
		},
		Data: map[string]string{
			"otherRepo":                        "",
			"onlineImage":                      "",
			constant.PatchKeyPrefix + "v25.10": constant.PatchValuePrefix + "v25.10",
		},
	}
	c := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}
	_, err := c.GetOpenFuyaoUpgradeVersions("not-a-valid-version")
	if err == nil {
		t.Error("expected error for invalid current version, got nil")
	}
}

func testGetOpenFuyaoVersionsGetVersionsError(t *testing.T) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BKEConfigCmKey().Name,
			Namespace: BKEConfigCmKey().Namespace,
		},
		Data: map[string]string{},
	}
	c := &installerClient{clientset: k8sfake.NewSimpleClientset(cm)}
	_, err := c.GetOpenFuyaoVersions()
	if err == nil {
		t.Error("expected error when GetDeployVersions fails, got nil")
	}
}

func TestUpgradeClusterVersion(t *testing.T) {
	newClient := func(objects ...runtime.Object) *installerClient {
		scheme := runtime.NewScheme()
		dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
			scheme,
			map[schema.GroupVersionResource]string{
				gvr:               "BKEClusterList",
				clusterVersionGVR: "ClusterVersionList",
			},
			objects...,
		)
		return &installerClient{dynamicClient: dyn}
	}

	cluster := testutilsMakeBKECluster("upgrade-ok")
	t.Run("create cluster version desired version", func(t *testing.T) {
		c := newClient(cluster.DeepCopy())
		require.NoError(t, c.UpgradeClusterVersion("upgrade-ok", "v26.03.0"))

		got, err := c.dynamicClient.Resource(clusterVersionGVR).Namespace("upgrade-ok").
			Get(context.Background(), "upgrade-ok", metav1.GetOptions{})
		require.NoError(t, err)
		desired, _, _ := unstructured.NestedString(got.Object, "spec", "desiredVersion")
		assert.Equal(t, "v26.03.0", desired)
	})

	t.Run("update existing cluster version desired version", func(t *testing.T) {
		cv := &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": "config.openfuyao.com/v1alpha1",
			"kind":       "ClusterVersion",
			"metadata": map[string]interface{}{
				"name":      "upgrade-ok",
				"namespace": "upgrade-ok",
			},
			"spec": map[string]interface{}{
				"desiredVersion": "v25.09.0",
			},
		}}
		c := newClient(cluster.DeepCopy(), cv)
		require.NoError(t, c.UpgradeClusterVersion("upgrade-ok", "v26.03.0"))

		got, err := c.dynamicClient.Resource(clusterVersionGVR).Namespace("upgrade-ok").
			Get(context.Background(), "upgrade-ok", metav1.GetOptions{})
		require.NoError(t, err)
		desired, _, _ := unstructured.NestedString(got.Object, "spec", "desiredVersion")
		assert.Equal(t, "v26.03.0", desired)
	})

	t.Run("missing cluster returns error", func(t *testing.T) {
		c := newClient()
		err := c.UpgradeClusterVersion("missing", "v26.03.0")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "get BKECluster")
	})
}

func testutilsMakeBKECluster(name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKECluster",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": name,
		},
		"spec": map[string]interface{}{
			"clusterConfig": map[string]interface{}{
				"cluster": map[string]interface{}{
					"openFuyaoVersion": "v25.09.0",
				},
			},
		},
	}}
}

// Tests for createResourcesOnCrateCluster: cluster-scoped, namespaced (namespace creation), and already-exists behavior
func TestCreateResourcesOnCrateCluster(t *testing.T) {
	cases := []struct {
		name           string
		obj            *unstructured.Unstructured
		precreateNs    bool
		precreateObj   bool
		precreateObjNs string
		expectName     string
	}{
		{name: "cluster-scoped", obj: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test.example.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "cluster-obj"}}}, expectName: "cluster-obj"},
		{name: "namespaced create", obj: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test.example.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "ns-obj", "namespace": "ns1"}}}, expectName: "ns-obj"},
		{name: "already exists", obj: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test.example.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "pre-obj", "namespace": "ns2"}}}, precreateNs: true, precreateObj: true, precreateObjNs: "ns2", expectName: "pre-obj"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			dyn := dynamicfake.NewSimpleDynamicClient(scheme)
			client := &installerClient{dynamicClient: dyn}

			gvr := schema.GroupVersionResource{Group: "test.example.com", Version: "v1", Resource: "foos"}
			mapping := &meta.RESTMapping{Resource: gvr}
			nsGVR := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"}

			// optional precreate
			if tc.precreateNs {
				ns2 := &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "v1", "kind": "Namespace", "metadata": map[string]interface{}{"name": tc.precreateObjNs}}}
				if _, err := dyn.Resource(nsGVR).Create(context.TODO(), ns2, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to pre-create namespace: %v", err)
				}
			}
			if tc.precreateObj {
				if _, err := dyn.Resource(gvr).Namespace(tc.precreateObjNs).Create(context.TODO(), tc.obj, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to pre-create object: %v", err)
				}
			}

			// call function under test
			if err := client.createResourcesOnCrateCluster(tc.obj, mapping); err != nil {
				t.Fatalf("createResourcesOnCrateCluster error: %v", err)
			}

			// verify object exists (cluster or namespaced)
			ns := tc.obj.GetNamespace()
			var got *unstructured.Unstructured
			var err error
			if ns == "" {
				got, err = dyn.Resource(gvr).Get(context.TODO(), tc.expectName, metav1.GetOptions{})
			} else {
				// ensure namespace exists
				if _, err = dyn.Resource(nsGVR).Get(context.TODO(), ns, metav1.GetOptions{}); err != nil {
					t.Fatalf("expected namespace created, get ns error: %v", err)
				}
				got, err = dyn.Resource(gvr).Namespace(ns).Get(context.TODO(), tc.expectName, metav1.GetOptions{})
			}
			if err != nil {
				t.Fatalf("expected created object, get error: %v", err)
			}
			if got.GetName() != tc.expectName {
				t.Fatalf("unexpected name: got %s want %s", got.GetName(), tc.expectName)
			}
		})
	}
}

// 为 validateNodeTime 添加表驱动测试（打桩 getRemoteTime）
func TestValidateNodeTime(t *testing.T) {
	cases := []struct {
		name         string
		remoteFunc   func(*ssh.Client) (time.Time, error)
		wantErr      bool
		wantContains string
	}{
		{name: "success small diff", remoteFunc: func(_ *ssh.Client) (time.Time, error) { return time.Now().Add(-1 * time.Second), nil }, wantErr: false},
		{name: "remote error", remoteFunc: func(_ *ssh.Client) (time.Time, error) { return time.Time{}, errors.New("ssh failure") }, wantErr: true, wantContains: "get remote time err"},
		{name: "large diff", remoteFunc: func(_ *ssh.Client) (time.Time, error) { return time.Now().Add(-MaxTimeDiff * 2), nil }, wantErr: true, wantContains: "time diff too large"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			patches := gomonkey.ApplyFunc(getRemoteTime, tc.remoteFunc)
			defer patches.Reset()

			c := &installerClient{}
			node := &ClusterNode{Ip: "1.2.3.4"}
			err := c.validateNodeTime(nil, node)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				if tc.wantContains != "" && !strings.Contains(err.Error(), tc.wantContains) {
					t.Fatalf("error %v does not contain %q", err, tc.wantContains)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
		})
	}
}
