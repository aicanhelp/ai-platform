/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"context"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/kubernetes"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

const (
	// testNodeSkipIP 测试中需要跳过的节点IP（使用本地回环地址用于测试）
	testNodeSkipIP = "127.0.0.100"
	// testNodeCheckIP 测试中需要检查的节点IP（使用本地回环地址用于测试）
	testNodeCheckIP = "127.0.0.101"
	// testClusterName 测试集群名称
	testClusterName = "test-cluster"
	// testNodeSkipName 需要跳过的节点名称
	testNodeSkipName = "node-skip"
	// testNodeCheckName 需要检查的节点名称
	testNodeCheckName = "node-check"
	// expectedNodeCount 期望的节点数量
	expectedNodeCount = 2
)

// createTestCluster 创建测试用的 BKECluster 对象
func createTestCluster() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterName,
		},
		Status: confv1beta1.BKEClusterStatus{},
	}
}

// createTestNode 创建测试用的 Kubernetes Node 对象
func createTestNode(name, ip string) corev1.Node {
	return corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Status: corev1.NodeStatus{
			Addresses: []corev1.NodeAddress{
				{
					Type:    corev1.NodeInternalIP,
					Address: ip,
				},
			},
			Conditions: []corev1.NodeCondition{
				{
					Type:   corev1.NodeReady,
					Status: corev1.ConditionTrue,
				},
			},
		},
	}
}

// createTestNodes 创建测试用的节点列表
func createTestNodes() []corev1.Node {
	return []corev1.Node{
		createTestNode(testNodeSkipName, testNodeSkipIP),
		createTestNode(testNodeCheckName, testNodeCheckIP),
	}
}

// createTestBKENodes 创建测试用的 BKENode 列表
func createTestBKENodes() bkev1beta1.BKENodes {
	return bkev1beta1.BKENodes{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: testNodeSkipName,
			},
			Spec: confv1beta1.BKENodeSpec{
				IP: testNodeSkipIP,
			},
			Status: confv1beta1.BKENodeStatus{
				NeedSkip: true,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: testNodeCheckName,
			},
			Spec: confv1beta1.BKENodeSpec{
				IP: testNodeCheckIP,
			},
			Status: confv1beta1.BKENodeStatus{
				NeedSkip: false,
			},
		},
	}
}

// verifyNodeSkipLogic 验证节点跳过逻辑
// 现在使用 BKENodes wrapper 的 GetNodeStateNeedSkip 方法
func verifyNodeSkipLogic(t *testing.T, bkeNodes bkev1beta1.BKENodes) {
	shouldSkipFirst := bkeNodes.GetNodeStateNeedSkip(testNodeSkipIP)
	shouldSkipSecond := bkeNodes.GetNodeStateNeedSkip(testNodeCheckIP)

	if !shouldSkipFirst {
		t.Errorf("Expected node %s to be skipped (needskip=true), "+
			"but GetNodeStateNeedSkip returned false", testNodeSkipIP)
	}

	if shouldSkipSecond {
		t.Errorf("Expected node %s not to be skipped (needskip=false), "+
			"but GetNodeStateNeedSkip returned true", testNodeCheckIP)
	}
}

// verifyNodeCount 验证节点数量
func verifyNodeCount(t *testing.T, nodes []corev1.Node) {
	if len(nodes) != expectedNodeCount {
		t.Errorf("Expected %d nodes in test data, but got %d", expectedNodeCount, len(nodes))
	}
}

// logTestCompletion 输出测试完成信息
func logTestCompletion(t *testing.T, cluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes, nodes []corev1.Node) {
	t.Logf("Test completed: nodes=%d, bkeNodes=%d, cluster=%s", len(nodes), len(bkeNodes), cluster.Name)
	t.Logf("Node %s (needskip=true) should be skipped", testNodeSkipIP)
	t.Logf("Node %s (needskip=false) should be checked", testNodeCheckIP)
	t.Logf("Skip logic uses BKENodes.GetNodeStateNeedSkip method in CheckClusterHealth")
}

// TestCheckClusterHealthWithSkip 测试 CheckClusterHealth 方法中的跳过逻辑
func TestCheckClusterHealthWithSkip(t *testing.T) {
	t.Run("nodes with needskip=true should be skipped in CheckClusterHealth", func(t *testing.T) {
		cluster := createTestCluster()
		nodes := createTestNodes()
		bkeNodes := createTestBKENodes()

		verifyNodeSkipLogic(t, bkeNodes)
		verifyNodeCount(t, nodes)
		logTestCompletion(t, cluster, bkeNodes, nodes)
	})
}

func TestStaticPodName(t *testing.T) {
	tests := []struct {
		component string
		nodeName  string
		want      string
	}{
		{"kube-apiserver", "master-1", "kube-apiserver-master-1"},
		{"etcd", "node-1", "etcd-node-1"},
		{"", "node", "-node"},
	}

	for _, tt := range tests {
		got := StaticPodName(tt.component, tt.nodeName)
		if got != tt.want {
			t.Errorf("StaticPodName(%q, %q) = %q, want %q", tt.component, tt.nodeName, got, tt.want)
		}
	}
}

func TestNeededComponentChecksExcludeOptionalAddons(t *testing.T) {
	for _, prefix := range []string{"coredns", "kube-proxy-"} {
		for _, check := range neededComponentChecks {
			for _, p := range check.Prefixes {
				if p == prefix {
					t.Fatalf("prefix %q should not be in neededComponentChecks", prefix)
				}
			}
		}
	}
}

func TestNodeReady(t *testing.T) {
	tests := []struct {
		name string
		node *corev1.Node
		want bool
	}{
		{
			name: "ready node",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Conditions: []corev1.NodeCondition{
						{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
					},
				},
			},
			want: true,
		},
		{
			name: "not ready node",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Conditions: []corev1.NodeCondition{
						{Type: corev1.NodeReady, Status: corev1.ConditionFalse},
					},
				},
			},
			want: false,
		},
		{
			name: "no conditions",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Conditions: []corev1.NodeCondition{},
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NodeReady(tt.node)
			if got != tt.want {
				t.Errorf("NodeReady() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCheckNodeReady(t *testing.T) {
	tests := []struct {
		name    string
		node    *corev1.Node
		wantErr bool
	}{
		{
			name: "readyNode",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Status: corev1.NodeStatus{
					Conditions: []corev1.NodeCondition{
						{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "notReadyNode",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node2"},
				Status: corev1.NodeStatus{
					Conditions: []corev1.NodeCondition{
						{Type: corev1.NodeReady, Status: corev1.ConditionFalse},
					},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkNodeReady(tt.node)
			if (err != nil) != tt.wantErr {
				t.Errorf("checkNodeReady() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCheckNodeVersion(t *testing.T) {
	const testVersion = "v1.28.0"

	tests := []struct {
		name          string
		node          *corev1.Node
		expectVersion string
		wantErr       bool
	}{
		{
			name: "matchingVersion",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Status: corev1.NodeStatus{
					NodeInfo: corev1.NodeSystemInfo{KubeletVersion: testVersion},
				},
			},
			expectVersion: testVersion,
			wantErr:       false,
		},
		{
			name: "mismatchVersion",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node2"},
				Status: corev1.NodeStatus{
					NodeInfo: corev1.NodeSystemInfo{KubeletVersion: "v1.27.0"},
				},
			},
			expectVersion: testVersion,
			wantErr:       true,
		},
		{
			name: "emptyExpectVersion",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node3"},
				Status: corev1.NodeStatus{
					NodeInfo: corev1.NodeSystemInfo{KubeletVersion: "v1.27.0"},
				},
			},
			expectVersion: "",
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkNodeVersion(tt.node, tt.expectVersion)
			if (err != nil) != tt.wantErr {
				t.Errorf("checkNodeVersion() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestFilterPodsWithPrefix(t *testing.T) {
	pods := []corev1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "coredns-123"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "coredns-456"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "etcd-master"}},
	}

	tests := []struct {
		name   string
		prefix string
		want   int
	}{
		{"corednsPrefix", "coredns", 2},
		{"etcdPrefix", "etcd", 1},
		{"noMatch", "kube-proxy", 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := filterPodsWithPrefix(pods, tt.prefix)
			if len(got) != tt.want {
				t.Errorf("filterPodsWithPrefix() = %v, want %v", len(got), tt.want)
			}
		})
	}
}

func TestFindAddonComponent(t *testing.T) {
	tests := []struct {
		name      string
		addon     string
		wantFound bool
	}{
		{"foundClusterAPI", "cluster-api", true},
		{"foundOpenfuyao", "openfuyao-system-controller", true},
		{"notFound", "unknown-addon", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, found := findAddonComponent(tt.addon)
			if found != tt.wantFound {
				t.Errorf("findAddonComponent() found = %v, want %v", found, tt.wantFound)
			}
		})
	}
}

func TestVerifyComponentPods(t *testing.T) {
	const testNamespace = "kube-system"

	readyCondition := []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
	notReadyCondition := []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}}

	tests := []struct {
		name    string
		pods    []corev1.Pod
		prefix  string
		wantErr bool
	}{
		{
			name: "allHealthy",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "etcd-master", Namespace: testNamespace},
					Status: corev1.PodStatus{
						Phase:      corev1.PodRunning,
						Conditions: readyCondition,
						ContainerStatuses: []corev1.ContainerStatus{{
							Name:  "etcd",
							Ready: true,
						}},
					},
				},
			},
			prefix:  "etcd",
			wantErr: false,
		},
		{
			name: "runningButPodNotReady",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "etcd-master", Namespace: testNamespace},
					Status: corev1.PodStatus{
						Phase:      corev1.PodRunning,
						Conditions: notReadyCondition,
						ContainerStatuses: []corev1.ContainerStatus{{
							Name:  "etcd",
							Ready: false,
						}},
					},
				},
			},
			prefix:  "etcd",
			wantErr: true,
		},
		{
			name: "runningButCrashLoopBackOff",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "etcd-master", Namespace: testNamespace},
					Status: corev1.PodStatus{
						Phase:      corev1.PodRunning,
						Conditions: readyCondition,
						ContainerStatuses: []corev1.ContainerStatus{{
							Name:  "etcd",
							Ready: false,
							State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"}},
						}},
					},
				},
			},
			prefix:  "etcd",
			wantErr: true,
		},
		{
			name:    "noPods",
			pods:    []corev1.Pod{},
			prefix:  "etcd",
			wantErr: true,
		},
		{
			name: "corednsAtLeastOneHealthy",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "coredns-1", Namespace: testNamespace},
					Status: corev1.PodStatus{
						Phase:      corev1.PodRunning,
						Conditions: notReadyCondition,
						ContainerStatuses: []corev1.ContainerStatus{{
							Name:  "coredns",
							Ready: false,
						}},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "coredns-2", Namespace: testNamespace},
					Status: corev1.PodStatus{
						Phase:      corev1.PodRunning,
						Conditions: readyCondition,
						ContainerStatuses: []corev1.ContainerStatus{{
							Name:  "coredns",
							Ready: true,
						}},
					},
				},
			},
			prefix:  "coredns",
			wantErr: false,
		},
		{
			name: "corednsAllUnhealthy",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "coredns-1", Namespace: testNamespace},
					Status: corev1.PodStatus{
						Phase:      corev1.PodRunning,
						Conditions: notReadyCondition,
						ContainerStatuses: []corev1.ContainerStatus{{
							Name:  "coredns",
							Ready: false,
							State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"}},
						}},
					},
				},
			},
			prefix:  "coredns",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{}
			err := client.verifyComponentPods(tt.pods, tt.prefix, testNamespace)
			if (err != nil) != tt.wantErr {
				t.Errorf("verifyComponentPods() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestNodeHealthCheck(t *testing.T) {
	const testVersion = "v1.28.0"

	tests := []struct {
		name              string
		node              *corev1.Node
		expectVersion     string
		componentCheckErr error
		wantErr           bool
	}{
		{
			name: "healthyMasterNode",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "master-1",
					Labels: map[string]string{"node-role.kubernetes.io/master": ""},
				},
				Status: corev1.NodeStatus{
					Conditions: []corev1.NodeCondition{
						{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
					},
					NodeInfo: corev1.NodeSystemInfo{KubeletVersion: testVersion},
				},
			},
			expectVersion:     testVersion,
			componentCheckErr: nil,
			wantErr:           false,
		},
		{
			name: "notReadyNode",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
				Status: corev1.NodeStatus{
					Conditions: []corev1.NodeCondition{
						{Type: corev1.NodeReady, Status: corev1.ConditionFalse},
					},
				},
			},
			expectVersion: testVersion,
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				Log: nil,
			}

			patches := gomonkey.ApplyMethod(client, "CheckComponentHealth",
				func(_ *Client, _ *corev1.Node) error {
					return tt.componentCheckErr
				})
			defer patches.Reset()

			err := client.NodeHealthCheck(tt.node, tt.expectVersion, client.Log)
			if (err != nil) != tt.wantErr {
				t.Errorf("NodeHealthCheck() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCheckComponentHealth(t *testing.T) {
	tests := []struct {
		name      string
		node      *corev1.Node
		getPodErr error
		podPhase  corev1.PodPhase
		wantErr   bool
	}{
		{
			name: "allComponentsRunning",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "master-1"},
			},
			getPodErr: nil,
			podPhase:  corev1.PodRunning,
			wantErr:   false,
		},
		{
			name: "componentNotRunning",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "master-2"},
			},
			getPodErr: nil,
			podPhase:  corev1.PodPending,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{}

			patches := gomonkey.ApplyMethod(client, "GetPod",
				func(_ *Client, _, _ string) (*corev1.Pod, error) {
					if tt.getPodErr != nil {
						return nil, tt.getPodErr
					}
					return &corev1.Pod{
						Status: corev1.PodStatus{Phase: tt.podPhase},
					}, nil
				})
			defer patches.Reset()

			err := client.CheckComponentHealth(tt.node)
			if (err != nil) != tt.wantErr {
				t.Errorf("CheckComponentHealth() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetPods(t *testing.T) {
	const testNamespace = "kube-system"

	client := &Client{
		ClientSet: &kubernetes.Clientset{},
		Ctx:       context.Background(),
	}

	patches := gomonkey.ApplyMethod(client.ClientSet.CoreV1().Pods(testNamespace), "List",
		func(_ interface{}, _ context.Context, _ metav1.ListOptions) (*corev1.PodList, error) {
			return &corev1.PodList{
				Items: []corev1.Pod{
					{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
				},
			}, nil
		})
	defer patches.Reset()

	pods, err := client.getPods(testNamespace)
	if err != nil {
		t.Errorf("getPods() error = %v", err)
	}
	if len(pods) != 1 {
		t.Errorf("getPods() returned %d pods, want 1", len(pods))
	}
}

func TestGetPodsError(t *testing.T) {
	const testNamespace = "kube-system"

	client := &Client{
		ClientSet: &kubernetes.Clientset{},
		Ctx:       context.Background(),
	}

	patches := gomonkey.ApplyMethod(client.ClientSet.CoreV1().Pods(testNamespace), "List",
		func(_ interface{}, _ context.Context, _ metav1.ListOptions) (*corev1.PodList, error) {
			return nil, errors.New("list failed")
		})
	defer patches.Reset()

	_, err := client.getPods(testNamespace)
	if err == nil {
		t.Error("getPods() expected error")
	}
}

func TestProcessAddonComponentCheck(t *testing.T) {
	client := &Client{}
	err := client.processAddonComponentCheck("unknown-addon")
	if err == nil {
		t.Error("processAddonComponentCheck() expected error for unknown addon")
	}
}

func TestOptionalAddonComponentsIncludeCoreDNSKubeProxyAndCalico(t *testing.T) {
	cases := []struct {
		addon    string
		prefixes []string
	}{
		{addon: "coredns", prefixes: []string{"coredns"}},
		{addon: "kubeproxy", prefixes: []string{"kube-proxy-"}},
		{addon: "calico", prefixes: []string{"calico-kube-controllers-", "calico-node-"}},
	}
	for _, tc := range cases {
		addonCheck, found := findAddonComponent(tc.addon)
		if !found {
			t.Fatalf("addon %q should be registered in extraAddonComponents", tc.addon)
		}
		if len(addonCheck.Components) == 0 || len(addonCheck.Components[0].Prefixes) == 0 {
			t.Fatalf("addon %q should have component prefixes configured", tc.addon)
		}
		if len(addonCheck.Components[0].Prefixes) != len(tc.prefixes) {
			t.Fatalf("addon %q prefixes = %v, want %v", tc.addon, addonCheck.Components[0].Prefixes, tc.prefixes)
		}
		for i, prefix := range tc.prefixes {
			if addonCheck.Components[0].Prefixes[i] != prefix {
				t.Fatalf("addon %q prefix[%d] = %q, want %q", tc.addon, i, addonCheck.Components[0].Prefixes[i], prefix)
			}
		}
	}
}

func TestDefaultHealthCheckConfigContainsCoreComponents(t *testing.T) {
	config := DefaultHealthCheckConfig()
	if len(config.Components) != 8 {
		t.Fatalf("DefaultHealthCheckConfig components = %d, want 8", len(config.Components))
	}

	want := map[ComponentName]HealthCheckPriority{
		NameEtcd:                  PriorityCritical,
		NameKubeAPIServer:         PriorityCritical,
		NameKubeControllerManager: PriorityCritical,
		NameKubeScheduler:         PriorityCritical,
		NameCalicoNode:            PriorityImportant,
		NameCalicoKubeControllers: PriorityImportant,
		NameKubeProxy:             PriorityImportant,
		NameCoreDNS:               PriorityImportant,
	}
	got := make(map[ComponentName]HealthCheckPriority, len(config.Components))
	for _, component := range config.Components {
		got[component.Name] = component.Priority
	}
	for name, priority := range want {
		if got[name] != priority {
			t.Fatalf("component %s priority = %s, want %s", name, got[name], priority)
		}
	}
}

func TestHealthCheckErrorPriorityHelpers(t *testing.T) {
	criticalErr := &HealthCheckError{
		Component: ComponentInfo{Name: NameEtcd, Namespace: metav1.NamespaceSystem, Priority: PriorityCritical},
		Reason:    "PodNotReady",
		Err:       errors.New("etcd not ready"),
	}
	importantErr := &HealthCheckError{
		Component: ComponentInfo{Name: NameCoreDNS, Namespace: metav1.NamespaceSystem, Priority: PriorityImportant},
		Reason:    "PodNotReady",
		Err:       errors.New("coredns not ready"),
	}
	agg := kerrors.NewAggregate([]error{criticalErr, importantErr})

	if !IsCriticalHealthCheckError(agg) {
		t.Fatal("aggregate error should contain critical health check error")
	}
	if !IsImportantHealthCheckError(agg) {
		t.Fatal("aggregate error should contain important health check error")
	}
	if len(ComponentErrorsByPriority(agg, PriorityCritical)) != 1 {
		t.Fatal("expected one critical health check error")
	}
	if len(ComponentErrorsByPriority(agg, PriorityImportant)) != 1 {
		t.Fatal("expected one important health check error")
	}
}

func TestGetHealthCheckRequeueInterval(t *testing.T) {
	if got := GetHealthCheckRequeueInterval(nil); got != 5*time.Minute {
		t.Fatalf("normal requeue = %v, want 5m", got)
	}

	criticalErr := &HealthCheckError{Component: ComponentInfo{Priority: PriorityCritical}, Err: errors.New("critical")}
	if got := GetHealthCheckRequeueInterval(criticalErr); got != 5*time.Second {
		t.Fatalf("critical requeue = %v, want 5s", got)
	}

	importantErr := &HealthCheckError{Component: ComponentInfo{Priority: PriorityImportant}, Err: errors.New("important")}
	if got := GetHealthCheckRequeueInterval(importantErr); got != 15*time.Second {
		t.Fatalf("important requeue = %v, want 15s", got)
	}

	wrappedImportantErr := errors.Wrap(kerrors.NewAggregate([]error{importantErr}), "CheckClusterHealth failed")
	if got := GetHealthCheckRequeueInterval(wrappedImportantErr); got != 15*time.Second {
		t.Fatalf("wrapped important requeue = %v, want 15s", got)
	}
}

func TestGetHealthCheckRequeueIntervalUsesWrappedConfigIntervals(t *testing.T) {
	importantErr := &HealthCheckError{Component: ComponentInfo{Priority: PriorityImportant}, Err: errors.New("important")}
	intervals := IntervalConfig{Critical: time.Second, Important: 45 * time.Second, Optional: time.Minute, Normal: 5 * time.Minute}
	wrappedErr := newHealthCheckRequeueError(errors.Wrap(kerrors.NewAggregate([]error{importantErr}), "CheckClusterHealth failed"), intervals)
	if got := GetHealthCheckRequeueInterval(wrappedErr); got != 45*time.Second {
		t.Fatalf("wrapped config important requeue = %v, want 45s", got)
	}
}

func TestUnifiedHealthCheckerKeepsCoreDNSSingleHealthySemantics(t *testing.T) {
	checker := NewUnifiedHealthChecker(&Client{}, DefaultHealthCheckConfig())
	pods := []corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "coredns-unready", Namespace: metav1.NamespaceSystem},
			Status: corev1.PodStatus{
				Phase: corev1.PodRunning,
				Conditions: []corev1.PodCondition{
					{Type: corev1.PodReady, Status: corev1.ConditionFalse},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "coredns-ready", Namespace: metav1.NamespaceSystem},
			Status: corev1.PodStatus{
				Phase: corev1.PodRunning,
				Conditions: []corev1.PodCondition{
					{Type: corev1.PodReady, Status: corev1.ConditionTrue},
				},
				ContainerStatuses: []corev1.ContainerStatus{{Name: "coredns", Ready: true}},
			},
		},
	}
	check := ComponentCheck{Name: NameCoreDNS, Namespace: metav1.NamespaceSystem, Prefixes: []string{"coredns"}, Priority: PriorityImportant}
	if err := checker.verifyComponentPods(check, pods, "coredns"); err != nil {
		t.Fatalf("coredns should pass when at least one pod is healthy: %v", err)
	}
}

func TestCoreHealthAddonsAreSkippedOnlyInAddonContinuation(t *testing.T) {
	for _, addon := range []string{"calico", "coredns", "kubeproxy"} {
		if _, covered := coreHealthAddons[addon]; !covered {
			t.Fatalf("addon %q should be covered by core health check", addon)
		}
		if _, found := findAddonComponent(addon); !found {
			t.Fatalf("addon %q should remain registered in extraAddonComponents", addon)
		}
	}
}

func TestHealthCheckCacheCachesNodesAndPods(t *testing.T) {
	client := &Client{Ctx: context.Background()}
	nodeCalls := 0
	patches := gomonkey.ApplyMethod(client, "ListNodes", func(_ *Client, _ *metav1.ListOptions) (*corev1.NodeList, error) {
		nodeCalls++
		return &corev1.NodeList{Items: []corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}}, nil
	})
	defer patches.Reset()
	cache := newHealthCheckCache(client)
	cache.pods[metav1.NamespaceSystem] = []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: metav1.NamespaceSystem}}}

	nodes, err := cache.GetNodes()
	if err != nil {
		t.Fatalf("GetNodes() error = %v", err)
	}
	cachedNodes, err := cache.GetNodes()
	if err != nil {
		t.Fatalf("second GetNodes() error = %v", err)
	}
	if nodes != cachedNodes || len(cachedNodes.Items) != 1 || nodeCalls != 1 {
		t.Fatalf("GetNodes() did not return cached node list")
	}

	pods, err := cache.GetPods(metav1.NamespaceSystem)
	if err != nil {
		t.Fatalf("GetPods() error = %v", err)
	}
	cachedPods, err := cache.GetPods(metav1.NamespaceSystem)
	if err != nil {
		t.Fatalf("second GetPods() error = %v", err)
	}
	if len(cachedPods) != 1 || pods[0].Name != cachedPods[0].Name {
		t.Fatalf("GetPods() did not return cached pod list")
	}
}

func TestHealthCheckCachePropagatesListErrors(t *testing.T) {
	client := &Client{Ctx: context.Background()}
	patches := gomonkey.ApplyMethod(client, "ListNodes", func(_ *Client, _ *metav1.ListOptions) (*corev1.NodeList, error) {
		return nil, errors.New("list nodes failed")
	})
	defer patches.Reset()
	cache := newHealthCheckCache(client)
	if _, err := cache.GetNodes(); err == nil {
		t.Fatal("GetNodes() expected error")
	}
}

func TestHealthUnifiedSmallBranches(t *testing.T) {
	if got := PriorityOptional.String(); got != "optional" {
		t.Fatalf("optional priority string = %q", got)
	}
	if got := HealthCheckPriority(99).String(); got != "unknown" {
		t.Fatalf("unknown priority string = %q", got)
	}
	if got := (ComponentInfo{Name: NameEtcd}).String(); got != "etcd" {
		t.Fatalf("component without namespace = %q", got)
	}
	if got := (ComponentInfo{Name: NameCoreDNS, Namespace: metav1.NamespaceSystem, Prefix: "coredns"}).String(); got != "kube-system/coredns(coredns)" {
		t.Fatalf("component with prefix = %q", got)
	}

	hcErr := &HealthCheckError{Component: ComponentInfo{Name: NameNode, Priority: PriorityCritical}, Reason: "NodeNotReady", Err: errors.New("boom")}
	if hcErr.Error() == "" {
		t.Fatal("HealthCheckError Error() should not be empty")
	}

	config := HealthCheckConfig{Components: []ComponentCheck{{Name: NameEtcd, Namespace: metav1.NamespaceSystem, Prefixes: []string{"etcd-"}, Priority: PriorityCritical}}}
	if got := healthCheckConfigSummary(config); got != "etcd:kube-system:etcd-:critical" {
		t.Fatalf("healthCheckConfigSummary() = %q", got)
	}
}

func TestHealthUnifiedAggregateNodeAndRequeueBranches(t *testing.T) {
	checker := NewUnifiedHealthChecker(&Client{}, DefaultHealthCheckConfig())
	if err := checker.aggregateResult(&HealthCheckResult{}); err != nil {
		t.Fatalf("empty aggregateResult() error = %v", err)
	}

	nodeErr := newNodeError("node-1", "NodeNotReady", errors.New("not ready"))
	if nodeErr.Component.Name != ComponentName("node-1") || nodeErr.Component.Priority != PriorityCritical {
		t.Fatalf("unexpected node health error: %+v", nodeErr)
	}
	defaultNodeErr := newNodeError("", "ListNodesFailed", errors.New("list failed"))
	if defaultNodeErr.Component.Name != NameNode {
		t.Fatalf("empty node name component = %s, want %s", defaultNodeErr.Component.Name, NameNode)
	}

	aggErr := checker.aggregateResult(&HealthCheckResult{
		NodeErrors:               []error{nodeErr},
		ImportantComponentErrors: []error{&HealthCheckError{Component: ComponentInfo{Name: NameCoreDNS, Priority: PriorityImportant}, Reason: "PodNotReady", Err: errors.New("important")}},
		OptionalComponentErrors:  []error{&HealthCheckError{Component: ComponentInfo{Name: ComponentName("optional"), Priority: PriorityOptional}, Reason: "PodNotFound", Err: errors.New("optional")}},
	})
	if aggErr == nil || !IsCriticalHealthCheckError(aggErr) || !IsImportantHealthCheckError(aggErr) {
		t.Fatalf("aggregateResult() did not preserve priorities: %v", aggErr)
	}

	wrapped := newHealthCheckRequeueError(aggErr, IntervalConfig{Critical: time.Second, Important: 2 * time.Second, Optional: 3 * time.Second, Normal: 4 * time.Second})
	requeueErr, ok := wrapped.(*healthCheckRequeueError)
	if !ok {
		t.Fatalf("newHealthCheckRequeueError type = %T", wrapped)
	}
	if requeueErr.Error() != aggErr.Error() || requeueErr.Unwrap().Error() != aggErr.Error() {
		t.Fatal("healthCheckRequeueError should proxy Error and Unwrap")
	}
	if got := requeueErr.RequeueInterval(); got != time.Second {
		t.Fatalf("RequeueInterval() = %v, want 1s", got)
	}
}

func TestGetPodUnhealthyReasonBranches(t *testing.T) {
	runningReady := corev1.Pod{Status: corev1.PodStatus{
		Phase:      corev1.PodRunning,
		Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
	}}
	if got := getPodUnhealthyReason(runningReady); got != "PodUnhealthy" {
		t.Fatalf("ready pod fallback reason = %s", got)
	}
	if got := getPodUnhealthyReason(corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodPending}}); got != "PodNotRunning" {
		t.Fatalf("pending pod reason = %s", got)
	}
	notReady := runningReady
	notReady.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}}
	if got := getPodUnhealthyReason(notReady); got != "PodNotReady" {
		t.Fatalf("not ready pod reason = %s", got)
	}
	initCrash := runningReady
	initCrash.Status.InitContainerStatuses = []corev1.ContainerStatus{{State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"}}}}
	if got := getPodUnhealthyReason(initCrash); got != "CrashLoopBackOff" {
		t.Fatalf("init crash reason = %s", got)
	}
	containerCrash := runningReady
	containerCrash.Status.ContainerStatuses = []corev1.ContainerStatus{{State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "ImagePullBackOff"}}}}
	if got := getPodUnhealthyReason(containerCrash); got != "ImagePullBackOff" {
		t.Fatalf("container crash reason = %s", got)
	}
}
