/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Muluan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// setupKubeletConfigFile 为测试创建临时 kubelet 配置文件并返回还原函数
func setupKubeletConfigFile(t *testing.T, content string) func() {
	t.Helper()
	oldPath := KubeletConfigPath
	KubeletConfigPath = filepath.Join(t.TempDir(), "kubelet-config.yaml")
	if err := os.WriteFile(KubeletConfigPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write temp kubelet config: %v", err)
	}
	return func() {
		KubeletConfigPath = oldPath
	}
}

// TestAddNodeToExistingConfig_IdempotentForExistingNode 验证节点已存在时不追加重复项（issue 91 复发路径）
func TestAddNodeToExistingConfig_IdempotentForExistingNode(t *testing.T) {
	ctx := context.Background()
	testScheme := setupTestScheme()
	restore := setupKubeletConfigFile(t, "cpuManagerPolicy: none\n")
	defer restore()

	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kubelet-config",
			Namespace: "default",
		},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "node-a", EnableStaticCPUManager: true, KubeletConfigPath: KubeletConfigPath},
			},
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(nodeConfig).
		Build()

	// 对已存在的 node-a 再次调用，应幂等返回且不追加
	err := addNodeToExistingConfig(ctx, cl, nil, nodeConfig, "node-a")
	if err != nil {
		t.Fatalf("addNodeToExistingConfig failed: %v", err)
	}

	updated := &NodeConfig{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "kubelet-config", Namespace: "default"}, updated); err != nil {
		t.Fatalf("failed to get updated NodeConfig: %v", err)
	}

	if len(updated.Spec.Nodes) != 1 {
		t.Errorf("expected 1 node after idempotent add, got %d", len(updated.Spec.Nodes))
	}
	if updated.Spec.Nodes[0].NodeName != "node-a" {
		t.Errorf("expected node-a, got %s", updated.Spec.Nodes[0].NodeName)
	}
}

// TestCreateNodeConfigCR_AlreadyExistsNoDuplicate 验证 AlreadyExists 分支不产生重复节点
func TestCreateNodeConfigCR_AlreadyExistsNoDuplicate(t *testing.T) {
	ctx := context.Background()
	testScheme := setupTestScheme()
	restore := setupKubeletConfigFile(t, "cpuManagerPolicy: none\n")
	defer restore()

	// 预置已包含 node-a 的 CR，模拟另一节点已先创建
	existing := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kubelet-config",
			Namespace: "default",
		},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "node-a", EnableStaticCPUManager: false, KubeletConfigPath: KubeletConfigPath},
			},
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(existing).
		Build()

	// node-a 尝试创建 CR 会得到 AlreadyExists，走新增分支后不应出现重复
	err := createNodeConfigCR(ctx, cl, "node-a", false, false)
	if err != nil {
		t.Fatalf("createNodeConfigCR failed: %v", err)
	}

	updated := &NodeConfig{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "kubelet-config", Namespace: "default"}, updated); err != nil {
		t.Fatalf("failed to get updated NodeConfig: %v", err)
	}

	count := 0
	for _, nc := range updated.Spec.Nodes {
		if nc.NodeName == "node-a" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected exactly 1 entry for node-a after AlreadyExists branch, got %d", count)
	}
}

// TestDeduplicateNodeConfigurations 验证按 nodeName 去重保留首次出现项
func TestDeduplicateNodeConfigurations(t *testing.T) {
	tests := []struct {
		name      string
		input     []NodeConfiguration
		wantNil   bool
		wantOrder []string
	}{
		{
			name: "无重复返回nil",
			input: []NodeConfiguration{
				{NodeName: "a"}, {NodeName: "b"},
			},
			wantNil:   true,
			wantOrder: nil,
		},
		{
			name: "保留首次出现项",
			input: []NodeConfiguration{
				{NodeName: "a", EnableStaticCPUManager: true},
				{NodeName: "b"},
				{NodeName: "a", EnableStaticCPUManager: false},
			},
			wantNil:   false,
			wantOrder: []string{"a", "b"},
		},
		{
			name:      "空列表返回nil",
			input:     []NodeConfiguration{},
			wantNil:   true,
			wantOrder: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := deduplicateNodeConfigurations(tt.input)
			if tt.wantNil {
				if got != nil {
					t.Errorf("expected nil, got %v", got)
				}
				return
			}
			if got == nil {
				t.Fatal("expected non-nil result")
			}
			if len(got) != len(tt.wantOrder) {
				t.Fatalf("expected %d nodes, got %d", len(tt.wantOrder), len(got))
			}
			for i, name := range tt.wantOrder {
				if got[i].NodeName != name {
					t.Errorf("position %d: expected %s, got %s", i, name, got[i].NodeName)
				}
			}
			// 验证保留的是首次出现项的值
			if got[0].NodeName == "a" && !got[0].EnableStaticCPUManager {
				t.Errorf("expected first occurrence value (true) to be preserved")
			}
		})
	}
}

// TestUpdateNodeStatusWithRetry_NoIndexOutOfRange 验证 Status 条目数超过 Spec.Nodes 时不 panic（issue 90）
func TestUpdateNodeStatusWithRetry_NoIndexOutOfRange(t *testing.T) {
	ctx := context.Background()
	testScheme := setupTestScheme()

	// 构造 Spec.Nodes 只有 1 个节点、Status 却有 3 个条目的不一致状态
	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kubelet-config",
			Namespace: "default",
		},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "node-a", EnableStaticCPUManager: true, EnableNUMADistance: true},
			},
		},
		Status: NodeConfigStatus{
			NodeStatuses: []NodeStatus{
				{NodeName: "node-a", State: "pending"},
				{NodeName: "node-b", State: "pending"},
				{NodeName: "node-c", State: "pending"},
			},
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(nodeConfig).
		WithStatusSubresource(&NodeConfig{}).
		Build()

	r := &NodeConfigController{Client: cl}

	// 更新 node-a 的状态为 configured：旧实现会用 Status 下标 0 索引 Spec.Nodes[0]，
	// 但若命中的是 node-b（下标 1）就会越界。这里对每个节点都执行一次，
	// 确保任何下标组合都不 panic。
	for _, nodeName := range []string{"node-a", "node-b", "node-c"} {
		err := r.updateNodeStatusWithRetry(ctx, nodeConfig, nodeName, NodeConfigStateConfigured, "")
		if err != nil {
			t.Fatalf("updateNodeStatusWithRetry failed for %s: %v", nodeName, err)
		}
	}

	updated := &NodeConfig{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "kubelet-config", Namespace: "default"}, updated); err != nil {
		t.Fatalf("failed to get updated NodeConfig: %v", err)
	}

	// node-a 在 Spec 中有配置，configured 状态应带上正确的开关值
	for _, status := range updated.Status.NodeStatuses {
		if status.NodeName == "node-a" {
			if status.State != NodeConfigStateConfigured {
				t.Errorf("expected node-a state=%s, got %s", NodeConfigStateConfigured, status.State)
			}
			if !status.StaticCPUManagerEnabled || !status.NUMADistanceEnabled {
				t.Errorf("expected node-a flags from Spec.Nodes to be applied, got static=%v numa=%v",
					status.StaticCPUManagerEnabled, status.NUMADistanceEnabled)
			}
		}
	}
}

// TestNodeConfigurationExists 验证存在性检查
func TestNodeConfigurationExists(t *testing.T) {
	nodes := []NodeConfiguration{
		{NodeName: "node-a"},
		{NodeName: "node-b"},
	}

	if !nodeConfigurationExists(nodes, "node-a") {
		t.Errorf("expected node-a to exist")
	}
	if nodeConfigurationExists(nodes, "node-c") {
		t.Errorf("expected node-c to NOT exist")
	}
	if nodeConfigurationExists(nil, "node-a") {
		t.Errorf("expected nil list to NOT contain node-a")
	}
}

// TestReconcile_DeduplicatesExistingDuplicateNodes 验证 Reconcile 对存量重复节点
// 的去重结果真正持久化到 CR（回归：去重结果必须回写到对象再提交）
func TestReconcile_DeduplicatesExistingDuplicateNodes(t *testing.T) {
	ctx := context.Background()
	testScheme := setupTestScheme()
	restore := setupKubeletConfigFile(t, "cpuManagerPolicy: none\n")
	defer restore()

	// 历史脏数据：node-a 出现两次（issue 91 现场），首次出现开关为 false
	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "kubelet-config", Namespace: "default"},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "node-a", EnableStaticCPUManager: false, EnableNUMADistance: false, KubeletConfigPath: KubeletConfigPath},
				{NodeName: "node-b", EnableStaticCPUManager: true, EnableNUMADistance: true, KubeletConfigPath: KubeletConfigPath},
				{NodeName: "node-a", EnableStaticCPUManager: true, EnableNUMADistance: true, KubeletConfigPath: KubeletConfigPath},
			},
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(nodeConfig).
		WithStatusSubresource(&NodeConfig{}).
		Build()

	r := &NodeConfigController{Client: cl, Scheme: testScheme}
	t.Setenv("NODE_NAME", "node-a")

	_, err := r.Reconcile(ctx, ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "kubelet-config", Namespace: "default"},
	})
	if err != nil {
		t.Fatalf("Reconcile failed: %v", err)
	}

	updated := &NodeConfig{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "kubelet-config", Namespace: "default"}, updated); err != nil {
		t.Fatalf("failed to get updated NodeConfig: %v", err)
	}

	if len(updated.Spec.Nodes) != 2 {
		t.Fatalf("expected 2 nodes after dedup, got %d", len(updated.Spec.Nodes))
	}

	// node-a 应只剩一条，且保留首次出现的开关值（false）
	count := 0
	for _, nc := range updated.Spec.Nodes {
		if nc.NodeName == "node-a" {
			count++
			if nc.EnableStaticCPUManager || nc.EnableNUMADistance {
				t.Errorf("expected first occurrence flags (false) to be preserved, got static=%v numa=%v",
					nc.EnableStaticCPUManager, nc.EnableNUMADistance)
			}
		}
	}
	if count != 1 {
		t.Errorf("expected exactly 1 entry for node-a after Reconcile dedup, got %d", count)
	}
}

// TestReconcile_HealsStuckConfiguringStatus 验证 kubelet 配置已生效但 status
// 卡在 Configuring 时，Reconcile 的无变化分支会自愈补写 Configured（issue 92）。
// 场景：多节点并发写 status 丢更新，导致部分节点状态永远停留在 Configuring。
func TestReconcile_HealsStuckConfiguringStatus(t *testing.T) {
	ctx := context.Background()
	testScheme := setupTestScheme()
	// kubelet 配置已与期望一致（static + numa distance 均已开启）
	restore := setupKubeletConfigFile(t, "cpuManagerPolicy: static\n"+
		"topologyManagerPolicyOptions:\n  prefer-closest-numa-nodes: \"true\"\n")
	defer restore()

	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "kubelet-config", Namespace: "default"},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "node-a", EnableStaticCPUManager: true,
					EnableNUMADistance: true, KubeletConfigPath: KubeletConfigPath},
			},
		},
		Status: NodeConfigStatus{
			NodeStatuses: []NodeStatus{
				// 模拟并发丢更新：配置已生效但状态卡在 Configuring 且开关值不对
				{NodeName: "node-a", State: NodeConfigStateConfiguring,
					StaticCPUManagerEnabled: false, NUMADistanceEnabled: false},
			},
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(nodeConfig).
		WithStatusSubresource(&NodeConfig{}).
		Build()

	r := &NodeConfigController{Client: cl, Scheme: testScheme}
	t.Setenv("NODE_NAME", "node-a")

	_, err := r.Reconcile(ctx, ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "kubelet-config", Namespace: "default"},
	})
	if err != nil {
		t.Fatalf("Reconcile failed: %v", err)
	}

	updated := &NodeConfig{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "kubelet-config", Namespace: "default"}, updated); err != nil {
		t.Fatalf("failed to get updated NodeConfig: %v", err)
	}

	found := false
	for _, status := range updated.Status.NodeStatuses {
		if status.NodeName == "node-a" {
			found = true
			if status.State != NodeConfigStateConfigured {
				t.Errorf("expected healed state=%s, got %s", NodeConfigStateConfigured, status.State)
			}
			if !status.StaticCPUManagerEnabled || !status.NUMADistanceEnabled {
				t.Errorf("expected healed flags static=true numa=true, got static=%v numa=%v",
					status.StaticCPUManagerEnabled, status.NUMADistanceEnabled)
			}
		}
	}
	if !found {
		t.Fatal("expected node-a status entry to exist")
	}
}

// TestReconcile_NoStatusChurnWhenAlreadyConfigured 验证状态已正确时，
// 无变化分支不产生额外的 status 写入（避免周期性 Requeue 造成的 resourceVersion 抖动）。
func TestReconcile_NoStatusChurnWhenAlreadyConfigured(t *testing.T) {
	ctx := context.Background()
	testScheme := setupTestScheme()
	restore := setupKubeletConfigFile(t, "cpuManagerPolicy: static\n"+
		"topologyManagerPolicyOptions:\n  prefer-closest-numa-nodes: \"true\"\n")
	defer restore()

	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "kubelet-config", Namespace: "default"},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "node-a", EnableStaticCPUManager: true,
					EnableNUMADistance: true, KubeletConfigPath: KubeletConfigPath},
			},
		},
		Status: NodeConfigStatus{
			NodeStatuses: []NodeStatus{
				{NodeName: "node-a", State: NodeConfigStateConfigured,
					StaticCPUManagerEnabled: true, NUMADistanceEnabled: true},
			},
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(nodeConfig).
		WithStatusSubresource(&NodeConfig{}).
		Build()

	origRV := nodeConfig.ResourceVersion
	_ = origRV

	r := &NodeConfigController{Client: cl, Scheme: testScheme}
	t.Setenv("NODE_NAME", "node-a")

	_, err := r.Reconcile(ctx, ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "kubelet-config", Namespace: "default"},
	})
	if err != nil {
		t.Fatalf("Reconcile failed: %v", err)
	}

	updated := &NodeConfig{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "kubelet-config", Namespace: "default"}, updated); err != nil {
		t.Fatalf("failed to get updated NodeConfig: %v", err)
	}

	// 状态未被改写（fake client 每次成功 Update 都会递增 RV）
	if len(updated.Status.NodeStatuses) != 1 {
		t.Fatalf("expected 1 status entry, got %d", len(updated.Status.NodeStatuses))
	}
	if updated.Status.NodeStatuses[0].State != NodeConfigStateConfigured {
		t.Errorf("expected state to remain %s, got %s",
			NodeConfigStateConfigured, updated.Status.NodeStatuses[0].State)
	}
}

// TestUpdateNodeStatusWithRetry_APIReaderOnlyWritesOwnEntry 验证设置 APIReader 后
// 状态更新基于 apiserver 最新数据，且不覆盖其他节点的状态条目。
func TestUpdateNodeStatusWithRetry_APIReaderOnlyWritesOwnEntry(t *testing.T) {
	ctx := context.Background()
	testScheme := setupTestScheme()

	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "kubelet-config", Namespace: "default"},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "node-a", EnableStaticCPUManager: true, EnableNUMADistance: true},
				{NodeName: "node-b", EnableStaticCPUManager: false, EnableNUMADistance: false},
			},
		},
		Status: NodeConfigStatus{
			NodeStatuses: []NodeStatus{
				{NodeName: "node-b", State: NodeConfigStateConfigured,
					StaticCPUManagerEnabled: false, NUMADistanceEnabled: false},
			},
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(nodeConfig).
		WithStatusSubresource(&NodeConfig{}).
		Build()

	// APIReader 与 cache client 使用同一 fake client
	r := &NodeConfigController{Client: cl, APIReader: cl}

	err := r.updateNodeStatusWithRetry(ctx, nodeConfig, "node-a", NodeConfigStateConfigured, "")
	if err != nil {
		t.Fatalf("updateNodeStatusWithRetry failed: %v", err)
	}

	updated := &NodeConfig{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "kubelet-config", Namespace: "default"}, updated); err != nil {
		t.Fatalf("failed to get updated NodeConfig: %v", err)
	}

	statuses := map[string]NodeStatus{}
	for _, s := range updated.Status.NodeStatuses {
		statuses[s.NodeName] = s
	}
	if len(statuses) != 2 {
		t.Fatalf("expected 2 status entries, got %d", len(statuses))
	}

	if s, ok := statuses["node-a"]; !ok || s.State != NodeConfigStateConfigured ||
		!s.StaticCPUManagerEnabled || !s.NUMADistanceEnabled {
		t.Errorf("node-a status not correctly written: %+v", s)
	}
	// node-b 的已有状态不被覆盖
	if s, ok := statuses["node-b"]; !ok || s.State != NodeConfigStateConfigured {
		t.Errorf("node-b status was overwritten: %+v", s)
	}
}
