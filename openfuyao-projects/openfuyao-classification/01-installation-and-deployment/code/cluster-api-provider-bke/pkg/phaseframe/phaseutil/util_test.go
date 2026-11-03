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

package phaseutil

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/coreos/go-semver/semver"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kubernetes "k8s.io/client-go/kubernetes"
	"sigs.k8s.io/cluster-api/util/version"
	"sigs.k8s.io/controller-runtime/pkg/client"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	clusterpkg "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/cluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
)

func compareVersions(oldVersion, newVersion string) (int, error) {
	oldv, err := version.ParseMajorMinorPatch(oldVersion)
	if err != nil {
		fmt.Printf("parse %v err %v\n", oldVersion, err)
		return 0, err
	}
	newv, err := version.ParseMajorMinorPatch(newVersion)
	if err != nil {
		fmt.Printf("parse %v err %v\n", newVersion, err)
		return 0, err
	}

	// step 2 compare cluster version upgrade
	return version.Compare(newv, oldv), nil
}

func TestVersionCompare(t *testing.T) {
	tests := []struct {
		name        string
		oldVersion  string
		newVersion  string
		expected    int
		shouldError bool
	}{
		{
			name:       "same versions",
			oldVersion: "v25.9.0",
			newVersion: "v25.9.0",
			expected:   0,
		},
		{
			name:       "old less than new - patch",
			oldVersion: "v25.9.0",
			newVersion: "v25.9.1",
			expected:   1,
		},
		{
			name:       "old greater than new - patch",
			oldVersion: "v25.9.1",
			newVersion: "v25.9.0",
			expected:   -1,
		},
		{
			name:        "version is illegal",
			oldVersion:  "v25.09",
			newVersion:  "v25.9",
			expected:    1,
			shouldError: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := compareVersions(tt.oldVersion, tt.newVersion)
			if err != nil {
				if !tt.shouldError {
					t.Errorf("compareVersions() error = %v, wantErr %v", err, tt.shouldError)
				}
			} else {
				if result != tt.expected {
					t.Errorf("compareVersions(%s, %s) = %d, expected %d",
						tt.oldVersion, tt.newVersion, result, tt.expected)
				}
			}
		})
	}
}

func isPatchVersion(version string) bool {
	cleanVersion := strings.TrimPrefix(version, "v")

	v, err := semver.NewVersion(cleanVersion)
	if err != nil {
		return false
	}

	return v.Patch > 0 && v.PreRelease == ""
}

func TestPatchVersion(t *testing.T) {
	tests := []struct {
		name    string
		version string
		isPatch bool
	}{
		{
			name:    "not patch case 1",
			version: "v25.09.0",
			isPatch: false,
		},
		{
			name:    "not patch case 2",
			version: "v25.9",
			isPatch: false,
		},
		{
			name:    "not patch case 3",
			version: "v25.09.1.rc",
			isPatch: false,
		},
		{
			name:    "patch case 1",
			version: "v25.09.1",
			isPatch: true,
		},
		{
			name:    "patch case 2",
			version: "v25.9.2",
			isPatch: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isPatch := isPatchVersion(tt.version)
			if isPatch != tt.isPatch {
				t.Errorf("version %v should %v but %v", tt.version, tt.isPatch, isPatch)
			}
		})
	}
}

func TestNeedUpgradeWithOF(t *testing.T) {
	tests := []struct {
		old  string
		new  string
		want bool
	}{
		{"v1.2.1-of.1", "v1.2.1-of.2", true}, // of 编号不同
		{"v1.2-of.1", "v1.2.1-of.1", true},   // 修订号不同
		{"v1.2.1-of.2", "v1.2.2-of.1", true}, // 修订号更大

		{"v1.2.0", "v1.2.1-of.1", true},        // 没有 of 后缀
		{"v1.2.1-of.2", "v1.2.1-of.1", false},  // of 编号更小
		{"v2.0.0-of.1", "v1.9.9-of.99", false}, // 主版本更小
		{"v1.2.1-of.1", "v1.2.1-of.1", false},  // 完全相同
		{"v1.2-of.2", "v1.2.0-of.1", false},    // 修订号相等，of 更大但整体更小
	}

	for _, tt := range tests {
		got := NeedUpgrade(tt.old, tt.new)
		if got != tt.want {
			t.Errorf("NeedUpgrade(%q, %q) = %v, want %v", tt.old, tt.new, got, tt.want)
		}
	}
}

func TestNodeInfo(t *testing.T) {
	tests := []struct {
		name     string
		node     confv1beta1.Node
		expected string
	}{
		{"both hostname and IP", confv1beta1.Node{Hostname: "node1", IP: "192.168.1.1"}, "node1/192.168.1.1"},
		{"only IP", confv1beta1.Node{IP: "192.168.1.1"}, "192.168.1.1"},
		{"only hostname", confv1beta1.Node{Hostname: "node1"}, "node1"},
		{"empty", confv1beta1.Node{}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NodeInfo(tt.node)
			if result != tt.expected {
				t.Errorf("NodeInfo() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestNodeRoleString(t *testing.T) {
	node := confv1beta1.Node{Role: []string{"master", "worker"}}
	result := NodeRoleString(node)
	if result != "master;worker" {
		t.Errorf("NodeRoleString() = %v, want master;worker", result)
	}
}

func TestIsMasterNode(t *testing.T) {
	tests := []struct {
		name     string
		node     *confv1beta1.Node
		expected bool
	}{
		{"is master", &confv1beta1.Node{Role: []string{"master"}}, true},
		{"not master", &confv1beta1.Node{Role: []string{"worker"}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsMasterNode(tt.node)
			if result != tt.expected {
				t.Errorf("IsMasterNode() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestIsWorkerNode(t *testing.T) {
	tests := []struct {
		name     string
		node     *confv1beta1.Node
		expected bool
	}{
		{"is worker", &confv1beta1.Node{Role: []string{"node"}}, true},
		{"not worker", &confv1beta1.Node{Role: []string{"master"}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsWorkerNode(tt.node)
			if result != tt.expected {
				t.Errorf("IsWorkerNode() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestIsEtcdNode(t *testing.T) {
	tests := []struct {
		name     string
		node     *confv1beta1.Node
		expected bool
	}{
		{"is etcd", &confv1beta1.Node{Role: []string{"etcd"}}, true},
		{"not etcd", &confv1beta1.Node{Role: []string{"master"}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsEtcdNode(tt.node)
			if result != tt.expected {
				t.Errorf("IsEtcdNode() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetBKEAllNodesFromNodesStatusWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Hostname: "node1"}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Hostname: "node2"}},
	}
	result := GetBKEAllNodesFromNodesStatusWithBKENodes(bkeNodes)
	require.Len(t, result, 2)
	assert.Equal(t, "192.168.1.1", result[0].IP)
	assert.Equal(t, "192.168.1.2", result[1].IP)
}

func TestGetBKENodesFromNodesStatus(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{NeedSkip: false}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.3"}, Status: confv1beta1.BKENodeStatus{NeedSkip: false}},
		}
	})
	defer patches.Reset()

	result := GetBKENodesFromNodesStatus(cluster)
	require.Len(t, result, 2)
	assert.Equal(t, "192.168.1.1", result[0].Spec.IP)
	assert.Equal(t, "192.168.1.3", result[1].Spec.IP)
}

func TestGetBKENodesFromNodesStatusWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{NeedSkip: false}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.3"}, Status: confv1beta1.BKENodeStatus{NeedSkip: false}},
	}
	result := GetBKENodesFromNodesStatusWithBKENodes(bkeNodes)
	require.Len(t, result, 2)
	assert.Equal(t, "192.168.1.1", result[0].Spec.IP)
	assert.Equal(t, "192.168.1.3", result[1].Spec.IP)
}

func TestConvertELBNodesToBKENodes(t *testing.T) {
	src := bkenode.Nodes{{IP: "192.168.1.1"}, {IP: "192.168.1.2"}, {IP: "192.168.1.3"}}
	elbNodes := []string{"192.168.1.1", "192.168.1.3"}
	result := ConvertELBNodesToBKENodes(elbNodes, src)
	assert.Len(t, result, 2)
	assert.Equal(t, "192.168.1.1", result[0].IP)
	assert.Equal(t, "192.168.1.3", result[1].IP)
}

// ---------- Node filter options ----------

func TestWithExcludeAppointmentNodes(t *testing.T) {
	cfg := &nodeFilterConfig{}
	opt := WithExcludeAppointmentNodes()
	opt(cfg)
	assert.True(t, cfg.excludeAppointment)
}

func TestWithBKENodes(t *testing.T) {
	nodes := bkev1beta1.BKENodes{}
	cfg := &nodeFilterConfig{}
	opt := WithBKENodes(nodes)
	opt(cfg)
	assert.NotNil(t, cfg.bkeNodes)
}

// ---------- filterNodes ----------

func TestFilterNodes(t *testing.T) {
	t.Run("with bkeNodes option", func(t *testing.T) {
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.3"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeDeletingFlag}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.4"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.5"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
		result := filterNodes(&bkev1beta1.BKECluster{},
			func(ip string, bn *confv1beta1.BKENode) bool { return true },
			WithBKENodes(bkeNodes),
		)
		require.Len(t, result, 2)
		assert.Equal(t, "192.168.1.1", result[0].IP)
		assert.Equal(t, "192.168.1.5", result[1].IP)
	})

	t.Run("without bkeNodes option", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
			}
		})
		defer patches.Reset()

		result := filterNodes(&bkev1beta1.BKECluster{},
			func(ip string, bn *confv1beta1.BKENode) bool { return true },
		)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})

	t.Run("with excludeAppointment", func(t *testing.T) {
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
		patches := gomonkey.ApplyFunc(GetAppointmentAddNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1"}}
		})
		defer patches.Reset()

		result := filterNodes(&bkev1beta1.BKECluster{},
			func(ip string, bn *confv1beta1.BKENode) bool { return true },
			WithBKENodes(bkeNodes),
			WithExcludeAppointmentNodes(),
		)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.2", result[0].IP)
	})

	t.Run("with excludeAppointment and empty appointment", func(t *testing.T) {
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
		patches := gomonkey.ApplyFunc(GetAppointmentAddNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{}
		})
		defer patches.Reset()

		result := filterNodes(&bkev1beta1.BKECluster{},
			func(ip string, bn *confv1beta1.BKENode) bool { return true },
			WithBKENodes(bkeNodes),
			WithExcludeAppointmentNodes(),
		)
		require.Len(t, result, 1)
	})
}

// ---------- Agent push / env / postprocess / join ----------

func TestGetNeedPushAgentNodes(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentPushedFlag}},
		}
	})
	defer patches.Reset()

	result := GetNeedPushAgentNodes(&bkev1beta1.BKECluster{})
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedPushAgentNodesWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentPushedFlag}},
	}
	result := GetNeedPushAgentNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkeNodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetAgentPushedNodes(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentPushedFlag}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
	})
	defer patches.Reset()

	result := GetAgentPushedNodes(&bkev1beta1.BKECluster{})
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetAgentPushedNodesWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentPushedFlag}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
	}
	result := GetAgentPushedNodesWithBKENodes(bkeNodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedInitEnvNodes(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeEnvFlag}},
		}
	})
	defer patches.Reset()

	result := GetNeedInitEnvNodes(&bkev1beta1.BKECluster{})
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedInitEnvNodesWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeEnvFlag}},
	}
	result := GetNeedInitEnvNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkeNodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedPostProcessNodes(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeBootFlag}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeBootFlag | bkev1beta1.NodePostProcessFlag}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.3"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
	})
	defer patches.Reset()

	result := GetNeedPostProcessNodes(&bkev1beta1.BKECluster{})
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedPostProcessNodesWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeBootFlag}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeBootFlag | bkev1beta1.NodePostProcessFlag}},
	}
	result := GetNeedPostProcessNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkeNodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedJoinNodes(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeBootFlag}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.3"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.MasterInitFlag}},
		}
	})
	defer patches.Reset()

	result := GetNeedJoinNodes(&bkev1beta1.BKECluster{})
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedJoinNodesWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeBootFlag}},
	}
	result := GetNeedJoinNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkeNodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedJoinMasterNodes(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{bkenode.WorkerNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
	})
	defer patches.Reset()

	patches.ApplyFunc(GetAppointmentAddNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	result := GetNeedJoinMasterNodes(&bkev1beta1.BKECluster{})
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedJoinMasterNodesWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{"worker"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
	}
	result := GetNeedJoinMasterNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkeNodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedJoinWorkerNodes(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{bkenode.WorkerNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
	})
	defer patches.Reset()

	patches.ApplyFunc(GetAppointmentAddNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	result := GetNeedJoinWorkerNodes(&bkev1beta1.BKECluster{})
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.2", result[0].IP)
}

func TestGetNeedJoinWorkerNodesWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{bkenode.WorkerNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
	}
	result := GetNeedJoinWorkerNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkeNodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.2", result[0].IP)
}

// ---------- Appointment nodes ----------

func TestGetAppointmentAddNodes(t *testing.T) {
	t.Run("no annotation", func(t *testing.T) {
		assert.Nil(t, GetAppointmentAddNodes(&bkev1beta1.BKECluster{}))
	})

	t.Run("with annotation", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					annotation.AppointmentAddNodesAnnotationKey: "192.168.1.1,192.168.1.2",
				},
			},
		}
		patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}},
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}},
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.3"}},
			}
		})
		defer patches.Reset()

		result := GetAppointmentAddNodes(cluster)
		require.Len(t, result, 2)
		assert.Equal(t, "192.168.1.1", result[0].IP)
		assert.Equal(t, "192.168.1.2", result[1].IP)
	})
}

func TestGetAppointmentAddNodesWithBKENodes(t *testing.T) {
	t.Run("no annotation", func(t *testing.T) {
		assert.Nil(t, GetAppointmentAddNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkev1beta1.BKENodes{}))
	})

	t.Run("with annotation", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					annotation.AppointmentAddNodesAnnotationKey: "192.168.1.1",
				},
			},
		}
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}},
		}
		result := GetAppointmentAddNodesWithBKENodes(cluster, bkeNodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

func TestComputeFinalAddNodes(t *testing.T) {
	needAdd := bkenode.Nodes{{IP: "192.168.1.1"}, {IP: "192.168.1.2"}}
	appointment := bkenode.Nodes{{IP: "192.168.1.1"}}
	result := ComputeFinalAddNodes(needAdd, appointment)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.2", result[0].IP)
}

// ---------- Delete nodes ----------

func TestGetNeedDeleteNodes(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}

	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}},
		}
	})
	defer patches.Reset()

	result := GetNeedDeleteNodes(context.Background(), c, cluster)
	assert.Equal(t, 0, len(result))
}

func TestGetNeedDeleteWorkerNodes(t *testing.T) {
	t.Run("appointment mode", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test",
				Namespace: "default",
				Annotations: map[string]string{
					annotation.AppointmentDeletedNodesAnnotationKey: "192.168.1.1",
				},
			},
		}

		patches := gomonkey.ApplyFunc(GetAppointmentDeletedNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.WorkerNodeRole}}}
		})
		defer patches.Reset()

		patches.ApplyFunc(GetNeedDeleteNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.WorkerNodeRole}}}
		})

		result := GetNeedDeleteWorkerNodes(context.Background(), c, cluster)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})

	t.Run("no appointment", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		}
		assert.Nil(t, GetNeedDeleteWorkerNodes(context.Background(), c, cluster))
	})
}

func TestGetNeedDeleteMasterNodes(t *testing.T) {
	t.Run("appointment mode", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test",
				Namespace: "default",
				Annotations: map[string]string{
					annotation.AppointmentDeletedNodesAnnotationKey: "192.168.1.1",
				},
			},
		}

		patches := gomonkey.ApplyFunc(GetAppointmentDeletedNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}}
		})
		defer patches.Reset()

		patches.ApplyFunc(GetNeedDeleteNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}}
		})

		result := GetNeedDeleteMasterNodes(context.Background(), c, cluster)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})

	t.Run("no appointment", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		}
		assert.Nil(t, GetNeedDeleteMasterNodes(context.Background(), c, cluster))
	})
}

func TestGetNeedDeleteWorkerNodesWithTargetNodes(t *testing.T) {
	t.Run("appointment mode", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test",
				Namespace: "default",
				Annotations: map[string]string{
					annotation.AppointmentDeletedNodesAnnotationKey: "192.168.1.1",
				},
			},
		}
		targetNodes := bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.WorkerNodeRole}}}

		patches := gomonkey.ApplyFunc(GetAppointmentDeletedNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.WorkerNodeRole}}}
		})
		defer patches.Reset()

		patches.ApplyFunc(GetNeedDeleteNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.WorkerNodeRole}}}
		})

		result := GetNeedDeleteWorkerNodesWithTargetNodes(context.Background(), c, cluster, targetNodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})

	t.Run("no appointment", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		}
		targetNodes := bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.WorkerNodeRole}}}
		patches := gomonkey.ApplyFunc(GetNeedDeleteNodesFromTargetNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster, bkenode.Nodes) bkenode.Nodes {
			return bkenode.Nodes{}
		})
		defer patches.Reset()

		result := GetNeedDeleteWorkerNodesWithTargetNodes(context.Background(), c, cluster, targetNodes)
		assert.Equal(t, 0, len(result))
	})
}

func TestGetNeedDeleteMasterNodesWithTargetNodes(t *testing.T) {
	t.Run("appointment mode", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test",
				Namespace: "default",
				Annotations: map[string]string{
					annotation.AppointmentDeletedNodesAnnotationKey: "192.168.1.1",
				},
			},
		}
		targetNodes := bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}}

		patches := gomonkey.ApplyFunc(GetAppointmentDeletedNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}}
		})
		defer patches.Reset()

		patches.ApplyFunc(GetNeedDeleteNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}}
		})

		result := GetNeedDeleteMasterNodesWithTargetNodes(context.Background(), c, cluster, targetNodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})

	t.Run("no appointment", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		}
		targetNodes := bkenode.Nodes{{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}}
		patches := gomonkey.ApplyFunc(GetNeedDeleteNodesFromTargetNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster, bkenode.Nodes) bkenode.Nodes {
			return bkenode.Nodes{}
		})
		defer patches.Reset()

		result := GetNeedDeleteMasterNodesWithTargetNodes(context.Background(), c, cluster, targetNodes)
		assert.Equal(t, 0, len(result))
	})
}

func TestGetNeedDeleteNodesFromTargetNodes(t *testing.T) {
	t.Run("nil target nodes", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		}
		assert.Nil(t, GetNeedDeleteNodesFromTargetNodes(context.Background(), c, cluster, nil))
	})

	t.Run("empty target nodes", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		}
		assert.Nil(t, GetNeedDeleteNodesFromTargetNodes(context.Background(), c, cluster, bkenode.Nodes{}))
	})

	t.Run("with target nodes", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		}
		targetNodes := bkenode.Nodes{
			{IP: "192.168.1.1"},
			{IP: "192.168.1.2"},
		}

		patches := gomonkey.ApplyFunc(GetNeedDeleteNodesFromTargetNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster, bkenode.Nodes) bkenode.Nodes {
			return bkenode.Nodes{{IP: "192.168.1.2"}}
		})
		defer patches.Reset()

		result := GetNeedDeleteNodesFromTargetNodes(context.Background(), c, cluster, targetNodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.2", result[0].IP)
	})
}

func TestGetNodeRolesFromK8sNode(t *testing.T) {
	t.Run("control-plane and master", func(t *testing.T) {
		node := &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"node-role.kubernetes.io/control-plane": "",
					"node-role.kubernetes.io/master":        "",
				},
			},
		}
		assert.Equal(t, []string{bkenode.MasterNodeRole, bkenode.WorkerNodeRole}, GetNodeRolesFromK8sNode(node))
	})

	t.Run("control-plane only", func(t *testing.T) {
		node := &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"node-role.kubernetes.io/control-plane": "",
				},
			},
		}
		assert.Equal(t, []string{bkenode.MasterNodeRole, bkenode.WorkerNodeRole}, GetNodeRolesFromK8sNode(node))
	})

	t.Run("no labels", func(t *testing.T) {
		assert.Equal(t, []string{bkenode.WorkerNodeRole}, GetNodeRolesFromK8sNode(&corev1.Node{}))
	})
}

func TestGetAppointmentDeletedNodes(t *testing.T) {
	t.Run("no annotation", func(t *testing.T) {
		assert.Nil(t, GetAppointmentDeletedNodes(&bkev1beta1.BKECluster{}))
	})

	t.Run("with annotation", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					annotation.AppointmentDeletedNodesAnnotationKey: "192.168.1.1,192.168.1.2",
				},
			},
		}
		patches := gomonkey.ApplyFunc(GetNodesFromCluster, func(*bkev1beta1.BKECluster) bkenode.Nodes {
			return bkenode.Nodes{
				{IP: "192.168.1.1"},
				{IP: "192.168.1.2"},
				{IP: "192.168.1.3"},
			}
		})
		defer patches.Reset()

		result := GetAppointmentDeletedNodes(cluster)
		require.Len(t, result, 2)
		assert.Equal(t, "192.168.1.1", result[0].IP)
		assert.Equal(t, "192.168.1.2", result[1].IP)
	})
}

func TestRemoveAppointmentDeletedNodes(t *testing.T) {
	t.Run("no annotation", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		RemoveAppointmentDeletedNodes(cluster, "192.168.1.1")
		assert.Nil(t, cluster.Annotations)
	})

	t.Run("remove existing node", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					annotation.AppointmentDeletedNodesAnnotationKey: "192.168.1.1,192.168.1.2",
				},
			},
		}
		RemoveAppointmentDeletedNodes(cluster, "192.168.1.1")
		assert.Equal(t, "192.168.1.2", cluster.Annotations[annotation.AppointmentDeletedNodesAnnotationKey])
	})

	t.Run("remove non-existing node", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					annotation.AppointmentDeletedNodesAnnotationKey: "192.168.1.1,192.168.1.2",
				},
			},
		}
		RemoveAppointmentDeletedNodes(cluster, "192.168.1.3")
		assert.Equal(t, "192.168.1.1,192.168.1.2", cluster.Annotations[annotation.AppointmentDeletedNodesAnnotationKey])
	})
}

func TestComputeFinalDeleteNodes(t *testing.T) {
	needDelete := bkenode.Nodes{{IP: "192.168.1.1"}, {IP: "192.168.1.2"}}
	appointment := bkenode.Nodes{{IP: "192.168.1.1"}}
	result := ComputeFinalDeleteNodes(needDelete, appointment)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

// ---------- Version ----------

func TestNormalizeVersion(t *testing.T) {
	tests := []struct {
		name     string
		version  string
		expected string
		wantErr  bool
	}{
		{"normal version", "v1.25.0", "v1.25.0", false},
		{"without v prefix", "1.25.0", "v1.25.0", false},
		{"with leading zeros", "v01.025.000", "v1.25.0", false},
		{"with rc", "v1.25.0-rc.1", "v1.25.0", false},
		{"with of suffix", "v1.25.0-of.1", "v1.25.0", false},
		{"invalid format", "invalid", "", true},
		{"empty string", "", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := NormalizeVersion(tt.version)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestNeedUpgrade(t *testing.T) {
	tests := []struct {
		name string
		oldV string
		newV string
		want bool
	}{
		{"same version", "v1.2.3", "v1.2.3", false},
		{"upgrade patch", "v1.2.3", "v1.2.4", true},
		{"upgrade minor", "v1.2.3", "v1.3.0", true},
		{"upgrade major", "v1.2.3", "v2.0.0", true},
		{"downgrade", "v1.2.3", "v1.1.0", false},
		{"rc to final", "v1.2.3-rc.1", "v1.2.3", true},
		{"final to rc", "v1.2.3", "v1.2.3-rc.1", false},
		{"rc to newer rc", "v1.2.3-rc.1", "v1.2.3-rc.2", true},
		{"invalid old", "invalid", "v1.2.3", false},
		{"invalid new", "v1.2.3", "invalid", false},
		{"both invalid", "invalid", "invalid", false},
		{"of suffix upgrade", "v1.2.3-of.1", "v1.2.3-of.2", true},
		{"of suffix downgrade", "v1.2.3-of.2", "v1.2.3-of.1", false},
		{"of suffix same base", "v1.2.3-of.1", "v1.2.3-of.1", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, NeedUpgrade(tt.oldV, tt.newV))
		})
	}
}

func TestSplitVersion(t *testing.T) {
	t.Run("with of suffix", func(t *testing.T) {
		base, of := splitVersion("v1.2.3-of.4")
		assert.Equal(t, "v1.2.3", base)
		assert.Equal(t, 4, of)
	})

	t.Run("without of suffix", func(t *testing.T) {
		base, of := splitVersion("v1.2.3")
		assert.Equal(t, "v1.2.3", base)
		assert.Equal(t, 0, of)
	})

	t.Run("without v prefix", func(t *testing.T) {
		base, of := splitVersion("1.2.3")
		assert.Equal(t, "v1.2.3", base)
		assert.Equal(t, 0, of)
	})

	t.Run("invalid of number", func(t *testing.T) {
		base, of := splitVersion("v1.2.3-of.abc")
		assert.Equal(t, "v1.2.3", base)
		assert.Equal(t, 0, of)
	})
}

// ---------- Upgrade nodes ----------

func TestGetNeedUpgradeComponentNodes(t *testing.T) {
	t.Run("empty version", func(t *testing.T) {
		assert.Nil(t, GetNeedUpgradeComponentNodes(&bkev1beta1.BKECluster{}))
	})

	t.Run("no upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: "v1.25.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v1.25.0"},
				},
			},
		}
		assert.Nil(t, GetNeedUpgradeComponentNodes(cluster))
	})

	t.Run("upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: "v1.24.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v1.25.0"},
				},
			},
		}
		patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
			}
		})
		defer patches.Reset()

		result := GetNeedUpgradeComponentNodes(cluster)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

func TestGetNeedUpgradeComponentNodesWithBKENodes(t *testing.T) {
	t.Run("empty version", func(t *testing.T) {
		assert.Nil(t, GetNeedUpgradeComponentNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkev1beta1.BKENodes{}))
	})

	t.Run("upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: "v1.24.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v1.25.0"},
				},
			},
		}
		nodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
		}
		result := GetNeedUpgradeComponentNodesWithBKENodes(cluster, nodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

func TestGetNeedUpgradeContainerdNodesWithBKENodes(t *testing.T) {
	t.Run("nil cluster", func(t *testing.T) {
		assert.Nil(t, GetNeedUpgradeContainerdNodesWithBKENodes(nil, bkev1beta1.BKENodes{}))
	})

	t.Run("nil cluster config", func(t *testing.T) {
		assert.Nil(t, GetNeedUpgradeContainerdNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkev1beta1.BKENodes{}))
	})

	t.Run("empty containerd version", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{},
			},
		}
		assert.Nil(t, GetNeedUpgradeContainerdNodesWithBKENodes(cluster, bkev1beta1.BKENodes{}))
	})

	t.Run("no upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "v1.2.3"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{ContainerdVersion: "v1.2.3"},
				},
			},
		}
		assert.Nil(t, GetNeedUpgradeContainerdNodesWithBKENodes(cluster, bkev1beta1.BKENodes{}))
	})

	t.Run("upgrade needed with mixed states", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "v1.2.3"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{ContainerdVersion: "v1.2.4"},
				},
			},
		}
		nodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
		}
		result := GetNeedUpgradeContainerdNodesWithBKENodes(cluster, nodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

func TestGetNeedUpgradeNodes(t *testing.T) {
	t.Run("empty version", func(t *testing.T) {
		assert.Nil(t, GetNeedUpgradeNodes(&bkev1beta1.BKECluster{}))
	})

	t.Run("upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: "v1.24.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v1.25.0"},
				},
			},
		}
		patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			}
		})
		defer patches.Reset()

		result := GetNeedUpgradeNodes(cluster)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

func TestGetNeedUpgradeNodesWithBKENodes(t *testing.T) {
	t.Run("empty version", func(t *testing.T) {
		assert.Nil(t, GetNeedUpgradeNodesWithBKENodes(&bkev1beta1.BKECluster{}, bkev1beta1.BKENodes{}))
	})

	t.Run("upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: "v1.24.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v1.25.0"},
				},
			},
		}
		nodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
		result := GetNeedUpgradeNodesWithBKENodes(cluster, nodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

func TestGetNeedUpgradeK8sNodes(t *testing.T) {
	t.Run("upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.24.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.25.0"},
				},
			},
		}
		patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			}
		})
		defer patches.Reset()

		result := GetNeedUpgradeK8sNodes(cluster)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})

	t.Run("no upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.25.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.25.0"},
				},
			},
		}
		assert.Nil(t, GetNeedUpgradeK8sNodes(cluster))
	})
}

func TestGetNeedUpgradeK8sNodesWithBKENodes(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.24.0"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.25.0"},
			},
		},
	}
	nodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
	}
	result := GetNeedUpgradeK8sNodesWithBKENodes(cluster, nodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedUpgradeMasterNodes(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.24.0"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.25.0"},
			},
		},
	}
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{"worker"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
	})
	defer patches.Reset()

	result := GetNeedUpgradeMasterNodes(cluster)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedUpgradeMasterNodesWithBKENodes(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.24.0"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.25.0"},
			},
		},
	}
	nodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{"worker"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
	}
	result := GetNeedUpgradeMasterNodesWithBKENodes(cluster, nodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedUpgradeWorkerNodes(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.24.0"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.25.0"},
			},
		},
	}
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{bkenode.WorkerNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
	})
	defer patches.Reset()

	result := GetNeedUpgradeWorkerNodes(cluster)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.2", result[0].IP)
}

func TestGetNeedUpgradeWorkerNodesWithBKENodes(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.24.0"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.25.0"},
			},
		},
	}
	nodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{bkenode.MasterNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{bkenode.WorkerNodeRole}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
	}
	result := GetNeedUpgradeWorkerNodesWithBKENodes(cluster, nodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.2", result[0].IP)
}

// ---------- Load balance ----------

func TestMergeNodesToMap(t *testing.T) {
	list1 := bkenode.Nodes{{IP: "192.168.1.1"}, {IP: "192.168.1.2"}}
	list2 := bkenode.Nodes{{IP: "192.168.1.2"}, {IP: "192.168.1.3"}}
	result := mergeNodesToMap(list1, list2)
	assert.Equal(t, 3, len(result))
}

func TestGetNeedLoadBalanceNodes(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	patches := gomonkey.ApplyFunc(GetNeedJoinMasterNodes, func(*bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{{IP: "192.168.1.1"}}
	})
	defer patches.Reset()

	patches.ApplyFunc(GetNeedDeleteMasterNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{{IP: "192.168.1.2"}}
	})

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
	result := GetNeedLoadBalanceNodes(context.Background(), c, cluster)
	assert.Equal(t, 2, len(result))
}

func TestGetNeedLoadBalanceNodesWithBKENodes(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{}
	nodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
	}
	patches := gomonkey.ApplyFunc(GetNeedDeleteMasterNodes, func(context.Context, client.Client, *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{{IP: "192.168.1.2"}}
	})
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
	result := GetNeedLoadBalanceNodesWithBKENodes(context.Background(), c, cluster, nodes)
	assert.Equal(t, 2, len(result))
}

// ---------- BKE Config ----------

func TestGetBKEConfigCMData(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      clusterutil.BKEConfigCmKey().Name,
				Namespace: clusterutil.BKEConfigCmKey().Namespace,
			},
			Data: map[string]string{"key": "value"},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
		result, err := GetBKEConfigCMData(context.Background(), c)
		assert.NoError(t, err)
		assert.Equal(t, "value", result["key"])
	})

	t.Run("not found", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		_, err := GetBKEConfigCMData(context.Background(), c)
		assert.Error(t, err)
	})
}

func TestGetRemoteBKEConfigCM(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(GetRemoteBKEConfigCM, func(_ context.Context, _ *kubernetes.Clientset) (*corev1.ConfigMap, error) {
			return &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      clusterutil.BKEConfigCmKey().Name,
					Namespace: clusterutil.BKEConfigCmKey().Namespace,
				},
				Data: map[string]string{"key": "value"},
			}, nil
		})
		defer patches.Reset()

		result, err := GetRemoteBKEConfigCM(context.Background(), &kubernetes.Clientset{})
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "value", result.Data["key"])
	})

	t.Run("not found", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(GetRemoteBKEConfigCM, func(_ context.Context, _ *kubernetes.Clientset) (*corev1.ConfigMap, error) {
			return nil, nil
		})
		defer patches.Reset()

		result, err := GetRemoteBKEConfigCM(context.Background(), &kubernetes.Clientset{})
		assert.NoError(t, err)
		assert.Nil(t, result)
	})
}

func TestMigrateBKEConfigCM(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      clusterutil.BKEConfigCmKey().Name,
				Namespace: clusterutil.BKEConfigCmKey().Namespace,
			},
			Data: map[string]string{"key": "value"},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
		patches := gomonkey.ApplyFunc(MigrateBKEConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return nil
		})
		defer patches.Reset()

		err := MigrateBKEConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.NoError(t, err)
	})

	t.Run("config not found", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		patches := gomonkey.ApplyFunc(MigrateBKEConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return assert.AnError
		})
		defer patches.Reset()

		err := MigrateBKEConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.Error(t, err)
	})

	t.Run("already exists", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      clusterutil.BKEConfigCmKey().Name,
				Namespace: clusterutil.BKEConfigCmKey().Namespace,
			},
			Data: map[string]string{"key": "value"},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
		patches := gomonkey.ApplyFunc(MigrateBKEConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return nil
		})
		defer patches.Reset()

		err := MigrateBKEConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.NoError(t, err)
	})
}

// ---------- Health check ----------

func TestEnsureLocalHealthCheckConfigCM(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		err := EnsureLocalHealthCheckConfigCM(context.Background(), c)
		assert.NoError(t, err)
	})

	t.Run("already exists", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "health-check",
				Namespace: "openfuyao-system",
			},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
		err := EnsureLocalHealthCheckConfigCM(context.Background(), c)
		assert.NoError(t, err)
	})
}

func TestEnsureHealthCheckConfigNamespace(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		err := ensureHealthCheckConfigNamespace(context.Background(), c)
		assert.NoError(t, err)
	})

	t.Run("already exists", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		ns := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: "openfuyao-system"},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(ns).Build()
		err := ensureHealthCheckConfigNamespace(context.Background(), c)
		assert.NoError(t, err)
	})
}

func TestGetRemoteHealthCheckConfigCM(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(GetRemoteHealthCheckConfigCM, func(_ context.Context, _ *kubernetes.Clientset) (*corev1.ConfigMap, error) {
			return &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "health-check",
					Namespace: "openfuyao-system",
				},
				Data: map[string]string{"key": "value"},
			}, nil
		})
		defer patches.Reset()

		result, err := GetRemoteHealthCheckConfigCM(context.Background(), &kubernetes.Clientset{})
		assert.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("not found", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(GetRemoteHealthCheckConfigCM, func(_ context.Context, _ *kubernetes.Clientset) (*corev1.ConfigMap, error) {
			return nil, nil
		})
		defer patches.Reset()

		result, err := GetRemoteHealthCheckConfigCM(context.Background(), &kubernetes.Clientset{})
		assert.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("nil clientset", func(t *testing.T) {
		result, err := GetRemoteHealthCheckConfigCM(context.Background(), nil)
		assert.Error(t, err)
		assert.Nil(t, result)
	})
}

func TestMigrateHealthCheckConfigCM(t *testing.T) {
	t.Run("nil clientset", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		err := MigrateHealthCheckConfigCM(context.Background(), c, nil)
		assert.Error(t, err)
	})

	t.Run("success", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "health-check",
				Namespace: "openfuyao-system",
			},
			Data: map[string]string{"key": "value"},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
		patches := gomonkey.ApplyFunc(MigrateHealthCheckConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return nil
		})
		defer patches.Reset()

		err := MigrateHealthCheckConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.NoError(t, err)
	})

	t.Run("config not found", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		patches := gomonkey.ApplyFunc(MigrateHealthCheckConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return assert.AnError
		})
		defer patches.Reset()

		err := MigrateHealthCheckConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.Error(t, err)
	})

	t.Run("already exists", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "health-check",
				Namespace: "openfuyao-system",
			},
			Data: map[string]string{"key": "value"},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
		patches := gomonkey.ApplyFunc(MigrateHealthCheckConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return nil
		})
		defer patches.Reset()

		err := MigrateHealthCheckConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.NoError(t, err)
	})
}

func TestMigratePatchConfigCM(t *testing.T) {
	t.Run("success with no configmaps", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		c := fakeclient.NewClientBuilder().WithScheme(scheme).Build()
		patches := gomonkey.ApplyFunc(MigratePatchConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return nil
		})
		defer patches.Reset()

		err := MigratePatchConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.NoError(t, err)
	})

	t.Run("success with configmaps", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "cm.test",
				Namespace: "openfuyao-patch",
			},
			Data: map[string]string{"key": "value"},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
		patches := gomonkey.ApplyFunc(MigratePatchConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return nil
		})
		defer patches.Reset()

		err := MigratePatchConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.NoError(t, err)
	})

	t.Run("already exists", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = bkev1beta1.AddToScheme(scheme)
		_ = corev1.AddToScheme(scheme)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "cm.test",
				Namespace: "openfuyao-patch",
			},
			Data: map[string]string{"key": "value"},
		}
		c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
		patches := gomonkey.ApplyFunc(MigratePatchConfigCM, func(_ context.Context, _ client.Client, _ *kubernetes.Clientset) error {
			return nil
		})
		defer patches.Reset()

		err := MigratePatchConfigCM(context.Background(), c, &kubernetes.Clientset{})
		assert.NoError(t, err)
	})
}

// ---------- Bootstrap ----------

func TestGetReadyBootstrapNodes(t *testing.T) {
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentReadyFlag | bkev1beta1.NodeEnvFlag}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentReadyFlag | bkev1beta1.NodeEnvFlag | bkev1beta1.NodeBootFlag}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.3"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
	})
	defer patches.Reset()

	result := GetReadyBootstrapNodes(&bkev1beta1.BKECluster{})
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetReadyBootstrapNodesWithBKENodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentReadyFlag | bkev1beta1.NodeEnvFlag}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentReadyFlag | bkev1beta1.NodeEnvFlag | bkev1beta1.NodeBootFlag}},
	}
	result := GetReadyBootstrapNodesWithBKENodes(bkeNodes)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

// ---------- Remote node ----------

func TestGetRemoteNodeByBKENode(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(GetRemoteNodeByBKENode, func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
			return &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
			}, nil
		})
		defer patches.Reset()

		result, err := GetRemoteNodeByBKENode(context.Background(), &kubernetes.Clientset{}, confv1beta1.Node{Hostname: "node1"})
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "node1", result.Name)
	})

	t.Run("not found", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(GetRemoteNodeByBKENode, func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
			return nil, assert.AnError
		})
		defer patches.Reset()

		result, err := GetRemoteNodeByBKENode(context.Background(), &kubernetes.Clientset{}, confv1beta1.Node{Hostname: "node1"})
		assert.Error(t, err)
		assert.Nil(t, result)
	})
}

func TestGetClientURLByIP(t *testing.T) {
	assert.Equal(t, "https://192.168.1.1:2379", GetClientURLByIP("192.168.1.1"))
}

// ---------- Etcd upgrade ----------

func TestGetNeedUpgradeEtcds(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{EtcdVersion: "v3.5.0"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{EtcdVersion: "v3.5.1"},
			},
		},
	}
	patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
		return bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"etcd"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{"worker"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
	})
	defer patches.Reset()

	result := GetNeedUpgradeEtcds(cluster)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].IP)
}

func TestGetNeedUpgradeEtcdsWithBKENodes(t *testing.T) {
	t.Run("nil cluster config", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		assert.Nil(t, GetNeedUpgradeEtcdsWithBKENodes(cluster, bkev1beta1.BKENodes{}))
	})

	t.Run("no upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{EtcdVersion: "v3.5.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{EtcdVersion: "v3.5.0"},
				},
			},
		}
		assert.Nil(t, GetNeedUpgradeEtcdsWithBKENodes(cluster, bkev1beta1.BKENodes{}))
	})

	t.Run("upgrade needed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Status: confv1beta1.BKEClusterStatus{EtcdVersion: "v3.5.0"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{EtcdVersion: "v3.5.1"},
				},
			},
		}
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"etcd"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2", Role: []string{"worker"}}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
		result := GetNeedUpgradeEtcdsWithBKENodes(cluster, bkeNodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

// ---------- Helpers ----------

func TestFilterNonFailedNodes(t *testing.T) {
	statusNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
		{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.3"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
	}
	result := filterNonFailedNodes(statusNodes)
	require.Len(t, result, 2)
	assert.Equal(t, "192.168.1.1", result[0].IP)
	assert.Equal(t, "192.168.1.3", result[1].IP)
}

func TestCompareVersionAndGetNodes(t *testing.T) {
	t.Run("no upgrade needed", func(t *testing.T) {
		assert.Nil(t, compareVersionAndGetNodes(&bkev1beta1.BKECluster{}, "v1.2.3", "v1.2.3"))
	})

	t.Run("upgrade needed", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(GetBKENodesFromCluster, func(*bkev1beta1.BKECluster) bkev1beta1.BKENodes {
			return bkev1beta1.BKENodes{
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
			}
		})
		defer patches.Reset()

		result := compareVersionAndGetNodes(&bkev1beta1.BKECluster{}, "v1.2.3", "v1.2.4")
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

func TestCompareVersionAndGetNodesWithBKENodes(t *testing.T) {
	t.Run("no upgrade needed", func(t *testing.T) {
		assert.Nil(t, compareVersionAndGetNodesWithBKENodes(&bkev1beta1.BKECluster{}, "v1.2.3", "v1.2.3", bkev1beta1.BKENodes{}))
	})

	t.Run("upgrade needed", func(t *testing.T) {
		nodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
		}
		result := compareVersionAndGetNodesWithBKENodes(&bkev1beta1.BKECluster{}, "v1.2.3", "v1.2.4", nodes)
		require.Len(t, result, 1)
		assert.Equal(t, "192.168.1.1", result[0].IP)
	})
}

func TestGetNodesFromCluster(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	// GetNodesData is small enough to be inlined, so patch the callee it always calls.
	patches := gomonkey.ApplyFunc(clusterpkg.GetClusterData, func(string, string) (*clusterpkg.ClusterData, error) {
		return &clusterpkg.ClusterData{
			Nodes: bkenode.Nodes{
				{IP: "192.168.1.1"},
			},
		}, nil
	})
	defer patches.Reset()

	result := GetNodesFromCluster(cluster)
	assert.Equal(t, 1, len(result))
}

func TestGetBKENodesFromCluster(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	patches := gomonkey.ApplyFunc(GetNodesFromCluster, func(*bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{
			{IP: "192.168.1.1", Hostname: "node1"},
		}
	})
	defer patches.Reset()

	result := GetBKENodesFromCluster(cluster)
	require.Len(t, result, 1)
	assert.Equal(t, "192.168.1.1", result[0].Spec.IP)
}

func TestGetNodeStateFlag(t *testing.T) {
	node := &confv1beta1.BKENode{
		Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
		Status: confv1beta1.BKENodeStatus{StateCode: 4},
	}
	assert.True(t, GetNodeStateFlag(node, "192.168.1.1", 4))
	assert.False(t, GetNodeStateFlag(node, "192.168.1.2", 4))
	assert.False(t, GetNodeStateFlag(node, "192.168.1.1", 8))
}

func TestHasNodesNeedingPhase(t *testing.T) {
	t.Run("has nodes needing phase", func(t *testing.T) {
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 0}},
		}
		assert.True(t, HasNodesNeedingPhase(bkeNodes, 4))
	})

	t.Run("all have flag", func(t *testing.T) {
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: 4}},
		}
		assert.False(t, HasNodesNeedingPhase(bkeNodes, 4))
	})

	t.Run("failed", func(t *testing.T) {
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeFailedFlag}},
		}
		assert.False(t, HasNodesNeedingPhase(bkeNodes, 4))
	})

	t.Run("deleting", func(t *testing.T) {
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeDeletingFlag}},
		}
		assert.False(t, HasNodesNeedingPhase(bkeNodes, 4))
	})

	t.Run("need skip", func(t *testing.T) {
		bkeNodes := bkev1beta1.BKENodes{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}, Status: confv1beta1.BKENodeStatus{NeedSkip: true}},
		}
		assert.False(t, HasNodesNeedingPhase(bkeNodes, 4))
	})

	t.Run("empty nodes", func(t *testing.T) {
		assert.False(t, HasNodesNeedingPhase(bkev1beta1.BKENodes{}, 4))
	})
}
