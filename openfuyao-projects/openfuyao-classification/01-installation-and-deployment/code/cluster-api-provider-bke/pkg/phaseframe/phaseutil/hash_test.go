/******************************************************************
 * Copyright (c) 2026 ICBC Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

func TestStringSlicesEqual(t *testing.T) {
	if !utils.StringSlicesEqual([]string{"a", "b"}, []string{"a", "b"}) {
		t.Errorf("expected equal slices to return true")
	}
	if utils.StringSlicesEqual([]string{"a", "b"}, []string{"a", "c"}) {
		t.Errorf("expected unequal slices to return false")
	}
	if utils.StringSlicesEqual([]string{"a"}, []string{"a", "b"}) {
		t.Errorf("expected slices of different length to return false")
	}
	if !utils.StringSlicesEqual(nil, nil) {
		t.Errorf("expected nil slices to return true")
	}
}

// TestCalculateComponentConfigHash verifies the basic behavior of the hash
// calculation: nil input, empty cluster, and clusters with various configurations.
func TestCalculateComponentConfigHash(t *testing.T) {
	tests := []struct {
		name    string
		cluster *confv1beta1.Cluster
		wantErr bool
	}{
		{
			name:    "nil cluster",
			cluster: nil,
			wantErr: false,
		},
		{
			name:    "empty cluster",
			cluster: &confv1beta1.Cluster{},
			wantErr: false,
		},
		{
			name: "cluster with apiServer extraArgs",
			cluster: &confv1beta1.Cluster{
				ControlPlane: confv1beta1.ControlPlane{
					APIServer: &confv1beta1.APIServer{
						ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
							ExtraArgs: map[string]string{
								"--audit-log-path": "/var/log/audit.log",
								"--feature-gates":  "FeatureGateName=true",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "full control plane config",
			cluster: &confv1beta1.Cluster{
				ControlPlane: confv1beta1.ControlPlane{
					APIServer: &confv1beta1.APIServer{
						ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
							ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"},
						},
						CertSANs: []string{"example.com"},
					},
					ControllerManager: &confv1beta1.ControlPlaneComponent{
						ExtraArgs: map[string]string{"--bind-address": "0.0.0.0"},
					},
					Scheduler: &confv1beta1.ControlPlaneComponent{
						ExtraArgs: map[string]string{"--profiles": "/etc/kubernetes/scheduler-config.yaml"},
					},
					Etcd: &confv1beta1.Etcd{ // Etcd 包含在这里但不应该影响 ComponentConfigHash
						DataDir: "/var/lib/etcd",
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := CalculateComponentConfigHash(tt.cluster)
			fmt.Println(got)
			if (err != nil) != tt.wantErr {
				t.Errorf("CalculateComponentConfigHash() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			emptyHash := confv1beta1.ComponentConfigHash{}
			if !tt.wantErr && reflect.DeepEqual(got, emptyHash) && tt.cluster != nil && tt.cluster.APIServer != nil {
				t.Error("CalculateComponentConfigHash() returned empty hash for non-nil populated cluster")
			}
		})
	}
}

// TestCalculateComponentConfigHash_Consistency verifies that the same config produces the same hash.
func TestCalculateComponentConfigHash_Consistency(t *testing.T) {
	cluster1 := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			APIServer: &confv1beta1.APIServer{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
					ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"},
				},
			},
		},
		KubernetesVersion: "v1.25.6",
	}

	cluster2 := &confv1beta1.Cluster{
		KubernetesVersion: "v1.25.6",
		ControlPlane: confv1beta1.ControlPlane{
			APIServer: &confv1beta1.APIServer{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
					ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"},
				},
			},
		},
	}

	hash1, _ := CalculateComponentConfigHash(cluster1)
	hash2, _ := CalculateComponentConfigHash(cluster2)

	if !reflect.DeepEqual(hash1, hash2) {
		t.Errorf("Same ControlPlane config should produce same hash: %+v != %+v", hash1, hash2)
	}
}

// TestCalculateComponentConfigHash_Difference verifies that different configs produce different hashes.
func TestCalculateComponentConfigHash_Difference(t *testing.T) {
	cluster1 := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			APIServer: &confv1beta1.APIServer{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
					ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"},
				},
			},
		},
	}

	cluster2 := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			APIServer: &confv1beta1.APIServer{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
					ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit2.log"}, // Different path
				},
			},
		},
	}

	hash1, _ := CalculateComponentConfigHash(cluster1)
	hash2, _ := CalculateComponentConfigHash(cluster2)

	if reflect.DeepEqual(hash1, hash2) {
		t.Error("Different ControlPlane configs should produce different hashes")
	}
}

// TestCalculateComponentConfigHash_EachControlPlaneComponent verifies that
// changing each individual ControlPlane component produces a different component hash.
// Note: Etcd is excluded from CalculateComponentConfigHash.
func TestCalculateComponentConfigHash_EachControlPlaneComponent(t *testing.T) {
	base := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			APIServer: &confv1beta1.APIServer{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"}},
			},
			ControllerManager: &confv1beta1.ControlPlaneComponent{
				ExtraArgs: map[string]string{"--bind-address": "0.0.0.0"},
			},
			Scheduler: &confv1beta1.ControlPlaneComponent{
				ExtraArgs: map[string]string{"--profiles": "v1"},
			},
			Etcd: &confv1beta1.Etcd{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--data-dir": "/var/lib/etcd"}},
			},
		},
	}

	baseHash, _ := CalculateComponentConfigHash(base)

	// Change APIServer -> APIServer hash should change, others stay same
	changed := base.DeepCopy()
	changed.ControlPlane.APIServer.ExtraArgs["--audit-log-path"] = "/var/log/audit2.log"
	h, _ := CalculateComponentConfigHash(changed)
	if h.APIServer == baseHash.APIServer {
		t.Error("Changing APIServer should produce a different APIServer hash")
	}
	if h.ControllerManager != baseHash.ControllerManager || h.Scheduler != baseHash.Scheduler {
		t.Error("Changing APIServer should NOT affect CM or Scheduler hashes")
	}

	// Change ControllerManager
	changed = base.DeepCopy()
	changed.ControlPlane.ControllerManager.ExtraArgs["--bind-address"] = "127.0.0.1"
	h, _ = CalculateComponentConfigHash(changed)
	if h.ControllerManager == baseHash.ControllerManager {
		t.Error("Changing ControllerManager should produce a different CM hash")
	}

	// Change Scheduler
	changed = base.DeepCopy()
	changed.ControlPlane.Scheduler.ExtraArgs["--profiles"] = "v2"
	h, _ = CalculateComponentConfigHash(changed)
	if h.Scheduler == baseHash.Scheduler {
		t.Error("Changing Scheduler should produce a different Scheduler hash")
	}

	// Change Etcd -> should NOT change ComponentConfigHash
	changed = base.DeepCopy()
	changed.ControlPlane.Etcd.ExtraArgs["--data-dir"] = "/var/lib/etcd2"
	h, _ = CalculateComponentConfigHash(changed)
	if !reflect.DeepEqual(h, baseHash) {
		t.Error("Changing Etcd should NOT change the ComponentConfigHash")
	}
}

// TestCalculateComponentConfigHash_ExcludesNonControlPlaneFields verifies that
// fields outside of ControlPlane (e.g. KubernetesVersion, Networking) do NOT affect the hash.
func TestCalculateComponentConfigHash_ExcludesNonControlPlaneFields(t *testing.T) {
	baseControlPlane := confv1beta1.ControlPlane{
		APIServer: &confv1beta1.APIServer{
			ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"}},
		},
	}

	cluster1 := &confv1beta1.Cluster{
		ControlPlane:      baseControlPlane,
		KubernetesVersion: "v1.25.6",
		Networking:        confv1beta1.Networking{ServiceSubnet: "10.96.0.0/12"},
	}

	cluster2 := &confv1beta1.Cluster{
		ControlPlane:      baseControlPlane,
		KubernetesVersion: "v1.26.0",                                               // Different
		Networking:        confv1beta1.Networking{ServiceSubnet: "192.168.0.0/16"}, // Different
	}

	hash1, _ := CalculateComponentConfigHash(cluster1)
	hash2, _ := CalculateComponentConfigHash(cluster2)

	if !reflect.DeepEqual(hash1, hash2) {
		t.Errorf("Clusters differing only in non-ControlPlane fields should produce same hash: %+v != %+v", hash1, hash2)
	}
}

// TestCalculateComponentConfigHash_KubeletExcluded verifies Kubelet config changes do NOT affect the hash.
func TestCalculateComponentConfigHash_KubeletExcluded(t *testing.T) {
	controlPlane := confv1beta1.ControlPlane{
		APIServer: &confv1beta1.APIServer{
			ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"}},
		},
	}

	cluster1 := &confv1beta1.Cluster{
		ControlPlane: controlPlane,
		Kubelet: &confv1beta1.Kubelet{
			ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--max-pods": "110"}},
		},
	}

	cluster2 := &confv1beta1.Cluster{
		ControlPlane: controlPlane,
		Kubelet: &confv1beta1.Kubelet{
			ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--max-pods": "220"}}, // Different
		},
	}

	hash1, _ := CalculateComponentConfigHash(cluster1)
	hash2, _ := CalculateComponentConfigHash(cluster2)

	if !reflect.DeepEqual(hash1, hash2) {
		t.Error("Clusters differing only in Kubelet config should produce same hash")
	}
}

// ======================= 新增函数单测 =======================

// TestComponentConfigHashChanged 验证 ComponentConfigHashChanged 函数
func TestComponentConfigHashChanged(t *testing.T) {
	hash1 := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-1",
		ControllerManager: "hash-cm-1",
		Scheduler:         "hash-sched-1",
	}
	hash2 := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-1",
		ControllerManager: "hash-cm-1",
		Scheduler:         "hash-sched-1",
	}
	hash3 := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-1",
		ControllerManager: "hash-cm-2",
		Scheduler:         "hash-sched-1",
	}

	if ComponentConfigHashChanged(hash1, hash2) {
		t.Error("Same hashes should NOT be considered changed")
	}
	if !ComponentConfigHashChanged(hash1, hash3) {
		t.Error("Different hashes should be considered changed")
	}
}

// TestComponentConfigHashToJSON 验证 ComponentConfigHashToJSON 函数
func TestComponentConfigHashToJSON(t *testing.T) {
	hash := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-1",
		ControllerManager: "hash-cm-1",
		Scheduler:         "hash-sched-1",
	}

	jsonStr, err := ComponentConfigHashToJSON(hash)
	if err != nil {
		t.Fatalf("ComponentConfigHashToJSON() error = %v", err)
	}
	if jsonStr == "" {
		t.Error("ComponentConfigHashToJSON() returned empty string")
	}
	// 验证 JSON 包含所有字段
	if !strings.Contains(jsonStr, "hash-api-1") ||
		!strings.Contains(jsonStr, "hash-cm-1") ||
		!strings.Contains(jsonStr, "hash-sched-1") {
		t.Errorf("ComponentConfigHashToJSON() JSON missing fields: %s", jsonStr)
	}
}

// TestComponentConfigHashFromJSON 验证 ComponentConfigHashFromJSON 函数
func TestComponentConfigHashFromJSON(t *testing.T) {
	// 空字符串返回空结构体
	empty, err := ComponentConfigHashFromJSON("")
	if err != nil {
		t.Fatalf("ComponentConfigHashFromJSON(\"\") error = %v", err)
	}
	if empty != (confv1beta1.ComponentConfigHash{}) {
		t.Errorf("ComponentConfigHashFromJSON(\"\") should return empty hash, got %+v", empty)
	}

	// 正常 JSON 反序列化
	jsonStr := `{"apiServer":"hash-api-1","controllerManager":"hash-cm-1","scheduler":"hash-sched-1"}`
	hash, err := ComponentConfigHashFromJSON(jsonStr)
	if err != nil {
		t.Fatalf("ComponentConfigHashFromJSON() error = %v", err)
	}
	if hash.APIServer != "hash-api-1" ||
		hash.ControllerManager != "hash-cm-1" ||
		hash.Scheduler != "hash-sched-1" {
		t.Errorf("ComponentConfigHashFromJSON() got %+v", hash)
	}

	// 非法 JSON 返回错误
	_, err = ComponentConfigHashFromJSON("invalid-json")
	if err == nil {
		t.Error("ComponentConfigHashFromJSON() should return error for invalid JSON")
	}
}

// TestComponentConfigHashChangedComponents 验证 ComponentConfigHashChangedComponents 函数
func TestComponentConfigHashChangedComponents(t *testing.T) {
	oldHash := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-1",
		ControllerManager: "hash-cm-1",
		Scheduler:         "hash-sched-1",
	}
	newHash := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-2",   // 变化
		ControllerManager: "hash-cm-1",    // 未变化
		Scheduler:         "hash-sched-2", // 变化
	}

	changed := ComponentConfigHashChangedComponents(oldHash, newHash)

	if !changed["kube-apiserver"] {
		t.Error("APIServer change should mark kube-apiserver as changed")
	}
	if changed["kube-controller-manager"] {
		t.Error("ControllerManager unchanged should NOT mark kube-controller-manager as changed")
	}
	if !changed["kube-scheduler"] {
		t.Error("Scheduler change should mark kube-scheduler as changed")
	}

	// 完全相同的 hash 不应标记任何组件
	changed = ComponentConfigHashChangedComponents(oldHash, oldHash)
	if len(changed) != 0 {
		t.Errorf("Same hashes should not mark any component as changed, got %v", changed)
	}
}

// ======================= Etcd 专属的独立校验测试 =======================

func TestCalculateEtcdConfigHash_Difference(t *testing.T) {
	cluster1 := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			Etcd: &confv1beta1.Etcd{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--heartbeat-interval": "100"}},
			},
		},
	}

	cluster2 := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			Etcd: &confv1beta1.Etcd{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--heartbeat-interval": "200"}},
			},
		},
	}

	hash1, _ := CalculateEtcdConfigHash(cluster1)
	hash2, _ := CalculateEtcdConfigHash(cluster2)

	if hash1 == hash2 {
		t.Error("Different etcd ExtraArgs should produce different hashes")
	}
}

func TestCalculateEtcdConfigHash_IgnoresImmutableFields(t *testing.T) {
	base := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			Etcd: &confv1beta1.Etcd{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--heartbeat-interval": "100"}},
				DataDir:               "/var/lib/etcd",
				ServerCertSANs:        []string{"etcd.example.com"},
			},
		},
	}

	baseHash, _ := CalculateEtcdConfigHash(base)

	// Change DataDir → should NOT change the hash
	changed := base.DeepCopy()
	changed.ControlPlane.Etcd.DataDir = "/var/lib/etcd2"
	h, _ := CalculateEtcdConfigHash(changed)
	if h != baseHash {
		t.Error("Changing DataDir should NOT change the etcd hash (immutable field)")
	}

	// Change ServerCertSANs → should NOT change the hash
	changed = base.DeepCopy()
	changed.ControlPlane.Etcd.ServerCertSANs = []string{"etcd2.example.com"}
	h, _ = CalculateEtcdConfigHash(changed)
	if h != baseHash {
		t.Error("Changing ServerCertSANs should NOT change the etcd hash (immutable field)")
	}
}

func TestEtcdImmutableFieldsChanged(t *testing.T) {
	base := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			Etcd: &confv1beta1.Etcd{
				DataDir:        "/var/lib/etcd",
				ServerCertSANs: []string{"etcd.example.com"},
			},
		},
	}

	changed, field := EtcdImmutableFieldsChanged(base, base.DeepCopy())
	if changed {
		t.Errorf("Same config should not report immutable field change, got field=%q", field)
	}

	modified := base.DeepCopy()
	modified.ControlPlane.Etcd.DataDir = "/var/lib/etcd2"
	changed, field = EtcdImmutableFieldsChanged(base, modified)
	if !changed || field != "dataDir" {
		t.Errorf("DataDir change should report field=dataDir, got changed=%v field=%q", changed, field)
	}

	modified = base.DeepCopy()
	modified.ControlPlane.Etcd.ExtraArgs = map[string]string{"--heartbeat-interval": "200"}
	changed, field = EtcdImmutableFieldsChanged(base, modified)
	if changed {
		t.Errorf("ExtraArgs change should NOT be reported as immutable field change, got field=%q", field)
	}
}

func TestCalculateEtcdConfigHash_IgnoresNonEtcdFields(t *testing.T) {
	base := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			APIServer: &confv1beta1.APIServer{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"}},
			},
			Etcd: &confv1beta1.Etcd{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{ExtraArgs: map[string]string{"--data-dir": "/var/lib/etcd"}},
			},
		},
	}

	baseHash, _ := CalculateEtcdConfigHash(base)

	changed := base.DeepCopy()
	changed.ControlPlane.APIServer.ExtraArgs["--audit-log-path"] = "/var/log/audit2.log"
	h, _ := CalculateEtcdConfigHash(changed)
	if h != baseHash {
		t.Error("Changing APIServer should NOT change the etcd hash")
	}
}

// ======================= Normalize*ForHash 包装函数 =======================

// TestNormalizeAPIServerForHash 验证 APIServer 归一化结构只包含参与 hash 的可变字段
func TestNormalizeAPIServerForHash(t *testing.T) {
	apiServer := &confv1beta1.APIServer{
		APIEndpoint: confv1beta1.APIEndpoint{Host: "10.0.0.1", Port: 6443},
		ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
			ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"},
			ExtraVolumes: []confv1beta1.HostPathMount{
				{Name: "certs", HostPath: "/etc/certs", MountPath: "/certs", ReadOnly: true, PathType: "Directory"},
			},
		},
		CertSANs: []string{"example.com"},
	}

	data, err := json.Marshal(NormalizeAPIServerForHash(apiServer))
	require.NoError(t, err)

	want := `{
		"extraArgs": {"--audit-log-path": "/var/log/audit.log"},
		"extraVolumes": [{"name":"certs","hostPath":"/etc/certs","mountPath":"/certs","readOnly":true,"pathType":"Directory"}],
		"certSANs": ["example.com"],
		"apiEndpoint": {"host":"10.0.0.1","port":6443}
	}`
	assert.JSONEq(t, want, string(data))
}

// TestNormalizeControlPlaneComponentForHash 验证控制面组件归一化结构只含 ExtraArgs 与 ExtraVolumes
func TestNormalizeControlPlaneComponentForHash(t *testing.T) {
	comp := &confv1beta1.ControlPlaneComponent{
		ExtraArgs: map[string]string{"--bind-address": "0.0.0.0"},
		ExtraVolumes: []confv1beta1.HostPathMount{
			{Name: "vol", HostPath: "/host", MountPath: "/mnt", PathType: "Directory"},
		},
	}

	data, err := json.Marshal(NormalizeControlPlaneComponentForHash(comp))
	require.NoError(t, err)

	want := `{
		"extraArgs": {"--bind-address": "0.0.0.0"},
		"extraVolumes": [{"name":"vol","hostPath":"/host","mountPath":"/mnt","pathType":"Directory"}]
	}`
	assert.JSONEq(t, want, string(data))
}

// TestNormalizeEtcdForHash 验证 etcd 归一化结构只含可变字段，DataDir / CertSANs 被排除
func TestNormalizeEtcdForHash(t *testing.T) {
	etcd := &confv1beta1.Etcd{
		ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
			ExtraArgs: map[string]string{"--heartbeat-interval": "100"},
			ExtraVolumes: []confv1beta1.HostPathMount{
				{Name: "data", HostPath: "/data", MountPath: "/data", ReadOnly: false, PathType: "Directory"},
			},
		},
		DataDir:        "/var/lib/etcd",
		ServerCertSANs: []string{"etcd.example.com"},
		PeerCertSANs:   []string{"etcd-peer.example.com"},
	}

	data, err := json.Marshal(NormalizeEtcdForHash(etcd))
	require.NoError(t, err)

	// DataDir / ServerCertSANs / PeerCertSANs 属于启动期不可变字段，不应出现在归一化结果中
	want := `{
		"extraArgs": {"--heartbeat-interval": "100"},
		"extraVolumes": [{"name":"data","hostPath":"/data","mountPath":"/data","pathType":"Directory"}]
	}`
	assert.JSONEq(t, want, string(data))
	assert.NotContains(t, string(data), "dataDir")
	assert.NotContains(t, string(data), "etcd.example.com")
}

// ======================= 单组件 Hash 计算 =======================

func newTestCluster() *confv1beta1.Cluster {
	return &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			APIServer: &confv1beta1.APIServer{
				APIEndpoint: confv1beta1.APIEndpoint{Host: "10.0.0.1", Port: 6443},
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
					ExtraArgs: map[string]string{"--audit-log-path": "/var/log/audit.log"},
				},
				CertSANs: []string{"example.com"},
			},
			ControllerManager: &confv1beta1.ControlPlaneComponent{
				ExtraArgs: map[string]string{"--bind-address": "0.0.0.0"},
			},
			Scheduler: &confv1beta1.ControlPlaneComponent{
				ExtraArgs: map[string]string{"--profiles": "v1"},
			},
			Etcd: &confv1beta1.Etcd{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
					ExtraArgs: map[string]string{"--heartbeat-interval": "100"},
				},
				DataDir: "/var/lib/etcd",
			},
		},
	}
}

// TestCalculateAPIServerConfigHash 覆盖 nil 入参、正常计算以及各可变字段对 hash 的影响
func TestCalculateAPIServerConfigHash(t *testing.T) {
	t.Run("nil cluster", func(t *testing.T) {
		got, err := CalculateAPIServerConfigHash(nil)
		require.NoError(t, err)
		assert.Empty(t, got)
	})

	t.Run("nil apiServer", func(t *testing.T) {
		got, err := CalculateAPIServerConfigHash(&confv1beta1.Cluster{})
		require.NoError(t, err)
		assert.Empty(t, got)
	})

	t.Run("populated", func(t *testing.T) {
		got, err := CalculateAPIServerConfigHash(newTestCluster())
		require.NoError(t, err)
		assert.Len(t, got, 64) // sha256 的十六进制长度
	})

	t.Run("extraArgs change affects hash", func(t *testing.T) {
		base, _ := CalculateAPIServerConfigHash(newTestCluster())
		changed := newTestCluster()
		changed.ControlPlane.APIServer.ExtraArgs["--audit-log-path"] = "/var/log/audit2.log"
		got, _ := CalculateAPIServerConfigHash(changed)
		assert.NotEqual(t, base, got)
	})

	t.Run("extraVolumes change affects hash", func(t *testing.T) {
		base, _ := CalculateAPIServerConfigHash(newTestCluster())
		changed := newTestCluster()
		changed.ControlPlane.APIServer.ExtraVolumes = []confv1beta1.HostPathMount{
			{Name: "certs", HostPath: "/etc/certs", MountPath: "/certs"},
		}
		got, _ := CalculateAPIServerConfigHash(changed)
		assert.NotEqual(t, base, got)
	})

	t.Run("certSANs change affects hash", func(t *testing.T) {
		base, _ := CalculateAPIServerConfigHash(newTestCluster())
		changed := newTestCluster()
		changed.ControlPlane.APIServer.CertSANs = []string{"other.example.com"}
		got, _ := CalculateAPIServerConfigHash(changed)
		assert.NotEqual(t, base, got)
	})

	t.Run("apiEndpoint change affects hash", func(t *testing.T) {
		base, _ := CalculateAPIServerConfigHash(newTestCluster())
		changed := newTestCluster()
		changed.ControlPlane.APIServer.APIEndpoint = confv1beta1.APIEndpoint{Host: "10.0.0.2", Port: 6443}
		got, _ := CalculateAPIServerConfigHash(changed)
		assert.NotEqual(t, base, got)
	})

	t.Run("other components do not affect hash", func(t *testing.T) {
		base, _ := CalculateAPIServerConfigHash(newTestCluster())
		changed := newTestCluster()
		changed.ControlPlane.ControllerManager.ExtraArgs["--bind-address"] = "127.0.0.1"
		changed.ControlPlane.Scheduler.ExtraArgs["--profiles"] = "v2"
		got, _ := CalculateAPIServerConfigHash(changed)
		assert.Equal(t, base, got)
	})
}

// TestCalculateControllerManagerConfigHash 覆盖 ControllerManager hash 的正常与边界场景
func TestCalculateControllerManagerConfigHash(t *testing.T) {
	t.Run("nil cluster", func(t *testing.T) {
		got, err := CalculateControllerManagerConfigHash(nil)
		require.NoError(t, err)
		assert.Empty(t, got)
	})

	t.Run("nil controllerManager", func(t *testing.T) {
		got, err := CalculateControllerManagerConfigHash(&confv1beta1.Cluster{})
		require.NoError(t, err)
		assert.Empty(t, got)
	})

	t.Run("populated and sensitive to extraArgs/extraVolumes", func(t *testing.T) {
		base, err := CalculateControllerManagerConfigHash(newTestCluster())
		require.NoError(t, err)
		assert.Len(t, base, 64)

		changed := newTestCluster()
		changed.ControlPlane.ControllerManager.ExtraArgs["--bind-address"] = "127.0.0.1"
		got, _ := CalculateControllerManagerConfigHash(changed)
		assert.NotEqual(t, base, got)

		changed = newTestCluster()
		changed.ControlPlane.ControllerManager.ExtraVolumes = []confv1beta1.HostPathMount{
			{Name: "kubeconfig", HostPath: "/etc/kubernetes", MountPath: "/kubernetes"},
		}
		got, _ = CalculateControllerManagerConfigHash(changed)
		assert.NotEqual(t, base, got)
	})
}

// TestCalculateSchedulerConfigHash 覆盖 Scheduler hash 的正常与边界场景
func TestCalculateSchedulerConfigHash(t *testing.T) {
	t.Run("nil cluster", func(t *testing.T) {
		got, err := CalculateSchedulerConfigHash(nil)
		require.NoError(t, err)
		assert.Empty(t, got)
	})

	t.Run("nil scheduler", func(t *testing.T) {
		got, err := CalculateSchedulerConfigHash(&confv1beta1.Cluster{})
		require.NoError(t, err)
		assert.Empty(t, got)
	})

	t.Run("populated and sensitive to extraArgs/extraVolumes", func(t *testing.T) {
		base, err := CalculateSchedulerConfigHash(newTestCluster())
		require.NoError(t, err)
		assert.Len(t, base, 64)

		changed := newTestCluster()
		changed.ControlPlane.Scheduler.ExtraArgs["--profiles"] = "v2"
		got, _ := CalculateSchedulerConfigHash(changed)
		assert.NotEqual(t, base, got)

		changed = newTestCluster()
		changed.ControlPlane.Scheduler.ExtraVolumes = []confv1beta1.HostPathMount{
			{Name: "config", HostPath: "/etc/scheduler", MountPath: "/scheduler"},
		}
		got, _ = CalculateSchedulerConfigHash(changed)
		assert.NotEqual(t, base, got)
	})
}

// TestCalculateEtcdConfigHash_NilCluster 覆盖 cluster 为 nil 的提前返回分支
func TestCalculateEtcdConfigHash_NilCluster(t *testing.T) {
	got, err := CalculateEtcdConfigHash(nil)
	require.NoError(t, err)
	assert.Empty(t, got)
}

// TestCalculateEtcdConfigHash_NilEtcd 覆盖 cluster 非 nil 但 etcd 为 nil 的分支
func TestCalculateEtcdConfigHash_NilEtcd(t *testing.T) {
	nilEtcdHash, err := CalculateEtcdConfigHash(&confv1beta1.Cluster{})
	require.NoError(t, err)
	// Etcd == nil 与控制面组件对齐，返回 ""（无 etcd 可变配置）
	assert.Empty(t, nilEtcdHash)

	// 相同输入应得到相同结果
	again, err := CalculateEtcdConfigHash(&confv1beta1.Cluster{})
	require.NoError(t, err)
	assert.Equal(t, nilEtcdHash, again)

	// 配置了 etcd 后 hash 应变为非空且与原值不同
	withEtcd, err := CalculateEtcdConfigHash(newTestCluster())
	require.NoError(t, err)
	assert.NotEmpty(t, withEtcd)
	assert.NotEqual(t, nilEtcdHash, withEtcd)
}

// TestEtcdImmutableFieldsChanged_NilInputs 覆盖 old/new 为 nil，以及 Etcd 配置段新增/删除的分支
func TestEtcdImmutableFieldsChanged_NilInputs(t *testing.T) {
	cluster := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			Etcd: &confv1beta1.Etcd{DataDir: "/var/lib/etcd"},
		},
	}

	t.Run("old/new 为 nil 时不判定为不可变字段变更", func(t *testing.T) {
		tests := []struct {
			name    string
			oldConf *confv1beta1.Cluster
			newConf *confv1beta1.Cluster
		}{
			{"nil old", nil, cluster},
			{"nil new", cluster, nil},
			{"both nil", nil, nil},
			{"both etcd nil", &confv1beta1.Cluster{}, &confv1beta1.Cluster{}},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				changed, field := EtcdImmutableFieldsChanged(tt.oldConf, tt.newConf)
				assert.False(t, changed)
				assert.Empty(t, field)
			})
		}
	})

	t.Run("etcd 配置段新增时引入 DataDir 应判定为变更", func(t *testing.T) {
		// old.Etcd == nil 时不可变字段按零值参与比较，新增 DataDir 属于不可变字段变更
		changed, field := EtcdImmutableFieldsChanged(&confv1beta1.Cluster{}, cluster)
		assert.True(t, changed)
		assert.Equal(t, "dataDir", field)
	})

	t.Run("etcd 配置段删除时移除 DataDir 应判定为变更", func(t *testing.T) {
		// new.Etcd == nil 时不可变字段按零值参与比较，移除 DataDir 属于不可变字段变更
		changed, field := EtcdImmutableFieldsChanged(cluster, &confv1beta1.Cluster{})
		assert.True(t, changed)
		assert.Equal(t, "dataDir", field)
	})
}

// TestEtcdImmutableFieldsChanged_SingleImmutableFieldTriggers 验证 etcd 段新增/删除时，
// DataDir / ServerCertSANs / PeerCertSANs 任意一个出现即判定为变更，无需三个同时配置
func TestEtcdImmutableFieldsChanged_SingleImmutableFieldTriggers(t *testing.T) {
	cases := []struct {
		name      string
		etcd      *confv1beta1.Etcd
		wantField string
	}{
		{"仅 DataDir", &confv1beta1.Etcd{DataDir: "/var/lib/etcd"}, "dataDir"},
		{"仅 ServerCertSANs", &confv1beta1.Etcd{ServerCertSANs: []string{"etcd.example.com"}}, "serverCertSANs"},
		{"仅 PeerCertSANs", &confv1beta1.Etcd{PeerCertSANs: []string{"peer.example.com"}}, "peerCertSANs"},
	}

	for _, tt := range cases {
		t.Run("新增etcd段-"+tt.name, func(t *testing.T) {
			newCluster := &confv1beta1.Cluster{
				ControlPlane: confv1beta1.ControlPlane{Etcd: tt.etcd},
			}
			changed, field := EtcdImmutableFieldsChanged(&confv1beta1.Cluster{}, newCluster)
			assert.True(t, changed, "仅配置 %s 也应判定为不可变字段变更", tt.wantField)
			assert.Equal(t, tt.wantField, field)
		})

		t.Run("删除etcd段-"+tt.name, func(t *testing.T) {
			oldCluster := &confv1beta1.Cluster{
				ControlPlane: confv1beta1.ControlPlane{Etcd: tt.etcd},
			}
			changed, field := EtcdImmutableFieldsChanged(oldCluster, &confv1beta1.Cluster{})
			assert.True(t, changed, "仅移除 %s 也应判定为不可变字段变更", tt.wantField)
			assert.Equal(t, tt.wantField, field)
		})
	}
}

// TestEtcdImmutableFieldsChanged_EtcdSectionChangedWithMutableFields 覆盖 etcd 段新增/删除
// 时只涉及可变字段（ExtraArgs / ExtraVolumes）的场景：不应判定为变更
func TestEtcdImmutableFieldsChanged_EtcdSectionChangedWithMutableFields(t *testing.T) {
	etcdWithMutableFields := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			Etcd: &confv1beta1.Etcd{
				ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
					ExtraArgs: map[string]string{"heartbeat-interval": "500"},
					ExtraVolumes: []confv1beta1.HostPathMount{
						{Name: "etcd-extra", HostPath: "/data/etcd", MountPath: "/data/etcd"},
					},
				},
			},
		},
	}

	t.Run("新增 etcd 段仅配置可变字段", func(t *testing.T) {
		changed, field := EtcdImmutableFieldsChanged(&confv1beta1.Cluster{}, etcdWithMutableFields)
		assert.False(t, changed)
		assert.Empty(t, field)
	})

	t.Run("删除仅含可变字段的 etcd 段", func(t *testing.T) {
		changed, field := EtcdImmutableFieldsChanged(etcdWithMutableFields, &confv1beta1.Cluster{})
		assert.False(t, changed)
		assert.Empty(t, field)
	})
}

// TestEtcdImmutableFieldsChanged_CertSANs 覆盖 serverCertSANs / peerCertSANs 变化分支
func TestEtcdImmutableFieldsChanged_CertSANs(t *testing.T) {
	base := &confv1beta1.Cluster{
		ControlPlane: confv1beta1.ControlPlane{
			Etcd: &confv1beta1.Etcd{
				DataDir:        "/var/lib/etcd",
				ServerCertSANs: []string{"etcd1.example.com", "etcd2.example.com"},
				PeerCertSANs:   []string{"peer1.example.com"},
			},
		},
	}

	t.Run("serverCertSANs value changed", func(t *testing.T) {
		modified := base.DeepCopy()
		modified.ControlPlane.Etcd.ServerCertSANs = []string{"etcd1.example.com", "etcd3.example.com"}
		changed, field := EtcdImmutableFieldsChanged(base, modified)
		assert.True(t, changed)
		assert.Equal(t, "serverCertSANs", field)
	})

	t.Run("serverCertSANs length changed", func(t *testing.T) {
		modified := base.DeepCopy()
		modified.ControlPlane.Etcd.ServerCertSANs = []string{"etcd1.example.com"}
		changed, field := EtcdImmutableFieldsChanged(base, modified)
		assert.True(t, changed)
		assert.Equal(t, "serverCertSANs", field)
	})

	t.Run("serverCertSANs order changed", func(t *testing.T) {
		modified := base.DeepCopy()
		modified.ControlPlane.Etcd.ServerCertSANs = []string{"etcd2.example.com", "etcd1.example.com"}
		changed, field := EtcdImmutableFieldsChanged(base, modified)
		assert.True(t, changed)
		assert.Equal(t, "serverCertSANs", field)
	})

	t.Run("peerCertSANs changed", func(t *testing.T) {
		modified := base.DeepCopy()
		modified.ControlPlane.Etcd.PeerCertSANs = []string{"peer2.example.com"}
		changed, field := EtcdImmutableFieldsChanged(base, modified)
		assert.True(t, changed)
		assert.Equal(t, "peerCertSANs", field)
	})

	t.Run("peerCertSANs removed", func(t *testing.T) {
		modified := base.DeepCopy()
		modified.ControlPlane.Etcd.PeerCertSANs = nil
		changed, field := EtcdImmutableFieldsChanged(base, modified)
		assert.True(t, changed)
		assert.Equal(t, "peerCertSANs", field)
	})

	t.Run("nothing changed", func(t *testing.T) {
		changed, field := EtcdImmutableFieldsChanged(base, base.DeepCopy())
		assert.False(t, changed)
		assert.Empty(t, field)
	})
}

// TestComponentConfigHashChangedComponents_ControllerManager 覆盖 ControllerManager 单独变化的分支
func TestComponentConfigHashChangedComponents_ControllerManager(t *testing.T) {
	oldHash := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-1",
		ControllerManager: "hash-cm-1",
		Scheduler:         "hash-sched-1",
	}
	newHash := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-1",
		ControllerManager: "hash-cm-2",
		Scheduler:         "hash-sched-1",
	}

	changed := ComponentConfigHashChangedComponents(oldHash, newHash)
	assert.Equal(t, map[string]bool{"kube-controller-manager": true}, changed)

	// 三个组件同时变化
	allChanged := confv1beta1.ComponentConfigHash{
		APIServer:         "hash-api-2",
		ControllerManager: "hash-cm-2",
		Scheduler:         "hash-sched-2",
	}
	changed = ComponentConfigHashChangedComponents(oldHash, allChanged)
	assert.Len(t, changed, 3)
	assert.True(t, changed["kube-apiserver"])
	assert.True(t, changed["kube-controller-manager"])
	assert.True(t, changed["kube-scheduler"])
}

// TestComponentConfigHashJSONRoundTrip 验证序列化与反序列化的幂等性
func TestComponentConfigHashJSONRoundTrip(t *testing.T) {
	tests := []confv1beta1.ComponentConfigHash{
		{},
		{APIServer: "hash-api-1"},
		{APIServer: "hash-api-1", ControllerManager: "hash-cm-1", Scheduler: "hash-sched-1"},
	}

	for _, want := range tests {
		data, err := ComponentConfigHashToJSON(want)
		require.NoError(t, err)

		got, err := ComponentConfigHashFromJSON(data)
		require.NoError(t, err)
		assert.Equal(t, want, got)
	}
}

// TestComponentConfigHashFromJSON_PartialFields 验证部分字段与空数组的反序列化
func TestComponentConfigHashFromJSON_PartialFields(t *testing.T) {
	got, err := ComponentConfigHashFromJSON(`{"controllerManager":"hash-cm-1"}`)
	require.NoError(t, err)
	assert.Equal(t, confv1beta1.ComponentConfigHash{ControllerManager: "hash-cm-1"}, got)

	got, err = ComponentConfigHashFromJSON(`{}`)
	require.NoError(t, err)
	assert.Equal(t, confv1beta1.ComponentConfigHash{}, got)
}

// TestStringSlicesEqual_EdgeCases 覆盖空切片、nil 与顺序敏感的比较场景
func TestStringSlicesEqual_EdgeCases(t *testing.T) {
	tests := []struct {
		name string
		a    []string
		b    []string
		want bool
	}{
		{"nil vs empty", nil, []string{}, true},
		{"empty vs empty", []string{}, []string{}, true},
		{"same values different order", []string{"a", "b"}, []string{"b", "a"}, false},
		{"one element duplicated", []string{"a", "a"}, []string{"a"}, false},
		{"single element equal", []string{"a"}, []string{"a"}, true},
		{"single element different", []string{"a"}, []string{"b"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, utils.StringSlicesEqual(tt.a, tt.b))
		})
	}
}

// ======================= 序列化失败的错误分支 =======================

// patchJSONMarshalError 在第 failOn 次调用 json.Marshal 时返回错误，用于覆盖序列化失败分支。
// 未到指定次数时返回固定的合法 JSON，避免桩函数递归调用真实的 json.Marshal。
func patchJSONMarshalError(failOn int) *gomonkey.Patches {
	calls := 0
	return gomonkey.ApplyFunc(json.Marshal, func(any) ([]byte, error) {
		calls++
		if calls == failOn {
			return nil, assert.AnError
		}
		return []byte(`{}`), nil
	})
}

// TestComponentConfigHashToJSON_MarshalError 验证序列化失败时返回错误与空字符串
func TestComponentConfigHashToJSON_MarshalError(t *testing.T) {
	patches := patchJSONMarshalError(1)
	got, err := ComponentConfigHashToJSON(confv1beta1.ComponentConfigHash{APIServer: "hash-api-1"})
	patches.Reset()

	assert.Error(t, err)
	assert.Empty(t, got)
}

// TestCalculateAPIServerConfigHash_MarshalError 验证 APIServer 序列化失败时错误向上传递
func TestCalculateAPIServerConfigHash_MarshalError(t *testing.T) {
	patches := patchJSONMarshalError(1)
	got, err := CalculateAPIServerConfigHash(newTestCluster())
	patches.Reset()

	assert.Error(t, err)
	assert.Empty(t, got)
}

// TestCalculateControllerManagerConfigHash_MarshalError 验证 ControllerManager 序列化失败时错误向上传递
func TestCalculateControllerManagerConfigHash_MarshalError(t *testing.T) {
	patches := patchJSONMarshalError(1)
	got, err := CalculateControllerManagerConfigHash(newTestCluster())
	patches.Reset()

	assert.Error(t, err)
	assert.Empty(t, got)
}

// TestCalculateSchedulerConfigHash_MarshalError 验证 Scheduler 序列化失败时错误向上传递
func TestCalculateSchedulerConfigHash_MarshalError(t *testing.T) {
	patches := patchJSONMarshalError(1)
	got, err := CalculateSchedulerConfigHash(newTestCluster())
	patches.Reset()

	assert.Error(t, err)
	assert.Empty(t, got)
}

// TestCalculateEtcdConfigHash_MarshalError 验证 Etcd 序列化失败时错误向上传递
func TestCalculateEtcdConfigHash_MarshalError(t *testing.T) {
	patches := patchJSONMarshalError(1)
	got, err := CalculateEtcdConfigHash(newTestCluster())
	patches.Reset()

	assert.Error(t, err)
	assert.Empty(t, got)
}

// TestCalculateComponentConfigHash_MarshalError 验证三个组件任意一个序列化失败都会中断并返回错误
func TestCalculateComponentConfigHash_MarshalError(t *testing.T) {
	tests := []struct {
		name   string
		failOn int
	}{
		{"apiServer marshal failed", 1},
		{"controllerManager marshal failed", 2},
		{"scheduler marshal failed", 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := patchJSONMarshalError(tt.failOn)
			got, err := CalculateComponentConfigHash(newTestCluster())
			patches.Reset()

			assert.Error(t, err)
			assert.Equal(t, confv1beta1.ComponentConfigHash{}, got)
		})
	}
}
