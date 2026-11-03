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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"reflect"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	mfutil "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
)

// CalculateComponentConfigHash 根据 Cluster 配置计算控制面组件的期望 Hash 结构体
func CalculateComponentConfigHash(cluster *confv1beta1.Cluster) (confv1beta1.ComponentConfigHash, error) {
	apiHash, err := CalculateAPIServerConfigHash(cluster)
	if err != nil {
		return confv1beta1.ComponentConfigHash{}, err
	}
	cmHash, err := CalculateControllerManagerConfigHash(cluster)
	if err != nil {
		return confv1beta1.ComponentConfigHash{}, err
	}
	schedulerHash, err := CalculateSchedulerConfigHash(cluster)
	if err != nil {
		return confv1beta1.ComponentConfigHash{}, err
	}

	return confv1beta1.ComponentConfigHash{
		APIServer:         apiHash,
		ControllerManager: cmHash,
		Scheduler:         schedulerHash,
	}, nil
}

// ComponentConfigHashChanged 比较新旧两个 Hash 结构体是否发生变化，供 phase 触发判定使用。
func ComponentConfigHashChanged(oldHash, newHash confv1beta1.ComponentConfigHash) bool {
	return !reflect.DeepEqual(oldHash, newHash)
}

// ComponentConfigHashToJSON 将 hash 结构体序列化为 JSON 字符串，用于 command 参数传递。
func ComponentConfigHashToJSON(hash confv1beta1.ComponentConfigHash) (string, error) {
	data, err := json.Marshal(hash)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ComponentConfigHashFromJSON 将 JSON 字符串反序列化为 hash 结构体，供 bkeagent 解析。
func ComponentConfigHashFromJSON(data string) (confv1beta1.ComponentConfigHash, error) {
	var hash confv1beta1.ComponentConfigHash
	if data == "" {
		return hash, nil
	}
	if err := json.Unmarshal([]byte(data), &hash); err != nil {
		return hash, err
	}
	return hash, nil
}

// ComponentConfigHashChangedComponents 逐组件比较新旧两个 Hash 结构体，
// 返回哪些组件发生变化。key 为组件名，与 mfutil.GetControlPlaneComponents() 保持一致：
// kube-apiserver / kube-controller-manager / kube-scheduler。
func ComponentConfigHashChangedComponents(oldHash, newHash confv1beta1.ComponentConfigHash) map[string]bool {
	changed := map[string]bool{}
	if oldHash.APIServer != newHash.APIServer {
		changed[mfutil.KubeAPIServer] = true
	}
	if oldHash.ControllerManager != newHash.ControllerManager {
		changed[mfutil.KubeControllerManager] = true
	}
	if oldHash.Scheduler != newHash.Scheduler {
		changed[mfutil.KubeScheduler] = true
	}
	return changed
}

// normalizeAPIServer normalizes APIServer configuration
func normalizeAPIServer(apiServer *confv1beta1.APIServer) interface{} {
	return struct {
		ExtraArgs    map[string]string           `json:"extraArgs"`
		ExtraVolumes []confv1beta1.HostPathMount `json:"extraVolumes"`
		CertSANs     []string                    `json:"certSANs"`
		APIEndpoint  confv1beta1.APIEndpoint     `json:"apiEndpoint"`
	}{
		ExtraArgs:    apiServer.ExtraArgs,
		ExtraVolumes: apiServer.ExtraVolumes,
		CertSANs:     apiServer.CertSANs,
		APIEndpoint:  apiServer.APIEndpoint,
	}
}

// normalizeControlPlaneComponent normalizes a control plane component configuration
func normalizeControlPlaneComponent(comp *confv1beta1.ControlPlaneComponent) interface{} {
	return struct {
		ExtraArgs    map[string]string           `json:"extraArgs"`
		ExtraVolumes []confv1beta1.HostPathMount `json:"extraVolumes"`
	}{
		ExtraArgs:    comp.ExtraArgs,
		ExtraVolumes: comp.ExtraVolumes,
	}
}

// normalizeEtcd normalizes Etcd configuration for mutable fields only.
// Only ExtraArgs and ExtraVolumes are included in the hash because
// DataDir, ServerCertSANs, PeerCertSANs are bootstrap-time immutable
// fields that must not change after cluster creation.
func normalizeEtcd(etcd *confv1beta1.Etcd) interface{} {
	return struct {
		ExtraArgs    map[string]string           `json:"extraArgs"`
		ExtraVolumes []confv1beta1.HostPathMount `json:"extraVolumes"`
	}{
		ExtraArgs:    etcd.ExtraArgs,
		ExtraVolumes: etcd.ExtraVolumes,
	}
}

// NormalizeAPIServerForHash normalizes APIServer configuration into a hashable structure.
func NormalizeAPIServerForHash(apiServer *confv1beta1.APIServer) interface{} {
	return normalizeAPIServer(apiServer)
}

// NormalizeControlPlaneComponentForHash normalizes a control plane component into a hashable structure.
func NormalizeControlPlaneComponentForHash(comp *confv1beta1.ControlPlaneComponent) interface{} {
	return normalizeControlPlaneComponent(comp)
}

// NormalizeEtcdForHash normalizes etcd configuration (mutable fields only) into a hashable structure.
func NormalizeEtcdForHash(etcd *confv1beta1.Etcd) interface{} {
	return normalizeEtcd(etcd)
}

// CalculateAPIServerConfigHash calculates the hash of the APIServer static Pod configuration.
func CalculateAPIServerConfigHash(cluster *confv1beta1.Cluster) (string, error) {
	if cluster == nil || cluster.APIServer == nil {
		return "", nil
	}
	data, err := json.Marshal(normalizeAPIServer(cluster.APIServer))
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

// CalculateControllerManagerConfigHash calculates the hash of the ControllerManager static Pod configuration.
func CalculateControllerManagerConfigHash(cluster *confv1beta1.Cluster) (string, error) {
	if cluster == nil || cluster.ControllerManager == nil {
		return "", nil
	}
	data, err := json.Marshal(normalizeControlPlaneComponent(cluster.ControllerManager))
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

// CalculateSchedulerConfigHash calculates the hash of the Scheduler static Pod configuration.
func CalculateSchedulerConfigHash(cluster *confv1beta1.Cluster) (string, error) {
	if cluster == nil || cluster.Scheduler == nil {
		return "", nil
	}
	data, err := json.Marshal(normalizeControlPlaneComponent(cluster.Scheduler))
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

// CalculateEtcdConfigHash calculates the hash of etcd runtime configuration only.
// 与控制面组件对齐：Etcd == nil（无任何 etcd 可变配置）时返回 ""，
// 使上层 desiredHash == "" 的「无 etcd 可变配置」语义成立。
func CalculateEtcdConfigHash(cluster *confv1beta1.Cluster) (string, error) {
	if cluster == nil || cluster.Etcd == nil {
		return "", nil
	}

	etcd := normalizeEtcd(cluster.Etcd)

	data, err := json.Marshal(etcd)
	if err != nil {
		return "", err
	}

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

// EtcdImmutableFieldsChanged checks whether any bootstrap-time immutable etcd
// fields (DataDir, ServerCertSANs, PeerCertSANs) have changed between old and new
// cluster configurations.
//
// etcd 配置段缺失（Etcd == nil）时，其不可变字段按零值参与比较，因此：
//   - 新增 etcd 配置段本身是允许的；只有当新增的内容中 DataDir / ServerCertSANs /
//     PeerCertSANs 任意一个有值时才判定为变更，即不允许事后引入引导期配置；
//   - 删除整个 etcd 配置段时，若原配置中上述任一字段有值，同样判定为变更；
//   - 仅涉及 ExtraArgs / ExtraVolumes 等可变字段的新增、修改、删除不受影响。
func EtcdImmutableFieldsChanged(old, new *confv1beta1.Cluster) (changed bool, fieldName string) {
	if old == nil || new == nil {
		return false, ""
	}
	oldDataDir, oldServerSANs, oldPeerSANs := etcdImmutableFields(old.Etcd)
	newDataDir, newServerSANs, newPeerSANs := etcdImmutableFields(new.Etcd)

	if oldDataDir != newDataDir {
		return true, "dataDir"
	}
	if !utils.StringSlicesEqual(oldServerSANs, newServerSANs) {
		return true, "serverCertSANs"
	}
	if !utils.StringSlicesEqual(oldPeerSANs, newPeerSANs) {
		return true, "peerCertSANs"
	}
	return false, ""
}

// etcdImmutableFields 提取 etcd 的引导期不可变字段；etcd 配置段缺失时返回零值，
// 使 etcd 段的新增/删除可以等价为不可变字段值的新增/删除参与比较。
func etcdImmutableFields(etcd *confv1beta1.Etcd) (dataDir string, serverCertSANs, peerCertSANs []string) {
	if etcd == nil {
		return "", nil, nil
	}
	return etcd.DataDir, etcd.ServerCertSANs, etcd.PeerCertSANs
}

// stringSlicesEqual compares two string slices for equality.
