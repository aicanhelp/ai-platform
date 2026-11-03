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

import "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"

type ClusterInfo struct {
	Yaml string `json:"yamlString"`
}

type ClusterRequest struct {
	Name        string `json:"name"`
	Status      string `json:"status"`
	CurrentPage int    `json:"currentPage"`
	PageSize    int    `json:"pageSize"`
}

type ClusterData struct {
	Name              string   `json:"name"`
	OpenFuyaoVersion  string   `json:"openFuyaoVersion"`
	KubernetesVersion string   `json:"kubernetesVersion"`
	ContainerdVersion string   `json:"containerdVersion"`
	Status            string   `json:"status"`
	CreateTime        string   `json:"createTime"`
	NodeSum           int      `json:"nodeSum"`
	IsAmd             bool     `json:"isAmd"`
	IsArm             bool     `json:"isArm"`
	OsList            []string `json:"osList"`
	ContainerRuntime  string   `json:"containerRuntime"`
	// IsAdminCluster indicates whether the cluster is a management cluster.
	// True when addons contain cluster-api, or the self-managed CR carries
	// bke.bocloud.com/cluster-api-manager-applied=true (addons may be absent).
	IsAdminCluster bool `json:"isAdminCluster"`
}

// AutoUpgradeRequest POST请求体结构
type AutoUpgradeRequest struct {
	PatchDir  string `json:"patchDir"`  // 补丁文件所在的目录路径。
	PatchName string `json:"patchName"` // 补丁文件的名称，不含扩展名。
}

// AutoUpgradeResponse POST响应体结构，包含任务ID
type AutoUpgradeResponse struct {
	TaskID string `json:"taskID"`
}

// AutoUpgradeStatusResponse GET响应体结构
type AutoUpgradeStatusResponse struct {
	TaskID string `json:"taskID"`
	Status string `json:"status"`
	Log    string `json:"log,omitempty"` // 仅在 FAILED 状态时包含
}

type ClusterResponse struct {
	Items []*ClusterData `json:"items"`
}

// ClusterFullResponse 表示 GET /clusters/{cluster-name} 的响应体结构
// 复用已有 ClusterConfig 和 NodeData，避免重复字段定义。
type ClusterFullResponse struct {
	ClusterConfig
	Nodes []NodeData `json:"nodes"`
}

type NodeData struct {
	Hostname     string   `json:"hostname"`
	Ip           string   `json:"ip"`
	Role         []string `json:"role"`
	Cpu          int64    `json:"cpu"`
	Memory       float64  `json:"memory"`
	Architecture string   `json:"architecture"`
	Status       string   `json:"status"`
	Os           string   `json:"os"`
}

type NodeResponse struct {
	Items []NodeData `json:"items"`
}

type DefaultResp struct {
	ImageRepo         v1beta1.Repo `json:"imageRepo"`
	HttpRepo          v1beta1.Repo `json:"httpRepo"`
	Ip                string       `json:"ip"`
	KubernetesVersion string       `json:"kubernetesVersion"`
	ContainerRuntime  string       `json:"containerRuntime"`
	AgentHealthPort   string       `json:"agentHealthPort"`
}

type ClusterConfig struct {
	ClusterName      string `json:"clusterName"`
	ClusterStatus    string `json:"clusterStatus"`
	OpenFuyaoVersion string `json:"openFuyaoVersion"`
	CreateTime       string `json:"createTime"`
	Host             string `json:"Host"`
	IsHA             bool   `json:"isHA"`
	DefaultResp
}

type DefaultYaml struct {
	Yaml string `json:"yaml" yaml:"yaml"`
}

type NodeRequest struct {
	ClusterName string `json:"clusterName"`
	NodeName    string `json:"nodeName"`
}

type ClusterName struct {
	ClusterName string `json:"clusterName"`
}

type ClusterDownInfo struct {
	Yaml string `json:"yamlString"`
	Ip   string `json:"nodeIp"`
}

// ScaleDownRequest 前端传入的缩容请求体，包含要删除的 BKENode 名称列表
type ScaleDownRequest struct {
	Nodes []string `json:"nodes"`
}

// ScaleUpRequest 前端传入的扩容请求体，包含要新增的节点信息
type ScaleUpRequest struct {
	Nodes []ClusterNode `json:"nodes"`
}

// ClusterNode /* 节点校验中节点详细信息 */
type ClusterNode struct {
	Hostname string `json:"hostname"`
	Ip       string `json:"ip"`
	Port     string `json:"port"`
	Username string `json:"username"`
	Password string `json:"password"`
	// Role roles assigned to this node, e.g. ["master","etcd"]
	Role []string `json:"role"`
}

// ClusterNodeInfo /* 节点校验中前后端约定的json格式数据 */
type ClusterNodeInfo struct {
	NameSpace string        `json:"nameSpace"`
	Nodes     []ClusterNode `json:"nodes"`
	BalanceIp string        `json:"balanceIp"`
}

// PatchFileData The content of frontend uploaded request
type PatchFileData struct {
	FileName    string `json:"PatchFileName"`
	FileContent string `json:"PatchFileContent"`
}

type ClusterPatchInfo struct {
	ClusterVersion string `json:"clusterVersion"`
}

// UpgradeRequest represents POST /clusters/{cluster-name}/upgrade payload
type UpgradeRequest struct {
	Version string `json:"version"`
}

// CreateClusterRequest is the create-cluster payload (frontend sends values only).
type CreateClusterRequest struct {
	Cluster              CreateClusterCluster `json:"cluster"`
	ControlPlaneEndpoint string               `json:"controlPlaneEndpoint"`
	Addons               []CreateClusterAddon `json:"addons"`
	Nodes                []CreateClusterNode  `json:"nodes"`
}

type CreateClusterCluster struct {
	Name             string                 `json:"name"`
	OpenFuyaoVersion string                 `json:"openFuyaoVersion"`
	ImageRepo        CreateClusterImageRepo `json:"imageRepo"`
}

type CreateClusterImageRepo struct {
	Url string `json:"url"`
	Ip  string `json:"ip"`
}

type CreateClusterAddon struct {
	Name   string            `json:"name"`
	Params map[string]string `json:"params"`
}

type CreateClusterNode struct {
	Hostname string   `json:"hostname"`
	Ip       string   `json:"ip"`
	Port     string   `json:"port"`
	Username string   `json:"username"`
	Password string   `json:"password"`
	Role     []string `json:"role"`
}

// RemotePatchIndexResponse The content of index.yaml from remote
type RemotePatchIndexResponse []map[string]string

// PatchVersionInfo The needed versions to patch base versions in bkeCluster
// Needed to be read from patch file
type PatchVersionInfo struct {
	OpenFuyaoVersion  string `yaml:"openFuyaoVersion" json:"openFuyaoVersion"`
	KubernetesVersion string `yaml:"kubernetesVersion" json:"kubernetesVersion"`
	ContainerdVersion string `yaml:"containerdVersion" json:"containerdVersion"`
	EtcdVersion       string `yaml:"etcdVersion" json:"etcdVersion"`
}

// PatchConfig addons
type PatchConfig struct {
	Addons []AddonInfo `yaml:"addons" json:"addons"`
}

// AddonInfo Addon info in bke cluster
type AddonInfo struct {
	Name    string            `yaml:"name" json:"name"`
	Version string            `yaml:"version" json:"version"`
	Param   map[string]string `yaml:"param" json:"param"`
	Block   bool              `yaml:"block" json:"block"`
}
