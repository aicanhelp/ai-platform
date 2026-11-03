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
	"github.com/gorilla/websocket"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"installer-service/pkg/utils/httputil"
	"installer-service/pkg/zlog"
)

// Operation all operations for installer
// 大写的方法导出，在别的文件也可以用
type Operation interface {
	GetClientSet() kubernetes.Interface
	GetDynamicClient() dynamic.Interface

	// 创建 - 接收结构化创建请求（来自 POST /clusters）
	CreateCluster(object string) error

	// 判断集群节点信息
	JudgeClusterNode(nodeInfo *ClusterNodeInfo) error

	// 日志
	GetClusterLog(namespace string, conn *websocket.Conn) (*httputil.ResponseJson, int)

	// 获取集群列表
	GetClusters() ([]configv1beta1.BKECluster, error)
	GetClustersByName(name string) (*configv1beta1.BKECluster, error)
	GetClustersByQuery(req ClusterRequest) (ClusterResponse, error)
	// GetClusterFull returns assembled cluster config + nodes for API GET /clusters/{cluster-name}
	GetClusterFull(clusterName string) (ClusterFullResponse, error)
	// GetAllClusters returns all clusters mapped to installer.ClusterResponse (used by GET /clusters)
	GetAllClusters() (ClusterResponse, error)

	// 获取节点详情列表
	GetNodesByQuery(req NodeRequest) (NodeResponse, error)

	// 删除
	DeleteCluster(clusterName string) (*httputil.ResponseJson, int)

	// 离线安装的配置
	GetDefaultConfig() (DefaultResp, error)
	GetClusterConfig(clusterName string) (ClusterConfig, error)

	PatchYaml(object string, isUpgrade bool) error
	ScaleDownCluster(yaml string, ip string) error
	// CreateBKENodes creates BKENode CRs in the given cluster namespace (scale-up)
	CreateBKENodes(clusterName string, nodes []ClusterNode) error
	// DeleteBKENodes deletes BKENode CRs by metadata.name, hostname, or IP
	DeleteBKENodes(clusterName string, nodeNames []string) error

	// 上传升级patch文件
	UploadPatchFile(fileName string, fileContent string) error

	// 获取可用的更新openFuyaoVersion
	GetOpenFuyaoVersions() ([]string, error)

	// GetInstallableVersions returns versions that can be used for fresh installs.
	GetInstallableVersions() ([]string, error)

	// 获取可升级的openFuyaoVersion
	GetOpenFuyaoUpgradeVersions(currentVersion string) ([]string, error)

	// GetUpgradeableVersions returns versions reachable from currentVersion.
	GetUpgradeableVersions(currentVersion string) ([]string, error)

	// test
	AutoUpgradePatchPrepare(clusterName string, req AutoUpgradeRequest) error

	// UpgradeOpenFuyao upgrades the openFuyao version of the given cluster by patching the BKECluster CR
	UpgradeOpenFuyao(clusterName string, version string) error

	// UpgradeClusterVersion declares the desired version on the ClusterVersion CR.
	UpgradeClusterVersion(clusterName string, version string) error
}

type installerClient struct {
	kubeConfig    *rest.Config
	dynamicClient dynamic.Interface
	clientset     kubernetes.Interface
}

// installer operation requires client for kubernetes resource operation
func NewInstallerOperation(kubeConfig *rest.Config) (Operation, error) {
	dynamicClient, err := dynamic.NewForConfig(kubeConfig)
	if err != nil {
		zlog.Error("error creating dynamic client, err: %v", err)
		return nil, err
	}

	clientset, err := kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		zlog.Error("error creating client set,err: %v", err)
		return nil, err
	}

	return &installerClient{
		kubeConfig:    kubeConfig,
		dynamicClient: dynamicClient,
		clientset:     clientset,
	}, nil
}

// NewInstallerOperationWithClients returns an Operation backed by the provided
// kubernetes clientset and dynamic client. This helper is intended for tests
// where a fake clientset / dynamic client are injected so that the real
// installer implementation is exercised while avoiding network calls to a
// real kube-apiserver.
func NewInstallerOperationWithClients(clientset kubernetes.Interface, dynamicClient dynamic.Interface) (Operation, error) {
	return &installerClient{
		kubeConfig:    nil,
		dynamicClient: dynamicClient,
		clientset:     clientset,
	}, nil
}
