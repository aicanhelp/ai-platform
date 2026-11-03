/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package common

import (
	"os"
	"path/filepath"
	"strings"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
)

// Argument is common parameter
type Argument struct {
	FilePath      string
	AllComponent  bool
	Debug         bool
	FuyaoZone     string
	ArtifactPath  string
	EnableOffline bool
	InitCfg       string
	SkipRmTmpFile bool
	EnableAI      bool

	// 创建集群配置文件
	EnableAllParams      bool
	EnableAdvancedParams bool
	EnableHaParams       bool
	// 文件保存路径
	FileSaveFolder string

	AddNodes       bool
	DestroyCluster bool

	EnableSubpackageTar bool

	// 镜像仓
	RegistryNodeUsername   string
	RegistryNodeIP         string
	RegistryNodePasswd     []byte
	RegistryDomain         string
	RegistryPasswd         []byte
	RegistryDBPasswd       []byte
	RegistryCertExpiryDays string
	SkipDestroyHub         bool
}

// ClearSensitiveData 清除敏感数据
func (a *Argument) ClearSensitiveData() {
	a.RegistryPasswd = []byte{}
	a.RegistryDBPasswd = []byte{}
	a.RegistryNodePasswd = []byte{}
}

// KubeRuntime is load config file struct
type KubeRuntime struct {
	ClusterName string
	Cluster     *cluster.Cluster
	Kubeconfig  string
	Arg         Argument
}

// ClearSensitiveData 清除敏感数据
func (r *KubeRuntime) ClearSensitiveData() {
	r.Cluster.Spec.Nodes.ClearNodesPasswd()
	r.Arg.ClearSensitiveData()
}

// NewKubeRuntime 构建运行时的数据
func NewKubeRuntime(arg Argument) (*KubeRuntime, error) {
	loader := NewLoader(arg)
	// cluster 集群结构体
	clusterCfg, err := loader.Load()
	if err != nil {
		return nil, err
	}

	r := &KubeRuntime{
		Cluster:     clusterCfg,
		ClusterName: clusterCfg.Metadata.Name,
		Arg:         arg,
	}

	return r, nil
}

// GetServiceName 获取当前服务名字
func GetServiceName() string {
	serviceName := constants.Fuyao
	exePath, err := os.Executable()
	if err == nil {
		serviceName = strings.TrimSuffix(filepath.Base(exePath), filepath.Ext(filepath.Base(exePath)))
	}
	return serviceName
}
