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

// Package k8s used for k8s config
package k8s

import (
	"os"
	"os/user"
	"path"

	"github.com/pkg/errors"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"monitoring-dashboard-service/pkg/utils/zlog"
)

// KubernetesCfg k8s 配置
type KubernetesCfg struct {
	KubeConfigFile string `json:"kubeconfig" yaml:"kubeconfig"`

	KubeConfig *rest.Config

	QPS float32 `json:"qps,omitempty" yaml:"qps,omitempty"`

	Burst int `json:"burst,omitempty" yaml:"burst,omitempty"`
}

// NewKubernetesCfg 返回默认的 k8s 相关配置（如KubeConfig）
func NewKubernetesCfg() *KubernetesCfg {
	return &KubernetesCfg{
		KubeConfigFile: getKubeConfigFile(),
		KubeConfig:     GetKubeConfig(),
		QPS:            1e6,
		Burst:          1e6,
	}
}

// Validate 验证 k8s 配置
func (k *KubernetesCfg) Validate() []error {
	var errs []error
	if k.KubeConfigFile != "" {
		if _, err := os.Stat(k.KubeConfigFile); err != nil {
			errs = append(errs, err)
		}
	}
	if k.KubeConfig == nil {
		errs = append(errs, errors.New("k8s config get nil"))
	}
	return errs
}

// GetKubeConfig 获取k8s配置
func GetKubeConfig() *rest.Config {
	// 在集群中获取k8s配置信息
	config, err := rest.InClusterConfig()
	if err != nil {
		zlog.LogWarn("Get KubeConfig In Cluster Config error, Attempting to obtain from the configuration file")
		kubeConfigFile := getKubeConfigFile()
		if kubeConfigFile == "" {
			zlog.LogError("Error creating in-cluster config: %v", err)
		}
		if _, err1 := os.Stat(kubeConfigFile); err1 != nil {
			zlog.LogError("Error creating in-filePath config: %v", err)
		}
		if configFromFile, err2 := clientcmd.BuildConfigFromFlags("", kubeConfigFile); err2 == nil {
			return configFromFile
		}
		zlog.LogError("Error creating in-filePath config: %v", err)
	}
	return config
}

// getKubeConfigFile 获取指定路径的k8s配置
func getKubeConfigFile() string {
	kubeConfigPath := ""

	// 获取用户主目录
	userHomeDir := homedir.HomeDir()
	if userHomeDir == "" {
		currentUser, userErr := user.Current()
		if userErr == nil && currentUser.HomeDir != "" {
			userHomeDir = currentUser.HomeDir
		}
	}

	// 检查主目录下的 kube/config 文件是否存在
	if userHomeDir != "" {
		configFilePath := path.Join(userHomeDir, ".kube/config")
		if fileInfo, statErr := os.Stat(configFilePath); statErr == nil && !fileInfo.IsDir() {
			kubeConfigPath = configFilePath
		}
	}

	return kubeConfigPath
}
