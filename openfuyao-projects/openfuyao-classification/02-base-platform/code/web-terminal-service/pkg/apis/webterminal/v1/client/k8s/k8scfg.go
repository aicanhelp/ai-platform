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

// Package k8s create kubernetes config
package k8s

import (
	"errors"
	"os"
	"os/user"
	"path"

	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"openfuyao.com/web-terminal-service/pkg/zlog"
)

// KubernetesCfg kubernetes config
type KubernetesCfg struct {
	KubeConfigFile string `json:"kubeconfig" yaml:"kubeconfig"`

	KubeConfig *rest.Config

	// kubernetes clientset qps
	QPS float32 `json:"qps,omitempty" yaml:"qps,omitempty"`

	// kubernetes clientset burst
	Burst int `json:"burst,omitempty" yaml:"burst,omitempty"`
}

// NewKubernetesCfg return default k8s related config (KubeConfig)
func NewKubernetesCfg() *KubernetesCfg {
	return &KubernetesCfg{
		KubeConfigFile: getKubeConfigFile(),
		KubeConfig:     GetKubeConfig(),
		QPS:            1e6,
		Burst:          1e6,
	}
}

// Validate validate kubernetes config
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

// GetKubeConfig get kubernetes config, either from cluster or from local path
func GetKubeConfig() *rest.Config {
	// 在集群中获取k8s配置信息
	config, err := rest.InClusterConfig()
	if err != nil {
		zlog.LogWarn("Get KubeConfig In Cluster Config error, Attempting to obtain from the configuration file")
		kubeConfigFile := getKubeConfigFile()
		if kubeConfigFile == "" {
			zlog.LogFatalf("LogError creating in-cluster config: %v", err)
		}
		if _, err1 := os.Stat(kubeConfigFile); err1 != nil {
			zlog.LogFatalf("LogError creating in-filePath config: %v", err)
		}
		if configFromFile, err2 := clientcmd.BuildConfigFromFlags("", kubeConfigFile); err2 == nil {
			return configFromFile
		}
		zlog.LogFatalf("LogError creating in-filePath config: %v", err)
	}
	return config
}

func getKubeConfigFile() string {
	kubeConfig := ""
	homePath := homedir.HomeDir()
	if homePath == "" {
		if u, err := user.Current(); err == nil {
			homePath = u.HomeDir
		}
	}

	userHomeConfig := path.Join(homePath, ".kube/config")
	if _, err := os.Stat(userHomeConfig); err == nil {
		kubeConfig = userHomeConfig
	}
	return kubeConfig
}
