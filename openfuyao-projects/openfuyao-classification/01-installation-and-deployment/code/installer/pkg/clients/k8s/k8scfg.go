/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package k8s

import (
	"fmt"
	"os"
	"os/user"
	"path"
	"path/filepath"

	"github.com/pkg/errors"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

const (
	kubeconfigPath = ".kube/config"
)

// KubernetesCfg is k8s config
type KubernetesCfg struct {
	KubeConfigFile string `json:"kubeconfig" yaml:"kubeconfig"`

	KubeConfig *rest.Config

	// kubernetes clientset qps
	QPS float32 `json:"qps,omitempty" yaml:"qps,omitempty"`

	// kubernetes clientset burst
	Burst int `json:"burst,omitempty" yaml:"burst,omitempty"`
}

// NewKubernetesCfg 返回默认的 k8s 相关配置（如KubeConfig）
func NewKubernetesCfg(kubeCfgFilePath string) *KubernetesCfg {
	return &KubernetesCfg{
		KubeConfigFile: getKubeConfigFile(kubeCfgFilePath),
		KubeConfig:     GetKubeConfig(kubeCfgFilePath),
		QPS:            1e6,
		Burst:          1e6,
	}
}

func getKubeConfigFile(kubeCfgFilePath string) string {
	if kubeCfgFilePath != "" {
		return kubeCfgFilePath
	}
	homePath := homedir.HomeDir()
	if homePath == "" {
		if u, err := user.Current(); err == nil {
			homePath = u.HomeDir
		}
	}

	userHomeConfig := path.Join(homePath, kubeconfigPath)
	if _, err := os.Stat(userHomeConfig); err == nil {
		kubeCfgFilePath = userHomeConfig
	}
	return kubeCfgFilePath
}

// Validate is validate kubeconfig
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

// GetKubeConfig is get kubeconfig
func GetKubeConfig(kubeCfgFilePath string) *rest.Config {
	if kubeCfgFilePath != "" {
		return readKubeConfig(kubeCfgFilePath)
	}

	// 在集群中获取k8s配置信息
	config, err := rest.InClusterConfig()
	if err != nil {
		zlog.Warn("Get KubeConfig In Cluster Config error, Attempting to obtain from the configuration file")
		kubeConfigFile := getKubeConfigFile(kubeCfgFilePath)
		return readKubeConfig(kubeConfigFile)
	}
	return config
}

func readKubeConfig(kubeCfgFilePath string) *rest.Config {
	if kubeCfgFilePath == "" {
		zlog.Fatalf("Kubeconfig file path is empty")
	}
	if _, err := os.Stat(kubeCfgFilePath); err != nil {
		zlog.Fatalf("Not found kubeconfig: %v", err)
	}

	configFromFile, err := clientcmd.BuildConfigFromFlags("", kubeCfgFilePath)
	if err != nil {
		zlog.Fatalf("Error creating in-filePath config: %v", err)
	}
	return configFromFile
}

// DownLoadKubeConfig is download kubeconfig
func DownLoadKubeConfig(node *cluster.Node) (string, error) {
	cfgSavePath := filepath.Join(constants.OptPath, constants.OpenFuyao, constants.AuthDir)
	kubeconfigFile := filepath.Join(cfgSavePath, constants.KubeConfig)
	if _, err := os.Stat(kubeconfigFile); err == nil {
		return kubeconfigFile, nil
	}

	if err := utils.MkdirAll(cfgSavePath, constants.RwxRxRx); err != nil {
		return "", err
	}

	client := node.Client()
	remoteConfigPath := filepath.Join(constants.OptPath, constants.OpenFuyao)
	_, err := client.SudoExecCmd(fmt.Sprintf("cd %s && kubectl config view --raw > %s && chmod %o %s",
		remoteConfigPath, constants.KubeConfig, constants.RwxRxRx,
		filepath.Join(remoteConfigPath, constants.KubeConfig)))
	if err != nil {
		return "", err
	}

	err = client.DownloadFile(kubeconfigFile, filepath.Join(remoteConfigPath, constants.KubeConfig))
	if err != nil {
		return "", err
	}
	_, err = utils.ExecLocalSudoCmd(fmt.Sprintf("chmod %o %s", constants.RwxRxRx, kubeconfigFile))
	if err != nil {
		zlog.Error("failed to chmod kube-config file")
		return "", err
	}
	return kubeconfigFile, nil
}
