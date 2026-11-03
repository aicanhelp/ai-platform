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
package switchcluster

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	Name = "SwitchCluster"
	// RwRR is the permission of the file
	RwRR   = 0644
	thirty = 30
)

var (
	kubeconfig = fmt.Sprintf("%s/%s", utils.Workspace, "config")
	node       = fmt.Sprintf("%s/%s", utils.Workspace, "node")
	cluster    = fmt.Sprintf("%s/%s", utils.Workspace, "cluster")
)

type SwitchClusterPlugin struct {
	K8sClient client.Client
}

func New(k8sClient client.Client) plugin.Plugin {
	return &SwitchClusterPlugin{
		K8sClient: k8sClient,
	}
}

func (s *SwitchClusterPlugin) Name() string {
	return Name
}

func (s *SwitchClusterPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"kubeconfig": plugin.PluginParam{
			Key:         "kubeconfig",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "Kubeconfig in Secret, format ns/secret",
		},
		"nodeName": plugin.PluginParam{
			Key:         "nodeName",
			Value:       "",
			Required:    false,
			Default:     utils.HostName(),
			Description: "Switch the cluster to the target cluster nodeName, default os.hostname",
		},
		"clusterName": plugin.PluginParam{
			Key:         "clusterName",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "Switch the cluster to the target cluster clusterName, default os.hostname",
		},
	}
}

func (s *SwitchClusterPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	var result []string
	// Parse command
	runtimeParam, err := plugin.ParseCommands(s, commands)
	if err != nil {
		return result, err
	}
	namespace, name, err := cache.SplitMetaNamespaceKey(runtimeParam["kubeconfig"])
	if err != nil {
		return result, err
	}
	if namespace == "" {
		namespace = "default"
	}
	config := &corev1.Secret{}
	err = s.K8sClient.Get(context.Background(), client.ObjectKey{Namespace: namespace, Name: name}, config)
	if err != nil {
		return result, err
	}
	if !utils.Exists(utils.Workspace) {
		err = os.MkdirAll(utils.Workspace, os.ModePerm)
		if err != nil {
			return result, err
		}
	}
	for _, value := range config.Data {
		err = ioutil.WriteFile(kubeconfig, value, RwRR)
		if err != nil {
			return result, err
		}
	}
	// Write the nodeName
	err = ioutil.WriteFile(node, []byte(runtimeParam["nodeName"]), RwRR)
	if err != nil {
		return result, nil
	}
	// Write the clusterName
	err = ioutil.WriteFile(cluster, []byte(runtimeParam["clusterName"]), RwRR)
	if err != nil {
		return result, nil
	}

	time.AfterFunc(thirty*time.Second, func() {
		log.Info("Switch over the listening cluster, exit normally...")
		// 设置为1 ，systemd 的service中设置了 Restart=on-failure SuccessExitStatus=0
		os.Exit(1)
	})
	result = append(result, "The listening cluster switch will take place in 30 seconds")
	return result, nil
}
