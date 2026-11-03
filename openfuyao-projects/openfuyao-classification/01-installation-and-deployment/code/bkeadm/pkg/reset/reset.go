/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package reset

import (
	"os"
	"path"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/source"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/syscompat"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/timezone"
	"gopkg.openfuyao.cn/bkeadm/pkg/root"
	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

type Options struct {
	root.Options
	Args  []string `json:"args"`
	All   bool     `json:"all"`
	Mount bool     `json:"mount"`
}

func (op *Options) Reset() {
	RemoveLocalKubernetes()
	if op.All {
		RemoveContainerService()
		RemoveNtpService()
		removeAllInOne()
		err := source.ResetSource()
		if err != nil {
			log.Warnf("Reset source error: %v", err)
		}
		err = syscompat.RepoUpdate()
		if err != nil {
			log.Warnf("Update source error: %v", err)
		}
	}
	if op.Mount {
		RemoveDataDir()
		// 只在 bke reset --all --mount 路径下调用：syscompat.RemoveBKEManagedConfig() 和 timezone.RestoreTimeZone()。
		if op.All {
			syscompat.RemoveBKEManagedConfig()
			timezone.RestoreTimeZone()
		}
		if !op.All {
			RemoveContainerService()
			RemoveNtpService()
			err := source.ResetSource()
			if err != nil {
				log.Warnf("Reset source error: %v", err)
			}
		}
	}
	log.Info("BKE reset completed")
}

func RemoveLocalKubernetes() {
	if !infrastructure.IsDocker() && !infrastructure.IsContainerd() {
		return
	}
	log.Infof("Remove local Kubernetes")
	removeContainerWithProcessFallback(utils.LocalKubernetesName, isResidualK3sProcess)
	if err := os.RemoveAll("/etc/rancher"); err != nil {
		log.Warnf("Failed to remove /etc/rancher: %v", err)
	}
	if err := os.RemoveAll("/var/lib/rancher"); err != nil {
		log.Warnf("Failed to remove /var/lib/rancher: %v", err)
	}
}

func RemoveContainerService() {
	_ = server.RemoveImageRegistry(utils.LocalImageRegistryName)
	_ = server.RemoveChartRegistry(utils.LocalChartRegistryName)
	_ = server.RemoveYumRegistry(utils.LocalYumRegistryName)
	_ = server.RemoveNFSServer(utils.LocalNFSRegistryName)
}

func RemoveNtpService() {
	_ = server.RemoveNTPServer()
}

func RemoveDataDir() {
	if err := os.RemoveAll(path.Join(global.Workspace, "mount")); err != nil {
		log.Warnf("Failed to remove mount directory: %v", err)
	}
}

func removeAllInOne() {
	cleanKubeletBin()
	cleanKubernetesContainer()
	cleanOtherContainer()
	killResidualKubeProcesses()
	cleanContainerRuntime()
	cleanNetwork()
	cleanBkeAgent()
	cleanNeedDeleteFile()
}
