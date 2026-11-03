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

package manifests

import (
	"context"
	"crypto/sha256"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	Name = "Manifests"
)

type ManifestPlugin struct {
	bootScope *mfutil.BootScope
	exec      exec.Executor
}

func New(cfg *mfutil.BootScope, exec exec.Executor) plugin.Plugin {
	return &ManifestPlugin{
		bootScope: cfg,
		exec:      exec,
	}
}

func (mp *ManifestPlugin) Name() string {
	return Name
}

func (mp *ManifestPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"bkeConfig": {
			Key:         "bkeConfig",
			Value:       "NameSpace:Name",
			Required:    false,
			Default:     "",
			Description: "bkeconfig ConfigMap  ns/name",
		},
		"scope": {
			Key:         "scope",
			Value:       "kube-apiserver,kube-controller-manager,kube-scheduler,etcd",
			Required:    false,
			Default:     strings.Join([]string{mfutil.KubeAPIServer, mfutil.KubeControllerManager, mfutil.KubeScheduler, mfutil.Etcd}, ","),
			Description: "The scope of the manifests which to render, split by ',' eg: kube-apiserver,kube-controller-manager,kube-scheduler,etcd ",
		},
		"check": {
			Key:         "check",
			Value:       "",
			Required:    false,
			Default:     "false",
			Description: "Check the manifests static pod is running",
		},
		"gpuEnable": {
			Key:         "gpuEnable",
			Value:       "",
			Required:    false,
			Default:     "false",
			Description: "Enable gpu manager config for scheduler",
		},
		"manifestDir": {
			Key:         "manifestDir",
			Value:       "",
			Required:    false,
			Default:     mfutil.GetDefaultManifestsPath(),
			Description: "The path of the manifests",
		},
		"etcdDataDir": {
			Key:         "etcdDataDir",
			Value:       "",
			Required:    false,
			Default:     mfutil.EtcdDataDir,
			Description: "The path of the etcd data dir",
		},
	}
}

// setupEtcdEnvironment sets up etcd data directory and user
func (mp *ManifestPlugin) setupEtcdEnvironment(etcdDataDir string) error {
	// create etcd data dir if not exist
	if !utils.Exists(etcdDataDir) {
		mkdirCmd := fmt.Sprintf("mkdir -p -m 700 %s", etcdDataDir)
		if err := mp.exec.ExecuteCommand("/usr/bin/sh", "-c", mkdirCmd); err != nil {
			return err
		}
	}
	// create etcd user if not exist
	out, err := mp.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", "id etcd")
	if err != nil {
		log.Warnf("get etcd user failed: %s, err: %v", out, err)
		log.Infof("create user etcd")
		createEtcdUserCmd := fmt.Sprintf(`useradd -r -c "etcd user" -s /sbin/nologin etcd -d %s`, etcdDataDir)
		out, err = mp.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", createEtcdUserCmd)
		if err != nil {
			return fmt.Errorf("create etcd user failed: %s, err: %w", out, err)
		}
	}
	// change etcd data dir owner
	ownerCmd := fmt.Sprintf("chown -R etcd:etcd %s", etcdDataDir)
	out, err = mp.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", ownerCmd)
	if err != nil {
		return fmt.Errorf("change etcd data dir owner failed: %s, err: %w", out, err)
	}
	return nil
}

func (mp *ManifestPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	mpc := *mp
	mpc.exec = exec.Bound(mp.exec, ctx)
	mp = &mpc

	parseCommands, err := plugin.ParseCommands(mp, commands)
	if err != nil {
		return nil, err
	}

	// 这仅出现在单独调用manifests插件时，即不是在boot时调用
	if mp.bootScope == nil {
		log.Info("new boot scope")
		if err = mp.newBootScope(parseCommands); err != nil {
			return nil, err
		}
	}

	scope := strings.Split(parseCommands["scope"], ",")
	components := mfutil.Components{}
	for _, component := range mfutil.GetDefaultComponentList() {
		if utils.ContainsString(scope, component.Name) {
			components = append(components, component)
		}
		if component.Name == mfutil.Etcd {
			if err := mp.setupEtcdEnvironment(parseCommands["etcdDataDir"]); err != nil {
				return nil, err
			}
		}
	}
	components.SetMfPath(parseCommands["manifestDir"])
	before := mp.manifestFileHashes(components)
	if err := mfutil.GenerateManifestYaml(components, mp.bootScope); err != nil {
		return nil, err
	}
	after := mp.manifestFileHashes(components)

	// 仅在 manifest 内容变化或 kubelet 未运行时 restart kubelet：
	// 配置更新失败重试期间 manifest 内容不变，重复 restart 会反复打断 kubelet
	// 恢复静态 Pod，导致 etcd 等组件长时间无法拉起（重试风暴）。
	if !maps.Equal(before, after) || !mp.kubeletActive() {
		// join control plane restart kubelet for start static pod
		cmd := fmt.Sprintf("if [ -f %s ]; then systemctl restart kubelet; fi", utils.GetKubeletServicePath())
		if out, err := mp.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", cmd); err != nil {
			log.Warnf("restart kubelet failed: %s, err: %v", out, err)
		} else if !maps.Equal(before, after) {
			log.Infof("manifests changed, kubelet restarted")
		} else {
			log.Infof("manifests unchanged, kubelet restarted for inactive kubelet")
		}
	} else {
		log.Infof("manifests unchanged and kubelet active, skip restart kubelet")
	}
	return nil, nil
}

// manifestFileHashes 计算各组件 static pod manifest 文件内容的 sha256（文件不存在则跳过），
// 用于渲染前后对比判断 manifest 是否实际发生变化。
func (mp *ManifestPlugin) manifestFileHashes(components mfutil.Components) map[string]string {
	hashes := make(map[string]string, len(components))
	for _, component := range components {
		filePath := filepath.Join(component.MfPath, component.Name+".yaml")
		data, err := os.ReadFile(filePath)
		if err != nil {
			continue
		}
		hashes[filePath] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	return hashes
}

// kubeletActive 检查 kubelet 服务是否处于运行状态。
func (mp *ManifestPlugin) kubeletActive() bool {
	return mp.exec.ExecuteCommand("/usr/bin/sh", "-c", "systemctl is-active --quiet kubelet") == nil
}

func (mp *ManifestPlugin) newBootScope(parseCommands map[string]string) error {
	bkeConfigNS := ""
	if v, ok := parseCommands["bkeConfig"]; ok {
		bkeConfigNS = v
	} else {
		return errors.Errorf("bkeConfig param is required,when use manifest plugin alone")
	}

	enableGPU := "false"
	if v, ok := parseCommands["gpuEnable"]; ok {
		enableGPU = v
	}

	bkeCluster, err := plugin.GetBKECluster(bkeConfigNS)
	if err != nil {
		return err
	}
	config, err := plugin.GetBkeConfigFromBkeCluster(bkeCluster)
	if err != nil {
		return err
	}

	clusterData, err := plugin.GetClusterData(bkeConfigNS)
	if err != nil {
		return err
	}

	nodes := bkenode.Nodes(clusterData.Nodes)
	currentNode, err := nodes.CurrentNode()
	if err != nil {
		return errors.Wrapf(err, "failed to get current node")
	}

	// 构建一个bootScope
	mp.bootScope = &mfutil.BootScope{
		BkeConfig:        config,
		ClusterName:      bkeCluster.GetName(),
		ClusterNamespace: bkeCluster.GetNamespace(),
		HostName:         utils.HostName(),
		HostIP:           currentNode.IP,
		CurrentNode:      currentNode,
		Extra: map[string]interface{}{
			"Init":          false,
			"KubernetesDir": pkiutil.KubernetesDir,
			"mccs":          []string{bkeCluster.GetNamespace(), bkeCluster.GetName()},
			"gpuEnable":     enableGPU,
		},
	}
	mp.bootScope.Extra["upgradeWithOpenFuyao"] = mp.bootScope.HasOpenFuyaoAddon()
	return nil
}
