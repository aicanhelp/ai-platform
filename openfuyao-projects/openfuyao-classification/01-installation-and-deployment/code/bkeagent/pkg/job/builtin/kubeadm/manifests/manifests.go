package manifests

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/mfutil"
	"gopkg.openfuyao.cn/bkeagent/utils/pkiutil"
)

const (
	Name = "Manifests"
)

var (
	DefaultScope = strings.Join([]string{mfutil.KubeAPIServer, mfutil.KubeControllerManager, mfutil.KubeScheduler, mfutil.Etcd}, ",")
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
			Default:     DefaultScope,
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

func (mp *ManifestPlugin) Execute(commands []string) ([]string, error) {
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
			// create etcd data dir if not exist
			if !utils.Exists(parseCommands["etcdDataDir"]) {
				mkdirCmd := fmt.Sprintf("mkdir -p -m 700 %s", parseCommands["etcdDataDir"])
				if err := mp.exec.ExecuteCommand("/usr/bin/sh", "-c", mkdirCmd); err != nil {
					return nil, err
				}
			}
			// create etcd user if not exist
			out, err := mp.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", "id etcd")
			if err != nil {
				log.Warnf("get etcd user failed: %s, err: %v", out, err)

				log.Infof("create user etcd")
				createEtcdUserCmd := fmt.Sprintf("useradd -r -c \"etcd user\" -s /sbin/nologin etcd -d %s", parseCommands["etcdDataDir"])
				if out, err = mp.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", createEtcdUserCmd); err != nil {
					return nil, errors.Errorf("create etcd user failed: %s, err: %v", out, err)
				}
			}
			// change etcd data dir owner
			ownerCmd := fmt.Sprintf("chown -R etcd:etcd %s", parseCommands["etcdDataDir"])
			if out, err = mp.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", ownerCmd); err != nil {
				return nil, errors.Errorf("change etcd data dir owner failed: %s, err: %v", out, err)
			}
		}
	}
	components.SetMfPath(parseCommands["manifestDir"])
	if err := mfutil.GenerateManifestYaml(components, mp.bootScope); err != nil {
		return nil, err
	}

	// join control plane restart kubelet for start static pod
	cmd := fmt.Sprintf("if [ -f %s ]; then systemctl restart kubelet; fi", utils.GetKubeletServicePath())
	if out, err := mp.exec.ExecuteCommandWithCombinedOutput("/usr/bin/sh", "-c", cmd); err != nil {
		log.Warnf("restart kubelet failed: %s, err: %v", out, err)
	}
	return nil, nil
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

	nodes := bkenode.Nodes(config.Nodes)
	currentNode, err := nodes.CurrentNode()
	if err != nil {
		return errors.Wrapf(err, "failed to get current node")
	}

	// 构建一个bootScope
	mp.bootScope = &mfutil.BootScope{
		BkeConfig:   config,
		ClusterName: bkeCluster.GetName(),
		HostName:    utils.HostName(),
		HostIP:      currentNode.IP,
		CurrentNode: currentNode,
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
