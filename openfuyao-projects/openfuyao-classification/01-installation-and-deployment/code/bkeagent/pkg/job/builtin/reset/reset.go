package reset

import (
	"net"
	"path"
	"strings"

	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	bkenet "gopkg.openfuyao.cn/bkecommon/utils/net"

	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

const (
	Name = "Reset"
)

type ResetPlugin struct {
}

func New() plugin.Plugin {
	return &ResetPlugin{}
}

func (r ResetPlugin) Name() string {
	return Name
}

func (r ResetPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"bkeConfig": {
			Key:         "bkeConfig",
			Value:       "ns:name",
			Required:    false,
			Default:     "",
			Description: "bke config",
		},
		"scope": {
			Key:         "scope",
			Value:       "cert,manifests,containerd-cfg,container,kubelet,containerRuntime,source,extra",
			Required:    false,
			Default:     "cert,manifests,container,kubelet,containerRuntime,source,extra",
			Description: "reset scope(cert,)",
		},
		"extra": {
			Key:         "extra",
			Value:       "extra",
			Required:    false,
			Default:     "",
			Description: "extra file/dir/ipaddr to clean,split by ',' ",
		},
	}
}

func (r ResetPlugin) Execute(commands []string) ([]string, error) {
	commandsMap, err := plugin.ParseCommands(r, commands)
	if err != nil {
		return nil, err
	}

	var cfg *bkev1beta1.BKEConfig
	if v, ok := commandsMap["bkeConfig"]; ok && v != "" {
		cfg, err = plugin.GetBkeConfig(v)
		if err != nil {
			log.Errorf("get bke config failed: %v", err)
			return nil, err
		}
	}
	cfg.CustomExtra["allInOne"] = "false"

	nodes := bkenode.Nodes(cfg.Nodes)
	cnode, err := nodes.CurrentNode()
	if err != nil {
		log.Warnf("get current node failed: %v", err)
	}

	if v, ok := cfg.CustomExtra["host"]; ok && v != "" {
		if cnode.IP != "" && cnode.IP == v {
			cfg.CustomExtra["allInOne"] = "true"
		}
		inter, err := bkenet.GetInterfaceFromIp(v)
		if err == nil && inter != "" {
			cfg.CustomExtra["allInOne"] = "true"
		}
	}

	scopes := strings.Split(commandsMap["scope"], ",")

	for _, phase := range DefaultCleanPhases() {
		if utils.ContainsString(scopes, phase.Name) {
			if phase.Name == "extra" {
				args := strings.Split(commandsMap["extra"], ",")
				for _, arg := range args {
					if arg == "" {
						continue
					}
					if path.IsAbs(arg) {
						if utils.Exists(arg) {
							if utils.IsDir(arg) {
								phase.AddDirToClean(arg)
							} else {
								phase.AddFileToClean(arg)
							}
						}
						log.Warnf("extra file/dir %s not exists, skip remove", arg)
					}
					if ip := net.ParseIP(arg); ip != nil {
						phase.AddIPToClean(arg)
					} else {
						log.Warnf("not a valid ip addr, skip remove", arg)
					}
				}
			}

			if err = phase.Clean(cfg); err != nil {
				return nil, err
			}
		}
	}

	return nil, nil
}
