package ha

import (
	"strings"
	"time"

	"github.com/pkg/errors"
	bkenet "gopkg.openfuyao.cn/bkecommon/utils/net"
	"k8s.io/apimachinery/pkg/util/wait"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	envPlugin "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/env"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/mfutil"
)

const (
	Name = "HA"

	defaultHAproxyImageTag  = "2.1.4"
	defaultHAproxyImageName = "haproxy"

	defaultKeepAlivedImageName = "keepalived"
	defaultKeepAlivedImageTag  = "1.3.5"
)

type HA struct {
	exec       exec.Executor
	isMasterHa bool
}

type Endpoint struct {
	host string
	port int32
}

func (h *HA) Name() string {
	return Name
}

func New(exec exec.Executor) plugin.Plugin {
	return &HA{
		exec: exec,
	}
}

func (h *HA) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		// haproxy config command
		"haproxyConfigDir": {
			Key:         "haproxyConfigDir",
			Value:       "",
			Required:    false,
			Default:     mfutil.HAProxyConfPath,
			Description: "haproxy config dir",
		},
		"haproxyImageName": {
			Key:      "haproxyImageName",
			Value:    "haproxy",
			Required: false,
			Default:  defaultHAproxyImageName,
		},
		"haproxyImageTag": {
			Key:      "haproxyImageTag",
			Value:    "latest",
			Required: false,
			Default:  defaultHAproxyImageTag,
		},
		// keepalived config command
		"keepAlivedConfigDir": {
			Key:         "keepAlivedConfigDir",
			Value:       "",
			Required:    false,
			Default:     mfutil.KeepAlivedConfPath,
			Description: "keepalived config dir",
		},
		"keepAlivedImageName": {
			Key:         "keepalivedImageName",
			Value:       "keepalived",
			Required:    false,
			Default:     defaultKeepAlivedImageName,
			Description: "keepalived image name",
		},
		"keepAlivedImageTag": {
			Key:         "keepalivedImageTag",
			Value:       "latest",
			Required:    false,
			Default:     defaultKeepAlivedImageTag,
			Description: "keepalived image tag",
		},
		"haNodes": {
			Key:         "haNodes",
			Value:       "hostName:IP,hostName:IP",
			Required:    true,
			Default:     "",
			Description: "master nodes",
		},
		"ingressVIP": {
			Key:         "ingressVIP",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "ingress vip",
		},
		"controlPlaneEndpointVIP": {
			Key:         "controlPlaneEndpointVIP",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "control plane endpoint vip",
		},
		"controlPlaneEndpointPort": {
			Key:         "controlPlaneEndpointPort",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "control plane endpoint port",
		},
		"imageRepo": {
			Key:         "imageRepo",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "ha components image repo",
		},
		"manifestsDir": {
			Key:         "manifestsDir",
			Value:       "",
			Required:    false,
			Default:     mfutil.GetDefaultManifestsPath(),
			Description: "manifests dir",
		},
		"virtualRouterId": {
			Key:         "virtualRouterId",
			Value:       "51",
			Required:    false,
			Default:     "51",
			Description: "vrrp route id",
		},
		"wait": {
			Key:         "wait",
			Value:       "false",
			Required:    false,
			Default:     "false",
			Description: "wait for vip to be ready",
		},
	}
}

func (h *HA) Execute(commands []string) ([]string, error) {
	parseCommands, err := plugin.ParseCommands(h, commands)
	if err != nil {
		return nil, err
	}

	// sure load ip_vs
	h.initIPVS()

	// prepareRendCfg
	cfg, err := h.prepareRendCfg(parseCommands)
	if err != nil {
		return nil, err
	}

	var haComponents mfutil.HAComponents
	if h.isMasterHa {
		haComponents = mfutil.GetHAComponentList()
	} else {
		haComponents = mfutil.GetIngressHaComponentList()
	}

	haComponents.SetMfPath(parseCommands["manifestsDir"])

	// render and write to disk
	if err := mfutil.GenerateHAManifestYaml(haComponents, cfg); err != nil {
		return nil, err
	}

	if cfg["wait"] == "true" {
		if err := h.Wait(cfg); err != nil {
			return nil, err
		}
	}

	return nil, nil
}

func (h *HA) prepareRendCfg(commands map[string]string) (map[string]interface{}, error) {
	cfg := make(map[string]interface{})
	for k, v := range commands {
		cfg[k] = v
	}

	var nodes []mfutil.HANode
	if v, ok := commands["haNodes"]; ok && v != "" {
		ns := strings.Split(v, ",")
		for _, n := range ns {
			info := strings.Split(n, ":")
			if len(info) != 2 {
				return nil, errors.Errorf("haNodes format error")
			}
			nodes = append(nodes, mfutil.HANode{
				Hostname: info[0],
				IP:       info[1],
			})
		}
	}

	var vipInter string
	for _, node := range nodes {
		// get node interface
		if node.Hostname == utils.HostName() {
			inter, err := bkenet.GetInterfaceFromIp(node.IP)
			if err != nil {
				return nil, errors.Errorf("get interface from ip %s error", node.IP)
			}
			if inter == "" {
				return nil, errors.Errorf("can not find interface from ip %s", node.IP)
			}
			log.Infof("VIP Will be built on network card %s", inter)
			vipInter = inter
		}
	}
	if vipInter == "" {
		return nil, errors.Errorf("can not find local IP associated network card")
	}

	cfg["nodes"] = nodes
	cfg["interface"] = vipInter
	cfg["keepalivedAdvertInt"] = "1"
	cfg["keepalivedAuthPass"] = "22222222"
	//cfg["virtualRouterId"] = "51"

	log.Debug("HA config: ")
	if v, ok := cfg["controlPlaneEndpointVIP"]; ok && v != "" {
		h.isMasterHa = true
		cfg["isMasterHa"] = h.isMasterHa
		for _, node := range nodes {
			log.Debugf("add master node %q to ha config", node.IP)
		}
		log.Debugf("controlPlaneEndpointVIP is %v", v)
		log.Debugf("controlPlaneEndpointPort is %v", cfg["controlPlaneEndpointPort"])
		log.Debugf("network interface: %v", cfg["interface"])
		log.Debugf("imageRepo: %v", cfg["imageRepo"])
		cfg["vip"] = v
		return cfg, nil
	}

	if v, ok := cfg["ingressVIP"]; ok && v != "" {
		h.isMasterHa = false
		cfg["isMasterHa"] = h.isMasterHa
		for _, node := range nodes {
			log.Debugf("add ingress node %q to keepalived config", node.IP)
		}
		log.Debugf("ingress VIP is %v", v)
		cfg["vip"] = v
		return cfg, nil
	}

	return cfg, nil
}

// initIPVS load ip_vs and ip_vs_wrr mod
func (h *HA) initIPVS() {
	envCommand := []string{
		"K8sEnvInit",
		"init=true",
		"check=false",
		"scope=kernel",
	}
	_, _ = envPlugin.New(h.exec, nil).Execute(envCommand)
}

func (h *HA) Wait(cfg map[string]interface{}) error {
	if !mfutil.KeepalivedInstanceIsMaster(cfg["nodes"].([]mfutil.HANode)) {
		log.Infof("this node is not master, skip wait vip")
		return nil
	}

	waitIPs := []string{}
	if h.isMasterHa {
		waitIPs = append(waitIPs, cfg["controlPlaneEndpointVIP"].(string))
	} else {
		waitIPs = append(waitIPs, cfg["ingressVIP"].(string))
	}
	log.Infof("wait vip(s) %q ready", strings.Join(waitIPs, ","))

	err := wait.Poll(10*time.Second, 5*time.Minute, func() (bool, error) {
		for _, ip := range waitIPs {
			if ip == "" {
				continue
			}
			ok, err := bkenet.GetInterfaceFromIp(ip)
			if err != nil || ok == "" {
				log.Warnf("VIP %s is not ready on this node", ip)
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		return errors.Errorf("wait vip(s) %q ready failed, err: %v", strings.Join(waitIPs, ","), err)
	}
	log.Infof("vip(s) %q ready now", strings.Join(waitIPs, ","))
	return nil
}
