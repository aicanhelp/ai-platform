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

package ha

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/wait"

	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// KernelEnvRunner loads kernel modules (IPVS) via the env builtin plugin.
// Wired from pkg/job/builtin to avoid ha importing the heavy kubeadm/env package
// (which inflates ha.test under -coverpkg=./... and can OOM the linker in CI).
// The executor is captured when wired; callers only pass ctx and commands.
var KernelEnvRunner func(ctx context.Context, commands []string) error

const (
	Name                    = "HA"
	two                     = 2
	ten                     = 10
	five                    = 5
	defaultHAproxyImageTag  = "2.1.4"
	defaultHAproxyImageName = "haproxy"

	defaultKeepAlivedImageName = "keepalived/keepalived"
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
		"haproxyConfigDir":         {Key: "haproxyConfigDir", Value: "", Required: false, Default: mfutil.HAProxyConfPath, Description: "haproxy config dir"},
		"haproxyImageName":         {Key: "haproxyImageName", Value: "haproxy", Required: false, Default: defaultHAproxyImageName},
		"haproxyImageTag":          {Key: "haproxyImageTag", Value: "latest", Required: false, Default: defaultHAproxyImageTag},
		"keepAlivedConfigDir":      {Key: "keepAlivedConfigDir", Value: "", Required: false, Default: mfutil.KeepAlivedConfPath, Description: "keepalived config dir"},
		"keepAlivedImageName":      {Key: "keepalivedImageName", Value: "keepalived", Required: false, Default: defaultKeepAlivedImageName, Description: "keepalived image name"},
		"keepAlivedImageTag":       {Key: "keepalivedImageTag", Value: "latest", Required: false, Default: defaultKeepAlivedImageTag, Description: "keepalived image tag"},
		"haNodes":                  {Key: "haNodes", Value: "hostName:IP,hostName:IP", Required: true, Default: "", Description: "master nodes"},
		"ingressVIP":               {Key: "ingressVIP", Value: "", Required: false, Default: "", Description: "ingress vip"},
		"controlPlaneEndpointVIP":  {Key: "controlPlaneEndpointVIP", Value: "", Required: false, Default: "", Description: "control plane endpoint vip"},
		"controlPlaneEndpointPort": {Key: "controlPlaneEndpointPort", Value: "", Required: false, Default: "", Description: "control plane endpoint port"},
		"thirdImageRepo":           {Key: "thirdImageRepo", Value: "", Required: true, Default: "", Description: "ha proxy image repo"},
		"fuyaoImageRepo":           {Key: "fuyaoImageRepo", Value: "", Required: true, Default: "", Description: "ha keepalived image repo"},
		"manifestsDir":             {Key: "manifestsDir", Value: "", Required: false, Default: mfutil.GetDefaultManifestsPath(), Description: "manifests dir"},
		"virtualRouterId":          {Key: "virtualRouterId", Value: "51", Required: false, Default: "51", Description: "vrrp route id"},
		"wait":                     {Key: "wait", Value: "false", Required: false, Default: "false", Description: "wait for vip to be ready"},
	}
}

func (h *HA) Execute(ctx context.Context, commands []string) ([]string, error) {
	hc := *h
	hc.exec = exec.Bound(h.exec, ctx)
	h = &hc

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
		if err := h.Wait(ctx, cfg); err != nil {
			return nil, err
		}
	}

	return nil, nil
}

// parseHANodes parses haNodes string into HANode slice
func (h *HA) parseHANodes(haNodesStr string) ([]mfutil.HANode, error) {
	if haNodesStr == "" {
		return nil, nil
	}
	var nodes []mfutil.HANode
	ns := strings.Split(haNodesStr, ",")
	for _, n := range ns {
		info := strings.Split(n, ":")
		if len(info) != two {
			return nil, errors.Errorf("haNodes format error")
		}
		nodes = append(nodes, mfutil.HANode{
			Hostname: info[0],
			IP:       info[1],
		})
	}
	return nodes, nil
}

// findVIPInterface finds the network interface for VIP from nodes
func (h *HA) findVIPInterface(nodes []mfutil.HANode) (string, error) {
	for _, node := range nodes {
		if node.Hostname == utils.HostName() {
			inter, err := bkenet.GetInterfaceFromIp(node.IP)
			if err != nil {
				return "", errors.Errorf("get interface from ip %s error", node.IP)
			}
			if inter == "" {
				return "", errors.Errorf("can not find interface from ip %s", node.IP)
			}
			log.Infof("VIP Will be built on network card %s", inter)
			return inter, nil
		}
	}
	return "", errors.Errorf("can not find local IP associated network card")
}

func (h *HA) prepareRendCfg(commands map[string]string) (map[string]interface{}, error) {
	cfg := make(map[string]interface{})
	for k, v := range commands {
		cfg[k] = v
	}

	haNodesStr := commands["haNodes"]
	nodes, err := h.parseHANodes(haNodesStr)
	if err != nil {
		return nil, err
	}

	vipInter, err := h.findVIPInterface(nodes)
	if err != nil {
		return nil, err
	}

	cfg["nodes"] = nodes
	cfg["interface"] = vipInter
	cfg["keepalivedAdvertInt"] = "1"
	cfg["keepalivedAuthPass"] = "22222222"

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
	if KernelEnvRunner == nil {
		log.Warnf("HA KernelEnvRunner unset, skip IPVS kernel init")
		return
	}
	envCommand := []string{
		"K8sEnvInit",
		"init=true",
		"check=false",
		"scope=kernel",
	}
	if err := KernelEnvRunner(context.Background(), envCommand); err != nil {
		log.Warnf("init IPVS via env plugin failed: %v", err)
	}
}

func (h *HA) Wait(ctx context.Context, cfg map[string]interface{}) error {
	if !mfutil.KeepalivedInstanceIsMaster(cfg["nodes"].([]mfutil.HANode)) {
		log.Infof("this node is not master, skip wait vip")
		return nil
	}

	var waitIPs []string
	if h.isMasterHa {
		waitIPs = append(waitIPs, cfg["controlPlaneEndpointVIP"].(string))
	} else {
		waitIPs = append(waitIPs, cfg["ingressVIP"].(string))
	}
	log.Infof("wait vip(s) %q ready", strings.Join(waitIPs, ","))

	if ctx == nil {
		ctx = context.Background()
	}
	err := wait.PollImmediateWithContext(ctx, ten*time.Second, five*time.Minute, func(ctx context.Context) (bool, error) {
		if err := ctx.Err(); err != nil {
			return false, err
		}
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
		return fmt.Errorf("wait vip(s) %q ready failed, err: %w", strings.Join(waitIPs, ","), err)
	}
	log.Infof("vip(s) %q ready now", strings.Join(waitIPs, ","))
	return nil
}
