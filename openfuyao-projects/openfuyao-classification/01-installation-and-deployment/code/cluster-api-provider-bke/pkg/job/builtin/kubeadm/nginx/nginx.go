/******************************************************************
 * Copyright (c) 2026 China Mobile (Suzhou) Software Technology Co., Ltd
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package nginx implements the nginx-proxy load balancer plugin: it renders
// nginx.conf and the nginx-proxy static pod manifest on each worker node so
// the kubelet and kubeconfig traffic reach kube-apiserver via 127.0.0.1.
package nginx

import (
	"context"
	"embed"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/template"

	"github.com/pkg/errors"

	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// Name is the plugin name registered with the Kubeadm plugin framework.
	Name                  = "Nginx"
	nginxConfPath         = "/etc/kubernetes/nginx-proxy"
	nginxConfFileName     = "nginx.conf"
	nginxManifestFileName = "nginx-proxy.yaml"

	defaultAPIServerPort  = 6443
	defaultNginxImageName = "nginx"
	// defaultNginxImageTag uses the maintained stable version
	defaultNginxImageTag = "1.28.0"

	// minPort and maxPort bound the kube-apiserver port accepted by parsePort.
	minPort = 1
	maxPort = 65535
)

// templates holds the embedded nginx.conf and static pod manifest templates.
//
//go:embed tmpl/*
var templates embed.FS

// Plugin renders nginx.conf and the nginx-proxy static pod manifest on a
// worker node. It implements the plugin.Plugin interface.
type Plugin struct {
	exec exec.Executor
	boot *mfutil.BootScope
}

// New returns a new nginx plugin instance.
func New(exec exec.Executor, boot *mfutil.BootScope) *Plugin {
	return &Plugin{
		exec: exec,
		boot: boot,
	}
}

// Name returns the plugin name.
func (n *Plugin) Name() string {
	return Name
}

// Param declares the plugin parameters and their defaults.
func (n *Plugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"masterNodes": {
			Key:         "masterNodes",
			Value:       "192.168.1.1,192.168.1.2,192.168.1.3",
			Required:    true,
			Description: "comma separated master node IPs",
		},
		"apiServerPort": {
			Key:         "apiServerPort",
			Value:       "6443",
			Required:    false,
			Default:     "6443",
			Description: "kube-apiserver port",
		},
		"manifestsDir": {
			Key:         "manifestsDir",
			Value:       nginxConfPath,
			Required:    false,
			Default:     nginxConfPath,
			Description: "nginx config dir",
		},
		"manifestPath": {
			Key:         "manifestPath",
			Value:       mfutil.GetDefaultManifestsPath(),
			Required:    false,
			Default:     mfutil.GetDefaultManifestsPath(),
			Description: "static pod manifest dir",
		},
		"nginxImageName": {
			Key:         "nginxImageName",
			Value:       defaultNginxImageName,
			Required:    false,
			Default:     defaultNginxImageName,
			Description: "nginx image name",
		},
		"nginxImageTag": {
			Key:         "nginxImageTag",
			Value:       defaultNginxImageTag,
			Required:    false,
			Default:     defaultNginxImageTag,
			Description: "nginx image tag",
		},
	}
}

// Execute handles the plugin command: it creates the nginx config directory,
// renders nginx.conf and generates the static pod manifest.
func (n *Plugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	parsed, err := plugin.ParseCommands(n, commands)
	if err != nil {
		return nil, err
	}
	params, err := n.buildParams(parsed)
	if err != nil {
		return nil, err
	}

	log.Infof("Start to configure nginx load balancer")

	// 1. Create nginx config directory
	if err := n.ensureConfigDirectory(params.ConfPath); err != nil {
		return nil, err
	}

	// 2. Generate nginx config file
	if err := n.generateNginxConf(params); err != nil {
		return nil, err
	}

	// 3. Generate static pod manifest
	if err := n.generateNginxManifest(params); err != nil {
		return nil, err
	}

	log.Infof("Nginx load balancer configured successfully")
	return nil, nil
}

// Params carries the parsed plugin parameters for rendering nginx.conf and
// the static pod manifest.
type Params struct {
	MasterNodes   []string
	APIServerPort int
	ConfPath      string
	ManifestPath  string
	ImageName     string
	ImageTag      string
}

func (n *Plugin) buildParams(parsed map[string]string) (*Params, error) {
	masterNodes, err := parseMasterNodes(parsed["masterNodes"])
	if err != nil {
		return nil, err
	}
	port, err := parsePort(parsed["apiServerPort"])
	if err != nil {
		return nil, err
	}
	return &Params{
		MasterNodes:   masterNodes,
		APIServerPort: port,
		ConfPath:      parsed["manifestsDir"],
		ManifestPath:  parsed["manifestPath"],
		ImageName:     parsed["nginxImageName"],
		ImageTag:      parsed["nginxImageTag"],
	}, nil
}

func parseMasterNodes(masterNodesStr string) ([]string, error) {
	var nodes []string
	ns := strings.Split(masterNodesStr, ",")
	for _, n := range ns {
		ip := strings.TrimSpace(n)
		if ip == "" {
			continue
		}
		nodes = append(nodes, ip)
	}
	if len(nodes) == 0 {
		return nil, errors.New("masterNodes format error, expected comma separated IPs")
	}
	return nodes, nil
}

func parsePort(portStr string) (int, error) {
	port := strings.TrimSpace(portStr)
	if port == "" {
		return defaultAPIServerPort, nil
	}
	p, err := strconv.Atoi(port)
	if err != nil {
		return 0, errors.Errorf("invalid port format: %s", port)
	}
	if p < minPort || p > maxPort {
		return 0, errors.Errorf("port must be between %d and %d, got: %d", minPort, maxPort, p)
	}
	return p, nil
}

func (n *Plugin) ensureConfigDirectory(confPath string) error {
	if err := os.MkdirAll(confPath, utils.RwxRxRx); err != nil {
		return errors.Errorf("failed to create nginx config directory: %v", err)
	}
	log.Infof("nginx config directory: %s", confPath)
	return nil
}

func (n *Plugin) generateNginxConf(params *Params) error {
	tmplBytes, err := templates.ReadFile("tmpl/nginx.conf.tmpl")
	if err != nil {
		return errors.Wrapf(err, "failed to read nginx.conf template")
	}

	data := struct {
		MasterNodes   []string
		APIServerPort int
	}{
		MasterNodes:   params.MasterNodes,
		APIServerPort: params.APIServerPort,
	}

	confContent, err := renderTemplate("nginx.conf", string(tmplBytes), data)
	if err != nil {
		return errors.Wrapf(err, "failed to render nginx.conf")
	}

	confPath := filepath.Join(params.ConfPath, nginxConfFileName)
	if err := os.WriteFile(confPath, []byte(confContent), utils.RwRR); err != nil {
		return errors.Wrapf(err, "failed to write nginx.conf")
	}

	log.Infof("nginx.conf generated: %s", confPath)
	return nil
}

func (n *Plugin) generateNginxManifest(params *Params) error {
	if n.boot == nil || n.boot.BkeConfig == nil {
		return errors.New("the BootScope is not initialized, use New(exec, boot) with a valid one")
	}

	if err := os.MkdirAll(params.ManifestPath, utils.RwxRxRx); err != nil {
		return errors.Wrapf(err, "failed to create manifest directory")
	}

	tmplBytes, err := templates.ReadFile("tmpl/nginx.yaml.tmpl")
	if err != nil {
		return errors.Wrapf(err, "failed to read nginx.yaml template")
	}

	// Create a copy of BootScope to avoid modifying the original data
	data := &mfutil.BootScope{
		BkeConfig:   n.boot.BkeConfig,
		ClusterName: n.boot.ClusterName,
		HostName:    n.boot.HostName,
		HostIP:      n.boot.HostIP,
		CurrentNode: n.boot.CurrentNode,
		Extra:       make(map[string]interface{}),
	}

	// Copy original Extra data
	for k, v := range n.boot.Extra {
		data.Extra[k] = v
	}

	data.Extra["ManifestsDir"] = params.ConfPath
	data.Extra["NginxImageName"] = params.ImageName
	data.Extra["NginxImageTag"] = params.ImageTag
	data.Extra["APIServerPort"] = params.APIServerPort
	// nginx is a third-party image, image repo prefix uses the project third-repo mechanism
	cfg := bkeinit.BkeConfig(*n.boot.BkeConfig)
	data.Extra["ImageRepo"] = cfg.ImageThirdRepo()

	manifestContent, err := renderTemplateWithFuncs("nginx-proxy", string(tmplBytes), data)
	if err != nil {
		return errors.Wrapf(err, "failed to render nginx manifest")
	}

	manifestPath := filepath.Join(params.ManifestPath, nginxManifestFileName)
	if err := os.WriteFile(manifestPath, []byte(manifestContent), utils.RwRR); err != nil {
		return errors.Wrapf(err, "failed to write nginx manifest")
	}

	log.Infof("nginx static pod manifest generated: %s", manifestPath)
	return nil
}

func renderTemplate(name, content string, data interface{}) (string, error) {
	tmpl, err := template.New(name).Parse(content)
	if err != nil {
		return "", errors.Wrapf(err, "failed to parse template")
	}

	var buf strings.Builder
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", errors.Wrapf(err, "failed to execute template")
	}

	return buf.String(), nil
}

// renderTemplateWithFuncs renders template with global functions
func renderTemplateWithFuncs(name, content string, data interface{}) (string, error) {
	funcMap := mfutil.GlobalFuncMap()

	tmpl, err := template.New(name).Funcs(*funcMap).Parse(content)
	if err != nil {
		return "", errors.Wrapf(err, "failed to parse template")
	}

	var buf strings.Builder
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", errors.Wrapf(err, "failed to execute template")
	}

	return buf.String(), nil
}
