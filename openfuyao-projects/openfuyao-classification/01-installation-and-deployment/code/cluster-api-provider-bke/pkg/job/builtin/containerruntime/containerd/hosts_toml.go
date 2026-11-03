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

package containerd

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"text/template"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	HostsTomlDir = "/etc/containerd/certs.d"
)

var (
	//go:embed hosts.toml.tmpl
	hostsToml string
)

type TemplateData struct {
	Server string
	Hosts  []HostEntry
}

type HostEntry struct {
	URL    string
	Config HostConfig
}

type HostConfig struct {
	Capabilities []string
	SkipVerify   bool
	PlainHTTP    bool
	Insecure     bool
	OverridePath bool
	CA           string
	ClientCert   string
	ClientKey    string
	Headers      map[string][]string
}

type HostsTOMLGenerator struct {
	ConfigPath string
}

func NewHostsTOMLGenerator(configPath string) *HostsTOMLGenerator {
	if configPath == "" {
		configPath = HostsTomlDir
	}
	return &HostsTOMLGenerator{
		ConfigPath: configPath,
	}
}

func (g *HostsTOMLGenerator) GenerateHostsTOML(registryName string, config *bkev1beta1.RegistryHostConfig) error {
	if config == nil {
		return fmt.Errorf("registry config is nil")
	}

	registryDir := filepath.Join(g.ConfigPath, registryName)
	if err := os.MkdirAll(registryDir, utils.RwxRxRx); err != nil {
		return fmt.Errorf("failed to create registry directory: %w", err)
	}

	tomlContent, err := g.generateTOMLContent(registryName, config)
	if err != nil {
		return fmt.Errorf("failed to generate TOML content: %w", err)
	}

	hostsTOMLPath := filepath.Join(registryDir, "hosts.toml")
	if err = os.WriteFile(hostsTOMLPath, []byte(tomlContent), utils.RwRR); err != nil {
		return fmt.Errorf("failed to write hosts.toml: %w", err)
	}

	return nil
}

func (g *HostsTOMLGenerator) GenerateMultipleHostsTOML(registryConfigs map[string]bkev1beta1.RegistryHostConfig) error {
	if registryConfigs == nil {
		log.Debugf("registry config is nil, skip generate containerd hosts.toml")
		return nil
	}

	var errors []string

	for registryName, config := range registryConfigs {
		if err := g.GenerateHostsTOML(registryName, &config); err != nil {
			errors = append(errors, fmt.Sprintf("failed to generate hosts.toml for %s: %v", registryName, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("multiple errors occurred: %v", errors)
	}

	return nil
}

func (g *HostsTOMLGenerator) generateTOMLContent(registryName string, config *bkev1beta1.RegistryHostConfig) (string, error) {
	tmplData := g.prepareTemplateData(registryName, config)

	tpl, err := template.New("hosts.toml").Parse(hostsToml)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	if err = tpl.Execute(&buf, tmplData); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

func (g *HostsTOMLGenerator) prepareTemplateData(registryName string, config *bkev1beta1.RegistryHostConfig) *TemplateData {
	data := &TemplateData{
		Server: fmt.Sprintf("https://%s", registryName),
	}

	hostConfig := HostConfig{
		Capabilities: config.Capabilities,
		SkipVerify:   config.SkipVerify,
		PlainHTTP:    config.PlainHTTP,
		Insecure:     config.Insecure,
		OverridePath: config.OverridePath,
	}

	if config.TLS != nil {
		hostConfig.CA = config.TLS.CAFile
		hostConfig.ClientCert = config.TLS.CertFile
		hostConfig.ClientKey = config.TLS.KeyFile
		if config.TLS.InsecureSkipVerify {
			hostConfig.SkipVerify = true
		}
	}

	if config.Auth != nil {
		hostConfig.Headers = g.prepareAuthHeaders(config.Auth)
	}

	if config.Header != nil {
		if hostConfig.Headers == nil {
			hostConfig.Headers = make(map[string][]string)
		}
		for k, v := range config.Header {
			hostConfig.Headers[k] = v
		}
	}

	data.Hosts = append(data.Hosts, HostEntry{
		URL:    fmt.Sprintf("https://%s", config.Host),
		Config: hostConfig,
	})

	return data
}

func (g *HostsTOMLGenerator) prepareAuthHeaders(auth *bkev1beta1.RegistryAuthConfig) map[string][]string {
	headers := make(map[string][]string)
	if auth == nil {
		return headers
	}

	if auth.Auth != "" {
		headers["authorization"] = []string{auth.Auth}
		return headers
	}
	if auth.Username != "" && auth.Password != "" {
		headers["authorization"] = []string{fmt.Sprintf("Basic %s", fmt.Sprintf("%s:%s", auth.Username, auth.Password))}
		return headers
	}
	if auth.IdentityToken != "" {
		headers["authorization"] = []string{fmt.Sprintf("Bearer %s", auth.IdentityToken)}
		return headers
	}
	if auth.RegistryToken != "" {
		headers["authorization"] = []string{fmt.Sprintf("Bearer %s", auth.RegistryToken)}
	}

	return headers
}
