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
)

const (
	// DropInFileName is the generated containerd systemd override filename.
	DropInFileName = "10-override.conf"
	// ServiceDropInDir is the default containerd systemd override directory.
	ServiceDropInDir = "/etc/systemd/system/containerd.service.d"
)

//go:embed service-dropin.tmpl
var serviceDropInTemplate string

type ServiceDropInGenerator struct {
	ConfigPath string
}

func NewServiceDropInGenerator(configPath string) *ServiceDropInGenerator {
	if configPath == "" {
		configPath = ServiceDropInDir
	}
	return &ServiceDropInGenerator{
		ConfigPath: configPath,
	}
}

type ServiceDropInData struct {
	ServiceConfig *bkev1beta1.ServiceConfig
}

func (g *ServiceDropInGenerator) GenerateServiceDropIn(config *bkev1beta1.ServiceConfig) error {
	if config == nil {
		return fmt.Errorf("service config is nil")
	}

	if err := os.MkdirAll(g.ConfigPath, utils.RwxRxRx); err != nil {
		return fmt.Errorf("failed to create drop-in directory: %w", err)
	}

	configContent, err := g.generateDropInContent(config)
	if err != nil {
		return fmt.Errorf("failed to generate drop-in content: %w", err)
	}

	dropInPath := filepath.Join(g.ConfigPath, DropInFileName)
	if err = os.WriteFile(dropInPath, []byte(configContent), utils.RwRR); err != nil {
		return fmt.Errorf("failed to write drop-in file: %w", err)
	}

	return nil
}

func (g *ServiceDropInGenerator) generateDropInContent(config *bkev1beta1.ServiceConfig) (string, error) {
	tmplData := &ServiceDropInData{
		ServiceConfig: config,
	}

	tpl, err := template.New("service-dropin").Parse(serviceDropInTemplate)
	if err != nil {
		return "", fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err = tpl.Execute(&buf, tmplData); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

func (g *ServiceDropInGenerator) RemoveServiceDropIn() error {
	dropInPath := filepath.Join(g.ConfigPath, DropInFileName)

	if _, err := os.Stat(dropInPath); os.IsNotExist(err) {
		return nil
	}

	if err := os.Remove(dropInPath); err != nil {
		return fmt.Errorf("failed to remove drop-in file: %w", err)
	}

	return nil
}
