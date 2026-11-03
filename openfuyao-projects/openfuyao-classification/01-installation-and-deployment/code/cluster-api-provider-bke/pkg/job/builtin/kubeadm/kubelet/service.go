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

package kubelet

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	ServiceDir      = "/etc/systemd/system"
	KubeletFileName = "kubelet.service"
)

var (
	//go:embed kubelet-service.tmpl
	kubeletservice string
)

// applyDefaultHostnameOverrideIfMissing appends --hostname-override to ExecStart when KCT (KubeletConfig)
// does not already specify it. The value comes from utils.HostName() (same as the non-KCT kubelet path).
func applyDefaultHostnameOverrideIfMissing(s *confv1beta1.KubeletService) {
	if s == nil {
		return
	}
	execStart := strings.TrimSpace(s.ExecStart)
	if execStart == "" {
		return
	}
	if strings.Contains(execStart, "--hostname-override") {
		return
	}
	name := utils.HostName()
	if name == "" {
		return
	}
	s.ExecStart = strings.TrimRight(execStart, " \t") + " --hostname-override=" + name
}

type ServiceData struct {
	KubeletService *confv1beta1.KubeletService
}

type ServiceGenerator struct {
	ConfigPath string
}

func NewServiceData(configPath string) *ServiceGenerator {
	if configPath == "" {
		configPath = ServiceDir
	}
	return &ServiceGenerator{
		ConfigPath: configPath,
	}
}

// GenerateService generate kubelet.service file
func (s *ServiceGenerator) GenerateService(config *confv1beta1.KubeletService, commandConfig map[string]string,
	exec exec.Executor) error {
	if config == nil {
		return fmt.Errorf("service config is nil")
	}

	if err := os.MkdirAll(s.ConfigPath, utils.RwxRxRx); err != nil {
		return fmt.Errorf("failed to create drop-in directory: %w", err)
	}

	configContent, err := s.generateServiceContent(config)
	if err != nil {
		return fmt.Errorf("failed to generate content: %w", err)
	}

	// 如果启用了变量替换，对服务内容进行变量替换
	if commandConfig != nil && commandConfig["enableVariableSubstitution"] == "true" {
		substitutor := &VariableSubstitutor{
			config: commandConfig,
			exec:   exec,
		}
		substitutedContent, err := substitutor.Substitute(configContent)
		if err != nil {
			return fmt.Errorf("failed to substitute variables in service content: %w", err)
		}
		configContent = substitutedContent
	}

	ServicePath := filepath.Join(s.ConfigPath, KubeletFileName)
	if err = os.WriteFile(ServicePath, []byte(configContent), utils.RwRR); err != nil {
		return fmt.Errorf("failed to write drop-in file: %w", err)
	}

	return nil
}

func (s *ServiceGenerator) generateServiceContent(config *confv1beta1.KubeletService) (string, error) {
	applyDefaultHostnameOverrideIfMissing(config)

	tmplData := &ServiceData{
		KubeletService: config,
	}

	tpl, err := template.New("kubelet-service").Parse(kubeletservice)
	if err != nil {
		return "", fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err = tpl.Execute(&buf, tmplData); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}
