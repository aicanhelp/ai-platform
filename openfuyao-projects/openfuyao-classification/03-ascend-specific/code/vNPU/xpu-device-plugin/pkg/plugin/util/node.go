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

package util

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"huawei.com/vxpu-device-plugin/pkg/lock"
	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

const (
	nodeConfigCmName   = "vnpu.node-mode.config"
	nodeConfigYamlName = "node-mode.config.yaml"

	templateConfigCmName   = "vnpu.virtual-template.config"
	templateConfigYamlName = "virtual-template-config.yaml"

	softMode = "soft"
	hardMode = "hard"

	fullCardTemplate = "full"
)

func SetNodeConfig() {
	parseNodeModeFromConfigMap()
	log.Infof("node mode is %s", config.NodeMode)

	enableDeviceShareMode()

	if !IsSoftMode() {
		disablePresetVirtualDevice()
	}

	parseNodeTemplateFromConfigMap()
}

func parseNodeModeFromConfigMap() {
	configMap, err := lock.GetClient().CoreV1().ConfigMaps(os.Getenv("POD_NAMESPACE")).
		Get(context.Background(), nodeConfigCmName, metav1.GetOptions{})
	if err != nil {
		panic(err)
	}

	yamlData, exists := configMap.Data[nodeConfigYamlName]
	if !exists {
		panic(fmt.Errorf("node mode configmap does not exist"))
	}

	var configData types.NodeConfigData
	if err := yaml.Unmarshal([]byte(yamlData), &configData); err != nil {
		panic(err)
	}

	for _, c := range configData.NodeConfigs {
		if c.Name == config.NodeName {
			config.NodeMode = c.Mode
			break
		}
	}

	if config.NodeMode == "" {
		config.NodeMode = softMode
	}

	if config.NodeMode != softMode && config.NodeMode != hardMode {
		panic("invalid node mode: " + config.NodeMode)
	}
}

func disablePresetVirtualDevice() {
	cmd := exec.Command("npu-smi", "set", "-t", "vnpu-cfg-recover", "-d", "0")
	output, err := cmd.Output()
	if err != nil {
		log.Errorf("failed to disable virtual device: %v, output: %s", err, string(output))
		return
	}
	log.Infof("disable virtual device\n %s", string(output))
}

func enableDeviceShareMode() {
	cardIDs, err := xpu.GetCardIDs()
	if err != nil {
		log.Errorf("failed to get card list: %v", err)
		return
	}
	if len(cardIDs) == 0 {
		log.Errorf("no NPU devices found")
		return
	}
	for _, cardID := range cardIDs {
		cmd := exec.Command("npu-smi", "set", "-t", "device-share",
			"-i", strconv.FormatInt(int64(cardID), 10), "-d", "1")
		cmd.Stdin = strings.NewReader("Y\n")
		output, err := cmd.CombinedOutput()
		if err != nil {
			log.Errorf("failed to enable device share for card %d: %v, output: %s",
				cardID, err, string(output))
			continue
		}
		log.Infof("enabled device share for card %d", cardID)
	}
	cmd := exec.Command("npu-smi", "set", "-t", "device-share-cfg-recover", "-d", "1")
	cmd.Stdin = strings.NewReader("Y\n")
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Errorf("failed to enable device share persistence: %v, output: %s",
			err, string(output))
		return
	}
	log.Infof("device share persistence enabled")
}

func IsSoftMode() bool {
	return config.NodeMode == softMode
}

func IsHardMode() bool {
	return config.NodeMode == hardMode
}

func parseNodeTemplateFromConfigMap() {
	configMap, err := lock.GetClient().CoreV1().ConfigMaps(os.Getenv("POD_NAMESPACE")).
		Get(context.Background(), templateConfigCmName, metav1.GetOptions{})
	if err != nil {
		log.Errorf("parseNodeTemplateFromConfigMap err: %v", err)
		return
	}

	yamlData, exists := configMap.Data[templateConfigYamlName]
	if !exists {
		log.Errorf("parseNodeTemplateFromConfigMap err: ConfigMap not exist key: %s", templateConfigYamlName)
		return
	}

	var templateData config.VnpuTemplateData
	if err := yaml.Unmarshal([]byte(yamlData), &templateData); err != nil {
		log.Errorf("parseNodeTemplateFromConfigMap err: parse template yaml failed %v", err)
		return
	}

	config.NodeTemplateInfos = make(config.TemplateInfos)
	for _, v := range templateData.Vnpus {
		templateMap := map[string]*config.TemplateInfo{}
		for _, t := range v.Templates {
			templateMap[t.Name] = &config.TemplateInfo{
				Name:        t.Name,
				AiCore:      t.AiCore,
				AiCPU:       t.AiCPU,
				TotalAiCore: v.AiCore,
				TotalAiCpu:  v.AiCPU,
			}
		}
		templateMap[fullCardTemplate] = &config.TemplateInfo{
			Name:        fullCardTemplate,
			AiCore:      v.AiCore,
			AiCPU:       v.AiCPU,
			TotalAiCore: v.AiCore,
			TotalAiCpu:  v.AiCPU,
		}
		config.NodeTemplateInfos[v.ChipName] = templateMap
	}
	log.Infof("parseNodeTemplateFromConfigMap success, loaded %d device types", len(config.NodeTemplateInfos))
}
