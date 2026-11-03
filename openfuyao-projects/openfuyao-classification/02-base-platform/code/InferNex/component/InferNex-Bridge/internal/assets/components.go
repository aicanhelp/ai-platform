/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package assets

import (
	"embed"
	"fmt"

	"sigs.k8s.io/yaml"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

//go:embed components/*.yaml
var componentTemplates embed.FS

var componentTemplateFiles = map[string]string{
	"pd-orchestrator-elastic-scaler":         "components/pd-orchestrator-elastic-scaler.yaml",
	"pd-orchestrator-tidal":                  "components/pd-orchestrator-tidal.yaml",
	"pd-orchestrator-rsg":                    "components/pd-orchestrator-rsg.yaml",
	"eagle-eye-hardware-monitor":             "components/eagle-eye-hardware-monitor.yaml",
	"eagle-eye-hardware-diagnosis":           "components/eagle-eye-hardware-diagnosis.yaml",
	"eagle-eye-network-performance-exporter": "components/eagle-eye-network-performance-exporter.yaml",
	"mooncake-metadata":                      "components/mooncake-metadata.yaml",
	"cache-indexer":                          "components/cache-indexer.yaml",
	"proxy-server":                           "components/proxy-server.yaml",
}

func LoadManagedComponent(name string) (*infernexv1alpha1.ComponentSpec, error) {
	file, ok := componentTemplateFiles[name]
	if !ok {
		return nil, fmt.Errorf("unknown managed component %q", name)
	}

	data, err := componentTemplates.ReadFile(file)
	if err != nil {
		return nil, err
	}

	var spec infernexv1alpha1.ComponentSpec
	if err := yaml.Unmarshal(data, &spec); err != nil {
		return nil, err
	}
	if spec.Template == nil {
		return nil, fmt.Errorf("component asset %q has no template", name)
	}
	return &spec, nil
}
