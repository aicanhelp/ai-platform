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

// Package parser handles parsing of CLI input files: the node info file passed
// via --nodes and the Helm values.yaml passed via --values.
package parser

import (
	"fmt"
	"os"

	"sigs.k8s.io/yaml"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// ParseNodes parses the node info file specified by --nodes
func ParseNodes(path string) ([]types.NodeInfo, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read node file: %w", err)
	}

	var cfg types.NodesConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse node file: %w", err)
	}

	if len(cfg.Nodes) == 0 {
		return nil, fmt.Errorf("no node configuration found in node file")
	}

	for i, node := range cfg.Nodes {
		if node.Name == "" {
			return nil, fmt.Errorf("node #%d is missing the name field", i+1)
		}
		if node.IP == "" {
			return nil, fmt.Errorf("node %s is missing the ip field", node.Name)
		}
		if node.User == "" {
			return nil, fmt.Errorf("node %s is missing the user field", node.Name)
		}
		if node.Password == "" && node.KeyFile == "" {
			return nil, fmt.Errorf("node %s requires either password or keyFile", node.Name)
		}
		if node.Port == 0 {
			cfg.Nodes[i].Port = 22
		}
	}

	return cfg.Nodes, nil
}

// ParseValues parses the Helm values.yaml specified by --values
func ParseValues(path string) (map[string]interface{}, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read values file: %w", err)
	}

	var values map[string]interface{}
	if err := yaml.Unmarshal(data, &values); err != nil {
		return nil, fmt.Errorf("failed to parse values file: %w", err)
	}

	return values, nil
}

// GetNestedString retrieves a string value from a nested map using a dot-separated path.
// Example: GetNestedString(values, "inference-backend.images.inferenceEngine.tag")
func GetNestedString(values map[string]interface{}, path string) (string, error) {
	current := interface{}(values)
	key := ""
	start := 0
	for i := 0; i <= len(path); i++ {
		if i == len(path) || path[i] == '.' {
			key = path[start:i]
			m, ok := current.(map[string]interface{})
			if !ok {
				return "", fmt.Errorf("path %s is not a map type", path[:i])
			}
			val, exists := m[key]
			if !exists {
				return "", fmt.Errorf("path %s does not exist", path[:i])
			}
			current = val
			start = i + 1
		}
	}
	str, ok := current.(string)
	if !ok {
		return "", fmt.Errorf("value at path %s is not a string type", path)
	}
	return str, nil
}
