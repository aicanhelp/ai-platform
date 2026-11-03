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

package config

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"

	"gopkg.in/yaml.v3"
)

var bandwidthItemFieldTypes = yamlFieldTypesOf[BandwidthItem]()

// ErrInvalidConfig indicates that the config file content is invalid.
var ErrInvalidConfig = errors.New("invalid config")

// IsInvalidConfigError reports whether an error was caused by invalid config content.
func IsInvalidConfigError(err error) bool {
	return errors.Is(err, ErrInvalidConfig)
}

// ParseConfig loads bandwidth configuration from path.
func ParseConfig(path string) (*BandwidthConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read config file %s: %w", path, err)
	}

	if len(data) == 0 {
		return nil, nil
	}

	var root yaml.Node
	if err := yaml.Unmarshal(data, &root); err != nil {
		return nil, fmt.Errorf("parse config file %s: %w: %v", path, ErrInvalidConfig, err)
	}
	if err := validateConfigNodeTypes(&root); err != nil {
		return nil, fmt.Errorf("parse config file %s: %w: %v", path, ErrInvalidConfig, err)
	}

	var cfg BandwidthConfig
	if err := root.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parse config file %s: %w: %v", path, ErrInvalidConfig, err)
	}

	return &cfg, nil
}

func validateConfigNodeTypes(root *yaml.Node) error {
	if root == nil || len(root.Content) == 0 {
		return nil
	}
	node := root.Content[0]
	if node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value != "bandwidths" {
			continue
		}
		return validateBandwidthsNode(node.Content[i+1])
	}
	return nil
}

func validateBandwidthsNode(node *yaml.Node) error {
	if node.Kind != yaml.SequenceNode {
		return nil
	}
	for itemIndex, item := range node.Content {
		if item.Kind != yaml.MappingNode {
			continue
		}
		for i := 0; i+1 < len(item.Content); i += 2 {
			field := item.Content[i].Value
			fieldType, ok := bandwidthItemFieldTypes[field]
			if !ok {
				continue
			}
			if err := validateNodeType(item.Content[i+1], fmt.Sprintf("bandwidths[%d].%s", itemIndex, field), fieldType); err != nil {
				return err
			}
		}
	}
	return nil
}

func yamlFieldTypesOf[T any]() map[string]reflect.Type {
	structType := reflect.TypeOf((*T)(nil)).Elem()
	fields := make(map[string]reflect.Type, structType.NumField())
	for i := 0; i < structType.NumField(); i++ {
		field := structType.Field(i)
		if !field.IsExported() {
			continue
		}
		yamlName := yamlFieldName(field)
		if yamlName == "" || yamlName == "-" {
			continue
		}
		fields[yamlName] = field.Type
	}
	return fields
}

func yamlFieldName(field reflect.StructField) string {
	tag := field.Tag.Get("yaml")
	if tag == "" {
		return field.Name
	}
	name, _, _ := strings.Cut(tag, ",")
	return name
}

func validateNodeType(node *yaml.Node, fieldPath string, fieldType reflect.Type) error {
	expectedTag, expectedType, ok := expectedYAMLTag(fieldType)
	if !ok {
		return nil
	}
	if node.Tag != expectedTag {
		return fmt.Errorf("%s must be %s, got %s", fieldPath, expectedType, node.Tag)
	}
	return nil
}

func expectedYAMLTag(fieldType reflect.Type) (string, string, bool) {
	switch fieldType.Kind() {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return "!!int", fieldType.String() + " integer", true
	case reflect.String:
		return "!!str", fieldType.String(), true
	case reflect.Bool:
		return "!!bool", fieldType.String(), true
	case reflect.Float32, reflect.Float64:
		return "!!float", fieldType.String(), true
	default:
		return "", "", false
	}
}
