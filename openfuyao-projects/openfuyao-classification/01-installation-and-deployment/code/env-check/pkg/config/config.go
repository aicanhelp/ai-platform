/******************************************************************
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package config handles reading and parsing configuration
package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// LoadConfigFromFile loads configuration from file
func LoadConfigFromFile(filename string, config *AppConfig) error {
	file, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	return json.Unmarshal(file, config)
}

// ValidateConfig validates configuration
func ValidateConfig(config *AppConfig) error {
	validModes := map[OperationMode]bool{
		ModeQuery:         true,
		ModeClean:         true,
		ModeProgramCheck:  true,
		ModeClockCheck:    true,
		ModeRouteCheck:    true,
		ModeIptablesCheck: true,
		"kernelCheck":     true,
		"portCheck":       true,
		"diskCheck":       true,
		"dispatch":        true,
	}

	if !validModes[config.Mode] {
		return fmt.Errorf("unsupported mode: %s", config.Mode)
	}

	// For query and clean modes, paths are required
	if (config.Mode == ModeQuery || config.Mode == ModeClean) && len(config.Paths) == 0 {
		return fmt.Errorf("at least one path is required")
	}

	// For program check mode, program list is required
	if config.Mode == ModeProgramCheck && len(config.ProgramList) == 0 {
		return fmt.Errorf("program list is required")
	}

	// For clock check mode, hosts is required
	if config.Mode == ModeClockCheck && len(config.Hosts) == 0 {
		return fmt.Errorf("hosts field is required")
	}

	// For route check and iptables check modes, no additional config is required
	return nil
}
