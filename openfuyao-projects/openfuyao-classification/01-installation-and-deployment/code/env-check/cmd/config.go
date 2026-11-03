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

package main

import (
	"fmt"
	"os"

	"env-check/pkg/config"
)

// loadConfig loads configuration
func loadConfig(defaultMode config.OperationMode) (*config.AppConfig, error) {
	cfg := &config.AppConfig{}

	// Load from configuration file (if specified)
	if cfgFile != "" {
		if err := config.LoadConfigFromFile(cfgFile, cfg); err != nil {
			// Ignore error if config file doesn't exist, use default configuration
			if !os.IsNotExist(err) {
				return nil, fmt.Errorf("failed to load configuration file: %v", err)
			}
		}
	}
	cfg.Mode = defaultMode

	// Validate configuration
	if err := config.ValidateConfig(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}
