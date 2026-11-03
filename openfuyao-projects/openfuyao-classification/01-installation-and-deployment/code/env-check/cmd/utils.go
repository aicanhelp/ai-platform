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
	"path/filepath"

	"github.com/spf13/cobra"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

// getOutputFilename generates output filename
func getOutputFilename(baseName, format string) string {
	// Only JSON format is saved to file
	if format == "json" {
		return fmt.Sprintf("%s.json", baseName)
	}
	return ""
}

// saveResultToFile saves result to file
func saveResultToFile(filename string, content string) error {
	// Ensure directory exists
	dir := filepath.Dir(filename)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, config.DirectoryMode); err != nil {
			return fmt.Errorf("failed to create result directory: %v", err)
		}
	}

	// Write to file
	if err := os.WriteFile(filename, []byte(content), config.FileMode); err != nil {
		return fmt.Errorf("failed to write result file: %v", err)
	}

	return nil
}

// actionHandler holds callbacks for a mode-specific operation
type actionHandler struct {
	// execute performs the core operation and returns a result (opaque)
	execute func(cmd *cobra.Command, cfg *config.AppConfig, log logger.Logger) (interface{}, error)
	// generateHTML optionally generates HTML from result
	generateHTML func(result interface{}) error
	// format formats the result into a string for console/file
	format func(result interface{}, outputFormat string) (string, error)
	// baseName used to generate filename when saving
	baseName string
}

// runAction is a generic runner used by subcommands to avoid duplicate code.
// It loads config, initializes logger, executes the action, generates HTML,
// formats output, prints to console (for text) and saves result to file if applicable.
func runAction(cmd *cobra.Command, args []string, mode config.OperationMode, h actionHandler) error {
	// Load configuration
	cfg, err := loadConfig(mode)
	if err != nil {
		return fmt.Errorf("failed to load configuration: %v", err)
	}

	// Initialize logger
	log := logger.NewLogger(cfg.LogFile)
	log.Info("envCheck tool start")
	log.Info(fmt.Sprintf("mode: %s", cfg.Mode))

	// Execute core action
	result, err := h.execute(cmd, cfg, log)
	if err != nil {
		log.Error("operation failed: " + err.Error())
		return err
	}

	// Generate HTML if provided
	if h.generateHTML != nil {
		if err := h.generateHTML(result); err != nil {
			panic(err)
		}
	}

	// Format output if formatter provided
	var outputStr string
	if h.format != nil {
		outputStr, err = h.format(result, cfg.OutputFormat)
		if err != nil {
			log.Error("format output failed: " + err.Error())
			return err
		}
	}

	// Console output when text
	if cfg.OutputFormat == "text" && outputStr != "" {
		fmt.Println(outputStr)
	}

	// Save to file when applicable
	if h.baseName != "" && h.format != nil {
		outputFile := getOutputFilename(h.baseName, cfg.OutputFormat)
		if outputFile != "" {
			if err := saveResultToFile(outputFile, outputStr); err != nil {
				log.Error("failed to save result file: " + err.Error())
			} else {
				log.Info("results have been saved to: " + outputFile)
			}
		}
	}

	log.Info("completed")
	return nil
}
