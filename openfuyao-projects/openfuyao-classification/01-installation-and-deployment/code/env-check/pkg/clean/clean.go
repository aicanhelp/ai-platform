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

// Package clean performs specific cleanup operations
package clean

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"env-check/pkg/config"
	"env-check/pkg/logger"
	"env-check/pkg/utils"
)

// FileCleaner file cleaner
type FileCleaner struct {
	config *config.AppConfig
	logger logger.Logger
}

// NewCleaner creates a new cleaner
func NewCleaner(cfg *config.AppConfig, log logger.Logger) *FileCleaner {
	return &FileCleaner{
		config: cfg,
		logger: log,
	}
}

// Execute executes file cleanup
func (c *FileCleaner) Execute() (*config.CleanResult, error) {
	c.logger.Info("start clean file ......")
	c.logger.Info(fmt.Sprintf("path(s) to be cleaned up: %v", c.config.Paths))

	result := &config.CleanResult{
		Timestamp: time.Now(),
		Deleted:   make([]config.FileInfo, 0),
		Failed:    make([]config.FileInfo, 0),
		Skipped:   make([]config.FileInfo, 0),
		Summary:   config.CleanSummary{},
	}

	// Expand environment variables and wildcards on each configured path
	expander := utils.NewPathExpander()
	for _, path := range c.config.Paths {
		expanded, err := expander.ExpandPath(path)
		if err != nil {
			result.Failed = append(result.Failed, config.FileInfo{
				Path:  path,
				Error: fmt.Sprintf("failed to expand path: %v", err),
			})
			c.logger.Warning(fmt.Sprintf("failed to expand path %s: %v", path, err))
			continue
		}

		if len(expanded) == 0 {
			// Wildcard pattern matched no files
			result.Skipped = append(result.Skipped, config.FileInfo{
				Path:   expander.ExpandEnv(path),
				Exists: false,
				Error:  "no files matched the wildcard pattern",
			})
			c.logger.Info(fmt.Sprintf("wildcard %s matched no files", path))
			continue
		}

		for _, p := range expanded {
			c.cleanPath(p, result)
		}
	}

	// Update summary information
	result.Summary.TotalChecked = len(result.Deleted) + len(result.Failed) + len(result.Skipped)
	result.Summary.TotalDeleted = len(result.Deleted)
	result.Summary.TotalFailed = len(result.Failed)
	result.Summary.TotalSkipped = len(result.Skipped)

	c.logger.Info("file clean completed")
	return result, nil
}

// cleanPath cleans a single path
func (c *FileCleaner) cleanPath(path string, result *config.CleanResult) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		fileInfo := config.FileInfo{
			Path:  path,
			Error: fmt.Sprintf("unable to obtain the absolute path: %v", err),
		}
		result.Failed = append(result.Failed, fileInfo)
		c.logger.Error(fmt.Sprintf("deldete failed: %s - %v", path, err))
		return
	}

	// Check if path exists
	info, err := os.Lstat(absPath)
	if err != nil {
		c.dealLstatError(err, absPath, result)
		return
	}

	fileInfo := config.FileInfo{
		Path:        absPath,
		Exists:      true,
		IsDir:       info.IsDir(),
		Permissions: info.Mode().String(),
	}

	// Ask user for confirmation (if needed)
	if !c.shouldDelete(fileInfo) {
		result.Skipped = append(result.Skipped, fileInfo)
		c.logger.Info(fmt.Sprintf("user skip delete: %s", absPath))
		return
	}

	// Execute deletion
	if err := c.deletePath(fileInfo); err != nil {
		fileInfo.Error = err.Error()
		result.Failed = append(result.Failed, fileInfo)
		c.logger.Error(fmt.Sprintf("delete failed: %s - %v", absPath, err))
	} else {
		result.Deleted = append(result.Deleted, fileInfo)
		c.logger.Success(fmt.Sprintf("delete successfully: %s", absPath))
	}
}

func (c *FileCleaner) dealLstatError(err error, absPath string, result *config.CleanResult) {
	if os.IsNotExist(err) {
		fileInfo := config.FileInfo{
			Path:   absPath,
			Exists: false,
			Error:  "path does not exist",
		}
		result.Skipped = append(result.Skipped, fileInfo)
		c.logger.Info(fmt.Sprintf("skip non-existent path: %s", absPath))
		return
	}

	fileInfo := config.FileInfo{
		Path:  absPath,
		Error: fmt.Sprintf("unable to access the path: %v", err),
	}
	result.Failed = append(result.Failed, fileInfo)
	c.logger.Error(fmt.Sprintf("delete failed: %s - %v", absPath, err))
	return
}

// shouldDelete determines whether to delete
func (c *FileCleaner) shouldDelete(fileInfo config.FileInfo) bool {
	// Auto mode deletes directly
	if c.config.CleanForce {
		return true
	}

	// Ask user for confirmation
	return c.askForDeletion(fileInfo.Path)
}

// deletePath deletes a path
func (c *FileCleaner) deletePath(fileInfo config.FileInfo) error {
	if fileInfo.IsDir {
		return os.RemoveAll(fileInfo.Path)
	} else {
		return os.Remove(fileInfo.Path)
	}
}

// askForDeletion asks user whether to delete
func (c *FileCleaner) askForDeletion(path string) bool {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Printf("delete '%s'? [y/n]: ", path)
		response, err := reader.ReadString('\n')
		if err != nil {
			c.logger.Error("read user input failed: " + err.Error())
			return false
		}

		response = strings.TrimSpace(strings.ToLower(response))

		switch response {
		case "y", "yes":
			return true
		case "n", "no", "":
			return false
		default:
			fmt.Println("please input 'y' or 'n'")
		}
	}
}
