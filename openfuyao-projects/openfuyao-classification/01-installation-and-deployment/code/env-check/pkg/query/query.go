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

// Package query performs specific query operations
package query

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"env-check/pkg/config"
	"env-check/pkg/logger"
	"env-check/pkg/utils"
)

const cmdResMinLen = 3 // Minimum number of parameters returned by ls command result

// FileQuery file query handler
type FileQuery struct {
	config *config.AppConfig
	logger logger.Logger
}

// NewQuery creates a new query handler
func NewQuery(cfg *config.AppConfig, log logger.Logger) *FileQuery {
	return &FileQuery{
		config: cfg,
		logger: log,
	}
}

// Execute executes file query
func (q *FileQuery) Execute() (*config.QueryResult, error) {
	q.logger.Info("file query start ......")
	q.logger.Info(fmt.Sprintf("query path: %v", q.config.Paths))

	result := &config.QueryResult{
		Timestamp: time.Now(),
		Files:     make([]config.FileInfo, 0),
		Summary:   config.QuerySummary{},
	}

	// Expand environment variables and wildcards on each configured path.
	// Expansion runs on the executing node so $HOME / ~ resolve to local values.
	expander := utils.NewPathExpander()
	for _, path := range q.config.Paths {
		q.queryExpandedPath(path, expander, result)
	}

	q.logger.Info("file query completed")
	return result, nil
}

// queryExpandedPath expands a single configured path (env vars + wildcards)
// and appends the resulting file infos and summary counters to result.
func (q *FileQuery) queryExpandedPath(path string, expander *utils.PathExpander, result *config.QueryResult) {
	expanded, err := expander.ExpandPath(path)
	if err != nil {
		result.Files = append(result.Files, config.FileInfo{
			Path:   path,
			Exists: false,
			Error:  fmt.Sprintf("failed to expand path: %v", err),
		})
		result.Summary.TotalChecked++
		result.Summary.TotalMissing++
		q.logger.Warning(fmt.Sprintf("failed to expand path %s: %v", path, err))
		return
	}

	if len(expanded) == 0 {
		// Wildcard pattern matched no files
		result.Files = append(result.Files, config.FileInfo{
			Path:   expander.ExpandEnv(path),
			Exists: false,
			Error:  "no files matched the wildcard pattern",
		})
		result.Summary.TotalChecked++
		result.Summary.TotalMissing++
		q.logger.Info(fmt.Sprintf("wildcard %s matched no files", path))
		return
	}

	for _, p := range expanded {
		fileInfo := q.queryPath(p)
		result.Files = append(result.Files, fileInfo)
		result.Summary.TotalChecked++
		if fileInfo.Exists {
			result.Summary.TotalExists++
			if fileInfo.IsDir {
				result.Summary.TotalDirs++
			} else {
				result.Summary.TotalFiles++
			}
		} else {
			result.Summary.TotalMissing++
		}
	}
}

// queryPath queries a single path
func (q *FileQuery) queryPath(path string) config.FileInfo {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return config.FileInfo{
			Path:   path,
			Exists: false,
			Error:  fmt.Sprintf("unable to obtain the absolute path: %v", err),
		}
	}

	info, err := os.Lstat(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			q.logger.Info(fmt.Sprintf("path not exist: %s", absPath))
			return config.FileInfo{
				Path:   absPath,
				Exists: false,
				Error:  "path not exist",
			}
		}

		q.logger.Warning(fmt.Sprintf("Unable to access path: %s - %v", absPath, err))
		return config.FileInfo{
			Path:   absPath,
			Exists: false,
			Error:  fmt.Sprintf("access err: %v", err),
		}
	}

	user, group, err := getFileOwnerCrossPlatform(absPath)

	if err != nil {
		q.logger.Warning(fmt.Sprintf("Failed to retrieve file owner and group information: %s - %v", absPath, err))
		return config.FileInfo{
			Path:   absPath,
			Exists: true,
			IsDir:  info.IsDir(),
			Error:  fmt.Sprintf("Failed to retrieve file owner and group information: %v", err),
		}
	}

	// Build file information
	fileInfo := config.FileInfo{
		Path:        absPath,
		Exists:      true,
		IsDir:       info.IsDir(),
		Owner:       user,
		Group:       group,
		Permissions: info.Mode().String(),
	}

	// Log success information
	if fileInfo.IsDir {
		q.logger.Warning(fmt.Sprintf("direction exist: %s", absPath))
	} else {
		q.logger.Warning(fmt.Sprintf("file exist: %s", absPath))
	}
	return fileInfo
}

func getFileOwnerCrossPlatform(filePath string) (string, string, error) {
	switch runtime.GOOS {
	case "linux", "darwin", "freebsd":
		return getUnixFileOwner(filePath)
	default:
		return "", "", fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}

func getUnixFileOwner(filePath string) (string, string, error) {
	// Use ls -l command to get file owner information
	cmd := exec.Command("ls", "-ld", filePath)
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		return "", "", err
	}

	output := out.String()
	parts := strings.Fields(output)
	if len(parts) < cmdResMinLen {
		return "", "", fmt.Errorf("unable to parse ls output")
	}

	// ls -l output format: permissions links owner group size date filename
	owner := parts[2]
	group := parts[3]

	return owner, group, nil
}
