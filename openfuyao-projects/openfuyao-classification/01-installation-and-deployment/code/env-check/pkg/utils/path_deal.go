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

// Package utils contains utility methods
package utils

import (
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"strings"
)

const commonPathBeginIdx = 2 // Normal path start index when including home directory

// PathExpander path expander
type PathExpander struct {
	// Whether to expand wildcards
	ExpandWildcards bool
	// Whether to expand environment variables
	ExpandEnvVars bool
}

// NewPathExpander creates a new path expander
func NewPathExpander() *PathExpander {
	return &PathExpander{
		ExpandWildcards: true,
		ExpandEnvVars:   true,
	}
}

// ExpandPaths expands file path list
func (p *PathExpander) ExpandPaths(paths []string) ([]string, error) {
	var result []string

	for _, path := range paths {
		expanded, err := p.ExpandPath(path)
		if err != nil {
			return nil, err
		}
		result = append(result, expanded...)
	}

	return result, nil
}

// ExpandPath expands a single file path
func (p *PathExpander) ExpandPath(path string) ([]string, error) {
	var result []string

	// Expand environment variables
	if p.ExpandEnvVars {
		path = p.expandEnvironmentVariables(path)
	}

	// Check if path contains wildcards
	if p.ExpandWildcards && containsWildcard(path) {
		matches, err := filepath.Glob(path)
		if err != nil {
			return nil, fmt.Errorf("expand wildcard failed: %v", err)
		}

		for _, match := range matches {
			result = append(result, match)
		}
	} else {
		// No wildcards, add directly
		result = append(result, path)
	}

	return result, nil
}

// expandEnvironmentVariables expands environment variables
func (p *PathExpander) expandEnvironmentVariables(path string) string {
	// Expand standard environment variables
	path = os.ExpandEnv(path)

	// Special handling for ~ (home directory)
	if strings.HasPrefix(path, "~") {
		if usr, err := user.Current(); err == nil {
			homeDir := usr.HomeDir
			if path == "~" {
				path = homeDir
			} else if strings.HasPrefix(path, "~/") {
				path = filepath.Join(homeDir, path[commonPathBeginIdx:])
			} else {
				return path
			}
		}
	}

	return path
}

// ExpandEnv expands environment variables and ~ in the path without performing
// wildcard globbing. Useful for reporting the resolved pattern when a wildcard
// has no matches.
func (p *PathExpander) ExpandEnv(path string) string {
	if !p.ExpandEnvVars {
		return path
	}
	return p.expandEnvironmentVariables(path)
}

// containsWildcard checks if path contains wildcards
func containsWildcard(path string) bool {
	wildcards := []string{"*", "?", "[", "]"}
	for _, wc := range wildcards {
		if strings.Contains(path, wc) {
			return true
		}
	}
	return false
}
