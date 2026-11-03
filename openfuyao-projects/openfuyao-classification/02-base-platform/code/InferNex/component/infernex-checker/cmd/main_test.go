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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

func TestNodeNames(t *testing.T) {
	nodes := []types.NodeInfo{
		{Name: "node-1"},
		{Name: "node-2"},
		{Name: "node-3"},
	}
	names := nodeNames(nodes)
	if len(names) != 3 {
		t.Errorf("expected 3 names, got %d", len(names))
	}
	if names[0] != "node-1" || names[1] != "node-2" || names[2] != "node-3" {
		t.Errorf("unexpected names: %v", names)
	}
}

func TestNodeNames_Empty(t *testing.T) {
	nodes := []types.NodeInfo{}
	names := nodeNames(nodes)
	if len(names) != 0 {
		t.Errorf("expected 0 names, got %d", len(names))
	}
}

func TestDefaultKubeconfig(t *testing.T) {
	path := defaultKubeconfig()
	// Should return a path containing .kube/config or empty string
	if path != "" && !strings.Contains(path, ".kube") {
		t.Errorf("expected .kube in path, got %s", path)
	}
}

func TestLoadNodes(t *testing.T) {
	tmpDir := t.TempDir()
	nodesPath := filepath.Join(tmpDir, "nodes.yaml")
	content := `nodes:
  - name: test-node
    ip: 192.168.1.100
    port: 22
    user: root
    password: test123
`
	if err := os.WriteFile(nodesPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write nodes file: %v", err)
	}

	// Set global variable
	nodesFile = nodesPath
	nodes, err := loadNodes()
	if err != nil {
		t.Fatalf("loadNodes failed: %v", err)
	}

	if len(nodes) != 1 {
		t.Errorf("expected 1 node, got %d", len(nodes))
	}
	if nodes[0].Name != "test-node" {
		t.Errorf("expected test-node, got %s", nodes[0].Name)
	}
}

func TestLoadNodes_FileNotFound(t *testing.T) {
	nodesFile = "/non/existent/path.yaml"
	_, err := loadNodes()
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestLoadValues(t *testing.T) {
	tmpDir := t.TempDir()
	valuesPath := filepath.Join(tmpDir, "values.yaml")
	content := `global:
  cachePath: /tmp/cache
inference-backend:
  images:
    inferenceEngine:
      tag: "1.0.0"
`
	if err := os.WriteFile(valuesPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write values file: %v", err)
	}

	valuesFile = valuesPath
	values, err := loadValues()
	if err != nil {
		t.Fatalf("loadValues failed: %v", err)
	}

	if values == nil {
		t.Error("expected non-nil values")
	}
}

func TestLoadValues_Empty(t *testing.T) {
	valuesFile = ""
	_, err := loadValues()
	if err == nil {
		t.Error("expected error when valuesFile is empty")
	}
	if !strings.Contains(err.Error(), "values.yaml") {
		t.Errorf("error should mention values.yaml, got: %v", err)
	}
}

func TestLoadValues_FileNotFound(t *testing.T) {
	valuesFile = "/non/existent/values.yaml"
	_, err := loadValues()
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestLoadNodesAndValues(t *testing.T) {
	tmpDir := t.TempDir()

	nodesPath := filepath.Join(tmpDir, "nodes.yaml")
	nodesContent := `nodes:
  - name: node-1
    ip: 192.168.1.100
    port: 22
    user: root
    password: test123
`
	if err := os.WriteFile(nodesPath, []byte(nodesContent), 0644); err != nil {
		t.Fatalf("failed to write nodes file: %v", err)
	}

	valuesPath := filepath.Join(tmpDir, "values.yaml")
	valuesContent := `global:
  cachePath: /tmp/cache
`
	if err := os.WriteFile(valuesPath, []byte(valuesContent), 0644); err != nil {
		t.Fatalf("failed to write values file: %v", err)
	}

	nodesFile = nodesPath
	valuesFile = valuesPath

	nodes, values, err := loadNodesAndValues()
	if err != nil {
		t.Fatalf("loadNodesAndValues failed: %v", err)
	}

	if len(nodes) != 1 {
		t.Errorf("expected 1 node, got %d", len(nodes))
	}
	if values == nil {
		t.Error("expected non-nil values")
	}
}

func TestWriteJSON(t *testing.T) {
	tmpDir := t.TempDir()
	jsonPath := filepath.Join(tmpDir, "result.json")

	outputJSON = jsonPath
	report := &types.Report{
		Summary: types.Summary{Total: 1, Passed: 1},
	}

	err := writeJSON(report)
	if err != nil {
		t.Fatalf("writeJSON failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(jsonPath); os.IsNotExist(err) {
		t.Error("JSON file was not created")
	}

	// Clean up
	outputJSON = ""
}

func TestWriteJSON_EmptyPath(t *testing.T) {
	outputJSON = ""
	report := &types.Report{}

	err := writeJSON(report)
	if err != nil {
		t.Errorf("writeJSON should not error when outputJSON is empty, got: %v", err)
	}
}

func TestWriteJSON_InvalidPath(t *testing.T) {
	outputJSON = "/invalid/path/result.json"
	report := &types.Report{}

	err := writeJSON(report)
	if err == nil {
		t.Error("expected error for invalid path")
	}

	// Clean up
	outputJSON = ""
}
