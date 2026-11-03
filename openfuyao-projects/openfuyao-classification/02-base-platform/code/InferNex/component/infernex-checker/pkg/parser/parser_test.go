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

package parser

import (
	"os"
	"path/filepath"
	"testing"
)

// writeTemp writes content to a temp file and returns its path.
func writeTemp(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "*.yaml")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	if _, err := f.WriteString(content); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}
	f.Close()
	return f.Name()
}

// ---------------------------------------------------------------------------
// ParseNodes
// ---------------------------------------------------------------------------

func TestParseNodes_Valid(t *testing.T) {
	content := `
nodes:
  - name: node1
    ip: 192.168.1.1
    port: 2222
    user: root
    password: secret
  - name: node2
    ip: 192.168.1.2
    user: admin
    keyFile: /home/admin/.ssh/id_rsa
`
	path := writeTemp(t, content)
	nodes, err := ParseNodes(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(nodes))
	}
	if nodes[0].Name != "node1" || nodes[0].IP != "192.168.1.1" || nodes[0].Port != 2222 {
		t.Errorf("node1 fields mismatch: %+v", nodes[0])
	}
	if nodes[1].Name != "node2" || nodes[1].KeyFile != "/home/admin/.ssh/id_rsa" {
		t.Errorf("node2 fields mismatch: %+v", nodes[1])
	}
}

func TestParseNodes_DefaultPort(t *testing.T) {
	content := `
nodes:
  - name: node1
    ip: 10.0.0.1
    user: root
    password: pass
`
	path := writeTemp(t, content)
	nodes, err := ParseNodes(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if nodes[0].Port != 22 {
		t.Errorf("expected default port 22, got %d", nodes[0].Port)
	}
}

func TestParseNodes_FileNotFound(t *testing.T) {
	_, err := ParseNodes(filepath.Join(t.TempDir(), "nonexistent.yaml"))
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

func TestParseNodes_InvalidYAML(t *testing.T) {
	path := writeTemp(t, "nodes: [invalid: yaml: :")
	_, err := ParseNodes(path)
	if err == nil {
		t.Fatal("expected error for invalid YAML, got nil")
	}
}

func TestParseNodes_EmptyNodes(t *testing.T) {
	path := writeTemp(t, "nodes: []\n")
	_, err := ParseNodes(path)
	if err == nil {
		t.Fatal("expected error for empty nodes list, got nil")
	}
}

func TestParseNodes_MissingName(t *testing.T) {
	content := `
nodes:
  - ip: 10.0.0.1
    user: root
    password: pass
`
	path := writeTemp(t, content)
	_, err := ParseNodes(path)
	if err == nil {
		t.Fatal("expected error for missing name, got nil")
	}
}

func TestParseNodes_MissingIP(t *testing.T) {
	content := `
nodes:
  - name: node1
    user: root
    password: pass
`
	path := writeTemp(t, content)
	_, err := ParseNodes(path)
	if err == nil {
		t.Fatal("expected error for missing ip, got nil")
	}
}

func TestParseNodes_MissingUser(t *testing.T) {
	content := `
nodes:
  - name: node1
    ip: 10.0.0.1
    password: pass
`
	path := writeTemp(t, content)
	_, err := ParseNodes(path)
	if err == nil {
		t.Fatal("expected error for missing user, got nil")
	}
}

func TestParseNodes_MissingAuth(t *testing.T) {
	content := `
nodes:
  - name: node1
    ip: 10.0.0.1
    user: root
`
	path := writeTemp(t, content)
	_, err := ParseNodes(path)
	if err == nil {
		t.Fatal("expected error when both password and keyFile are absent, got nil")
	}
}

func TestParseNodes_KeyFileOnly(t *testing.T) {
	content := `
nodes:
  - name: node1
    ip: 10.0.0.1
    user: root
    keyFile: /root/.ssh/id_rsa
`
	path := writeTemp(t, content)
	nodes, err := ParseNodes(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if nodes[0].KeyFile != "/root/.ssh/id_rsa" {
		t.Errorf("unexpected keyFile: %s", nodes[0].KeyFile)
	}
}

// ---------------------------------------------------------------------------
// ParseValues
// ---------------------------------------------------------------------------

func TestParseValues_Valid(t *testing.T) {
	content := `
inference-backend:
  images:
    inferenceEngine:
      tag: v1.2.3
replicas: 3
`
	path := writeTemp(t, content)
	values, err := ParseValues(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if values == nil {
		t.Fatal("expected non-nil values map")
	}
	if _, ok := values["inference-backend"]; !ok {
		t.Error("expected inference-backend key in values")
	}
}

func TestParseValues_FileNotFound(t *testing.T) {
	_, err := ParseValues(filepath.Join(t.TempDir(), "nonexistent.yaml"))
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

func TestParseValues_InvalidYAML(t *testing.T) {
	path := writeTemp(t, "key: [bad: yaml: :")
	_, err := ParseValues(path)
	if err == nil {
		t.Fatal("expected error for invalid YAML, got nil")
	}
}

func TestParseValues_EmptyFile(t *testing.T) {
	path := writeTemp(t, "")
	// empty YAML unmarshals to nil map — no error expected
	_, err := ParseValues(path)
	if err != nil {
		t.Fatalf("unexpected error for empty file: %v", err)
	}
}

// ---------------------------------------------------------------------------
// GetNestedString
// ---------------------------------------------------------------------------

func TestGetNestedString_SingleKey(t *testing.T) {
	values := map[string]interface{}{
		"tag": "v1.0.0",
	}
	got, err := GetNestedString(values, "tag")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "v1.0.0" {
		t.Errorf("expected v1.0.0, got %s", got)
	}
}

func TestGetNestedString_NestedKey(t *testing.T) {
	values := map[string]interface{}{
		"inference-backend": map[string]interface{}{
			"images": map[string]interface{}{
				"inferenceEngine": map[string]interface{}{
					"tag": "v2.3.4",
				},
			},
		},
	}
	got, err := GetNestedString(values, "inference-backend.images.inferenceEngine.tag")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "v2.3.4" {
		t.Errorf("expected v2.3.4, got %s", got)
	}
}

func TestGetNestedString_MissingKey(t *testing.T) {
	values := map[string]interface{}{
		"a": map[string]interface{}{},
	}
	_, err := GetNestedString(values, "a.missing")
	if err == nil {
		t.Fatal("expected error for missing key, got nil")
	}
}

func TestGetNestedString_NonMapIntermediate(t *testing.T) {
	values := map[string]interface{}{
		"a": "not-a-map",
	}
	_, err := GetNestedString(values, "a.b")
	if err == nil {
		t.Fatal("expected error when intermediate node is not a map, got nil")
	}
}

func TestGetNestedString_NonStringLeaf(t *testing.T) {
	values := map[string]interface{}{
		"count": 42,
	}
	_, err := GetNestedString(values, "count")
	if err == nil {
		t.Fatal("expected error when leaf value is not a string, got nil")
	}
}

func TestGetNestedString_TopLevelMissing(t *testing.T) {
	values := map[string]interface{}{}
	_, err := GetNestedString(values, "nonexistent")
	if err == nil {
		t.Fatal("expected error for missing top-level key, got nil")
	}
}
