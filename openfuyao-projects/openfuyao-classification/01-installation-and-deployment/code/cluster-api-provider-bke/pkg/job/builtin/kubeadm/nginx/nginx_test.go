/******************************************************************
 * Copyright (c) 2026 China Mobile (Suzhou) Software Technology Co., Ltd
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package nginx

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
)

func TestParsePort(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    int
		wantErr bool
	}{
		{name: "empty defaults to 6443", input: "", want: 6443},
		{name: "valid port", input: "8443", want: 8443},
		{name: "dirty input rejected", input: "6443abc", wantErr: true},
		{name: "zero rejected", input: "0", wantErr: true},
		{name: "out of range rejected", input: "65536", wantErr: true},
		{name: "negative rejected", input: "-1", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePort(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parsePort(%q) expected error, got %d", tt.input, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parsePort(%q) unexpected error: %v", tt.input, err)
			}
			if got != tt.want {
				t.Errorf("parsePort(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestParseMasterNodes(t *testing.T) {
	if _, err := parseMasterNodes(""); err == nil {
		t.Error("parseMasterNodes(\"\") expected error")
	}
	if _, err := parseMasterNodes(" , ,"); err == nil {
		t.Error("parseMasterNodes(\" , ,\") expected error")
	}
	nodes, err := parseMasterNodes("10.0.0.1, 10.0.0.2 ,10.0.0.3")
	if err != nil {
		t.Fatalf("parseMasterNodes unexpected error: %v", err)
	}
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}
	if nodes[1] != "10.0.0.2" {
		t.Errorf("expected trimmed IP 10.0.0.2, got %q", nodes[1])
	}
}

func TestExecuteMissingMasterNodes(t *testing.T) {
	n := New(nil, nil)
	if _, err := n.Execute(context.Background(), []string{"Nginx", "apiServerPort=6443"}); err == nil {
		t.Fatal("expected error for missing required masterNodes")
	}
}

func TestGenerateNginxConf(t *testing.T) {
	n := New(nil, nil)
	confDir := t.TempDir()
	params := &Params{
		MasterNodes:   []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		APIServerPort: 6443,
		ConfPath:      confDir,
	}
	if err := n.generateNginxConf(params); err != nil {
		t.Fatalf("generateNginxConf error: %v", err)
	}

	content, err := os.ReadFile(filepath.Join(confDir, nginxConfFileName))
	if err != nil {
		t.Fatalf("read nginx.conf error: %v", err)
	}
	conf := string(content)
	for _, ip := range params.MasterNodes {
		if !strings.Contains(conf, "server "+ip+":6443;") {
			t.Errorf("nginx.conf missing upstream server %s:6443", ip)
		}
	}
	if !strings.Contains(conf, "listen        127.0.0.1:6443;") {
		t.Error("nginx.conf missing listen 127.0.0.1:6443")
	}
	if !strings.Contains(conf, "least_conn;") {
		t.Error("nginx.conf missing least_conn")
	}
}

func TestExecuteEndToEnd(t *testing.T) {
	confDir := t.TempDir()
	manifestDir := t.TempDir()
	boot := &mfutil.BootScope{
		BkeConfig:   &bkev1beta1.BKEConfig{},
		ClusterName: "test-cluster",
		HostName:    "worker-1",
		HostIP:      "10.0.0.4",
	}
	n := New(nil, boot)

	commands := []string{
		"Nginx",
		"masterNodes=10.0.0.1,10.0.0.2",
		"apiServerPort=6443",
		"manifestsDir=" + confDir,
		"manifestPath=" + manifestDir,
	}
	if _, err := n.Execute(context.Background(), commands); err != nil {
		t.Fatalf("Execute error: %v", err)
	}

	conf, err := os.ReadFile(filepath.Join(confDir, nginxConfFileName))
	if err != nil {
		t.Fatalf("read nginx.conf error: %v", err)
	}
	for _, ip := range []string{"10.0.0.1", "10.0.0.2"} {
		if !strings.Contains(string(conf), "server "+ip+":6443;") {
			t.Errorf("nginx.conf missing upstream server %s:6443", ip)
		}
	}

	manifest, err := os.ReadFile(filepath.Join(manifestDir, nginxManifestFileName))
	if err != nil {
		t.Fatalf("read nginx-proxy.yaml error: %v", err)
	}
	yaml := string(manifest)

	mustContain := []string{
		"nginx:1.28.0",
		"type: DirectoryOrCreate",
		"livenessProbe:",
		"readinessProbe:",
		"hostNetwork: true",
		"priorityClassName: system-node-critical",
		"memory: 512Mi",
	}
	for _, s := range mustContain {
		if !strings.Contains(yaml, s) {
			t.Errorf("nginx-proxy.yaml missing %q", s)
		}
	}
	mustNotContain := []string{
		"privileged",
		"1.21.4",
	}
	for _, s := range mustNotContain {
		if strings.Contains(yaml, s) {
			t.Errorf("nginx-proxy.yaml must not contain %q", s)
		}
	}
}

func TestGenerateNginxManifestNilBoot(t *testing.T) {
	n := New(nil, nil)
	params := &Params{
		MasterNodes:   []string{"10.0.0.1"},
		APIServerPort: 6443,
		ConfPath:      t.TempDir(),
		ManifestPath:  t.TempDir(),
	}
	if err := n.generateNginxManifest(params); err == nil {
		t.Fatal("expected error for nil BootScope")
	}
}
