/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package config

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
)

func TestConfigurationFlag(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	flag.CommandLine = fs

	ConfigurationFlag()

	if MetricsAddr != "0" {
		t.Errorf("expected default 0, got %s", MetricsAddr)
	}
	if EnableLeaderElection != false {
		t.Errorf("expected default false, got %v", EnableLeaderElection)
	}
	if ProbeAddr != ":8081" {
		t.Errorf("expected default :8081, got %s", ProbeAddr)
	}
	if ProbeScheme != "http" {
		t.Errorf("expected default http, got %s", ProbeScheme)
	}
	if ProbePort != 9444 {
		t.Errorf("expected default 9444, got %d", ProbePort)
	}
	if WebhookPort != 9443 {
		t.Errorf("expected default 9443, got %d", WebhookPort)
	}
	if EnableInternalUpdate != false {
		t.Errorf("expected default false, got %v", EnableInternalUpdate)
	}
}
func TestResolveClientConfigDefaults(t *testing.T) {
	resetClientConfigTestState(t)

	ResolveClientConfig()

	if ClientQPS != DefaultClientQPS {
		t.Fatalf("expected qps %v, got %v", DefaultClientQPS, ClientQPS)
	}
	if ClientBurst != DefaultClientBurst {
		t.Fatalf("expected burst %v, got %v", DefaultClientBurst, ClientBurst)
	}
}

func TestResolveClientConfigPriority(t *testing.T) {
	resetClientConfigTestState(t)
	t.Setenv("KUBE_CLIENT_QPS", "80")
	t.Setenv("KUBE_CLIENT_BURST", "160")

	configFile := filepath.Join(t.TempDir(), "client-config.yaml")
	if err := os.WriteFile(configFile, []byte("qps: 60\nburst: 120\n"), 0644); err != nil {
		t.Fatal(err)
	}

	ConfigurationFlag()
	if err := flag.CommandLine.Parse([]string{"--client-config-file", configFile, "--client-qps", "100", "--client-burst", "200"}); err != nil {
		t.Fatal(err)
	}
	ResolveClientConfig()

	if ClientQPS != 100 {
		t.Fatalf("expected qps 100, got %v", ClientQPS)
	}
	if ClientBurst != 200 {
		t.Fatalf("expected burst 200, got %v", ClientBurst)
	}
}

func resetClientConfigTestState(t *testing.T) {
	t.Helper()
	flag.CommandLine = flag.NewFlagSet(t.Name(), flag.ContinueOnError)
	ClientQPS = 0
	ClientBurst = 0
	ClientConfigFile = ""
	t.Setenv("KUBE_CLIENT_QPS", "")
	t.Setenv("KUBE_CLIENT_BURST", "")
}
