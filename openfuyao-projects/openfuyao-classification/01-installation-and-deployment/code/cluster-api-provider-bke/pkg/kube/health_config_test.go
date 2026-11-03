/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"context"
	"testing"
	"time"

	healthconfig "gopkg.openfuyao.cn/cluster-api-provider-bke/config/health"
)

func TestParseHealthCheckConfigYAMLFromEmbeddedFile(t *testing.T) {
	config, err := parseHealthCheckConfigYAML(healthconfig.DefaultConfig)
	if err != nil {
		t.Fatalf("parse embedded health check config failed: %v", err)
	}
	if len(config.Components) != 8 {
		t.Fatalf("embedded health check components = %d, want 8", len(config.Components))
	}
	if config.Intervals.Critical != 5*time.Second || config.Intervals.Important != 15*time.Second || config.Intervals.Optional != 30*time.Second || config.Intervals.Normal != 5*time.Minute {
		t.Fatalf("unexpected intervals: %+v", config.Intervals)
	}
}

func TestParseHealthCheckConfigYAMLCustomConfigMapData(t *testing.T) {
	config, err := parseHealthCheckConfigYAML(`intervals:
  critical: 7s
  important: 17s
  optional: 37s
  normal: 6m
components:
  - name: etcd
    namespace: kube-system
    prefixes:
      - etcd-
    priority: critical
`)
	if err != nil {
		t.Fatalf("parse custom health check config failed: %v", err)
	}
	if len(config.Components) != 1 {
		t.Fatalf("components = %d, want 1", len(config.Components))
	}
	if config.Components[0].Name != NameEtcd || config.Components[0].Prefixes[0] != "etcd-" {
		t.Fatalf("unexpected component: %+v", config.Components[0])
	}
	if config.Intervals.Critical != 7*time.Second || config.Intervals.Normal != 6*time.Minute {
		t.Fatalf("unexpected intervals: %+v", config.Intervals)
	}
}

func TestLoadHealthCheckConfigFallbackWhenClientSetMissing(t *testing.T) {
	client := &Client{Ctx: context.Background()}

	config := client.LoadHealthCheckConfig()
	if len(config.Components) != 8 {
		t.Fatalf("fallback components = %d, want 8", len(config.Components))
	}
}
