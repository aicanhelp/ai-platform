/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package npu

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func validParameters() Parameters {
	return Parameters{
		Path:                     "/metrics",
		ExporterNamespace:        "npu-exporter",
		ExporterPodLabelSelector: "app=npu-exporter",
		ExporterPort:             8082,
		CacheTTL:                 Duration(100 * time.Millisecond),
		ScrapeTimeout:            Duration(2 * time.Second),
	}
}

func TestParametersValidate(t *testing.T) {
	t.Run("ok", func(t *testing.T) {
		p := validParameters()
		p.applyDefaults()
		if err := p.Validate(); err != nil {
			t.Fatalf("expected ok, got %v", err)
		}
	})

	cases := map[string]func(*Parameters){
		"path no slash":             func(p *Parameters) { p.Path = "metrics" },
		"invalid exporter selector": func(p *Parameters) { p.ExporterPodLabelSelector = "app in" },
		"invalid exporter port":     func(p *Parameters) { p.ExporterPort = 65536 },
		"zero cache ttl":            func(p *Parameters) { p.CacheTTL = 0 },
		"zero scrape":               func(p *Parameters) { p.ScrapeTimeout = 0 },
		"namespace == pod label":    func(p *Parameters) { p.NamespaceLabelKey = "x"; p.PodLabelKey = "x" },
		"namespace == device label": func(p *Parameters) { p.NamespaceLabelKey = deviceLabelKey },
		"pod == device label":       func(p *Parameters) { p.PodLabelKey = deviceLabelKey },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			p := validParameters()
			mutate(&p)
			// Note: we deliberately do NOT call applyDefaults here for the
			// "path no slash" case since defaults would normalize it.
			if name != "path no slash" {
				p.applyDefaults()
			}
			if err := p.Validate(); err == nil {
				t.Fatalf("expected validation error for %s", name)
			}
		})
	}
}

func TestParametersApplyDefaults(t *testing.T) {
	var p Parameters
	p.applyDefaults()
	if p.Path != defaultPath ||
		p.NamespaceLabelKey != defaultNamespaceLabelKey ||
		p.PodLabelKey != defaultPodLabelKey ||
		p.ExporterNamespace != defaultExporterNamespace ||
		p.ExporterPodLabelSelector != defaultExporterPodLabelSelector ||
		p.ExporterPort != defaultExporterPort {
		t.Fatalf("unexpected defaults: %+v", p)
	}
}

func TestDurationUnmarshalJSONString(t *testing.T) {
	var p Parameters
	if err := json.Unmarshal([]byte(`{"cacheTTL":"250ms","scrapeTimeout":"3s"}`), &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if p.CacheTTL.Std() != 250*time.Millisecond || p.ScrapeTimeout.Std() != 3*time.Second {
		t.Fatalf("unexpected durations: %+v", p)
	}
}

func TestDurationUnmarshalJSONInteger(t *testing.T) {
	var p Parameters
	if err := json.Unmarshal([]byte(`{"cacheTTL":123456}`), &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if p.CacheTTL.Std() != time.Duration(123456) {
		t.Fatalf("unexpected duration: %s", p.CacheTTL.Std())
	}
}

func TestDurationUnmarshalJSONInvalidString(t *testing.T) {
	var p Parameters
	err := json.Unmarshal([]byte(`{"cacheTTL":"not-a-duration"}`), &p)
	if err == nil || !strings.Contains(err.Error(), "invalid duration") {
		t.Fatalf("expected invalid duration error, got %v", err)
	}
}
