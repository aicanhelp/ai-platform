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
	"fmt"
	"strings"

	metav1labels "k8s.io/apimachinery/pkg/labels"

	"hermes-router/pkg/epp/internal/utils"
)

const (
	defaultPath                     = "/metrics"
	defaultNamespaceLabelKey        = "namespace"
	defaultPodLabelKey              = "pod_name"
	defaultExporterNamespace        = "npu-exporter"
	defaultExporterPodLabelSelector = "app=npu-exporter"
	defaultExporterPort             = 8082
	maxTCPPort                      = 65535
)

// Parameters is the user-facing configuration for the NPU exporter datasource.
// Fields are decoded directly from the plugin "parameters" YAML block.
//
// HTTPS is intentionally unsupported in this iteration: the exporter is always
// scraped over plaintext HTTP. TLS/auth knobs will be reintroduced as part of
// a separate production-hardening change.
type Parameters struct {
	Path                     string   `json:"path"`
	NamespaceLabelKey        string   `json:"namespaceLabelKey"`
	PodLabelKey              string   `json:"podLabelKey"`
	ExporterNamespace        string   `json:"exporterNamespace"`
	ExporterPodLabelSelector string   `json:"exporterPodLabelSelector"`
	ExporterPort             int32    `json:"exporterPort"`
	CacheTTL                 Duration `json:"cacheTTL"`
	ScrapeTimeout            Duration `json:"scrapeTimeout"`
}

// Duration accepts either a Go duration string ("100ms") or a numeric
// nanosecond value, mirroring how the GIE config loader serializes durations.
type Duration = utils.Duration

// applyDefaults fills zero-valued fields with package defaults. Required
// fields with no sensible default are not touched here; Validate enforces them.
func (p *Parameters) applyDefaults() {
	if p.Path == "" {
		p.Path = defaultPath
	}
	if p.NamespaceLabelKey == "" {
		p.NamespaceLabelKey = defaultNamespaceLabelKey
	}
	if p.PodLabelKey == "" {
		p.PodLabelKey = defaultPodLabelKey
	}
	if p.ExporterNamespace == "" {
		p.ExporterNamespace = defaultExporterNamespace
	}
	if p.ExporterPodLabelSelector == "" {
		p.ExporterPodLabelSelector = defaultExporterPodLabelSelector
	}
	if p.ExporterPort == 0 {
		p.ExporterPort = defaultExporterPort
	}
	if !strings.HasPrefix(p.Path, "/") {
		p.Path = "/" + p.Path
	}
}

// Validate enforces the fixed contract of the NPU exporter datasource.
func (p Parameters) Validate() error {
	if !strings.HasPrefix(p.Path, "/") {
		return fmt.Errorf("path must start with /")
	}
	if err := validateLabelKeys(p.NamespaceLabelKey, p.PodLabelKey); err != nil {
		return err
	}
	if p.ExporterNamespace == "" {
		return fmt.Errorf("exporterNamespace must not be empty")
	}
	if p.ExporterPodLabelSelector == "" {
		return fmt.Errorf("exporterPodLabelSelector must not be empty")
	}
	if _, err := metav1labels.Parse(p.ExporterPodLabelSelector); err != nil {
		return fmt.Errorf("exporterPodLabelSelector must be a valid label selector: %w", err)
	}
	if p.ExporterPort <= 0 || p.ExporterPort > maxTCPPort {
		return fmt.Errorf("exporterPort must be between 1 and %d", maxTCPPort)
	}
	if p.CacheTTL.Std() <= 0 {
		return fmt.Errorf("cacheTTL must be positive")
	}
	if p.ScrapeTimeout.Std() <= 0 {
		return fmt.Errorf("scrapeTimeout must be positive")
	}
	return nil
}

// validateLabelKeys rejects label-key configurations that would cause the
// per-sample label scan in snapshot.requiredLabels to silently misclassify
// labels (e.g. a single label being treated as both pod and device id, or
// the namespace key shadowing the pod key in the switch statement).
func validateLabelKeys(namespaceKey, podKey string) error {
	if namespaceKey == "" {
		return fmt.Errorf("namespaceLabelKey must not be empty")
	}
	if podKey == "" {
		return fmt.Errorf("podLabelKey must not be empty")
	}
	if namespaceKey == podKey {
		return fmt.Errorf("namespaceLabelKey and podLabelKey must differ (both = %q)", namespaceKey)
	}
	if namespaceKey == deviceLabelKey {
		return fmt.Errorf("namespaceLabelKey must not equal the device label %q", deviceLabelKey)
	}
	if podKey == deviceLabelKey {
		return fmt.Errorf("podLabelKey must not equal the device label %q", deviceLabelKey)
	}
	return nil
}
