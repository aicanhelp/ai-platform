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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"

	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	ctrlconfig "sigs.k8s.io/controller-runtime/pkg/client/config"
	"sigs.k8s.io/controller-runtime/pkg/log"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// NPUExporterSourceType is the registered plugin type identifier for the
// source (matches the YAML "type" field).
const NPUExporterSourceType = "npu-exporter-data-source"

var (
	_ fwkdl.DataSource        = (*DataSource)(nil)
	_ fwkdl.PollingDataSource = (*DataSource)(nil)
)

// DataSource is the polling NPU exporter datasource. One instance is shared
// across all endpoints; scrape results are cached by discovered exporter URL.
type DataSource struct {
	typedName fwkplugin.TypedName
	params    Parameters
	client    *http.Client
	cache     *targetCache
	clock     clock
	resolver  targetResolver
}

// Factory is the registered plugin factory function for NPUExporterSourceType.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params Parameters
	if len(raw) != 0 {
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("npu source %q: invalid parameters: %w", name, err)
		}
	}
	params.applyDefaults()
	if err := params.Validate(); err != nil {
		return nil, fmt.Errorf("npu source %q: %w", name, err)
	}
	kubeConfig, err := ctrlconfig.GetConfig()
	if err != nil {
		return nil, fmt.Errorf("npu source %q: get kube config: %w", name, err)
	}
	kubeClient, err := ctrlclient.New(kubeConfig, ctrlclient.Options{})
	if err != nil {
		return nil, fmt.Errorf("npu source %q: create kube client: %w", name, err)
	}
	return NewDataSource(name, params, kubeClient)
}

// NewDataSource constructs a DataSource instance.
func NewDataSource(name string, params Parameters, kubeClient ctrlclient.Client) (*DataSource, error) {
	resolver, err := newKubeTargetResolver(kubeClient, params)
	if err != nil {
		return nil, err
	}
	return newDataSourceWithResolver(name, params, resolver), nil
}

func newDataSourceWithResolver(name string, params Parameters, resolver targetResolver) *DataSource {
	c := realClock{}
	return &DataSource{
		typedName: fwkplugin.TypedName{Type: NPUExporterSourceType, Name: name},
		params:    params,
		client:    &http.Client{Timeout: params.ScrapeTimeout.Std()},
		cache:     newTargetCache(params.CacheTTL.Std(), c),
		clock:     c,
		resolver:  resolver,
	}
}

// TypedName returns the framework plugin type and instance name.
func (s *DataSource) TypedName() fwkplugin.TypedName { return s.typedName }

// OutputType returns the data source output type.
func (s *DataSource) OutputType() reflect.Type { return PodType() }

// ExtractorType returns the framework extractor type.
func (s *DataSource) ExtractorType() reflect.Type { return fwkdl.ExtractorType }

// Poll discovers the same-node exporter Pod for the target workload Pod,
// scrapes that exporter, and then reuses the existing metric-label aggregation
// to extract the matching pod snapshot. Discovery and scrape failures are
// softened into unmatched snapshots so one failed poll does not fail the whole
// datasource.
func (s *DataSource) Poll(ctx context.Context, ep fwkdl.Endpoint) (any, error) {
	logger := log.FromContext(ctx).WithName("npu-exporter-source")
	metadata := ep.GetMetadata()
	if metadata == nil {
		return Pod{}, nil
	}
	namespace := metadata.NamespacedName.Namespace
	podName := metadata.PodName
	if namespace == "" || podName == "" {
		return Pod{Namespace: namespace, PodName: podName}, nil
	}

	targetURL, err := s.resolver.resolveTargetURL(ctx, namespace, podName)
	if err != nil {
		logger.Error(err, "npu exporter discovery failed", "namespace", namespace, "pod", podName)
		return Pod{Namespace: namespace, PodName: podName}, nil
	}

	entry, err := s.cache.get(ctx, targetURL, func(ctx context.Context) (parsedFamilies, error) {
		return scrapeTarget(ctx, s.client, s.clock, targetURL)
	})
	if err != nil {
		logger.Error(err, "npu exporter scrape failed", "target", targetURL)
		return Pod{Namespace: namespace, PodName: podName}, nil
	}

	return buildPod(s.params, namespace, podName, []targetResult{{
		target:    targetURL,
		families:  entry.families,
		fetchedAt: entry.fetchedAt,
	}}), nil
}
