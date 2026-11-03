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

// Package prefixcache provides prefix-cache request data and scheduling filters.
package prefixcache

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	tokenizedpromptattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/tokenizedprompt"
	"hermes-router/pkg/epp/internal/endpointmeta"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PluginType is the registered plugin type identifier.
	PluginType                 = "prefix-cache-producer"
	defaultBlockSize           = 128
	defaultTimeout             = 3 * time.Second
	defaultL1Weight            = 0.7
	defaultL3Weight            = 0.3
	cacheIndexerHitRateAPIPath = "/kv-cache/hit-rate"
	maxHTTPErrorBodyBytes      = 4 << 10
)

type httpClient interface {
	Do(request *http.Request) (*http.Response, error)
}

type pluginParameters struct {
	CacheIndexerAddress string                            `json:"cacheIndexerAddress"`
	BlockSize           *int                              `json:"blockSize,omitempty"`
	Timeout             *utils.Duration                   `json:"timeout,omitempty"`
	L1Weight            *float64                          `json:"l1Weight,omitempty"`
	L3Weight            *float64                          `json:"l3Weight,omitempty"`
	TargetPodSelector   *endpointmeta.LabelSelectorConfig `json:"targetPodSelector,omitempty"`
	TargetPodRoleLabel  string                            `json:"targetPodRoleLabel,omitempty"`
	TargetPodRoles      []string                          `json:"targetPodRoles,omitempty"`
}

// Config controls runtime access to cache-indexer.
type Config struct {
	// CacheIndexerAddress stores the CacheIndexerAddress value.
	CacheIndexerAddress string
	// BlockSize stores the BlockSize value.
	BlockSize int
	// Timeout stores the Timeout value.
	Timeout time.Duration
	// L1Weight stores the L1Weight value.
	L1Weight float64
	// L3Weight stores the L3Weight value.
	L3Weight float64
	// TargetPodSelector stores the TargetPodSelector value.
	TargetPodSelector *endpointmeta.LabelSelectorConfig
	// TargetPodRoleLabel stores the TargetPodRoleLabel value.
	TargetPodRoleLabel string
	// TargetPodRoles stores the TargetPodRoles value.
	TargetPodRoles []string
	// HTTPClient stores the HTTPClient value.
	HTTPClient httpClient
}

// Producer enriches candidate endpoints with prefix cache match info.
type Producer struct {
	typedName       fwkplugin.TypedName
	client          httpClient
	cacheIndexerURL *url.URL
	blockSize       int
	timeout         time.Duration
	l1Weight        float64
	l3Weight        float64
	selector        endpointmeta.LabelSelector
	hasSelector     bool
}

var _ requestcontrol.PrepareDataPlugin = (*Producer)(nil)

// Factory constructs a prefix cache producer with validated parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("prefix cache producer %q: %w", name, err)
	}
	return newProducer(name, config)
}

// New builds a producer from runtime config.
func New(name string, config Config) *Producer {
	producer, err := newProducer(name, config)
	if err != nil {
		panic(fmt.Sprintf("prefix cache producer %q: %v", name, err))
	}
	return producer
}

func newProducer(name string, config Config) (*Producer, error) {
	resolved := resolveConfig(config)
	selector, hasSelector, err := endpointmeta.CompileProducerTargetSelector(
		resolved.TargetPodSelector,
		resolved.TargetPodRoleLabel,
		resolved.TargetPodRoles,
	)
	if err != nil {
		return nil, err
	}
	return &Producer{
		typedName:       fwkplugin.TypedName{Type: PluginType, Name: name},
		client:          resolved.HTTPClient,
		cacheIndexerURL: parseCacheIndexerURL(resolved.CacheIndexerAddress),
		blockSize:       resolved.BlockSize,
		timeout:         resolved.Timeout,
		l1Weight:        resolved.L1Weight,
		l3Weight:        resolved.L3Weight,
		selector:        selector,
		hasSelector:     hasSelector,
	}, nil
}

// TypedName returns the plugin identity.
func (p *Producer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the prefix cache match info contract.
func (p *Producer) Produces() map[string]any {
	return map[string]any{
		prefixattr.PrefixCacheMatchInfoKey: prefixattr.PrefixCacheMatchInfo{},
	}
}

// Consumes declares the tokenized prompt contract required for cache lookup.
func (p *Producer) Consumes() map[string]any {
	return map[string]any{
		tokenizedpromptattr.TokenizedPromptKey: tokenizedpromptattr.Info{},
	}
}

// PrepareRequestData queries cache-indexer and publishes per-endpoint cache match info.
func (p *Producer) PrepareRequestData(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	endpoints []fwkscheduling.Endpoint,
) error {
	if ctx == nil {
		ctx = context.Background()
	}
	logger := log.FromContext(ctx).WithName("Producer.PrepareRequestData")
	start := time.Now()
	if p == nil || p.client == nil || p.cacheIndexerURL == nil {
		return nil
	}
	tokenIDs, ok := tokenIDsFromRequest(request)
	if !ok {
		logger.V(logutil.DEBUG).Info(
			"Prefix cache producer skipped cache lookup",
			"reason", "missing_tokenized_prompt",
			"durationMs", utils.DurationMillisecondsSince(start),
		)
		return nil
	}
	addresses := make([]string, 0, len(endpoints))
	endpointByIP := make(map[string][]fwkscheduling.Endpoint, len(endpoints))
	for _, endpoint := range endpoints {
		if endpoint == nil || endpoint.GetMetadata() == nil || endpoint.GetMetadata().Address == "" {
			continue
		}
		if p.hasSelector && !p.selector.Matches(endpoint) {
			continue
		}
		address := endpoint.GetMetadata().Address
		if _, seen := endpointByIP[address]; !seen {
			addresses = append(addresses, address)
		}
		endpointByIP[address] = append(endpointByIP[address], endpoint)
	}
	if len(addresses) == 0 {
		logger.V(logutil.DEBUG).Info(
			"Prefix cache producer skipped cache lookup",
			"reason", "no_queryable_endpoints",
			"durationMs", utils.DurationMillisecondsSince(start),
		)
		return nil
	}
	totalBlocks := len(tokenIDs) / p.blockSize
	if totalBlocks == 0 {
		logger.V(logutil.DEBUG).Info(
			"Prefix cache producer skipped cache lookup",
			"reason", "zero_total_blocks",
			"durationMs", utils.DurationMillisecondsSince(start),
		)
		return nil
	}
	payload := hitRateRequest{
		ServerIP: addresses,
		Body: hitRateRequestBody{
			TokenIDs:  tokenIDs,
			CacheSalt: "",
			BlockSize: p.blockSize,
		},
	}
	response, err := p.fetchHitRates(ctx, payload)
	if err != nil {
		logger.Error(
			err,
			"Prefix cache producer failed to publish cache match info; continuing without prefix cache data",
			"cacheIndexerAddress", p.cacheIndexerURL.String(),
			"durationMs", utils.DurationMillisecondsSince(start),
		)
		return nil
	}

	for _, score := range response.ServerScoreList {
		logger.V(logutil.DEBUG).Info(
			"Prefix cache producer received cache-indexer score",
			"serverIP", score.ServerIP,
			"l1HitRatio", clamp(score.L1HitRatio),
			"l3HitRatio", clamp(score.L3HitRatio),
		)
		matchedEndpoints := endpointByIP[score.ServerIP]
		if len(matchedEndpoints) == 0 {
			continue
		}
		combined := p.l1Weight*clamp(score.L1HitRatio) + p.l3Weight*clamp(score.L3HitRatio)
		matchBlocks := int(math.Round(float64(totalBlocks) * combined))
		for _, endpoint := range matchedEndpoints {
			endpoint.Put(
				prefixattr.PrefixCacheMatchInfoKey,
				prefixattr.NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, p.blockSize),
			)
		}
	}
	logger.V(logutil.DEBUG).Info(
		"Prefix cache match info published to matched endpoints",
		"cacheIndexerScoreCount", len(response.ServerScoreList),
		"durationMs", utils.DurationMillisecondsSince(start),
	)
	return nil
}

func (p *Producer) fetchHitRates(ctx context.Context, payload hitRateRequest) (*hitRateResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	requestCtx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal cache-indexer request: %w", err)
	}
	request, err := http.NewRequestWithContext(
		requestCtx,
		http.MethodPost,
		p.cacheIndexerURL.String(),
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("build cache-indexer request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("call cache-indexer: %w", err)
	}
	defer func() {
		if err := drainAndCloseBody(response.Body); err != nil {
			return
		}
	}()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		bodyPreview, err := io.ReadAll(io.LimitReader(response.Body, maxHTTPErrorBodyBytes))
		if err != nil {
			return nil, fmt.Errorf("cache-indexer returned HTTP %d and body read failed: %w", response.StatusCode, err)
		}
		trimmedPreview := strings.TrimSpace(string(bodyPreview))
		if trimmedPreview == "" {
			return nil, fmt.Errorf("cache-indexer returned HTTP %d", response.StatusCode)
		}
		return nil, fmt.Errorf("cache-indexer returned HTTP %d: %s", response.StatusCode, trimmedPreview)
	}

	decoded := &hitRateResponse{}
	if err := json.NewDecoder(response.Body).Decode(decoded); err != nil {
		return nil, fmt.Errorf("decode cache-indexer response: %w", err)
	}
	if decoded.Status != 0 {
		if strings.TrimSpace(decoded.Message) == "" {
			return nil, fmt.Errorf("cache-indexer returned status %d", decoded.Status)
		}
		return nil, fmt.Errorf("cache-indexer returned status %d: %s", decoded.Status, strings.TrimSpace(decoded.Message))
	}
	return decoded, nil
}

func drainAndCloseBody(body io.ReadCloser) error {
	if body == nil {
		return nil
	}
	_, drainErr := io.Copy(io.Discard, body)
	closeErr := body.Close()
	if drainErr != nil {
		return drainErr
	}
	return closeErr
}

func parseAndValidateParameters(raw json.RawMessage) (Config, error) {
	params, err := decodeParameters(raw)
	if err != nil {
		return Config{}, err
	}
	if err := validateCacheIndexerAddress(params.CacheIndexerAddress); err != nil {
		return Config{}, err
	}

	config := defaultConfig(params.CacheIndexerAddress)
	if err := applyOptionalParameters(&config, params); err != nil {
		return Config{}, err
	}
	if err := applyTargetSelector(&config, params); err != nil {
		return Config{}, err
	}

	return resolveConfig(config), nil
}

func decodeParameters(raw json.RawMessage) (pluginParameters, error) {
	if len(raw) == 0 {
		return pluginParameters{}, fmt.Errorf("cacheIndexerAddress is required")
	}
	params := pluginParameters{}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&params); err != nil {
		return pluginParameters{}, fmt.Errorf("invalid parameters: %w", err)
	}
	return params, nil
}

func validateCacheIndexerAddress(address string) error {
	if strings.TrimSpace(address) == "" {
		return fmt.Errorf("cacheIndexerAddress is required")
	}
	parsedURL, err := url.Parse(address)
	if err != nil || parsedURL.Scheme == "" || parsedURL.Host == "" {
		return fmt.Errorf("cacheIndexerAddress must be a valid URL with scheme")
	}
	return nil
}

func defaultConfig(cacheIndexerAddress string) Config {
	return Config{
		CacheIndexerAddress: cacheIndexerAddress,
		BlockSize:           defaultBlockSize,
		Timeout:             defaultTimeout,
		L1Weight:            defaultL1Weight,
		L3Weight:            defaultL3Weight,
	}
}

func applyOptionalParameters(config *Config, params pluginParameters) error {
	if params.BlockSize != nil {
		if *params.BlockSize <= 0 {
			return fmt.Errorf("blockSize must be greater than 0")
		}
		config.BlockSize = *params.BlockSize
	}
	if params.Timeout != nil {
		if params.Timeout.Std() <= 0 {
			return fmt.Errorf("timeout must be greater than 0")
		}
		config.Timeout = params.Timeout.Std()
	}
	if params.L1Weight != nil {
		if *params.L1Weight < 0 {
			return fmt.Errorf("l1Weight must be greater than or equal to 0")
		}
		config.L1Weight = *params.L1Weight
	}
	if params.L3Weight != nil {
		if *params.L3Weight < 0 {
			return fmt.Errorf("l3Weight must be greater than or equal to 0")
		}
		config.L3Weight = *params.L3Weight
	}
	if config.L1Weight == 0 && config.L3Weight == 0 {
		return fmt.Errorf("at least one of l1Weight or l3Weight must be greater than 0")
	}
	return nil
}

func applyTargetSelector(config *Config, params pluginParameters) error {
	config.TargetPodSelector = params.TargetPodSelector
	config.TargetPodRoleLabel = strings.TrimSpace(params.TargetPodRoleLabel)
	config.TargetPodRoles = endpointmeta.NormalizeLabelValues(params.TargetPodRoles)
	if _, _, err := endpointmeta.CompileProducerTargetSelector(
		config.TargetPodSelector,
		config.TargetPodRoleLabel,
		config.TargetPodRoles,
	); err != nil {
		return err
	}
	return nil
}

func resolveConfig(config Config) Config {
	resolved := config
	if resolved.BlockSize <= 0 {
		resolved.BlockSize = defaultBlockSize
	}
	if resolved.Timeout <= 0 {
		resolved.Timeout = defaultTimeout
	}
	if resolved.L1Weight == 0 && resolved.L3Weight == 0 {
		resolved.L1Weight = defaultL1Weight
		resolved.L3Weight = defaultL3Weight
	}
	resolved.TargetPodSelector = config.TargetPodSelector
	resolved.TargetPodRoleLabel = strings.TrimSpace(resolved.TargetPodRoleLabel)
	resolved.TargetPodRoles = endpointmeta.NormalizeLabelValues(resolved.TargetPodRoles)
	resolved.L1Weight, resolved.L3Weight = normalizeWeights(resolved.L1Weight, resolved.L3Weight)
	if resolved.HTTPClient == nil {
		resolved.HTTPClient = &http.Client{}
	}
	return resolved
}

func normalizeWeights(l1Weight, l3Weight float64) (float64, float64) {
	total := l1Weight + l3Weight
	if total <= 0 {
		return defaultL1Weight, defaultL3Weight
	}
	return l1Weight / total, l3Weight / total
}

func tokenIDsFromRequest(request *fwkscheduling.LLMRequest) ([]uint32, bool) {
	if request == nil || request.TokenizedPrompt == nil || len(request.TokenizedPrompt.TokenIDs) == 0 {
		return nil, false
	}
	return append([]uint32(nil), request.TokenizedPrompt.TokenIDs...), true
}

func parseCacheIndexerURL(raw string) *url.URL {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return nil
	}
	resolved := parsed.ResolveReference(&url.URL{Path: cacheIndexerHitRateAPIPath})
	return resolved
}

func clamp(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}
