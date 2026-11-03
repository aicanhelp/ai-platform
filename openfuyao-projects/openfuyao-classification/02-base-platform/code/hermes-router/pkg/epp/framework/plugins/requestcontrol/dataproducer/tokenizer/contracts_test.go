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

package tokenizer

import (
	"testing"

	datalayer "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"

	tokenizedpromptattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/tokenizedprompt"
	inflightproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflight"
	predictionfeatureextractor "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictionfeature"
	prefixcacheproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixcache"
	internalinflight "hermes-router/pkg/epp/internal/inflight"
)

func TestProducesAndConsumesDeclarations(t *testing.T) {
	plugin := &Plugin{typedName: fwkplugin.TypedName{Type: PluginType, Name: "tokenizer"}}

	produces := plugin.Produces()
	if len(produces) != 1 {
		t.Fatalf("expected one produced key, got %d", len(produces))
	}
	value, ok := produces[tokenizedpromptattr.TokenizedPromptKey]
	if !ok {
		t.Fatalf("expected produced key %q", tokenizedpromptattr.TokenizedPromptKey)
	}
	if _, ok := value.(tokenizedpromptattr.Info); !ok {
		t.Fatalf("expected produced value to be TokenizedPromptInfo, got %T", value)
	}
	if len(plugin.Consumes()) != 0 {
		t.Fatalf("expected no consumed keys, got %d", len(plugin.Consumes()))
	}
}

func TestValidateAndOrderDataDependenciesOrdersTokenizerBeforeConsumers(t *testing.T) {
	tokenizerPlugin := &Plugin{typedName: fwkplugin.TypedName{Type: PluginType, Name: "tokenizer"}}
	inflightPlugin := inflightproducer.NewLoadProducer("inflight", internalinflight.StoreForName("dag-order-test"), "")
	prefixCachePlugin := prefixcacheproducer.New("prefix-cache", prefixcacheproducer.Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
	})
	extractorPlugin, err := predictionfeatureextractor.New("prediction-features", predictionfeatureextractor.Config{})
	if err != nil {
		t.Fatalf("failed to construct feature extractor: %v", err)
	}

	ordered, err := datalayer.ValidateAndOrderDataDependencies([]fwkplugin.Plugin{
		extractorPlugin,
		prefixCachePlugin,
		inflightPlugin,
		tokenizerPlugin,
	})
	if err != nil {
		t.Fatalf("ValidateAndOrderDataDependencies() error = %v", err)
	}

	// Tokenizer feeds prefix-cache directly and the feature extractor indirectly.
	mustAppearBefore(t, ordered, tokenizerPlugin.TypedName().String(), prefixCachePlugin.TypedName().String())
	mustAppearBefore(t, ordered, tokenizerPlugin.TypedName().String(), extractorPlugin.TypedName().String())

	// The extractor is the single owner of feature assembly. Every raw-signal
	// producer must run before it so the serialized input snapshot is complete.
	mustAppearBefore(t, ordered, inflightPlugin.TypedName().String(), extractorPlugin.TypedName().String())
	mustAppearBefore(t, ordered, prefixCachePlugin.TypedName().String(), extractorPlugin.TypedName().String())
}

func mustAppearBefore(t *testing.T, ordered []string, first, second string) {
	t.Helper()
	firstIndex := -1
	secondIndex := -1
	for index, name := range ordered {
		if name == first {
			firstIndex = index
		}
		if name == second {
			secondIndex = index
		}
	}
	if firstIndex == -1 || secondIndex == -1 {
		t.Fatalf("ordered plugin list %v missing %q or %q", ordered, first, second)
	}
	if firstIndex >= secondIndex {
		t.Fatalf("expected %q before %q, got %v", first, second, ordered)
	}
}
