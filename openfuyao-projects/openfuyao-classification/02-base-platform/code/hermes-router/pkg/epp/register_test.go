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

package epp

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrequestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requestcontrol/requestdataproducer/inflightload"

	extractorNPU "hermes-router/pkg/epp/framework/plugins/datalayer/extractor/npu"
	sourceNPU "hermes-router/pkg/epp/framework/plugins/datalayer/source/npu"
	inflightproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflight"
	pdgroupproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/pdgroup"
	predictionfeatureextractor "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictionfeature"
	prefixcacheproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixcache"
	tokenizerproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	requestlifecycle "hermes-router/pkg/epp/framework/plugins/requestcontrol/lifecycle"
	requestprerequest "hermes-router/pkg/epp/framework/plugins/requestcontrol/prerequest"
	pdgroupfilter "hermes-router/pkg/epp/framework/plugins/scheduling/filter/pdgroup"
	prefixcachefilter "hermes-router/pkg/epp/framework/plugins/scheduling/filter/prefixcache"
	maxscorepicker "hermes-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	minscorepicker "hermes-router/pkg/epp/framework/plugins/scheduling/picker/minscore"
	randompicker "hermes-router/pkg/epp/framework/plugins/scheduling/picker/random"
	softmaxpotcpicker "hermes-router/pkg/epp/framework/plugins/scheduling/picker/softmaxpotc"
	alwaysdisaggdecider "hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider/alwaysdisagg"
	prefixbaseddecider "hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider/prefixbased"
	disaggprofilehandler "hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	autokvawarescorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/autokvaware"
	bucketscorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/bucket"
	decodekvawarescorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/decodekvaware"
	kvawarescorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/kvaware"
	pdkvawarescorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/pdkvaware"
	predictionscorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/prediction"

	"hermes-router/pkg/epp/internal/pdgroup"
)

func TestRegisterAllPluginsRegistersFactories(t *testing.T) {
	original := make(map[string]fwkplugin.FactoryFunc, len(fwkplugin.Registry))
	for pluginType, factory := range fwkplugin.Registry {
		original[pluginType] = factory
		delete(fwkplugin.Registry, pluginType)
	}
	t.Cleanup(func() {
		for pluginType := range fwkplugin.Registry {
			delete(fwkplugin.Registry, pluginType)
		}
		for pluginType, factory := range original {
			fwkplugin.Registry[pluginType] = factory
		}
	})

	RegisterAllPlugins()

	expected := map[string]fwkplugin.FactoryFunc{
		kvawarescorer.PluginType:              kvawarescorer.Factory,
		pdkvawarescorer.PluginType:            pdkvawarescorer.Factory,
		decodekvawarescorer.PluginType:       decodekvawarescorer.Factory,
		autokvawarescorer.PluginType:          autokvawarescorer.Factory,
		pdgroupfilter.PluginType:              pdgroupfilter.Factory,
		prefixcachefilter.PluginType:          prefixcachefilter.Factory,
		inflightproducer.PluginType:           inflightproducer.Factory,
		pdgroupproducer.PluginType:            pdgroupproducer.Factory,
		predictionfeatureextractor.PluginType: predictionfeatureextractor.Factory,
		predictionscorer.PluginType:           predictionscorer.Factory,
		prefixcacheproducer.PluginType:        prefixcacheproducer.Factory,
		randompicker.PluginType:               randompicker.Factory,
		minscorepicker.PluginType:             minscorepicker.Factory,
		maxscorepicker.PluginType:             maxscorepicker.Factory,
		softmaxpotcpicker.PluginType:          softmaxpotcpicker.Factory,
		alwaysdisaggdecider.PluginType:        alwaysdisaggdecider.Factory,
		prefixbaseddecider.PluginType:         prefixbaseddecider.Factory,
		disaggprofilehandler.PluginType:       disaggprofilehandler.Factory,
		tokenizerproducer.PluginType:          tokenizerproducer.Factory,
		requestlifecycle.PluginType:           requestlifecycle.Factory,
		requestprerequest.HeaderHandlerType:   requestprerequest.HeaderHandlerFactory,
		bucketscorer.PDBucketType:             bucketscorer.PDBucketFactory,
		sourceNPU.NPUExporterSourceType:       sourceNPU.Factory,
		extractorNPU.NPUExporterExtractorType: extractorNPU.Factory,
	}

	if got, want := len(fwkplugin.Registry), len(expected); got != want {
		t.Fatalf("expected %d registered plugin types, got %d", want, got)
	}

	for pluginType, factory := range expected {
		registered, ok := fwkplugin.Registry[pluginType]
		if !ok {
			t.Fatalf("expected plugin type %q to be registered", pluginType)
		}
		if reflect.ValueOf(registered).Pointer() != reflect.ValueOf(factory).Pointer() {
			t.Fatalf("expected plugin type %q to register the correct factory", pluginType)
		}
	}
}

func TestRegisterAllPluginsBuildsExplicitKVScorerTypes(t *testing.T) {
	original := make(map[string]fwkplugin.FactoryFunc, len(fwkplugin.Registry))
	for pluginType, factory := range fwkplugin.Registry {
		original[pluginType] = factory
		delete(fwkplugin.Registry, pluginType)
	}
	t.Cleanup(func() {
		for pluginType := range fwkplugin.Registry {
			delete(fwkplugin.Registry, pluginType)
		}
		for pluginType, factory := range original {
			fwkplugin.Registry[pluginType] = factory
		}
	})

	RegisterAllPlugins()

	testCases := []struct {
		pluginType string
		wantType   string
		parameters json.RawMessage
	}{
		{pluginType: kvawarescorer.PluginType, wantType: kvawarescorer.PluginType, parameters: json.RawMessage(`{}`)},
		{
			pluginType: pdkvawarescorer.PluginType,
			wantType:   pdkvawarescorer.PluginType,
			parameters: json.RawMessage(`{"prefillScoreWeight":1,"decodeScoreWeight":1}`),
		},
		{
			pluginType: decodekvawarescorer.PluginType,
			wantType:   decodekvawarescorer.PluginType,
			parameters: json.RawMessage(`{}`),
		},
		{pluginType: autokvawarescorer.PluginType, wantType: autokvawarescorer.PluginType, parameters: json.RawMessage(`{}`)},
	}

	for _, tc := range testCases {
		factory, ok := fwkplugin.Registry[tc.pluginType]
		if !ok {
			t.Fatalf("expected plugin type %q to be registered", tc.pluginType)
		}
		plugin, err := factory("configured", tc.parameters, fwkplugin.NewEppHandle(context.Background(), nil))
		if err != nil {
			t.Fatalf("unexpected factory error for plugin type %q: %v", tc.pluginType, err)
		}
		if got := plugin.TypedName().Type; got != tc.wantType {
			t.Fatalf("expected plugin type %q to build canonical type %q, got %q", tc.pluginType, tc.wantType, got)
		}
	}
}

func TestRegisterAllPluginsUsesUnifiedPickerRegistrations(t *testing.T) {
	original := make(map[string]fwkplugin.FactoryFunc, len(fwkplugin.Registry))
	for pluginType, factory := range fwkplugin.Registry {
		original[pluginType] = factory
		delete(fwkplugin.Registry, pluginType)
	}
	t.Cleanup(func() {
		for pluginType := range fwkplugin.Registry {
			delete(fwkplugin.Registry, pluginType)
		}
		for pluginType, factory := range original {
			fwkplugin.Registry[pluginType] = factory
		}
	})

	RegisterAllPlugins()

	for _, legacy := range []string{
		"min-random-picker",
		"pd-softmaxpotc-picker",
		"pd-random-picker",
		"pd-min-score-picker",
	} {
		if _, ok := fwkplugin.Registry[legacy]; ok {
			t.Fatalf("expected legacy plugin type %q to be removed", legacy)
		}
	}

	for _, unified := range []string{
		randompicker.PluginType,
		minscorepicker.PluginType,
		maxscorepicker.PluginType,
		softmaxpotcpicker.PluginType,
	} {
		if _, ok := fwkplugin.Registry[unified]; !ok {
			t.Fatalf("expected unified plugin type %q to be registered", unified)
		}
	}
}

func TestRegisterAllPluginsBuildsUnifiedSoftmaxPicker(t *testing.T) {
	original := make(map[string]fwkplugin.FactoryFunc, len(fwkplugin.Registry))
	for pluginType, factory := range fwkplugin.Registry {
		original[pluginType] = factory
		delete(fwkplugin.Registry, pluginType)
	}
	t.Cleanup(func() {
		for pluginType := range fwkplugin.Registry {
			delete(fwkplugin.Registry, pluginType)
		}
		for pluginType, factory := range original {
			fwkplugin.Registry[pluginType] = factory
		}
	})

	RegisterAllPlugins()

	factory, ok := fwkplugin.Registry[softmaxpotcpicker.PluginType]
	if !ok {
		t.Fatalf("expected %q to be registered", softmaxpotcpicker.PluginType)
	}

	plugin, err := factory(
		"softmax",
		json.RawMessage(`{"temperature":1.0}`),
		fwkplugin.NewEppHandle(context.Background(), nil),
	)
	if err != nil {
		t.Fatalf("unexpected factory error: %v", err)
	}
	picker, ok := plugin.(fwkscheduling.Picker)
	if !ok {
		t.Fatalf("expected scheduling.Picker, got %T", plugin)
	}
	if got := plugin.TypedName().Type; got != softmaxpotcpicker.PluginType {
		t.Fatalf("expected plugin type %q, got %q", softmaxpotcpicker.PluginType, got)
	}

	invalidLeader := newRegisterEndpoint("leader-invalid")
	validLeader := newRegisterEndpoint("leader-valid")
	pdgroup.Set(invalidLeader, &pdgroup.Info{GroupID: "invalid"})
	pdgroup.Set(validLeader, &pdgroup.Info{
		GroupID:            "group-a",
		PrefillPods:        []fwkscheduling.ScoredEndpoint{{Endpoint: newRegisterEndpoint("prefill-a"), Score: 0.2}},
		DecodePods:         []fwkscheduling.ScoredEndpoint{{Endpoint: newRegisterEndpoint("decode-a"), Score: 0.3}},
		SelectedPrefillPod: &fwkscheduling.ScoredEndpoint{Endpoint: newRegisterEndpoint("prefill-a-selected"), Score: 0.2},
		SelectedDecodePod:  &fwkscheduling.ScoredEndpoint{Endpoint: newRegisterEndpoint("decode-a-selected"), Score: 0.3},
	})

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{
		{Endpoint: invalidLeader, Score: 0.9},
		{Endpoint: validLeader, Score: 0.1},
	})
	if result == nil || len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected one selected leader, got %+v", result)
	}
	if !fwkscheduling.EndpointComparer(result.TargetEndpoints[0], validLeader) {
		t.Fatal("expected only selectable PD group leader to be returned")
	}
}

func TestRegisterAllPluginsValidatesUnifiedSoftmaxTemperature(t *testing.T) {
	original := make(map[string]fwkplugin.FactoryFunc, len(fwkplugin.Registry))
	for pluginType, factory := range fwkplugin.Registry {
		original[pluginType] = factory
		delete(fwkplugin.Registry, pluginType)
	}
	t.Cleanup(func() {
		for pluginType := range fwkplugin.Registry {
			delete(fwkplugin.Registry, pluginType)
		}
		for pluginType, factory := range original {
			fwkplugin.Registry[pluginType] = factory
		}
	})

	RegisterAllPlugins()

	factory, ok := fwkplugin.Registry[softmaxpotcpicker.PluginType]
	if !ok {
		t.Fatalf("expected %q to be registered", softmaxpotcpicker.PluginType)
	}

	_, err := factory("softmax", json.RawMessage(`{"temperature":0}`), fwkplugin.NewEppHandle(context.Background(), nil))
	if err == nil || !strings.Contains(err.Error(), "temperature must be greater than 0") {
		t.Fatalf("expected temperature validation error, got %v", err)
	}
}

func TestRegisterAllPluginsKeepsHermesInflightFactoryDistinctFromInTreeRegistration(t *testing.T) {
	original := make(map[string]fwkplugin.FactoryFunc, len(fwkplugin.Registry))
	for pluginType, factory := range fwkplugin.Registry {
		original[pluginType] = factory
		delete(fwkplugin.Registry, pluginType)
	}
	t.Cleanup(func() {
		for pluginType := range fwkplugin.Registry {
			delete(fwkplugin.Registry, pluginType)
		}
		for pluginType, factory := range original {
			fwkplugin.Registry[pluginType] = factory
		}
	})

	RegisterAllPlugins()
	fwkplugin.Register(inflightload.InFlightLoadProducerType, inflightload.InFlightLoadProducerFactory)

	hermesFactory, ok := fwkplugin.Registry[inflightproducer.PluginType]
	if !ok {
		t.Fatalf("expected Hermes inflight plugin type %q to remain registered", inflightproducer.PluginType)
	}
	if reflect.ValueOf(hermesFactory).Pointer() != reflect.ValueOf(inflightproducer.Factory).Pointer() {
		t.Fatalf("expected Hermes inflight plugin type %q to keep its own factory", inflightproducer.PluginType)
	}

	builtInFactory, ok := fwkplugin.Registry[inflightload.InFlightLoadProducerType]
	if !ok {
		t.Fatalf("expected built-in inflight plugin type %q to be registered", inflightload.InFlightLoadProducerType)
	}
	if reflect.ValueOf(builtInFactory).Pointer() !=
		reflect.ValueOf(inflightload.InFlightLoadProducerFactory).Pointer() {
		t.Fatalf(
			"expected built-in inflight plugin type %q to keep the in-tree factory",
			inflightload.InFlightLoadProducerType,
		)
	}
	if inflightproducer.PluginType == inflightload.InFlightLoadProducerType {
		t.Fatalf(
			"expected Hermes inflight plugin type %q to differ from in-tree type %q",
			inflightproducer.PluginType,
			inflightload.InFlightLoadProducerType,
		)
	}

	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	hermesPlugin, err := hermesFactory("hermes", nil, handle)
	if err != nil {
		t.Fatalf("unexpected Hermes inflight factory error: %v", err)
	}
	if _, ok := hermesPlugin.(fwkrequestcontrol.PreRequest); ok {
		t.Fatalf("expected Hermes inflight plugin to remain PrepareData-only, got %T", hermesPlugin)
	}

	builtInPlugin, err := builtInFactory("builtin", nil, handle)
	if err != nil {
		t.Fatalf("unexpected in-tree inflight factory error: %v", err)
	}
	if _, ok := builtInPlugin.(fwkrequestcontrol.PreRequest); !ok {
		t.Fatalf("expected in-tree inflight plugin to expose PreRequest, got %T", builtInPlugin)
	}
}

func newRegisterEndpoint(name string) fwkscheduling.Endpoint {
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Address:        name + ".svc.cluster.local",
		Port:           "8080",
	}
	return fwkscheduling.NewEndpoint(metadata, fwkdl.NewMetrics(), fwkdl.NewAttributes())
}
