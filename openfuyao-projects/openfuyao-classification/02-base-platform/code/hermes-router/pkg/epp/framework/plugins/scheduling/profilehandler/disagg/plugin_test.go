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

package disagg

import (
	"context"
	"strings"
	"testing"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	"hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider"
	alwaysdisaggdecider "hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider/alwaysdisagg"
	prefixbaseddecider "hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider/prefixbased"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/testutil"
)

type stubProfile struct{}

func (stubProfile) Run(
	context.Context,
	*fwkscheduling.LLMRequest,
	*fwkscheduling.CycleState,
	[]fwkscheduling.Endpoint,
) (*fwkscheduling.ProfileRunResult, error) {
	return &fwkscheduling.ProfileRunResult{}, nil
}

// alwaysSkipPrefillDecider is a test-only Decider that forces the d path.
type alwaysSkipPrefillDecider struct{}

func (alwaysSkipPrefillDecider) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "test-always-skip-prefill", Name: "force-d"}
}

func (alwaysSkipPrefillDecider) Decide(
	context.Context,
	*fwkscheduling.LLMRequest,
	fwkscheduling.Endpoint,
) (bool, error) {
	return false, nil
}

func newHandlerWithDecider(d decider.Strategy) *Handler {
	handler := New("disagg", Config{PrefillDeciderPluginRef: "unused"}, nil)
	handler.decider = d
	return handler
}

func TestPickSkipsPDOnDPath(t *testing.T) {
	handler := newHandlerWithDecider(alwaysSkipPrefillDecider{})

	profiles := map[string]fwkscheduling.SchedulerProfile{
		"decode": stubProfile{},
		"pd":     stubProfile{},
	}
	cycleState := fwkscheduling.NewCycleState()

	first := handler.Pick(context.Background(), cycleState, nil, profiles, map[string]*fwkscheduling.ProfileRunResult{})
	if len(first) != 1 || first["decode"] == nil {
		t.Fatalf("first Pick should select decode only, got %#v", first)
	}

	leader, _ := newPDLeaderWithDecode(t)
	decodeResult := &fwkscheduling.ProfileRunResult{
		TargetEndpoints: []fwkscheduling.Endpoint{leader},
	}
	second := handler.Pick(
		context.Background(),
		cycleState,
		nil,
		profiles,
		map[string]*fwkscheduling.ProfileRunResult{"decode": decodeResult},
	)
	if len(second) != 0 {
		t.Fatalf("d path must not schedule pd profile, got %#v", second)
	}
}

func TestPickSchedulesPDOnPDPath(t *testing.T) {
	handler := newHandlerWithDecider(alwaysdisaggdecider.New("always"))

	profiles := map[string]fwkscheduling.SchedulerProfile{
		"decode": stubProfile{},
		"pd":     stubProfile{},
	}
	cycleState := fwkscheduling.NewCycleState()
	leader, _ := newPDLeaderWithDecode(t)
	decodeResult := &fwkscheduling.ProfileRunResult{
		TargetEndpoints: []fwkscheduling.Endpoint{leader},
	}

	picked := handler.Pick(
		context.Background(),
		cycleState,
		nil,
		profiles,
		map[string]*fwkscheduling.ProfileRunResult{"decode": decodeResult},
	)
	if len(picked) != 1 || picked["pd"] == nil {
		t.Fatalf("pd path should schedule pd profile, got %#v", picked)
	}
}

func TestProcessResultsDOmitsPD(t *testing.T) {
	handler := newHandlerWithDecider(alwaysSkipPrefillDecider{})

	cycleState := fwkscheduling.NewCycleState()
	leader, _ := newPDLeaderWithDecode(t)
	decodeResult := &fwkscheduling.ProfileRunResult{
		TargetEndpoints: []fwkscheduling.Endpoint{leader},
	}
	_ = handler.Pick(
		context.Background(),
		cycleState,
		nil,
		map[string]fwkscheduling.SchedulerProfile{"decode": stubProfile{}, "pd": stubProfile{}},
		map[string]*fwkscheduling.ProfileRunResult{"decode": decodeResult},
	)

	result, err := handler.ProcessResults(
		context.Background(),
		cycleState,
		nil,
		map[string]*fwkscheduling.ProfileRunResult{"decode": decodeResult},
	)
	if err != nil {
		t.Fatalf("ProcessResults: %v", err)
	}
	if result.PrimaryProfileName != "decode" {
		t.Fatalf("PrimaryProfileName=%q, want decode", result.PrimaryProfileName)
	}
	if result.ProfileResults["pd"] != nil {
		t.Fatal("d path must not include pd profile result")
	}
	if result.ProfileResults["decode"] == nil {
		t.Fatal("d path requires decode profile result")
	}
}

func TestProcessResultsPDOmitsDecode(t *testing.T) {
	handler := New("disagg", Config{PrefillDeciderPluginRef: "unused"}, nil)

	leader, _ := newPDLeaderWithDecode(t)
	decodeResult := &fwkscheduling.ProfileRunResult{
		TargetEndpoints: []fwkscheduling.Endpoint{leader},
	}
	pdResult := &fwkscheduling.ProfileRunResult{
		TargetEndpoints: []fwkscheduling.Endpoint{leader},
	}

	result, err := handler.ProcessResults(
		context.Background(),
		fwkscheduling.NewCycleState(),
		nil,
		map[string]*fwkscheduling.ProfileRunResult{
			"decode": decodeResult,
			"pd":     pdResult,
		},
	)
	if err != nil {
		t.Fatalf("ProcessResults: %v", err)
	}
	if result.PrimaryProfileName != "pd" {
		t.Fatalf("PrimaryProfileName=%q, want pd", result.PrimaryProfileName)
	}
	if result.ProfileResults["pd"] == nil {
		t.Fatal("pd path requires pd profile result")
	}
	if result.ProfileResults["decode"] != nil {
		t.Fatal("pd path must not include intermediate decode profile result")
	}
}

func TestProcessResultsPDSucceedsWhenDecodeFailed(t *testing.T) {
	handler := New("disagg", Config{PrefillDeciderPluginRef: "unused"}, nil)

	leader, _ := newPDLeaderWithDecode(t)
	pdResult := &fwkscheduling.ProfileRunResult{
		TargetEndpoints: []fwkscheduling.Endpoint{leader},
	}

	result, err := handler.ProcessResults(
		context.Background(),
		fwkscheduling.NewCycleState(),
		nil,
		map[string]*fwkscheduling.ProfileRunResult{
			"decode": nil,
			"pd":     pdResult,
		},
	)
	if err != nil {
		t.Fatalf("ProcessResults should accept pd after decode failure: %v", err)
	}
	if result.PrimaryProfileName != "pd" {
		t.Fatalf("PrimaryProfileName=%q, want pd", result.PrimaryProfileName)
	}
	if len(result.ProfileResults) != 1 || result.ProfileResults["pd"] == nil {
		t.Fatalf("expected only pd profile result, got %#v", result.ProfileResults)
	}
}

func TestProcessResultsErrorsWhenPDFailed(t *testing.T) {
	handler := New("disagg", Config{PrefillDeciderPluginRef: "unused"}, nil)

	_, err := handler.ProcessResults(
		context.Background(),
		fwkscheduling.NewCycleState(),
		nil,
		map[string]*fwkscheduling.ProfileRunResult{
			"decode": {TargetEndpoints: []fwkscheduling.Endpoint{}},
			"pd":     nil,
		},
	)
	if err == nil || !strings.Contains(err.Error(), `pd profile "pd" failed`) {
		t.Fatalf("expected pd failure error, got %v", err)
	}
}

func TestProcessResultsErrorsWhenDecodeIncomplete(t *testing.T) {
	handler := New("disagg", Config{PrefillDeciderPluginRef: "unused"}, nil)
	leader := testutil.NewEndpoint("leader-incomplete", "10.0.0.10", "8000", nil)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:     "group-incomplete",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: testutil.NewEndpoint("p", "10.0.0.11", "8000", nil), Score: 0}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: testutil.NewEndpoint("d", "10.0.0.12", "8000", nil), Score: 0}},
	})

	_, err := handler.ProcessResults(
		context.Background(),
		fwkscheduling.NewCycleState(),
		nil,
		map[string]*fwkscheduling.ProfileRunResult{
			"decode": {TargetEndpoints: []fwkscheduling.Endpoint{leader}},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "is incomplete") {
		t.Fatalf("expected incomplete decode error, got %v", err)
	}
}

func TestPickStopsWhenDecodeResultMissingSelection(t *testing.T) {
	handler := newHandlerWithDecider(alwaysSkipPrefillDecider{})

	profiles := map[string]fwkscheduling.SchedulerProfile{
		"decode": stubProfile{},
		"pd":     stubProfile{},
	}
	picked := handler.Pick(
		context.Background(),
		fwkscheduling.NewCycleState(),
		nil,
		profiles,
		map[string]*fwkscheduling.ProfileRunResult{"decode": nil},
	)
	if len(picked) != 0 {
		t.Fatalf("incomplete decode must stop scheduling, got %#v", picked)
	}
}

func TestPrefixBasedDeciderDrivesDSkip(t *testing.T) {
	d := prefixbaseddecider.New("prefix", prefixbaseddecider.Config{
		MinHitRatio:       0.8,
		MaxUncachedTokens: 256,
	})
	handler := newHandlerWithDecider(d)

	leader := testutil.NewEndpoint("leader-a", "10.0.0.10", "8000", nil)
	prefill := testutil.NewEndpoint("prefill-a", "10.0.0.11", "8000", nil)
	decode := testutil.NewEndpoint("decode-a", "10.0.0.12", "8000", nil)
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(9, 10, 16))
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
		SelectedDecodePod: &fwkscheduling.ScoredEndpoint{
			Endpoint: decode,
			Score:    1,
		},
	})

	tokenIDs := make([]uint32, 160)
	for i := range tokenIDs {
		tokenIDs[i] = uint32(i + 1)
	}
	request := &fwkscheduling.LLMRequest{
		TokenizedPrompt: &fwkscheduling.TokenizedPrompt{TokenIDs: tokenIDs},
	}

	picked := handler.Pick(
		context.Background(),
		fwkscheduling.NewCycleState(),
		request,
		map[string]fwkscheduling.SchedulerProfile{"decode": stubProfile{}, "pd": stubProfile{}},
		map[string]*fwkscheduling.ProfileRunResult{
			"decode": {TargetEndpoints: []fwkscheduling.Endpoint{leader}},
		},
	)
	if len(picked) != 0 {
		t.Fatalf("prefix-based d decision must skip pd scheduling, got %#v", picked)
	}
}

func TestFactoryRejectsUnknownPrefillProfileName(t *testing.T) {
	_, err := Factory(
		"disagg",
		[]byte(`{"prefillDeciderPluginRef":"decider","prefillProfileName":"prefill"}`),
		nil,
	)
	if err == nil {
		t.Fatal("expected unknown field error for prefillProfileName")
	}
}

func TestFactoryRequiresPrefillDeciderPluginRef(t *testing.T) {
	_, err := Factory("disagg", []byte(`{}`), nil)
	if err == nil || !strings.Contains(err.Error(), "prefillDeciderPluginRef is required") {
		t.Fatalf("expected missing ref error, got %v", err)
	}
}

func TestFactoryAndTypedName(t *testing.T) {
	plugin, err := Factory(
		"disagg",
		[]byte(`{"prefillDeciderPluginRef":"prefix-decider"}`),
		nil,
	)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	handler, ok := plugin.(*Handler)
	if !ok {
		t.Fatalf("unexpected type %T", plugin)
	}
	if got := handler.TypedName(); got.Type != PluginType || got.Name != "disagg" {
		t.Fatalf("TypedName=%+v, want type=%q name=disagg", got, PluginType)
	}
	if handler.config.PrefillDeciderPluginRef != "prefix-decider" {
		t.Fatalf("PrefillDeciderPluginRef=%q", handler.config.PrefillDeciderPluginRef)
	}
	if handler.config.DecodeProfileName != "decode" || handler.config.PDProfileName != "pd" {
		t.Fatalf("unexpected defaults %#v", handler.config)
	}
}

func newPDLeaderWithDecode(t *testing.T) (fwkscheduling.Endpoint, fwkscheduling.Endpoint) {
	t.Helper()
	leader := testutil.NewEndpoint("leader-a", "10.0.0.10", "8000", nil)
	prefill := testutil.NewEndpoint("prefill-a", "10.0.0.11", "8000", nil)
	decode := testutil.NewEndpoint("decode-a", "10.0.0.12", "8000", nil)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
		SelectedDecodePod: &fwkscheduling.ScoredEndpoint{
			Endpoint: decode,
			Score:    1,
		},
	})
	return leader, decode
}
