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

package minscore

import (
	"context"
	"encoding/json"
	"math/rand"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestFactory(t *testing.T) {
	plugin, err := Factory("min-score", json.RawMessage(`{}`), fwkplugin.NewEppHandle(context.Background(), nil))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	picker, ok := plugin.(*Picker)
	if !ok {
		t.Fatalf("expected *Picker, got %T", plugin)
	}
	if picker.TypedName().Type != PluginType {
		t.Fatalf("expected plugin type %q, got %q", PluginType, picker.TypedName().Type)
	}
}

func TestFactoryRejectsInvalidParameters(t *testing.T) {
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	if _, err := Factory("min-score", json.RawMessage(`null`), handle); err == nil {
		t.Fatal("expected null parameters error")
	}
	if _, err := Factory("min-score", json.RawMessage(`{"unexpected":1}`), handle); err == nil {
		t.Fatal("expected unknown field error")
	}
}

func TestPickChoosesOneLowestScoreCandidate(t *testing.T) {
	picker := &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: "min-score"},
		rng:       rand.New(rand.NewSource(1)),
	}
	first := newMinScoreTestScoredPod("pod-a", 1.2)
	second := newMinScoreTestScoredPod("pod-b", 0.4)
	third := newMinScoreTestScoredPod("pod-c", 0.8)

	result := picker.Pick(
		context.Background(),
		fwkscheduling.NewCycleState(),
		[]*fwkscheduling.ScoredEndpoint{first, second, third},
	)
	if len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected one selected endpoint, got %d", len(result.TargetEndpoints))
	}
	if !fwkscheduling.EndpointComparer(result.TargetEndpoints[0], second.Endpoint) {
		t.Fatal("expected lowest score endpoint to be selected")
	}
}

func newMinScoreTestEndpoint(name string) fwkscheduling.Endpoint {
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Address:        name + ".svc.cluster.local",
		Port:           "8080",
	}
	return fwkscheduling.NewEndpoint(metadata, fwkdl.NewMetrics(), fwkdl.NewAttributes())
}

func newMinScoreTestScoredPod(name string, score float64) *fwkscheduling.ScoredEndpoint {
	return &fwkscheduling.ScoredEndpoint{Endpoint: newMinScoreTestEndpoint(name), Score: score}
}
