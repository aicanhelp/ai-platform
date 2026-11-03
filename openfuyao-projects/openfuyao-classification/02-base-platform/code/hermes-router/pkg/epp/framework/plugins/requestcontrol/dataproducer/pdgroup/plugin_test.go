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

package pdgroup

import (
	"context"
	"encoding/json"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"

	pdg "hermes-router/pkg/epp/internal/pdgroup"
)

const (
	testFactoryName    = "pd-group-producer"
	testRoleLabel      = "openfuyao.com/pdRole"
	testGroupLabel     = "openfuyao.com/pdGroupID"
	testPrefillValue   = "prefill"
	testDecodeValue    = "decode"
	testLeaderValue    = "leader"
	testValidGroupID   = "group-a"
	testInvalidGroupID = "group-b"
)

func TestFactoryRejectsEmptyRequiredParameters(t *testing.T) {
	raw := json.RawMessage(`{"pdLabelName":"","pdGroupLabelName":"group","prefillValue":"prefill","decodeValue":"decode","leaderValue":"leader"}`)
	handle := fwkplugin.NewEppHandle(context.Background(), nil)

	plugin, err := Factory(testFactoryName, raw, handle)
	if err == nil {
		t.Fatal("expected validation error")
	}
	if plugin != nil {
		t.Fatalf("expected nil plugin, got %T", plugin)
	}
}

func TestFactoryBuildsProducerWithValidatedConfig(t *testing.T) {
	raw := json.RawMessage(`{"pdLabelName":"openfuyao.com/pdRole","pdGroupLabelName":"openfuyao.com/pdGroupID","prefillValue":"prefill","decodeValue":"decode","leaderValue":"leader"}`)

	plugin, err := Factory(testFactoryName, raw, fwkplugin.NewEppHandle(context.Background(), nil))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	producer, ok := plugin.(*Producer)
	if !ok {
		t.Fatalf("expected *Producer, got %T", plugin)
	}
	if producer.roleLabelName != testRoleLabel || producer.groupLabelName != testGroupLabel || producer.prefillValue != testPrefillValue || producer.decodeValue != testDecodeValue || producer.leaderValue != testLeaderValue {
		t.Fatalf("unexpected producer config: %+v", producer)
	}
}

func TestPrepareRequestDataPublishesGroupInfoToUniqueLeader(t *testing.T) {
	producer := New(testFactoryName, Config{
		RoleLabelName:  testRoleLabel,
		GroupLabelName: testGroupLabel,
		PrefillValue:   testPrefillValue,
		DecodeValue:    testDecodeValue,
		LeaderValue:    testLeaderValue,
	})
	leader := newTestEndpoint("leader-a", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: testValidGroupID})
	prefillA := newTestEndpoint("prefill-a", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: testValidGroupID})
	prefillB := newTestEndpoint("prefill-b", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: testValidGroupID})
	decodeA := newTestEndpoint("decode-a", map[string]string{testRoleLabel: testDecodeValue, testGroupLabel: testValidGroupID})
	decodeB := newTestEndpoint("decode-b", map[string]string{testRoleLabel: testDecodeValue, testGroupLabel: testValidGroupID})
	invalidLeader := newTestEndpoint("leader-b", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: testInvalidGroupID})
	invalidPrefill := newTestEndpoint("prefill-b", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: testInvalidGroupID})
	aggregate := newTestEndpoint("aggregate", nil)

	err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{leader, prefillA, prefillB, decodeA, decodeB, invalidLeader, invalidPrefill, aggregate})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	info, ok := pdg.Get(leader)
	if !ok {
		t.Fatal("expected leader endpoint to receive Info")
	}
	if info.GroupID != testValidGroupID {
		t.Fatalf("expected group ID %q, got %q", testValidGroupID, info.GroupID)
	}
	if len(info.PrefillPods) != 2 || len(info.DecodePods) != 2 {
		t.Fatalf("unexpected pod counts: %+v", info)
	}
	if _, ok := pdg.Get(prefillA); ok {
		t.Fatal("did not expect prefill endpoint to receive leader attribute")
	}
	if _, ok := pdg.Get(prefillB); ok {
		t.Fatal("did not expect prefill endpoint to receive leader attribute")
	}
	if _, ok := pdg.Get(decodeA); ok {
		t.Fatal("did not expect decode endpoint to receive leader attribute")
	}
	if _, ok := pdg.Get(decodeB); ok {
		t.Fatal("did not expect decode endpoint to receive leader attribute")
	}
	if _, ok := pdg.Get(invalidLeader); ok {
		t.Fatal("did not expect invalid group leader to receive Info")
	}
	if _, ok := pdg.Get(aggregate); ok {
		t.Fatal("did not expect aggregate endpoint to receive Info")
	}
}

func TestPrepareRequestDataKeepsMemberAttributesVisibleAfterGroupCreation(t *testing.T) {
	producer := New(testFactoryName, Config{
		RoleLabelName:  testRoleLabel,
		GroupLabelName: testGroupLabel,
		PrefillValue:   testPrefillValue,
		DecodeValue:    testDecodeValue,
		LeaderValue:    testLeaderValue,
	})
	leader := newTestEndpoint("leader-late-attr", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: testValidGroupID})
	prefill := newTestEndpoint("prefill-late-attr", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: testValidGroupID})
	decode := newTestEndpoint("decode-late-attr", map[string]string{testRoleLabel: testDecodeValue, testGroupLabel: testValidGroupID})

	if err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{leader, prefill, decode}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	prefill.Put(latencyattr.LatencyPredictionInfoKey, latencyattr.NewLatencyPredictionInfo(true, true, 0, 0, 11, 1))

	info, ok := pdg.GetDirect(leader)
	if !ok || len(info.PrefillPods) != 1 {
		t.Fatalf("expected leader group info, got %+v", info)
	}
	if _, ok := info.PrefillPods[0].Endpoint.Get(latencyattr.LatencyPredictionInfoKey); !ok {
		t.Fatal("expected member attributes added after PD group creation to be visible through group info")
	}
}

func TestPrepareRequestDataPicksFirstLeaderWhenMultiple(t *testing.T) {
	producer := New(testFactoryName, Config{
		RoleLabelName:  testRoleLabel,
		GroupLabelName: testGroupLabel,
		PrefillValue:   testPrefillValue,
		DecodeValue:    testDecodeValue,
		LeaderValue:    testLeaderValue,
	})
	leaderA := newTestEndpoint("leader-a", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: testValidGroupID})
	leaderB := newTestEndpoint("leader-b", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: testValidGroupID})
	prefillA := newTestEndpoint("prefill-a", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: testValidGroupID})
	prefillB := newTestEndpoint("prefill-b", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: testValidGroupID})
	decodeA := newTestEndpoint("decode-a", map[string]string{testRoleLabel: testDecodeValue, testGroupLabel: testValidGroupID})
	decodeB := newTestEndpoint("decode-b", map[string]string{testRoleLabel: testDecodeValue, testGroupLabel: testValidGroupID})

	err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{leaderA, leaderB, prefillA, prefillB, decodeA, decodeB})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	infoA, okA := pdg.Get(leaderA)
	infoB, okB := pdg.Get(leaderB)
	if !okA && !okB {
		t.Fatal("expected exactly one leader to receive Info")
	}
	if okA && okB {
		t.Fatal("expected exactly one leader to receive Info, got two")
	}
	var info *pdg.Info
	if okA {
		info = infoA
	} else {
		info = infoB
	}
	if len(info.PrefillPods) != 2 || len(info.DecodePods) != 2 {
		t.Fatalf("unexpected pod counts: %+v", info)
	}
}

func TestPrepareRequestDataPublishesIndependentInfosForMultipleValidGroups(t *testing.T) {
	producer := New(testFactoryName, Config{
		RoleLabelName:  testRoleLabel,
		GroupLabelName: testGroupLabel,
		PrefillValue:   testPrefillValue,
		DecodeValue:    testDecodeValue,
		LeaderValue:    testLeaderValue,
	})
	groupALeader := newTestEndpoint("leader-a", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: "group-a"})
	groupAPrefill := newTestEndpoint("prefill-a", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: "group-a"})
	groupADecode := newTestEndpoint("decode-a", map[string]string{testRoleLabel: testDecodeValue, testGroupLabel: "group-a"})
	groupBLeader := newTestEndpoint("leader-b", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: "group-b"})
	groupBPrefill := newTestEndpoint("prefill-b", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: "group-b"})
	groupBDecode := newTestEndpoint("decode-b", map[string]string{testRoleLabel: testDecodeValue, testGroupLabel: "group-b"})

	err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{groupALeader, groupAPrefill, groupADecode, groupBLeader, groupBPrefill, groupBDecode})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	groupAInfo, ok := pdg.Get(groupALeader)
	if !ok || groupAInfo.GroupID != "group-a" {
		t.Fatalf("expected group-a info, got %+v", groupAInfo)
	}
	groupBInfo, ok := pdg.Get(groupBLeader)
	if !ok || groupBInfo.GroupID != "group-b" {
		t.Fatalf("expected group-b info, got %+v", groupBInfo)
	}
}

func TestPrepareRequestDataSkipsGroupMissingDecode(t *testing.T) {
	producer := New(testFactoryName, Config{
		RoleLabelName:  testRoleLabel,
		GroupLabelName: testGroupLabel,
		PrefillValue:   testPrefillValue,
		DecodeValue:    testDecodeValue,
		LeaderValue:    testLeaderValue,
	})
	leader := newTestEndpoint("leader-a", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: testValidGroupID})
	prefill := newTestEndpoint("prefill-a", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: testValidGroupID})

	err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{leader, prefill})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := pdg.Get(leader); ok {
		t.Fatal("did not expect Info for group missing decode")
	}
}

func TestPrepareRequestDataAcceptsEmptyEndpoints(t *testing.T) {
	producer := New(testFactoryName, Config{
		RoleLabelName:  testRoleLabel,
		GroupLabelName: testGroupLabel,
		PrefillValue:   testPrefillValue,
		DecodeValue:    testDecodeValue,
		LeaderValue:    testLeaderValue,
	})

	if err := producer.PrepareRequestData(context.Background(), nil, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPrepareRequestDataSkipsBlankGroupID(t *testing.T) {
	producer := New(testFactoryName, Config{
		RoleLabelName:  testRoleLabel,
		GroupLabelName: testGroupLabel,
		PrefillValue:   testPrefillValue,
		DecodeValue:    testDecodeValue,
		LeaderValue:    testLeaderValue,
	})
	leader := newTestEndpoint("leader-a", map[string]string{testRoleLabel: testLeaderValue, testGroupLabel: "   "})
	prefill := newTestEndpoint("prefill-a", map[string]string{testRoleLabel: testPrefillValue, testGroupLabel: "   "})
	decode := newTestEndpoint("decode-a", map[string]string{testRoleLabel: testDecodeValue, testGroupLabel: "   "})

	err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{leader, prefill, decode})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := pdg.Get(leader); ok {
		t.Fatal("did not expect Info for blank group ID")
	}
}

func TestProducesDeclaresPDGroupAttribute(t *testing.T) {
	producer := New(testFactoryName, Config{
		RoleLabelName:  testRoleLabel,
		GroupLabelName: testGroupLabel,
		PrefillValue:   testPrefillValue,
		DecodeValue:    testDecodeValue,
		LeaderValue:    testLeaderValue,
	})

	produces := producer.Produces()
	value, ok := produces[pdg.AttributeKey]
	if !ok {
		t.Fatalf("expected produced key %q", pdg.AttributeKey)
	}
	if _, ok := value.(*pdg.Info); !ok {
		t.Fatalf("expected Info prototype, got %T", value)
	}
}

func newTestEndpoint(name string, labels map[string]string) fwkscheduling.Endpoint {
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Address:        name + ".svc.cluster.local",
		Port:           "8080",
		Labels:         cloneLabels(labels),
	}
	return fwkscheduling.NewEndpoint(metadata, fwkdl.NewMetrics(), fwkdl.NewAttributes())
}

func cloneLabels(labels map[string]string) map[string]string {
	if labels == nil {
		return map[string]string{}
	}
	cloned := make(map[string]string, len(labels))
	for key, value := range labels {
		cloned[key] = value
	}
	return cloned
}
