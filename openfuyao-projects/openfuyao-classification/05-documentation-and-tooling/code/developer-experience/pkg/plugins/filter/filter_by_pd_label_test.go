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

package filter

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	"hermes-router/pkg/plugins/common"
)

// TestFilterByPDLabelFactory tests FilterByPDLabelFactory function.
const (
	testFactoryName = "test-factory-filter"
)

func TestFilterByPDLabelFactory(t *testing.T) {
	rawParams := json.RawMessage(`{}`)
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := FilterByPDLabelFactory(testFactoryName, rawParams, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin, got nil")
	}
	filter, ok := plugin.(*FilterByPDLabel)
	if !ok {
		t.Fatal("expected *FilterByPDLabel type")
	}
	if filter.typedName.Name != testFactoryName {
		t.Errorf("expected name %s, got %s", testFactoryName, filter.typedName.Name)
	}
}

// TestNewFilterByPDLabel tests NewFilterByPDLabel function.
const (
	testNewFilterName       = "test-new-filter"
	testParamRoleLabelName  = "openfuyao.com/pdRole"
	testParamGroupLabelName = "openfuyao.com/pdGroupID"
	testParamPrefillValue   = "prefill"
	testParamDecodeValue    = "decode"
	testParamLeaderValue    = "leader"
)

func TestNewFilterByPDLabel(t *testing.T) {
	rawParams := json.RawMessage(`{
		"pdLabelName": "openfuyao.com/pdRole",
		"pdGroupLabelName": "openfuyao.com/pdGroupID",
		"prefillValue": "prefill",
		"decodeValue": "decode",
		"leaderValue": "leader"
	}`)

	filter := NewFilterByPDLabel(context.Background(), testNewFilterName, rawParams)

	if filter.typedName.Name != testNewFilterName {
		t.Errorf("expected name %s, got %s", testNewFilterName, filter.typedName.Name)
	}
	if filter.roleLabelName != testParamRoleLabelName {
		t.Errorf("expected roleLabelName %s, got %s", testParamRoleLabelName, filter.roleLabelName)
	}
	if filter.groupLabelName != testParamGroupLabelName {
		t.Errorf("expected groupLabelName %s, got %s", testParamGroupLabelName, filter.groupLabelName)
	}
	if filter.prefillValue != testParamPrefillValue {
		t.Errorf("expected prefillValue %s, got %s", testParamPrefillValue, filter.prefillValue)
	}
	if filter.decodeValue != testParamDecodeValue {
		t.Errorf("expected decodeValue %s, got %s", testParamDecodeValue, filter.decodeValue)
	}
	if filter.leaderValue != testParamLeaderValue {
		t.Errorf("expected leaderValue %s, got %s", testParamLeaderValue, filter.leaderValue)
	}
}

// TestNewFilterByPDLabelEmptyParams tests NewFilterByPDLabel with empty parameters.
func TestNewFilterByPDLabelEmptyParams(t *testing.T) {
	rawParams := json.RawMessage(`{}`)

	filter := NewFilterByPDLabel(context.Background(), testNewFilterName, rawParams)

	if filter.typedName.Name != testNewFilterName {
		t.Errorf("expected name %s, got %s", testNewFilterName, filter.typedName.Name)
	}
	if filter.roleLabelName != "" {
		t.Errorf("expected empty roleLabelName, got %s", filter.roleLabelName)
	}
}

// TestTypedName tests TypedName method.
const (
	testTypedNameName = "test-typed-name"
)

func TestTypedName(t *testing.T) {
	filter := &FilterByPDLabel{
		typedName: plugins.TypedName{Type: FilterByPDLabelType, Name: testTypedNameName},
	}

	typedName := filter.TypedName()

	if typedName.Type != FilterByPDLabelType {
		t.Errorf("expected type %s, got %s", FilterByPDLabelType, typedName.Type)
	}
	if typedName.Name != testTypedNameName {
		t.Errorf("expected name %s, got %s", testTypedNameName, typedName.Name)
	}
}

// TestWithName tests WithName method.
const (
	testOriginalName = "original-name"
	testNewName      = "new-name"
)

func TestWithName(t *testing.T) {
	original := &FilterByPDLabel{
		typedName:      plugins.TypedName{Type: FilterByPDLabelType, Name: testOriginalName},
		roleLabelName:  testParamRoleLabelName,
		groupLabelName: testParamGroupLabelName,
		prefillValue:   testParamPrefillValue,
		decodeValue:    testParamDecodeValue,
		leaderValue:    testParamLeaderValue,
	}

	newFilter := original.WithName(testNewName)

	if newFilter.typedName.Name != testNewName {
		t.Errorf("expected name %s, got %s", testNewName, newFilter.typedName.Name)
	}
	if newFilter.roleLabelName != testParamRoleLabelName {
		t.Errorf("expected roleLabelName %s, got %s", testParamRoleLabelName, newFilter.roleLabelName)
	}
	if original.typedName.Name != testOriginalName {
		t.Error("original filter should not be modified")
	}
}

// TestFilterEmptyPods tests Filter with empty pods list.
const (
	testEmptyPodsLength = 0
	testRequestID       = "test-request"
)

func TestFilterEmptyPods(t *testing.T) {
	filter := createTestFilter()
	state := types.NewCycleState()
	request := &types.LLMRequest{RequestId: testRequestID}

	resultPods := filter.Filter(context.Background(), state, request, []types.Pod{})

	if len(resultPods) != testEmptyPodsLength {
		t.Errorf("expected %d pods, got %d", testEmptyPodsLength, len(resultPods))
	}
	_, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err == nil {
		t.Error("expected state to be deleted when no pods")
	}
}

// TestFilterGroupsPods tests Filter with multiple groups.
const (
	testGroup1ID        = "group-1"
	testGroup2ID        = "group-2"
	testGroup3ID        = "group-3"
	testPrefillPodCount = 2
	testDecodePodCount  = 2
	testExpectedGroups  = 3
)

func TestFilterGroupsPods(t *testing.T) {
	filter := createTestFilter()
	pods := createPodsWithGroups(testGroup1ID, testGroup2ID, testGroup3ID)
	state := types.NewCycleState()
	request := &types.LLMRequest{RequestId: testRequestID}

	resultPods := filter.Filter(context.Background(), state, request, pods)

	if len(resultPods) != len(pods) {
		t.Errorf("expected %d pods returned, got %d", len(pods), len(resultPods))
	}
	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err != nil {
		t.Fatalf("failed to read pd groups: %v", err)
	}
	if len(pdGroupList.Groups) != testExpectedGroups {
		t.Errorf("expected %d groups, got %d", testExpectedGroups, len(pdGroupList.Groups))
	}
	verifyGroupStructure(t, pdGroupList.Groups[0], testPrefillPodCount, testDecodePodCount)
}

// TestFilterPodsMissingLabels tests Filter with pods missing labels.
const (
	testPodWithoutRoleLabelName  = "pod-no-role"
	testPodWithoutGroupLabelName = "pod-no-group"
	testPodWithEmptyGroupIDName  = "pod-empty-group"
	testEmptyString              = ""
	testOldGroupID               = "old-group"
)

func TestFilterPodsMissingLabels(t *testing.T) {
	filter := createTestFilter()
	pods := []types.Pod{
		createPodWithLabels(testPodWithoutRoleLabelName, map[string]string{
			testParamGroupLabelName: testGroup1ID,
		}),
		createPodWithLabels(testPodWithoutGroupLabelName, map[string]string{
			testParamRoleLabelName: testParamPrefillValue,
		}),
		createPodWithLabels(testPodWithEmptyGroupIDName, map[string]string{
			testParamRoleLabelName:  testParamPrefillValue,
			testParamGroupLabelName: testEmptyString,
		}),
	}
	state := types.NewCycleState()
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{Groups: []common.PDGroup{{ID: testOldGroupID}}})
	request := &types.LLMRequest{RequestId: testRequestID}

	resultPods := filter.Filter(context.Background(), state, request, pods)

	if len(resultPods) != len(pods) {
		t.Errorf("expected %d pods returned, got %d", len(pods), len(resultPods))
	}
	_, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err == nil {
		t.Error("expected state key to be deleted when no valid groups")
	}
}

// TestFilterPodsWithInvalidRole tests Filter with pods having invalid role values.
const (
	testPodInvalidRoleName = "pod-invalid-role"
	testInvalidRoleValue   = "invalid-role"
	testEmptyPodCount      = 0
)

func TestFilterPodsWithInvalidRole(t *testing.T) {
	filter := createTestFilter()
	pods := []types.Pod{
		createPodWithLabels(testPodInvalidRoleName, map[string]string{
			testParamRoleLabelName:  testInvalidRoleValue,
			testParamGroupLabelName: testGroup1ID,
		}),
	}
	state := types.NewCycleState()
	request := &types.LLMRequest{RequestId: testRequestID}

	resultPods := filter.Filter(context.Background(), state, request, pods)

	if len(resultPods) != len(pods) {
		t.Errorf("expected %d pods returned, got %d", len(pods), len(resultPods))
	}
	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err != nil {
		t.Fatalf("failed to read pd groups: %v", err)
	}
	if len(pdGroupList.Groups) != testSingleGroup {
		t.Errorf("expected %d group, got %d", testSingleGroup, len(pdGroupList.Groups))
	}
	group := pdGroupList.Groups[0]
	if group.ID != testGroup1ID {
		t.Errorf("expected group ID %s, got %s", testGroup1ID, group.ID)
	}
	if len(group.PrefillPods) != testEmptyPodCount {
		t.Errorf("expected %d prefill pods, got %d", testEmptyPodCount, len(group.PrefillPods))
	}
	if len(group.DecodePods) != testEmptyPodCount {
		t.Errorf("expected %d decode pods, got %d", testEmptyPodCount, len(group.DecodePods))
	}
	if group.LeaderPod.Pod != nil {
		t.Error("expected leader pod to be nil")
	}
}

// TestFilterPodsWithValidGroup tests Filter with pods that form a valid group.
const (
	testPodPrefillName = "pod-prefill"
	testPodDecodeName  = "pod-decode"
	testPodLeaderName  = "pod-leader"
	testSingleGroup    = 1
)

func TestFilterPodsWithValidGroup(t *testing.T) {
	filter := createTestFilter()
	pods := []types.Pod{
		createPodWithLabels(testPodPrefillName, map[string]string{
			testParamRoleLabelName:  testParamPrefillValue,
			testParamGroupLabelName: testGroup1ID,
		}),
		createPodWithLabels(testPodDecodeName, map[string]string{
			testParamRoleLabelName:  testParamDecodeValue,
			testParamGroupLabelName: testGroup1ID,
		}),
		createPodWithLabels(testPodLeaderName, map[string]string{
			testParamRoleLabelName:  testParamLeaderValue,
			testParamGroupLabelName: testGroup1ID,
		}),
	}
	state := types.NewCycleState()
	request := &types.LLMRequest{RequestId: testRequestID}

	resultPods := filter.Filter(context.Background(), state, request, pods)

	if len(resultPods) != len(pods) {
		t.Errorf("expected %d pods returned, got %d", len(pods), len(resultPods))
	}
	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err != nil {
		t.Fatalf("failed to read pd groups: %v", err)
	}
	if len(pdGroupList.Groups) != testSingleGroup {
		t.Errorf("expected %d group, got %d", testSingleGroup, len(pdGroupList.Groups))
	}
}

// TestFilterSortsGroupIDs tests that Filter sorts group IDs alphabetically.
const (
	testGroupAID = "group-a"
	testGroupBID = "group-b"
	testGroupCID = "group-c"
)

const (
	testGroupIndex0 = 0
	testGroupIndex1 = 1
	testGroupIndex2 = 2
)

func TestFilterSortsGroupIDs(t *testing.T) {
	filter := createTestFilter()
	pods := createPodsWithGroups(testGroupCID, testGroupAID, testGroupBID)
	state := types.NewCycleState()
	request := &types.LLMRequest{RequestId: testRequestID}

	filter.Filter(context.Background(), state, request, pods)

	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err != nil {
		t.Fatalf("failed to read pd groups: %v", err)
	}
	if len(pdGroupList.Groups) != testExpectedGroups {
		t.Fatalf("expected %d groups, got %d", testExpectedGroups, len(pdGroupList.Groups))
	}
	if pdGroupList.Groups[testGroupIndex0].ID != testGroupAID {
		t.Errorf("expected first group ID %s, got %s", testGroupAID, pdGroupList.Groups[testGroupIndex0].ID)
	}
	if pdGroupList.Groups[testGroupIndex1].ID != testGroupBID {
		t.Errorf("expected second group ID %s, got %s", testGroupBID, pdGroupList.Groups[testGroupIndex1].ID)
	}
	if pdGroupList.Groups[testGroupIndex2].ID != testGroupCID {
		t.Errorf("expected third group ID %s, got %s", testGroupCID, pdGroupList.Groups[testGroupIndex2].ID)
	}
}

// Helper functions
const (
	testFilterName = "test-filter"
)

func createTestFilter() *FilterByPDLabel {
	return &FilterByPDLabel{
		typedName:      plugins.TypedName{Type: FilterByPDLabelType, Name: testFilterName},
		roleLabelName:  testParamRoleLabelName,
		groupLabelName: testParamGroupLabelName,
		prefillValue:   testParamPrefillValue,
		decodeValue:    testParamDecodeValue,
		leaderValue:    testParamLeaderValue,
	}
}

func createPodsWithGroups(groupIDs ...string) []types.Pod {
	pods := make([]types.Pod, 0)
	for _, groupID := range groupIDs {
		for i := 0; i < testPrefillPodCount; i++ {
			pods = append(pods, createPodWithLabels(
				groupID+"-prefill-"+strconv.Itoa(i),
				map[string]string{
					testParamRoleLabelName:  testParamPrefillValue,
					testParamGroupLabelName: groupID,
				}))
		}
		for i := 0; i < testDecodePodCount; i++ {
			pods = append(pods, createPodWithLabels(
				groupID+"-decode-"+strconv.Itoa(i),
				map[string]string{
					testParamRoleLabelName:  testParamDecodeValue,
					testParamGroupLabelName: groupID,
				}))
		}
		pods = append(pods, createPodWithLabels(
			groupID+"-leader-0",
			map[string]string{
				testParamRoleLabelName:  testParamLeaderValue,
				testParamGroupLabelName: groupID,
			}))
	}
	return pods
}

func createPodWithLabels(name string, labels map[string]string) types.Pod {
	return &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: name,
			Labels:  labels,
		},
		MetricsState: &metrics.MetricsState{},
	}
}

func verifyGroupStructure(t *testing.T, group common.PDGroup, expectedPrefill, expectedDecode int) {
	if group.ID == "" {
		t.Error("group ID should not be empty")
	}
	if len(group.PrefillPods) != expectedPrefill {
		t.Errorf("expected %d prefill pods, got %d", expectedPrefill, len(group.PrefillPods))
	}
	if len(group.DecodePods) != expectedDecode {
		t.Errorf("expected %d decode pods, got %d", expectedDecode, len(group.DecodePods))
	}
	if group.LeaderPod.Pod == nil {
		t.Error("leader pod should not be nil")
	}
}
