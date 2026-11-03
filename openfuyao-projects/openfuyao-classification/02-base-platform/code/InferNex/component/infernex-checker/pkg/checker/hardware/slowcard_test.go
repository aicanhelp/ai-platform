/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package hardware

import (
	"errors"
	"strings"
	"testing"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// ---- checkH08 ----

func TestCheckH08_NpuSmiFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {err: errors.New("npu-smi not found")},
		},
	}
	r := checkH08(m)
	if r.Status != types.StatusFailed {
		t.Errorf("expected StatusFailed, got %v", r.Status)
	}
	if r.ID != "H-08" {
		t.Errorf("expected ID H-08, got %s", r.ID)
	}
}

func TestCheckH08_InsufficientIdleNPUs(t *testing.T) {
	// npu-smi output with 1 NPU occupied (only 1 idle)
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: npuSmiOccupiedOutput},
		},
	}
	r := checkH08(m)
	if r.Status != types.StatusSkipped {
		t.Errorf("expected StatusSkipped, got %v: %s", r.Status, r.Message)
	}
	if !strings.Contains(r.Message, "need >= 2 idle NPUs") {
		t.Errorf("unexpected message: %s", r.Message)
	}
}

func TestCheckH08_ZeroIdleNPUs(t *testing.T) {
	// Empty npu-smi output = 0 NPUs
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: ""},
		},
	}
	r := checkH08(m)
	if r.Status != types.StatusSkipped {
		t.Errorf("expected StatusSkipped, got %v", r.Status)
	}
}

// ---- getIdleNPUDevices ----

func TestGetIdleNPUDevices_Normal(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: npuSmiOutput},
		},
	}
	devices, err := getIdleNPUDevices(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(devices) != 2 {
		t.Errorf("expected 2 idle NPUs, got %d", len(devices))
	}
	// Check that devices are [0, 1]
	if devices[0] != 0 || devices[1] != 1 {
		t.Errorf("expected devices [0, 1], got %v", devices)
	}
}

func TestGetIdleNPUDevices_Error(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {err: errors.New("command not found")},
		},
	}
	_, err := getIdleNPUDevices(m)
	if err == nil {
		t.Error("expected error")
	}
}

func TestGetIdleNPUDevices_Occupied(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: npuSmiOccupiedOutput},
		},
	}
	devices, err := getIdleNPUDevices(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(devices) != 1 {
		t.Errorf("expected 1 idle NPU, got %d", len(devices))
	}
	// Device 0 is occupied, only device 1 should be idle
	if devices[0] != 1 {
		t.Errorf("expected device [1], got %v", devices)
	}
}

// ---- detectNPUResourceName ----

func TestDetectNPUResourceName_Found(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `'{"cpu":"64","huawei.com/Ascend910B4":"8","memory":"256Gi"}'`,
			},
		},
	}
	name, err := detectNPUResourceName(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "huawei.com/Ascend910B4" {
		t.Errorf("expected huawei.com/Ascend910B4, got %s", name)
	}
}

func TestDetectNPUResourceName_310P(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `'{"cpu":"32","huawei.com/Ascend310P3":"4","memory":"128Gi"}'`,
			},
		},
	}
	name, err := detectNPUResourceName(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "huawei.com/Ascend310P3" {
		t.Errorf("expected huawei.com/Ascend310P3, got %s", name)
	}
}

func TestDetectNPUResourceName_NotFound(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `'{"cpu":"64","memory":"256Gi"}'`,
			},
		},
	}
	_, err := detectNPUResourceName(m)
	if err == nil {
		t.Error("expected error when no Ascend resource found")
	}
}

func TestDetectNPUResourceName_KubectlFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				err: errors.New("connection refused"),
			},
		},
	}
	_, err := detectNPUResourceName(m)
	if err == nil {
		t.Error("expected error on kubectl failure")
	}
}

func TestDetectNPUResourceName_InvalidJSON(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `'not valid json'`,
			},
		},
	}
	_, err := detectNPUResourceName(m)
	if err == nil {
		t.Error("expected error on invalid JSON")
	}
}

// ---- parseSlowCardResult ----

func TestParseSlowCardResult_NoSlowRank(t *testing.T) {
	output := `some output
Suggestion: There is no slow rank in the current cluster communication
more output`
	suggestion, err := parseSlowCardResult(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(suggestion, "no slow rank") {
		t.Errorf("expected 'no slow rank' in suggestion, got %s", suggestion)
	}
}

func TestParseSlowCardResult_SlowRankComputation(t *testing.T) {
	output := `=== Exporting summary ===
Suggestion: There is a low efficiency issue in the current cluster communication, Waiting take up most of the time in communication action for rank 0 . The max wait ratio is: 0.98
 The reason is that the the ranks are not synchronized. Rank 1 has a lot longer computation time than other ranks.
=== Done ===`
	suggestion, err := parseSlowCardResult(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := "Rank 1 is the root cause (longer computation time), affecting Rank 0 (max wait ratio: 0.98)"
	if suggestion != expected {
		t.Errorf("expected %q, got %q", expected, suggestion)
	}
}

func TestParseSlowCardResult_SlowRankBandwidth(t *testing.T) {
	output := `=== Exporting summary ===
Suggestion: There is a low efficiency issue in the current cluster communication, Waiting take up most of the time in communication action for rank 0 . The max wait ratio is: 0.85
 The possible reason is that the communication bandwidths between ranks are inconsistent.
=== Done ===`
	suggestion, err := parseSlowCardResult(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := "Communication bandwidth inconsistency detected, affecting Rank 0 (max wait ratio: 0.85)"
	if suggestion != expected {
		t.Errorf("expected %q, got %q", expected, suggestion)
	}
}

func TestParseSlowCardResult_RawSuggestion(t *testing.T) {
	output := `some output
Suggestion: some unrecognized format

more output`
	suggestion, err := parseSlowCardResult(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if suggestion != "some unrecognized format" {
		t.Errorf("expected raw suggestion, got %s", suggestion)
	}
}

func TestParseSlowCardResult_NoSuggestion(t *testing.T) {
	output := `some output without suggestion line`
	_, err := parseSlowCardResult(output)
	if err == nil {
		t.Error("expected error when no Suggestion found")
	}
}

// ---- getPhysicalDeviceMapping ----

func TestGetPhysicalDeviceMapping_Complete(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl exec msprof-slowcard-check -n default -- cat /tmp/rank_0_device.txt": {
				stdout: "rank=0\nactual_device=0\nphysical_devices=0,1,2,4,5,6",
			},
			"kubectl exec msprof-slowcard-check -n default -- cat /tmp/rank_1_device.txt": {
				stdout: "rank=1\nactual_device=1\nphysical_devices=0,1,2,4,5,6",
			},
			"kubectl exec msprof-slowcard-check -n default -- cat /tmp/rank_2_device.txt": {
				stdout: "rank=2\nactual_device=2\nphysical_devices=0,1,2,4,5,6",
			},
		},
	}
	mapping, err := getPhysicalDeviceMapping(m, "msprof-slowcard-check", "default", 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping[0] != 0 || mapping[1] != 1 || mapping[2] != 2 {
		t.Errorf("unexpected mapping: %v", mapping)
	}
}

func TestGetPhysicalDeviceMapping_NonContiguousDevices(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl exec pod -n ns -- cat /tmp/rank_0_device.txt": {
				stdout: "rank=0\nactual_device=0\nphysical_devices=0,1,2,4,5,6",
			},
			"kubectl exec pod -n ns -- cat /tmp/rank_1_device.txt": {
				stdout: "rank=1\nactual_device=1\nphysical_devices=0,1,2,4,5,6",
			},
			"kubectl exec pod -n ns -- cat /tmp/rank_2_device.txt": {
				stdout: "rank=2\nactual_device=2\nphysical_devices=0,1,2,4,5,6",
			},
			"kubectl exec pod -n ns -- cat /tmp/rank_3_device.txt": {
				stdout: "rank=3\nactual_device=3\nphysical_devices=0,1,2,4,5,6",
			},
		},
	}
	mapping, err := getPhysicalDeviceMapping(m, "pod", "ns", 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// rank 3, actual_device=3, phys_list[3]=4
	if mapping[3] != 4 {
		t.Errorf("expected rank 3 -> physical device 4, got %d", mapping[3])
	}
}

func TestGetPhysicalDeviceMapping_PartialFailure(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl exec pod -n ns -- cat /tmp/rank_0_device.txt": {
				stdout: "rank=0\nactual_device=0\nphysical_devices=0,1",
			},
			"kubectl exec pod -n ns -- cat /tmp/rank_1_device.txt": {
				err: errors.New("file not found"),
			},
		},
	}
	mapping, err := getPhysicalDeviceMapping(m, "pod", "ns", 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// rank 0 should be mapped, rank 1 should be missing
	if _, ok := mapping[0]; !ok {
		t.Error("expected rank 0 in mapping")
	}
	if _, ok := mapping[1]; ok {
		t.Error("expected rank 1 to be missing from mapping")
	}
}

func TestGetPhysicalDeviceMapping_NotSet(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl exec pod -n ns -- cat /tmp/rank_0_device.txt": {
				stdout: "rank=0\nactual_device=0\nphysical_devices=NOT_SET",
			},
		},
	}
	mapping, err := getPhysicalDeviceMapping(m, "pod", "ns", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// physical_devices=NOT_SET means no mapping available
	if _, ok := mapping[0]; ok {
		t.Error("expected no mapping when physical_devices=NOT_SET")
	}
}

// ---- collectRankMetrics ----

func TestCollectRankMetrics_Normal(t *testing.T) {
	commJSON := `{
		"Total HCCL Operators": {
			"0": {"Communication Time Info": {"Wait Time(ms)": 10.5, "Transit Time(ms)": 20.1, "Synchronization Time(ms)": 5.2, "Wait Time Ratio": 0.35, "Synchronization Time Ratio": 0.15}},
			"1": {"Communication Time Info": {"Wait Time(ms)": 11.0, "Transit Time(ms)": 19.8, "Synchronization Time(ms)": 5.5, "Wait Time Ratio": 0.37, "Synchronization Time Ratio": 0.16}},
			"Slow Rank Suggestion": "no slow rank"
		}
	}`
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"find": {stdout: "/tmp/prof_data/PROF_xxx/query/communication_0.json"},
			"cat":  {stdout: commJSON},
		},
	}
	deviceMapping := map[int]int{0: 0, 1: 1}
	metrics, err := collectRankMetrics(m, "pod", "ns", "node1", deviceMapping)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(metrics) != 2 {
		t.Fatalf("expected 2 metrics, got %d", len(metrics))
	}
	// Verify metrics content
	for _, metric := range metrics {
		if metric.NodeName != "node1" {
			t.Errorf("expected node_name=node1, got %s", metric.NodeName)
		}
		if metric.WaitTimeMS == 0 {
			t.Errorf("expected non-zero WaitTimeMS for rank %d", metric.RankID)
		}
	}
}

func TestCollectRankMetrics_JSONNotFound(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"find": {stdout: ""},
		},
	}
	_, err := collectRankMetrics(m, "pod", "ns", "node1", map[int]int{})
	if err == nil {
		t.Error("expected error when JSON not found")
	}
}

func TestCollectRankMetrics_MissingDeviceMapping(t *testing.T) {
	commJSON := `{
		"Total HCCL Operators": {
			"0": {"Communication Time Info": {"Wait Time(ms)": 10.5, "Wait Time Ratio": 0.35}},
			"3": {"Communication Time Info": {"Wait Time(ms)": 15.0, "Wait Time Ratio": 0.85}}
		}
	}`
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"find": {stdout: "/tmp/prof_data/PROF_xxx/query/communication_0.json"},
			"cat":  {stdout: commJSON},
		},
	}
	// Only rank 0 has mapping, rank 3 does not
	deviceMapping := map[int]int{0: 0}
	metrics, err := collectRankMetrics(m, "pod", "ns", "node1", deviceMapping)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(metrics) != 2 {
		t.Fatalf("expected 2 metrics, got %d", len(metrics))
	}
	// rank 3 should have PhysicalDeviceID = -1
	for _, metric := range metrics {
		if metric.RankID == 3 && metric.PhysicalDeviceID != -1 {
			t.Errorf("expected PhysicalDeviceID=-1 for unmapped rank 3, got %d", metric.PhysicalDeviceID)
		}
	}
}

// ---- getPhysicalDevicesFromAnnotation ----

func TestGetPhysicalDevicesFromAnnotation_Normal(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pod": {stdout: "Ascend910-2,Ascend910-4,Ascend910-5"},
		},
	}
	devices, err := getPhysicalDevicesFromAnnotation(m, "pod", "ns")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if devices != "2,4,5" {
		t.Errorf("expected '2,4,5', got '%s'", devices)
	}
}

func TestGetPhysicalDevicesFromAnnotation_KubectlFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pod": {err: errors.New("pod not found")},
		},
	}
	_, err := getPhysicalDevicesFromAnnotation(m, "pod", "ns")
	if err == nil {
		t.Error("expected error")
	}
}

func TestGetPhysicalDevicesFromAnnotation_EmptyAnnotation(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pod": {stdout: ""},
		},
	}
	_, err := getPhysicalDevicesFromAnnotation(m, "pod", "ns")
	if err == nil {
		t.Error("expected error when annotation is empty")
	}
}

// ---- detectCANNImage ----

func TestDetectCANNImage_910B(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"source /dev/stdin": {
				stdout: "swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11-devel\n910B",
			},
		},
	}
	image, chipName, err := detectCANNImage(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(image, "910b") {
		t.Errorf("expected 910b image, got %s", image)
	}
	if chipName != "910B" {
		t.Errorf("expected chip name 910B, got %s", chipName)
	}
}

func TestDetectCANNImage_CommandFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"source /dev/stdin": {err: errors.New("lspci not found")},
		},
	}
	_, _, err := detectCANNImage(m)
	if err == nil {
		t.Error("expected error")
	}
}

func TestDetectCANNImage_InvalidOutput(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"source /dev/stdin": {stdout: "only-one-line"},
		},
	}
	_, _, err := detectCANNImage(m)
	if err == nil {
		t.Error("expected error for invalid output")
	}
}

func TestDetectCANNImage_EmptyOutput(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"source /dev/stdin": {stdout: "\n"},
		},
	}
	_, _, err := detectCANNImage(m)
	if err == nil {
		t.Error("expected error for empty output")
	}
}

// ---- generateMsprofPodYAML ----

func TestGenerateMsprofPodYAML(t *testing.T) {
	config := msprofPodConfig{
		NodeName:        "node1",
		PodName:         "pod1",
		Namespace:       "default",
		NPUCount:        4,
		Image:           "cann:latest",
		NPUResourceName: "huawei.com/Ascend910B4",
	}
	yaml := generateMsprofPodYAML(config)
	if !strings.Contains(yaml, "nodeName: node1") {
		t.Error("expected nodeName in YAML")
	}
	if !strings.Contains(yaml, "name: pod1") {
		t.Error("expected pod name in YAML")
	}
	if !strings.Contains(yaml, "namespace: default") {
		t.Error("expected namespace in YAML")
	}
	if !strings.Contains(yaml, "image: cann:latest") {
		t.Error("expected image in YAML")
	}
	if !strings.Contains(yaml, "huawei.com/Ascend910B4") {
		t.Error("expected NPU resource name in YAML")
	}
	if !strings.Contains(yaml, `"4"`) {
		t.Error("expected NPU count in YAML")
	}
}

// ---- prepareScriptsInPod ----

func TestPrepareScriptsInPod_Success(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl exec": {stdout: ""},
		},
	}
	err := prepareScriptsInPod(m, "msprof-slowcard-check")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPrepareScriptsInPod_BenchmarkCopyFail(t *testing.T) {
	callCount := 0
	m := &mockRunner{
		nodeName:  "node1",
		responses: map[string]mockResponse{},
	}
	// Override Run to fail on first kubectl exec call
	origRun := m.Run
	_ = origRun
	// Use a custom mock that fails on first call
	customMock := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl exec": {err: errors.New("exec failed")},
		},
	}
	_ = callCount
	err := prepareScriptsInPod(customMock, "pod")
	if err == nil {
		t.Error("expected error when benchmark copy fails")
	}
}

// ---- executeMsprofWorkflow ----

func TestExecuteMsprofWorkflow_ExecFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pod": {stdout: "Ascend910-0,Ascend910-1"},
			"kubectl exec":    {stdout: "partial output", stderr: "torchrun not found", err: errors.New("exit status 1")},
		},
	}
	_, err := executeMsprofWorkflow(m, "pod", []int{0, 1})
	if err == nil {
		t.Error("expected error when exec fails")
	}
	if !strings.Contains(err.Error(), "msprof check failed") {
		t.Errorf("expected msprof error, got: %v", err)
	}
}

func TestExecuteMsprofWorkflow_Success(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pod": {stdout: "Ascend910-0,Ascend910-1"},
			"kubectl exec":    {stdout: "=== Exporting summary ===\nSuggestion: no slow rank"},
		},
	}
	output, err := executeMsprofWorkflow(m, "pod", []int{0, 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(output, "Suggestion") {
		t.Errorf("expected Suggestion in output, got: %s", output)
	}
}

// ---- createAndStartMsprofPod ----

func TestCreateAndStartMsprofPod_WriteFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat >": {err: errors.New("write failed")},
		},
	}
	_, _, err := createAndStartMsprofPod(m, "node1", 2, "cann:latest", "huawei.com/Ascend910B4")
	if err == nil {
		t.Error("expected error when yaml write fails")
	}
}

func TestCreateAndStartMsprofPod_ApplyFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat >":          {stdout: ""},
			"kubectl apply":  {err: errors.New("apply failed")},
			"kubectl delete": {stdout: ""},
			"rm -f":          {stdout: ""},
		},
	}
	_, _, err := createAndStartMsprofPod(m, "node1", 2, "cann:latest", "huawei.com/Ascend910B4")
	if err == nil {
		t.Error("expected error when apply fails")
	}
}

func TestCreateAndStartMsprofPod_WaitFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat >":          {stdout: ""},
			"kubectl apply":  {stdout: ""},
			"kubectl wait":   {err: errors.New("timeout")},
			"kubectl delete": {stdout: ""},
			"rm -f":          {stdout: ""},
		},
	}
	_, _, err := createAndStartMsprofPod(m, "node1", 2, "cann:latest", "huawei.com/Ascend910B4")
	if err == nil {
		t.Error("expected error when wait fails")
	}
}

func TestCreateAndStartMsprofPod_Success(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat >":          {stdout: ""},
			"kubectl apply":  {stdout: "pod/msprof-slowcard-check created"},
			"kubectl wait":   {stdout: "pod/msprof-slowcard-check condition met"},
			"kubectl delete": {stdout: ""},
			"rm -f":          {stdout: ""},
		},
	}
	podName, cleanup, err := createAndStartMsprofPod(m, "node1", 2, "cann:latest", "huawei.com/Ascend910B4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(podName, "msprof-slowcard-") {
		t.Errorf("expected pod name with prefix msprof-slowcard-, got %s", podName)
	}
	if cleanup == nil {
		t.Error("expected non-nil cleanup function")
	}
	// Call cleanup to verify it doesn't panic
	cleanup()
}

// ---- runMsprofCheck (integration) ----

func TestRunMsprofCheck_DetectCANNFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: npuSmiOutput},
			// detectCANNImage will fail because no matching response
		},
	}
	_, err := runMsprofCheck(m, []int{0, 1})
	if err == nil {
		t.Error("expected error when CANN image detection fails")
	}
}

func TestRunMsprofCheck_DetectResourceNameFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"source /dev/stdin": {
				stdout: "swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11-devel\n910B",
			},
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `'{"cpu":"64","memory":"256Gi"}'`, // no Ascend resource
			},
		},
	}
	_, err := runMsprofCheck(m, []int{0, 1})
	if err == nil {
		t.Error("expected error when NPU resource name not found")
	}
	if !strings.Contains(err.Error(), "NPU resource name") {
		t.Errorf("expected NPU resource name error, got: %v", err)
	}
}

func TestRunMsprofCheck_CreatePodFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"source /dev/stdin": {
				stdout: "swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11-devel\n910B",
			},
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `'{"cpu":"64","huawei.com/Ascend910B4":"8","memory":"256Gi"}'`,
			},
			"cat >": {err: errors.New("write failed")},
		},
	}
	_, err := runMsprofCheck(m, []int{0, 1})
	if err == nil {
		t.Error("expected error when pod creation fails")
	}
}

// ---- checkH08 full path tests ----

func TestCheckH08_MsprofCheckFail(t *testing.T) {
	// Simulate: idle NPUs available but msprof workflow fails
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: npuSmiOutput}, // 2 idle NPUs
			// detectCANNImage will fail → runMsprofCheck returns error
		},
	}
	r := checkH08(m)
	if r.Status != types.StatusWarning {
		t.Errorf("expected StatusWarning when msprof fails, got %v: %s", r.Status, r.Message)
	}
	if !strings.Contains(r.Message, "Slow card detection failed") {
		t.Errorf("unexpected message: %s", r.Message)
	}
}
