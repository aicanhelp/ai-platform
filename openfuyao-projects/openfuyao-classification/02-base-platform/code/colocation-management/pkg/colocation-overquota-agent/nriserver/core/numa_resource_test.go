/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * Author: Junlin Liu
 * Date: 2025-06-19
 */

package core

import (
	"reflect"
	"runtime"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// Constants for test data
const (
	testCPUZero  = 0
	testCPUOne   = 1
	testCPUTwo   = 2
	testCPUThree = 3
	testCPUFour  = 4
	testCPUFive  = 5
	testCPUSix   = 6
	testCPUSeven = 7
	testCPUEight = 8
	testCPUNine  = 9
	testCPUTen   = 10

	nodeZero = 0
	nodeOne  = 1
	nodeTwo  = 2
	nodeFive = 5

	minCPUOne   = 1
	minCPUThree = 3
	minCPUFour  = 4
	minCPUTen   = 10

	memoryOneGB = 1024 * 1024 * 1024
	memoryTwoGB = 2 * 1024 * 1024 * 1024

	goroutineCount = 10
	iterationCount = 100
)

// Test data structures
type parseCPUListTestCase struct {
	name     string
	input    string
	expected []int
	hasError bool
}

type cpuListToStringTestCase struct {
	name     string
	input    []int
	expected string
}

type numaNodeTestCase struct {
	name       string
	cpu        int
	expectedID int
	found      bool
}

type numaNodeCPUsTestCase struct {
	name         string
	nodeID       int
	expectedCPUs []int
	hasError     bool
}

type bestNumaNodeTestCase struct {
	name           string
	minCPUCount    int
	expectedNodeID int
	hasError       bool
}

// Helper function to create test topology
func createTestTopology() *NumaTopology {
	return &NumaTopology{
		Nodes: map[int]*NumaNode{
			nodeZero: {ID: nodeZero, CPUList: []int{testCPUZero, testCPUOne, testCPUTwo, testCPUThree}},
			nodeOne:  {ID: nodeOne, CPUList: []int{testCPUFour, testCPUFive, testCPUSix, testCPUSeven}},
		},
	}
}

// Helper function to create test topology for best node selection
func createBestNodeTestTopology() *NumaTopology {
	return &NumaTopology{
		Nodes: map[int]*NumaNode{
			nodeZero: {ID: nodeZero, CPUList: []int{testCPUZero, testCPUOne}},                           // 2 CPUs
			nodeOne:  {ID: nodeOne, CPUList: []int{testCPUTwo, testCPUThree, testCPUFour, testCPUFive}}, // 4 CPUs
			nodeTwo:  {ID: nodeTwo, CPUList: []int{testCPUSix, testCPUSeven, testCPUEight}},             // 3 CPUs
		},
	}
}

// Helper function to validate error expectation
func validateErrorExpectation(t *testing.T, err error, hasError bool, testName string) bool {
	if hasError {
		if err == nil {
			t.Errorf("%s: expected error but got none", testName)
			return false
		}
		return true
	}
	if err != nil {
		t.Errorf("%s: unexpected error: %v", testName, err)
		return false
	}
	return true
}

// Test cases for parseCPUList
func getParseCPUListTestCases() []parseCPUListTestCase {
	return []parseCPUListTestCase{
		{
			name:     "single CPU",
			input:    "0",
			expected: []int{testCPUZero},
			hasError: false,
		},
		{
			name:     "multiple single CPUs",
			input:    "0,2,4",
			expected: []int{testCPUZero, testCPUTwo, testCPUFour},
			hasError: false,
		},
		{
			name:     "CPU range",
			input:    "0-3",
			expected: []int{testCPUZero, testCPUOne, testCPUTwo, testCPUThree},
			hasError: false,
		},
		{
			name:     "mixed format",
			input:    "0-2,5,7-9",
			expected: []int{testCPUZero, testCPUOne, testCPUTwo, testCPUFive, testCPUSeven, testCPUEight, testCPUNine},
			hasError: false,
		},
		{
			name:     "whitespace trimmed",
			input:    " 0-2,5 ",
			expected: []int{testCPUZero, testCPUOne, testCPUTwo, testCPUFive},
			hasError: false,
		},
		{
			name:     "empty string",
			input:    "",
			expected: []int{},
			hasError: false,
		},
	}
}

// Test cases for parseCPUList error scenarios
func getParseCPUListErrorTestCases() []parseCPUListTestCase {
	return []parseCPUListTestCase{
		{
			name:     "invalid range format",
			input:    "0-2-4",
			expected: nil,
			hasError: true,
		},
		{
			name:     "invalid CPU number",
			input:    "0,abc",
			expected: nil,
			hasError: true,
		},
		{
			name:     "invalid range start",
			input:    "abc-2",
			expected: nil,
			hasError: true,
		},
		{
			name:     "invalid range end",
			input:    "0-abc",
			expected: nil,
			hasError: true,
		},
	}
}

func TestGivenValidAndInvalidCPUListWhenParseThenExpectedResult(t *testing.T) {
	testCases := append(getParseCPUListTestCases(), getParseCPUListErrorTestCases()...)

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseCPUList(tt.input)
			if !validateErrorExpectation(t, err, tt.hasError, tt.name) {
				return
			}
			if tt.hasError {
				return
			}
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test cases for CPUListToString
func getCPUListToStringTestCases() []cpuListToStringTestCase {
	return []cpuListToStringTestCase{
		{
			name:     "empty list",
			input:    []int{},
			expected: "",
		},
		{
			name:     "single CPU",
			input:    []int{testCPUZero},
			expected: "0",
		},
		{
			name:     "consecutive range",
			input:    []int{testCPUZero, testCPUOne, testCPUTwo, testCPUThree},
			expected: "0-3",
		},
		{
			name:     "multiple single CPUs",
			input:    []int{testCPUZero, testCPUTwo, testCPUFour},
			expected: "0,2,4",
		},
		{
			name:     "mixed ranges and singles",
			input:    []int{testCPUZero, testCPUOne, testCPUTwo, testCPUFive, testCPUSeven, testCPUEight, testCPUNine},
			expected: "0-2,5,7-9",
		},
		{
			name:     "unsorted input",
			input:    []int{testCPUFive, testCPUZero, testCPUTwo, testCPUOne},
			expected: "0-2,5",
		},
		{
			name:     "two consecutive pairs",
			input:    []int{testCPUZero, testCPUOne, testCPUThree, testCPUFour},
			expected: "0-1,3-4",
		},
	}
}

func TestGivenCPUListWhenConvertToStringThenExpectedString(t *testing.T) {
	tests := getCPUListToStringTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CPUListToString(tt.input)
			if result != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, result)
			}
		})
	}
}

// Test cases for GetNumaNodeForCPU
func getNumaNodeForCPUTestCases() []numaNodeTestCase {
	return []numaNodeTestCase{
		{
			name:       "CPU in node 0",
			cpu:        testCPUTwo,
			expectedID: nodeZero,
			found:      true,
		},
		{
			name:       "CPU in node 1",
			cpu:        testCPUSix,
			expectedID: nodeOne,
			found:      true,
		},
		{
			name:       "CPU not found",
			cpu:        testCPUTen,
			expectedID: -1,
			found:      false,
		},
	}
}

func TestGivenTopologyWhenGetNumaNodeForCPUThenExpectedNodeID(t *testing.T) {
	topology := createTestTopology()
	tests := getNumaNodeForCPUTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeID, found := topology.GetNumaNodeForCPU(tt.cpu)
			if found != tt.found {
				t.Errorf("expected found=%v, got found=%v", tt.found, found)
			}
			if nodeID != tt.expectedID {
				t.Errorf("expected nodeID=%d, got nodeID=%d", tt.expectedID, nodeID)
			}
		})
	}
}

// Test cases for GetNumaNodeCPUs
func getNumaNodeCPUsTestCases() []numaNodeCPUsTestCase {
	return []numaNodeCPUsTestCase{
		{
			name:         "valid node 0",
			nodeID:       nodeZero,
			expectedCPUs: []int{testCPUZero, testCPUOne, testCPUTwo, testCPUThree},
			hasError:     false,
		},
		{
			name:         "valid node 1",
			nodeID:       nodeOne,
			expectedCPUs: []int{testCPUFour, testCPUFive, testCPUSix, testCPUSeven},
			hasError:     false,
		},
		{
			name:         "invalid node",
			nodeID:       nodeFive,
			expectedCPUs: nil,
			hasError:     true,
		},
	}
}

func TestGivenTopologyWhenGetNumaNodeCPUsThenExpectedCPUs(t *testing.T) {
	topology := createTestTopology()
	tests := getNumaNodeCPUsTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cpus, err := topology.GetNumaNodeCPUs(tt.nodeID)
			if !validateErrorExpectation(t, err, tt.hasError, tt.name) {
				return
			}
			if tt.hasError {
				return
			}
			if !reflect.DeepEqual(cpus, tt.expectedCPUs) {
				t.Errorf("expected %v, got %v", tt.expectedCPUs, cpus)
			}
		})
	}
}

// Test cases for GetBestNumaNodeForAffinity
func getBestNumaNodeTestCases() []bestNumaNodeTestCase {
	return []bestNumaNodeTestCase{
		{
			name:           "request 1 CPU - should get node with most CPUs (node 1)",
			minCPUCount:    minCPUOne,
			expectedNodeID: nodeOne,
			hasError:       false,
		},
		{
			name:           "request 3 CPUs - should get node 1",
			minCPUCount:    minCPUThree,
			expectedNodeID: nodeOne,
			hasError:       false,
		},
		{
			name:           "request 4 CPUs - should get node 1",
			minCPUCount:    minCPUFour,
			expectedNodeID: nodeOne,
			hasError:       false,
		},
		{
			name:           "request more CPUs than available",
			minCPUCount:    minCPUTen,
			expectedNodeID: -1,
			hasError:       true,
		},
	}
}

func TestGivenTopologyWhenGetBestNumaNodeForAffinityThenExpectedNodeID(t *testing.T) {
	topology := createBestNodeTestTopology()
	tests := getBestNumaNodeTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeID, err := topology.GetBestNumaNodeForAffinity(tt.minCPUCount)
			if !validateErrorExpectation(t, err, tt.hasError, tt.name) {
				return
			}
			if tt.hasError {
				return
			}
			if nodeID != tt.expectedNodeID {
				t.Errorf("expected nodeID=%d, got nodeID=%d", tt.expectedNodeID, nodeID)
			}
		})
	}
}

func TestGivenNumaResourceSpecWhenValidateThenExpectedValues(t *testing.T) {
	spec := createTestNumaResourceSpec()
	validateNumaResourceSpec(t, spec)
}

// Helper function to create test NumaResourceSpec
func createTestNumaResourceSpec() NumaResourceSpec {
	return NumaResourceSpec{
		CPUset: "0-3,6,8-10",
		NUMANodeResourcesMap: map[int]v1.ResourceList{
			nodeZero: {
				v1.ResourceCPU:    *resource.NewQuantity(testCPUTwo, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memoryOneGB, resource.BinarySI),
			},
			nodeOne: {
				v1.ResourceCPU:    *resource.NewQuantity(testCPUFour, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memoryTwoGB, resource.BinarySI),
			},
		},
	}
}

// Helper function to validate NumaResourceSpec
func validateNumaResourceSpec(t *testing.T, spec NumaResourceSpec) {
	expectedCPUset := "0-3,6,8-10"
	if spec.CPUset != expectedCPUset {
		t.Errorf("expected CPUset to be '%s', got '%s'", expectedCPUset, spec.CPUset)
	}

	expectedNodeCount := 2
	if len(spec.NUMANodeResourcesMap) != expectedNodeCount {
		t.Errorf("expected %d NUMA nodes, got %d", expectedNodeCount, len(spec.NUMANodeResourcesMap))
	}

	validateNodeResources(t, spec.NUMANodeResourcesMap)
}

// Helper function to validate node resources
func validateNodeResources(t *testing.T, nodeResourcesMap map[int]v1.ResourceList) {
	node0Resources := nodeResourcesMap[nodeZero]
	if cpu := node0Resources[v1.ResourceCPU]; cpu.Value() != testCPUTwo {
		t.Errorf("expected node 0 CPU to be %d, got %d", testCPUTwo, cpu.Value())
	}

	node1Resources := nodeResourcesMap[nodeOne]
	if cpu := node1Resources[v1.ResourceCPU]; cpu.Value() != testCPUFour {
		t.Errorf("expected node 1 CPU to be %d, got %d", testCPUFour, cpu.Value())
	}
}

func TestGivenWindowsOrNoHardwareWhenGetNumaTopologyThenSkip(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Skipping GetNumaTopology test - not supported on Windows")
		return
	}
	t.Skip("Skipping GetNumaTopology test - requires real NUMA hardware")
}

func TestGivenConcurrentAccessWhenNumaTopologyThenNoRace(t *testing.T) {
	topology := createTestTopology()
	done := make(chan bool, goroutineCount)

	for i := 0; i < goroutineCount; i++ {
		go runConcurrentTest(topology, done)
	}

	waitForConcurrentTests(done)
}

// Helper function to run concurrent test
func runConcurrentTest(topology *NumaTopology, done chan bool) {
	defer func() { done <- true }()

	for j := 0; j < iterationCount; j++ {
		_, _ = topology.GetNumaNodeCPUs(nodeZero)
		_, _ = topology.GetNumaNodeForCPU(testCPUTwo)
		_, _ = topology.GetBestNumaNodeForAffinity(testCPUTwo)
	}
}

// Helper function to wait for concurrent tests
func waitForConcurrentTests(done chan bool) {
	for i := 0; i < goroutineCount; i++ {
		<-done
	}
}

func TestGivenEmptyInputWhenParseCPUListThenEmptyResult(t *testing.T) {
	result, err := parseCPUList("")
	if err != nil {
		t.Errorf("unexpected error for empty input: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("expected empty result for empty input, got %v", result)
	}
}
