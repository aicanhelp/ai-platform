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

package model

import (
	"errors"
	"reflect"
	"testing"

	"monitoring-dashboard-service/pkg/config/global"
)

// LineGraph test cases
func TestLineGraphGetBoardMin(t *testing.T) {
	lineGraph := NewLineGraph(LineGraphType)
	expected := 0
	actual := lineGraph.GetBoardMin()
	if actual != expected {
		t.Errorf("Expected GetBoardMin to return %d, but got %d", expected, actual)
	}
}

func TestLineGraphGetBoardMax(t *testing.T) {
	lineGraph := NewLineGraph(LineGraphType)
	expected := 0
	actual := lineGraph.GetBoardMax()
	if actual != expected {
		t.Errorf("Expected GetBoardMax to return %d, but got %d", expected, actual)
	}
}

func TestLineGraphGetInstancesMax(t *testing.T) {
	lineGraph := NewLineGraph(LineGraphType)
	expected := global.MaxInstanceOfWidget
	actual := lineGraph.GetInstancesMax()
	if actual != expected {
		t.Errorf("Expected GetInstancesMax to return %d, but got %d", expected, actual)
	}
}

func TestLineGraphValidate(t *testing.T) {
	lineGraph := NewLineGraph(LineGraphType)
	actual := lineGraph.Validate()
	if actual != nil {
		t.Errorf("Expected Validate to return nil, but got %v", actual)
	}

	invalidGraph := NewLineGraph(StackedAreaGraphType)
	expectedError := errors.New("lineGraph type error")
	actualError := invalidGraph.Validate()
	if errors.Is(actualError, nil) {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
	if actualError.Error() != expectedError.Error() {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
}

func TestLineGraphGetGraphType(t *testing.T) {
	lineGraph := NewLineGraph(LineGraphType)
	expected := LineGraphType
	actual := lineGraph.GetGraphType()
	if actual != expected {
		t.Errorf("Expected GetGraphType to return %v, but got %v", expected, actual)
	}
}

// StackedAreaGraph test cases
func TestStackedAreaGraphGetBoardMin(t *testing.T) {
	stackedAreaGraph := NewStackedArea(StackedAreaGraphType)
	expected := 0
	actual := stackedAreaGraph.GetBoardMin()
	if actual != expected {
		t.Errorf("Expected GetBoardMin to return %d, but got %d", expected, actual)
	}
}

func TestStackedAreaGraphGetBoardMax(t *testing.T) {
	stackedAreaGraph := NewStackedArea(StackedAreaGraphType)
	expected := 0
	actual := stackedAreaGraph.GetBoardMax()
	if actual != expected {
		t.Errorf("Expected GetBoardMax to return %d, but got %d", expected, actual)
	}
}

func TestStackedAreaGraphGetInstancesMax(t *testing.T) {
	stackedAreaGraph := NewStackedArea(StackedAreaGraphType)
	expected := global.MaxInstanceOfStackedArea
	actual := stackedAreaGraph.GetInstancesMax()
	if actual != expected {
		t.Errorf("Expected GetInstancesMax to return %d, but got %d", expected, actual)
	}
}

func TestStackedAreaGraphValidate(t *testing.T) {
	stackedAreaGraph := NewStackedArea(StackedAreaGraphType)
	actual := stackedAreaGraph.Validate()
	if actual != nil {
		t.Errorf("Expected Validate to return nil, but got %v", actual)
	}

	invalidGraph := NewStackedArea(NumbersGraphType)
	expectedError := errors.New("stackedAreaGraph type error")
	actualError := invalidGraph.Validate()
	if errors.Is(actualError, nil) {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
	if actualError.Error() != expectedError.Error() {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
}

func TestStackedAreaGraphGetGraphType(t *testing.T) {
	stackedAreaGraph := NewStackedArea(StackedAreaGraphType)
	expected := StackedAreaGraphType
	actual := stackedAreaGraph.GetGraphType()
	if actual != expected {
		t.Errorf("Expected GetGraphType to return %v, but got %v", expected, actual)
	}
}

// NewNumbers test cases
func TestNumbersGraphGetBoardMin(t *testing.T) {
	numbersGraph := NewNumbers(NumbersGraphType)
	expected := 0
	actual := numbersGraph.GetBoardMin()
	if actual != expected {
		t.Errorf("Expected GetBoardMin to return %d, but got %d", expected, actual)
	}
}

func TestNumbersGraphGetBoardMax(t *testing.T) {
	numbersGraph := NewNumbers(NumbersGraphType)
	expected := 0
	actual := numbersGraph.GetBoardMax()
	if actual != expected {
		t.Errorf("Expected GetBoardMax to return %d, but got %d", expected, actual)
	}
}

func TestNumbersGraphGetInstancesMax(t *testing.T) {
	numbersGraph := NewNumbers(NumbersGraphType)
	expected := global.MaxInstanceOfWidget
	actual := numbersGraph.GetInstancesMax()
	if actual != expected {
		t.Errorf("Expected GetInstancesMax to return %d, but got %d", expected, actual)
	}
}

func TestNumbersGraphValidate(t *testing.T) {
	numbersGraph := NewNumbers(NumbersGraphType)
	actual := numbersGraph.Validate()
	if actual != nil {
		t.Errorf("Expected Validate to return nil, but got %v", actual)
	}

	invalidGraph := NewNumbers(DashboardGraphType)
	expectedError := errors.New("numbersGraph type error")
	actualError := invalidGraph.Validate()
	if errors.Is(actualError, expectedError) {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
}

func TestNumbersGraphGetGraphType(t *testing.T) {
	numbersGraph := NewNumbers(NumbersGraphType)
	expected := NumbersGraphType
	actual := numbersGraph.GetGraphType()
	if actual != expected {
		t.Errorf("Expected GetGraphType to return %v, but got %v", expected, actual)
	}
}

const (
	tbMax = 100
)

// BoardGraph test cases
func TestBoardGraphGetBoardMin(t *testing.T) {
	boardGraph := NewBoardGraph(DashboardGraphType, 0, tbMax)
	expected := 0
	actual := boardGraph.GetBoardMin()
	if actual != expected {
		t.Errorf("Expected GetBoardMin to return %d, but got %d", expected, actual)
	}
}

func TestBoardGraphGetBoardMax(t *testing.T) {
	boardGraph := NewBoardGraph(DashboardGraphType, 0, tbMax)
	expected := tbMax
	actual := boardGraph.GetBoardMax()
	if actual != expected {
		t.Errorf("Expected GetBoardMax to return %d, but got %d", expected, actual)
	}
}

func TestBoardGraphGetInstancesMax(t *testing.T) {
	boardGraph := NewBoardGraph(DashboardGraphType, 0, tbMax)
	expected := global.MaxInstanceOfWidget
	actual := boardGraph.GetInstancesMax()
	if actual != expected {
		t.Errorf("Expected GetInstancesMax to return %d, but got %d", expected, actual)
	}
}

func TestBoardGraphValidate(t *testing.T) {
	boardGraph := NewBoardGraph(DashboardGraphType, 0, tbMax)
	actual := boardGraph.Validate()
	if actual != nil {
		t.Errorf("Expected Validate to return nil, but got %v", actual)
	}

	invalidGraph := NewBoardGraph(RingGraphType, 0, tbMax)
	expectedError := errors.New("boardGraph type error")
	actualError := invalidGraph.Validate()
	if errors.Is(actualError, nil) {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
	if actualError.Error() != expectedError.Error() {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}

	boardGraph = NewBoardGraph(DashboardGraphType, tbMax, 0)
	expectedError = errors.New("board min must be less than board max")
	actualError = boardGraph.Validate()
	if errors.Is(actualError, nil) {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
	if actualError.Error() != expectedError.Error() {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
}

func TestBoardGraphGetGraphType(t *testing.T) {
	boardGraph := NewBoardGraph(DashboardGraphType, 0, tbMax)
	expected := DashboardGraphType
	actual := boardGraph.GetGraphType()
	if actual != expected {
		t.Errorf("Expected GetGraphType to return %v, but got %v", expected, actual)
	}
}

// RingGraph test cases
func TestRingGraphGetBoardMin(t *testing.T) {
	ringGraph := NewRingGraph(RingGraphType)
	expected := 0
	actual := ringGraph.GetBoardMin()
	if actual != expected {
		t.Errorf("Expected GetBoardMin to return %d, but got %d", expected, actual)
	}
}

func TestRingGraphGetBoardMax(t *testing.T) {
	ringGraph := NewRingGraph(RingGraphType)
	expected := 0
	actual := ringGraph.GetBoardMax()
	if actual != expected {
		t.Errorf("Expected GetBoardMax to return %d, but got %d", expected, actual)
	}
}

func TestRingGraphGetInstancesMax(t *testing.T) {
	ringGraph := NewRingGraph(RingGraphType)
	expected := global.MaxInstanceOfWidget
	actual := ringGraph.GetInstancesMax()
	if actual != expected {
		t.Errorf("Expected GetInstancesMax to return %d, but got %d", expected, actual)
	}
}

func TestRingGraphValidate(t *testing.T) {
	ringGraph := NewRingGraph(RingGraphType)
	actual := ringGraph.Validate()
	if actual != nil {
		t.Errorf("Expected Validate to return nil, but got %v", actual)
	}

	invalidGraph := NewRingGraph(BarGraphType)
	expectedError := errors.New("ringGraph type error")
	actualError := invalidGraph.Validate()
	if errors.Is(actualError, nil) {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
	if actualError.Error() != expectedError.Error() {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
}

func TestRingGraphGetGraphType(t *testing.T) {
	ringGraph := NewRingGraph(RingGraphType)
	expected := RingGraphType
	actual := ringGraph.GetGraphType()
	if actual != expected {
		t.Errorf("Expected GetGraphType to return %v, but got %v", expected, actual)
	}
}

// BarGraph test cases
func TestBarGraphGetBoardMin(t *testing.T) {
	barGraph := NewBarGraph(BarGraphType)
	expected := 0
	actual := barGraph.GetBoardMin()
	if actual != expected {
		t.Errorf("Expected GetBoardMin to return %d, but got %d", expected, actual)
	}
}

func TestBarGraphGetBoardMax(t *testing.T) {
	barGraph := NewBarGraph(BarGraphType)
	expected := 0
	actual := barGraph.GetBoardMax()
	if actual != expected {
		t.Errorf("Expected GetBoardMax to return %d, but got %d", expected, actual)
	}
}

func TestBarGraphGetInstancesMax(t *testing.T) {
	barGraph := NewBarGraph(BarGraphType)
	expected := global.MaxInstanceOfWidget
	actual := barGraph.GetInstancesMax()
	if actual != expected {
		t.Errorf("Expected GetInstancesMax to return %d, but got %d", expected, actual)
	}
}

func TestBarGraphValidate(t *testing.T) {
	barGraph := NewBarGraph(BarGraphType)
	actual := barGraph.Validate()
	if actual != nil {
		t.Errorf("Expected Validate to return nil, but got %v", actual)
	}

	invalidGraph := NewBarGraph(LineGraphType)
	expectedError := errors.New("barGraph type error")
	actualError := invalidGraph.Validate()
	if errors.Is(actualError, nil) {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
	if actualError.Error() != expectedError.Error() {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedError, actualError)
	}
}

func TestBarGraphGetGraphType(t *testing.T) {
	barGraph := NewBarGraph(BarGraphType)
	expected := BarGraphType
	actual := barGraph.GetGraphType()
	if actual != expected {
		t.Errorf("Expected GetGraphType to return %v, but got %v", expected, actual)
	}
}

// GraphFactoryByWidgetVO test cases
func TestGraphFactoryByWidgetVO(t *testing.T) {
	tests := []struct {
		graphType WidgetGraphType
		vo        *WidgetVO
		expected  IGraph
		err       error
	}{
		{
			graphType: LineGraphType, vo: &WidgetVO{},
			expected: NewLineGraph(LineGraphType), err: nil,
		},
		{
			graphType: StackedAreaGraphType, vo: &WidgetVO{},
			expected: NewStackedArea(StackedAreaGraphType), err: nil,
		},
		{
			graphType: NumbersGraphType, vo: &WidgetVO{},
			expected: NewNumbers(NumbersGraphType), err: nil,
		},
		{
			graphType: DashboardGraphType, vo: &WidgetVO{BoardMin: "0", BoardMax: "100"},
			expected: NewBoardGraph(DashboardGraphType, 0, tbMax), err: nil,
		},
		{
			graphType: RingGraphType, vo: &WidgetVO{},
			expected: NewRingGraph(RingGraphType), err: nil,
		},
		{
			graphType: BarGraphType,
			vo:        &WidgetVO{},
			expected:  NewBarGraph(BarGraphType),
			err:       nil,
		},
	}
	for _, test := range tests {
		actual, err := GraphFactoryByWidgetVO(test.graphType, test.vo)
		if !errors.Is(err, test.err) {
			t.Errorf("Expected GraphFactoryByWidgetVO to return error %v, but got %v", test.err, err)
		}
		if !reflect.DeepEqual(actual, test.expected) {
			t.Errorf("Expected GraphFactoryByWidgetVO to return %v, but got %v", test.expected, actual)
		}
	}
}

// GraphFactoryByWidgetCRD test cases
func TestGraphFactoryByWidgetCRD(t *testing.T) {
	tests := []struct {
		crd      *WidgetCRD
		expected IGraph
		err      error
	}{
		{
			crd:      &WidgetCRD{Spec: WidgetSpec{WidgetType: LineGraphType}},
			expected: NewLineGraph(LineGraphType),
			err:      nil,
		},
		{
			crd:      &WidgetCRD{Spec: WidgetSpec{WidgetType: StackedAreaGraphType}},
			expected: NewStackedArea(StackedAreaGraphType),
			err:      nil,
		},
		{
			crd:      &WidgetCRD{Spec: WidgetSpec{WidgetType: NumbersGraphType}},
			expected: NewNumbers(NumbersGraphType),
			err:      nil,
		},
		{
			crd:      &WidgetCRD{Spec: WidgetSpec{WidgetType: DashboardGraphType, BoardMin: 0, BoardMax: tbMax}},
			expected: NewBoardGraph(DashboardGraphType, 0, tbMax),
			err:      nil,
		},
		{
			crd:      &WidgetCRD{Spec: WidgetSpec{WidgetType: RingGraphType}},
			expected: NewRingGraph(RingGraphType),
			err:      nil,
		},
		{
			crd:      &WidgetCRD{Spec: WidgetSpec{WidgetType: BarGraphType}},
			expected: NewBarGraph(BarGraphType),
			err:      nil,
		},
	}

	for _, test := range tests {
		actual, err := GraphFactoryByWidgetCRD(test.crd)
		if !errors.Is(err, test.err) {
			t.Errorf("Expected GraphFactoryByWidgetCRD to return error %v, but got %v", test.err, err)
		}
		if !reflect.DeepEqual(actual, test.expected) {
			t.Errorf("Expected GraphFactoryByWidgetCRD to return %v, but got %v", test.expected, actual)
		}
	}
}
