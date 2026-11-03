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
)

// 全局测试变量
var (
	datasourceURL  = "http://prometheus-k8s.monitoring.svc.cluster.local:9090"
	validStartTime = "2024-01-01T00:00:00Z"
	validEndTime   = "2024-01-01T01:00:00Z"
	validStep      = indicatorSixty
	validInstance  = "instance1"
	validInstances = []string{"instance1", "instance2"}

	// 数字常量
	indicatorSixty                   = "60"
	indicatorThreeHundred            = "300"
	indicatorThreeThousandSixHundred = "3600"
	// 时间常量
	indicatorTestDifferentEndTime   = "2024-01-01T02:00:00Z"
	indicatorTestEndBeforeStartTime = "2024-01-01T01:00:00Z"
	indicatorTestStartAfterEndTime  = "2024-01-01T00:00:00Z"
	indicatorTestShortEndTime       = "2024-01-01T00:30:00Z"
)

// 辅助函数：创建预期的InstanceQuery
func createExpectedInstanceQuery(dataSource, start, end, step string, instances []string) *InstanceQuery {
	return &InstanceQuery{
		DataSource: dataSource,
		Start:      start,
		End:        end,
		Step:       step,
		Instances:  instances,
	}
}

// NewInstanceQuery test cases
func TestNewInstanceQueryValidInput(t *testing.T) {
	expected := createExpectedInstanceQuery(datasourceURL, validStartTime, validEndTime, validStep, validInstances)
	actual := NewInstanceQuery(datasourceURL, validStartTime, validEndTime, validStep, validInstances)

	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("Expected NewInstanceQuery to return %v, but got %v", expected, actual)
	}
}

func TestNewInstanceQueryEmptyAndNilInstances(t *testing.T) {
	var emptyInstances []string
	var nilInstances []string

	tests := []struct {
		name      string
		instances []string
	}{
		{"EmptyInstances", emptyInstances},
		{"NilInstances", nilInstances},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			expected := createExpectedInstanceQuery(datasourceURL, validStartTime, validEndTime, validStep, test.instances)
			actual := NewInstanceQuery(datasourceURL, validStartTime, validEndTime, validStep, test.instances)

			if !reflect.DeepEqual(actual, expected) {
				t.Errorf("Expected NewInstanceQuery to return %v, but got %v", expected, actual)
			}
		})
	}
}

// 测试参数结构体
type instanceQueryTestParams struct {
	dataSource string
	start      string
	end        string
	step       string
	instances  []string
}

// 辅助函数：创建InstanceQuery并验证
func validateInstanceQuery(t *testing.T, params instanceQueryTestParams) {
	instanceQuery := &InstanceQuery{
		DataSource: params.dataSource,
		Start:      params.start,
		End:        params.end,
		Step:       params.step,
		Instances:  params.instances,
	}
	err := instanceQuery.Validate()
	if !errors.Is(err, nil) {
		t.Errorf("Expected Validate to return nil, but got %v", err)
	}
}

// 辅助函数：验证InstanceQuery返回预期错误
func validateInstanceQueryWithError(t *testing.T, params instanceQueryTestParams, expectedError string) {
	instanceQuery := &InstanceQuery{
		DataSource: params.dataSource,
		Start:      params.start,
		End:        params.end,
		Step:       params.step,
		Instances:  params.instances,
	}
	expectedErr := errors.New(expectedError)
	err := instanceQuery.Validate()
	if errors.Is(err, nil) {
		t.Errorf("Expected Validate to return error %v, but got nil", expectedErr)
	}
	if err.Error() != expectedErr.Error() {
		t.Errorf("Expected Validate to return error %v, but got %v", expectedErr, err)
	}
}

// Validate test cases
func TestInstanceQueryValidateValidInputs(t *testing.T) {
	singleInstance := []string{validInstance}
	tests := []struct {
		nameValidInputs       string
		dataSourceValidInputs string
		startValidInputs      string
		endValidInputs        string
		stepValidInputs       string
		instancesValidInputs  []string
	}{
		{
			nameValidInputs:       "ValidWithMultipleInstances",
			dataSourceValidInputs: datasourceURL,
			startValidInputs:      validStartTime,
			endValidInputs:        validEndTime,
			stepValidInputs:       validStep,
			instancesValidInputs:  validInstances,
		},
		{
			nameValidInputs:       "ValidWithSingleInstance",
			dataSourceValidInputs: datasourceURL,
			startValidInputs:      validStartTime,
			endValidInputs:        validEndTime,
			stepValidInputs:       validStep,
			instancesValidInputs:  singleInstance,
		},
	}

	for _, test := range tests {
		t.Run(test.nameValidInputs, func(t *testing.T) {
			validateInstanceQuery(t, instanceQueryTestParams{
				dataSource: test.dataSourceValidInputs,
				start:      test.startValidInputs,
				end:        test.endValidInputs,
				step:       test.stepValidInputs,
				instances:  test.instancesValidInputs,
			})
		})
	}
}

func TestInstanceQueryValidateValidWithDifferentConfigs(t *testing.T) {
	singleInstance := []string{validInstance}

	tests := []struct {
		nameConfigs       string
		dataSourceConfigs string
		startConfigs      string
		endConfigs        string
		stepConfigs       string
		instancesConfigs  []string
	}{
		{
			nameConfigs:       "ValidWithDifferentStep",
			dataSourceConfigs: datasourceURL,
			startConfigs:      validStartTime,
			endConfigs:        indicatorTestDifferentEndTime,
			stepConfigs:       indicatorThreeHundred,
			instancesConfigs:  singleInstance,
		},
	}

	for _, test := range tests {
		t.Run(test.nameConfigs, func(t *testing.T) {
			validateInstanceQuery(t, instanceQueryTestParams{
				dataSource: test.dataSourceConfigs,
				start:      test.startConfigs,
				end:        test.endConfigs,
				step:       test.stepConfigs,
				instances:  test.instancesConfigs,
			})
		})
	}
}

func TestInstanceQueryValidateInvalidInputs(t *testing.T) {
	InvalidInputs := []struct {
		nameInvalidInputs       string
		dataSourceInvalidInputs string
		startInvalidInputs      string
		endInvalidInputs        string
		stepInvalidInputs       string
		instancesInvalidInputs  []string
		expectedError           string
	}{{
		nameInvalidInputs:       "InvalidDataSource",
		dataSourceInvalidInputs: "invalid-host",
		startInvalidInputs:      validStartTime,
		endInvalidInputs:        validEndTime,
		stepInvalidInputs:       validStep,
		instancesInvalidInputs:  []string{validInstance},
		expectedError:           "invalid host format",
	},
		{
			nameInvalidInputs:       "InvalidStartTime",
			dataSourceInvalidInputs: datasourceURL,
			startInvalidInputs:      "invalid-time",
			endInvalidInputs:        validEndTime,
			stepInvalidInputs:       validStep,
			instancesInvalidInputs:  []string{validInstance},
			expectedError:           "time format is not valid",
		},
	}

	for _, test := range InvalidInputs {
		t.Run(test.nameInvalidInputs, func(t *testing.T) {
			validateInstanceQueryWithError(t, instanceQueryTestParams{
				dataSource: test.dataSourceInvalidInputs,
				start:      test.startInvalidInputs,
				end:        test.endInvalidInputs,
				step:       test.stepInvalidInputs,
				instances:  test.instancesInvalidInputs,
			}, test.expectedError)
		})
	}
}

func TestInstanceQueryValidateInvalidTimeAndStep(t *testing.T) {
	testsInvalidTimeAndStep := []struct {
		nameInvalidInputs       string
		dataSourceInvalidInputs string
		startInvalidInputs      string
		endInvalidInputs        string
		stepInvalidInputs       string
		instancesInvalidInputs  []string
		expectedError           string
	}{{
		nameInvalidInputs:       "InvalidEndTime",
		dataSourceInvalidInputs: datasourceURL,
		startInvalidInputs:      validStartTime,
		endInvalidInputs:        "invalid-time",
		stepInvalidInputs:       validStep,
		instancesInvalidInputs:  []string{validInstance},
		expectedError:           "time format is not valid",
	},
		{
			nameInvalidInputs:       "InvalidStep",
			dataSourceInvalidInputs: datasourceURL,
			startInvalidInputs:      validStartTime,
			endInvalidInputs:        validEndTime,
			stepInvalidInputs:       "invalid-step",
			instancesInvalidInputs:  []string{validInstance},
			expectedError:           "invalid step duration",
		},
	}

	for _, test := range testsInvalidTimeAndStep {
		t.Run(test.nameInvalidInputs, func(t *testing.T) {
			validateInstanceQueryWithError(t, instanceQueryTestParams{
				dataSource: test.dataSourceInvalidInputs, start: test.startInvalidInputs,
				end: test.endInvalidInputs, step: test.stepInvalidInputs, instances: test.instancesInvalidInputs,
			}, test.expectedError)
		})
	}
}

func TestInstanceQueryValidateTimeLogicErrors(t *testing.T) {
	tests := []struct {
		nameTimeLogic       string
		dataSourceTimeLogic string
		start               string
		end                 string
		step                string
		instances           []string
		expectedError       string
	}{
		{
			nameTimeLogic: "EndBeforeStart", dataSourceTimeLogic: datasourceURL,
			start: indicatorTestEndBeforeStartTime, end: indicatorTestStartAfterEndTime,
			step:      validStep,
			instances: []string{validInstance}, expectedError: "end time is less than start time",
		},
		{
			nameTimeLogic: "StepTooLarge", dataSourceTimeLogic: datasourceURL, start: validStartTime,
			end: indicatorTestShortEndTime, step: indicatorThreeThousandSixHundred,
			instances: []string{validInstance}, expectedError: "end time is less than start time plus step",
		},
	}

	for _, test := range tests {
		t.Run(test.nameTimeLogic, func(t *testing.T) {
			validateInstanceQueryWithError(t, instanceQueryTestParams{
				dataSource: test.dataSourceTimeLogic,
				start:      test.start,
				end:        test.end,
				step:       test.step,
				instances:  test.instances,
			}, test.expectedError)
		})
	}
}

func TestInstanceQueryValidateInstanceErrors(t *testing.T) {
	var emptyInstances []string
	instancesWithEmpty := []string{validInstance, ""}
	instancesWithFirstEmpty := []string{"", "instance2"}
	tests := []struct {
		nameInstanceErrors       string
		dataSourceInstanceErrors string
		start                    string
		end                      string
		step                     string
		instances                []string
		expectedError            string
	}{
		{
			nameInstanceErrors: "EmptyInstances", dataSourceInstanceErrors: datasourceURL,
			start: validStartTime, end: validEndTime,
			step: validStep, instances: emptyInstances,
			expectedError: "instances List is empty",
		},
		{
			nameInstanceErrors: "EmptyInstanceString", dataSourceInstanceErrors: datasourceURL,
			start: validStartTime, end: validEndTime,
			step: validStep, instances: instancesWithEmpty,
			expectedError: "instance is empty string",
		},
		{
			nameInstanceErrors: "FirstInstanceEmpty", dataSourceInstanceErrors: datasourceURL,
			start: validStartTime, end: validEndTime,
			step: validStep, instances: instancesWithFirstEmpty,
			expectedError: "instance is empty string",
		},
	}
	for _, test := range tests {
		t.Run(test.nameInstanceErrors, func(t *testing.T) {
			validateInstanceQueryWithError(t, instanceQueryTestParams{
				dataSource: test.dataSourceInstanceErrors,
				start:      test.start,
				end:        test.end,
				step:       test.step,
				instances:  test.instances,
			}, test.expectedError)
		})
	}
}
