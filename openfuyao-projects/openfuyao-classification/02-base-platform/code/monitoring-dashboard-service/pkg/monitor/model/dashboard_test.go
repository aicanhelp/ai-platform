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
	"strings"
	"testing"

	"monitoring-dashboard-service/pkg/config/global"
)

// 全局测试变量
var (
	validDisplayName        = "Test Dashboard"
	validCreator            = "testuser"
	idRepeatTimes           = 36
	validDashboardID        = "2024-01-01-12-00-00-" + strings.Repeat("a", idRepeatTimes)
	dashboardValidStartTime = "2024-01-01T00:00:00Z"
	dashboardValidEndTime   = "2024-01-01T01:00:00Z"
	dashboardValidStep      = dashboardSixty

	// 数字常量
	dashboardSixty                   = "60"
	dashboardThreeThousandSixHundred = "3600"
	// 时间常量
	dashboardTestEndBeforeStartTime = "2024-01-01T01:00:00Z"
	dashboardTestStartAfterEndTime  = "2024-01-01T00:00:00Z"
	dashboardTestShortEndTime       = "2024-01-01T00:30:00Z"
)

// 辅助函数：验证DashboardVO的ValidateCreate结果
func validateDashboardCreate(t *testing.T, displayName, creator string, expectError bool, expectedError string) {
	dashboardVO := &DashboardVO{
		DisplayName: displayName,
		Creator:     creator,
	}
	err := dashboardVO.ValidateCreate()

	if expectError {
		if errors.Is(err, nil) {
			t.Errorf("Expected ValidateCreate to return error %s, but got nil", expectedError)
		} else {
			if err.Error() != expectedError {
				t.Errorf("Expected ValidateCreate to return error %s, but got %v", expectedError, err)
			}
		}
	} else {
		if err != nil {
			t.Errorf("Expected ValidateCreate to return nil, but got %v", err)
		}
	}
}

// ValidateCreate test cases
func TestDashboardVOValidateCreate(t *testing.T) {
	tests := []struct {
		name          string
		displayName   string
		creator       string
		expectError   bool
		expectedError string
	}{
		{
			name:        "ValidInput",
			displayName: validDisplayName,
			creator:     validCreator,
			expectError: false,
		},
		{
			name:          "EmptyDisplayName",
			displayName:   "",
			creator:       validCreator,
			expectError:   true,
			expectedError: "displayName is empty",
		},
		{
			name:          "EmptyCreator",
			displayName:   validDisplayName,
			creator:       "",
			expectError:   true,
			expectedError: "creator is empty",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			validateDashboardCreate(t, test.displayName, test.creator, test.expectError, test.expectedError)
		})
	}
}

func TestDashboardVOValidateCreateLengthErrors(t *testing.T) {
	tooLongDisplayName := strings.Repeat("a", global.MaxNameLength+1)
	tooLongCreator := strings.Repeat("a", global.MaxNameLength+1)

	tests := []struct {
		name          string
		displayName   string
		creator       string
		expectError   bool
		expectedError string
	}{
		{
			name:          "DisplayNameTooLong",
			displayName:   tooLongDisplayName,
			creator:       validCreator,
			expectError:   true,
			expectedError: "displayName is too long",
		},
		{
			name:          "CreatorTooLong",
			displayName:   validDisplayName,
			creator:       tooLongCreator,
			expectError:   true,
			expectedError: "creator is too long",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			validateDashboardCreate(t, test.displayName, test.creator, test.expectError, test.expectedError)
		})
	}
}

// TestDashboardVOValidatePatchUpdateDisplayNameErrors test cases
func TestDashboardVOValidatePatchUpdateDisplayNameErrors(t *testing.T) {
	tooLongDisplayName := strings.Repeat("a", global.MaxNameLength+1)
	tests := []struct {
		name          string
		dashboardID   string
		displayName   string
		expectError   bool
		expectedError string
	}{
		{name: "EmptyDisplayName", dashboardID: validDashboardID, displayName: "",
			expectError: true, expectedError: "displayName is empty"},
		{name: "DisplayNameTooLong", dashboardID: validDashboardID,
			displayName: tooLongDisplayName, expectError: true, expectedError: "displayName is too long",
		},
		{
			name:        "ValidInput",
			dashboardID: validDashboardID,
			displayName: "Updated Dashboard",
			expectError: false,
		},
		{
			name:          "EmptyID",
			dashboardID:   "",
			displayName:   "Updated Dashboard",
			expectError:   true,
			expectedError: "id is empty",
		},
		{name: "InvalidIDLength", dashboardID: "invalid-id", displayName: "Updated Dashboard",
			expectError: true, expectedError: "id length is not valid"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dashboardVO := &DashboardVO{DashboardID: test.dashboardID, DisplayName: test.displayName}
			err := dashboardVO.ValidatePatchUpdate()
			if test.expectError {
				if errors.Is(err, nil) {
					t.Errorf("Expected ValidatePatchUpdate to return error %s, but got nil", test.expectedError)
				} else {
					if err.Error() != test.expectedError {
						t.Errorf("Expected ValidatePatchUpdate to return error %s, but got %v", test.expectedError, err)
					}
				}
			} else {
				if err != nil {
					t.Errorf("Expected ValidatePatchUpdate to return nil, but got %v", err)
				}
			}
		})
	}
}

// 测试参数结构体
type dashboardPostQueryTestParams struct {
	dashboardID   string
	start         string
	end           string
	step          string
	expectError   bool
	expectedError string
}

// 辅助函数：验证DashboardVO的ValidatePostQuery结果
func validateDashboardPostQuery(t *testing.T, params dashboardPostQueryTestParams) {
	dashboardVO := &DashboardVO{
		DashboardID: params.dashboardID,
		Start:       params.start,
		End:         params.end,
		Step:        params.step,
	}
	err := dashboardVO.ValidatePostQuery()

	if params.expectError {
		if errors.Is(err, nil) {
			t.Errorf("Expected ValidatePostQuery to return error %s, but got nil", params.expectedError)
		} else {
			if err.Error() != params.expectedError {
				t.Errorf("Expected ValidatePostQuery to return error %s, but got %v", params.expectedError, err)
			}
		}
	} else {
		if err != nil {
			t.Errorf("Expected ValidatePostQuery to return nil, but got %v", err)
		}
	}
}

// ValidatePostQuery test cases
func TestDashboardVOValidatePostQueryValidInput(t *testing.T) {
	validateDashboardPostQuery(t, dashboardPostQueryTestParams{
		dashboardID:   validDashboardID,
		start:         dashboardValidStartTime,
		end:           dashboardValidEndTime,
		step:          dashboardValidStep,
		expectError:   false,
		expectedError: "",
	})
}

func TestDashboardVOValidatePostQueryInvalidID(t *testing.T) {
	tests := []struct {
		name          string
		dashboardID   string
		expectedError string
	}{
		{
			name:          "EmptyID",
			dashboardID:   "",
			expectedError: "id is empty",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			validateDashboardPostQuery(t, dashboardPostQueryTestParams{
				dashboardID:   test.dashboardID,
				start:         dashboardValidStartTime,
				end:           dashboardValidEndTime,
				step:          dashboardValidStep,
				expectError:   true,
				expectedError: test.expectedError,
			})
		})
	}
}

func TestDashboardVOValidatePostQueryInvalidTime(t *testing.T) {
	tests := []struct {
		name          string
		start         string
		end           string
		expectedError string
	}{
		{
			name:          "InvalidStartTime",
			start:         "invalid-time",
			end:           dashboardValidEndTime,
			expectedError: "time format is not valid",
		},
		{
			name:          "InvalidEndTime",
			start:         dashboardValidStartTime,
			end:           "invalid-time",
			expectedError: "time format is not valid",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			validateDashboardPostQuery(t, dashboardPostQueryTestParams{
				dashboardID:   validDashboardID,
				start:         test.start,
				end:           test.end,
				step:          dashboardValidStep,
				expectError:   true,
				expectedError: test.expectedError,
			})
		})
	}
}

func TestDashboardVOValidatePostQueryInvalidStep(t *testing.T) {
	validateDashboardPostQuery(t, dashboardPostQueryTestParams{
		dashboardID:   validDashboardID,
		start:         dashboardValidStartTime,
		end:           dashboardValidEndTime,
		step:          "invalid-step",
		expectError:   true,
		expectedError: "invalid step duration",
	})
}

func TestDashboardVOValidatePostQueryTimeLogicErrors(t *testing.T) {
	tests := []struct {
		name          string
		start         string
		end           string
		step          string
		expectedError string
	}{
		{
			name:          "EndBeforeStart",
			start:         dashboardTestEndBeforeStartTime,
			end:           dashboardTestStartAfterEndTime,
			step:          dashboardValidStep,
			expectedError: "end time is less than start time",
		},
		{
			name:          "StepTooLarge",
			start:         dashboardValidStartTime,
			end:           dashboardTestShortEndTime,
			step:          dashboardThreeThousandSixHundred,
			expectedError: "end time is less than start time plus step",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			validateDashboardPostQuery(t, dashboardPostQueryTestParams{
				dashboardID:   validDashboardID,
				start:         test.start,
				end:           test.end,
				step:          test.step,
				expectError:   true,
				expectedError: test.expectedError,
			})
		})
	}
}

// IsDisplayNameExists test cases
func TestDashboardVOIsDisplayNameExists(t *testing.T) {
	var emptyDashboards []*DashboardSpec
	tests := []struct {
		name        string
		displayName string
		dashboards  []*DashboardSpec
		expected    bool
	}{
		{
			name:        "Exists",
			displayName: "Existing Dashboard",
			dashboards: []*DashboardSpec{
				{DisplayName: "Existing Dashboard"},
				{DisplayName: "Other Dashboard"},
			},
			expected: true,
		},
		{
			name:        "NotExists",
			displayName: "New Dashboard",
			dashboards: []*DashboardSpec{
				{DisplayName: "Existing Dashboard"},
				{DisplayName: "Other Dashboard"},
			},
			expected: false,
		},
		{
			name:        "EmptyList",
			displayName: "New Dashboard",
			dashboards:  emptyDashboards,
			expected:    false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dashboardVO := &DashboardVO{DisplayName: test.displayName}
			actual := dashboardVO.IsDisplayNameExists(test.dashboards)
			if actual != test.expected {
				t.Errorf("Expected IsDisplayNameExists to return %v, but got %v", test.expected, actual)
			}
		})
	}
}
