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
	widgetDatasourceURL    = "http://prometheus-k8s.monitoring.svc.cluster.local:9090"
	validWidgetDisplayName = "Test Widget"
	widgetIDRepeatTimes    = 36
	validWidgetDashboardID = "2024-01-01-12-00-00-" + strings.Repeat("a", widgetIDRepeatTimes)
	validWidgetInstance    = "instance1"
	validWidgetInstances   = []string{"instance1", "instance2"}

	// Widget类型相关常量
	lineGraphType      = "0"
	stackedAreaType    = "1"
	numbersGraphType   = two
	dashboardGraphType = three
	ringGraphType      = four
	barGraphType       = five
	outOfRangeType     = ten

	// 默认值常量
	defaultBoardMin = "0"
	defaultBoardMax = "0"
	testBoardMin    = ten
	testBoardMax    = oneHundred

	// 数字常量
	two                           = "2"
	three                         = "3"
	four                          = "4"
	five                          = "5"
	six                           = "6"
	seven                         = "7"
	eight                         = "8"
	nine                          = "9"
	ten                           = "10"
	oneHundred                    = "100"
	widgetSixty                   = "60"
	widgetThreeHundred            = "300"
	widgetThreeThousandSixHundred = "3600"

	// 数字常量（用于函数调用）
	fiveInt = 5
	sixInt  = 6

	// 实例数量限制
	maxInstanceCount     = 8
	tooManyInstanceCount = 9

	// 测试用的实例列表
	maxValidInstances = []string{"1", two, three, four, five, six, seven, eight}
	tooManyInstances  = []string{"1", two, three, four, five, six, seven, eight, nine}

	// 有效的Widget类型列表
	validWidgetTypes                = []string{"0", "1", two, three, four, five}
	validWidgetTypesExceptDashboard = []string{"0", "1", two, four, five}
)

// GetWidgetGraphTypeFromInt test cases
func TestGetWidgetGraphTypeFromInt(t *testing.T) {
	tests := []struct {
		name     string
		input    int
		expected WidgetGraphType
	}{
		{"ValidValue0", 0, LineGraphType},
		{"ValidValue1", 1, StackedAreaGraphType},
		{"ValidValue2", 2, NumbersGraphType},
		{"ValidValue3", 3, DashboardGraphType},
		{"ValidValue4", 4, RingGraphType},
		{"ValidValue5", 5, BarGraphType},
		{"InvalidValue-1", -1, WidgetGraphType(-1)},
		{"InvalidValue6", 6, WidgetGraphType(-1)},
		{"InvalidValue10", 10, WidgetGraphType(-1)},
		{"InvalidValue100", 100, WidgetGraphType(-1)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := GetWidgetGraphTypeFromInt(test.input)
			if actual != test.expected {
				t.Errorf("Expected GetWidgetGraphTypeFromInt(%d) to return %v, but got %v",
					test.input, test.expected, actual)
			}
		})
	}
}

// ValidateCreate test cases
func TestWidgetVOValidateCreate(t *testing.T) {
	singleInstance := []string{validWidgetInstance}
	tests := []struct {
		name          string
		dashboardID   string
		displayName   string
		dataSource    string
		widgetType    string
		instances     []string
		expectError   bool
		expectedError string
	}{
		{
			name: "ValidInput", dashboardID: validWidgetDashboardID, displayName: validWidgetDisplayName,
			dataSource: widgetDatasourceURL, widgetType: lineGraphType, instances: singleInstance, expectError: false,
		},
		{
			name: "InvalidDashboardID", dashboardID: "invalid-id",
			displayName: validWidgetDisplayName, dataSource: widgetDatasourceURL,
			widgetType: lineGraphType, instances: singleInstance,
			expectError: true, expectedError: "id length is not valid",
		},
		{
			name: "EmptyDashboardID", dashboardID: "",
			displayName: validWidgetDisplayName, dataSource: widgetDatasourceURL,
			widgetType: lineGraphType, instances: singleInstance,
			expectError: true, expectedError: "id is empty",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			widgetVO := &WidgetVO{
				DashboardID: test.dashboardID, DisplayName: test.displayName, DataSource: test.dataSource,
				WidgetType: test.widgetType, Instances: test.instances,
			}
			err := widgetVO.ValidateCreate()
			if test.expectError {
				if errors.Is(err, nil) {
					t.Errorf("Expected ValidateCreate to return error %s, but got nil", test.expectedError)
				} else {
					if err.Error() != test.expectedError {
						t.Errorf("Expected ValidateCreate to return error %s, but got %v", test.expectedError, err)
					}
				}
			} else {
				if !errors.Is(err, nil) {
					t.Errorf("Expected ValidateCreate to return nil, but got %v", err)
				}
			}
		})
	}
}

// ValidatePatchUpdate test cases
func TestWidgetVOValidatePatchUpdateValidInput(t *testing.T) {
	validDisplayName := "Test Widget"
	validDataSource := widgetDatasourceURL
	validWidgetType := lineGraphType
	validInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		DisplayName: validDisplayName,
		DataSource:  validDataSource,
		WidgetType:  validWidgetType,
		Instances:   validInstances,
	}
	err := widgetVO.ValidatePatchUpdate()
	if !errors.Is(err, nil) {
		t.Errorf("Expected ValidatePatchUpdate to return nil, but got %v", err)
	}
}

func TestWidgetVOValidatePatchUpdateEmptyDisplayName(t *testing.T) {
	emptyDisplayName := ""
	validDataSource := widgetDatasourceURL
	validWidgetType := lineGraphType
	validInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		DisplayName: emptyDisplayName,
		DataSource:  validDataSource,
		WidgetType:  validWidgetType,
		Instances:   validInstances,
	}
	expectedError := errors.New("displayName is empty")
	err := widgetVO.ValidatePatchUpdate()
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got %v", expectedError, err)
	}
}

func TestWidgetVOValidatePatchUpdateInvalidDataSource(t *testing.T) {
	validDisplayName := "Test Widget"
	invalidDataSource := "invalid-host"
	validWidgetType := lineGraphType
	validInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		DisplayName: validDisplayName,
		DataSource:  invalidDataSource,
		WidgetType:  validWidgetType,
		Instances:   validInstances,
	}
	expectedError := errors.New("invalid host format")
	err := widgetVO.ValidatePatchUpdate()
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got %v", expectedError, err)
	}
}

func TestWidgetVOValidatePatchUpdateEmptyWidgetType(t *testing.T) {
	validDisplayName := "Test Widget"
	validDataSource := widgetDatasourceURL
	emptyWidgetType := ""
	validInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		DisplayName: validDisplayName,
		DataSource:  validDataSource,
		WidgetType:  emptyWidgetType,
		Instances:   validInstances,
	}
	expectedError := errors.New("widgetType is empty")
	err := widgetVO.ValidatePatchUpdate()
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got %v", expectedError, err)
	}
}

func TestWidgetVOValidatePatchUpdateInvalidWidgetType(t *testing.T) {
	validDisplayName := "Test Widget"
	validDataSource := widgetDatasourceURL
	invalidWidgetType := "invalid"
	validInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		DisplayName: validDisplayName,
		DataSource:  validDataSource,
		WidgetType:  invalidWidgetType,
		Instances:   validInstances,
	}
	expectedError := errors.New("widgetType is invalid")
	err := widgetVO.ValidatePatchUpdate()
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got %v", expectedError, err)
	}
}

// InitialGraphParams test cases
func TestWidgetVOInitialGraphParamsEmptyStrings(t *testing.T) {
	emptyBoardMin := ""
	emptyBoardMax := ""
	widgetVO := &WidgetVO{
		BoardMin: emptyBoardMin,
		BoardMax: emptyBoardMax,
	}
	widgetVO.InitialGraphParams()
	if widgetVO.BoardMin != defaultBoardMin {
		t.Errorf("Expected BoardMin to be '%s', but got %s", defaultBoardMin, widgetVO.BoardMin)
	}
	if widgetVO.BoardMax != defaultBoardMax {
		t.Errorf("Expected BoardMax to be '%s', but got %s", defaultBoardMax, widgetVO.BoardMax)
	}
}

func TestWidgetVOInitialGraphParamsNonEmptyStrings(t *testing.T) {
	validBoardMin := testBoardMin
	validBoardMax := testBoardMax
	widgetVO := &WidgetVO{
		BoardMin: validBoardMin,
		BoardMax: validBoardMax,
	}
	widgetVO.InitialGraphParams()
	if widgetVO.BoardMin != testBoardMin {
		t.Errorf("Expected BoardMin to be '%s', but got %s", testBoardMin, widgetVO.BoardMin)
	}
	if widgetVO.BoardMax != testBoardMax {
		t.Errorf("Expected BoardMax to be '%s', but got %s", testBoardMax, widgetVO.BoardMax)
	}
}

func TestWidgetVOInitialGraphParamsMixedEmpty(t *testing.T) {
	validBoardMin := testBoardMin
	emptyBoardMax := ""
	widgetVO := &WidgetVO{
		BoardMin: validBoardMin,
		BoardMax: emptyBoardMax,
	}
	widgetVO.InitialGraphParams()
	if widgetVO.BoardMin != testBoardMin {
		t.Errorf("Expected BoardMin to be '%s', but got %s", testBoardMin, widgetVO.BoardMin)
	}
	if widgetVO.BoardMax != defaultBoardMax {
		t.Errorf("Expected BoardMax to be '%s', but got %s", defaultBoardMax, widgetVO.BoardMax)
	}
}

// ValidateWidgetTypeParam test cases
func TestWidgetVOValidateWidgetTypeParamValidTypes(t *testing.T) {
	validWidgetTypes := validWidgetTypes
	for _, widgetType := range validWidgetTypes {
		widgetVO := &WidgetVO{WidgetType: widgetType}
		err := widgetVO.ValidateWidgetTypeParam()
		if !errors.Is(err, nil) {
			t.Errorf("Expected ValidateWidgetTypeParam to return nil for type %s, but got %v", widgetType, err)
		}
	}
}

func TestWidgetVOValidateWidgetTypeParamEmptyType(t *testing.T) {
	emptyWidgetType := ""
	widgetVO := &WidgetVO{WidgetType: emptyWidgetType}
	expectedError := errors.New("widgetType is empty")
	err := widgetVO.ValidateWidgetTypeParam()
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidateWidgetTypeParam to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidateWidgetTypeParam to return error %v, but got %v", expectedError, err)
	}
}

func TestWidgetVOValidateWidgetTypeParamInvalidType(t *testing.T) {
	invalidWidgetType := "abc"
	widgetVO := &WidgetVO{WidgetType: invalidWidgetType}
	expectedError := errors.New("widgetType is invalid")
	err := widgetVO.ValidateWidgetTypeParam()
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidateWidgetTypeParam to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidateWidgetTypeParam to return error %v, but got %v", expectedError, err)
	}
}

func TestWidgetVOValidateWidgetTypeParamOutOfRangeType(t *testing.T) {
	outOfRangeWidgetType := outOfRangeType
	widgetVO := &WidgetVO{WidgetType: outOfRangeWidgetType}
	expectedError := errors.New("widgetType is invalid")
	err := widgetVO.ValidateWidgetTypeParam()
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidateWidgetTypeParam to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidateWidgetTypeParam to return error %v, but got %v", expectedError, err)
	}
}

// ValidateInstancesParam test cases
func TestWidgetVOValidateInstancesParamValidInstances(t *testing.T) {
	validInstances := []string{"instance1", "instance2"}
	widgetVO := &WidgetVO{
		Instances: validInstances,
	}
	err := widgetVO.ValidateInstancesParam(1, maxInstanceCount)
	if !errors.Is(err, nil) {
		t.Errorf("Expected ValidateInstancesParam to return nil, but got %v", err)
	}
}

func TestWidgetVOValidateInstancesParamNilInstances(t *testing.T) {
	nilInstances := []string(nil)
	widgetVO := &WidgetVO{
		Instances: nilInstances,
	}
	expectedError := errors.New("instances is empty")
	err := widgetVO.ValidateInstancesParam(1, maxInstanceCount)
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidateInstancesParam to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidateInstancesParam to return error %v, but got %v", expectedError, err)
	}
}

func TestWidgetVOValidateInstancesParamTooManyInstances(t *testing.T) {
	tooManyInstances := tooManyInstances
	widgetVO := &WidgetVO{
		Instances: tooManyInstances,
	}
	expectedError := errors.New("length of instances is invalid")
	err := widgetVO.ValidateInstancesParam(1, maxInstanceCount)
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidateInstancesParam to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidateInstancesParam to return error %v, but got %v", expectedError, err)
	}
}

func TestWidgetVOValidateInstancesParamEmptyInstanceString(t *testing.T) {
	instancesWithEmpty := []string{"instance1", ""}
	widgetVO := &WidgetVO{
		Instances: instancesWithEmpty,
	}
	expectedError := errors.New("instance is empty")
	err := widgetVO.ValidateInstancesParam(1, maxInstanceCount)
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidateInstancesParam to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidateInstancesParam to return error %v, but got %v", expectedError, err)
	}
}

// GetGraphByWidgetType test cases
func TestGetGraphByWidgetTypeValidTypes(t *testing.T) {
	validWidgetTypes := validWidgetTypesExceptDashboard
	defaultBoardMin := defaultBoardMin
	defaultBoardMax := defaultBoardMax
	for _, widgetType := range validWidgetTypes {
		widgetVO := &WidgetVO{
			WidgetType: widgetType,
			BoardMin:   defaultBoardMin,
			BoardMax:   defaultBoardMax,
		}
		graph, err := GetGraphByWidgetType(widgetVO)
		if !errors.Is(err, nil) {
			t.Errorf("Expected GetGraphByWidgetType to return nil for type %s, but got %v", widgetType, err)
		}
		if graph == nil {
			t.Errorf("Expected GetGraphByWidgetType to return non-nil graph for type %s", widgetType)
		}
	}
}

func TestGetGraphByWidgetTypeDashboardType(t *testing.T) {
	dashboardWidgetType := dashboardGraphType
	defaultBoardMin := "0"
	validBoardMax := testBoardMax
	widgetVO := &WidgetVO{
		WidgetType: dashboardWidgetType,
		BoardMin:   defaultBoardMin,
		BoardMax:   validBoardMax,
	}
	graph, err := GetGraphByWidgetType(widgetVO)
	if !errors.Is(err, nil) {
		t.Errorf("Expected GetGraphByWidgetType to return nil for dashboard type, but got %v", err)
	}
	if graph == nil {
		t.Errorf("Expected GetGraphByWidgetType to return non-nil graph for dashboard type")
	}
}

func TestGetGraphByWidgetTypeInvalidType(t *testing.T) {
	invalidWidgetType := "invalid"
	defaultBoardMin := defaultBoardMin
	defaultBoardMax := defaultBoardMax
	widgetVO := &WidgetVO{
		WidgetType: invalidWidgetType,
		BoardMin:   defaultBoardMin,
		BoardMax:   defaultBoardMax,
	}
	expectedError := errors.New("widgetType is invalid")
	_, err := GetGraphByWidgetType(widgetVO)
	if errors.Is(err, nil) {
		t.Errorf("Expected GetGraphByWidgetType to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected GetGraphByWidgetType to return error %v, but got %v", expectedError, err)
	}
}

// getWidgetCRTypeMeta test cases
func TestGetWidgetCRTypeMeta(t *testing.T) {
	typeMeta := getWidgetCRTypeMeta()
	expectedKind := global.WidgetCRDRepoKind
	expectedAPIVersion := global.CRDRepoGroup + "/" + global.CRDRepoVersion

	if typeMeta.Kind != expectedKind {
		t.Errorf("Expected Kind to be %s, but got %s", expectedKind, typeMeta.Kind)
	}
	if typeMeta.APIVersion != expectedAPIVersion {
		t.Errorf("Expected APIVersion to be %s, but got %s", expectedAPIVersion, typeMeta.APIVersion)
	}
}

// getWidgetCRObjectMeta test cases
func TestGetWidgetCRObjectMeta(t *testing.T) {
	testRepoName := "test-widget"
	testDashboardID := "test-dashboard"
	objectMeta := getWidgetCRObjectMeta(testRepoName, testDashboardID)

	if objectMeta.Name != testRepoName {
		t.Errorf("Expected Name to be %s, but got %s", testRepoName, objectMeta.Name)
	}
	if objectMeta.Namespace != global.MonitorServiceDefaultNamespace {
		t.Errorf("Expected Namespace to be %s, but got %s",
			global.MonitorServiceDefaultNamespace, objectMeta.Namespace)
	}
	if objectMeta.Labels["dashboardID"] != testDashboardID {
		t.Errorf("Expected dashboardID label to be %s, but got %s",
			testDashboardID, objectMeta.Labels["dashboardID"])
	}
	if objectMeta.Annotations == nil {
		t.Errorf("Expected Annotations to be non-nil")
	}
}

// NewWidgetCRDByCreateVO test cases
func TestNewWidgetCRDByCreateVOValidInput(t *testing.T) {
	validDashboardID := "2024-01-01-12-00-00-" + strings.Repeat("a", widgetIDRepeatTimes)
	validDisplayName := "Test Widget"
	validDataSource := widgetDatasourceURL
	validWidgetType := lineGraphType
	validInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		DashboardID: validDashboardID,
		DisplayName: validDisplayName,
		DataSource:  validDataSource,
		WidgetType:  validWidgetType,
		Instances:   validInstances,
	}
	widgetCRD, err := NewWidgetCRDByCreateVO(widgetVO)
	if !errors.Is(err, nil) {
		t.Errorf("Expected NewWidgetCRDByCreateVO to return nil, but got %v", err)
	}
	if widgetCRD == nil {
		t.Errorf("Expected NewWidgetCRDByCreateVO to return non-nil WidgetCRD")
	}
	if widgetCRD.Spec.DashboardID != widgetVO.DashboardID {
		t.Errorf("Expected DashboardID to be %s, but got %s",
			widgetVO.DashboardID, widgetCRD.Spec.DashboardID)
	}
	if widgetCRD.Spec.DisplayName != widgetVO.DisplayName {
		t.Errorf("Expected DisplayName to be %s, but got %s",
			widgetVO.DisplayName, widgetCRD.Spec.DisplayName)
	}
}

func TestNewWidgetCRDByCreateVOInvalidWidgetType(t *testing.T) {
	validDashboardID := "2024-01-01-12-00-00-" + strings.Repeat("a", widgetIDRepeatTimes)
	validDisplayName := "Test Widget"
	validDataSource := widgetDatasourceURL
	invalidWidgetType := "invalid"
	validInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		DashboardID: validDashboardID,
		DisplayName: validDisplayName,
		DataSource:  validDataSource,
		WidgetType:  invalidWidgetType,
		Instances:   validInstances,
	}
	_, err := NewWidgetCRDByCreateVO(widgetVO)
	if errors.Is(err, nil) {
		t.Errorf("Expected NewWidgetCRDByCreateVO to return error for invalid widget type, but got nil")
	}
}

func TestNewWidgetCRDByCreateVOInvalidInstances(t *testing.T) {
	validDashboardID := "2024-01-01-12-00-00-" + strings.Repeat("a", widgetIDRepeatTimes)
	validDisplayName := "Test Widget"
	validDataSource := widgetDatasourceURL
	validWidgetType := lineGraphType
	var emptyInstances []string
	widgetVO := &WidgetVO{
		DashboardID: validDashboardID,
		DisplayName: validDisplayName,
		DataSource:  validDataSource,
		WidgetType:  validWidgetType,
		Instances:   emptyInstances,
	}
	_, err := NewWidgetCRDByCreateVO(widgetVO)
	if errors.Is(err, nil) {
		t.Errorf("Expected NewWidgetCRDByCreateVO to return error for empty instances, but got nil")
	}
}

// NewPatchUpdateVO test cases
func TestNewPatchUpdateVOValidInput(t *testing.T) {
	validDisplayName := "Updated Widget"
	validDataSource := widgetDatasourceURL
	validWidgetType := lineGraphType
	validInstances := []string{"instance1"}
	defaultBoardMin := defaultBoardMin
	defaultBoardMax := defaultBoardMax
	widgetVO := &WidgetVO{
		DisplayName: validDisplayName,
		DataSource:  validDataSource,
		WidgetType:  validWidgetType,
		Instances:   validInstances,
		BoardMin:    defaultBoardMin,
		BoardMax:    defaultBoardMax,
	}
	patchMap, err := NewPatchUpdateVO(widgetVO)
	if !errors.Is(err, nil) {
		t.Errorf("Expected NewPatchUpdateVO to return nil, but got %v", err)
	}
	if patchMap == nil {
		t.Errorf("Expected NewPatchUpdateVO to return non-nil patch map")
	}
	spec, ok := patchMap["spec"].(map[string]interface{})
	if !ok {
		t.Errorf("Expected patch map to contain 'spec' key")
	}
	if spec["displayName"] != widgetVO.DisplayName {
		t.Errorf("Expected displayName to be %s, but got %v",
			widgetVO.DisplayName, spec["displayName"])
	}
}

func TestNewPatchUpdateVOInvalidWidgetType(t *testing.T) {
	validDisplayName := "Updated Widget"
	validDataSource := widgetDatasourceURL
	invalidWidgetType := "invalid"
	validInstances := []string{"instance1"}
	defaultBoardMin := defaultBoardMin
	defaultBoardMax := defaultBoardMax
	widgetVO := &WidgetVO{
		DisplayName: validDisplayName,
		DataSource:  validDataSource,
		WidgetType:  invalidWidgetType,
		Instances:   validInstances,
		BoardMin:    defaultBoardMin,
		BoardMax:    defaultBoardMax,
	}
	_, err := NewPatchUpdateVO(widgetVO)
	if errors.Is(err, nil) {
		t.Errorf("Expected NewPatchUpdateVO to return error for invalid widget type, but got nil")
	}
}

// WidgetGraphType constants test cases
func TestWidgetGraphTypeConstants(t *testing.T) {
	expectedValues := map[WidgetGraphType]int{
		LineGraphType:        0,
		StackedAreaGraphType: 1,
		NumbersGraphType:     2,
		DashboardGraphType:   3,
		RingGraphType:        4,
		BarGraphType:         5,
	}

	for graphType, expectedValue := range expectedValues {
		if int(graphType) != expectedValue {
			t.Errorf("Expected %v to have value %d, but got %d",
				graphType, expectedValue, int(graphType))
		}
	}
}

// Edge cases and boundary tests
func TestWidgetVOValidatePatchUpdateDisplayNameTooLong(t *testing.T) {
	tooLongDisplayName := strings.Repeat("a", global.MaxNameLength+1)
	validDataSource := widgetDatasourceURL
	validWidgetType := lineGraphType
	validInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		DisplayName: tooLongDisplayName,
		DataSource:  validDataSource,
		WidgetType:  validWidgetType,
		Instances:   validInstances,
	}
	expectedError := errors.New("displayName is too long")
	err := widgetVO.ValidatePatchUpdate()
	if errors.Is(err, nil) {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got nil", expectedError)
	}
	if err.Error() != expectedError.Error() {
		t.Errorf("Expected ValidatePatchUpdate to return error %v, but got %v", expectedError, err)
	}
}

func TestWidgetVOValidateInstancesParamBoundaryValues(t *testing.T) {
	// Test minimum valid instances
	minValidInstances := []string{"instance1"}
	widgetVO := &WidgetVO{
		Instances: minValidInstances,
	}
	err := widgetVO.ValidateInstancesParam(1, 1)
	if !errors.Is(err, nil) {
		t.Errorf("Expected ValidateInstancesParam to return nil for minimum valid instances, but got %v", err)
	}

	// Test maximum valid instances
	maxValidInstances := maxValidInstances
	widgetVO = &WidgetVO{
		Instances: maxValidInstances,
	}
	err = widgetVO.ValidateInstancesParam(1, maxInstanceCount)
	if !errors.Is(err, nil) {
		t.Errorf("Expected ValidateInstancesParam to return nil for maximum valid instances, but got %v", err)
	}
}

func TestGetWidgetGraphTypeFromIntBoundaryValues(t *testing.T) {
	// Test boundary values
	actual := GetWidgetGraphTypeFromInt(0)
	if actual != LineGraphType {
		t.Errorf("Expected GetWidgetGraphTypeFromInt(0) to return LineGraphType, but got %v", actual)
	}

	actual = GetWidgetGraphTypeFromInt(fiveInt)
	if actual != BarGraphType {
		t.Errorf("Expected GetWidgetGraphTypeFromInt(5) to return BarGraphType, but got %v", actual)
	}

	// Test just outside boundary
	actual = GetWidgetGraphTypeFromInt(-1)
	if actual != WidgetGraphType(-1) {
		t.Errorf("Expected GetWidgetGraphTypeFromInt(-1) to return WidgetGraphType(-1), but got %v", actual)
	}

	actual = GetWidgetGraphTypeFromInt(sixInt)
	if actual != WidgetGraphType(-1) {
		t.Errorf("Expected GetWidgetGraphTypeFromInt(6) to return WidgetGraphType(-1), but got %v", actual)
	}
}
