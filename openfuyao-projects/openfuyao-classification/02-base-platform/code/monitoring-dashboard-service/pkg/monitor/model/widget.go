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

// Package model 定义了monitor模块涉及的的 DO VO BO
package model

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/utils/uutil"
)

// WidgetCRD 用于创建/修改 CR 的 DO
type WidgetCRD struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty"`

	Spec WidgetSpec `json:"spec"`
}

// WidgetSpec WidgetCRD 的 spec 自定义部分
type WidgetSpec struct {
	WidgetID    string          `json:"widgetID"`
	DashboardID string          `json:"dashboardID"`
	DisplayName string          `json:"displayName"`
	DataSource  string          `json:"dataSource"`
	WidgetType  WidgetGraphType `json:"widgetType"`
	Instances   []string        `json:"instances"` // JSON
	BoardMin    int             `json:"boardMin"`
	BoardMax    int             `json:"boardMax"`
}

// WidgetVO 用于前端 POST Create 请求的 VO
type WidgetVO struct {
	WidgetID    string   `json:"widgetID"` // 创建VO时可以为空
	DashboardID string   `json:"dashboardID"`
	DisplayName string   `json:"displayName"`
	DataSource  string   `json:"dataSource"`
	WidgetType  string   `json:"widgetType"`
	Instances   []string `json:"instances"` // 前端直接传入JSON
	BoardMin    string   `json:"boardMin"`  // 可以为空，当Type为dashboard时非空
	BoardMax    string   `json:"boardMax"`  // 同上
}

// Widget 用于执行业务的完整 BO
type Widget struct {
	WidgetID    string          `json:"widgetID"`
	DashboardID string          `json:"dashboardID"`
	DisplayName string          `json:"displayName"`
	DataSource  string          `json:"dataSource"`
	WidgetType  WidgetGraphType `json:"widgetType"`
	Instances   []Instance      `json:"instances"` // 需要装配
	WidgetGraph IGraph          `json:"graphType"`
}

// WidgetGraphType 定义枚举类型
type WidgetGraphType int

// WidgetGraphType 枚举值
const (
	LineGraphType        WidgetGraphType = 0
	StackedAreaGraphType WidgetGraphType = 1
	NumbersGraphType     WidgetGraphType = 2
	DashboardGraphType   WidgetGraphType = 3
	RingGraphType        WidgetGraphType = 4
	BarGraphType         WidgetGraphType = 5
)

// graphTypeMap 用于枚举类型 value-key 映射
var graphTypeMap = map[int]WidgetGraphType{
	0: LineGraphType,
	1: StackedAreaGraphType,
	2: NumbersGraphType,
	3: DashboardGraphType,
	4: RingGraphType,
	5: BarGraphType,
}

// GetWidgetGraphTypeFromInt 用于从 int 获取枚举类型
func GetWidgetGraphTypeFromInt(value int) WidgetGraphType {
	if widgetGraphType, found := graphTypeMap[value]; found {
		return widgetGraphType
	}
	return -1 // 或者返回一个错误值，例如 WidgetGraphType(-1)
}

// ValidateCreate 对Create请求的WidgetVO参数做合法性校验，图合法性校验放在构造Widget中
// dashboardID、displayName、widgetType、dataSource、instances、boardMin、boardMax
func (v *WidgetVO) ValidateCreate() error {
	errID := uutil.ValidateID(v.DashboardID)
	if errID != nil {
		return errID
	}
	errUpdate := v.ValidatePatchUpdate() // update中会被校验的VO参数是create的子集
	if errUpdate != nil {
		return errUpdate
	}
	return nil
}

// ValidatePatchUpdate 对Patch请求的WidgetVO参数做合法性校验，图合法性校验放在构造Widget中
// displayName、widgetType、dataSource、instances、boardMin、boardMax
func (v *WidgetVO) ValidatePatchUpdate() error {
	errName := uutil.ValidateDisplayName(v.DisplayName, global.MaxNameLength)
	if errName != nil {
		return errName
	}
	errHost := uutil.ValidateHost(v.DataSource)
	if errHost != nil {
		return errHost
	}
	errType := v.ValidateWidgetTypeParam()
	if errType != nil {
		return errType
	}
	v.InitialGraphParams()
	return nil
}

// InitialGraphParams 对图表参数为空字符串的情况进行初始化，初始化为0
// 目的是在业务开始之前将字符串类型转为int类型
func (v *WidgetVO) InitialGraphParams() {
	if v.BoardMin == "" {
		v.BoardMin = "0"
	}
	if v.BoardMax == "" {
		v.BoardMax = "0"
	}
}

// ValidateWidgetTypeParam 校验widgetType参数合法：1.widgetType不能为空；2.widgetType必须是0-5的整数
func (v *WidgetVO) ValidateWidgetTypeParam() error {
	if v.WidgetType == "" {
		return errors.New("widgetType is empty")
	}
	widgetTypeInt, errType := strconv.Atoi(v.WidgetType)
	if errType != nil {
		return errors.New("widgetType is invalid")
	}
	if widgetTypeInt < 0 || widgetTypeInt > len(graphTypeMap) {
		return errors.New("widgetType is invalid")
	}
	return nil
}

// ValidateInstancesParam 校验instances参数合法性
func (v *WidgetVO) ValidateInstancesParam(InstancesMin int, InstancesMax int) error {
	if v.Instances == nil {
		return errors.New("instances is empty")
	}
	length := len(v.Instances)
	if length < InstancesMin || length > InstancesMax {
		return errors.New("length of instances is invalid")
	}
	for _, instance := range v.Instances {
		if instance == "" {
			return errors.New("instance is empty")
		}
	}
	return nil
}

// NewWidgetCRDByCreateVO 用于Create操作，直接根据VO获得CRD，绕过BO
func NewWidgetCRDByCreateVO(vo *WidgetVO) (*WidgetCRD, error) {
	wID := uutil.GenerateID()
	graph, err := GetGraphByWidgetType(vo)
	if err != nil {
		return nil, err
	}
	err = vo.ValidateInstancesParam(global.LeastInstanceOfWidget, graph.GetInstancesMax())
	if err != nil {
		return nil, err
	}

	return &WidgetCRD{
		TypeMeta:   getWidgetCRTypeMeta(),
		ObjectMeta: getWidgetCRObjectMeta(strings.ToLower(wID), strings.ToLower(vo.DashboardID)),
		Spec: WidgetSpec{
			WidgetID:    wID,
			DashboardID: vo.DashboardID,
			DisplayName: vo.DisplayName,
			DataSource:  vo.DataSource,
			WidgetType:  graph.GetGraphType(),
			Instances:   vo.Instances,
			BoardMin:    graph.GetBoardMin(),
			BoardMax:    graph.GetBoardMax(),
		},
	}, nil
}

// NewPatchUpdateVO 用于Update操作，直接根据VO获得patch键值对
func NewPatchUpdateVO(vo *WidgetVO) (map[string]interface{}, error) {
	wIGraph, err := GetGraphByWidgetType(vo)
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"spec": map[string]interface{}{
			"displayName": vo.DisplayName,
			"dataSource":  vo.DataSource,
			"widgetType":  wIGraph.GetGraphType(),
			"instances":   vo.Instances,
			"boardMin":    wIGraph.GetBoardMin(),
			"boardMax":    wIGraph.GetBoardMax(),
		},
	}, nil
}

// GetGraphByWidgetType 根据widgetType获得对应的IGraph接口
func GetGraphByWidgetType(vo *WidgetVO) (IGraph, error) {
	widgetTypeInt, err := strconv.Atoi(vo.WidgetType)
	if err != nil {
		return nil, errors.New("widgetType is invalid")
	}
	wGraphType := GetWidgetGraphTypeFromInt(widgetTypeInt)
	graph, err := GraphFactoryByWidgetVO(wGraphType, vo)
	if err != nil {
		return nil, err
	}
	err = graph.Validate()
	if err != nil {
		return nil, err
	}
	return graph, nil
}

// getWidgetCRObjectMeta 获取CRD中的ObjectMeta
func getWidgetCRObjectMeta(repoName string, dashboardId string) v1.ObjectMeta {
	objectMeta := v1.ObjectMeta{
		Name:      repoName,
		Namespace: global.MonitorServiceDefaultNamespace,
		Annotations: map[string]string{
			fmt.Sprintf("%s/%s", global.MonitorServiceDefaultOrgName,
				global.ModificationTimestamp): time.Now().UTC().Format(time.RFC3339),
		},
		Labels: map[string]string{
			"dashboardID": dashboardId,
		},
	}
	return objectMeta
}

// getWidgetCRTypeMeta 获取CRD中的TypeMeta
func getWidgetCRTypeMeta() v1.TypeMeta {
	typeMeta := v1.TypeMeta{
		Kind:       global.WidgetCRDRepoKind,
		APIVersion: fmt.Sprintf("%s/%s", global.CRDRepoGroup, global.CRDRepoVersion),
	}
	return typeMeta
}
