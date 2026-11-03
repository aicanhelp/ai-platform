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
	"testing"

	"monitoring-dashboard-service/pkg/config/global"
)

// TestNewGraph 校验Graph基类构造
func TestNewGraph(t *testing.T) {
	graph := NewGraph(LineGraphType, global.MaxInstanceOfWidget)
	if graph.GraphType != LineGraphType {
		t.Errorf("unexpected graph type: %v", graph.GraphType)
	}
	if graph.InstancesMax != global.MaxInstanceOfWidget {
		t.Errorf("unexpected instances max: %d", graph.InstancesMax)
	}
}

// TestGraphFactoryByWidgetVOUnknownType 未知图表类型报错
func TestGraphFactoryByWidgetVOUnknownType(t *testing.T) {
	if _, err := GraphFactoryByWidgetVO(WidgetGraphType(99), &WidgetVO{}); err == nil {
		t.Error("expected error for unknown graph type, got nil")
	}
}

// TestGraphFactoryByWidgetVOBadBoardParam 仪表盘图表min/max非法时报错
func TestGraphFactoryByWidgetVOBadBoardParam(t *testing.T) {
	vo := &WidgetVO{BoardMin: "abc", BoardMax: "100"}
	if _, err := GraphFactoryByWidgetVO(DashboardGraphType, vo); err == nil {
		t.Error("expected error for non-numeric board params, got nil")
	}
}

// TestGraphFactoryByWidgetCRDUnknownType CRD未知图表类型报错
func TestGraphFactoryByWidgetCRDUnknownType(t *testing.T) {
	crd := &WidgetCRD{Spec: WidgetSpec{WidgetType: WidgetGraphType(99)}}
	if _, err := GraphFactoryByWidgetCRD(crd); err == nil {
		t.Error("expected error for unknown graph type, got nil")
	}
}

// TestWidgetVOValidateCreateBadDashboardID 非法dashboardID报错
func TestWidgetVOValidateCreateBadDashboardID(t *testing.T) {
	vo := &WidgetVO{
		DashboardID: "short-id",
		DisplayName: "组件",
		DataSource:  "http://prometheus-k8s.monitoring.svc.cluster.local:9090",
		WidgetType:  "0",
	}
	if err := vo.ValidateCreate(); err == nil {
		t.Error("expected error for invalid dashboard ID, got nil")
	}
}

// TestGetGraphByWidgetTypeOutOfRange 超出范围的widgetType报错
func TestGetGraphByWidgetTypeOutOfRange(t *testing.T) {
	vo := &WidgetVO{
		WidgetType: "6",
		BoardMin:   "0",
		BoardMax:   "100",
	}
	if _, err := GetGraphByWidgetType(vo); err == nil {
		t.Error("expected error for out-of-range widget type, got nil")
	}
}
