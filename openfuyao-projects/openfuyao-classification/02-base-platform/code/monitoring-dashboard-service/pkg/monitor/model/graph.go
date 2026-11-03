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
	"strconv"

	"monitoring-dashboard-service/pkg/config/global"
)

// IGraph 所有Graph的子类必须实现的接口，widget组成部分
type IGraph interface {
	Validate() error
	GetBoardMin() int
	GetBoardMax() int
	GetInstancesMax() int
	GetGraphType() WidgetGraphType
}

// Graph 所有Graph的基类，以组合的形式继承，存放公共属性
type Graph struct {
	GraphType    WidgetGraphType `json:"graphType"`
	InstancesMax int             `json:"instancesMax"`
}

// NewGraph 创建Graph对象
func NewGraph(graphType WidgetGraphType, instancesMax int) *Graph {
	return &Graph{
		GraphType:    graphType,
		InstancesMax: instancesMax,
	}
}

// LineGraph 线形图类
type LineGraph struct {
	*Graph
}

// NewLineGraph 创建LineGraph对象
func NewLineGraph(graphType WidgetGraphType) *LineGraph {
	return &LineGraph{
		Graph: &Graph{
			GraphType:    graphType,
			InstancesMax: global.MaxInstanceOfWidget,
		},
	}
}

// GetBoardMin 获取仪表盘最小值
func (g *LineGraph) GetBoardMin() int {
	return 0
}

// GetBoardMax 获取仪表盘最大值
func (g *LineGraph) GetBoardMax() int {
	return 0
}

// GetInstancesMax 获取图表可包含的实例instance最大值
func (g *LineGraph) GetInstancesMax() int {
	return g.Graph.InstancesMax
}

// Validate 校验参数合法性
func (g *LineGraph) Validate() error {
	if g.Graph.GraphType != LineGraphType {
		return errors.New("lineGraph type error")
	}
	return nil
}

// GetGraphType 获取图表类型
func (g *LineGraph) GetGraphType() WidgetGraphType {
	return g.GraphType
}

// StackedAreaGraph 堆叠面积图类
type StackedAreaGraph struct {
	*Graph
}

// NewStackedArea 创建StackedAreaGraph对象
func NewStackedArea(graphType WidgetGraphType) *StackedAreaGraph {
	return &StackedAreaGraph{
		Graph: &Graph{
			GraphType:    graphType,
			InstancesMax: global.MaxInstanceOfStackedArea,
		},
	}
}

// Validate 校验参数合法性
func (g *StackedAreaGraph) Validate() error {
	if g.Graph.GraphType != StackedAreaGraphType {
		return errors.New("stackedAreaGraph type error")
	}
	return nil
}

// GetBoardMin 获取仪表盘最小值
func (g *StackedAreaGraph) GetBoardMin() int {
	return 0
}

// GetBoardMax 获取仪表盘最大值
func (g *StackedAreaGraph) GetBoardMax() int {
	return 0
}

// GetInstancesMax 获取图表可包含的实例instance最大值
func (g *StackedAreaGraph) GetInstancesMax() int {
	return g.Graph.InstancesMax
}

// GetGraphType 获取图表类型
func (g *StackedAreaGraph) GetGraphType() WidgetGraphType {
	return g.GraphType
}

// NumbersGraph 数字图类
type NumbersGraph struct {
	*Graph
}

// NewNumbers 创建NumbersGraph对象
func NewNumbers(graphType WidgetGraphType) *NumbersGraph {
	return &NumbersGraph{
		Graph: &Graph{
			GraphType:    graphType,
			InstancesMax: global.MaxInstanceOfWidget,
		},
	}
}

// Validate 校验参数合法性
func (g *NumbersGraph) Validate() error {
	if g.Graph.GraphType != NumbersGraphType {
		return errors.New("numbersGraph type error")
	}
	return nil
}

// GetBoardMin 获取仪表盘最小值
func (g *NumbersGraph) GetBoardMin() int {
	return 0
}

// GetBoardMax 获取仪表盘最大值
func (g *NumbersGraph) GetBoardMax() int {
	return 0
}

// GetInstancesMax 获取图表可包含的实例instance最大值
func (g *NumbersGraph) GetInstancesMax() int {
	return g.Graph.InstancesMax
}

// GetGraphType 获取图表类型
func (g *NumbersGraph) GetGraphType() WidgetGraphType {
	return g.GraphType
}

// BoardGraph 仪表盘图类
type BoardGraph struct {
	*Graph
	BoardMin int `json:"boardMin"`
	BoardMax int `json:"boardMax"`
}

// NewBoardGraph 创建BoardGraph对象
func NewBoardGraph(graphType WidgetGraphType, boardMin int, boardMax int) *BoardGraph {
	return &BoardGraph{
		Graph: &Graph{
			GraphType:    graphType,
			InstancesMax: global.MaxInstanceOfWidget,
		},
		BoardMin: boardMin,
		BoardMax: boardMax,
	}
}

// Validate 校验参数合法性
func (g *BoardGraph) Validate() error {
	if g.Graph.GraphType != DashboardGraphType {
		return errors.New("boardGraph type error")
	}
	if g.BoardMin >= g.BoardMax {
		return errors.New("board min must be less than board max")
	}
	return nil
}

// GetBoardMin 获取仪表盘最小值
func (g *BoardGraph) GetBoardMin() int {
	return g.BoardMin
}

// GetBoardMax 获取仪表盘最大值
func (g *BoardGraph) GetBoardMax() int {
	return g.BoardMax
}

// GetInstancesMax 获取图表可包含的实例instance最大值
func (g *BoardGraph) GetInstancesMax() int {
	return g.Graph.InstancesMax
}

// GetGraphType 获取图表类型
func (g *BoardGraph) GetGraphType() WidgetGraphType {
	return g.GraphType
}

// RingGraph 环图类
type RingGraph struct {
	*Graph
}

// NewRingGraph 创建RingGraph对象
func NewRingGraph(graphType WidgetGraphType) *RingGraph {
	return &RingGraph{
		Graph: &Graph{
			GraphType:    graphType,
			InstancesMax: global.MaxInstanceOfWidget,
		},
	}
}

// Validate 校验参数合法性
func (g *RingGraph) Validate() error {
	if g.Graph.GraphType != RingGraphType {
		return errors.New("ringGraph type error")
	}
	return nil
}

// GetBoardMin 获取仪表盘最小值
func (g *RingGraph) GetBoardMin() int {
	return 0
}

// GetBoardMax 获取仪表盘最大值
func (g *RingGraph) GetBoardMax() int {
	return 0
}

// GetInstancesMax 获取图表可包含的实例instance最大值
func (g *RingGraph) GetInstancesMax() int {
	return g.Graph.InstancesMax
}

// GetGraphType 获取图表类型
func (g *RingGraph) GetGraphType() WidgetGraphType {
	return g.GraphType
}

// BarGraph 柱状图类
type BarGraph struct {
	*Graph
}

// NewBarGraph 创建BarGraph对象
func NewBarGraph(graphType WidgetGraphType) *BarGraph {
	return &BarGraph{
		Graph: &Graph{
			GraphType:    graphType,
			InstancesMax: global.MaxInstanceOfWidget,
		},
	}
}

// Validate 校验参数合法性
func (g *BarGraph) Validate() error {
	if g.Graph.GraphType != BarGraphType {
		return errors.New("barGraph type error")
	}

	return nil
}

// GetBoardMin 获取仪表盘最小值
func (g *BarGraph) GetBoardMin() int {
	return 0
}

// GetBoardMax 获取仪表盘最大值
func (g *BarGraph) GetBoardMax() int {
	return 0
}

// GetInstancesMax 获取图表可包含的实例instance最大值
func (g *BarGraph) GetInstancesMax() int {
	return g.Graph.InstancesMax
}

// GetGraphType 获取图表类型
func (g *BarGraph) GetGraphType() WidgetGraphType {
	return g.GraphType
}

// GraphFactoryByWidgetVO 简单工厂函数，通过widgetVO创建Graph对象
func GraphFactoryByWidgetVO(graphType WidgetGraphType, vo *WidgetVO) (IGraph, error) {
	switch graphType {
	case LineGraphType:
		return NewLineGraph(graphType), nil
	case StackedAreaGraphType:
		return NewStackedArea(graphType), nil
	case NumbersGraphType:
		return NewNumbers(graphType), nil
	case DashboardGraphType:
		boardMin, err := strconv.Atoi(vo.BoardMin)
		boardMax, err2 := strconv.Atoi(vo.BoardMax)
		if err != nil || err2 != nil {
			return nil, errors.New("widgetType is invalid")
		}
		return NewBoardGraph(graphType, boardMin, boardMax), nil
	case RingGraphType:
		return NewRingGraph(graphType), nil
	case BarGraphType:
		return NewBarGraph(graphType), nil
	default:
		return nil, errors.New("unknown graph type")
	}
}

// GraphFactoryByWidgetCRD 简单工厂函数，通过widgetCRD创建Graph对象
func GraphFactoryByWidgetCRD(crd *WidgetCRD) (IGraph, error) {
	graphType := crd.Spec.WidgetType
	switch graphType {
	case LineGraphType:
		return NewLineGraph(graphType), nil
	case StackedAreaGraphType:
		return NewStackedArea(graphType), nil
	case NumbersGraphType:
		return NewNumbers(graphType), nil
	case DashboardGraphType:
		return NewBoardGraph(graphType, crd.Spec.BoardMin, crd.Spec.BoardMax), nil
	case RingGraphType:
		return NewRingGraph(graphType), nil
	case BarGraphType:
		return NewBarGraph(graphType), nil
	default:
		return nil, errors.New("unknown graph type")
	}
}
