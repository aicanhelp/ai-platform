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

package service

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"

	colocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
	"openfuyao.com/colocation-service/pkg/utils"
)

func (server *Server) initColocationNodeRouteNew() {
	ws := server.ws
	// 提供混部Node的管理接口, 通过真实节点标签来返回当前节点是否开启混部
	ws.Route(ws.GET("/colocation-configs/").To(server.getAllColocationNodesNew).
		Doc("get all colocation nodes"))

	// 注册 PATCH 路由，修改节点混部状态
	ws.Route(
		ws.PATCH("/colocation-nodes/{name}").
			To(server.toggleColocationNodeByNameWithRetry).
			Doc("toggle single node colocation status").
			Param(ws.QueryParameter("colocation-status", "true / false").DataType("boolean").Required(true)),
	)

	// 新增 POST 路由，接收配置数据
	ws.Route(
		ws.POST("/colocation-configs/").
			To(server.handleColocationConfigWithRetry).
			Doc("receive post colocation configuration").
			Reads(ColocationConfigRequest{}), // 这里需要定义 ColocationConfigRequest 结构体
	)
}

const (
	// node-role.kubernetes.io/master
	NodeMasterLabelKey string = "node-role.kubernetes.io/master"
	// node-role.kubernetes.io/control-plane
	NodeControlPlaneLabelKey string = "node-role.kubernetes.io/control-plane"
)

// NodeAttr 表示节点信息
type NodeAttr struct {
	Uid            string                  `json:"uid"`
	Name           string                  `json:"name"`
	Addresses      []string                `json:"addresses"`
	OSImage        string                  `json:"osImage"`
	Status         string                  `json:"status"`
	IsColocated    bool                    `json:"isColocated"`
	FeatureSupport *NodeFeatureSupportInfo `json:"featureSupport,omitempty"`
}

type NodeFeatureSupportInfo struct {
	CPUPreemption    string `json:"cpuPreemption"`
	MemoryPreemption string `json:"memoryPreemption"`
	QuotaTurbo       string `json:"quotaTurbo"`
	DynMemory        string `json:"dynMemory"`
	DynCache         string `json:"dynCache"`
	PSIDetection     string `json:"psiDetection"`
}

// RubikConfig Rubik配置信息
type RubikConfig struct {
	Eviction   RubikEviction     `json:"eviction"`
	QuotaTurbo *QuotaTurboConfig `json:"quotaTurbo,omitempty"`
	DynMemory  *DynMemoryConfig  `json:"dynMemory,omitempty"`
	DynCache   *DynCacheConfig   `json:"dynCache,omitempty"`
	PSI        *PSIConfig        `json:"psi,omitempty"`
}

// RubikEviction Rubik水位线驱逐配置信息
type RubikEviction struct {
	Enabled     bool              `json:"enable"`
	CPUEvict    CPUEvictConfig    `json:"cpuevict"`
	MemoryEvict MemoryEvictConfig `json:"memoryevict"`
}

// QuotaTurboConfig 配额CPU弹性限流配置
type QuotaTurboConfig struct {
	Enable         bool `json:"enable"`
	HighWaterMark  int  `json:"highWaterMark"`
	AlarmWaterMark int  `json:"alarmWaterMark"`
}

// DynMemoryConfig 动态内存配置
type DynMemoryConfig struct {
	Enable bool `json:"enable"`
}

// DynCacheConfig 动态缓存配置
type DynCacheConfig struct {
	Enable         bool                `json:"enable"`
	L3Percent      *CachePercentConfig `json:"l3Percent,omitempty"`
	MemBandPercent *CachePercentConfig `json:"memBandPercent,omitempty"`
}

// CachePercentConfig 缓存百分比配置
type CachePercentConfig struct {
	Low  int `json:"low"`
	Mid  int `json:"mid"`
	High int `json:"high"`
}

// PSIConfig 表示节点的PSI配置
type PSIConfig struct {
	Enable         bool     `json:"enable"`
	Resource       []string `json:"resource"`
	Avg10Threshold float64  `json:"avg10Threshold"`
}

// UsageThresholdConfig volcano调度器阈值信息
type UsageThresholdConfig struct {
	Enabled         bool    `json:"enable"`
	CPUThreshold    float64 `json:"cpu,omitempty"`
	MemoryThreshold float64 `json:"memory,omitempty"`
}

// UsagePluginConfig volcano调度器插件信息
type UsagePluginConfig struct {
	Enabled        bool                 `json:"enable"`
	UsageThreshold UsageThresholdConfig `json:"usageThreshold,omitempty"`
}

// VolcanoSchedulerConfig volcano调度器配置信息
type VolcanoSchedulerConfig struct {
	UsagePlugin UsagePluginConfig `json:"usagePlugin"`
}

// NodeInfoWrapper 包装节点信息
type NodeInfoWrapper struct {
	NodeInfo               NodeAttr               `json:"nodeInfo"`
	RubikConfig            RubikConfig            `json:"rubikConfig"`
	VolcanoSchedulerConfig VolcanoSchedulerConfig `json:"volcanoSchedulerConfig"`
}

// ColocationConfigRequest get colocation时完整响应结构体
type ColocationConfigRequest struct {
	NodeInfo               []NodeAttr             `json:"nodeInfo"`
	RubikConfig            RubikConfig            `json:"rubikConfig"`
	VolcanoSchedulerConfig VolcanoSchedulerConfig `json:"volcanoSchedulerConfig"`
}

func (server *Server) buildNodeAttr(node *corev1.Node) NodeAttr {
	addr := getNodeAddresses(*node)
	var addresses []string
	if addr != "" {
		addresses = append(addresses, addr)
	}

	isColocated := isColocationNode(node)
	status := getNodeStatus(*node)
	featureSupport := server.getNodeFeatureSupport(node)

	return NodeAttr{
		Uid:            string(node.ObjectMeta.UID),
		Name:           node.ObjectMeta.Name,
		Addresses:      addresses,
		OSImage:        node.Status.NodeInfo.OSImage,
		Status:         status,
		IsColocated:    isColocated,
		FeatureSupport: featureSupport,
	}
}

func (server *Server) getAllColocationNodesNew(request *restful.Request, response *restful.Response) {
	klog.Info("getAllColocationNodes")

	// 获取所有节点
	nodes, err := server.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.Error(err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError,
			utils.NewResponseResultWithError(AppCode, FeatureCode, err))
		return
	}

	rubikConfig, err := server.getRubikConfig()
	if err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError,
			utils.NewResponseResultWithError(AppCode, FeatureCode, err))
		return
	}

	volcanoSchedulerConfig, err := server.getVolcanoSchedulerConfig()
	if err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError,
			utils.NewResponseResultWithError(AppCode, FeatureCode, err))
		return
	}

	// 筛选混部节点并构建结果
	var result ColocationConfigRequest
	var nodeList []NodeAttr
	for _, node := range nodes.Items {
		n := node
		nodeList = append(nodeList, server.buildNodeAttr(&n))
	}

	result = ColocationConfigRequest{
		NodeInfo:               nodeList,
		RubikConfig:            rubikConfig,
		VolcanoSchedulerConfig: volcanoSchedulerConfig,
	}
	response.WriteHeaderAndEntity(http.StatusOK, utils.NewResponseResultOk(result))
}

// getNodeFeatureSupport 获取节点特性支持信息
func (server *Server) getNodeFeatureSupport(node *corev1.Node) *NodeFeatureSupportInfo {
	featureSupport := &NodeFeatureSupportInfo{
		CPUPreemption:    "检测中",
		MemoryPreemption: "检测中",
		QuotaTurbo:       "检测中",
		DynMemory:        "检测中",
		DynCache:         "检测中",
		PSIDetection:     "检测中",
	}

	if labels := node.Labels; labels != nil {
		// 检查基础混部能力
		if val, exists := labels["openfuyao.com/colocation.feature.cpupreemption"]; exists && val == "true" {
			featureSupport.CPUPreemption = "OK"
		} else {
			featureSupport.CPUPreemption = "内核缺少cpu.qos_level接口"
		}

		if val, exists := labels["openfuyao.com/colocation.feature.memorypreemption"]; exists && val == "true" {
			featureSupport.MemoryPreemption = "OK"
		} else {
			featureSupport.MemoryPreemption = "内核缺少memory.qos_level接口"
		}

		// 检查高级功能
		if val, exists := labels["openfuyao.com/colocation.feature.quotaturbo"]; exists && val == "true" {
			featureSupport.QuotaTurbo = "OK"
		} else {
			featureSupport.QuotaTurbo = "内核缺少CPU.CFS控制接口"
		}

		if val, exists := labels["openfuyao.com/colocation.feature.dynmemory"]; exists && val == "true" {
			featureSupport.DynMemory = "OK"
		} else {
			featureSupport.DynMemory = "内核缺少memory.high_async_ratio接口"
		}

		if val, exists := labels["openfuyao.com/colocation.feature.dyncache"]; exists && val == "true" {
			featureSupport.DynCache = "OK"
		} else {
			featureSupport.DynCache = "硬件不支持RDT/MPAM特性"
		}

		if val, exists := labels["openfuyao.com/colocation.feature.psi"]; exists && val == "true" {
			featureSupport.PSIDetection = "OK"
		} else {
			featureSupport.PSIDetection = "内核不支持PSI特性"
		}
	}

	return featureSupport
}

// getRubikConfig retrieves and parses the rubik configuration from the colocation-config ConfigMap
func (server *Server) getRubikConfig() (RubikConfig, error) {
	// Get colocation-config ConfigMap
	colocationConfig, err := server.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).
		Get(context.Background(), "colocation-config", metav1.GetOptions{})
	if err != nil {
		klog.Errorf("Failed to get colocation-config ConfigMap: %v", err)
		return RubikConfig{}, err
	}

	// Initialize default config
	rubikConfig := RubikConfig{}

	// Parse rubik-options if exists
	if rubikOpts, exists := colocationConfig.Data["rubik-options"]; exists {
		if err := json.Unmarshal([]byte(rubikOpts), &rubikConfig); err != nil {
			klog.Errorf("Failed to parse rubik-options: %v", err)
			return rubikConfig, fmt.Errorf("rubik-options parse error: %w", err)
		}
	}

	return rubikConfig, nil
}

func (server *Server) getVolcanoSchedulerConfig() (VolcanoSchedulerConfig, error) {
	// Get colocation-config ConfigMap
	colocationConfig, err := server.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).
		Get(context.Background(), "colocation-config", metav1.GetOptions{})
	if err != nil {
		klog.Errorf("Failed to get colocation-config ConfigMap: %v", err)
		return VolcanoSchedulerConfig{}, err
	}

	volcanoConfig := VolcanoSchedulerConfig{}

	if volcanoOpts, exists := colocationConfig.Data["volcano-scheduler-options"]; exists {
		if err := json.Unmarshal([]byte(volcanoOpts), &volcanoConfig); err != nil {
			klog.Errorf("Failed to parse volcano-scheduler-options: %v", err)
			return volcanoConfig, fmt.Errorf("volcano-scheduler-options parse error: %w", err)
		}
	}

	return volcanoConfig, nil

}

// update node colocation status by colocation-status parameter
func (server *Server) toggleColocationNodeByNameWithRetry(request *restful.Request, response *restful.Response) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		return server.toggleColocationNodeByName(request, response)
	})
	if err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError,
			utils.NewResponseResultWithError(AppCode, FeatureCode, err))
	}
}

func (server *Server) handleColocationConfigWithRetry(request *restful.Request, response *restful.Response) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		return server.handleColocationConfig(request, response)
	})
	if err != nil {
		klog.Error(err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError,
			utils.NewResponseResultWithError(AppCode, FeatureCode, err))
	}
}

func (server *Server) handleColocationConfig(request *restful.Request, response *restful.Response) error {
	// 解析请求体
	newConfig := ColocationConfigRequest{}
	err := request.ReadEntity(&newConfig)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return err
	}

	err = server.coverColocationConfig(newConfig)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return err
	}

	updatedConfig, err := server.getUpdatedColocationConfig()
	if err != nil {
		// 即使获取失败，配置也已更新，返回一个成功的空响应
		klog.Errorf("Failed to get updated config after POST: %v", err)
		response.WriteHeaderAndEntity(http.StatusOK, utils.NewResponseResultOk(newConfig))
	} else {
		response.WriteHeaderAndEntity(http.StatusOK, utils.NewResponseResultOk(updatedConfig))
	}

	return nil
}

func (server *Server) getUpdatedColocationConfig() (*ColocationConfigRequest, error) {
	nodes, err := server.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	rubikConfig, err := server.getRubikConfig()
	if err != nil {
		return nil, err
	}

	volcanoSchedulerConfig, err := server.getVolcanoSchedulerConfig()
	if err != nil {
		return nil, err
	}

	var nodeList []NodeAttr
	for _, node := range nodes.Items {
		n := node
		nodeList = append(nodeList, server.buildNodeAttr(&n))
	}

	return &ColocationConfigRequest{
		NodeInfo:               nodeList,
		RubikConfig:            rubikConfig,
		VolcanoSchedulerConfig: volcanoSchedulerConfig,
	}, nil
}

func (server *Server) toggleColocationNodeByName(request *restful.Request, response *restful.Response) error {
	targetStatus := request.QueryParameter("colocation-status")
	if targetStatus != "true" && targetStatus != "false" {
		return response.WriteErrorString(http.StatusBadRequest, "colocation-status must be true or false")
	}
	if targetStatus == "true" {
		return server.addColocationNodeByName(request, response)
	}
	return server.deleteColocationNodeByName(request, response)
}

func (server *Server) addColocationNodeByName(request *restful.Request, response *restful.Response) error {
	klog.InfoS(request.Request.URL.Path)
	klog.InfoS(request.Request.URL.RawPath)
	klog.InfoS(request.Request.URL.RawQuery)
	nodeName := request.PathParameter("name")
	klog.Info("Adding colocation node:", nodeName)

	findNode, err := server.client.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		klog.InfoS("failed to get node:%#v", nodeName)
		klog.Error(err)
		return err
	}

	coloNode := []string{findNode.Name}

	if err := server.modifyColocationConfig(coloNode, "add"); err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError,
			utils.NewResponseResultWithError(AppCode, FeatureCode, err))
		return err
	}

	response.WriteHeaderAndEntity(
		http.StatusOK,
		utils.NewResponseResultOk(map[string]interface{}{
			"name":              nodeName,
			"colocation-status": "true",
		}),
	)
	return nil
}

// deletelColocationNodeById 把一个混部节点转变成普通节点
func (server *Server) deleteColocationNodeByName(request *restful.Request, response *restful.Response) error {
	nodeName := request.PathParameter("name")
	klog.Info("Deleting colocation node:", nodeName)

	findNode, err := server.client.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		klog.InfoS("failed to get node:%#v", nodeName)
		klog.Error(err)
		return err
	}

	coloNode := []string{findNode.Name}

	if err := server.modifyColocationConfig(coloNode, "delete"); err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError,
			utils.NewResponseResultWithError(AppCode, FeatureCode, err))
		return err
	}

	response.WriteHeaderAndEntity(
		http.StatusOK,
		utils.NewResponseResultOk(map[string]interface{}{
			"name":              nodeName,
			"colocation-status": "false",
		}),
	)
	return nil
}

// 检查是否是混部节点
func isColocationNode(node *corev1.Node) bool {
	if value, ok := node.Labels[colocationv1.ColocationNodeLabel]; ok {
		return value == "true"
	}
	return false
}
