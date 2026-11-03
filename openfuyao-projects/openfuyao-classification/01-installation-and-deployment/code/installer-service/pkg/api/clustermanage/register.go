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
 */

package clustermanage

import (
	"strings"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"

	"installer-service/pkg/installer"
	"installer-service/pkg/zlog"
)

const (
	groupName   = "/rest/cluster"
	groupNameWs = "/ws/cluster"
	version     = "v1"
)

// GroupVersion 集群版本信息
var groupVersion = schema.GroupVersion{Group: groupName, Version: version}

var groupVersionWs = schema.GroupVersion{Group: groupNameWs, Version: version}

// NewWbeSevice 新建webService
func NewWebService(gv schema.GroupVersion) *restful.WebService {
	webservice := restful.WebService{}

	// 去除末尾的“/”
	webservice.Path(strings.TrimRight(gv.String(), "/")).Produces(restful.MIME_JSON)

	return &webservice
}

func ConfigInstallerWs(c *restful.Container, KubeConfig *rest.Config) error {
	webService := NewWebService(groupVersionWs)
	operation, err := installer.NewInstallerOperation(KubeConfig)
	if err != nil {
		zlog.Error("installer handler init failed,err :%v", err)
	}
	h := newHandler(operation)

	// 日志单独提出来
	webService.Route(webService.GET("/clusters/{cluster-name}/logs").
		Doc("get the log of work cluster ").
		To(h.getClusterLog))

	c.Add(webService)
	return nil
}

func ConfigInstaller(c *restful.Container, KubeConfig *rest.Config) error {
	webService := NewWebService(groupVersion)
	operation, err := installer.NewInstallerOperation(KubeConfig)
	if err != nil {
		zlog.Error("installer handler init failed,err :%v", err)
		return err
	}
	h := newHandler(operation)

	// Register routes according to swagger/OpenAPI spec
	registerClusterRoutes(webService, h)
	registerNodeRoutes(webService, h)
	registerUpgradeRoutes(webService, h)
	registerConfigRoutes(webService, h)

	c.Add(webService)
	return nil
}

func registerClusterRoutes(webService *restful.WebService, h *Handler) {
	// 创建集群 (swagger: POST /clusters)
	webService.Route(webService.POST("/clusters").
		Doc("create a work cluster ").
		To(h.createCluster))

	// 展示集群列表 (swagger: GET /clusters)
	webService.Route(webService.GET("/clusters").
		Doc(" list all the cluster ").
		To(h.listClusterGet))

	// 删除集群
	webService.Route(webService.DELETE("/clusters/{cluster-name}").
		Doc("delete a work cluster ").
		To(h.deleteCluster))

	// 集群详情页（swagger: GET /clusters/{cluster-name} 返回合并信息，包括 nodes）
	webService.Route(webService.GET("/clusters/{cluster-name}").
		Doc("get cluster info and nodes").
		To(h.getClusterFull))
}

func registerNodeRoutes(webService *restful.WebService, h *Handler) {
	// 集群操作前节点信息校验 (swagger: POST /nodes/validate)
	webService.Route(webService.POST("/nodes/validate").
		Doc("cluster nodes judge ").
		To(h.judgeClusterNode))

	// 展示集群节点列表
	webService.Route(webService.GET("/clusters/{cluster-name}/nodes").
		Doc("list all the node of a cluster").
		To(h.listNode))

	// 扩容 (swagger: POST /clusters/{cluster-name}/scale-up)
	webService.Route(webService.POST("/clusters/{cluster-name}/scale-up").
		Doc("scale up a work cluster ").
		To(h.scaleUpCluster))

	// 缩容 (swagger: POST /clusters/{cluster-name}/scale-down)
	webService.Route(webService.POST("/clusters/{cluster-name}/scale-down").
		Doc("scale down a work cluster ").
		To(h.scaleDownCluster))
}

func registerUpgradeRoutes(webService *restful.WebService, h *Handler) {
	// 升级 (swagger: POST /clusters/{cluster-name}/upgrade)
	webService.Route(webService.POST("/clusters/{cluster-name}/upgrade").
		Doc("upgrade a work cluster ").
		To(h.upgradeCluster))

	// 上传升级patch文件
	webService.Route(webService.POST("/patches").
		Doc("upload a patch upgrade file").
		To(h.uploadPatchFile))

	webService.Route(webService.GET("/versions").
		Doc("get available versions of openFuyao").
		To(h.getOpenFuyaoVersions))

	// 获取升级版本 swagger: GET /clusters/{cluster-name}/upgrade-versions
	webService.Route(webService.GET("/clusters/{cluster-name}/upgrade-versions").
		Doc("get available upgrade versions of openFuyao").
		To(h.getUpgradeOpenFuyaoVersions))

	// 开始自动升级补丁准备 swagger: POST /clusters/{cluster-name}/auto-upgrade
	webService.Route(webService.POST("/clusters/{cluster-name}/auto-upgrade").
		Doc("start auto upgrade patch preparation").
		To(h.autoUpgrade))
}

func registerConfigRoutes(webService *restful.WebService, h *Handler) {
	// 获得集群配置消息,用于离线安装
	webService.Route(webService.GET("/configs").
		Doc("get cluster default configs").
		To(h.getDefaultConfig))
}
