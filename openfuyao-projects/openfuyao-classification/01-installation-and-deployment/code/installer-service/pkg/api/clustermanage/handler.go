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
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"github.com/gorilla/websocket"

	"installer-service/pkg/constant"
	"installer-service/pkg/installer"
	"installer-service/pkg/utils/httputil"
	"installer-service/pkg/utils/wsutil"
	"installer-service/pkg/zlog"
)

// Handler installer handler that contains all installer operations
type Handler struct {
	installerHandler installer.Operation
}

func newHandler(handler installer.Operation) *Handler {
	return &Handler{installerHandler: handler}
}

func (h *Handler) createCluster(request *restful.Request, response *restful.Response) {
	req := installer.CreateClusterRequest{}
	if err := request.ReadEntity(&req); err != nil {
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	defaults, err := h.installerHandler.GetDefaultConfig()
	if err != nil {
		zlog.Errorf("get default config failed: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}
	yamlString, err := installer.BuildCreateClusterYaml(req, defaults)
	if err != nil {
		zlog.Errorf("build cluster yaml failed: %v", err)
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetDefaultClientFailureResponseJson())
		return
	}

	if err := h.installerHandler.CreateCluster(yamlString); err != nil {
		zlog.Errorf("create cluster err: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}

	response.WriteHeaderAndEntity(http.StatusOK, httputil.GetDefaultSuccessResponseJson())
}

func (h *Handler) judgeClusterNode(request *restful.Request, response *restful.Response) {
	req := installer.ClusterNodeInfo{}

	if err := request.ReadEntity(&req); err != nil {
		zlog.Errorf("req: %v, err: %v", req, err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}

	if err := h.installerHandler.JudgeClusterNode(&req); err != nil {
		zlog.Errorf("req: %v, err: %v", req, err)
		response.WriteHeaderAndEntity(http.StatusBadRequest, &httputil.ResponseJson{
			Code:    constant.ClientError,
			Message: fmt.Sprintf("%v", err),
			Data:    nil,
		})
		return
	}

	response.WriteHeaderAndEntity(http.StatusOK, httputil.GetDefaultSuccessResponseJson())
}

// [MOD-2] WebSocket 升级器：CheckOrigin 校验 Origin，防止 CSWSH
var upgrade = websocket.Upgrader{
	CheckOrigin: wsutil.CheckOrigin,
}

// 命名空间 和  集群名称  一一对应
func (h *Handler) getClusterLog(request *restful.Request, response *restful.Response) {

	clusterName := request.PathParameter("cluster-name")
	zlog.Info(clusterName)

	conn, err := upgrade.Upgrade(response.ResponseWriter, request.Request, nil)
	if err != nil {
		zlog.Warn("Failed to upgrade connection:", err)
		response.WriteHeader(http.StatusInternalServerError)
		response.Write([]byte("Internal Server Error - Failed to upgrade connection."))
		return
	}
	defer func(conn *websocket.Conn) {
		err := conn.Close()
		if err != nil {
			zlog.Warn("Failed to close the conn")
		}
	}(conn)

	zlog.Info("1")
	result, status := h.installerHandler.GetClusterLog(clusterName, conn)
	zlog.Info("2")
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) listNode(request *restful.Request, response *restful.Response) {
	clusterName := request.PathParameter("cluster-name")
	if clusterName == "" {
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	// 从查询参数获取 nodeName（可选）
	nodeName := request.QueryParameter("nodeName") // 如果未提供，返回空字符串

	// 构造 NodeRequest
	req := installer.NodeRequest{
		ClusterName: clusterName,
		NodeName:    nodeName,
	}
	zlog.Info("req data: %v", req)
	data, err := h.installerHandler.GetNodesByQuery(req)
	if err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}
	res := httputil.GetDefaultSuccessResponseJson()
	res.Data = data
	response.WriteHeaderAndEntity(http.StatusOK, res)
}

// listClusterGet handles GET /clusters (supports query params)
func (h *Handler) listClusterGet(request *restful.Request, response *restful.Response) {
	// Use installer.GetAllClusters to return all created clusters
	data, err := h.installerHandler.GetAllClusters()
	if err != nil {
		zlog.Errorf("failed to get clusters: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}
	res := httputil.GetDefaultSuccessResponseJson()
	res.Data = data
	response.WriteHeaderAndEntity(http.StatusOK, res)
}

// getClusterFull handles GET /clusters/{cluster-name} and returns merged cluster config and nodes
func (h *Handler) getClusterFull(request *restful.Request, response *restful.Response) {
	clusterName := request.PathParameter("cluster-name")
	if clusterName == "" {
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	data, err := h.installerHandler.GetClusterFull(clusterName)
	if err != nil {
		zlog.Errorf("failed to get cluster full info: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}

	res := httputil.GetDefaultSuccessResponseJson()
	res.Data = data
	response.WriteHeaderAndEntity(http.StatusOK, res)
}

func (h *Handler) deleteCluster(request *restful.Request, response *restful.Response) {
	clusterName := request.PathParameter("cluster-name")
	zlog.Info(clusterName)
	result, status := h.installerHandler.DeleteCluster(clusterName)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getDefaultConfig(request *restful.Request, response *restful.Response) {
	data, err := h.installerHandler.GetDefaultConfig()
	if err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}
	res := httputil.GetDefaultSuccessResponseJson()
	res.Data = data
	response.WriteHeaderAndEntity(http.StatusOK, res)
}

// 扩容
func (h *Handler) scaleUpCluster(request *restful.Request, response *restful.Response) {
	// 新的扩容实现：接收节点列表并创建对应的 BKENode CRs
	req := installer.ScaleUpRequest{}

	if err := request.ReadEntity(&req); err != nil {
		zlog.Errorf("read req failed: %v", err)
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	if len(req.Nodes) == 0 {
		zlog.Errorf("req nodes is empty")
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	clusterName := request.PathParameter("cluster-name")
	if clusterName != "" {
		zlog.Infof("scale up cluster: %s, nodes: %v", clusterName, req.Nodes)
	}

	if err := h.installerHandler.CreateBKENodes(clusterName, req.Nodes); err != nil {
		zlog.Errorf("CreateBKENodes failed: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}

	response.WriteHeaderAndEntity(http.StatusOK, httputil.GetDefaultSuccessResponseJson())
}

// 缩容
func (h *Handler) scaleDownCluster(request *restful.Request, response *restful.Response) {
	req := installer.ScaleDownRequest{}

	if err := request.ReadEntity(&req); err != nil {
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	if len(req.Nodes) == 0 {
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	clusterName := request.PathParameter("cluster-name")
	if clusterName != "" {
		zlog.Infof("scale down cluster: %s, nodes: %v", clusterName, req.Nodes)
	}

	// 删除相应的 BKENode CRs
	if err := h.installerHandler.DeleteBKENodes(clusterName, req.Nodes); err != nil {
		zlog.Errorf("DeleteBKENodes failed: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}

	response.WriteHeaderAndEntity(http.StatusOK, httputil.GetDefaultSuccessResponseJson())
}

// 升级
func (h *Handler) upgradeCluster(request *restful.Request, response *restful.Response) {
	// 新的升级实现：前端传入 {"version":"vX"}，服务端将 BKECluster 的 openFuyaoVersion 打补丁
	req := installer.UpgradeRequest{}

	if err := request.ReadEntity(&req); err != nil {
		zlog.Errorf("Read upgrade entity failed: %v", err)
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	if req.Version == "" {
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	clusterName := request.PathParameter("cluster-name")
	if clusterName != "" {
		zlog.Infof("upgrade cluster: %s to version %s", clusterName, req.Version)
	}

	if err := h.installerHandler.UpgradeClusterVersion(clusterName, req.Version); err != nil {
		zlog.Errorf("upgrade cluster failed: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}

	response.WriteHeaderAndEntity(http.StatusOK, httputil.GetDefaultSuccessResponseJson())
}

// 上传升级patch文件
func (h *Handler) uploadPatchFile(request *restful.Request, response *restful.Response) {
	req := installer.PatchFileData{}

	if err := request.ReadEntity(&req); err != nil {
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	err := h.installerHandler.UploadPatchFile(req.FileName, req.FileContent)
	if err != nil {
		zlog.Errorf("upload patch file err: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}
	response.WriteHeaderAndEntity(http.StatusOK, httputil.GetDefaultSuccessResponseJson())
}

func (h *Handler) getOpenFuyaoVersions(request *restful.Request, response *restful.Response) {
	data, err := h.installerHandler.GetOpenFuyaoVersions()
	if err != nil {
		zlog.Errorf("get openFuyao versions err: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}
	res := httputil.GetDefaultSuccessResponseJson()
	res.Data = map[string][]string{"versions": data}
	response.WriteHeaderAndEntity(http.StatusOK, res)
}

func (h *Handler) getInstallableVersions(request *restful.Request, response *restful.Response) {
	data, err := h.installerHandler.GetInstallableVersions()
	if err != nil {
		zlog.Errorf("get installable versions err: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}
	res := httputil.GetDefaultSuccessResponseJson()
	res.Data = map[string][]string{"versions": data}
	response.WriteHeaderAndEntity(http.StatusOK, res)
}

func (h *Handler) getUpgradeOpenFuyaoVersions(request *restful.Request, response *restful.Response) {
	// For GET /clusters/{cluster-name}/upgrade-versions we read the cluster's current
	// openFuyaoVersion from the cluster resource and pass it to GetUpgradeableVersions.
	// Fallback: accept currentVersion via query/body for backward compatibility.
	currentVersion := request.QueryParameter("currentVersion")
	clusterName := request.PathParameter("cluster-name")

	if clusterName != "" {
		// try to obtain current version from cluster resource
		if cluster, err := h.installerHandler.GetClustersByName(clusterName); err == nil && cluster != nil {
			// read nested field where installer stores openFuyaoVersion
			if cluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion != "" {
				currentVersion = cluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion
			}
		} else if currentVersion == "" {
			// unable to determine current version from cluster and none provided
			zlog.Errorf("failed to get cluster %s info: %v", clusterName, err)
			response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
			return
		} else {
			// failed to read version from cluster resource, keep currentVersion provided by query param
			zlog.Warnf("failed to get cluster %s info: %v, fallback to provided currentVersion %s",
				clusterName, err, currentVersion)
		}
		zlog.Infof("get upgrade versions for cluster %s, currentVersion=%s", clusterName, currentVersion)
	} else {
		// fallback to body if no path param provided
		if currentVersion == "" {
			req := installer.ClusterPatchInfo{}
			if err := request.ReadEntity(&req); err == nil {
				currentVersion = req.ClusterVersion
			}
		}
	}

	data, err := h.installerHandler.GetUpgradeableVersions(currentVersion)
	if err != nil {
		zlog.Errorf("get upgradeable versions err: %v", err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, httputil.GetDefaultServerFailureResponseJson())
		return
	}
	res := httputil.GetDefaultSuccessResponseJson()
	res.Data = map[string][]string{"versions": data}
	response.WriteHeaderAndEntity(http.StatusOK, res)
}

// autoUpgrade 同步执行升级准备脚本
func (h *Handler) autoUpgrade(request *restful.Request, response *restful.Response) {
	req := installer.AutoUpgradeRequest{}
	if err := request.ReadEntity(&req); err != nil {
		zlog.Errorf("Read autoUpgrade entity failed: %v", err)
		response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetParamsEmptyErrorResponseJson())
		return
	}

	clusterName := request.PathParameter("cluster-name")
	if clusterName != "" {
		zlog.Infof("auto-upgrade for cluster: %s", clusterName)
	}

	if err := h.installerHandler.AutoUpgradePatchPrepare(clusterName, req); err != nil {
		// 业务逻辑执行失败，返回 500 或 400，并带上业务错误信息
		zlog.Errorf("Auto upgrade preparation failed: %v", err)

		// 假设业务层返回的 error 包含了用户可读的失败信息
		response.WriteHeaderAndEntity(
			http.StatusInternalServerError,
			httputil.GetResponseJson(constant.ServerError, err.Error(), nil),
		)
		return
	}

	// 成功
	response.WriteHeaderAndEntity(http.StatusOK, httputil.GetDefaultSuccessResponseJson())
}
