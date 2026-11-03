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

// Package v1 provides the necessary tools and utilities to interact with mcs api service.
// It is responsible for creating API clients, API servers, registering endpoints and handling
// requests and reponses efficiently.
package v1

import (
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/runtime"
	"openfuyao.com/multi-cluster-service/pkg/cluster"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// AddToContainer initializes and adds routes to a RESTful container for mcs API service.
func AddToContainer(container *restful.Container, client client.Client) error {
	ws := runtime.NewWebService()
	handler := NewHandler()
	handler.ApiClient = NewAPIClient()

	kClient, kVersionedClient, err := utils.CreateKarmadaClusterClient()
	if err != nil {
		zlog.LogErrorf("Error Creating karmada kubernetes client : %v", err)
		return nil
	}
	handler.ClusterClient = cluster.NewClusteClient(client, kClient, kVersionedClient)

	go handler.autoRegisterHost()

	sayHello(ws, handler)
	clusterRegistration(ws, handler)
	clusterUnregistration(ws, handler)
	clusterEdition(ws, handler)
	clusterInspectUsage(ws, handler)
	resourceUsageInspection(ws, handler)
	policyRuleUpdation(ws, handler)
	kubeconfigRetrieval(ws, handler)

	container.Add(ws)
	return nil
}

func sayHello(ws *restful.WebService, h *handler) {
	ws.Route(ws.GET("/hello").To(h.sayHello))
}

func clusterInspectUsage(ws *restful.WebService, h *handler) {
	ws.Route(ws.GET("/clusters/{clusterTag}/resources").
		To(h.getClusterResourceUsage).
		Doc("Checking resource usage of the specified cluster").
		Returns(http.StatusOK, "Cluster unregister successfully", "successfully").
		Returns(http.StatusBadRequest, "Invalid request data",
			restful.ServiceError{Code: http.StatusBadRequest, Message: "Invalid data"}).
		Returns(http.StatusInternalServerError, "Internal server error",
			restful.ServiceError{Code: http.StatusInternalServerError, Message: "Server error"}).
		Produces(restful.MIME_JSON))
}

func resourceUsageInspection(ws *restful.WebService, h *handler) {
	ws.Route(ws.GET("/resources/clusters").
		To(h.getResourceUsage).
		Doc("Checking resource usage of the specified cluster").
		Returns(http.StatusOK, "Cluster unregister successfully", "successfully").
		Returns(http.StatusBadRequest, "Invalid request data",
			restful.ServiceError{Code: http.StatusBadRequest, Message: "Invalid data"}).
		Returns(http.StatusInternalServerError, "Internal server error",
			restful.ServiceError{Code: http.StatusInternalServerError, Message: "Server error"}).
		Produces(restful.MIME_JSON))
}

func clusterEdition(ws *restful.WebService, h *handler) {
	ws.Route(ws.PATCH("/clusters/{clusterTag}").
		To(h.editCluster).
		Doc("Tag the cluster").
		Reads(cluster.TagOfCluster{}).
		Returns(http.StatusOK, "Cluster labels modify successfully", cluster.InfoOfCluster{}).
		Returns(http.StatusBadRequest, "Invalid request data",
			restful.ServiceError{Code: http.StatusBadRequest, Message: "Invalid data"}).
		Returns(http.StatusInternalServerError, "Internal server error",
			restful.ServiceError{Code: http.StatusInternalServerError, Message: "Server error"}).
		Produces(restful.MIME_JSON))
}

func clusterRegistration(ws *restful.WebService, h *handler) {
	ws.Route(ws.POST("/clusters").
		To(h.registerCluster).
		Doc("Register a new cluster").
		Reads(cluster.InfoOfCluster{}).
		Returns(http.StatusOK, "Cluster register successfully", cluster.InfoOfCluster{}).
		Returns(http.StatusBadRequest, "Invalid request data",
			restful.ServiceError{Code: http.StatusBadRequest, Message: "Invalid data"}).
		Returns(http.StatusInternalServerError, "Internal server error",
			restful.ServiceError{Code: http.StatusInternalServerError, Message: "Server error"}).
		Produces(restful.MIME_JSON))
}

func clusterUnregistration(ws *restful.WebService, h *handler) {
	ws.Route(ws.DELETE("/clusters/{clusterTag}").
		To(h.unregisterCluster).
		Doc("Unregister the specified cluster").
		Returns(http.StatusOK, "Cluster unregister successfully", "successfully").
		Returns(http.StatusBadRequest, "Invalid request data",
			restful.ServiceError{Code: http.StatusBadRequest, Message: "Invalid data"}).
		Returns(http.StatusInternalServerError, "Internal server error",
			restful.ServiceError{Code: http.StatusInternalServerError, Message: "Server error"}).
		Produces(restful.MIME_JSON))
}

func policyRuleUpdation(ws *restful.WebService, h *handler) {
	ws.Route(ws.POST("/userinfo").
		To(h.authPolicyEnforcer).
		Doc("Update cluster policy rules").
		Reads(cluster.IdentityCollection{}).
		Returns(http.StatusOK, "Policy Rules update successfully", "successfully").
		Returns(http.StatusBadRequest, "Invalid request data",
			restful.ServiceError{Code: http.StatusBadRequest, Message: "Invalid data"}).
		Returns(http.StatusInternalServerError, "Internal server error",
			restful.ServiceError{Code: http.StatusInternalServerError, Message: "Server error"}).
		Produces(restful.MIME_JSON))
}

func kubeconfigRetrieval(ws *restful.WebService, h *handler) {
	ws.Route(ws.GET("/clusters/{clusterTag}/kubeconfig").
		To(h.kubeconfigRetrieval).
		Doc("Checking resource usage of the specified cluster").
		Returns(http.StatusOK, "Cluster Kubeconfig retrieve successfully", "successfully").
		Returns(http.StatusInternalServerError, "Internal server error",
			restful.ServiceError{Code: http.StatusInternalServerError, Message: "Server error"}).
		Produces(restful.MIME_JSON))
}
