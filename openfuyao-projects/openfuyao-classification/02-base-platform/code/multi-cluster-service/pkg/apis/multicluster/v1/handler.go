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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/api/resource"

	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/responsehandlers"
	"openfuyao.com/multi-cluster-service/pkg/cluster"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// handler centralizes the clients used to interface with different kubernetes APIs.
type handler struct {
	// ClusterClient is a operation client for interacting with the Cluster resources.
	ClusterClient cluster.OperationsInterface

	// http.client
	ApiClient *APIClient
}

// NewHandler defines a new handler structure.
func NewHandler() *handler {
	return &handler{}
}

func (h *handler) sayHello(req *restful.Request, resp *restful.Response) {
	responsehandlers.SendStatusOk(resp, "Hello, The MCS API Server Working Successfully !")
}

func (h *handler) autoRegisterHost() {
	h.ClusterClient.JoinHost(context.TODO())
}

func (h *handler) autoSendHello() {
	helloURL := fmt.Sprintf("%s%s", UserManagerEndpoint, UserManagerSignalUrl)
	_, err := h.ApiClient.Post(helloURL, "", "")
	if err != nil {
		zlog.LogErrorf("Error Sending Hello to User-Manager: %v", err)
	}
	fmt.Printf("Sending post to usermanager status code \n")
}

func (h *handler) authPolicyEnforcer(req *restful.Request, resp *restful.Response) {
	identityInfo := cluster.IdentityCollection{}
	identityInfo.UserInfo = make(map[string]*cluster.IdentityDescriptor)
	decoder := json.NewDecoder(req.Request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&identityInfo); err != nil {
		responsehandlers.SendStatusBadRequest(resp, "Invalid request data", err)
		return
	}

	authorizeOptions := cluster.Transfer2AuthorizeOpts(identityInfo)
	if err := h.ClusterClient.Authorize(context.TODO(), authorizeOptions); err != nil {
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		return
	}

	responsehandlers.SendStatusOk(resp, "AuthPolicies Updated successfully !")
	return

}

func (h *handler) editCluster(req *restful.Request, resp *restful.Response) {
	clusterTag := req.PathParameter("clusterTag")
	if clusterTag == "" {
		responsehandlers.SendStatusBadRequest(resp, "Invalid cluster name", nil)
		return
	}

	tags := cluster.TagOfCluster{}
	decoder := json.NewDecoder(req.Request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&tags); err != nil {
		responsehandlers.SendStatusBadRequest(resp, "Invalid request data", err)
		return
	}

	if !utils.IsLabelsValid(tags.ClusterLabels) {
		responsehandlers.SendStatusBadRequest(resp, "Invalid request data", nil)
		return
	}

	// check access
	if pass := checkUserAccess(req, resp, h.ClusterClient, "tag:cluster"); !pass {
		zlog.LogErrorf("Error Tagging cluster")
		return
	}

	editOptions := cluster.Transfer2EditOpts(clusterTag, tags)
	err := h.ClusterClient.Edit(context.TODO(), editOptions)
	if err != nil {
		if err.Error() == "cluster ID no exists" {
			responsehandlers.SendStatusBadRequest(resp, "Invalid request data, the cluster object dont exists.", err)
			return
		}
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		return
	}

	responsehandlers.SendStatusOk(resp, "Cluster tagged successfully")
	return
}

func (h *handler) registerCluster(req *restful.Request, resp *restful.Response) {
	kf := cluster.InfoOfCluster{}
	decoder := json.NewDecoder(req.Request.Body)
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&kf)
	if err != nil || kf.Name == "" {
		responsehandlers.SendStatusBadRequest(resp, "Invalid request data", err)
		return
	}

	// check if clustername and clusterlabels conform to Metadata naming conventions.
	if !utils.IsNameStringValid(kf.Name) || !utils.IsLabelsValid(kf.ClusterLabels) {
		responsehandlers.SendStatusBadRequest(resp, "Invalid request data, check name and clusterlabels again", nil)
		return
	}

	// check access
	if pass := checkUserAccess(req, resp, h.ClusterClient, "join:cluster"); !pass {
		return
	}

	joinOptions := cluster.Transfer2JoinOpts(kf)
	err = h.ClusterClient.Join(context.TODO(), joinOptions)
	if err != nil {
		if err.Error() == "cluster ID already exists" {
			responsehandlers.SendStatusBadRequest(resp, "Invalid request data, the cluster object already exists.", err)
			return
		}
		if err.Error() == "target Cluster ID contains karmada system or multicluster service" {
			responsehandlers.SendStatusBadRequest(resp, "Invalid join request, the cluster already contains karmada ", err)
			return
		}
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		return
	}

	time.Sleep(1 * time.Second)

	// send registering information to user-mgr
	msg := "MULTICLUSTERSERVICE sends cluster information for registering cluster"
	body := sync2UserMgr(h.ApiClient, req, msg, kf.Name, "join")
	if body != nil {
		// assert to clusterclient
		if client, ok := h.ClusterClient.(*cluster.OperationClient); ok {
			if err = client.MemberAuthPolicy(body, kf.Name, "join"); err != nil {
				zlog.LogErrorf("Error Synchronizing user permissions for the new joined cluster: %v", err)
			}
		} else {
			zlog.LogErrorf("Error asserting type from handler-interface to handler clusterclient")
		}
	}

	responsehandlers.SendStatusOk(resp, "Cluster registered successfully")
	return
}

func (h *handler) getResourceUsage(req *restful.Request, resp *restful.Response) {
	queryValues := req.Request.URL.Query()
	if len(queryValues) > 1 || (len(queryValues) == 1 && queryValues.Get("clusterTag") == "") {
		responsehandlers.SendStatusBadRequest(resp, "Invalid Resource Query.", nil)
		return
	}
	clusterTag := queryValues.Get("clusterTag")

	inspectOptions := cluster.Transfer2InspectOpts(clusterTag)
	clusterList, err := h.ClusterClient.Inspect(context.TODO(), inspectOptions)
	if err != nil {
		if err.Error() == "cluster ID no exists" {
			responsehandlers.SendStatusBadRequest(resp, "Invalid request data, the cluster object dont exists.", err)
			return
		}
		if err.Error() == "karmada service is not ready" {
			responsehandlers.SendtatusBadGateway(resp, "Karmada service is not ready", err)
			return
		}
		if err.Error() == "host Cluster is being onboarded" {
			responsehandlers.SendtatusBadGateway(resp, "Host cluster auto-registration is not ready", err)
			return
		}
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		return
	}

	// check username
	username, ok := req.Request.Context().Value("user").(string)
	if !ok {
		responsehandlers.SendStatusBadRequest(resp, "UnAuthorized user, user information no exists.", nil)
		return
	}

	// assert to clusterclient
	client, ok := h.ClusterClient.(*cluster.OperationClient)
	if !ok {
		responsehandlers.SendStatusServerError(resp, "Server error", nil)
		zlog.LogErrorf("Error asserting type from handler-interface to handler clusterclient")
		return
	}

	memList, err := utils.CheckUserAccessList(client.KarmadaClient, username)
	if err != nil {
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		return
	}

	clusterListAfterUserCheck := cluster.FilterClusterList(clusterList, memList)
	resp.WriteHeaderAndEntity(http.StatusOK, clusterListAfterUserCheck)
}

func (h *handler) getClusterResourceUsage(req *restful.Request, resp *restful.Response) {
	clusterTag := req.PathParameter("clusterTag")
	if clusterTag == "" {
		responsehandlers.SendStatusBadRequest(resp, "Invalid cluster name", nil)
		return
	}

	resourceUsage := &cluster.ResourcesInCluster{}

	// assert to clusterclient
	client, ok := h.ClusterClient.(*cluster.OperationClient)
	if !ok {
		responsehandlers.SendStatusServerError(resp, "Server error", nil)
		zlog.LogErrorf("Error asserting type from handler-interface to handler clusterclient")
		return
	}

	resourceUsage, err := client.Inspection(clusterTag)
	if err != nil {
		if err.Error() == "cluster ID no exists" {
			responsehandlers.SendStatusBadRequest(resp, "Invalid request data, the cluster object dont exists.", err)
			return
		}
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		return
	}

	cpuAndMemoryUsage := make(map[string]string)
	memStr := resourceUsage.Allocatable.Memory().String()
	memQuantity := resource.MustParse(memStr)
	cpuAndMemoryUsage["MEMORY"] = fmt.Sprintf("%.1fGi", float64(memQuantity.ScaledValue(resource.Giga)))

	usedMemStr := resourceUsage.Allocated.Memory().String()
	usedMemQuantity := resource.MustParse(usedMemStr)
	cpuAndMemoryUsage["USEDMEMORY"] = fmt.Sprintf("%.1fGi", float64(usedMemQuantity.ScaledValue(resource.Giga)))

	cpuStr := resourceUsage.Allocatable.Cpu().String()
	cpuQuantity := resource.MustParse(cpuStr)
	cpuAndMemoryUsage["CPU"] = fmt.Sprintf("%.1f", float64(cpuQuantity.MilliValue())/utils.MilliCoresPerCore)

	usedCpuStr := resourceUsage.Allocated.Cpu().String()
	usedCpuQuantity := resource.MustParse(usedCpuStr)
	cpuAndMemoryUsage["USEDCPU"] = fmt.Sprintf("%.1f", float64(usedCpuQuantity.MilliValue())/utils.MilliCoresPerCore)

	resp.WriteHeaderAndEntity(http.StatusOK, cpuAndMemoryUsage)
	return
}

func (h *handler) unregisterCluster(req *restful.Request, resp *restful.Response) {
	clusterTag := req.PathParameter("clusterTag")
	if clusterTag == "" {
		responsehandlers.SendStatusBadRequest(resp, "Invalid cluster name", nil)
		return
	}

	// check access
	if pass := checkUserAccess(req, resp, h.ClusterClient, "unjoin:cluster"); !pass {
		return
	}

	time.Sleep(1 * time.Second)

	// send unregistering information to user-mgr
	msg := "MULTICLUSTERSERVICE sends cluster information for unregistering cluster"
	body := sync2UserMgr(h.ApiClient, req, msg, clusterTag, "unjoin")
	if body != nil {
		// assert to clusterclient
		if client, ok := h.ClusterClient.(*cluster.OperationClient); ok {
			if err := client.MemberAuthPolicy(body, clusterTag, "unjoin"); err != nil {
				zlog.LogErrorf("Error Synchronizing user permissions for the new joined cluster: %v", err)
			}
		} else {
			zlog.LogErrorf("Error asserting type from handler-interface to handler clusterclient")
		}

	}

	unjoinOptions := cluster.Transfer2UnjoinOpts(clusterTag)
	err := h.ClusterClient.Unjoin(context.TODO(), unjoinOptions)
	if err != nil {
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		return
	}

	responsehandlers.SendStatusOk(resp, "Cluster unregistered successfully")
	return

}

func (h *handler) kubeconfigRetrieval(req *restful.Request, resp *restful.Response) {
	clusterTag := req.PathParameter("clusterTag")
	if clusterTag == "" {
		responsehandlers.SendStatusBadRequest(resp, "Invalid cluster name", nil)
		return
	}

	username, ok := req.Request.Context().Value("user").(string)
	if !ok {
		zlog.LogErrorf("Error Retrieving user information")
		responsehandlers.SendStatusBadRequest(resp, "UnAuthorized user, user information no exists.", nil)
		return
	}

	client, ok := h.ClusterClient.(*cluster.OperationClient)
	if !ok {
		responsehandlers.SendStatusServerError(resp, "Server error", nil)
		zlog.LogErrorf("Error asserting type from handler-interface to handler clusterclient")
		return
	}

	pass, config, err := client.CheckUserKubeconfigAccess(username, clusterTag)
	if err != nil {
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		return
	}
	if !pass {
		responsehandlers.SendStatusForbidden(resp, "Access Forbidden")
		return
	}

	resp.WriteHeaderAndEntity(http.StatusOK, config)
}
