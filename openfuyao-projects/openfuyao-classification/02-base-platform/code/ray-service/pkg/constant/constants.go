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

/*
Package constant
contains constant for console-service
*/
package constant

import (
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
)

type ContextKey string

// List Status
const (
	// running status
	Running string = "Running"
	// stopped status
	Suspended string = "Suspended"
	// failed status
	Failed string = "Failed"

	// ray cluster
	RayCluster = "raycluster"
	RayJob     = "rayjob"
	RayService = "rayservice"
	Default    = "default"

	// template status
	Instance = "实例"
	Template = "模板"
)

// GetValidCrdResources 返回有效的CRD资源映射
func GetValidCrdResources() map[string]any {
	return map[string]any{
		RayCluster: RayCluster,
		RayJob:     RayJob,
		RayService: RayService,
	}
}

// GetJobDeploymentStatusMap 返回作业部署状态映射
func GetJobDeploymentStatusMap() map[rayv1.JobDeploymentStatus]string {
	return map[rayv1.JobDeploymentStatus]string{
		rayv1.JobDeploymentStatusInitializing: "初始化",
		rayv1.JobDeploymentStatusRunning:      "运行中",
		rayv1.JobDeploymentStatusComplete:     "完成",
		rayv1.JobDeploymentStatusFailed:       "失败",
		rayv1.JobDeploymentStatusSuspending:   "暂停中",
		rayv1.JobDeploymentStatus("Retrying"): "重试中",
		rayv1.JobDeploymentStatusSuspended:    "未运行",
	}
}

// GetServiceStatusMap 返回服务状态映射
func GetServiceStatusMap() map[rayv1.ServiceStatus]string {
	return map[rayv1.ServiceStatus]string{
		rayv1.WaitForServeDeploymentReady: "服务部署中",
		rayv1.Running:                     "运行中",
		rayv1.Restarting:                  "重启中",
		rayv1.ServiceStatus(Suspended):    "未运行",
		rayv1.ServiceStatus(Failed):       "失败",
	}
}

// GetClusterStatusMap 返回集群状态映射
func GetClusterStatusMap() map[rayv1.ClusterState]string {
	return map[rayv1.ClusterState]string{
		rayv1.Ready:     "运行中",
		rayv1.Failed:    "失败",
		rayv1.Unhealthy: "不健康",
		rayv1.Suspended: "未运行",
	}
}

// GetGlobalStatusMap 返回全局状态映射
func GetGlobalStatusMap() map[string]string {
	return map[string]string{
		Running:   "运行中",
		Suspended: "未运行",
		Failed:    "失败",
	}
}

// GetServiceFailedMap 返回服务失败映射
func GetServiceFailedMap() map[rayv1.ServiceStatus]string {
	return map[rayv1.ServiceStatus]string{
		rayv1.FailedToGetServeDeploymentStatus: "服务部署失败",
		rayv1.FailedToGetOrCreateRayCluster:    "Ray集群部署失败",
		rayv1.FailedToUpdateServingPodLabel:    "更新服务标签失败",
		rayv1.FailedToUpdateService:            "更新服务失败",
	}
}

// openfuyao-rayservice host constant
const (
	// ContextDBKey is the key for db in context
	ContextDBKey                    = ContextKey("db")
	ResourcesPluralCluster          = "clusters"
	ConsoleServiceDefaultNamespace  = "fuyao-system"
	ConsoleServiceDefaultHost       = "console"
	ConsoleServiceDefaultAPIVersion = "v1beta1"
	ConsoleServiceDefaultOrgName    = "openfuyao.com"
	RayClusterCreateBy              = "raycluster-createby"
	KuberayCrdCreateBy              = "kuberaycrd-createby"
	BadRequest                      = "bad request"
)

// regular expression constant
const (
	MetadataNameRegExPattern = "[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*"
)

// restful response code
const (
	Success          = 200
	FileCreated      = 201
	NoContent        = 204
	ClientError      = 400
	ResourceNotFound = 404
	ServerError      = 500
)
