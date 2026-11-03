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

// Package global contains global constants
package global

// used for construct the uri of openfuyao
const (
	ResourcesPluralCluster          = "clusters"
	MonitorServiceDefaultNamespace  = "monitoring-dashboard"
	MonitorServiceDefaultHost       = "monitoring-dashboard"
	MonitorServiceDefaultAPIVersion = "v1beta1"
	MonitorServiceDefaultOrgName    = "openfuyao.com"
	ApiRootPath                     = "/rest"
)

// restful response code
const (
	Success                          = 200
	FileCreated                      = 201
	NoContent                        = 204
	ClientError                      = 400
	ClientInvalidRequestBody         = 401
	ClientInvalidParameters          = 402
	ClientDuplicateNameExist         = 403
	ResourceNotFound                 = 404
	ServerError                      = 500
	ServerCreateWidgetFailed         = 501
	ServerUpdateWidgetFailed         = 502
	ServerDeleteWidgetFailed         = 503
	ServerGetIndicatorNameListFailed = 504
	ServerGetInstancesMetricFailed   = 505
	ServerGetInstancesFailed         = 506
	ServerCreateDashboardFailed      = 507
	ServerGetDashboardFailed         = 508
	ServerGetDashboardListFailed     = 509
	ServerUpdateDashboardFailed      = 510
	ServerDeleteDashboardFailed      = 511
)

const ()

// Dashboard: used for create crd in k8s
const (
	CRDRepoGroup   = "monitoring-dashboard.openfuyao.com"
	CRDRepoVersion = "v1beta1"
	// Widget CRD
	WidgetCRDRepoResource = "widgets"
	WidgetCRDRepoKind     = "Widget"
	WidgetCRDRepoListKind = "WidgetList"
	// Dashboard CRD
	DashboardRepoResource = "dashboards"
	DashboardRepoKind     = "Dashboard"
	DashboardRepoListKind = "DashboardList"
)

// MetadataNameRegExPattern regular expression constant
const (
	MetadataNameRegExPattern = "[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*"
)

// map key used for create crd in k8s
const (
	ModificationTimestamp = "modificationTimestamp"
	syncInProgressMsg     = "in_progress"
	syncCompleteMsg       = "complete"
	syncFailedMsg         = "failed"
)

// for prometheus query
const (
	PrometheusAPIPath          = "/api/v1"
	TimeSeriesSamplingInterval = 60
)

// for host/port validation
const (
	MinPortNumber = 1
	MaxPortNumber = 65535
)

// for external http response protection
const (
	MaxExternalResponseBytes = 10 * 1024 * 1024
)

// for model validate
const (
	MaxNameLength            = 50
	StandardIDLength         = 56
	UTCTimeLayout            = "2006-01-02T15:04:05Z"
	MaxIndicatorNameLength   = 127
	MaxInstanceOfWidget      = 8
	MaxInstanceOfStackedArea = 3
	LeastInstanceOfWidget    = 1
)

// for ID generate
const (
	UTCTimePartOfID = "2006-01-02-15-04-05-"
)
