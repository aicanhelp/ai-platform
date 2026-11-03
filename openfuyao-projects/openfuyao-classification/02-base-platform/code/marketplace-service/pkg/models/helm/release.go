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

package helm

import (
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/time"
	"sigs.k8s.io/cli-utils/pkg/kstatus/status"
)

// ReleaseResponse restful response release model fields
type ReleaseResponse struct {
	Name      string                 `json:"name,omitempty"`
	Info      *Info                  `json:"info,omitempty"`
	Chart     *Chart                 `json:"chart,omitempty"`
	Values    map[string]interface{} `json:"values,omitempty"`
	Resources []ReleaseResource      `json:"resources,omitempty"`
	Version   int                    `json:"version,omitempty"`
	Namespace string                 `json:"namespace,omitempty"`
	Labels    map[string]string      `json:"labels,omitempty"`
}

// ReleaseResource restful response release resource fields
type ReleaseResource struct {
	APIVersion string         `json:"apiVersion"`
	Kind       string         `json:"kind"`
	Name       string         `json:"name"`
	Namespace  string         `json:"namespace"`
	Status     *status.Result `json:"status"`
	Uid        string         `json:"uid"`
}

// Chart restful response chart fields
type Chart struct {
	Metadata *chart.Metadata        `json:"metadata,omitempty"`
	Values   map[string]interface{} `json:"values,omitempty"`
}

// Info describes release information.
type Info struct {
	FirstDeployed time.Time `json:"firstDeployed,omitempty"`
	LastDeployed  time.Time `json:"lastDeployed,omitempty"`
	Deleted       time.Time `json:"deleted,omitempty"`
	Description   string    `json:"description,omitempty"`
	// unknown|deployed|uninstalled|superseded|failed|uninstalling|pending-install|pending-upgrade|pending-rollback
	Status string `json:"status,omitempty"`
	// Contains the rendered templates/NOTES.txt if available
	Notes  string `json:"notes,omitempty"`
	Readme string `json:"readme,omitempty"`
}
