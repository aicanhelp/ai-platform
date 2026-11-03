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
package helm

import (
	"helm.sh/helm/v3/pkg/chart"

	"application-management-service/pkg/rpc"
)

// ChartGetter abstracts how a Helm Chart is obtained.
// This allows tests to inject a fake implementation, avoiding real RPC calls.
type ChartGetter interface {
	Get(repoName, chartName, version string) (*chart.Chart, error)
}

// rpcChartGetter is the default production implementation of ChartGetter.
type rpcChartGetter struct{}

func (g rpcChartGetter) Get(repoName, chartName, version string) (*chart.Chart, error) {
	return rpc.GetHelmChart(repoName, chartName, version)
}
