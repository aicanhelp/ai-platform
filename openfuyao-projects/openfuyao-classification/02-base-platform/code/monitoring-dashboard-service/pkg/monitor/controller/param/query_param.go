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

/*
Package param contains constant for console-service url parsing
*/
package param

// Resources Params is the resource params in uri path like {%s}
const (
	WidgetID    = "widgetId"
	DashboardID = "dashboardId"
	DisplayName = "displayName"
)

// Path Params is the query params in uri path like ?query=
const (
	DataSource = "dataSource"
	Start      = "start"
	End        = "end"
	Step       = "step"
)
