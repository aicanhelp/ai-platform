/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package initadapter

// Input is the resolved OpenFuyao init projection consumed by this product
// adapter. It is intentionally outside the generic SDK api package.
type Input struct {
	HostIP    string
	Domain    string
	Workspace string

	Runtime        string
	RuntimeStorage string

	OnlineImage   string
	OtherRepo     string
	OtherSource   string
	OtherChart    string
	ImageFilePath string

	KubernetesPort  int
	ConsolePort     int
	ImageRepoPort   int
	YumRepoPort     int
	ChartRepoPort   int
	AgentHealthPort int
	// ExplicitPorts records policy port keys explicitly supplied by the
	// operation input. It is adapter control metadata and never enters PlanHash.
	ExplicitPorts map[string]bool

	NtpServer      string
	InstallConsole bool
	EnableNTP      bool

	ImageRepoTLSVerify bool
	ImageRepoCAFile    string
	ImageRepoAuth      bool
}
