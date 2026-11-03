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

package preflightcontract_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/render"
)

func TestRenderContractCoversOperatorStates(t *testing.T) {
	report := operatorReport()
	var detailed bytes.Buffer
	require.NoError(t, render.Text(&detailed, report))
	detailText := detailed.String()
	for _, expected := range []string{
		"openFuyao cluster-create 前置检查报告", "运行问题", "检查结果",
		"节点 CPU 资源", "SSH 认证", "管理集群 API", "依赖阻塞或跳过",
	} {
		assert.Contains(t, detailText, expected)
	}

	var summary bytes.Buffer
	require.NoError(t, render.Summary(&summary, report, "/tmp/preflight-report.txt"))
	assert.Contains(t, summary.String(), "需要处理")
	assert.Contains(t, summary.String(), "详细报告：/tmp/preflight-report.txt")
	assert.NotContains(t, summary.String(), "健康检查")

	var machine bytes.Buffer
	require.NoError(t, render.JSON(&machine, report))
	var decoded api.Report
	require.NoError(t, json.Unmarshal(machine.Bytes(), &decoded))
	require.NotEmpty(t, decoded.Results)
	assert.Equal(t, api.StatusFail, decoded.Results[0].Status)
}

func TestRenderContractCoversRunLifecycle(t *testing.T) {
	cases := []api.Report{
		{
			Operation: api.OperationInit, Overall: api.OverallSkip,
			Summary: api.Summary{},
		},
		{
			Operation: api.OperationClusterCreate, Overall: api.OverallInputError,
			Issues: []api.RunIssue{{
				Code: "CLUSTER-INPUT-SCHEMA", Severity: api.IssueError,
				Stage: api.IssueStageInput, Effect: api.IssueBlockStart, Field: "cluster",
				Cause:  "read cluster file /tmp/cluster.yaml: no such file or directory",
				Action: "correct the cluster file, then run preflight again",
			}},
		},
		{
			Operation: api.OperationClusterCreate, Overall: api.OverallInternalError,
			Issues: []api.RunIssue{{
				Code: "PREFLIGHT-CANCELED", Severity: api.IssueError,
				Stage: api.IssueStageExecution, Effect: api.IssueInterrupt,
				Cause: "context canceled", Action: "rerun preflight",
			}},
		},
	}
	for _, overall := range []api.Overall{
		api.OverallPass, api.OverallWarn, api.OverallFail, api.OverallInputError,
		api.OverallBundleError, api.OverallInternalError, api.OverallTimedOut,
		api.OverallCanceled, api.Overall("CUSTOM"),
	} {
		cases = append(cases, api.Report{Operation: api.OperationInit, Overall: overall})
	}
	for _, report := range cases {
		var text bytes.Buffer
		require.NoError(t, render.Text(&text, report))
		var summary bytes.Buffer
		require.NoError(t, render.Summary(&summary, report, ""))
		assert.NotEmpty(t, text.String())
		assert.NotEmpty(t, summary.String())
	}
}

func TestRenderContractCoversRepairCatalog(t *testing.T) {
	results := make([]api.Result, 0, len(repairRuleIDs())+len(collectionFailures()))
	for index, ruleID := range repairRuleIDs() {
		name := "Validation check"
		if strings.Contains(ruleID, "PORT") {
			name = "Required port [6443]"
		}
		results = append(results, result(fmt.Sprintf("repair-%d", index), ruleID, name,
			api.StatusFail, true, false))
	}
	results = append(results, collectionFailures()...)
	report := api.Report{
		Operation: api.OperationClusterCreate, Overall: api.OverallFail,
		Summary: api.Summary{Total: len(results), Fail: len(results)},
		Issues: []api.RunIssue{
			{Code: "PREFLIGHT-TIMEOUT", Severity: api.IssueError, Stage: api.IssueStageExecution,
				Effect: api.IssueContinue, Cause: "deadline exceeded"},
			{Code: "BUNDLE-INVALID", Severity: api.IssueError, Stage: api.IssueStagePlan,
				Effect: api.IssueContinue, Cause: "digest mismatch"},
			{Code: "CLUSTER-INPUT-REQUIRED", Severity: api.IssueError, Stage: api.IssueStageInput,
				Effect: api.IssueContinue, Cause: "cluster input missing"},
		},
		Results: results,
	}
	var output bytes.Buffer
	require.NoError(t, render.Text(&output, report))
	assert.Contains(t, output.String(), "Pod CIDR 无效")
	assert.Contains(t, output.String(), "来源不可达")
}

func TestRenderContractPropagatesWriterErrors(t *testing.T) {
	want := errors.New("write failed")
	writer := &failingWriter{err: want}
	assert.ErrorIs(t, render.Text(writer, operatorReport()), want)
	assert.ErrorIs(t, render.Summary(writer, operatorReport(), ""), want)
	assert.ErrorIs(t, render.JSON(writer, operatorReport()), want)
}

func operatorReport() api.Report {
	results := []api.Result{
		result("pass", "INIT-PLAT-OS", "健康检查 [uname -s]", api.StatusPass, "linux", "linux"),
		result("cpu", "CLUSTER-RES-CPU", "Node CPU [node-1]", api.StatusFail, 8, 4),
		result("memory", "CLUSTER-RES-MEMORY", "Node memory [node-1]", api.StatusFail,
			int64(16<<30), int64(8<<30)),
		result("disk", "CLUSTER-RES-SYSTEM-DISK", "Node disk [node-1]", api.StatusWarn,
			int64(100<<30), int64(80<<30)),
		result("ssh", "CLUSTER-SSH-AUTH", "SSH auth [node-1]", api.StatusFail, true, false),
		result("api", "CLUSTER-MGMT-API", "Management API", api.StatusFail, true, false),
		result("port", "CLUSTER-PORT-APISERVER", "API port [6443]", api.StatusFail, true, false),
		result("kernel", "CLUSTER-PLAT-KERNEL", "Kernel compatibility", api.StatusFail, true, false),
		result("skip", "CLUSTER-MGMT-DISCOVERY", "Discovery", api.StatusSkip, true, nil),
	}
	results[3].Actual.Type = api.ValueTypeBytes
	results[3].Expected.Type = api.ValueTypeBytes
	results[7].Evidence = []api.Evidence{{
		Kind: "command", Source: "ssh-node", Locator: "node-1",
		Observed: "distribution=openeuler version=24.03 arch=arm64 kernel=6.6.0",
	}}
	results[8].CauseCode = api.CauseDependencyFailed
	results[8].DependsOn = "api"
	return api.Report{
		SchemaVersion: api.SchemaVersion, Operation: api.OperationClusterCreate,
		Overall: api.OverallFail, ExitCode: api.ExitCode(api.OverallFail),
		Summary: api.Summary{Total: len(results), Pass: 1, Fail: 6, Warn: 1, Skip: 1},
		Issues: []api.RunIssue{
			{Code: "PREFLIGHT-OPTIONS", Severity: api.IssueWarn, Stage: api.IssueStageInput,
				Effect: api.IssueContinue, Field: "max-concurrency", Cause: "invalid value", Action: "use 1 to 64"},
			{Code: "NOTICE", Severity: api.IssueInfo, Stage: api.IssueStageSetup,
				Effect: api.IssueContinue, Cause: "informational message"},
		},
		Results: results,
	}
}

func result(id, ruleID, name string, status api.Status, expected, actual any) api.Result {
	return api.Result{
		InstanceID: id, RuleID: ruleID, Name: name, Status: status, Subject: "node-1",
		Expected: api.DisplayValue{Value: expected}, Actual: api.DisplayValue{Value: actual},
		Message:     "The observed value does not meet the requirement.",
		Remediation: "fix the reported value, then run preflight again",
	}
}

func repairRuleIDs() []string {
	return []string{
		"INIT-RES-CPU", "CLUSTER-RES-CPU", "INIT-RES-MEMORY", "CLUSTER-RES-MEMORY",
		"INIT-RES-SYSTEM-DISK", "CLUSTER-RES-SYSTEM-DISK", "INIT-PORT-K8S", "CLUSTER-PORT-APISERVER",
		"INIT-PLAT-OS", "CLUSTER-PLAT-OS", "INIT-PLAT-DISTRO", "CLUSTER-PLAT-DISTRO",
		"INIT-PLAT-ARCH", "INIT-TOOL-TAR", "INIT-TOOL-SYSTEMCTL", "CLUSTER-AGENT-SYSTEMD",
		"INIT-PATH-CREATE", "CLUSTER-LOCAL-PATH-CREATE", "INIT-PATH-MOUNT", "CLUSTER-LOCAL-PATH-MOUNT",
		"CLUSTER-LOCAL-PATH-ACCESS", "INIT-OFFLINE-EXISTS", "INIT-OFFLINE-READ", "INIT-OFFLINE-FORMAT",
		"INIT-REPO-URL", "INIT-REPO-CA-EXISTS", "INIT-REPO-CA-READ", "INIT-REPO-CA-PARSE",
		"INIT-COLOCATE", "CLUSTER-HA-COLOCATE", "CLUSTER-UNIQUE-IP", "CLUSTER-UNIQUE-HOSTNAME",
		"CLUSTER-TOPO-MASTER", "CLUSTER-TOPO-MASTER-ODD", "CLUSTER-TOPO-ETCD", "CLUSTER-TOPO-WORKER",
		"CLUSTER-MGMT-API", "CLUSTER-MGMT-DISCOVERY", "CLUSTER-MGMT-CRD-CLUSTER", "CLUSTER-MGMT-CRD-NODE",
		"CLUSTER-MGMT-AGENT-KUBECONFIG-GET", "CLUSTER-MGMT-AGENT-KUBECONFIG-DATA",
		"CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE", "CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE",
		"CLUSTER-MGMT-LOCAL-CLUSTER", "CLUSTER-MGMT-LOCAL-NODE", "CLUSTER-NET-POD-CIDR",
		"CLUSTER-NET-SERVICE-CIDR", "CLUSTER-REF-VERSION", "CLUSTER-ADDON-CLOSURE", "CLUSTER-ADDON-SOURCE",
	}
}

func collectionFailures() []api.Result {
	rules := []string{
		"CLUSTER-SSH-TRANSPORT", "CLUSTER-SSH-AUTH", "CLUSTER-SSH-PRIVILEGE",
		"CLUSTER-MGMT-AGENT-KUBECONFIG-GET", "CLUSTER-MGMT-API", "CLUSTER-MGMT-DISCOVERY",
		"CLUSTER-MGMT-CRD-BKECLUSTER", "CLUSTER-MGMT-CRD-BKENODE", "CUSTOM-NODE-CHECK", "CUSTOM-HOST-CHECK",
	}
	results := make([]api.Result, 0, len(rules)+1)
	for index, ruleID := range rules {
		item := result(fmt.Sprintf("collection-%d", index), ruleID, "Node check", api.StatusFail, true, nil)
		item.CauseCode = api.FactErrUnreachable
		item.Message = "connection failed"
		results = append(results, item)
	}
	source := result("source", "INIT-SRC-REACHABLE", "Image source is accessible",
		api.StatusFail, true, nil)
	source.CauseCode = api.FactErrTimeout
	results = append(results, source)
	return results
}

type failingWriter struct{ err error }

func (w *failingWriter) Write([]byte) (int, error) { return 0, w.err }
