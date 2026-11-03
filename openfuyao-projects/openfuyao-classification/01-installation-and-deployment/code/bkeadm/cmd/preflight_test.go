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

package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/preflightbridge"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
	preflightbundle "gopkg.openfuyao.cn/preflight/bundle"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type unavailableTestCollector struct{}

func (unavailableTestCollector) ID() string { return "unavailable-test" }

func (unavailableTestCollector) Collect(_ context.Context, request api.FactRequest) ([]api.Fact, error) {
	subject := ""
	if len(request.Subjects) > 0 {
		subject = request.Subjects[0].ID
	}
	facts := make([]api.Fact, 0, len(request.Keys))
	for _, key := range request.Keys {
		facts = append(facts, api.Fact{
			Key: key, Subject: subject, Available: false,
			Error: &api.FactError{Code: api.FactErrUnsupported, Message: "test collector has no fact"},
		})
	}
	return facts, nil
}

// TestPreflightCommandRegistered freezes that the preflight command tree is
// registered on the root and exposes the expected subcommands.
func TestPreflightCommandRegistered(t *testing.T) {
	// Find preflight in the root's subcommands.
	var preflightFound bool
	for _, c := range rootCmd.Commands() {
		if c.Use == "preflight" {
			preflightFound = true
			// Verify subcommands.
			var initFound, clusterFound bool
			for _, sub := range c.Commands() {
				switch sub.Use {
				case "init [args same as bke init]":
					initFound = true
				case "cluster-create [args same as bke cluster create]":
					clusterFound = true
				}
			}
			assert.True(t, initFound, "preflight init subcommand must be registered")
			assert.True(t, clusterFound, "preflight cluster-create subcommand must be registered")
			assert.Nil(t, c.PersistentFlags().Lookup("request"), "preflight must not expose a second request input protocol")
		}
	}
	assert.True(t, preflightFound, "preflight command must be registered on root")
}

func TestPreflightTotalTimeoutContextStartsBeforePlanPreparation(t *testing.T) {
	previous := preflightTimeout
	preflightTimeout = 20 * time.Millisecond
	t.Cleanup(func() { preflightTimeout = previous })
	ctx, cancel, err := totalPreflightContext(context.Background())
	assert.NoError(t, err)
	defer cancel()
	select {
	case <-ctx.Done():
		assert.ErrorIs(t, ctx.Err(), context.DeadlineExceeded)
	case <-time.After(time.Second):
		t.Fatal("overall preflight context did not expire")
	}
}

func TestResolveContextErrorReportPreservesTimeoutAndCancellation(t *testing.T) {
	setup := []api.RunIssue{{Code: "SETUP", Severity: api.IssueError, Stage: api.IssueStageSetup, Effect: api.IssueContinue, Cause: "setup failed"}}
	deadlineCtx, deadlineCancel := context.WithCancelCause(context.Background())
	deadlineCancel(context.DeadlineExceeded)
	deadlineReport, ok := resolveContextErrorReport(deadlineCtx, api.OperationClusterCreate, setup, context.DeadlineExceeded)
	assert.True(t, ok)
	assert.Equal(t, api.OverallTimedOut, deadlineReport.Overall)
	assert.Equal(t, 124, deadlineReport.ExitCode)
	assert.Equal(t, "SETUP", deadlineReport.Issues[0].Code)
	assert.Equal(t, "PREFLIGHT-TIMEOUT", deadlineReport.Issues[1].Code)

	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	canceledReport, ok := resolveContextErrorReport(canceledCtx, api.OperationInit, nil, context.Canceled)
	assert.True(t, ok)
	assert.Equal(t, api.OverallCanceled, canceledReport.Overall)
	assert.Equal(t, 130, canceledReport.ExitCode)
	assert.Equal(t, "PREFLIGHT-CANCELED", canceledReport.Issues[0].Code)

	_, ok = resolveContextErrorReport(context.Background(), api.OperationInit, nil, errors.New("resolver failed"))
	assert.False(t, ok)
}

// TestPreflightInitFlagsFromSharedSpec freezes that the init subcommand binds
// flags from the shared InitFlagSpec, so bke init and bke preflight init cannot
// drift.
func TestPreflightInitFlagsFromSharedSpec(t *testing.T) {
	// Verify key flags are bound.
	for _, name := range []string{"hostIP", "kubernetesPort", "consolePort",
		"imageRepoPort", "runtime", "installConsole", "enableNTP", "confirm"} {
		f := preflightInitCmd.Flags().Lookup(name)
		assert.NotNil(t, f, "flag %s must be bound on preflight init", name)
	}
	// hostIP must be required.
	for _, e := range preflightInitCmd.Flags().Lookup("hostIP").Annotations["cobra_annotation_bash_completion_one_required_flag"] {
		_ = e
	}
}

func TestPreflightInitFlagSetMatchesOriginalInit(t *testing.T) {
	type signature struct{ shorthand, valueType, defaultValue, usage string }
	collect := func(flags *pflag.FlagSet) map[string]signature {
		result := map[string]signature{}
		flags.VisitAll(func(flag *pflag.Flag) {
			result[flag.Name] = signature{flag.Shorthand, flag.Value.Type(), flag.DefValue, flag.Usage}
		})
		return result
	}
	assert.Equal(t, collect(initCmd.Flags()), collect(preflightInitCmd.Flags()),
		"bke init and bke preflight init flags must share name, shorthand, type, default and usage")
}

// TestPreflightOutputFlags freeze the presentation flags.
func TestPreflightOutputFlags(t *testing.T) {
	assert.NotNil(t, preflightCmd.PersistentFlags().Lookup("output"))
	assert.NotNil(t, preflightCmd.PersistentFlags().Lookup("output-file"))
	assert.NotNil(t, preflightCmd.PersistentFlags().Lookup("timeout"))
	assert.NotNil(t, preflightCmd.PersistentFlags().Lookup("fact-timeout"))
	assert.NotNil(t, preflightCmd.PersistentFlags().Lookup("max-concurrency"))
	assert.NotNil(t, preflightCmd.PersistentFlags().Lookup("preflight-config"))
	assert.Nil(t, preflightCmd.PersistentFlags().Lookup("preflight-bundle"))
	assert.Nil(t, preflightCmd.PersistentFlags().Lookup("provider-defaults-file"))
}

func TestPreflightProgressUsesChineseOperatorText(t *testing.T) {
	var output bytes.Buffer
	reporter := newPreflightProgressReporter(&output, 5*time.Second)
	reporter.printLocked(api.ProgressEvent{Phase: api.ProgressCollecting, Completed: 1, Total: 2, Current: "CLUSTER-SSH-AUTH@node-1"}, false)
	reporter.printLocked(api.ProgressEvent{Phase: api.ProgressEvaluating, Completed: 1, Total: 2, Current: "CLUSTER-SSH-AUTH@node-1", Status: api.StatusFail}, false)
	assert.Contains(t, output.String(), "前置检查进度：采集事实 1/2")
	assert.Contains(t, output.String(), "计算结果 1/2")
	assert.Contains(t, output.String(), "结果=失败")
	assert.NotContains(t, output.String(), "Preflight progress")
}

func TestLoadPreflightAssetsRejectsBrokenEmbeddedPolicy(t *testing.T) {
	originalPolicy := append([]byte(nil), embeddedPreflightPolicy...)
	t.Cleanup(func() { SetEmbeddedPreflightPolicy(originalPolicy) })
	SetEmbeddedPreflightPolicy([]byte("not: [valid, yaml"))

	_, err := loadPreflightAssets()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "parse embedded preflight policy")
}

// seedEmbeddedPolicy installs the reviewed policy from the repo as the
// embedded policy for the duration of one test (cmd tests run without main's
// go:embed registration).
func seedEmbeddedPolicy(t *testing.T) {
	t.Helper()
	policy, err := os.ReadFile("../configs/preflight/preflight.yaml")
	require.NoError(t, err)
	original := append([]byte(nil), embeddedPreflightPolicy...)
	SetEmbeddedPreflightPolicy(policy)
	t.Cleanup(func() { SetEmbeddedPreflightPolicy(original) })
}

func TestLoadPreflightAssetsCompilesEmbeddedPolicy(t *testing.T) {
	seedEmbeddedPolicy(t)
	assets, err := loadPreflightAssets()
	require.NoError(t, err)
	b, err := preflightbundle.Load(assets.bundle)
	require.NoError(t, err)
	assert.NotEmpty(t, b.Rules)
}

func TestLoadPreflightRuntimeConfigUsesExternalThreshold(t *testing.T) {
	seedEmbeddedPolicy(t)
	base, err := loadPreflightAssets()
	require.NoError(t, err)
	policyRaw, err := os.ReadFile("../configs/preflight/preflight.yaml")
	require.NoError(t, err)

	modified := strings.Replace(string(policyRaw), "minimum: 2", "minimum: 13", 1)
	path := filepath.Join(t.TempDir(), "preflight.yaml")
	require.NoError(t, os.WriteFile(path, []byte(modified), 0o600))
	runtimeAssets, err := loadPreflightRuntimeConfig(path, base)
	require.NoError(t, err)

	runtimeBundle, err := preflightbundle.Load(runtimeAssets.bundle)
	require.NoError(t, err)
	for _, rule := range runtimeBundle.Rules {
		if rule.ID == "INIT-RES-CPU" {
			assert.Equal(t, 13, rule.Expected)
			return
		}
	}
	t.Fatal("INIT-RES-CPU is missing from runtime Bundle")
}

func TestLoadPreflightRuntimeConfigRejectsInvalidInput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "preflight.yaml")
	require.NoError(t, os.WriteFile(path, []byte("unknown: true\n"), 0o600))
	_, err := loadPreflightRuntimeConfig(path, preflightAssets{})
	require.ErrorContains(t, err, "field unknown not found")
}

func TestLoadPreflightRuntimeConfigRejectsIncompletePolicy(t *testing.T) {
	seedEmbeddedPolicy(t)
	base, err := loadPreflightAssets()
	require.NoError(t, err)
	minimal := []byte(`apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata: {}
operations: {init: {}, clusterCreate: {}}
`)
	path := filepath.Join(t.TempDir(), "preflight.yaml")
	require.NoError(t, os.WriteFile(path, minimal, 0o600))
	_, err = loadPreflightRuntimeConfig(path, base)
	require.ErrorContains(t, err, "operations.init.platform.distributions must not be empty")
}

func TestReleaseBundleErrorKeepsBundleOverall(t *testing.T) {
	report := releaseAssetIssueReport(api.OperationInit, errors.New("digest mismatch"))
	assert.Equal(t, api.OverallBundleError, report.Overall)
	assert.Equal(t, api.ExitInputError, report.ExitCode)
	require.Len(t, report.Issues, 1)
	assert.Equal(t, "BUNDLE-INVALID", report.Issues[0].Code)
}

// TestPreflightClusterCreateFlags freezes the cluster-create flag parity.
func TestPreflightClusterCreateFlags(t *testing.T) {
	f := preflightClusterCreateCmd.Flags().Lookup("file")
	assert.NotNil(t, f)
	assert.Equal(t, "f", f.Shorthand)
	n := preflightClusterCreateCmd.Flags().Lookup("nodes")
	assert.NotNil(t, n)
	assert.Equal(t, "n", n.Shorthand)
}

func TestManagementDefaultsFailureRemainsVisible(t *testing.T) {
	issues := managementSetupIssues(context.DeadlineExceeded)
	require.Len(t, issues, 1)
	assert.Equal(t, "CLUSTER-MGMT-PROVIDER-DEFAULTS", issues[0].Code)
	assert.Equal(t, api.IssueContinue, issues[0].Effect)
}

func TestInvalidPreflightOptionsReportsEachInvalidField(t *testing.T) {
	originalOutput := preflightOutput
	originalTimeout := preflightTimeout
	originalFactTimeout := preflightFactTimeout
	originalConcurrency := preflightMaxConcurrency
	t.Cleanup(func() {
		preflightOutput = originalOutput
		preflightTimeout = originalTimeout
		preflightFactTimeout = originalFactTimeout
		preflightMaxConcurrency = originalConcurrency
	})
	tests := []struct {
		name, output, field string
		timeout, fact       time.Duration
		concurrency         int
	}{
		{name: "output", output: "yaml", field: "output", timeout: time.Second, fact: time.Second},
		{name: "timeout", output: "text", field: "timeout", timeout: 0, fact: time.Second},
		{name: "fact timeout", output: "text", field: "fact-timeout", timeout: time.Second, fact: 0},
		{name: "concurrency", output: "text", field: "max-concurrency",
			timeout: time.Second, fact: time.Second, concurrency: maximumPreflightConcurrency + 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preflightOutput, preflightTimeout = tt.output, tt.timeout
			preflightFactTimeout, preflightMaxConcurrency = tt.fact, tt.concurrency
			report := invalidPreflightOptions(api.OperationInit)
			require.NotNil(t, report)
			require.Len(t, report.Issues, 1)
			assert.Equal(t, tt.field, report.Issues[0].Field)
			assert.Equal(t, api.OverallInputError, report.Overall)
		})
	}
	preflightOutput, preflightTimeout = "json", time.Second
	preflightFactTimeout, preflightMaxConcurrency = time.Second, 0
	assert.Nil(t, invalidPreflightOptions(api.OperationInit))
}

func TestInputIssueReportsAndSetupMerge(t *testing.T) {
	report := singleInputIssueReport(
		api.OperationInit, "INVALID", "init.options", errors.New("bad value"), "correct it",
	)
	assert.Equal(t, api.OverallInputError, report.Overall)
	assert.Equal(t, api.ExitInputError, report.ExitCode)
	require.Len(t, report.Issues, 1)
	assert.Equal(t, api.IssueBlockStart, report.Issues[0].Effect)

	passing := api.Report{Overall: api.OverallPass, ExitCode: api.ExitSuccess}
	mergeSetupIssues(&passing, []api.RunIssue{{Severity: api.IssueError}})
	assert.Equal(t, api.OverallFail, passing.Overall)
	assert.Equal(t, api.ExitFail, passing.ExitCode)
	warn := api.Report{Overall: api.OverallWarn, ExitCode: api.ExitWarn}
	mergeSetupIssues(&warn, []api.RunIssue{{Severity: api.IssueWarn}})
	assert.Equal(t, api.OverallWarn, warn.Overall)
	mergeSetupIssues(&warn, nil)
}

func TestManagementRouteOnlyMatchesClusterFacts(t *testing.T) {
	collector := preflightbridge.NewUnavailableManagementCollector("unavailable")
	routes := managementRoutes(collector)
	require.Len(t, routes, 1)
	cluster := api.Subject{ID: "cluster"}
	node := api.Subject{ID: "node-1"}
	assert.True(t, routes[0].Match(cluster, "kubernetes.api.reachable"))
	assert.True(t, routes[0].Match(cluster, "secret.kubeconfig.exists"))
	assert.False(t, routes[0].Match(node, "kubernetes.api.reachable"))
	assert.False(t, routes[0].Match(cluster, "node.cpu.logical"))
	assert.False(t, isManagementFactKey("node.cpu.logical"))
	assert.Empty(t, managementSetupIssues(nil))
}

func TestReportFileHelpers(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.json")
	require.NoError(t, writeReportAtomic(path, []byte("report")))
	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, "report", string(raw))
	info, err := os.Stat(path)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(preflightReportFileMode), info.Mode().Perm())

	generated, err := defaultPreflightReportPath(api.OperationClusterCreate, "json")
	require.NoError(t, err)
	assert.Contains(t, filepath.Base(generated), "bke-preflight-cluster-create-")
	assert.Equal(t, ".json", filepath.Ext(generated))
	_, err = os.Stat(generated)
	assert.ErrorIs(t, err, os.ErrNotExist)
	cleanupTemporaryReport(generated)
}

func TestProgressReporterCoversOperatorVisibleStates(t *testing.T) {
	var output bytes.Buffer
	reporter := newPreflightProgressReporter(&output, time.Second)
	reporter.Start()
	reporter.Report(api.ProgressEvent{Phase: api.ProgressPlan, Total: 3})
	reporter.Report(api.ProgressEvent{Phase: api.ProgressCollecting, Completed: 1, Total: 3, Current: "collecting facts"})
	reporter.Report(api.ProgressEvent{Phase: api.ProgressEvaluating, Completed: 2, Total: 3, Status: api.StatusWarn})
	reporter.Report(api.ProgressEvent{Phase: api.ProgressBlocked, Current: "blocked"})
	reporter.Report(api.ProgressEvent{Phase: api.ProgressIncomplete, Current: "incomplete"})
	reporter.Report(api.ProgressEvent{Phase: api.ProgressComplete, Completed: 3, Total: 3})
	reporter.Close()

	text := output.String()
	for _, expected := range []string{"解析输入", "采集事实", "计算结果", "检查未启动", "检查未完成", "完成 3/3 项"} {
		assert.Contains(t, text, expected)
	}
	assert.Equal(t, "通过", localizedProgressStatus(api.StatusPass))
	assert.Equal(t, "采集检查事实", localizedProgressCurrent("collecting facts"))
}

func TestWriteAndRenderJSONReport(t *testing.T) {
	originalOutput, originalFile := preflightOutput, preflightOutputFile
	originalStdout := os.Stdout
	reportOutput, err := os.CreateTemp(t.TempDir(), "stdout-*.json")
	require.NoError(t, err)
	preflightOutput, preflightOutputFile, os.Stdout = "json", filepath.Join(t.TempDir(), "report.json"), reportOutput
	t.Cleanup(func() {
		preflightOutput, preflightOutputFile, os.Stdout = originalOutput, originalFile, originalStdout
		_ = reportOutput.Close()
	})
	report := api.Report{
		SchemaVersion: api.SchemaVersion, Operation: api.OperationInit,
		Overall: api.OverallPass, ExitCode: api.ExitSuccess,
	}
	require.NoError(t, writeAndRenderPreflightReport(report))
	require.NoError(t, reportOutput.Close())
	raw, err := os.ReadFile(reportOutput.Name())
	require.NoError(t, err)
	assert.Contains(t, string(raw), `"overall": "PASS"`)
	detail, err := os.ReadFile(preflightOutputFile)
	require.NoError(t, err)
	assert.Equal(t, raw, detail)
	assert.Contains(t, string(mustRenderBytes(report, "text")), "前置检查报告")
}

func TestRunCheckExecutesSDKAndRendersFailureReport(t *testing.T) {
	seedEmbeddedPolicy(t)

	originalOutput, originalFile := preflightOutput, preflightOutputFile
	originalTimeout, originalFactTimeout := preflightTimeout, preflightFactTimeout
	originalConcurrency, originalStdout := preflightMaxConcurrency, os.Stdout
	stdout, err := os.CreateTemp(t.TempDir(), "stdout-*.json")
	require.NoError(t, err)
	preflightOutput, preflightOutputFile = "json", filepath.Join(t.TempDir(), "report.json")
	preflightTimeout, preflightFactTimeout, preflightMaxConcurrency = time.Second, time.Second, 1
	os.Stdout = stdout
	t.Cleanup(func() {
		preflightOutput, preflightOutputFile = originalOutput, originalFile
		preflightTimeout, preflightFactTimeout = originalTimeout, originalFactTimeout
		preflightMaxConcurrency, os.Stdout = originalConcurrency, originalStdout
		_ = stdout.Close()
	})

	input := preflightbridge.AdapterInput{
		Operation: api.OperationInit,
		Init: &initadapter.Input{
			HostIP: "192.0.2.10", Domain: "example.test", KubernetesPort: 36443,
			ImageRepoPort: 40443, ChartRepoPort: 38080, YumRepoPort: 40080,
		},
	}
	runErr := runCheckWithCollector(context.Background(), input, unavailableTestCollector{})
	var exitErr *commandExitError
	require.ErrorAs(t, runErr, &exitErr)
	assert.Equal(t, api.ExitFail, exitErr.code)
	require.NoError(t, stdout.Close())
	raw, err := os.ReadFile(stdout.Name())
	require.NoError(t, err)
	assert.Contains(t, string(raw), `"overall": "FAIL"`)
}

func TestCommandHandlersRenderTimeoutAsInputReport(t *testing.T) {
	originalOutput, originalFile := preflightOutput, preflightOutputFile
	originalTimeout, originalStdout := preflightTimeout, os.Stdout
	stdout, err := os.CreateTemp(t.TempDir(), "stdout-*.json")
	require.NoError(t, err)
	preflightOutput, preflightOutputFile = "json", filepath.Join(t.TempDir(), "report.json")
	preflightTimeout, os.Stdout = 0, stdout
	t.Cleanup(func() {
		preflightOutput, preflightOutputFile = originalOutput, originalFile
		preflightTimeout, os.Stdout = originalTimeout, originalStdout
		_ = stdout.Close()
	})

	for _, run := range []func(*cobra.Command, []string) error{runPreflightInit, runPreflightClusterCreate} {
		runErr := run(&cobra.Command{}, nil)
		var exitErr *commandExitError
		require.ErrorAs(t, runErr, &exitErr)
		assert.Equal(t, api.ExitInputError, exitErr.code)
	}
	renderErr := renderInvalidInitOptions(errors.New("invalid option"))
	var exitErr *commandExitError
	require.ErrorAs(t, renderErr, &exitErr)
	assert.Equal(t, api.ExitInputError, exitErr.code)
}

func TestExecuteRendersPreflightFlagParseErrors(t *testing.T) {
	originalOutput, originalFile := preflightOutput, preflightOutputFile
	originalStdout := os.Stdout
	t.Cleanup(func() {
		preflightOutput, preflightOutputFile = originalOutput, originalFile
		os.Stdout = originalStdout
		rootCmd.SetArgs(nil)
	})

	for _, args := range [][]string{
		{"preflight", "init", "--timeout", "invalid"},
		{"preflight", "init", "--not-a-real-flag"},
		{"preflight", "init", "--installConsole=not-bool"},
	} {
		stdout, err := os.CreateTemp(t.TempDir(), "stdout-*.txt")
		require.NoError(t, err)
		preflightOutput, preflightOutputFile, os.Stdout = "text", filepath.Join(t.TempDir(), "report.txt"), stdout
		rootCmd.SetArgs(args)

		assert.Equal(t, api.ExitInputError, Execute(), args)
		require.NoError(t, stdout.Close())
		raw, err := os.ReadFile(stdout.Name())
		require.NoError(t, err)
		assert.Contains(t, string(raw), "openFuyao init 前置检查", args)
		assert.Contains(t, string(raw), "状态：未启动", args)
		assert.Contains(t, string(raw), "命令参数无效；修正参数后重试", args)
	}
}

func TestPreflightFlagField(t *testing.T) {
	assert.Equal(t, "--timeout", preflightFlagField(errors.New(`invalid argument "bad" for "--timeout" flag`)))
	assert.Equal(t, "--unknown", preflightFlagField(errors.New("unknown flag: --unknown")))
	assert.Equal(t, "命令行参数", preflightFlagField(errors.New("unknown shorthand flag")))
	assert.Equal(t, "命令行参数", preflightFlagField(nil))
}

func TestExecuteRendersPreflightFlagParseErrorAsJSON(t *testing.T) {
	originalOutput, originalFile := preflightOutput, preflightOutputFile
	originalStdout := os.Stdout
	stdout, err := os.CreateTemp(t.TempDir(), "stdout-*.json")
	require.NoError(t, err)
	t.Cleanup(func() {
		preflightOutput, preflightOutputFile = originalOutput, originalFile
		os.Stdout = originalStdout
		rootCmd.SetArgs(nil)
		_ = stdout.Close()
	})
	preflightOutput, preflightOutputFile, os.Stdout = "text", filepath.Join(t.TempDir(), "report.json"), stdout
	rootCmd.SetArgs([]string{"preflight", "init", "--output", "json", "--timeout", "invalid"})

	assert.Equal(t, api.ExitInputError, Execute())
	require.NoError(t, stdout.Close())
	raw, err := os.ReadFile(stdout.Name())
	require.NoError(t, err)
	var report api.Report
	require.NoError(t, json.Unmarshal(raw, &report))
	assert.Equal(t, api.OverallInputError, report.Overall)
	require.Len(t, report.Issues, 1)
	assert.Equal(t, "--timeout", report.Issues[0].Field)
}

func TestWorkspaceAndInitFlagProjection(t *testing.T) {
	initInput := preflightbridge.AdapterInput{Init: &initadapter.Input{}}
	applyPreflightWorkspace(&initInput)
	assert.Equal(t, global.Workspace, initInput.Init.Workspace)
	clusterInput := preflightbridge.AdapterInput{ClusterCreate: &preflightbridge.ClusterCreateInput{}}
	applyPreflightWorkspace(&clusterInput)
	assert.Equal(t, global.Workspace, clusterInput.ClusterCreate.Workspace)

	flags := collectInitFlags(preflightInitCmd)
	assert.Contains(t, flags, "hostIP")
	assert.Contains(t, flags, "runtime")
}

func TestCommandExitErrorFormatting(t *testing.T) {
	assert.Equal(t, "cause", (&commandExitError{err: errors.New("cause")}).Error())
	assert.Equal(t, "preflight exit 3", (&commandExitError{code: 3}).Error())
}

// TestOriginalCommandsUnchanged freezes that the original bke commands are
// still registered alongside preflight.
func TestOriginalCommandsUnchanged(t *testing.T) {
	uses := make(map[string]bool)
	for _, c := range rootCmd.Commands() {
		uses[c.Use] = true
	}
	assert.True(t, uses["init"], "bke init must still be registered")
	assert.True(t, uses["cluster"], "bke cluster must still be registered")
	assert.True(t, uses["build"], "bke build must still be registered")
	assert.True(t, uses["version"], "bke version must still be registered")
}

// TestManagementHelpers freezes the management routing and setup-issue
// classification used before the SDK builds a cluster plan.
func TestManagementHelpers(t *testing.T) {
	// isManagementFactKey: only management-scoped keys match.
	for _, key := range []string{
		"kubernetes.api.reachable", "crd.bkeclusters", "object.exists.x",
		"auth.get.localkubeconfig", "secret.localkubeconfig.dataNonEmpty", "kubeconfig.parseable",
	} {
		assert.True(t, isManagementFactKey(key), key)
	}
	for _, key := range []string{"port.6443.free", "ssh.transport.node-1", "platform.os"} {
		assert.False(t, isManagementFactKey(key), key)
	}

	// managementRoutes matches only the cluster subject with management keys.
	routes := managementRoutes(preflightbridge.NewUnavailableManagementCollector("down"))
	require.Len(t, routes, 1)
	assert.True(t, routes[0].Match(api.Subject{ID: "cluster"}, "kubernetes.api.reachable"))
	assert.False(t, routes[0].Match(api.Subject{ID: "node-1"}, "kubernetes.api.reachable"))
	assert.False(t, routes[0].Match(api.Subject{ID: "cluster"}, "port.6443.free"))

	// managementSetupIssues: nil error yields no issues; generic errors keep
	// the connectivity wording; RBAC errors switch to the permission wording.
	assert.Nil(t, managementSetupIssues(nil))
	issues := managementSetupIssues(errors.New("connection refused"))
	require.Len(t, issues, 1)
	assert.Equal(t, "CLUSTER-MGMT-PROVIDER-DEFAULTS", issues[0].Code)
	assert.Contains(t, issues[0].Cause, "不可用")
	forbidden := managementSetupIssues(apierrors.NewForbidden(
		schema.GroupResource{Resource: "configmaps"}, "bke-config", errors.New("denied")))
	require.Len(t, forbidden, 1)
	assert.Contains(t, forbidden[0].Cause, "无权")
	assert.Contains(t, forbidden[0].Action, "只读权限")
}

// TestInputIssueReportHelpers freezes the structured input-error reports.
func TestInputIssueReportHelpers(t *testing.T) {
	report := singleInputIssueReport(api.OperationInit, "INIT-INPUT-INVALID",
		"init.options", errors.New("bad port"), "correct the option")
	assert.Equal(t, api.OverallInputError, report.Overall)
	assert.Equal(t, api.ExitInputError, report.ExitCode)
	require.Len(t, report.Issues, 1)
	assert.Equal(t, "INIT-INPUT-INVALID", report.Issues[0].Code)
	assert.Equal(t, api.IssueBlockStart, report.Issues[0].Effect)

	multi := inputIssueReport(api.OperationClusterCreate, []api.RunIssue{{
		Code: "CLUSTER-INPUT-REQUIRED", Severity: api.IssueError,
		Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
		Field: "nodes", Cause: "nodes file is required",
	}})
	assert.Equal(t, api.OperationClusterCreate, multi.Operation)
	require.Len(t, multi.Issues, 1)
}

// TestMergeSetupIssuesDowngradesPassToFail freezes the setup-issue merge:
// an ERROR setup issue must never leave a PASS/WARN overall in place.
func TestMergeSetupIssuesDowngradesPassToFail(t *testing.T) {
	report := api.Report{Overall: api.OverallPass, ExitCode: 0}
	mergeSetupIssues(&report, []api.RunIssue{{
		Code: "CLUSTER-MGMT-PROVIDER-DEFAULTS", Severity: api.IssueError,
		Stage: api.IssueStageSetup, Effect: api.IssueContinue, Cause: "down",
	}})
	assert.Equal(t, api.OverallFail, report.Overall)
	assert.Equal(t, api.ExitFail, report.ExitCode)
	require.Len(t, report.Issues, 1)

	// Non-error setup issues do not change the overall conclusion.
	warnReport := api.Report{Overall: api.OverallPass, ExitCode: 0}
	mergeSetupIssues(&warnReport, nil)
	assert.Equal(t, api.OverallPass, warnReport.Overall)
}

// TestApplyPreflightWorkspaceDefaults freezes the workspace defaulting.
func TestApplyPreflightWorkspaceDefaults(t *testing.T) {
	initInput := preflightbridge.AdapterInput{Init: &initadapter.Input{}}
	applyPreflightWorkspace(&initInput)
	assert.Equal(t, global.Workspace, initInput.Init.Workspace)

	clusterInput := preflightbridge.AdapterInput{ClusterCreate: &preflightbridge.ClusterCreateInput{}}
	applyPreflightWorkspace(&clusterInput)
	assert.Equal(t, global.Workspace, clusterInput.ClusterCreate.Workspace)

	explicit := preflightbridge.AdapterInput{Init: &initadapter.Input{Workspace: "/custom"}}
	applyPreflightWorkspace(&explicit)
	assert.Equal(t, "/custom", explicit.Init.Workspace)
}

// TestMarkExplicitInitPorts freezes the explicit-port marker used by the
// policy fallback precedence: flag names are recorded under their policy
// port keys (kubernetesPort -> kubernetes), and untouched flags stay absent.
func TestMarkExplicitInitPorts(t *testing.T) {
	options := preflightbridge.InitOptions{}
	command := preflightInitCmd
	require.NoError(t, command.Flags().Set("kubernetesPort", "36443"))
	markExplicitInitPorts(command, &options)
	assert.Equal(t, map[string]bool{"kubernetes": true}, options.ExplicitPorts)
}
