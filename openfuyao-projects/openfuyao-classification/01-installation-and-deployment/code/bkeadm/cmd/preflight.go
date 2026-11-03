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
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/spf13/cobra"
	apierrors "k8s.io/apimachinery/pkg/api/errors"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/preflightbridge"
	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	preflight "gopkg.openfuyao.cn/preflight"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/collectors/router"
	"gopkg.openfuyao.cn/preflight/render"
)

const (
	defaultPreflightTimeout     = 10 * time.Minute
	defaultPreflightFactTimeout = 30 * time.Second
	maximumPreflightConcurrency = 64
	progressHeartbeatInterval   = 5 * time.Second
	progressPercentTotal        = 100.0
	preflightReportFileMode     = 0o600
)

// preflight output options (presentation only; do not enter PlanHash).
var (
	preflightOutput         string
	preflightOutputFile     string
	preflightConfigFile     string
	preflightTimeout        time.Duration
	preflightFactTimeout    time.Duration
	preflightMaxConcurrency int
)

// preflightCmd is the top-level preflight command. It is independent and
// read-only: it does not call bke init / cluster create, and registering it
// must not create any directories (PF-01 guarantees pkg/global no longer does
// so at package init).
var preflightCmd = &cobra.Command{
	Use:           "preflight",
	Short:         "Run read-only installation preflight checks",
	SilenceErrors: true,
	SilenceUsage:  true,
	Long: `Run read-only installation preflight checks before bke init or bke cluster create.
Preflight never modifies the host, nodes or Kubernetes resources; it only reads
and reports PASS/FAIL/WARN/SKIP with stable exit codes.`,
	RunE: func(cmd *cobra.Command, _ []string) error { return cmd.Help() },
}

var preflightInitCmd = &cobra.Command{
	Use: "init [args same as bke init]", Short: "Preflight check for bke init",
	Long: `Run the init preflight with the same parameters as 'bke init'.`, RunE: runPreflightInit,
	SilenceErrors: true, SilenceUsage: true,
}

var preflightClusterCreateCmd = &cobra.Command{
	Use: "cluster-create [args same as bke cluster create]", Short: "Preflight check for bke cluster create",
	RunE: runPreflightClusterCreate, SilenceErrors: true, SilenceUsage: true,
}

func registerPreflightCommands() {
	rootCmd.AddCommand(preflightCmd)
	preflightCmd.AddCommand(preflightInitCmd)
	preflightCmd.AddCommand(preflightClusterCreateCmd)
	registerPreflightRuntimeFlags()
	registerPreflightInitFlags()
	preflightClusterCreateCmd.Flags().
		StringVarP(&preflightClusterFile, "file", "f", "", "BKE Cluster Configuration File")
	preflightClusterCreateCmd.Flags().
		StringVarP(&preflightClusterNodes, "nodes", "n", "", "BKE Nodes Configuration File")
}

func registerPreflightRuntimeFlags() {
	preflightCmd.PersistentFlags().StringVar(&preflightOutput, "output", "text", "output format: text|json")
	preflightCmd.PersistentFlags().StringVar(&preflightOutputFile, "output-file", "", "write report to file")
	preflightCmd.PersistentFlags().StringVar(
		&preflightConfigFile,
		"preflight-config",
		"",
		"use a complete external preflight.yaml policy",
	)
	preflightCmd.PersistentFlags().DurationVar(
		&preflightTimeout,
		"timeout",
		defaultPreflightTimeout,
		"overall timeout",
	)
	preflightCmd.PersistentFlags().
		DurationVar(
			&preflightFactTimeout,
			"fact-timeout",
			defaultPreflightFactTimeout,
			"per-fact collection timeout",
		)
	preflightCmd.PersistentFlags().
		IntVar(
			&preflightMaxConcurrency,
			"max-concurrency",
			0,
			"maximum concurrent read-only fact probes (0: adaptive, bounded)",
		)
}

func registerPreflightInitFlags() {
	for _, spec := range preflightbridge.InitFlagSpecs() {
		if def, ok := spec.Default.(bool); ok {
			var b bool
			preflightInitCmd.Flags().BoolVar(&b, spec.Name, def, spec.Usage)
			if spec.Shorthand != "" {
				_ = preflightInitCmd.Flags().Lookup(spec.Name)
			}
			continue
		}
		var s string
		if spec.Shorthand != "" {
			preflightInitCmd.Flags().
				StringVarP(&s, spec.Name, spec.Shorthand, fmt.Sprintf("%v", spec.Default), spec.Usage)
		} else {
			preflightInitCmd.Flags().StringVar(&s, spec.Name, fmt.Sprintf("%v", spec.Default), spec.Usage)
		}
	}
}

var (
	preflightClusterFile  string
	preflightClusterNodes string
)

var embeddedPreflightPolicy []byte

// SetEmbeddedPreflightPolicy installs the release policy carried by the bke
// binary. The copy prevents callers from mutating the runtime policy after
// registration.
func SetEmbeddedPreflightPolicy(policy []byte) {
	embeddedPreflightPolicy = append([]byte(nil), policy...)
}

// runPreflightInit resolves init flags via the shared spec, builds the
// AdapterInput, runs the SDK, and renders the report.
func runPreflightInit(cmd *cobra.Command, args []string) error {
	ctx, cancel, err := totalPreflightContext(cmd.Context())
	if err != nil {
		return writeAndRenderPreflightReport(
			singleInputIssueReport(
				api.OperationInit,
				"PREFLIGHT-OPTIONS",
				"timeout",
				err,
				"set --timeout to a duration greater than zero, then rerun preflight",
			),
		)
	}
	defer cancel()
	// Collect raw flag values from Cobra into the shared map.
	raw := collectInitFlags(cmd)
	opts, err := preflightbridge.ResolveInitOptions(raw)
	if err != nil {
		return renderInvalidInitOptions(err)
	}
	markExplicitInitPorts(cmd, &opts)
	preflightbridge.ResolveInitHostIP(&opts)
	if err := preflightbridge.ApplyInitFileOptions(&opts); err != nil {
		return writeAndRenderPreflightReport(
			singleInputIssueReport(
				api.OperationInit,
				"INIT-INPUT-FILE",
				"init.file",
				err,
				"correct the init configuration file, then rerun preflight",
			),
		)
	}
	if err := preflightbridge.ValidateInitInput(opts); err != nil {
		return renderInvalidInitOptions(err)
	}
	in := opts.ToAdapterInput()
	in.Init.Workspace = global.Workspace
	return runCheckWithCollectorFactory(ctx, in, func(config preflightconfig.Config) (api.Collector, error) {
		return preflightbridge.NewInitRuntimeCollectorWithCustom(
			opts,
			preflightbridge.VerifiedPlatforms(config),
			config.Operations.Init,
		)
	})
}

func markExplicitInitPorts(command *cobra.Command, options *preflightbridge.InitOptions) {
	for _, name := range []string{
		"kubernetesPort", "consolePort", "imageRepoPort", "yumRepoPort", "chartRepoPort",
	} {
		if command.Flags().Changed(name) {
			preflightbridge.MarkExplicitInitPort(options, name)
		}
	}
}

func renderInvalidInitOptions(err error) error {
	return writeAndRenderPreflightReport(singleInputIssueReport(
		api.OperationInit,
		"INIT-INPUT-INVALID",
		"init.options",
		err,
		"correct the reported init option, then rerun preflight",
	))
}

func runPreflightClusterCreate(cmd *cobra.Command, args []string) error {
	ctx, cancel, err := totalPreflightContext(cmd.Context())
	if err != nil {
		return writeAndRenderPreflightReport(
			singleInputIssueReport(
				api.OperationClusterCreate,
				"PREFLIGHT-OPTIONS",
				"timeout",
				err,
				"set --timeout to a duration greater than zero, then rerun preflight",
			),
		)
	}
	defer cancel()
	res, issues, err := preflightbridge.ResolveClusterCreate(preflightClusterFile, preflightClusterNodes)
	if err != nil {
		return writeAndRenderPreflightReport(inputIssueReport(api.OperationClusterCreate, issues))
	}
	managementRoutes, defaults, managementErr := preflightManagementRuntimeWithProgress(ctx)
	var setupIssues []api.RunIssue
	if managementErr != nil {
		setupIssues = managementSetupIssues(managementErr)
	}
	if err := preflightbridge.ApplyProviderManagementDefaults(res, defaults); err != nil {
		return writeAndRenderPreflightReport(
			singleInputIssueReport(
				api.OperationClusterCreate,
				"CLUSTER-INPUT-INVALID",
				"provider.defaults",
				err,
				"correct the Provider defaults, then rerun preflight",
			),
		)
	}
	in, err := res.ToAdapterInput()
	if err != nil {
		return writeAndRenderPreflightReport(
			singleInputIssueReport(
				api.OperationClusterCreate,
				"CLUSTER-INPUT-INVALID",
				"cluster",
				err,
				"correct the cluster configuration, then rerun preflight",
			),
		)
	}
	return runCheckWithCollectorFactoryAndIssues(ctx, in, func(config preflightconfig.Config) (api.Collector, error) {
		return preflightbridge.NewClusterRuntimeCollectorWithCustom(
			res.BKENodes,
			preflightbridge.VerifiedPlatforms(config),
			config.Operations.ClusterCreate,
			managementRoutes...)
	}, setupIssues)
}

func preflightManagementRuntime(ctx context.Context) ([]router.Route, map[string]string, error) {
	managementCollector, err := preflightbridge.NewManagementCollectorWithTimeout(
		options.KubeConfig,
		preflightFactTimeout,
	)
	if err != nil {
		return managementRoutes(preflightbridge.NewUnavailableManagementCollector(err.Error())), nil, err
	}
	defaults, err := managementCollector.ProviderDefaults(ctx)
	// Keep the read-only management route even when only the defaults read
	// failed. The SDK can then continue with built-in/static defaults and
	// report the management failure together with node and input findings.
	return managementRoutes(managementCollector), defaults, err
}

func managementRoutes(managementCollector api.Collector) []router.Route {
	return []router.Route{{Match: func(subject api.Subject, key string) bool {
		return subject.ID == "cluster" && isManagementFactKey(key)
	}, Collector: managementCollector}}
}

func isManagementFactKey(key string) bool {
	prefixes := [...]string{"kubernetes.", "crd.", "object.", "auth.", "secret.", "kubeconfig."}
	for _, prefix := range prefixes {
		if strings.HasPrefix(key, prefix) {
			return true
		}
	}
	return false
}

func managementSetupIssues(err error) []api.RunIssue {
	if err == nil {
		return nil
	}
	cause := "管理集群 Provider 默认配置不可用，检查计划使用静态默认值"
	action := "恢复管理集群 API 连接并确认 ConfigMap cluster-system/bke-config 可读后重试"
	if apierrors.IsForbidden(err) || apierrors.IsUnauthorized(err) {
		cause = "无权读取管理集群 Provider 默认配置"
		action = "授予 ConfigMap cluster-system/bke-config 只读权限后重试"
	}
	return []api.RunIssue{{
		Code: "CLUSTER-MGMT-PROVIDER-DEFAULTS", Severity: api.IssueError,
		Stage: api.IssueStageSetup, Effect: api.IssueContinue,
		Cause: cause, Action: action,
	}}
}

// preflightManagementRuntimeWithProgress covers the management-cluster read
// that happens before the SDK can build a cluster plan. It must have the same
// operator-visible timeout and progress behavior as SDK fact collection;
// otherwise a stalled kubeconfig/API request looks like a hung preflight.
func preflightManagementRuntimeWithProgress(parent context.Context) ([]router.Route, map[string]string, error) {
	progress := newPreflightProgressReporter(os.Stderr, preflightFactTimeout)
	progress.Start()
	progress.Report(api.ProgressEvent{Phase: api.ProgressPlan, Current: "读取管理集群 Provider 默认配置"})
	managementCtx := parent
	managementTimeout := preflightFactTimeout
	if managementTimeout <= 0 {
		managementTimeout = defaultPreflightFactTimeout
	}
	if preflightTimeout > 0 && preflightTimeout < managementTimeout {
		managementTimeout = preflightTimeout
	}
	if managementTimeout > 0 {
		var cancel context.CancelFunc
		managementCtx, cancel = context.WithTimeout(parent, managementTimeout)
		defer cancel()
	}
	routes, defaults, err := preflightManagementRuntime(managementCtx)
	if err != nil {
		if errors.Is(managementCtx.Err(), context.DeadlineExceeded) {
			progress.Report(
				api.ProgressEvent{
					Phase: api.ProgressPlan,
					Current: fmt.Sprintf(
						"读取管理集群 Provider 默认配置超时（%s），将使用静态默认值",
						managementTimeout,
					),
				},
			)
			progress.Close()
			return routes, defaults, fmt.Errorf(
				"management-cluster Provider defaults timed out after %s",
				managementTimeout,
			)
		}
		if errors.Is(managementCtx.Err(), context.Canceled) {
			progress.Report(api.ProgressEvent{
				Phase:   api.ProgressPlan,
				Current: "读取管理集群 Provider 默认配置已取消，将使用静态默认值",
			})
			progress.Close()
			return routes, defaults, fmt.Errorf("management-cluster Provider defaults canceled")
		}
	}
	if err != nil {
		progress.Report(api.ProgressEvent{
			Phase:   api.ProgressPlan,
			Current: "管理集群 Provider 默认配置不可用，将使用静态默认值",
		})
	}
	progress.Close()
	return routes, defaults, err
}

func inputIssueReport(operation api.Operation, issues []api.RunIssue) api.Report {
	return api.Report{
		SchemaVersion: api.SchemaVersion,
		Operation:     operation,
		Overall:       api.OverallInputError,
		ExitCode:      api.ExitCode(api.OverallInputError),
		Issues:        issues,
	}
}

func singleInputIssueReport(operation api.Operation, code, field string, cause error, action string) api.Report {
	return inputIssueReport(operation, []api.RunIssue{{
		Code: code, Severity: api.IssueError, Stage: api.IssueStageInput,
		Effect: api.IssueBlockStart, Field: field, Cause: cause.Error(), Action: action,
	}})
}

func totalPreflightContext(parent context.Context) (context.Context, context.CancelFunc, error) {
	if preflightTimeout <= 0 {
		return nil, nil, &commandExitError{
			code: api.ExitInputError,
			err:  fmt.Errorf("--timeout must be greater than zero"),
		}
	}
	ctx, cancel := context.WithTimeout(parent, preflightTimeout)
	return ctx, cancel, nil
}

// runCheck builds the SDK CheckInput, runs Check, renders the report, writes
// to stdout (and optional output file), and exits with the report's exit code.
func runCheck(ctx context.Context, in preflightbridge.AdapterInput) error {
	return runCheckWithCollectorFactory(ctx, in, func(_ preflightconfig.Config) (api.Collector, error) {
		return preflightbridge.NewRuntimeCollector(), nil
	})
}

func runCheckWithCollector(ctx context.Context, in preflightbridge.AdapterInput, collector api.Collector) error {
	return runCheckWithCollectorFactory(
		ctx,
		in,
		func(_ preflightconfig.Config) (api.Collector, error) { return collector, nil },
	)
}

func runCheckWithCollectorFactory(
	ctx context.Context,
	in preflightbridge.AdapterInput,
	makeCollector func(preflightconfig.Config) (api.Collector, error),
) error {
	return runCheckWithCollectorFactoryAndIssues(ctx, in, makeCollector, nil)
}

func runCheckWithCollectorFactoryAndIssues(
	ctx context.Context,
	in preflightbridge.AdapterInput,
	makeCollector func(preflightconfig.Config) (api.Collector, error),
	setupIssues []api.RunIssue,
) error {
	if report := invalidPreflightOptions(in.Operation); report != nil {
		return writeAndRenderPreflightReport(*report)
	}
	applyPreflightWorkspace(&in)
	assets, err := loadPreflightAssets()
	if err != nil {
		return writeAndRenderPreflightReport(releaseAssetIssueReport(in.Operation, err))
	}
	if preflightConfigFile != "" {
		assets, err = loadPreflightRuntimeConfig(preflightConfigFile, assets)
		if err != nil {
			return writeAndRenderPreflightReport(singleInputIssueReport(
				in.Operation,
				"PREFLIGHT-CONFIG-INVALID",
				"preflight-config",
				err,
				"correct --preflight-config and rerun preflight",
			))
		}
	}
	providerResolver := &preflightbridge.ProviderResolverV122{Policy: &assets.config}
	adapter := preflightbridge.NewOpenFuyaoAdapter(providerResolver, assets.config)
	resolved, err := adapter.Resolve(ctx, in)
	if err != nil {
		if rep, ok := resolveContextErrorReport(ctx, in.Operation, setupIssues, err); ok {
			return writeAndRenderPreflightReport(rep)
		}
		return &commandExitError{code: api.ExitInternalError, err: fmt.Errorf("resolve operation plan: %w", err)}
	}

	collector, err := makeCollector(assets.config)
	if err != nil {
		return writeAndRenderPreflightReport(singleInputIssueReport(
			in.Operation,
			"PREFLIGHT-CUSTOM-COLLECTOR",
			"customChecks",
			err,
			"correct customChecks and rerun preflight",
		))
	}
	rep, err := runSDKPreflight(ctx, sdkPreflightRun{
		input: in, resolved: resolved, assets: assets, provider: providerResolver,
		collector: collector,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "internal error: %v\n", err)
		return &commandExitError{code: api.ExitInternalError, err: err}
	}
	mergeSetupIssues(&rep, setupIssues)
	return writeAndRenderPreflightReport(rep)
}

func invalidPreflightOptions(operation api.Operation) *api.Report {
	var field, cause, action string
	switch {
	case preflightOutput != "text" && preflightOutput != "json":
		field, cause = "output", "--output must be text or json"
		action = "set --output to text or json, then rerun preflight"
	case preflightTimeout <= 0:
		field, cause = "timeout", "--timeout must be greater than zero"
		action = "set --timeout to a duration greater than zero, then rerun preflight"
	case preflightFactTimeout <= 0:
		field, cause = "fact-timeout", "--fact-timeout must be greater than zero"
		action = "set --fact-timeout to a duration greater than zero, then rerun preflight"
	case preflightMaxConcurrency < 0 || preflightMaxConcurrency > maximumPreflightConcurrency:
		field, cause = "max-concurrency", "--max-concurrency must be between 0 and 64"
		action = "set --max-concurrency between 0 and 64, then rerun preflight"
	default:
		return nil
	}
	report := singleInputIssueReport(operation, "PREFLIGHT-OPTIONS", field, errors.New(cause), action)
	return &report
}

func applyPreflightWorkspace(in *preflightbridge.AdapterInput) {
	if in.Init != nil && in.Init.Workspace == "" {
		in.Init.Workspace = global.Workspace
	}
	if in.ClusterCreate != nil && in.ClusterCreate.Workspace == "" {
		in.ClusterCreate.Workspace = global.Workspace
	}
}

type sdkPreflightRun struct {
	input     preflightbridge.AdapterInput
	resolved  api.ResolveResult
	assets    preflightAssets
	provider  *preflightbridge.ProviderResolverV122
	collector api.Collector
}

func runSDKPreflight(ctx context.Context, run sdkPreflightRun) (api.Report, error) {
	checker := preflight.New(preflight.Dependencies{
		Collector: run.collector,
	})
	progress := newPreflightProgressReporter(os.Stderr, preflightFactTimeout)
	progress.Start()
	report, err := checker.Check(ctx, preflight.CheckInput{
		Resolved: run.resolved, Bundle: run.assets.bundle,
		Runtime: api.RuntimeOptions{
			OverallTimeout: preflightTimeout,
			FactTimeout:    preflightFactTimeout, MaxConcurrency: preflightMaxConcurrency,
		},
		Progress: progress.Report,
	})
	progress.Close()
	return report, err
}

func mergeSetupIssues(report *api.Report, setupIssues []api.RunIssue) {
	if len(setupIssues) == 0 {
		return
	}
	report.Issues = append(report.Issues, setupIssues...)
	if api.HasErrorIssue(setupIssues) && (report.Overall == api.OverallPass || report.Overall == api.OverallWarn) {
		report.Overall = api.OverallFail
		report.ExitCode = api.ExitCode(report.Overall)
	}
}

func releaseAssetIssueReport(operation api.Operation, err error) api.Report {
	return api.Report{SchemaVersion: api.SchemaVersion, Operation: operation,
		Overall: api.OverallBundleError, ExitCode: api.ExitCode(api.OverallBundleError),
		Issues: []api.RunIssue{{Code: "BUNDLE-INVALID", Severity: api.IssueError,
			Stage: api.IssueStageSetup, Effect: api.IssueBlockStart,
			Field: "embedded-policy", Cause: err.Error(),
			Action: "restore the trusted bke release binary, then rerun preflight"}}}
}

type preflightAssets struct {
	config preflightconfig.Config
	bundle []byte
}

func loadPreflightAssets() (preflightAssets, error) {
	config, err := preflightconfig.Load(embeddedPreflightPolicy)
	if err != nil {
		return preflightAssets{}, fmt.Errorf("parse embedded preflight policy: %w", err)
	}
	raw, err := preflightbridge.BuildBundle(config)
	if err != nil {
		return preflightAssets{}, fmt.Errorf("compile embedded preflight policy: %w", err)
	}
	return preflightAssets{config: config, bundle: raw}, nil
}

func loadPreflightRuntimeConfig(
	path string,
	releaseAssets preflightAssets,
) (preflightAssets, error) {
	config, err := preflightconfig.LoadFile(path)
	if err != nil {
		return preflightAssets{}, err
	}
	if err := preflightconfig.ValidateRuntimeCompleteness(releaseAssets.config, config); err != nil {
		return preflightAssets{}, fmt.Errorf("runtime policy is incomplete: %w", err)
	}
	raw, err := preflightbridge.BuildBundle(config)
	if err != nil {
		return preflightAssets{}, fmt.Errorf("compile external policy: %w", err)
	}
	releaseAssets.config = config
	releaseAssets.bundle = raw
	return releaseAssets, nil
}

func resolveContextErrorReport(
	ctx context.Context,
	operation api.Operation,
	setupIssues []api.RunIssue,
	resolveErr error,
) (api.Report, bool) {
	issues := append([]api.RunIssue(nil), setupIssues...)
	rep := api.Report{SchemaVersion: api.SchemaVersion, Operation: operation}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(resolveErr, context.DeadlineExceeded) {
		rep.Overall, rep.ExitCode = api.OverallTimedOut, api.ExitCode(api.OverallTimedOut)
		rep.Issues = append(issues, api.RunIssue{
			Code:     "PREFLIGHT-TIMEOUT",
			Severity: api.IssueError,
			Stage:    api.IssueStageExecution,
			Effect:   api.IssueInterrupt,
			Cause:    "preflight timed out while resolving the operation plan",
			Action: "restore management-cluster connectivity or increase --timeout, " +
				"then rerun preflight",
		})
		return rep, true
	}
	if errors.Is(ctx.Err(), context.Canceled) || errors.Is(resolveErr, context.Canceled) {
		rep.Overall, rep.ExitCode = api.OverallCanceled, api.ExitCode(api.OverallCanceled)
		rep.Issues = append(issues, api.RunIssue{
			Code:     "PREFLIGHT-CANCELED",
			Severity: api.IssueError,
			Stage:    api.IssueStageExecution,
			Effect:   api.IssueInterrupt,
			Cause:    "preflight was canceled while resolving the operation plan",
			Action:   "rerun preflight when the environment is reachable",
		})
		return rep, true
	}
	return api.Report{}, false
}

func writeAndRenderPreflightReport(rep api.Report) error {
	detailPath := preflightOutputFile
	compactText := preflightOutput == "text"
	if compactText && detailPath == "" {
		var err error
		detailPath, err = defaultPreflightReportPath(rep.Operation, preflightOutput)
		if err != nil {
			return &commandExitError{
				code: api.ExitInternalError,
				err:  fmt.Errorf("create default detailed report path: %w", err),
			}
		}
	}
	// Write the detailed report before rendering the compact console summary.
	if detailPath != "" {
		if err := writeReportAtomic(detailPath, mustRenderBytes(rep, preflightOutput)); err != nil {
			return &commandExitError{code: api.ExitInternalError, err: err}
		}
	}
	// Render text output as a compact actionable summary. The full report was
	// written above; JSON remains full and machine-readable on stdout.
	if compactText {
		if err := render.Summary(os.Stdout, rep, detailPath); err != nil {
			return &commandExitError{code: api.ExitInternalError, err: err}
		}
	} else if preflightOutput == "json" {
		if err := render.JSON(os.Stdout, rep); err != nil {
			return &commandExitError{code: api.ExitInternalError, err: err}
		}
	} else {
		if err := render.Text(os.Stdout, rep); err != nil {
			return &commandExitError{code: api.ExitInternalError, err: err}
		}
	}
	if rep.ExitCode != 0 {
		return &commandExitError{code: rep.ExitCode}
	}
	return nil
}

// preflightProgressReporter writes progress to stderr so text and JSON
// reports on stdout stay machine-readable. It rate-limits updates and emits
// a heartbeat while a collector is waiting on a remote host; this makes a
// slow or unresponsive node visible without flooding the terminal.
type preflightProgressReporter struct {
	w           io.Writer
	factTimeout time.Duration
	mu          sync.Mutex
	last        api.ProgressEvent
	lastPrinted time.Time
	lastPhase   api.ProgressPhase
	started     bool
	done        chan struct{}
	closeOnce   sync.Once
	heartbeatWg sync.WaitGroup
}

func newPreflightProgressReporter(w io.Writer, factTimeout time.Duration) *preflightProgressReporter {
	return &preflightProgressReporter{w: w, factTimeout: factTimeout, done: make(chan struct{})}
}

func (p *preflightProgressReporter) Start() {
	p.mu.Lock()
	p.started = true
	p.last = api.ProgressEvent{Phase: api.ProgressPlan, Current: "解析输入并生成检查计划"}
	p.lastPhase = api.ProgressPlan
	p.printLocked(p.last, false)
	p.mu.Unlock()
	p.heartbeatWg.Add(1)
	go func() {
		defer p.heartbeatWg.Done()
		ticker := time.NewTicker(progressHeartbeatInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				p.mu.Lock()
				if p.started && p.last.Phase != api.ProgressComplete {
					p.printLocked(p.last, true)
				}
				p.mu.Unlock()
			case <-p.done:
				return
			}
		}
	}()
}

func (p *preflightProgressReporter) Close() {
	p.closeOnce.Do(func() { close(p.done) })
	p.heartbeatWg.Wait()
}

func (p *preflightProgressReporter) Report(event api.ProgressEvent) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.last = event
	now := time.Now()
	if !p.started || event.Phase != p.lastPhase || event.Completed == event.Total ||
		now.Sub(p.lastPrinted) >= time.Second {
		p.printLocked(event, false)
	}
	p.started = true
	p.lastPhase = event.Phase
}

func (p *preflightProgressReporter) printLocked(event api.ProgressEvent, heartbeat bool) {
	if p.w == nil {
		return
	}
	message := "前置检查进度："
	current := localizedProgressCurrent(event.Current)
	switch event.Phase {
	case api.ProgressPlan:
		if current != "" {
			message += current
		} else {
			message += fmt.Sprintf("计划已生成，共 %d 项", event.Total)
		}
	case api.ProgressCollecting:
		message += fmt.Sprintf("采集事实 %d/%d", event.Completed, event.Total)
		if current != "" {
			message += "；当前=" + current
		}
		if p.factTimeout > 0 {
			message += fmt.Sprintf("（单项超时 %s）", p.factTimeout)
		}
	case api.ProgressEvaluating:
		percent := progressPercentTotal
		if event.Total > 0 {
			percent = float64(event.Completed) * progressPercentTotal / float64(event.Total)
		}
		message += fmt.Sprintf("计算结果 %d/%d（%.1f%%）", event.Completed, event.Total, percent)
		if current != "" {
			message += "；当前=" + current
		}
		// SKIP is a report-only per-check status. Do not leak it into the
		// progress stream; the detailed report remains the source of truth for
		// blocked applicable checks.
		if event.Status != "" && event.Status != api.StatusSkip {
			message += "；结果=" + localizedProgressStatus(event.Status)
		}
	case api.ProgressComplete:
		message += fmt.Sprintf("完成 %d/%d 项", event.Completed, event.Total)
	case api.ProgressBlocked:
		message += "检查未启动"
		if current != "" {
			message += "；" + current
		}
	case api.ProgressIncomplete:
		message += "检查未完成"
		if current != "" {
			message += "；" + current
		}
	default:
		message += string(event.Phase)
	}
	if heartbeat {
		message += "；仍在运行"
	}
	_, _ = fmt.Fprintln(p.w, message)
	p.lastPrinted = time.Now()
}

func localizedProgressCurrent(current string) string {
	switch current {
	case "collecting facts":
		return "采集检查事实"
	case "evaluating checks":
		return "计算检查结果"
	case "checks complete":
		return "检查完成"
	case "preflight timed out before all checks completed":
		return "前置检查超时，部分检查未完成"
	case "preflight canceled before all checks completed":
		return "前置检查已取消，部分检查未完成"
	default:
		return current
	}
}

func localizedProgressStatus(status api.Status) string {
	switch status {
	case api.StatusPass:
		return "通过"
	case api.StatusFail:
		return "失败"
	case api.StatusWarn:
		return "警告"
	default:
		return string(status)
	}
}

type commandExitError struct {
	code int
	err  error
}

func (e *commandExitError) Error() string {
	if e.err != nil {
		return e.err.Error()
	}
	return fmt.Sprintf("preflight exit %d", e.code)
}

func writeReportAtomic(path string, data []byte) error {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, ".preflight-report-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer cleanupTemporaryReport(name)
	if _, err = f.Write(data); err == nil {
		err = f.Chmod(preflightReportFileMode)
	}
	closeErr := f.Close()
	if err == nil && closeErr != nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, path)
}

func defaultPreflightReportPath(operation api.Operation, format string) (string, error) {
	ext := "txt"
	if format == "json" {
		ext = "json"
	}
	pattern := fmt.Sprintf("bke-preflight-%s-*.%s", strings.ReplaceAll(string(operation), "/", "-"), ext)
	f, err := os.CreateTemp("", pattern)
	if err != nil {
		return "", err
	}
	path := f.Name()
	if err := f.Close(); err != nil {
		if removeErr := os.Remove(path); removeErr != nil {
			return "", fmt.Errorf("close temporary report: %w (cleanup: %v)", err, removeErr)
		}
		return "", fmt.Errorf("close temporary report: %w", err)
	}
	if err := os.Remove(path); err != nil {
		return "", err
	}
	return path, nil
}

func cleanupTemporaryReport(path string) {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		fmt.Fprintf(os.Stderr, "preflight cleanup: remove temporary report: %v\n", err)
	}
}

// collectInitFlags reads the bound Cobra flags into the raw string map.
func collectInitFlags(cmd *cobra.Command) map[string]string {
	raw := map[string]string{}
	for _, spec := range preflightbridge.InitFlagSpecs() {
		flag := cmd.Flags().Lookup(spec.Name)
		if flag == nil {
			continue
		}
		// Cobra bool flags return "true"/"false"; keep as-is.
		raw[spec.Name] = flag.Value.String()
	}
	return raw
}

func mustRenderBytes(rep api.Report, format string) []byte {
	var sb strings.Builder
	if format == "json" {
		_ = render.JSON(&sb, rep)
	} else {
		_ = render.Text(&sb, rep)
	}
	return []byte(sb.String())
}
