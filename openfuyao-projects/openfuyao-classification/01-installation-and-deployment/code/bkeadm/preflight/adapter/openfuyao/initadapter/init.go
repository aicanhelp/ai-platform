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

// Package initadapter turns a resolved BKE init Input into an OperationPlan
// and the set of rule instances the engine evaluates. It is pure: no I/O, no
// DNS, no confirmation, no printing. The host resolves flags first.
package initadapter

import (
	"fmt"
	"net"
	"net/url"
	"strings"

	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/adapterutil"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/normalize"
)

const (
	sourceReferenceParts = 2
	initMinimumCPU       = 2
	initMinimumMemory    = int64(4 << 30)
	initMinimumDisk      = int64(100 << 30)
)

// Resolve builds an OperationPlan from a resolved init input. It projects the
// final ports, paths, runtime, sources and feature flags, then computes the
// PlanHash. Run issues carry any input problems; BLOCK_START means the plan
// cannot be formed (INPUT_ERROR).
func Resolve(in *Input) (api.OperationPlan, []api.RunIssue) {
	var issues []api.RunIssue
	if in == nil {
		return api.OperationPlan{}, []api.RunIssue{{
			Code: "INIT-INPUT-REQUIRED", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
			Cause: "init input is nil", Action: "provide resolved init input, then run preflight again",
		}}
	}
	plan := api.OperationPlan{
		SchemaVersion: api.SchemaVersion,
		Operation:     api.OperationInit,
		Subjects: []api.Subject{{ID: "bootstrap", Role: []string{"bootstrap"}, Address: in.HostIP,
			Metadata: map[string]string{
				"workspace": in.Workspace, "runtimeStorage": in.RuntimeStorage,
				"offlineArtifact": in.ImageFilePath, "imageRepoCAFile": in.ImageRepoCAFile,
			}}},
		Values: buildValues(in),
	}
	// Sources: online image / other repo / other source / other chart.
	if in.OnlineImage != "" {
		plan.Sources = append(plan.Sources, sourceFromRaw("online-image", "image", in.OnlineImage, ""))
	}
	if in.OtherRepo != "" {
		authRef := ""
		if in.ImageRepoAuth {
			authRef = "init-image-repo"
		}
		plan.Sources = append(plan.Sources, sourceFromRaw("other-repo", "image", in.OtherRepo, authRef))
	}
	if in.OtherSource != "" {
		plan.Sources = append(plan.Sources, sourceFromRaw("other-source", "package", in.OtherSource, ""))
	}
	if in.OtherChart != "" {
		plan.Sources = append(plan.Sources, sourceFromRaw("other-chart", "chart", in.OtherChart, ""))
	}
	// Compute PlanHash (sensitive keys are rejected by Canonicalize).
	h, err := normalize.ComputePlanHash(&plan)
	if err != nil {
		issues = append(issues, api.RunIssue{Code: "INIT-PLAN-HASH", Severity: api.IssueError,
			Stage: api.IssueStagePlan, Effect: api.IssueBlockStart,
			Cause: fmt.Sprintf(
				"compute plan hash: %v",
				err,
			), Action: "correct the reported init input, then run preflight again"})
		return plan, issues
	}
	plan.PlanHash = h
	return plan, issues
}

func sourceFromRaw(id, kind, raw, authRef string) api.Source {
	value := strings.TrimSpace(raw)
	scheme := "https"
	objectPath := ""
	if u, err := url.Parse(value); err == nil && u.Host != "" {
		value = u.Host
		if u.Scheme != "" {
			scheme = strings.ToLower(u.Scheme)
		}
		objectPath = u.EscapedPath()
	} else {
		parts := strings.SplitN(value, "/", sourceReferenceParts)
		value = parts[0]
		if len(parts) == sourceReferenceParts && id == "online-image" {
			objectPath = imageManifestPath(parts[1])
		}
	}
	if _, _, err := net.SplitHostPort(value); err == nil {
		return api.Source{ID: id, Kind: kind, Endpoint: value, Scheme: scheme, ObjectPath: objectPath, AuthRef: authRef}
	}
	return api.Source{
		ID:         id,
		Kind:       kind,
		Endpoint:   net.JoinHostPort(strings.Trim(value, "[]"), "443"),
		Scheme:     scheme,
		ObjectPath: objectPath,
		AuthRef:    authRef,
	}
}

func imageManifestPath(reference string) string {
	name, version := reference, "latest"
	if at := strings.LastIndex(reference, "@"); at >= 0 {
		name, version = reference[:at], reference[at+1:]
	} else if colon := strings.LastIndex(reference, ":"); colon > strings.LastIndex(reference, "/") {
		name, version = reference[:colon], reference[colon+1:]
	} else {
		// Keep the default latest tag when the reference has no explicit tag.
	}
	if strings.Trim(name, "/") == "" || strings.TrimSpace(version) == "" {
		return "/v2/"
	}
	return "/v2/" + strings.Trim(name, "/") + "/manifests/" + url.PathEscape(version)
}

// buildValues projects the init input into typed plan values keyed by fact key.
// It detects the install mode (online / offline / hybrid) from the source
// combination and sets feature flags that gate conditional rule instantiation.
func buildValues(in *Input) map[string]api.Value {
	v := make(map[string]api.Value, 20)
	v["platform.os"] = api.Value{Key: "platform.os", Type: api.ValueTypeEnum, Value: "linux"}
	v["resources.cpu.logical"] = api.Value{
		Key: "resources.cpu.logical", Type: api.ValueTypeInteger, Value: initMinimumCPU,
	}
	v["resources.memory.totalBytes"] = api.Value{
		Key:   "resources.memory.totalBytes",
		Type:  api.ValueTypeBytes,
		Value: initMinimumMemory,
	}
	v["resources.systemDisk.totalBytes"] = api.Value{
		Key:   "resources.systemDisk.totalBytes",
		Type:  api.ValueTypeBytes,
		Value: initMinimumDisk,
	}
	// Ports enabled by the plan.
	v["port.kubernetes"] = portVal("port.kubernetes", in.KubernetesPort)
	v["port.imageRepo"] = portVal("port.imageRepo", in.ImageRepoPort)
	v["port.chart"] = portVal("port.chart", in.ChartRepoPort)
	v["port.yum"] = portVal("port.yum", in.YumRepoPort)
	if in.InstallConsole {
		v["port.console"] = portVal("port.console", in.ConsolePort)
	}
	if in.EnableNTP && in.NtpServer == "local" {
		v["port.ntp"] = api.Value{Key: "port.ntp", Type: api.ValueTypeInteger, Value: 123}
	}
	v["feature.installConsole"] = api.Value{
		Key:   "feature.installConsole",
		Type:  api.ValueTypeBool,
		Value: in.InstallConsole,
	}
	v["feature.localNTP"] = api.Value{
		Key:   "feature.localNTP",
		Type:  api.ValueTypeBool,
		Value: in.EnableNTP && in.NtpServer == "local",
	}
	v["runtime"] = api.Value{Key: "runtime", Type: api.ValueTypeEnum, Value: in.Runtime}

	// --- Install mode detection (online / offline / hybrid) ---
	// Online: OnlineImage set, no local ImageFilePath.
	// Offline: ImageFilePath set (local tar.gz), requires OtherRepo.
	// Hybrid: OtherRepo/OtherSource/OtherChart set alongside OnlineImage.
	hasOnline := in.OnlineImage != ""
	hasOffline := in.ImageFilePath != ""
	hasCustomRepo := in.OtherRepo != "" || in.OtherSource != "" || in.OtherChart != ""

	if hasOnline && !hasOffline {
		v["init.mode"] = api.Value{Key: "init.mode", Type: api.ValueTypeEnum, Value: "online"}
	}
	if hasOffline {
		v["init.mode"] = api.Value{Key: "init.mode", Type: api.ValueTypeEnum, Value: "offline"}
		// Project the offline artifact path so §6.8 rules instantiate.
		v["offline.artifact"] = api.Value{Key: "offline.artifact", Type: api.ValueTypeString, Value: in.ImageFilePath}
	}
	if hasOnline && hasCustomRepo {
		v["init.mode"] = api.Value{Key: "init.mode", Type: api.ValueTypeEnum, Value: "hybrid"}
	}

	// Custom repo / CA feature flags (gate §6.9 rules).
	if hasCustomRepo {
		v["feature.customRepo"] = api.Value{Key: "feature.customRepo", Type: api.ValueTypeBool, Value: true}
	}
	if in.ImageRepoCAFile != "" {
		v["feature.repoCA"] = api.Value{Key: "feature.repoCA", Type: api.ValueTypeBool, Value: true}
	}

	return v
}

func portVal(key string, port int) api.Value {
	return api.Value{Key: key, Type: api.ValueTypeInteger, Value: port}
}

// Instances builds the rule instances for the init plan using the historical
// compatibility defaults. New hosts should call InstancesWithPorts and supply
// values compiled from PreflightConfig.
func Instances(plan api.OperationPlan) []InitInstance {
	return InstancesWithPorts(plan, map[string]int{"ntp": 123})
}

// InstancesWithPorts builds checks using caller-supplied policy ports. The
// operation plan values always remain authoritative; these defaults are only
// used when a policy port is absent.
func InstancesWithPorts(plan api.OperationPlan, defaults map[string]int) []InitInstance {
	return buildInstancesWithPorts(plan, defaults)
}

// buildInstancesWithPorts projects the supported init checks into
// InitInstance entries bound to the bootstrap subject. Conditional ports
// (console, NTP) and conditional sources/artifacts/CA only instantiate when
// enabled by the plan. The list is deliberately small: a check belongs here
// only when the init path really consumes the corresponding tool, path, port
// or artifact.
//
// Rule grouping (conditional source/artifact rules expand at runtime):
//  1. Input/identity/platform (§6.2): EXEC, PLAT-OS/DISTRO/KERNEL/ARCH
//  2. Resources (§6.3): CPU, MEMORY, SYSTEM-DISK
//  3. Tools & paths (§6.4): TAR, SYSTEMCTL, PATH-CREATE/MOUNT
//  4. Ports (§6.5): K8S, REGISTRY, CHART, HTTP + conditional CONSOLE, NTP
//  5. Sources (§6.6): one reachability result per source
//  6. Offline artifacts (§6.7): EXISTS, READ, FORMAT
//  7. Custom repo & CA (§6.8): REPO-URL/CA-*
//  8. Colocate (§6.9): COLOCATE
func buildInstancesWithPorts(plan api.OperationPlan, defaults map[string]int) []InitInstance {
	out := initIdentityInstances()
	out = append(out, initResourceInstances()...)
	out = append(out, initToolPathInstances()...)
	out = append(out, initPortInstances(plan, defaults)...)
	out = append(out, initSourceInstances(plan)...)
	out = append(out, initOfflineInstances(plan)...)
	out = append(out, initRepositoryInstances(plan)...)
	return append(out, initColocateInstance())
}

func initIdentityInstances() []InitInstance {
	return []InitInstance{
		Mk("INIT-EXEC-001", "exec.identity.euid", api.OpEq, 0),
		Mk("INIT-PLAT-OS", "platform.os", api.OpEq, "linux"),
		Mk("INIT-PLAT-DISTRO", "platform.distribution", api.OpIn, []any{
			[]any{"openeuler", "20.03"}, []any{"openeuler", "22.03"}, []any{"openeuler", "24.03"},
			[]any{"ubuntu", "22.04"}, []any{"uos", "20 1070e"},
		}),
		Mk("INIT-PLAT-KERNEL", "platform.kernel.matched", api.OpEq, true),
		Mk("INIT-PLAT-ARCH", "platform.architecture", api.OpIn, []any{"amd64", "arm64"}),
	}
}

func initResourceInstances() []InitInstance {
	return []InitInstance{
		Mk("INIT-RES-CPU", "resources.cpu.logical", api.OpGte, initMinimumCPU),
		Mk("INIT-RES-MEMORY", "resources.memory.totalBytes", api.OpGte, initMinimumMemory),
		Mk("INIT-RES-SYSTEM-DISK", "resources.systemDisk.totalBytes", api.OpGte, initMinimumDisk),
	}
}

func initToolPathInstances() []InitInstance {
	return []InitInstance{
		Mk("INIT-TOOL-TAR", "tool.tar", api.OpEq, true),
		Mk("INIT-TOOL-SYSTEMCTL", "tool.systemctl", api.OpEq, true),
		Mk("INIT-PATH-CREATE", "path.workspace.creatable", api.OpEq, true),
		Mk("INIT-PATH-MOUNT", "path.workspace.mountReadOnly", api.OpEq, false),
	}
}

func initPortInstances(plan api.OperationPlan, defaults map[string]int) []InitInstance {
	const bootstrap = "bootstrap"
	var out []InitInstance
	for _, p := range []struct {
		id       string
		valueKey string
	}{
		{"INIT-PORT-K8S", "port.kubernetes"},
		{"INIT-PORT-REGISTRY", "port.imageRepo"},
		{"INIT-PORT-CHART", "port.chart"},
		{"INIT-PORT-HTTP", "port.yum"},
	} {
		out = append(out, MkPort(p.id, planPort(plan, p.valueKey), bootstrap))
	}
	// Conditional: console port (installConsole=true).
	console := MkPort("INIT-PORT-CONSOLE", planPort(plan, "port.console"), bootstrap)
	console.Applicable = valueIsTrue(plan, "feature.installConsole")
	out = append(out, console)
	ntpPort := planPort(plan, "port.ntp")
	if ntpPort == 0 {
		ntpPort = defaults["ntp"]
	}
	ntp := Mk(
		"INIT-PORT-NTP",
		fmt.Sprintf("port.%d.udp.free", ntpPort),
		api.OpFree,
		true,
	)
	_, ntp.Applicable = plan.Values["port.ntp"]
	out = append(out, ntp)
	return out
}

func initSourceInstances(plan api.OperationPlan) []InitInstance {
	out := make([]InitInstance, 0, len(plan.Sources))
	for _, src := range plan.Sources {
		out = append(
			out,
			mkSrc(
				src,
				"endpoint.tcp."+src.Scheme+"://"+src.Endpoint,
				api.OpEq,
				true,
			),
		)
	}
	return out
}

func initOfflineInstances(plan api.OperationPlan) []InitInstance {
	var out []InitInstance
	if v, ok := plan.Values["offline.artifact"]; ok {
		path, _ := v.Value.(string)
		if path == "" {
			path = "offline-artifact"
		}
		out = append(out, mkOffline(path, "EXISTS", "artifact."+path+".exists", api.OpEq, true))
		out = append(out, mkOffline(path, "READ", "artifact."+path+".readable", api.OpEq, true))
		out = append(
			out,
			mkOffline(path, "FORMAT", "artifact."+path+".format", api.OpIn, []any{"tar.gz", "tgz", "tar"}),
		)
	} else {
		for _, instance := range []InitInstance{
			mkOffline("not-applicable", "EXISTS", "artifact.not-applicable.exists", api.OpEq, true),
			mkOffline("not-applicable", "READ", "artifact.not-applicable.readable", api.OpEq, true),
			mkOffline("not-applicable", "FORMAT", "artifact.not-applicable.format", api.OpIn, []any{"tar.gz", "tgz", "tar"}),
		} {
			instance.Applicable = false
			out = append(out, instance)
		}
	}
	return out
}

func initRepositoryInstances(plan api.OperationPlan) []InitInstance {
	var out []InitInstance
	customRepo := Mk(
		"INIT-REPO-URL",
		"plan.customRepo.valid",
		api.OpEq,
		true,
	)
	customRepo.Applicable = valueIsTrue(plan, "feature.customRepo")
	out = append(out, customRepo)
	for _, instance := range []InitInstance{
		Mk("INIT-REPO-CA-EXISTS", "path.customRepoCA.exists", api.OpEq, true),
		Mk("INIT-REPO-CA-READ", "path.customRepoCA.readable", api.OpEq, true),
		Mk(
			"INIT-REPO-CA-PARSE", "certificate.customRepoCA.parseable",
			api.OpEq, true,
		),
	} {
		instance.Applicable = valueIsTrue(plan, "feature.repoCA")
		out = append(out, instance)
	}
	return out
}

func initColocateInstance() InitInstance {
	colocate := MkWarn(
		"INIT-COLOCATE",
		"plan.colocate",
		api.OpRelation,
		false,
	)
	colocate.Applicable = false
	return colocate
}

// InitInstance is a local alias to avoid importing engine (which would create
// a cycle: adapter → engine → api; we want adapter → api only). The host
// bridges these to engine.Instance.
type InitInstance struct {
	Rule    api.RuleTemplate
	Subject string
	FactKey string
	// InstanceID distinguishes repeated instances of one stable Bundle rule
	// (for example one source rule per configured repository).
	InstanceID string
	Applicable bool
}

// Mk builds a blocking init check instance.
func Mk(
	id, fact string,
	op api.Op,
	expected any,
) InitInstance {
	name, fix := initRuleMetadata(id)
	return InitInstance{
		Rule: api.RuleTemplate{ID: id, Name: name, Operation: api.OperationInit, Fact: fact, Op: op,
			Expected: expected, Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: fix,
			DependsOn: initRuleDependencies(id)},
		Subject: "bootstrap", FactKey: fact, Applicable: true,
	}
}

// MkPort builds a blocking init port-availability check instance.
func MkPort(id string, port int, subject string) InitInstance {
	key := fmt.Sprintf("port.%d.free", port)
	name, _ := initRuleMetadata(id)
	if name == "" {
		name = "Required host port"
	}
	return InitInstance{
		Rule: api.RuleTemplate{ID: id, Name: name, Operation: api.OperationInit, Fact: key, Op: api.OpFree,
			Expected: true, Level: api.SeverityError, Unknown: api.UnknownBlock,
			Fix: fmt.Sprintf(
				"make %s (%d) available on the host; check the corresponding bke init flag",
				name, port,
			)},
		Subject: subject, FactKey: key, Applicable: true,
	}
}

// MkWarn builds a non-blocking init check instance.
func MkWarn(
	id, fact string,
	op api.Op,
	expected any,
) InitInstance {
	name, fix := initRuleMetadata(id)
	return InitInstance{
		Rule: api.RuleTemplate{ID: id, Name: name, Operation: api.OperationInit, Fact: fact, Op: op,
			Expected: expected, Level: api.SeverityWarning, Unknown: api.UnknownWarn,
			Fix: fix},
		Subject: "bootstrap", FactKey: fact, Applicable: true,
	}
}

// mkSrc builds the single public source-access rule. The collector performs
// only the endpoint connectivity probe used by this rule.
func mkSrc(
	source api.Source,
	factKey string,
	op api.Op,
	expected any,
) InitInstance {
	return InitInstance{
		Rule: api.RuleTemplate{
			ID: "INIT-SRC-REACHABLE",
			Name: adapterutil.SourceCheckName(
				sourceName(source.ID),
				source.Endpoint,
			),
			Operation: api.OperationInit,
			Fact:      factKey, Op: op, Expected: expected, Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: sourceCheckFix(source.ID, source.Endpoint),
		},
		Subject: "bootstrap", FactKey: factKey,
		InstanceID: "INIT-SRC-REACHABLE@" + source.ID, Applicable: true,
	}
}

func sourceName(sourceID string) string {
	switch sourceID {
	case "online-image":
		return "Online image source"
	case "other-repo":
		return "Image repository"
	case "other-source":
		return "Package repository"
	case "other-chart":
		return "Chart repository"
	default:
		return adapterutil.HumanizeSourceID(sourceID)
	}
}

func sourceCheckFix(sourceID, endpoint string) string {
	return adapterutil.SourceUnreachableFix(sourceName(sourceID), endpoint)
}

// CatalogRules returns every stable init RuleTemplate that can be bound at
// runtime. Dynamic source and artifact values are deliberately represented by
// harmless placeholders; the Adapter supplies the real fact key per instance.
func CatalogRules() []api.RuleTemplate {
	plan := api.OperationPlan{
		Operation: api.OperationInit,
		Subjects:  []api.Subject{{ID: "bootstrap"}},
		Values: map[string]api.Value{
			"feature.installConsole": {Key: "feature.installConsole", Type: api.ValueTypeBool, Value: true},
			"port.kubernetes":        {Key: "port.kubernetes", Type: api.ValueTypeInteger, Value: 0},
			"port.imageRepo":         {Key: "port.imageRepo", Type: api.ValueTypeInteger, Value: 0},
			"port.chart":             {Key: "port.chart", Type: api.ValueTypeInteger, Value: 0},
			"port.yum":               {Key: "port.yum", Type: api.ValueTypeInteger, Value: 0},
			"port.console":           {Key: "port.console", Type: api.ValueTypeInteger, Value: 0},
			"port.ntp":               {Key: "port.ntp", Type: api.ValueTypeInteger, Value: 0},
			"offline.artifact": {
				Key:   "offline.artifact",
				Type:  api.ValueTypeString,
				Value: "/release/offline.tar.gz",
			},
			"feature.customRepo": {Key: "feature.customRepo", Type: api.ValueTypeBool, Value: true},
			"feature.repoCA":     {Key: "feature.repoCA", Type: api.ValueTypeBool, Value: true},
		},
		Sources: []api.Source{{ID: "release-source", Endpoint: "registry.example.invalid:443", Scheme: "https"}},
	}
	instances := InstancesWithPorts(plan, nil)
	return adapterutil.UniqueRules(instances, func(instance InitInstance) api.RuleTemplate {
		return instance.Rule
	})
}

// mkOffline builds a per-offline-artifact ERROR rule.
func mkOffline(path, kind, factKey string, op api.Op, expected any) InitInstance {
	name, fix := offlineRuleMetadata(path, kind)
	return InitInstance{
		Rule: api.RuleTemplate{
			ID: "INIT-OFFLINE-" + kind, Name: name, Operation: api.OperationInit,
			Fact: factKey, Op: op, Expected: expected, Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: fix,
			DependsOn: initRuleDependencies("INIT-OFFLINE-" + kind),
		},
		Subject: "bootstrap", FactKey: factKey, Applicable: true,
	}
}

func initRuleDependencies(id string) []string {
	switch id {
	case "INIT-OFFLINE-READ":
		return []string{"INIT-OFFLINE-EXISTS"}
	case "INIT-OFFLINE-FORMAT":
		return []string{"INIT-OFFLINE-READ"}
	case "INIT-REPO-CA-READ":
		return []string{"INIT-REPO-CA-EXISTS"}
	case "INIT-REPO-CA-PARSE":
		return []string{"INIT-REPO-CA-READ"}
	default:
		return nil
	}
}

func offlineRuleMetadata(path, kind string) (string, string) {
	label := map[string]string{
		"EXISTS": "file exists",
		"READ":   "file is readable",
		"FORMAT": "archive format",
	}[kind]
	if label == "" {
		label = strings.ToLower(strings.ReplaceAll(kind, "-", " "))
	}
	name := fmt.Sprintf("Offline artifact %s (%s)", label, path)
	var fix string
	switch kind {
	case "EXISTS":
		fix = fmt.Sprintf("make sure the offline artifact file %s exists", path)
	case "READ":
		fix = fmt.Sprintf("make the offline artifact file %s readable by root", path)
	case "FORMAT":
		fix = fmt.Sprintf("replace %s with a tar, tar.gz or tgz archive", path)
	default:
		fix = fmt.Sprintf("check the offline artifact file %s", path)
	}
	return name, fix
}

func valueIsTrue(plan api.OperationPlan, key string) bool {
	value, ok := plan.Values[key]
	return ok && value.Value == true
}

func planPort(plan api.OperationPlan, key string) int {
	return adapterutil.PlanInt(plan, key)
}

type ruleMetadata struct {
	name string
	fix  string
}

func initRuleMetadata(id string) (string, string) {
	if item, ok := initHostRuleMetadata(id); ok {
		return item.name, item.fix
	}
	if item, ok := initPortRepositoryMetadata(id); ok {
		return item.name, item.fix
	}
	return "", "review the corresponding bke init option and rerun preflight"
}

func initHostRuleMetadata(id string) (ruleMetadata, bool) {
	metadata := map[string]ruleMetadata{
		"INIT-EXEC-001": {
			"Root user required",
			"run bke as root (UID 0); for example, start it from a root shell or use sudo",
		},
		"INIT-PLAT-OS": {"Host operating system", "use a supported Linux host"},
		"INIT-PLAT-DISTRO": {
			"Host Linux distribution and version",
			"use a distribution and version listed in the preflight policy",
		},
		"INIT-PLAT-KERNEL": {
			"Host kernel compatibility",
			"boot a kernel version covered by the release compatibility matrix",
		},
		"INIT-PLAT-ARCH": {
			"Host architecture",
			"run bke on an amd64 or arm64 host matching the installation artifacts",
		},
		"INIT-RES-CPU":         {"Host CPU capacity", "use a host with at least 2 logical CPUs"},
		"INIT-RES-MEMORY":      {"Host memory capacity", "use a host with at least 4 GiB memory"},
		"INIT-RES-SYSTEM-DISK": {"Host system disk capacity", "use a host with at least 100 GiB system disk capacity"},
		"INIT-TOOL-TAR":        {"tar executable on PATH", "install tar and ensure it is executable on PATH"},
		"INIT-TOOL-SYSTEMCTL": {
			"systemctl executable on PATH",
			"install systemd tools and ensure systemctl is executable on PATH",
		},
		"INIT-PATH-CREATE": {
			"Workspace path is creatable",
			"make the workspace path available and creatable by the root installation process",
		},
		"INIT-PATH-MOUNT": {
			"Workspace mount is writable",
			"remount the workspace filesystem read-write or choose a writable workspace",
		},
	}
	item, ok := metadata[id]
	return item, ok
}

func initPortRepositoryMetadata(id string) (ruleMetadata, bool) {
	metadata := map[string]ruleMetadata{
		"INIT-PORT-K8S": {
			"Kubernetes API host port (--kubernetesPort)",
			"change --kubernetesPort or stop the process using the configured port",
		},
		"INIT-PORT-REGISTRY": {
			"Image repository host port (--imageRepoPort)",
			"change --imageRepoPort or stop the process using the configured port",
		},
		"INIT-PORT-CHART": {
			"Chart repository host port (--chartRepoPort)",
			"change --chartRepoPort or stop the process using the configured port",
		},
		"INIT-PORT-HTTP": {
			"Package repository host port (--yumRepoPort)",
			"change --yumRepoPort or stop the process using the configured port",
		},
		"INIT-PORT-CONSOLE": {
			"BKE Console host port (--consolePort)",
			"change --consolePort or disable Console with --installConsole=false",
		},
		"INIT-PORT-NTP": {
			"Local NTP UDP port (--enableNTP/--ntpServer)",
			"stop the process using UDP port 123 or disable local NTP with --enableNTP=false",
		},
		"INIT-REPO-URL": {
			"Custom image repository configuration (--otherRepo)",
			"correct --otherRepo and verify its endpoint is reachable",
		},
		"INIT-REPO-CA-EXISTS": {
			"Custom repository CA file (--imageRepoCAFile) exists",
			"set --imageRepoCAFile to an existing CA certificate file",
		},
		"INIT-REPO-CA-READ": {
			"Custom repository CA file (--imageRepoCAFile) is readable",
			"make the file in --imageRepoCAFile readable by root",
		},
		"INIT-REPO-CA-PARSE": {
			"Custom repository CA file (--imageRepoCAFile) is valid",
			"replace --imageRepoCAFile with a valid PEM CA certificate",
		},
		"INIT-COLOCATE": {
			"Bootstrap node colocation risk",
			"review whether the bootstrap host is also a management or workload node",
		},
	}
	item, ok := metadata[id]
	return item, ok
}
