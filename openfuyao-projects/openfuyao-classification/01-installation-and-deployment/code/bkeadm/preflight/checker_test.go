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

package preflight

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/releasefacts"
)

type testAdapterInput struct {
	Operation api.Operation
	Init      *initadapter.Input
}

func resolveTestInput(in testAdapterInput) api.ResolveResult {
	if in.Operation != api.OperationInit {
		return api.ResolveResult{Plan: api.OperationPlan{Operation: in.Operation}, Issues: []api.RunIssue{{Code: "UNSUPPORTED", Severity: api.IssueError, Stage: api.IssueStageInput, Effect: api.IssueBlockStart, Cause: "unsupported operation"}}}
	}
	plan, issues := initadapter.Resolve(in.Init)
	result := api.ResolveResult{Plan: plan, Issues: issues}
	subjectsByID := make(map[string]api.Subject, len(plan.Subjects))
	for _, subject := range plan.Subjects {
		subjectsByID[subject.ID] = subject
	}
	for _, instance := range initadapter.Instances(plan) {
		result.Bindings = append(result.Bindings, api.RuleBinding{
			TemplateID: instance.Rule.ID, InstanceID: instance.InstanceID,
			Subject: subjectsByID[instance.Subject], Fact: instance.FactKey,
			Applicable: instance.Applicable,
		})
	}
	return result
}

func TestBindingNameOverridesGenericBundleName(t *testing.T) {
	bundle := &api.PreflightBundle{Rules: []api.RuleTemplate{{
		ID: "CHECK", Operation: api.OperationInit, Name: "Generic source check", Fact: "fact", Op: api.OpEq,
		Expected: true, Level: api.SeverityError, Unknown: api.UnknownBlock,
	}}}
	resolved := api.ResolveResult{
		Plan:     api.OperationPlan{Operation: api.OperationInit},
		Bindings: []api.RuleBinding{{TemplateID: "CHECK", InstanceID: "CHECK@bootstrap", Name: "Image repository is accessible (images.example.com:5443)", Fix: "unable to access Image repository (images.example.com:5443); check network connectivity, the endpoint address, proxy/NO_PROXY, and firewall settings, then run preflight again", Subject: api.Subject{ID: "bootstrap"}, Fact: "fact", Applicable: true}},
	}
	_, instances, err := bindRules(bundle, resolved)
	require.NoError(t, err)
	require.Len(t, instances, 1)
	assert.Equal(t, "Image repository is accessible (images.example.com:5443)", instances[0].Rule.Name)
	assert.Equal(t, "unable to access Image repository (images.example.com:5443); check network connectivity, the endpoint address, proxy/NO_PROXY, and firewall settings, then run preflight again", instances[0].Rule.Fix)
}

func TestCheckRejectsStalePlanHashAsInternalContractError(t *testing.T) {
	resolved := resolveTestInput(testAdapterInput{Operation: api.OperationInit, Init: &initadapter.Input{HostIP: "192.0.2.10", Domain: "x"}})
	resolved.Plan.PlanHash = "sha256:deadbeef"
	rep, err := New(Dependencies{}).Check(context.Background(), CheckInput{Resolved: resolved})
	require.NoError(t, err)
	assert.Equal(t, api.OverallInternalError, rep.Overall)
	require.Len(t, rep.Issues, 1)
	assert.Equal(t, "ADAPTER-CONTRACT", rep.Issues[0].Code)
}

func TestCheckRejectsMissingPlanHashAsInternalContractError(t *testing.T) {
	resolved := resolveTestInput(testAdapterInput{Operation: api.OperationInit, Init: &initadapter.Input{HostIP: "192.0.2.10", Domain: "x"}})
	resolved.Plan.PlanHash = ""
	rep, err := New(Dependencies{}).Check(context.Background(), CheckInput{Resolved: resolved})
	require.NoError(t, err)
	assert.Equal(t, api.OverallInternalError, rep.Overall)
	require.Len(t, rep.Issues, 1)
	assert.Contains(t, rep.Issues[0].Cause, "plan hash is required")
}

func TestBindingUsesSubjectFromHashedPlan(t *testing.T) {
	resolved := resolveTestInput(testAdapterInput{Operation: api.OperationInit, Init: &initadapter.Input{HostIP: "192.0.2.10", Domain: "x"}})
	require.NotEmpty(t, resolved.Bindings)
	resolved.Bindings[0].Subject.Address = "192.0.2.99"
	bundle := &api.PreflightBundle{Rules: []api.RuleTemplate{{ID: resolved.Bindings[0].TemplateID, Operation: api.OperationInit, Fact: resolved.Bindings[0].Fact, Op: api.OpEq, Expected: true, Level: api.SeverityError, Unknown: api.UnknownBlock}}}
	_, instances, err := bindRules(bundle, api.ResolveResult{Plan: resolved.Plan, Bindings: resolved.Bindings[:1]})
	require.NoError(t, err)
	require.Len(t, instances, 1)
	assert.Equal(t, "192.0.2.10", instances[0].Subject.Address)
}

func TestCheckRejectsBindingSubjectDifferentFromHashedPlan(t *testing.T) {
	resolved := resolveTestInput(testAdapterInput{Operation: api.OperationInit, Init: &initadapter.Input{HostIP: "192.0.2.10", Domain: "x"}})
	require.NotEmpty(t, resolved.Bindings)
	resolved.Bindings[0].Subject.Address = "192.0.2.99"
	rep, err := New(Dependencies{}).Check(context.Background(), CheckInput{Resolved: resolved})
	require.NoError(t, err)
	assert.Equal(t, api.OverallInternalError, rep.Overall)
	require.Len(t, rep.Issues, 1)
	assert.Equal(t, "ADAPTER-CONTRACT", rep.Issues[0].Code)
	assert.Contains(t, rep.Issues[0].Cause, "differs from the hashed plan subject")
}

func TestInapplicableBindingIsNotInstantiated(t *testing.T) {
	resolved := api.ResolveResult{Plan: api.OperationPlan{Operation: api.OperationInit}, Bindings: []api.RuleBinding{{TemplateID: "MISSING", Applicable: false}}}
	rules, instances, err := bindRules(&api.PreflightBundle{}, resolved)
	require.NoError(t, err)
	assert.Empty(t, rules)
	assert.Empty(t, instances)
}

// fakeCollector returns facts from a map keyed by subject\x00key.
type fakeCollector struct {
	facts map[string]api.Fact
}

func (f *fakeCollector) ID() string { return "fake" }
func (f *fakeCollector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	var out []api.Fact
	for _, s := range req.Subjects {
		for _, k := range req.Keys {
			if fact, ok := f.facts[s.ID+"\x00"+k]; ok {
				out = append(out, fact)
			} else {
				out = append(out, api.Fact{Key: k, Subject: s.ID, Available: false,
					Error: &api.FactError{Code: api.FactErrNotFound, Message: "no fact for " + k}})
			}
		}
	}
	if len(req.Subjects) == 0 {
		for _, k := range req.Keys {
			if fact, ok := f.facts["\x00"+k]; ok {
				out = append(out, fact)
			}
		}
	}
	return out, nil
}

type recordingCollector struct {
	request api.FactRequest
	facts   []api.Fact
}

func (*recordingCollector) ID() string { return "recording" }

func (c *recordingCollector) Collect(_ context.Context, request api.FactRequest) ([]api.Fact, error) {
	c.request = request
	return c.facts, nil
}

func TestFactOverlayCollectorUsesAdapterFactsBeforeRuntimeCollection(t *testing.T) {
	subject := api.Subject{ID: "node-1", Address: "192.0.2.10"}
	next := &recordingCollector{facts: []api.Fact{{
		Key: "runtime.fact", Subject: subject.ID, Available: true, Value: "runtime",
	}}}
	collector := &factOverlayCollector{
		facts: []api.Fact{{Key: "plan.fact", Subject: subject.ID, Available: true, Value: "plan"}},
		next:  next,
	}
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"plan.fact", "runtime.fact"}, Subjects: []api.Subject{subject}, Timeout: time.Second,
	})
	require.NoError(t, err)
	assert.Equal(t, "adapter-facts+recording", collector.ID())
	assert.Len(t, facts, 2)
	assert.Equal(t, []string{"runtime.fact"}, next.request.Keys)
	assert.Equal(t, []api.Subject{subject}, next.request.Subjects)
	assert.Equal(t, time.Second, next.request.Timeout)
}

func TestFactOverlayCollectorSkipsRuntimeWhenRequestIsSatisfied(t *testing.T) {
	next := &recordingCollector{}
	collector := &factOverlayCollector{
		facts: []api.Fact{{Key: "plan.fact", Available: true, Value: true}}, next: next,
	}
	facts, err := collector.Collect(context.Background(), api.FactRequest{Keys: []string{"plan.fact"}})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.Empty(t, next.request.Keys)
}

func TestNoopCollectorReturnsUnavailableFactForEachRequest(t *testing.T) {
	collector := noopCollector{}
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"one", "two"}, Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "noop", collector.ID())
	require.Len(t, facts, 2)
	for _, fact := range facts {
		assert.Equal(t, api.FactErrUnsupported, fact.Error.Code)
		assert.Equal(t, "node-1", fact.Subject)
	}
}

func TestValidFactErrorCodeAllowlist(t *testing.T) {
	valid := []string{
		api.FactErrTimeout, api.FactErrPermissionDenied, api.FactErrNotFound,
		api.FactErrUnreachable, api.FactErrParseError, api.FactErrUnsupported,
	}
	for _, code := range valid {
		assert.True(t, validFactErrorCode(code), code)
	}
	assert.False(t, validFactErrorCode("UNKNOWN"))
}

// allPassFacts returns a fact map where every init rule's fact key is present
// and satisfying. This lets the Checker E2E tests focus on the Overall
// conclusion rather than enumerating every fact.
func allPassFacts() map[string]api.Fact {
	return map[string]api.Fact{
		"bootstrap\x00exec.identity.euid":              {Key: "exec.identity.euid", Subject: "bootstrap", Value: 0, Available: true},
		"bootstrap\x00platform.os":                     {Key: "platform.os", Subject: "bootstrap", Value: "linux", Available: true},
		"bootstrap\x00platform.distribution":           {Key: "platform.distribution", Subject: "bootstrap", Value: []any{"openeuler", "22.03"}, Available: true},
		"bootstrap\x00platform.kernel.matched":         {Key: "platform.kernel.matched", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00platform.architecture":           {Key: "platform.architecture", Subject: "bootstrap", Value: "amd64", Available: true},
		"bootstrap\x00resources.cpu.logical":           {Key: "resources.cpu.logical", Subject: "bootstrap", Value: 8, Available: true},
		"bootstrap\x00resources.memory.totalBytes":     {Key: "resources.memory.totalBytes", Subject: "bootstrap", Value: int64(8 * 1024 * 1024 * 1024), Available: true},
		"bootstrap\x00resources.systemDisk.totalBytes": {Key: "resources.systemDisk.totalBytes", Subject: "bootstrap", Value: int64(200 * 1024 * 1024 * 1024), Available: true},
		"bootstrap\x00tool.tar":                        {Key: "tool.tar", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00tool.systemctl":                  {Key: "tool.systemctl", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00path.workspace.creatable":        {Key: "path.workspace.creatable", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00path.workspace.mountReadOnly":    {Key: "path.workspace.mountReadOnly", Subject: "bootstrap", Value: false, Available: true},
		"bootstrap\x00port.36443.free":                 {Key: "port.36443.free", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00port.40443.free":                 {Key: "port.40443.free", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00port.38080.free":                 {Key: "port.38080.free", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00port.40080.free":                 {Key: "port.40080.free", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00port.30010.free":                 {Key: "port.30010.free", Subject: "bootstrap", Value: true, Available: true},
		"bootstrap\x00port.123.udp.free":               {Key: "port.123.udp.free", Subject: "bootstrap", Value: true, Available: true},
		// Source facts (online-image source).
		"bootstrap\x00endpoint.tcp.cr.openfuyao.cn/openfuyao/bke-online:latest": {Key: "endpoint.tcp.cr.openfuyao.cn/openfuyao/bke-online:latest", Subject: "bootstrap", Value: true, Available: true},
	}
}

func mustBundleYAML(t *testing.T, op api.Operation) []byte {
	b := &api.PreflightBundle{
		Operations: []api.Operation{op},
		Rules:      releasefacts.StaticRules(),
	}
	if op == api.OperationInit {
		filtered := b.Rules[:0]
		for _, rule := range b.Rules {
			if rule.Operation == op {
				filtered = append(filtered, rule)
			}
		}
		b.Rules = filtered
	}
	out, err := yaml.Marshal(b)
	require.NoError(t, err)
	return out
}

// TestCheckInitAllPass freezes the full SDK pipeline happy path: init input →
// plan → bundle → engine → Report with Overall PASS. Uses a fake collector
// that returns matching facts.
func TestCheckInitAllPass(t *testing.T) {
	collector := &fakeCollector{facts: allPassFacts()}
	checker := New(Dependencies{Collector: collector})
	in := CheckInput{
		Resolved: resolveTestInput(testAdapterInput{
			Operation: api.OperationInit,
			Init: &initadapter.Input{
				HostIP: "192.0.2.10", Domain: "x",
				KubernetesPort: 36443, ConsolePort: 30010, ImageRepoPort: 40443,
				ChartRepoPort: 38080, YumRepoPort: 40080,
				InstallConsole: true, EnableNTP: false,
			},
		}),
		Bundle: mustBundleYAML(t, api.OperationInit),
	}
	rep, err := checker.Check(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, api.OverallPass, rep.Overall, "issues: %#v", rep.Issues)
	assert.Equal(t, api.SchemaVersion, rep.SchemaVersion)
	assert.Equal(t, api.OperationInit, rep.Operation)
	assert.NotEmpty(t, rep.Results)
	assert.Equal(t, len(rep.Results), rep.Summary.Total)
	// All applicable results PASS; non-applicable conditions are omitted.
	for _, r := range rep.Results {
		assert.NotEmpty(t, r.InstanceID)
		assert.NotEmpty(t, r.RuleID)
		assert.NotEmpty(t, r.Name)
		if r.Status == api.StatusSkip {
			continue
		}
		assert.Equal(t, api.StatusPass, r.Status, "rule %s should PASS", r.RuleID)
	}
	assert.Equal(t, api.ExitSuccess, rep.ExitCode)
}

// TestCheckInitPortFail freezes a FAIL path: an occupied port yields FAIL and
// Overall FAIL (exit 3).
func TestCheckInitPortFail(t *testing.T) {
	collector := &fakeCollector{facts: allPassFacts()}
	collector.facts["bootstrap\x00port.36443.free"] = api.Fact{Key: "port.36443.free", Subject: "bootstrap", Value: false, Available: true}
	collector.facts["bootstrap\x00port.30010.free"] = api.Fact{Key: "port.30010.free", Subject: "bootstrap", Value: true, Available: true}
	checker := New(Dependencies{Collector: collector})
	in := CheckInput{
		Resolved: resolveTestInput(testAdapterInput{
			Operation: api.OperationInit,
			Init: &initadapter.Input{HostIP: "192.0.2.10", Domain: "x",
				KubernetesPort: 36443, ConsolePort: 30010, ImageRepoPort: 40443,
				ChartRepoPort: 38080, YumRepoPort: 40080, InstallConsole: false, EnableNTP: false},
		}),
		Bundle: mustBundleYAML(t, api.OperationInit),
	}
	rep, err := checker.Check(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, api.OverallFail, rep.Overall)
	assert.Equal(t, api.ExitFail, rep.ExitCode)
	// The 36443 port rule is FAIL.
	var portFail bool
	for _, r := range rep.Results {
		if r.RuleID == "INIT-PORT-K8S" && r.Status == api.StatusFail {
			portFail = true
		}
	}
	assert.True(t, portFail, "INIT-PORT-K8S should be FAIL")
}

// TestCheckInitDoesNotEmitColocationWithoutTopology keeps init output quiet
// because init input has no management/workload node topology.
func TestCheckInitDoesNotEmitColocationWithoutTopology(t *testing.T) {
	collector := &fakeCollector{facts: allPassFacts()}
	checker := New(Dependencies{Collector: collector})
	in := CheckInput{
		Resolved: resolveTestInput(testAdapterInput{
			Operation: api.OperationInit,
			Init: &initadapter.Input{HostIP: "1.2.3.4", Domain: "x", KubernetesPort: 36443,
				ConsolePort: 30010, ImageRepoPort: 40443, ChartRepoPort: 38080, YumRepoPort: 40080,
				InstallConsole: false, EnableNTP: false},
		}),
		Bundle: mustBundleYAML(t, api.OperationInit),
	}
	rep, err := checker.Check(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, api.OverallPass, rep.Overall)
	assert.Equal(t, api.ExitSuccess, rep.ExitCode)
	for _, result := range rep.Results {
		assert.NotEqual(t, "INIT-COLOCATE", result.RuleID)
	}
}

// TestCheckBundleError freezes that a bad bundle yields BUNDLE_ERROR (exit 4).
func TestCheckBundleError(t *testing.T) {
	checker := New(Dependencies{})
	in := CheckInput{
		Resolved: resolveTestInput(testAdapterInput{
			Operation: api.OperationInit,
			Init:      &initadapter.Input{HostIP: "1.2.3.4", Domain: "x"},
		}),
		Bundle: []byte("::: not a bundle"),
	}
	rep, _ := checker.Check(context.Background(), in)
	assert.Equal(t, api.OverallBundleError, rep.Overall)
	assert.Equal(t, api.ExitInputError, rep.ExitCode)
}

// TestCheckInputError freezes that a nil init input yields INPUT_ERROR (exit 4).
func TestCheckInputError(t *testing.T) {
	checker := New(Dependencies{})
	in := CheckInput{
		Resolved: resolveTestInput(testAdapterInput{Operation: api.OperationInit, Init: nil}),
		Bundle:   mustBundleYAML(t, api.OperationInit),
	}
	rep, _ := checker.Check(context.Background(), in)
	assert.Equal(t, api.OverallInputError, rep.Overall)
	assert.Equal(t, api.ExitInputError, rep.ExitCode)
}

// TestCheckUnsupportedOperation freezes that an unknown operation is INPUT_ERROR.
func TestCheckUnsupportedOperation(t *testing.T) {
	checker := New(Dependencies{})
	in := CheckInput{
		Resolved: resolveTestInput(testAdapterInput{Operation: api.Operation("build")}),
		Bundle:   mustBundleYAML(t, api.OperationInit),
	}
	rep, _ := checker.Check(context.Background(), in)
	assert.Equal(t, api.OverallInputError, rep.Overall)
}

// TestCheckConditionalPortsSkip freezes that disabled console/NTP are not
// instantiated and are not collected.
func TestCheckConditionalPortsSkip(t *testing.T) {
	collector := &fakeCollector{facts: allPassFacts()}
	checker := New(Dependencies{Collector: collector})
	in := CheckInput{
		Resolved: resolveTestInput(testAdapterInput{
			Operation: api.OperationInit,
			Init: &initadapter.Input{HostIP: "1.2.3.4", Domain: "x", KubernetesPort: 36443,
				ConsolePort: 30010, ImageRepoPort: 40443, ChartRepoPort: 38080, YumRepoPort: 40080,
				InstallConsole: false, EnableNTP: false},
		}),
		Bundle: mustBundleYAML(t, api.OperationInit),
	}
	rep, _ := checker.Check(context.Background(), in)
	var consoleSeen, ntpSeen bool
	for _, r := range rep.Results {
		if r.RuleID == "INIT-PORT-CONSOLE" {
			consoleSeen = true
		}
		if r.RuleID == "INIT-PORT-NTP" {
			ntpSeen = true
		}
	}
	assert.False(t, consoleSeen)
	assert.False(t, ntpSeen)
}

// TestCheckNoCollectorIsUnknown freezes that a missing collector yields
// unavailable facts → unknown policy → never PASS.
func TestCheckNoCollectorIsUnknown(t *testing.T) {
	checker := New(Dependencies{}) // no collector
	in := CheckInput{
		Resolved: resolveTestInput(testAdapterInput{
			Operation: api.OperationInit,
			Init: &initadapter.Input{HostIP: "1.2.3.4", Domain: "x", KubernetesPort: 36443,
				ConsolePort: 30010, ImageRepoPort: 40443, ChartRepoPort: 38080, YumRepoPort: 40080,
				InstallConsole: false, EnableNTP: false},
		}),
		Bundle: mustBundleYAML(t, api.OperationInit),
	}
	rep, _ := checker.Check(context.Background(), in)
	// No PASS possible without facts; BLOCK unknown → FAIL.
	assert.NotEqual(t, api.OverallPass, rep.Overall)
}
