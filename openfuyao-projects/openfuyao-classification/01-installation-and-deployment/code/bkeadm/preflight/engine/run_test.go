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

package engine

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

// fakeCollector is a stub FactCollector for orchestration tests. It returns a
// fixed fact map keyed by subject+key, and an optional collector-level error.
type fakeCollector struct {
	facts map[string]api.Fact // keyed subject\x00key
	err   error
}

type recordingCollector struct {
	facts map[string]api.Fact
	mu    sync.Mutex
	calls []string
}

type concurrencyCollector struct {
	active  atomic.Int32
	peak    atomic.Int32
	release chan struct{}
}

func (c *concurrencyCollector) Collect(_ context.Context, req api.FactRequest) ([]api.Fact, error) {
	active := c.active.Add(1)
	for {
		peak := c.peak.Load()
		if active <= peak || c.peak.CompareAndSwap(peak, active) {
			break
		}
	}
	<-c.release
	c.active.Add(-1)
	return []api.Fact{{Key: req.Keys[0], Subject: req.Subjects[0].ID, Available: true, Value: true}}, nil
}

type panicCollector struct{}

func (panicCollector) Collect(context.Context, api.FactRequest) ([]api.Fact, error) { panic("boom") }

type timeoutCollector struct{}

func (timeoutCollector) Collect(ctx context.Context, _ api.FactRequest) ([]api.Fact, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (f *fakeCollector) ID() string { return "fake" }

func (f *fakeCollector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	if f.err != nil {
		return nil, f.err
	}
	var out []api.Fact
	for _, s := range req.Subjects {
		for _, k := range req.Keys {
			if fact, ok := f.facts[s.ID+"\x00"+k]; ok {
				out = append(out, fact)
			}
		}
	}
	return out, nil
}

func (c *recordingCollector) Collect(_ context.Context, req api.FactRequest) ([]api.Fact, error) {
	c.mu.Lock()
	c.calls = append(c.calls, req.Keys[0])
	c.mu.Unlock()
	var out []api.Fact
	for _, subject := range req.Subjects {
		for _, key := range req.Keys {
			if fact, ok := c.facts[subject.ID+"\x00"+key]; ok {
				out = append(out, fact)
			}
		}
	}
	return out, nil
}

// inst builds an Instance with the first fact key = rule.Fact.
func inst(rule api.RuleTemplate, subject string) Instance {
	return Instance{Rule: rule, Subject: api.Subject{ID: subject}, Applicable: true, FactKeys: []string{rule.Fact}}
}

func ruleOK(id string) api.RuleTemplate {
	return api.RuleTemplate{ID: id, Operation: api.OperationInit, Fact: "v", Op: api.OpEq,
		Expected: true, Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: "f"}
}

// TestRunAllPass freezes the happy path: all facts present and matching yields
// all PASS and Overall PASS.
func TestRunAllPass(t *testing.T) {
	rules := []api.RuleTemplate{ruleOK("A"), ruleOK("B")}
	instances := []Instance{inst(rules[0], "local"), inst(rules[1], "local")}
	col := &fakeCollector{facts: map[string]api.Fact{
		"local\x00v": {Key: "v", Subject: "local", Value: true, Available: true},
	}}
	results, overall := Run(context.Background(), rules, instances, col, RunOptions{})
	require.Len(t, results, 2)
	for _, r := range results {
		assert.Equal(t, api.StatusPass, r.Status)
		// A passing result must not suggest remediation, even when the rule
		// template carries a default fix.
		assert.Empty(t, r.Remediation)
		assert.Empty(t, r.Message)
	}
	assert.Equal(t, api.OverallPass, overall)
}

// TestRunFailDominates freezes that one FAIL makes Overall FAIL, while an
// independent rule still evaluates (not canceled).
func TestRunFailDominates(t *testing.T) {
	rules := []api.RuleTemplate{ruleOK("A"), ruleOK("B")}
	instances := []Instance{inst(rules[0], "local"), inst(rules[1], "local")}
	// A's fact is false (FAIL); B's is true (PASS).
	col := &fakeCollector{facts: map[string]api.Fact{
		"local\x00v": {Key: "v", Subject: "local", Value: false, Available: true},
	}}
	results, overall := Run(context.Background(), rules, instances, col, RunOptions{})
	require.Len(t, results, 2)
	// Both rules use the same fact key; both FAIL.
	assert.Equal(t, api.StatusFail, results[0].Status)
	assert.Equal(t, api.OverallFail, overall)
}

// TestRunCycleIsBundleError freezes that a dependency cycle marks every
// instance SKIP and the Overall is BUNDLE_ERROR.
func TestRunCycleIsBundleError(t *testing.T) {
	a := ruleOK("A")
	a.DependsOn = []string{"B"}
	b := ruleOK("B")
	b.DependsOn = []string{"A"}
	rules := []api.RuleTemplate{a, b}
	instances := []Instance{inst(a, "local"), inst(b, "local")}
	col := &fakeCollector{}
	results, overall := Run(context.Background(), rules, instances, col, RunOptions{})
	assert.Equal(t, api.OverallBundleError, overall)
	for _, r := range results {
		assert.Equal(t, api.StatusSkip, r.Status)
		assert.Equal(t, api.CauseBundleError, r.CauseCode)
	}
}

// TestRunDependencyFailedSkips freezes the root-cause contract: when a rule's
// dependency FAILs, the dependent is SKIP referencing the root, and does not
// add a second FAIL.
func TestRunDependencyFailedSkips(t *testing.T) {
	a := ruleOK("A") // will FAIL (fact false)
	b := ruleOK("B")
	b.DependsOn = []string{"A"}
	rules := []api.RuleTemplate{a, b}
	instances := []Instance{inst(a, "local"), inst(b, "local")}
	// Give A a distinct fact key so B looks up a different (missing) key.
	a.Fact = "a"
	b.Fact = "b"
	instances[0].FactKeys = []string{"a"}
	instances[1].FactKeys = []string{"b"}
	col := &fakeCollector{facts: map[string]api.Fact{
		"local\x00a": {Key: "a", Subject: "local", Value: false, Available: true}, // A FAILs
	}}
	results, overall := Run(context.Background(), rules, instances, col, RunOptions{})

	// Find A and B results.
	var resA, resB api.Result
	for _, r := range results {
		if r.RuleID == "A" {
			resA = r
		}
		if r.RuleID == "B" {
			resB = r
		}
	}
	assert.Equal(t, api.StatusFail, resA.Status)
	assert.Equal(t, api.StatusSkip, resB.Status)
	assert.Equal(t, api.CauseDependencyFailed, resB.CauseCode)
	assert.Equal(t, "A", resB.DependsOn)
	assert.Equal(t, api.OverallFail, overall)
}

func TestRunDependencyFailedDoesNotCollectDependentFacts(t *testing.T) {
	a := ruleOK("A")
	a.Fact = "a"
	b := ruleOK("B")
	b.Fact = "b"
	b.DependsOn = []string{"A"}
	collector := &recordingCollector{facts: map[string]api.Fact{
		"local\x00a": {Key: "a", Subject: "local", Value: false, Available: true},
	}}
	results, overall := Run(context.Background(), []api.RuleTemplate{a, b}, []Instance{
		{Rule: a, Subject: api.Subject{ID: "local"}, Applicable: true, FactKeys: []string{"a"}},
		{Rule: b, Subject: api.Subject{ID: "local"}, Applicable: true, FactKeys: []string{"b"}},
	}, collector, RunOptions{})
	assert.Equal(t, api.OverallFail, overall)
	assert.Len(t, collector.calls, 1)
	assert.Equal(t, "a", collector.calls[0])
	for _, result := range results {
		if result.RuleID == "B" {
			assert.Equal(t, api.StatusSkip, result.Status)
			assert.Equal(t, api.CauseDependencyFailed, result.CauseCode)
		}
	}
}

// TestRunCollectorErrorIsInternal freezes the collector contract: expected
// acquisition failures are Facts, while a returned Go error is a framework
// failure and cannot produce a readiness conclusion.
func TestRunCollectorErrorIsInternal(t *testing.T) {
	r := ruleOK("A")
	r.Unknown = api.UnknownIgnore // -> SKIP when unavailable
	instances := []Instance{inst(r, "local")}
	col := &fakeCollector{err: errors.New("connection refused")}
	results, overall := Run(context.Background(), []api.RuleTemplate{r}, instances, col, RunOptions{})
	require.Len(t, results, 1)
	assert.Equal(t, api.StatusSkip, results[0].Status)
	assert.Equal(t, api.OverallInternalError, overall)
}

func TestRunEmptyCollectorResultIsInternal(t *testing.T) {
	r := ruleOK("A")
	results, overall := Run(context.Background(), []api.RuleTemplate{r}, []Instance{inst(r, "local")}, &fakeCollector{}, RunOptions{})
	require.Len(t, results, 1)
	assert.Equal(t, api.OverallInternalError, overall)
	assert.Contains(t, results[0].Message, "collector contract error")
}

func TestRunFactTypeMismatchIsInternalAndPreservesDisplayTypes(t *testing.T) {
	r := api.RuleTemplate{ID: "BYTES", Operation: api.OperationInit, Fact: "resources.memory.totalBytes",
		Op: api.OpGte, Expected: int64(1024), Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: "f"}
	col := &fakeCollector{facts: map[string]api.Fact{
		"local\x00resources.memory.totalBytes": {Key: r.Fact, Subject: "local", Type: api.ValueTypeInteger, Value: int64(1024), Available: true},
	}}
	results, overall := Run(context.Background(), []api.RuleTemplate{r}, []Instance{inst(r, "local")}, col, RunOptions{})
	require.Len(t, results, 1)
	assert.Equal(t, "FACT_TYPE_MISMATCH", results[0].CauseCode)
	assert.Equal(t, api.OverallInternalError, overall)
	assert.Equal(t, api.ValueTypeBytes, results[0].Expected.Type)
	assert.Equal(t, api.ValueTypeInteger, results[0].Actual.Type)
}

func TestRunRecoversCollectorPanic(t *testing.T) {
	r := ruleOK("A")
	results, overall := Run(context.Background(), []api.RuleTemplate{r}, []Instance{inst(r, "local")}, panicCollector{}, RunOptions{})
	require.Len(t, results, 1)
	assert.Equal(t, api.OverallInternalError, overall)
}

func TestRunAppliesPerFactTimeout(t *testing.T) {
	r := ruleOK("A")
	results, overall := Run(context.Background(), []api.RuleTemplate{r}, []Instance{inst(r, "local")}, timeoutCollector{}, RunOptions{FactTimeout: 5 * time.Millisecond})
	require.Len(t, results, 1)
	assert.Equal(t, api.StatusFail, results[0].Status)
	assert.Equal(t, api.FactErrTimeout, results[0].CauseCode)
	assert.Empty(t, results[0].Remediation, "an unavailable fact must not carry a rule-specific fix")
	assert.Equal(t, api.OverallFail, overall)
}

func TestRunReportsDeterministicProgressCounts(t *testing.T) {
	r := ruleOK("A")
	notApplicable := Instance{Rule: ruleOK("B"), Subject: api.Subject{ID: "local"}, Applicable: false}
	var events []api.ProgressEvent
	col := &fakeCollector{facts: map[string]api.Fact{
		"local\x00v": {Key: "v", Subject: "local", Value: true, Available: true},
	}}
	results, _ := Run(context.Background(), []api.RuleTemplate{r, notApplicable.Rule}, []Instance{inst(r, "local"), notApplicable}, col, RunOptions{
		Progress: func(event api.ProgressEvent) { events = append(events, event) },
	})
	assert.Len(t, results, 1, "non-applicable instances must not produce public results")
	assert.Equal(t, "A", results[0].RuleID)
	require.NotEmpty(t, events)
	var collecting, evaluating bool
	for _, event := range events {
		if event.Phase == api.ProgressCollecting {
			collecting = true
			assert.Equal(t, 1, event.Total, "non-applicable instances are excluded")
		}
		if event.Phase == api.ProgressEvaluating {
			evaluating = true
			assert.Equal(t, 1, event.Total)
			assert.LessOrEqual(t, event.Completed, event.Total)
		}
	}
	assert.True(t, collecting)
	assert.True(t, evaluating)
	last := events[len(events)-1]
	assert.Equal(t, api.ProgressEvaluating, last.Phase)
	assert.Equal(t, 1, last.Completed)
}

func TestRunHonorsMaxConcurrency(t *testing.T) {
	collector := &concurrencyCollector{release: make(chan struct{})}
	var rules []api.RuleTemplate
	var instances []Instance
	for i := 0; i < 6; i++ {
		r := ruleOK(string(rune('A' + i)))
		r.Fact = r.ID
		rules = append(rules, r)
		instances = append(instances, inst(r, "local"))
	}
	done := make(chan struct{})
	go func() {
		Run(context.Background(), rules, instances, collector, RunOptions{MaxConcurrency: 2})
		close(done)
	}()
	require.Eventually(t, func() bool { return collector.peak.Load() == 2 }, time.Second, time.Millisecond)
	close(collector.release)
	<-done
	assert.LessOrEqual(t, collector.peak.Load(), int32(2))
}

// TestRunCanceledContext freezes that a canceled context before collection
// yields SKIP with CANCELED cause and Overall is not PASS.
func TestRunCanceledContext(t *testing.T) {
	r := ruleOK("A")
	instances := []Instance{inst(r, "local")}
	col := &fakeCollector{facts: map[string]api.Fact{
		"local\x00v": {Key: "v", Subject: "local", Value: true, Available: true},
	}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before Run
	results, overall := Run(ctx, []api.RuleTemplate{r}, instances, col, RunOptions{})
	require.Len(t, results, 1)
	assert.Equal(t, api.StatusSkip, results[0].Status)
	assert.Equal(t, api.CauseCanceledBeforeStart, results[0].CauseCode)
	assert.NotEqual(t, api.OverallPass, overall)
}

// TestRunTimeoutEvaluatesUnavailableFacts ensures a timeout during collection
// does not turn every remaining applicable rule into CANCELED_BEFORE_START.
func TestRunTimeoutEvaluatesUnavailableFacts(t *testing.T) {
	r := ruleOK("A")
	// Leave enough time for the collector goroutine to start under the race
	// detector. A very short deadline can expire before Run begins, which is a
	// different contract (CANCELED_BEFORE_START) from the one exercised here.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	results, overall := Run(ctx, []api.RuleTemplate{r}, []Instance{inst(r, "local")}, timeoutCollector{}, RunOptions{FactTimeout: 2 * time.Second})
	require.Len(t, results, 1)
	assert.Equal(t, api.StatusFail, results[0].Status)
	assert.Equal(t, api.FactErrTimeout, results[0].CauseCode)
	assert.Equal(t, api.OverallTimedOut, overall)
}

type wrongIdentityCollector struct{}

func (wrongIdentityCollector) Collect(context.Context, api.FactRequest) ([]api.Fact, error) {
	return []api.Fact{{Key: "wrong", Subject: "other", Available: true, Value: true}}, nil
}

type duplicateFactCollector struct{}

func (duplicateFactCollector) Collect(_ context.Context, req api.FactRequest) ([]api.Fact, error) {
	fact := api.Fact{Key: req.Keys[0], Subject: req.Subjects[0].ID, Available: true, Value: true}
	return []api.Fact{fact, fact}, nil
}

func TestRunRejectsDuplicateFactsForOneRequest(t *testing.T) {
	r := ruleOK("A")
	_, overall := Run(context.Background(), []api.RuleTemplate{r}, []Instance{inst(r, "local")}, duplicateFactCollector{}, RunOptions{})
	assert.Equal(t, api.OverallInternalError, overall)
}

type declaredTypeMismatchCollector struct{ valueType api.ValueType }

func (c declaredTypeMismatchCollector) Collect(_ context.Context, req api.FactRequest) ([]api.Fact, error) {
	return []api.Fact{{Key: req.Keys[0], Subject: req.Subjects[0].ID, Type: c.valueType, Available: true, Value: "true"}}, nil
}

func TestRunRejectsFactValueThatContradictsDeclaredType(t *testing.T) {
	r := ruleOK("A")
	for _, valueType := range []api.ValueType{api.ValueTypeBool, api.ValueType("bogus")} {
		_, overall := Run(context.Background(), []api.RuleTemplate{r}, []Instance{inst(r, "local")}, declaredTypeMismatchCollector{valueType: valueType}, RunOptions{})
		assert.Equal(t, api.OverallInternalError, overall, valueType)
	}
}

func TestRunRejectsCollectorFactForWrongRequest(t *testing.T) {
	r := ruleOK("A")
	results, overall := Run(context.Background(), []api.RuleTemplate{r}, []Instance{inst(r, "local")}, wrongIdentityCollector{}, RunOptions{})
	require.Len(t, results, 1)
	assert.Equal(t, api.OverallInternalError, overall)
}

// TestRunResultsSorted freezes that results are sorted by ruleID, subject,
// instanceID for stable output regardless of input order.
func TestRunResultsSorted(t *testing.T) {
	rules := []api.RuleTemplate{ruleOK("B"), ruleOK("A")}
	instances := []Instance{inst(rules[0], "local"), inst(rules[1], "local")}
	col := &fakeCollector{facts: map[string]api.Fact{
		"local\x00v": {Key: "v", Subject: "local", Value: true, Available: true},
	}}
	results, _ := Run(context.Background(), rules, instances, col, RunOptions{})
	// A before B.
	assert.Equal(t, "A", results[0].RuleID)
	assert.Equal(t, "B", results[1].RuleID)
}
