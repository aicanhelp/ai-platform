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
	"fmt"
	"sort"
	"sync"
	"time"

	"gopkg.openfuyao.cn/preflight/api"
)

// FactCollector is the narrow interface the engine uses to collect facts for
// rule instances. The host injects a concrete collector; the engine never
// touches files, SSH or Kubernetes directly.
type FactCollector interface {
	// Collect returns facts for the requested keys, scoped to subjects. A
	// single failure MUST be returned as a Fact with Available=false, not as
	// a Go error, so independent rules are not canceled.
	Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error)
}

// RunOptions controls execution; it does not change rules, levels, thresholds
// or unknown policies.
type RunOptions struct {
	Timeout        time.Duration
	FactTimeout    time.Duration
	MaxConcurrency int
	Progress       api.ProgressFunc
}

// Instance is one rule bound to a subject, ready to evaluate.
type Instance struct {
	Rule    api.RuleTemplate
	Subject api.Subject
	// ID is the stable adapter-provided identity. If empty, the engine derives
	// ruleID@subjectID.
	ID string
	// Applicable=false is omitted from public results without collection.
	Applicable bool
	// ExtraParams carries instance-specific parameters (e.g. port number for a
	// port rule); they feed the FactRequest keys.
	FactKeys []string
}

// Run evaluates a set of rule instances against collected facts and returns a
// slice of Results plus the Overall for a completed run.
//
// Run is for a COMPLETED run only. Timeout and cancellation are signaled by
// the caller via the context. If cancellation happens before Run starts,
// applicable instances become SKIP (cause CANCELED_BEFORE_START). If it
// happens during fact collection, unavailable facts are evaluated through the
// rule policy instead of being mass-labeled SKIP. Non-applicable instances
// are omitted. The Overall in that case is computed from the partial results
// and the caller MUST NOT treat it as a PASS conclusion.
func Run(
	ctx context.Context,
	rules []api.RuleTemplate,
	instances []Instance,
	collector FactCollector,
	opts RunOptions,
) ([]api.Result, api.Overall) {
	// Preserve the distinction between a caller that canceled before work
	// began and a run that timed out while collecting facts. The latter must
	// still evaluate unavailable facts through UnknownPolicy; otherwise every
	// remaining rule is turned into a noisy CANCELED_BEFORE_START SKIP.
	canceledBeforeRun := false
	if ctxErr := ctx.Err(); ctxErr != nil {
		canceledBeforeRun = true
	}
	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}
	order, err := TopoSort(rules)
	if err != nil {
		return dagErrorResults(instances), api.OverallBundleError
	}
	instances = orderInstances(instances, order)
	emitProgress := synchronizedProgress(opts.Progress)

	facts, collectionErr := collectInstanceFactsWithDependencies(
		ctx,
		collector,
		dependencyCollectionInput{rules: rules, order: order, instances: instances},
		collectionOptions{limit: opts.MaxConcurrency, factTimeout: opts.FactTimeout, progress: emitProgress},
	)
	results := evaluateInstances(ctx, instances, facts, canceledBeforeRun, emitProgress)
	sortResults(results)
	return results, completedOverall(ctx.Err(), results, collectionErr)
}

func synchronizedProgress(progress api.ProgressFunc) api.ProgressFunc {
	if progress == nil {
		return func(event api.ProgressEvent) {
			_ = event
		}
	}
	var mu sync.Mutex
	return func(event api.ProgressEvent) {
		mu.Lock()
		defer mu.Unlock()
		progress(event)
	}
}

func dagErrorResults(instances []Instance) []api.Result {
	results := make([]api.Result, 0, len(instances))
	for _, instance := range instances {
		if !instance.Applicable {
			continue
		}
		results = append(results, api.Result{
			InstanceID: instanceID(instance), RuleID: instance.Rule.ID,
			Name: instance.Rule.Name, Subject: instance.Subject.ID,
			Status: api.StatusSkip, CauseCode: api.CauseBundleError, DependsOn: "DAG_CYCLE",
		})
	}
	return results
}

type evaluationRun struct {
	ctx               context.Context
	canceledBeforeRun bool
	instances         []Instance
	facts             map[string]*api.Fact
	progress          api.ProgressFunc
	states            map[string]ruleState
	results           []api.Result
	completed         int
	total             int
}

func evaluateInstances(
	ctx context.Context,
	instances []Instance,
	facts []api.Fact,
	canceledBeforeRun bool,
	progress api.ProgressFunc,
) []api.Result {
	run := evaluationRun{
		ctx: ctx, canceledBeforeRun: canceledBeforeRun, instances: instances,
		facts: indexFacts(facts), progress: progress,
		states: make(map[string]ruleState), results: make([]api.Result, 0, len(instances)),
		total: applicableInstanceCount(instances),
	}
	return run.execute()
}

func (r *evaluationRun) execute() []api.Result {
	r.progress(api.ProgressEvent{Phase: api.ProgressEvaluating, Total: r.total, Current: "evaluating checks"})
	for _, instance := range r.instances {
		if !instance.Applicable {
			continue
		}
		result := r.evaluate(instance)
		r.results = append(r.results, result)
		r.record(instance, result)
	}
	return r.results
}

func (r *evaluationRun) evaluate(instance Instance) api.Result {
	if blocked, root := dependencyFailed(instance.Rule, instance.Subject.ID, r.states); blocked {
		return baseResult(instance, api.StatusSkip, api.CauseDependencyFailed, root)
	}
	if r.canceledBeforeRun && r.ctx.Err() != nil {
		return baseResult(instance, api.StatusSkip, api.CauseCanceledBeforeStart, "")
	}
	fact := findFact(r.facts, instance)
	if fact == nil {
		fact = missingFact(instance)
	}
	return evaluatedResult(instance, *fact)
}

func (r *evaluationRun) record(instance Instance, result api.Result) {
	r.states[stateKey(instance.Rule.ID, instance.Subject.ID)] = ruleState{
		status: result.Status,
		cause:  resultCause(result),
	}
	r.completed++
	r.progress(api.ProgressEvent{
		Phase: api.ProgressEvaluating, Completed: r.completed, Total: r.total,
		Current: instance.Rule.ID + "@" + instance.Subject.ID, Status: result.Status,
	})
}

func baseResult(instance Instance, status api.Status, cause, dependency string) api.Result {
	return api.Result{
		InstanceID: instanceID(instance), RuleID: instance.Rule.ID,
		Name: instance.Rule.Name, Subject: instance.Subject.ID,
		Status: status, CauseCode: cause, DependsOn: dependency,
	}
}

func missingFact(instance Instance) *api.Fact {
	return &api.Fact{
		Key: firstKey(instance), Subject: instance.Subject.ID, Available: false,
		Error: &api.FactError{Code: api.FactErrParseError, Message: "collector contract error: no fact collected"},
	}
}

func evaluatedResult(instance Instance, fact api.Fact) api.Result {
	evaluated := Evaluate(instance.Rule, fact)
	status := NormalizeStatus(instance.Rule, evaluated)
	result := api.Result{
		InstanceID: instanceID(instance), RuleID: instance.Rule.ID,
		Name: instance.Rule.Name, Subject: instance.Subject.ID, Status: status,
		Expected:  api.DisplayValue{Type: instance.Rule.ValueType(), Value: evaluated.Expected},
		Actual:    api.DisplayValue{Type: factValueType(instance.Rule, fact), Value: evaluated.Actual},
		CauseCode: evaluated.CauseCode, Evidence: fact.Evidence,
	}
	if status != api.StatusPass {
		result.Remediation = instance.Rule.Fix
	}
	if instance.Rule.ExpectedText != "" {
		result.Expected = api.DisplayValue{Type: api.ValueTypeString, Value: instance.Rule.ExpectedText}
	}
	if fact.Display != nil {
		result.Actual = *fact.Display
	}
	if fact.Error != nil {
		result.Remediation = ""
		result.Message = fact.Error.Message
	}
	if result.CauseCode == "" && status != api.StatusPass {
		result.CauseCode = string(status)
	}
	return result
}

func resultCause(result api.Result) string {
	if result.DependsOn != "" {
		return result.DependsOn
	}
	return result.CauseCode
}

func applicableInstanceCount(instances []Instance) int {
	count := 0
	for _, instance := range instances {
		if instance.Applicable {
			count++
		}
	}
	return count
}

func completedOverall(ctxErr error, results []api.Result, collectionErr error) api.Overall {
	if errors.Is(ctxErr, context.DeadlineExceeded) {
		return api.OverallTimedOut
	}
	if ctxErr != nil {
		return api.OverallCanceled
	}
	if collectionErr != nil || hasInternalEvaluationError(results) {
		return api.OverallInternalError
	}
	return api.AggregateOverall(statusesOf(results))
}

func hasInternalEvaluationError(results []api.Result) bool {
	for _, result := range results {
		switch result.CauseCode {
		case "FACT_TYPE_MISMATCH", "CEL_COMPILE_ERROR", "CEL_EVALUATION_ERROR", "UNSUPPORTED_OPERATOR":
			return true
		default:
			continue
		}
	}
	return false
}

func factValueType(rule api.RuleTemplate, fact api.Fact) api.ValueType {
	if fact.Type.Valid() {
		return fact.Type
	}
	typeOfValue := api.InferValueType(fact.Value)
	if expected := rule.ValueType(); expected == api.ValueTypeBytes || expected == api.ValueTypeDuration ||
		expected == api.ValueTypeEnum ||
		expected == api.ValueTypeAddress ||
		expected == api.ValueTypeRelation {
		return expected
	}
	return typeOfValue
}

const defaultFactTimeout = 30 * time.Second

type collectionOptions struct {
	limit       int
	factTimeout time.Duration
	progress    api.ProgressFunc
}

type factProbe struct {
	instance Instance
	key      string
}

type probeOutcome struct {
	facts []api.Fact
	err   error
	probe factProbe
}

type factCollection struct {
	ctx       context.Context
	collector FactCollector
	options   collectionOptions
	probes    []factProbe
	semaphore chan struct{}
}

func collectInstanceFacts(
	ctx context.Context,
	collector FactCollector,
	instances []Instance,
	options collectionOptions,
) ([]api.Fact, error) {
	if options.limit <= 0 {
		options.limit = adaptiveConcurrency(instances)
	}
	if options.factTimeout <= 0 {
		options.factTimeout = defaultFactTimeout
	}
	probes := buildFactProbes(instances)
	run := factCollection{
		ctx: ctx, collector: collector, options: options, probes: probes,
		semaphore: make(chan struct{}, options.limit),
	}
	return run.execute()
}

func buildFactProbes(instances []Instance) []factProbe {
	probes := make([]factProbe, 0, len(instances))
	seen := make(map[string]struct{}, len(instances))
	for _, instance := range instances {
		if !instance.Applicable {
			continue
		}
		key := instance.Subject.ID + "\x00" + firstKey(instance)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		probes = append(probes, factProbe{instance: instance, key: key})
	}
	return probes
}

func (r *factCollection) execute() ([]api.Fact, error) {
	outcomes := make(chan probeOutcome, len(r.probes))
	var wg sync.WaitGroup
	if r.options.progress != nil {
		r.options.progress(api.ProgressEvent{
			Phase: api.ProgressCollecting, Total: len(r.probes), Current: "collecting facts",
		})
	}
	for _, probe := range r.probes {
		wg.Add(1)
		go r.runProbe(probe, outcomes, &wg)
	}
	go func() {
		wg.Wait()
		close(outcomes)
	}()
	return summarizeProbeOutcomes(outcomes, len(r.probes), r.options.progress)
}

func (r *factCollection) runProbe(probe factProbe, outcomes chan<- probeOutcome, wg *sync.WaitGroup) {
	defer wg.Done()
	defer func() {
		if recovered := recover(); recovered != nil {
			outcomes <- probeOutcome{err: fmt.Errorf("collector panic: %v", recovered), probe: probe}
		}
	}()
	select {
	case r.semaphore <- struct{}{}:
		defer func() { <-r.semaphore }()
	case <-r.ctx.Done():
		ctxErr := context.Canceled
		if err := r.ctx.Err(); err != nil {
			ctxErr = err
		}
		outcomes <- probeOutcome{err: ctxErr, probe: probe}
		return
	}
	factCtx, cancel := context.WithTimeout(r.ctx, r.options.factTimeout)
	defer cancel()
	facts, err := r.collector.Collect(factCtx, api.FactRequest{
		Keys: []string{firstKey(probe.instance)}, Subjects: []api.Subject{probe.instance.Subject},
		Timeout: r.options.factTimeout,
	})
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			outcomes <- probeOutcome{facts: []api.Fact{timeoutFact(probe.instance, err)}, probe: probe}
			return
		}
		outcomes <- probeOutcome{err: err, probe: probe}
		return
	}
	outcomes <- probeOutcome{facts: facts, probe: probe}
}

func timeoutFact(instance Instance, err error) api.Fact {
	return api.Fact{
		Key: firstKey(instance), Subject: instance.Subject.ID, Available: false,
		Error: &api.FactError{Code: api.FactErrTimeout, Message: err.Error()},
	}
}

func summarizeProbeOutcomes(
	outcomes <-chan probeOutcome,
	total int,
	progress api.ProgressFunc,
) ([]api.Fact, error) {
	var facts []api.Fact
	var firstErr error
	completed := 0
	for outcome := range outcomes {
		outcome = validateProbeOutcome(outcome)
		facts = append(facts, outcome.facts...)
		if outcome.err != nil && len(outcome.facts) == 0 {
			facts = append(facts, errorFact(outcome))
		}
		if firstErr == nil && outcome.err != nil {
			firstErr = outcome.err
		}
		completed++
		if progress != nil {
			progress(api.ProgressEvent{
				Phase: api.ProgressCollecting, Completed: completed, Total: total,
				Current: outcome.probe.instance.Rule.ID + "@" + outcome.probe.instance.Subject.ID,
			})
		}
	}
	return facts, firstErr
}

func validateProbeOutcome(outcome probeOutcome) probeOutcome {
	if len(outcome.facts) != 1 {
		outcome.err = fmt.Errorf(
			"collector contract error: expected one fact for %s, got %d",
			outcome.probe.key,
			len(outcome.facts),
		)
	}
	for _, fact := range outcome.facts {
		if err := validateCollectedFact(
			fact,
			firstKey(outcome.probe.instance),
			outcome.probe.instance.Subject.ID,
		); err != nil && outcome.err == nil {
			outcome.err = err
		}
	}
	return outcome
}

func errorFact(outcome probeOutcome) api.Fact {
	code := api.FactErrUnreachable
	if errors.Is(outcome.err, context.DeadlineExceeded) || errors.Is(outcome.err, context.Canceled) {
		code = api.FactErrTimeout
	}
	return api.Fact{
		Key: firstKey(outcome.probe.instance), Subject: outcome.probe.instance.Subject.ID,
		Available: false, Error: &api.FactError{Code: code, Message: outcome.err.Error()},
	}
}

func validateCollectedFact(fact api.Fact, expectedKey, expectedSubject string) error {
	if fact.Key == "" {
		return fmt.Errorf("collector contract error: fact key is empty")
	}
	if fact.Key != expectedKey || fact.Subject != expectedSubject {
		return fmt.Errorf(
			"collector contract error: requested %s/%s, got %s/%s",
			expectedSubject,
			expectedKey,
			fact.Subject,
			fact.Key,
		)
	}
	if fact.Available == (fact.Error != nil) {
		return fmt.Errorf("collector contract error: fact %s has invalid availability/error state", fact.Key)
	}
	if fact.Available && fact.Type != "" {
		if !fact.Type.Valid() || !ValueMatchesDeclaredType(fact.Type, fact.Value) {
			return fmt.Errorf("collector contract error: fact %s does not match declared type %q", fact.Key, fact.Type)
		}
	}
	if fact.Error == nil {
		return nil
	}
	switch fact.Error.Code {
	case api.FactErrNotFound, api.FactErrPermissionDenied, api.FactErrUnreachable,
		api.FactErrTimeout, api.FactErrUnsupported, api.FactErrParseError:
		return nil
	default:
		return fmt.Errorf("collector contract error: fact %s has invalid error code %q", fact.Key, fact.Error.Code)
	}
}

// ValueMatchesDeclaredType reports whether a fact value matches its explicit
// SDK display type. An empty type is handled by inference at the call site.
func ValueMatchesDeclaredType(valueType api.ValueType, value any) bool {
	inferred := api.InferValueType(value)
	switch valueType {
	case api.ValueTypeBytes, api.ValueTypeDuration:
		return inferred == api.ValueTypeInteger
	case api.ValueTypeEnum, api.ValueTypeAddress:
		return inferred == api.ValueTypeString
	case api.ValueTypeRelation:
		return inferred == api.ValueTypeBool || inferred == api.ValueTypeObject
	case api.ValueTypeObject:
		return inferred == api.ValueTypeObject
	default:
		return inferred == valueType
	}
}

// collectInstanceFactsWithDependencies executes fact probes in dependency
// levels. A dependent probe is not sent to the collector when its dependency
// has already failed; the evaluation phase will expose it as SKIP with the
// dependency root. This keeps a failed remote-node connection from causing a
// second wave of pointless SSH commands for systemd, CPU, ports and paths.
type dependencyCollectionInput struct {
	rules     []api.RuleTemplate
	order     []string
	instances []Instance
}

type dependencyCollection struct {
	ctx             context.Context
	collector       FactCollector
	options         collectionOptions
	rulesByID       map[string]api.RuleTemplate
	instancesByRule map[string][]Instance
	probeKeys       map[string]struct{}
	levels          [][]string
	states          map[string]ruleState
	progressDone    map[string]struct{}
	facts           []api.Fact
	firstErr        error
}

func collectInstanceFactsWithDependencies(
	ctx context.Context,
	collector FactCollector,
	input dependencyCollectionInput,
	options collectionOptions,
) ([]api.Fact, error) {
	rulesByID, instancesByRule, probeKeys := indexDependencyInputs(input.rules, input.instances)
	run := dependencyCollection{
		ctx: ctx, collector: collector, options: options,
		rulesByID: rulesByID, instancesByRule: instancesByRule, probeKeys: probeKeys,
		levels:       dependencyLevels(input.order, rulesByID),
		states:       make(map[string]ruleState, len(input.instances)),
		progressDone: make(map[string]struct{}, len(probeKeys)),
	}
	return run.execute()
}

func indexDependencyInputs(
	rules []api.RuleTemplate,
	instances []Instance,
) (map[string]api.RuleTemplate, map[string][]Instance, map[string]struct{}) {
	rulesByID := make(map[string]api.RuleTemplate, len(rules))
	for _, rule := range rules {
		rulesByID[rule.ID] = rule
	}
	instancesByRule := make(map[string][]Instance, len(rules))
	probeKeys := make(map[string]struct{}, len(instances))
	for _, instance := range instances {
		if !instance.Applicable {
			continue
		}
		instancesByRule[instance.Rule.ID] = append(instancesByRule[instance.Rule.ID], instance)
		probeKeys[probeKey(instance)] = struct{}{}
	}
	return rulesByID, instancesByRule, probeKeys
}

func dependencyLevels(order []string, rulesByID map[string]api.RuleTemplate) [][]string {
	depth := make(map[string]int, len(order))
	levels := make([][]string, len(order))
	maxLevel := -1
	for _, ruleID := range order {
		rule, ok := rulesByID[ruleID]
		if !ok {
			continue
		}
		level := ruleDepth(rule, depth)
		depth[ruleID] = level
		levels[level] = append(levels[level], ruleID)
		if level > maxLevel {
			maxLevel = level
		}
	}
	return levels[:maxLevel+1]
}

func ruleDepth(rule api.RuleTemplate, depth map[string]int) int {
	level := 0
	for _, dependency := range rule.DependsOn {
		if dependencyDepth, ok := depth[dependency]; ok && dependencyDepth+1 > level {
			level = dependencyDepth + 1
		}
	}
	return level
}

func (r *dependencyCollection) execute() ([]api.Fact, error) {
	r.emit(api.ProgressEvent{Phase: api.ProgressCollecting, Total: len(r.probeKeys), Current: "collecting facts"})
	for _, level := range r.levels {
		eligible, blocked := r.eligibleInstances(level)
		r.collectLevel(eligible)
		r.updateLevelStates(level, blocked)
	}
	return r.facts, r.firstErr
}

func (r *dependencyCollection) eligibleInstances(level []string) ([]Instance, map[string]bool) {
	eligible := make([]Instance, 0)
	blocked := make(map[string]bool)
	seen := make(map[string]struct{})
	for _, ruleID := range level {
		for _, instance := range r.instancesByRule[ruleID] {
			if r.blockedByDependency(instance) {
				blocked[stateKey(instance.Rule.ID, instance.Subject.ID)] = true
				continue
			}
			key := probeKey(instance)
			if _, duplicate := seen[key]; duplicate {
				continue
			}
			seen[key] = struct{}{}
			eligible = append(eligible, instance)
		}
	}
	return eligible, blocked
}

func (r *dependencyCollection) blockedByDependency(instance Instance) bool {
	blocked, root := dependencyFailed(instance.Rule, instance.Subject.ID, r.states)
	if !blocked {
		return false
	}
	r.states[stateKey(instance.Rule.ID, instance.Subject.ID)] = ruleState{
		status: api.StatusSkip,
		cause:  root,
	}
	r.markProgress(instance, api.StatusSkip)
	return true
}

func (r *dependencyCollection) collectLevel(eligible []Instance) {
	levelFacts, levelErr := collectInstanceFacts(
		r.ctx,
		r.collector,
		eligible,
		collectionOptions{limit: r.options.limit, factTimeout: r.options.factTimeout},
	)
	r.facts = append(r.facts, levelFacts...)
	if r.firstErr == nil && levelErr != nil {
		r.firstErr = levelErr
	}
	for _, instance := range eligible {
		r.markProgress(instance, "")
	}
}

func (r *dependencyCollection) updateLevelStates(level []string, blocked map[string]bool) {
	facts := indexFacts(r.facts)
	for _, ruleID := range level {
		for _, instance := range r.instancesByRule[ruleID] {
			stateID := stateKey(instance.Rule.ID, instance.Subject.ID)
			if blocked[stateID] {
				continue
			}
			fact := findFact(facts, instance)
			if fact == nil {
				fact = missingFact(instance)
			}
			evaluated := Evaluate(instance.Rule, *fact)
			r.states[stateID] = ruleState{
				status: NormalizeStatus(instance.Rule, evaluated), cause: evaluated.CauseCode,
			}
		}
	}
}

func (r *dependencyCollection) markProgress(instance Instance, status api.Status) {
	key := probeKey(instance)
	if _, done := r.progressDone[key]; done {
		return
	}
	r.progressDone[key] = struct{}{}
	r.emit(api.ProgressEvent{
		Phase: api.ProgressCollecting, Completed: len(r.progressDone), Total: len(r.probeKeys),
		Current: instance.Rule.ID + "@" + instance.Subject.ID, Status: status,
	})
}

func (r *dependencyCollection) emit(event api.ProgressEvent) {
	if r.options.progress != nil {
		r.options.progress(event)
	}
}

func probeKey(instance Instance) string {
	return instance.Subject.ID + "\x00" + firstKey(instance)
}

// adaptiveConcurrency derives a bounded fact-probe budget from the number of
// remote subjects. Cluster-create expands many facts per node, so a fixed
// budget is unnecessarily slow for larger clusters while an unbounded budget
// can overwhelm SSH endpoints. The explicit RuntimeOptions value always wins.
func adaptiveConcurrency(instances []Instance) int {
	const (
		base                       = 8
		cap                        = 32
		nodeProbeConcurrencyFactor = 2
	)
	nodes := make(map[string]struct{})
	for _, instance := range instances {
		if !instance.Applicable || instance.Subject.ID == "" || instance.Subject.ID == "bootstrap" ||
			instance.Subject.ID == "cluster" {
			continue
		}
		nodes[instance.Subject.ID] = struct{}{}
	}
	if len(nodes) == 0 {
		return base
	}
	limit := len(nodes) * nodeProbeConcurrencyFactor
	if limit < base {
		return base
	}
	if limit > cap {
		return cap
	}
	return limit
}

// ruleState tracks a rule's evaluated status and root-cause for dependency
// propagation. It is package-level so dependencyFailed can reference it.
type ruleState struct {
	status api.Status
	cause  string
}

// dependencyFailed reports whether any DependsOn of the rule is in a failed
// state, and returns the root cause instance ID.
func dependencyFailed(rule api.RuleTemplate, subject string, states map[string]ruleState) (bool, string) {
	for _, dep := range rule.DependsOn {
		st, ok := states[stateKey(dep, subject)]
		if !ok {
			st, ok = states[stateKey(dep, "")]
		}
		if ok {
			if st.status == api.StatusFail || st.status == api.StatusWarn {
				return true, dep
			}
			if st.status == api.StatusSkip && st.cause != "" {
				// Propagate through skip chains: if our dependency was skipped
				// due to a root, reference that root.
				return true, st.cause
			}
		}
	}
	return false, ""
}

func stateKey(ruleID, subject string) string { return ruleID + "\x00" + subject }

func orderInstances(instances []Instance, order []string) []Instance {
	rank := make(map[string]int, len(order))
	for i, id := range order {
		rank[id] = i
	}
	out := append([]Instance(nil), instances...)
	sort.SliceStable(out, func(i, j int) bool {
		return rank[out[i].Rule.ID] < rank[out[j].Rule.ID]
	})
	return out
}

// indexFacts builds a lookup from (subject+key) to fact.
func indexFacts(facts []api.Fact) map[string]*api.Fact {
	m := make(map[string]*api.Fact, len(facts))
	for i := range facts {
		key := facts[i].Subject + "\x00" + facts[i].Key
		m[key] = &facts[i]
	}
	return m
}

// findFact returns the fact for an instance, preferring the first fact key.
func findFact(idx map[string]*api.Fact, in Instance) *api.Fact {
	k := firstKey(in)
	return idx[in.Subject.ID+"\x00"+k]
}

func firstKey(in Instance) string {
	if len(in.FactKeys) > 0 {
		return in.FactKeys[0]
	}
	return in.Rule.Fact
}

func instanceID(in Instance) string {
	if in.ID != "" {
		return in.ID
	}
	if in.Subject.ID != "" {
		return in.Rule.ID + "@" + in.Subject.ID
	}
	return in.Rule.ID
}

func statusesOf(results []api.Result) []api.Status {
	out := make([]api.Status, len(results))
	for i, r := range results {
		out[i] = r.Status
	}
	return out
}

func sortResults(results []api.Result) {
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].RuleID != results[j].RuleID {
			return results[i].RuleID < results[j].RuleID
		}
		if results[i].Subject != results[j].Subject {
			return results[i].Subject < results[j].Subject
		}
		return results[i].InstanceID < results[j].InstanceID
	})
}
