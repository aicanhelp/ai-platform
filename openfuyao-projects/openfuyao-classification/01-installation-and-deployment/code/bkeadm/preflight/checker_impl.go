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
	"fmt"
	"maps"
	"slices"
	"strings"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/bundle"
	"gopkg.openfuyao.cn/preflight/engine"
	"gopkg.openfuyao.cn/preflight/normalize"
)

// CheckerImpl is the host-agnostic SDK facade.
type CheckerImpl struct{ deps Dependencies }

// New constructs a Checker from explicit host capabilities.
func New(deps Dependencies) *CheckerImpl { return &CheckerImpl{deps: deps} }

// Check validates the neutral resolved input, binds reviewed checks and
// evaluates them using read-only facts.
func (c *CheckerImpl) Check(ctx context.Context, in CheckInput) (api.Report, error) {
	resolved := in.Resolved
	rep := api.Report{
		SchemaVersion: api.SchemaVersion,
		Operation:     resolved.Plan.Operation,
	}
	if err := validateResolved(resolved); err != nil {
		rep.Overall = api.OverallInternalError
		rep.ExitCode = api.ExitCode(rep.Overall)
		rep.Issues = []api.RunIssue{{Code: "ADAPTER-CONTRACT", Severity: api.IssueError,
			Stage: api.IssueStageSetup, Effect: api.IssueBlockStart,
			Cause: err.Error(), Action: "report this adapter contract error and use a corrected bke build"}}
		return rep, nil
	}

	b, err := c.loadCheckedBundle(in)
	if err != nil {
		return bundleErrorReport(rep, err), nil
	}

	rep.Issues = resolved.Issues
	if api.HasStartBlocker(resolved.Issues) {
		rep.Overall = api.OverallInputError
		rep.ExitCode = api.ExitCode(rep.Overall)
		return rep, nil
	}
	rep.Operation = resolved.Plan.Operation

	rules, instances, err := bindRules(b, resolved)
	if err != nil {
		return bundleErrorReport(rep, err), nil
	}
	applicable := applicableCount(instances)
	if in.Progress != nil {
		in.Progress(api.ProgressEvent{Phase: api.ProgressPlan, Total: applicable})
	}
	collector := c.collectorFor(resolved.Facts)
	results, overall := engine.Run(ctx, rules, instances, collector, engine.RunOptions{
		Timeout: in.Runtime.OverallTimeout, FactTimeout: in.Runtime.FactTimeout, MaxConcurrency: in.Runtime.MaxConcurrency,
		Progress: in.Progress,
	})
	rep.Results = results
	rep.Summary = api.Summarize(statusesOf(results))
	rep.Overall = overall
	rep.ExitCode = api.ExitCode(overall)
	rep.Issues = appendInterruptionIssue(rep.Issues, overall)
	reportCompletion(in.Progress, overall, len(results), applicable)
	return rep, nil
}

func (c *CheckerImpl) loadCheckedBundle(in CheckInput) (*api.PreflightBundle, error) {
	b, err := bundle.Load(in.Bundle)
	if err != nil {
		return nil, err
	}
	return b, nil
}

func applicableCount(instances []engine.Instance) int {
	count := 0
	for _, instance := range instances {
		if instance.Applicable {
			count++
		}
	}
	return count
}

func (c *CheckerImpl) collectorFor(facts []api.Fact) api.Collector {
	collector := api.Collector(noopCollector{})
	if c.deps.Collector != nil {
		collector = c.deps.Collector
	}
	if len(facts) > 0 {
		collector = &factOverlayCollector{facts: facts, next: collector}
	}
	return collector
}

func appendInterruptionIssue(issues []api.RunIssue, overall api.Overall) []api.RunIssue {
	switch overall {
	case api.OverallTimedOut:
		return append(issues, api.RunIssue{
			Code: "PREFLIGHT-TIMEOUT", Severity: api.IssueError,
			Stage: api.IssueStageExecution, Effect: api.IssueInterrupt,
			Cause: "preflight timed out before all applicable checks completed",
			Action: "inspect the last probe; fix connectivity, SSH, permission or management-cluster access, " +
				"then rerun with an appropriate --fact-timeout or --timeout",
		})
	case api.OverallCanceled:
		return append(issues, api.RunIssue{
			Code: "PREFLIGHT-CANCELED", Severity: api.IssueError,
			Stage: api.IssueStageExecution, Effect: api.IssueInterrupt,
			Cause:  "preflight was canceled before all applicable checks completed",
			Action: "rerun preflight when the environment is reachable",
		})
	default:
		return issues
	}
}

func reportCompletion(progress api.ProgressFunc, overall api.Overall, completed, total int) {
	if progress == nil {
		return
	}
	event := api.ProgressEvent{Completed: completed, Total: total}
	switch overall {
	case api.OverallTimedOut:
		event.Phase = api.ProgressIncomplete
		event.Current = "preflight timed out before all checks completed"
	case api.OverallCanceled:
		event.Phase = api.ProgressIncomplete
		event.Current = "preflight canceled before all checks completed"
	default:
		event.Phase = api.ProgressComplete
		event.Completed = total
		event.Current = "checks complete"
	}
	progress(event)
}

func validateResolved(resolved api.ResolveResult) error {
	plan := resolved.Plan
	blocked := api.HasStartBlocker(resolved.Issues)
	if blocked {
		// A blocked adapter result intentionally may not contain a complete plan;
		// the public INPUT_ERROR issues are the contract for this path.
		return nil
	}
	if err := validatePlanIdentity(plan); err != nil {
		return err
	}
	subjectsByID, err := validatePlanSubjects(plan.Subjects)
	if err != nil {
		return err
	}
	if err := validatePlanSources(plan.Sources); err != nil {
		return err
	}
	if err := validatePlanValues(plan.Values); err != nil {
		return err
	}
	recomputed, err := normalize.ComputePlanHash(&plan)
	if err != nil {
		return fmt.Errorf("recompute plan hash: %w", err)
	}
	if recomputed != plan.PlanHash {
		return fmt.Errorf("plan hash mismatch: supplied %s, recomputed %s", plan.PlanHash, recomputed)
	}
	if err := validateBindings(resolved.Bindings, subjectsByID); err != nil {
		return err
	}
	return validateAdapterFacts(resolved.Facts)
}

func validatePlanIdentity(plan api.OperationPlan) error {
	if plan.SchemaVersion != api.SchemaVersion {
		return fmt.Errorf("unsupported plan schema version %q", plan.SchemaVersion)
	}
	if !api.IsSupportedOperation(plan.Operation) {
		return fmt.Errorf("unsupported plan operation %q", plan.Operation)
	}
	if plan.PlanHash.IsZero() {
		return fmt.Errorf("plan hash is required")
	}
	return nil
}

func validatePlanSubjects(subjects []api.Subject) (map[string]api.Subject, error) {
	subjectsByID := make(map[string]api.Subject, len(subjects))
	for _, subject := range subjects {
		if subject.ID == "" {
			return nil, fmt.Errorf("plan subject ID is empty")
		}
		if _, exists := subjectsByID[subject.ID]; exists {
			return nil, fmt.Errorf("duplicate plan subject %q", subject.ID)
		}
		subjectsByID[subject.ID] = subject
	}
	return subjectsByID, nil
}

func validatePlanSources(sources []api.Source) error {
	seenSources := make(map[string]struct{}, len(sources))
	for _, source := range sources {
		if source.ID == "" {
			return fmt.Errorf("plan source ID is empty")
		}
		if _, exists := seenSources[source.ID]; exists {
			return fmt.Errorf("duplicate plan source %q", source.ID)
		}
		seenSources[source.ID] = struct{}{}
	}
	return nil
}

func validatePlanValues(values map[string]api.Value) error {
	for key, value := range values {
		if value.Key != key {
			return fmt.Errorf("plan value %q carries key %q", key, value.Key)
		}
		if !value.Type.Valid() || !engine.ValueMatchesDeclaredType(value.Type, value.Value) {
			return fmt.Errorf("plan value %q does not match declared type %q", key, value.Type)
		}
	}
	return nil
}

func validateBindings(bindings []api.RuleBinding, subjectsByID map[string]api.Subject) error {
	seenBindings := make(map[string]struct{}, len(bindings))
	for _, binding := range bindings {
		if !binding.Applicable {
			continue
		}
		if binding.TemplateID == "" || binding.Subject.ID == "" || binding.Fact == "" {
			return fmt.Errorf("binding %q is missing template, subject or fact", binding.InstanceID)
		}
		planSubject, exists := subjectsByID[binding.Subject.ID]
		if !exists {
			return fmt.Errorf("binding %q references unknown plan subject %q", binding.InstanceID, binding.Subject.ID)
		}
		if !sameSubject(binding.Subject, planSubject) {
			return fmt.Errorf(
				"binding %q subject %q differs from the hashed plan subject",
				binding.InstanceID,
				binding.Subject.ID,
			)
		}
		identity := binding.InstanceID
		if identity == "" {
			identity = binding.TemplateID + "\x00" + binding.Subject.ID + "\x00" + binding.Fact
		}
		if _, exists := seenBindings[identity]; exists {
			return fmt.Errorf("duplicate binding instance %q", identity)
		}
		seenBindings[identity] = struct{}{}
	}
	return nil
}

func validateAdapterFacts(facts []api.Fact) error {
	for _, fact := range facts {
		if fact.Key == "" {
			return fmt.Errorf("adapter fact key is empty")
		}
		if fact.Available == (fact.Error != nil) {
			return fmt.Errorf("adapter fact %s availability/error invariant is invalid", fact.Key)
		}
		if fact.Available && fact.Type != "" &&
			(!fact.Type.Valid() || !engine.ValueMatchesDeclaredType(fact.Type, fact.Value)) {
			return fmt.Errorf("adapter fact %s does not match declared type %q", fact.Key, fact.Type)
		}
		if fact.Error != nil && !validFactErrorCode(fact.Error.Code) {
			return fmt.Errorf("adapter fact %s has invalid error code %q", fact.Key, fact.Error.Code)
		}
	}
	return nil
}

func sameSubject(left, right api.Subject) bool {
	return left.ID == right.ID && left.Address == right.Address &&
		slices.Equal(left.Role, right.Role) && maps.Equal(left.Metadata, right.Metadata)
}

func validFactErrorCode(code string) bool {
	switch code {
	case api.FactErrNotFound, api.FactErrPermissionDenied, api.FactErrUnreachable,
		api.FactErrTimeout, api.FactErrUnsupported, api.FactErrParseError:
		return true
	default:
		return false
	}
}

func bundleErrorReport(rep api.Report, err error) api.Report {
	rep.Issues = append(rep.Issues, api.RunIssue{
		Code: "BUNDLE-INVALID", Severity: api.IssueError,
		Stage: api.IssueStageSetup, Effect: api.IssueBlockStart,
		Cause: err.Error(), Action: "restore the trusted preflight bundle, then run preflight again",
	})
	rep.Overall = api.OverallBundleError
	rep.ExitCode = api.ExitCode(rep.Overall)
	return rep
}

func bindRules(b *api.PreflightBundle, resolved api.ResolveResult) ([]api.RuleTemplate, []engine.Instance, error) {
	binder := newRuleBinder(b, resolved)
	for _, binding := range resolved.Bindings {
		if !binding.Applicable {
			continue
		}
		if err := binder.add(binding); err != nil {
			return nil, nil, err
		}
	}
	return binder.rules, binder.instances, nil
}

type ruleBinder struct {
	operation     api.Operation
	templates     map[string]api.RuleTemplate
	subjects      map[string]api.Subject
	seenRules     map[string]bool
	seenInstances map[string]bool
	rules         []api.RuleTemplate
	instances     []engine.Instance
}

func newRuleBinder(b *api.PreflightBundle, resolved api.ResolveResult) *ruleBinder {
	binder := &ruleBinder{
		operation: resolved.Plan.Operation, templates: make(map[string]api.RuleTemplate, len(b.Rules)),
		subjects:  make(map[string]api.Subject, len(resolved.Plan.Subjects)),
		seenRules: map[string]bool{}, seenInstances: map[string]bool{},
		rules:     make([]api.RuleTemplate, 0, len(resolved.Bindings)),
		instances: make([]engine.Instance, 0, len(resolved.Bindings)),
	}
	for _, template := range b.Rules {
		binder.templates[template.ID] = template
	}
	for _, subject := range resolved.Plan.Subjects {
		binder.subjects[subject.ID] = subject
	}
	return binder
}

func (b *ruleBinder) add(binding api.RuleBinding) error {
	template, ok := b.templates[binding.TemplateID]
	if !ok {
		return fmt.Errorf("binding references rule %q not present in Bundle", binding.TemplateID)
	}
	if template.Operation != b.operation {
		return fmt.Errorf(
			"rule %q operation %q does not match plan %q", template.ID, template.Operation, b.operation,
		)
	}
	instanceID := binding.InstanceID
	if instanceID == "" {
		instanceID = template.ID
		if binding.Subject.ID != "" {
			instanceID += "@" + binding.Subject.ID
		}
	}
	if b.seenInstances[instanceID] {
		return fmt.Errorf("duplicate rule instance %q", instanceID)
	}
	b.seenInstances[instanceID] = true
	if !b.seenRules[template.ID] {
		b.rules = append(b.rules, template)
		b.seenRules[template.ID] = true
	}
	b.instances = append(b.instances, boundInstance(template, binding, b.subjects[binding.Subject.ID], instanceID))
	return nil
}

func boundInstance(
	template api.RuleTemplate,
	binding api.RuleBinding,
	subject api.Subject,
	instanceID string,
) engine.Instance {
	fact := binding.Fact
	if fact == "" {
		fact = template.Fact
	}
	template.Fact = fact
	if binding.Name != "" {
		template.Name = binding.Name
	}
	if binding.Fix != "" {
		template.Fix = binding.Fix
	}
	return engine.Instance{
		Rule: template, Subject: subject, ID: instanceID, Applicable: true, FactKeys: []string{fact},
	}
}

func statusesOf(results []api.Result) []api.Status {
	out := make([]api.Status, len(results))
	for i, result := range results {
		out[i] = result.Status
	}
	return out
}

type noopCollector struct{}

func (noopCollector) ID() string { return "noop" }
func (noopCollector) Collect(_ context.Context, req api.FactRequest) ([]api.Fact, error) {
	out := make([]api.Fact, 0, len(req.Keys)*max(1, len(req.Subjects)))
	subjects := req.Subjects
	if len(subjects) == 0 {
		subjects = []api.Subject{{}}
	}
	for _, subject := range subjects {
		for _, key := range req.Keys {
			out = append(out, api.Fact{Key: key, Subject: subject.ID, Available: false,
				Error: &api.FactError{Code: api.FactErrUnsupported, Message: "no collector configured"}})
		}
	}
	return out, nil
}

type factOverlayCollector struct {
	facts []api.Fact
	next  api.Collector
}

func (c *factOverlayCollector) ID() string { return "adapter-facts+" + c.next.ID() }
func (c *factOverlayCollector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	wanted := requestedFactIDs(req)
	out := overlayFacts(c.facts, wanted)
	if len(wanted) == 0 {
		return out, nil
	}
	collected, err := c.next.Collect(ctx, remainingFactRequest(req, wanted))
	return append(out, collected...), err
}

func requestedFactIDs(req api.FactRequest) map[string]struct{} {
	wanted := make(map[string]struct{}, len(req.Keys)*max(1, len(req.Subjects)))
	subjects := req.Subjects
	if len(subjects) == 0 {
		subjects = []api.Subject{{}}
	}
	for _, subject := range subjects {
		for _, key := range req.Keys {
			wanted[subject.ID+"\x00"+key] = struct{}{}
		}
	}
	return wanted
}

func overlayFacts(facts []api.Fact, wanted map[string]struct{}) []api.Fact {
	out := make([]api.Fact, 0, len(wanted))
	for _, fact := range facts {
		key := fact.Subject + "\x00" + fact.Key
		if _, ok := wanted[key]; ok {
			out = append(out, fact)
			delete(wanted, key)
		}
	}
	return out
}

func remainingFactRequest(req api.FactRequest, wanted map[string]struct{}) api.FactRequest {
	remaining := api.FactRequest{Timeout: req.Timeout}
	keySeen := make(map[string]struct{})
	subjectSeen := make(map[string]struct{})
	subjectsByID := make(map[string]api.Subject, len(req.Subjects))
	for _, subject := range req.Subjects {
		subjectsByID[subject.ID] = subject
	}
	for composite := range wanted {
		subjectID, key, _ := strings.Cut(composite, "\x00")
		if _, exists := keySeen[key]; !exists {
			remaining.Keys = append(remaining.Keys, key)
			keySeen[key] = struct{}{}
		}
		if subjectID == "" {
			continue
		}
		if _, exists := subjectSeen[subjectID]; exists {
			continue
		}
		remaining.Subjects = append(remaining.Subjects, subjectsByID[subjectID])
		subjectSeen[subjectID] = struct{}{}
	}
	return remaining
}
