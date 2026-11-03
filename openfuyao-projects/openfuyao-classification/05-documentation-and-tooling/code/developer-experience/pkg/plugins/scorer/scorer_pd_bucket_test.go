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

package scorer

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"sync"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	"hermes-router/pkg/plugins/common"
)

const (
	// Test configuration constants
	defaultAlpha         = 0.5
	defaultBeta          = 0.5
	defaultDecay         = 0.5
	defaultBucketSepLen  = 10.0
	testReqLen50         = 50
	testReqLen80         = 80
	testReqLen100        = 100
	testReqLen120        = 120
	testReqCount50       = 50
	floatTolerance       = 1e-6
	bucketSepLen200      = 200.0
	bucketSepLen500      = 500.0
	bucketSepLen1000     = 1000.0
	bucketSepLen2000     = 2000.0
	prefillScaleFactor   = 0.0345
	prefillBaseConstant  = 120.0745
	tokenDivisor         = 4.0
	testProfileName      = "pd-profile"
	testPDLabelName      = "test/pdRole"
	testPDGroupLabelName = "test/pdGroupID"
	testTPLabelName      = "test/pdTP"
	testRolePrefill      = "prefill"
	testRoleDecode       = "decode"
	testRoleLeader       = "leader"
	highLoad             = 600.0
	mediumLoad           = 200.0
	lowLoad              = 50.0
	tinyLoad             = 10.0
	smallLoad            = 20.0
	mockScoreHigh        = 10.0
	mockScoreLow         = 5.0
	bucketUpdateStart    = 400.0
	bucketUpdateNext     = 1600.0
	bucketUpdateExp1     = 880.0
	bucketUpdateExp2     = 1024.0

	// TP size constants for test groups
	tpOne   = 1
	tpTwo   = 2
	tpThree = 3
	tpFour  = 4

	// Test load values for active factor calculation
	testLoadGroup1 = 2.0
	testLoadGroup2 = 3.0
	testLoadGroup3 = 5.0
	testEpsilon    = 1e-6

	// Expected pod counts
	expectedPDPodCount = 3

	// Array indices for buildTwoPDGroupsPods parameters
	idxGroup1TP      = 0
	idxGroup2PrefixA = 1
	idxGroup2PrefixB = 2
	minTPSizesOne    = 1
	minTPSizesTwo    = 2
	minTPSizesThree  = 3

	// Constants for balanced score calculation
	twoGroupsAvgTPShare = 0.5
	normalizeDivisor    = 1.0
)

// assertFloatApproxEqual verifies that two float64 values are approximately equal within a tolerance.
func assertFloatApproxEqual(t *testing.T, got, want float64) {
	t.Helper()
	tolerance := floatTolerance * math.Max(1.0, math.Abs(want))
	if math.Abs(got-want) > tolerance {
		t.Fatalf("got %f, want %f (tolerance=%f)", got, want, tolerance)
	}
}

func TestNewPDBucketValidParameters(t *testing.T) {
	t.Run("valid_parameters", func(t *testing.T) {
		// given
		ctx := context.Background()
		params := &scorerPDBucketParameters{
			Alpha:                defaultAlpha,
			Beta:                 defaultBeta,
			DecayFactor:          defaultDecay,
			BucketSeperateLength: defaultBucketSepLen,
		}

		// when
		pdb := NewPDBucket(ctx, params)

		// then
		validatePDBucketFields(t, pdb, params)
	})
}

func TestNewPDBucketValidParametersGreaterThanOne(t *testing.T) {
	t.Run("parameters_greater_than_one_are_valid", func(t *testing.T) {
		// given
		ctx := context.Background()
		params := &scorerPDBucketParameters{
			Alpha:                2.5,
			Beta:                 3.0,
			DecayFactor:          1.5,
			BucketSeperateLength: 100.0,
		}

		// when
		pdb := NewPDBucket(ctx, params)

		// then
		validatePDBucketFields(t, pdb, params)
	})
}

func TestNewPDBucketValidParametersVerySmall(t *testing.T) {
	t.Run("very_small_positive_parameters_are_valid", func(t *testing.T) {
		// given
		ctx := context.Background()
		params := &scorerPDBucketParameters{
			Alpha:                0.001,
			Beta:                 0.001,
			DecayFactor:          0.001,
			BucketSeperateLength: 0.001,
		}

		// when
		pdb := NewPDBucket(ctx, params)

		// then
		validatePDBucketFields(t, pdb, params)
	})
}

// validatePDBucketFields checks if PDBucket fields match the parameters.
func validatePDBucketFields(t *testing.T, pdb *PDBucket, params *scorerPDBucketParameters) {
	t.Helper()
	if pdb == nil {
		t.Fatal("expected non-nil PDBucket")
	}
	if pdb.alpha != params.Alpha {
		t.Errorf("alpha = %f, want %f", pdb.alpha, params.Alpha)
	}
	if pdb.beta != params.Beta {
		t.Errorf("beta = %f, want %f", pdb.beta, params.Beta)
	}
	if pdb.decayFactor != params.DecayFactor {
		t.Errorf("decayFactor = %f, want %f", pdb.decayFactor, params.DecayFactor)
	}
	if pdb.bucketSeperateLength != params.BucketSeperateLength {
		t.Errorf("bucketSeperateLength = %f, want %f", pdb.bucketSeperateLength, params.BucketSeperateLength)
	}
	if pdb.typedName.Type != PDBucketType {
		t.Errorf("typedName.Type = %s, want %s", pdb.typedName.Type, PDBucketType)
	}
	if pdb.activeTokens == nil || pdb.prefillTPSize == nil || pdb.inflight == nil {
		t.Error("expected maps to be initialized")
	}
}

// TestNewPDBucketInvalidParameters verifies PDBucket returns nil for invalid parameters.
func TestNewPDBucketInvalidParameters(t *testing.T) {
	cases := getInvalidParamTestCases()
	ctx := context.Background()

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			// when
			pdb := NewPDBucket(ctx, tc.params)

			// then
			if pdb != nil {
				t.Errorf("expected nil PDBucket for case %q, got non-nil", tc.name)
			}
		})
	}
}

// paramTestCase defines a test case for parameter validation.
type paramTestCase struct {
	name   string
	params *scorerPDBucketParameters
}

// getInvalidParamTestCases generates test cases for invalid parameters.
func getInvalidParamTestCases() []paramTestCase {
	base := scorerPDBucketParameters{
		Alpha:                defaultAlpha,
		Beta:                 defaultBeta,
		DecayFactor:          defaultDecay,
		BucketSeperateLength: defaultBucketSepLen,
	}

	return []paramTestCase{
		{"nil_params", nil},
		{"alpha_zero", func() *scorerPDBucketParameters { p := base; p.Alpha = 0; return &p }()},
		{"beta_zero", func() *scorerPDBucketParameters { p := base; p.Beta = 0; return &p }()},
		{"decayFactor_zero", func() *scorerPDBucketParameters { p := base; p.DecayFactor = 0; return &p }()},
		{"bucketSeperateLength_non_positive", func() *scorerPDBucketParameters {
			p := base
			p.BucketSeperateLength = 0
			return &p
		}()},
	}
}

// TestPDBucketTypedNameAndWithName verifies name updates.
func TestPDBucketTypedNameAndWithName(t *testing.T) {
	t.Run("with_name_updates_typed_name", func(t *testing.T) {
		// given
		pdb := &PDBucket{
			typedName: plugins.TypedName{Type: PDBucketType, Name: ""},
		}
		const name = "test-pd-bucket"

		// when
		pdb.WithName(name)

		// then
		typed := pdb.TypedName()
		if typed.Name != name {
			t.Errorf("typedName.Name = %s, want %s", typed.Name, name)
		}
		if typed.Type != PDBucketType {
			t.Errorf("typedName.Type = %s, want %s", typed.Type, PDBucketType)
		}
	})
}

// TestPDBucketPreRequestNoPrimaryRoute verifies behavior when no route is present.
func TestPDBucketPreRequestNoPrimaryRoute(t *testing.T) {
	// given
	ctx := context.Background()
	pdb := newEmptyPDBucket()
	req := newCompletionsRequest("req-no-route", testReqLen100)

	// when
	pdb.PreRequest(ctx, req, nil)

	// then
	verifyEmptyState(t, pdb, req)
}

// verifyEmptyState checks that PDBucket state remains empty.
func verifyEmptyState(t *testing.T, pdb *PDBucket, req *types.LLMRequest) {
	t.Helper()
	if len(pdb.inflight) != 0 {
		t.Fatalf("expected no inflight records when route is nil, got %d", len(pdb.inflight))
	}
	if len(pdb.activeTokens) != 0 {
		t.Fatalf("expected no activeTokens updated when route is nil, got %v", pdb.activeTokens)
	}
	if req.Headers != nil {
		t.Fatalf("expected no headers injected when route is nil, got %v", req.Headers)
	}
}

// TestPDBucketPreRequestUpdateTokensAndHeaders verifies token tracking and header injection.
func TestPDBucketPreRequestUpdateTokensAndHeaders(t *testing.T) {
	// given
	ctx := context.Background()
	pdb := newEmptyPDBucket()
	podsGroup1, _ := buildTwoPDGroupsPods(tpTwo, tpTwo)
	reqID := "req-1"
	req := newCompletionsRequest(reqID, testReqLen100)
	schedulingResult := createTestSchedulingResult(podsGroup1)

	// when
	pdb.PreRequest(ctx, req, schedulingResult)

	// then
	verifyInflightRecord(t, pdb, reqID, podsGroup1)
	verifyActiveTokens(t, pdb, podsGroup1, testReqLen100)
	verifyRequestHeaders(t, req)
}

// verifyInflightRecord checks the inflight map.
func verifyInflightRecord(t *testing.T, pdb *PDBucket, reqID string, pg pdGroupPods) {
	t.Helper()
	if len(pdb.inflight) != 1 {
		t.Fatalf("expected 1 inflight record, got %d", len(pdb.inflight))
	}
	route, ok := pdb.inflight[reqID]
	if !ok {
		t.Fatalf("expected inflight entry for request %s", reqID)
	}
	if route.leaderPodName != pg.leaderName ||
		route.prefillPodName != pg.prefillName ||
		route.decodePodName != pg.decodeName {
		t.Fatalf("unexpected inflight route, got %+v, want leader=%s prefill=%s decode=%s",
			route, pg.leaderName, pg.prefillName, pg.decodeName)
	}
}

// verifyActiveTokens checks the token counts.
func verifyActiveTokens(t *testing.T, pdb *PDBucket, pg pdGroupPods, reqLen int) {
	t.Helper()
	expectedPrefill := (float64(reqLen)/tokenDivisor)*prefillScaleFactor + prefillBaseConstant
	expectedDecode := float64(reqLen)

	assertFloatApproxEqual(t, pdb.activeTokens[pg.prefillName], expectedPrefill)
	assertFloatApproxEqual(t, pdb.activeTokens[pg.decodeName], expectedDecode)
}

// verifyRequestHeaders checks if headers are injected.
func verifyRequestHeaders(t *testing.T, req *types.LLMRequest) {
	t.Helper()
	if req.Headers == nil {
		t.Fatal("expected headers to be initialized")
	}
	if req.Headers[common.PrefillPodHeader] == "" {
		t.Fatalf("expected %s header to be set", common.PrefillPodHeader)
	}
	if req.Headers[common.DecodePodHeader] == "" {
		t.Fatalf("expected %s header to be set", common.DecodePodHeader)
	}
}

// TestPDBucketPreRequestConcurrentUpdates verifies thread safety of PreRequest.
func TestPDBucketPreRequestConcurrentUpdates(t *testing.T) {
	// given
	ctx := context.Background()
	pdb := newEmptyPDBucket()
	podsGroup1, _ := buildTwoPDGroupsPods(tpTwo, tpTwo)
	schedResult := createTestSchedulingResult(podsGroup1)

	// when
	runConcurrentPreRequests(ctx, pdb, schedResult, testReqCount50)

	// then
	if len(pdb.inflight) != testReqCount50 {
		t.Fatalf("expected %d inflight records, got %d", testReqCount50, len(pdb.inflight))
	}
	verifyAggregateTokens(t, pdb, podsGroup1, testReqCount50, testReqLen80)
}

// runConcurrentPreRequests executes PreRequest concurrently.
func runConcurrentPreRequests(
	ctx context.Context,
	pdb *PDBucket,
	res *types.SchedulingResult,
	count int,
) {
	var wg sync.WaitGroup
	wg.Add(count)
	for i := 0; i < count; i++ {
		i := i
		go func() {
			defer wg.Done()
			req := newCompletionsRequest(fmt.Sprintf("req-%d", i), testReqLen80)
			pdb.PreRequest(ctx, req, res)
		}()
	}
	wg.Wait()
}

// verifyAggregateTokens checks total token counts.
func verifyAggregateTokens(
	t *testing.T,
	pdb *PDBucket,
	pg pdGroupPods,
	count int,
	reqLen int,
) {
	t.Helper()
	unitPrefill := (float64(reqLen)/tokenDivisor)*prefillScaleFactor + prefillBaseConstant
	expectedPrefill := float64(count) * unitPrefill
	expectedDecode := float64(count * reqLen)

	assertFloatApproxEqual(t, pdb.activeTokens[pg.prefillName], expectedPrefill)
	assertFloatApproxEqual(t, pdb.activeTokens[pg.decodeName], expectedDecode)
}

// TestPDBucketResponseComplete verifies cleanup after request completion.
func TestPDBucketResponseComplete(t *testing.T) {
	// given
	ctx := context.Background()
	pdb := newEmptyPDBucket()
	podsGroup1, _ := buildTwoPDGroupsPods(tpOne, tpFour)
	req := newCompletionsRequest("req-functional", testReqLen120)
	schedResult := createTestSchedulingResult(podsGroup1)

	pdb.PreRequest(ctx, req, schedResult)
	checkPreRequestState(t, pdb, podsGroup1)

	// when
	pdb.ResponseComplete(ctx, req, &requestcontrol.Response{}, nil)

	// then
	checkPostResponseState(t, pdb, podsGroup1)
}

// checkPreRequestState verifies state before response completion.
func checkPreRequestState(t *testing.T, pdb *PDBucket, pg pdGroupPods) {
	t.Helper()
	if len(pdb.inflight) != 1 {
		t.Fatalf("expected 1 inflight record, got %d", len(pdb.inflight))
	}
	if pdb.activeTokens[pg.prefillName] <= 0 || pdb.activeTokens[pg.decodeName] <= 0 {
		t.Fatal("expected positive tokens before ResponseComplete")
	}
}

// checkPostResponseState verifies state after response completion.
func checkPostResponseState(t *testing.T, pdb *PDBucket, pg pdGroupPods) {
	t.Helper()
	if len(pdb.inflight) != 0 {
		t.Fatalf("expected 0 inflight records, got %d", len(pdb.inflight))
	}
	if pdb.activeTokens[pg.prefillName] != 0 {
		t.Errorf("expected zero prefill tokens, got %f", pdb.activeTokens[pg.prefillName])
	}
	if pdb.activeTokens[pg.decodeName] != 0 {
		t.Errorf("expected zero decode tokens, got %f", pdb.activeTokens[pg.decodeName])
	}
}

// TestPDBucketResponseCompleteConcurrent verifies thread safety of ResponseComplete.
func TestPDBucketResponseCompleteConcurrent(t *testing.T) {
	// given
	ctx := context.Background()
	pdb := newEmptyPDBucket()
	baseTokens := 100.0
	setupConcurrentInflightState(pdb, testReqCount50, baseTokens)

	// when
	runConcurrentResponseComplete(ctx, pdb, testReqCount50)

	// then
	verifyConcurrentCleanup(t, pdb)
}

// setupConcurrentInflightState prepares mock inflight data.
func setupConcurrentInflightState(pdb *PDBucket, count int, base float64) {
	for i := 0; i < count; i++ {
		reqID := fmt.Sprintf("req-%d", i)
		prefill := fmt.Sprintf("prefill-%d", i)
		decode := fmt.Sprintf("decode-%d", i)
		tokens := base + float64(i)

		pdb.activeTokens[prefill] += tokens
		pdb.activeTokens[decode] += tokens
		pdb.inflight[reqID] = &requestRoute{
			prefillPodName:     prefill,
			decodePodName:      decode,
			prefillTokensAdded: tokens,
			decodeTokensAdded:  tokens,
		}
	}
}

// runConcurrentResponseComplete executes ResponseComplete concurrently.
func runConcurrentResponseComplete(ctx context.Context, pdb *PDBucket, count int) {
	var wg sync.WaitGroup
	wg.Add(count)
	for i := 0; i < count; i++ {
		i := i
		go func() {
			defer wg.Done()
			req := &types.LLMRequest{RequestId: fmt.Sprintf("req-%d", i)}
			pdb.ResponseComplete(ctx, req, &requestcontrol.Response{}, nil)
		}()
	}
	wg.Wait()
}

// verifyConcurrentCleanup checks if all states are cleared.
func verifyConcurrentCleanup(t *testing.T, pdb *PDBucket) {
	t.Helper()
	if len(pdb.inflight) != 0 {
		t.Fatalf("expected 0 inflight records, got %d", len(pdb.inflight))
	}
	for name, v := range pdb.activeTokens {
		if v < 0 || v != 0 {
			t.Fatalf("token mismatch for %s: %f", name, v)
		}
	}
}

// TestCalculateActiveFactors verifies active factor calculation.
func TestCalculateActiveFactors(t *testing.T) {
	t.Run("basic_distribution", func(t *testing.T) {
		pdb := &PDBucket{}
		groupLoads := []float64{testLoadGroup1, testLoadGroup2, testLoadGroup3}

		activeFactors, minFactor, maxFactor := pdb.calculateActiveFactors(groupLoads)

		total := testLoadGroup1 + testLoadGroup2 + testLoadGroup3
		denominator := total + testEpsilon
		expected := []float64{
			testLoadGroup1 / denominator,
			testLoadGroup2 / denominator,
			testLoadGroup3 / denominator,
		}
		for i := range expected {
			assertFloatApproxEqual(t, activeFactors[i], expected[i])
		}
		assertFloatApproxEqual(t, minFactor, expected[0])
		assertFloatApproxEqual(t, maxFactor, expected[2])
	})

	t.Run("all_zero_loads", func(t *testing.T) {
		pdb := &PDBucket{}
		groupLoads := []float64{0, 0}

		activeFactors, minFactor, maxFactor := pdb.calculateActiveFactors(groupLoads)

		for _, factor := range activeFactors {
			if factor != 0 {
				t.Errorf("expected zero active factor, got %f", factor)
			}
		}
		if minFactor != 0 || maxFactor != 0 {
			t.Errorf("expected zero min/max, got min=%f max=%f", minFactor, maxFactor)
		}
	})
}

// TestCalculateTPFactors verifies TP factor logic.
func TestCalculateTPFactors(t *testing.T) {
	t.Run("tp_share_biases_scores", func(t *testing.T) {
		pdb := &PDBucket{alpha: 0.7, beta: 0.3}
		tpFactors := pdb.calculateTPFactors([]float64{1, 4}, 5.0, bucketSepLen1000, testReqLen100)
		finalScores := pdb.calculateFinalScores([]float64{0.4, 0.6}, tpFactors)

		assertFloatApproxEqual(t, finalScores[0], 0.199)
		assertFloatApproxEqual(t, finalScores[1], 0.501)
	})

	t.Run("larger_tp_share_gets_higher_score", func(t *testing.T) {
		pdb := &PDBucket{alpha: 0.5, beta: 0.5}
		tpFactors := pdb.calculateTPFactors([]float64{1, 2}, 3.0, bucketSepLen200, testReqLen100)
		finalScores := pdb.calculateFinalScores([]float64{0.5, 0.5}, tpFactors)

		if finalScores[1] <= finalScores[0] {
			t.Fatalf("expected group2 > group1, got %f vs %f", finalScores[1], finalScores[0])
		}
	})
}

// TestScorePDGroupListImbalancedLoadUsesActiveFactors verifies scoring under high load imbalance.
func TestScorePDGroupListImbalancedLoadUsesActiveFactors(t *testing.T) {
	// given
	pdb := &PDBucket{
		alpha:                0.6,
		beta:                 0.4,
		bucketSeperateLength: bucketSepLen1000,
		activeTokens:         make(map[string]float64),
		prefillTPSize:        make(map[string]int),
	}
	group1, group2 := buildTwoPDGroupsPods(tpOne, tpFour)
	list := createPDGroupList(group1, group2)
	group1.applyPrefillTPSizeTo(pdb.prefillTPSize)
	group2.applyPrefillTPSizeTo(pdb.prefillTPSize)

	pdb.activeTokens[group1.prefillName] = highLoad
	pdb.activeTokens[group1.decodeName] = 0
	pdb.activeTokens[group2.prefillName] = lowLoad
	pdb.activeTokens[group2.decodeName] = lowLoad

	// when
	scored := pdb.scorePDGroupList(list, bucketSepLen200)

	// then
	verifyImbalancedScores(t, pdb, scored, group1, group2)
}

// verifyImbalancedScores checks scores based on pure load.
func verifyImbalancedScores(
	t *testing.T,
	pdb *PDBucket,
	scored *common.PDGroupList,
	g1, g2 pdGroupPods,
) {
	t.Helper()
	loadG1 := pdb.activeTokens[g1.prefillName] + pdb.activeTokens[g1.decodeName]
	loadG2 := pdb.activeTokens[g2.prefillName] + pdb.activeTokens[g2.decodeName]
	total := loadG1 + loadG2 + testEpsilon
	expected := []float64{loadG1 / total, loadG2 / total}

	for i, want := range expected {
		assertFloatApproxEqual(t, scored.Groups[i].LeaderPod.Score, want)
	}
}

// TestScorePDGroupListBalancedLoadAppliesTPFactors verifies scoring under balanced load.
func TestScorePDGroupListBalancedLoadAppliesTPFactors(t *testing.T) {
	// given
	pdb := &PDBucket{
		alpha:                0.6,
		beta:                 0.4,
		bucketSeperateLength: bucketSepLen1000,
		activeTokens:         make(map[string]float64),
		prefillTPSize:        make(map[string]int),
	}
	group1, group2 := buildTwoPDGroupsPods(tpOne, tpFour)
	group1.applyPrefillTPSizeTo(pdb.prefillTPSize)
	group2.applyPrefillTPSizeTo(pdb.prefillTPSize)
	list := createPDGroupList(group1, group2)

	pdb.activeTokens[group1.prefillName] = 120
	pdb.activeTokens[group1.decodeName] = 80
	pdb.activeTokens[group2.prefillName] = 130
	pdb.activeTokens[group2.decodeName] = 70

	// when
	scored := pdb.scorePDGroupList(list, testReqLen100)

	// then
	verifyBalancedScores(t, balancedScoreContext{
		pdb:    pdb,
		scored: scored,
		g1:     group1,
		g2:     group2,
		reqLen: testReqLen100,
	})
}

// balancedScoreContext holds data for balanced score verification.
type balancedScoreContext struct {
	pdb    *PDBucket
	scored *common.PDGroupList
	g1     pdGroupPods
	g2     pdGroupPods
	reqLen int
}

// verifyBalancedScores checks scores with TP factors.
func verifyBalancedScores(t *testing.T, ctx balancedScoreContext) {
	t.Helper()
	actives := calculateExpectedActiveFactors(ctx.pdb, ctx.g1, ctx.g2)
	tpFactors := calculateExpectedTPFactors(ctx.pdb, ctx.g1, ctx.g2, ctx.reqLen)
	expectedScores := computeFinalScores(ctx.pdb, actives, tpFactors)

	verifyScoresMatch(t, ctx.scored, expectedScores)
}

// calculateExpectedActiveFactors computes normalized load factors for two groups.
func calculateExpectedActiveFactors(pdb *PDBucket, g1, g2 pdGroupPods) []float64 {
	l1 := (pdb.activeTokens[g1.prefillName] + pdb.activeTokens[g1.decodeName]) / normalizeDivisor
	l2 := (pdb.activeTokens[g2.prefillName] + pdb.activeTokens[g2.decodeName]) / normalizeDivisor
	totalL := l1 + l2 + testEpsilon
	return []float64{l1 / totalL, l2 / totalL}
}

// calculateExpectedTPFactors computes TP-based factors for two groups.
func calculateExpectedTPFactors(pdb *PDBucket, g1, g2 pdGroupPods, reqLen int) []float64 {
	totalTP := float64(g1.totalPrefillTP() + g2.totalPrefillTP())
	lenFactor := normalizeDivisor - float64(reqLen)/pdb.bucketSeperateLength

	tp1Share := float64(g1.totalPrefillTP())/totalTP - twoGroupsAvgTPShare
	tp2Share := float64(g2.totalPrefillTP())/totalTP - twoGroupsAvgTPShare

	return []float64{
		tp1Share * lenFactor,
		tp2Share * lenFactor,
	}
}

// computeFinalScores combines active factors and TP factors with weights.
func computeFinalScores(pdb *PDBucket, actives, tpFactors []float64) []float64 {
	scores := make([]float64, len(actives))
	for i := range actives {
		scores[i] = pdb.alpha*actives[i] + pdb.beta*tpFactors[i]
	}
	return scores
}

// verifyScoresMatch checks if actual scores match expected scores.
func verifyScoresMatch(t *testing.T, scored *common.PDGroupList, expected []float64) {
	t.Helper()
	for i := range scored.Groups {
		assertFloatApproxEqual(t, scored.Groups[i].LeaderPod.Score, expected[i])
	}
}

// TestCalculateGroupLoadsSingleGroupBasic verifies load calculation for a single group.
func TestCalculateGroupLoadsSingleGroupBasic(t *testing.T) {
	// given
	pdb := newEmptyPDBucket()
	pdb.bucketSeperateLength = bucketSepLen500
	group := buildPDGroupPodsWithTPSizes("group-1", []int{tpTwo, tpThree})
	pdb.activeTokens[group.prefillName] = 10.0
	pdb.activeTokens[group.decodeName] = 20.0
	group.applyPrefillTPSizeTo(pdb.prefillTPSize)

	list := common.PDGroupList{
		Groups: []common.PDGroup{createPDGroup(group)},
	}

	// when
	loads, tps, bucket := pdb.calculateGroupLoadsAndPrefillTP(list)

	// then
	if len(loads) != 1 {
		t.Fatal("expected 1 group load")
	}
	assertFloatApproxEqual(t, loads[0], 30.0)
	assertFloatApproxEqual(t, tps[0], float64(group.totalPrefillTP()))
	if bucket != pdb.bucketSeperateLength {
		t.Errorf("bucket mismatch")
	}
}

// TestCalculateGroupLoadsNoDecodeUsesSingleDecodeCount verifies fallback for missing decode pods.
func TestCalculateGroupLoadsNoDecodeUsesSingleDecodeCount(t *testing.T) {
	// given
	pdb := newEmptyPDBucket()
	pdb.bucketSeperateLength = bucketSepLen1000
	g1 := buildPDGroupPodsWithTPSizes("g1", []int{tpOne})
	g2 := buildPDGroupPodsWithTPSizes("g2", []int{tpOne})

	pdb.activeTokens[g1.prefillName] = 40
	pdb.activeTokens[g1.decodeName] = 60
	pdb.activeTokens[g2.prefillName] = 30
	g1.applyPrefillTPSizeTo(pdb.prefillTPSize)
	g2.applyPrefillTPSizeTo(pdb.prefillTPSize)

	g2Group := createPDGroup(g2)
	g2Group.DecodePods = nil // Explicitly remove decode pods
	list := common.PDGroupList{
		Groups: []common.PDGroup{createPDGroup(g1), g2Group},
	}

	// when
	loads, _, _ := pdb.calculateGroupLoadsAndPrefillTP(list)

	// then
	assertFloatApproxEqual(t, loads[0], 100.0)
	assertFloatApproxEqual(t, loads[1], 30.0)
}

// TestPDBucketScoreWithMissingPDGroups verifies behavior when state is missing.
func TestPDBucketScoreWithMissingPDGroups(t *testing.T) {
	ctx := context.Background()
	pdb := &PDBucket{}
	state := types.NewCycleState()
	req := newCompletionsRequest("req-score-missing", testReqLen50)

	result := pdb.Score(ctx, state, req, nil)

	if len(result) != 0 {
		t.Fatalf("expected empty score map, got %d", len(result))
	}
	_, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err == nil {
		t.Fatal("expected error for missing state")
	}
}

// TestPDBucketScoreUpdatesPDGroupScores verifies Score updates the state.
func TestPDBucketScoreUpdatesPDGroupScores(t *testing.T) {
	ctx := context.Background()
	pdb := &PDBucket{
		alpha: defaultAlpha, beta: defaultBeta,
		bucketSeperateLength: bucketSepLen1000,
		activeTokens:         make(map[string]float64),
		prefillTPSize:        make(map[string]int),
	}
	g1, g2 := buildTwoPDGroupsPods(tpOne, tpOne)
	g1.applyPrefillTPSizeTo(pdb.prefillTPSize)
	g2.applyPrefillTPSizeTo(pdb.prefillTPSize)

	pdb.activeTokens[g1.prefillName] = mediumLoad
	pdb.activeTokens[g1.decodeName] = lowLoad
	pdb.activeTokens[g2.prefillName] = smallLoad
	pdb.activeTokens[g2.decodeName] = tinyLoad

	state := types.NewCycleState()
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{
		Groups: []common.PDGroup{createPDGroup(g1), createPDGroup(g2)},
	})

	pdb.Score(ctx, state, newCompletionsRequest("r1", testReqLen120), nil)

	updated, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err != nil {
		t.Fatal(err)
	}
	if updated.Groups[0].LeaderPod.Score <= updated.Groups[1].LeaderPod.Score {
		t.Error("expected group1 score > group2 score")
	}
}

// TestPDBucketPickNoGroupsReturnsEmpty verifies Pick with no groups.
func TestPDBucketPickNoGroupsReturnsEmpty(t *testing.T) {
	ctx := context.Background()
	pdb := &PDBucket{}
	state := types.NewCycleState()
	result := pdb.Pick(ctx, state, nil)
	if len(result.TargetPods) != 0 {
		t.Error("expected zero target pods")
	}
}

// TestPDBucketPickSelectsMinScoreGroup verifies Pick selection logic.
func TestPDBucketPickSelectsMinScoreGroup(t *testing.T) {
	ctx := context.Background()
	pdb := &PDBucket{}
	state := types.NewCycleState()
	g1, g2 := buildTwoPDGroupsPods(tpOne, tpOne)

	list := &common.PDGroupList{
		Groups: []common.PDGroup{createPDGroup(g1), createPDGroup(g2)},
	}
	list.Groups[0].LeaderPod.Score = mockScoreHigh
	list.Groups[1].LeaderPod.Score = mockScoreLow
	state.Write(common.PDGroupsCycleStateKey, list)

	result := pdb.Pick(ctx, state, nil)
	if len(result.TargetPods) != expectedPDPodCount {
		t.Fatalf("expected %d pods, got %d", expectedPDPodCount, len(result.TargetPods))
	}
	if result.TargetPods[0].GetPod().PodName != g2.leaderName {
		t.Error("picked wrong leader")
	}
}

// TestCalculateGroupLoadsAndPrefillTPConcurrentAccess verifies concurrent safety.
func TestCalculateGroupLoadsAndPrefillTPConcurrentAccess(t *testing.T) {
	ctx := context.Background()
	pdb := newEmptyPDBucket()
	pdb.bucketSeperateLength = bucketSepLen2000
	group, _ := buildTwoPDGroupsPods(tpTwo, tpTwo)
	group.applyPrefillTPSizeTo(pdb.prefillTPSize)
	pdList := common.PDGroupList{Groups: []common.PDGroup{createPDGroup(group)}}
	schedRes := createTestSchedulingResult(group)

	errCh := make(chan error, 1)
	stopCh := make(chan struct{})
	var readerWG sync.WaitGroup

	// Start concurrent reader
	readerWG.Add(1)
	go runConcurrentStateReader(&readerWG, pdb, pdList, stopCh, errCh)

	// Start concurrent writers
	runConcurrentRequestWriters(ctx, pdb, schedRes, testReqCount50)

	close(stopCh)
	readerWG.Wait()

	select {
	case err := <-errCh:
		t.Fatalf("concurrent error: %v", err)
	default:
	}

	verifyFinalConcurrentState(t, pdb, pdList, group)
}

// runConcurrentStateReader continuously reads state in a loop.
func runConcurrentStateReader(
	wg *sync.WaitGroup,
	pdb *PDBucket,
	list common.PDGroupList,
	stopCh <-chan struct{},
	errCh chan<- error,
) {
	defer wg.Done()
	for {
		select {
		case <-stopCh:
			return
		default:
			loads, tps, bucket := pdb.calculateGroupLoadsAndPrefillTP(list)
			if len(loads) != 1 || len(tps) != 1 {
				errCh <- fmt.Errorf("bad length")
				return
			}
			if loads[0] < 0 || tps[0] < 0 {
				errCh <- fmt.Errorf("negative values")
				return
			}
			if bucket != pdb.bucketSeperateLength {
				errCh <- fmt.Errorf("bucket mismatch")
				return
			}
		}
	}
}

// runConcurrentRequestWriters simulates request traffic.
func runConcurrentRequestWriters(
	ctx context.Context,
	pdb *PDBucket,
	res *types.SchedulingResult,
	count int,
) {
	var wg sync.WaitGroup
	wg.Add(count)
	for i := 0; i < count; i++ {
		i := i
		go func() {
			defer wg.Done()
			req := newCompletionsRequest(fmt.Sprintf("req-calc-%d", i), testReqLen50+i)
			pdb.PreRequest(ctx, req, res)
			pdb.ResponseComplete(ctx, req, &requestcontrol.Response{}, nil)
		}()
	}
	wg.Wait()
}

// verifyFinalConcurrentState checks state after concurrency test.
func verifyFinalConcurrentState(
	t *testing.T,
	pdb *PDBucket,
	list common.PDGroupList,
	group pdGroupPods,
) {
	t.Helper()
	loads, tps, _ := pdb.calculateGroupLoadsAndPrefillTP(list)
	if len(loads) != 1 {
		t.Fatal("unexpected final length")
	}
	assertFloatApproxEqual(t, loads[0], 0)
	assertFloatApproxEqual(t, tps[0], float64(group.totalPrefillTP()))
}

// TestUpdateBucketSeperateLength verifies exponential moving average update.
func TestUpdateBucketSeperateLength(t *testing.T) {
	t.Run("moves_towards_request_length", func(t *testing.T) {
		pdb := &PDBucket{
			decayFactor:          0.8,
			bucketSeperateLength: bucketSepLen1000,
		}

		pdb.updateBucketSeperateLength(bucketUpdateStart)
		assertFloatApproxEqual(t, pdb.bucketSeperateLength, bucketUpdateExp1)

		pdb.updateBucketSeperateLength(bucketUpdateNext)
		assertFloatApproxEqual(t, pdb.bucketSeperateLength, bucketUpdateExp2)
	})
}

type pdGroupPods struct {
	groupID           string
	leaderName        string
	prefillName       string
	decodeName        string
	leaderPod         types.Pod
	decodePod         types.Pod
	prefillPod        types.Pod
	extraPrefillPods  []types.Pod
	extraPrefillNames []string
	prefillTPSizes    map[string]int
}

func (p pdGroupPods) prefillPods() []types.Pod {
	pods := []types.Pod{p.prefillPod}
	return append(pods, p.extraPrefillPods...)
}

func (p pdGroupPods) applyPrefillTPSizeTo(target map[string]int) {
	for name, size := range p.prefillTPSizes {
		target[name] = size
	}
}

func (p pdGroupPods) totalPrefillTP() int {
	total := 0
	for _, size := range p.prefillTPSizes {
		total += size
	}
	return total
}

func newEmptyPDBucket() *PDBucket {
	return &PDBucket{
		activeTokens:  make(map[string]float64),
		prefillTPSize: make(map[string]int),
		inflight:      make(map[string]*requestRoute),
	}
}

func createPDGroup(g pdGroupPods) common.PDGroup {
	return common.PDGroup{
		ID:          g.groupID,
		PrefillPods: toScoredPods(g.prefillPods()),
		DecodePods:  toScoredPods([]types.Pod{g.decodePod}),
		LeaderPod:   types.ScoredPod{Pod: g.leaderPod, Score: 0.0},
	}
}

func createPDGroupList(groups ...pdGroupPods) common.PDGroupList {
	list := make([]common.PDGroup, len(groups))
	for i, g := range groups {
		list[i] = createPDGroup(g)
	}
	return common.PDGroupList{Groups: list}
}

func createTestSchedulingResult(pg pdGroupPods) *types.SchedulingResult {
	return &types.SchedulingResult{
		ProfileResults: map[string]*types.ProfileRunResult{
			testProfileName: {
				TargetPods: []types.Pod{pg.leaderPod, pg.prefillPod, pg.decodePod},
			},
		},
		PrimaryProfileName: testProfileName,
	}
}

func toScoredPods(pods []types.Pod) []types.ScoredPod {
	scoredPods := make([]types.ScoredPod, len(pods))
	for i, pod := range pods {
		scoredPods[i] = types.ScoredPod{Pod: pod, Score: 0.0}
	}
	return scoredPods
}

func buildTwoPDGroupsPods(tpSizes ...int) (pdGroupPods, pdGroupPods) {
	g1TP, g2TP1, g2TP2 := tpOne, tpFour, tpTwo

	switch len(tpSizes) {
	case minTPSizesThree:
		g2TP2 = tpSizes[idxGroup2PrefixB]
		fallthrough
	case minTPSizesTwo:
		g2TP1 = tpSizes[idxGroup2PrefixA]
		fallthrough
	case minTPSizesOne:
		g1TP = tpSizes[idxGroup1TP]
	}

	return buildPDGroupPodsWithTPSizes("group-1", []int{g1TP}),
		buildPDGroupPodsWithTPSizes("group-2", []int{g2TP1, g2TP2})
}

func buildPDGroupPodsWithTPSizes(groupID string, prefillTPSizes []int) pdGroupPods {
	if len(prefillTPSizes) == 0 {
		prefillTPSizes = []int{tpOne}
	}
	leaderName := groupID + "-leader"
	decodeName := groupID + "-decode"

	leaderPod := newLabeledPod(leaderName, groupID, testRoleLeader, 0)
	decodePod := newLabeledPod(decodeName, groupID, testRoleDecode, 0)

	prefillPods, prefillNames, prefillTPMap := createPrefillPods(groupID, prefillTPSizes)
	var extraPods []types.Pod
	var extraNames []string
	if len(prefillPods) > 1 {
		extraPods = append(extraPods, prefillPods[1:]...)
		extraNames = append(extraNames, prefillNames[1:]...)
	}

	return pdGroupPods{
		groupID: groupID, leaderName: leaderName, prefillName: prefillNames[0], decodeName: decodeName,
		leaderPod: leaderPod, prefillPod: prefillPods[0], decodePod: decodePod,
		extraPrefillPods: extraPods, extraPrefillNames: extraNames, prefillTPSizes: prefillTPMap,
	}
}

func createPrefillPods(groupID string, sizes []int) ([]types.Pod, []string, map[string]int) {
	pods := make([]types.Pod, len(sizes))
	names := make([]string, len(sizes))
	tpMap := make(map[string]int, len(sizes))

	for i, size := range sizes {
		name := fmt.Sprintf("%s-prefill", groupID)
		if i > 0 {
			name = fmt.Sprintf("%s-prefill-%d", groupID, i+1)
		}
		pod := newLabeledPod(name, groupID, testRolePrefill, size)
		pods[i] = pod
		names[i] = name
		tpMap[name] = tpSizeFromLabel(pod)
	}
	return pods, names, tpMap
}

func newLabeledPod(podName, groupID, roleValue string, tpSize int) types.Pod {
	labels := map[string]string{
		testPDLabelName:      roleValue,
		testPDGroupLabelName: groupID,
	}
	if tpSize > 0 && roleValue == testRolePrefill {
		labels[testTPLabelName] = strconv.Itoa(tpSize)
	}
	return &types.PodMetrics{
		Pod:          &backend.Pod{PodName: podName, Labels: labels},
		MetricsState: &metrics.MetricsState{},
	}
}

func tpSizeFromLabel(pod types.Pod) int {
	if pod == nil || pod.GetPod() == nil {
		return 1
	}
	if val, ok := pod.GetPod().Labels[testTPLabelName]; ok {
		if size, err := strconv.Atoi(val); err == nil && size > 0 {
			return size
		}
	}
	return 1
}

func newCompletionsRequest(requestID string, promptLength int) *types.LLMRequest {
	prompt := make([]rune, promptLength)
	for i := range prompt {
		prompt[i] = 'a'
	}
	return &types.LLMRequest{
		RequestId: requestID,
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{Prompt: string(prompt)},
		},
	}
}
