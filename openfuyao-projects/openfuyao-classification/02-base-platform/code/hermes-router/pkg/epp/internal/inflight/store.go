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

package inflight

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/go-logr/logr"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"

	"hermes-router/pkg/epp/internal/predictiondata"
)

const (
	// DefaultAttributeKey is an exported constant used by Hermes Router.
	DefaultAttributeKey = "inflight_load"
	// DefaultStoreName is an exported constant used by Hermes Router.
	DefaultStoreName = "default"
	// CompletedRecordSchemaV1 is an exported constant used by Hermes Router.
	CompletedRecordSchemaV1     = 1
	defaultFlushThreshold       = 500
	defaultFlushInterval        = time.Minute
	defaultCompletedBufferLimit = 1024
	completedDirPerm            = 0o755
	completedFilePerm           = 0o644
	// Expire abandoned requests so their inflight contribution cannot leak forever.
	defaultRequestTimeout = 5 * time.Minute
	defaultRequestSweep   = time.Minute
)

// PredictionMode labels which prediction stream (if any) was attached to a
// request at scheduling time. The empty value means no prediction was recorded
// — either no predictor was running or the predictor produced no output for
// the selected endpoint.
const (
	// PredictionModeActive is an exported constant used by Hermes Router.
	PredictionModeActive = "active"
)

var stores sync.Map

// EndpointKey describes EndpointKey data used by Hermes Router.
type EndpointKey string

// Load is the per-endpoint inflight snapshot published to endpoint attributes.
type Load struct {
	// Tokens stores the Tokens value.
	Tokens int
	// Requests stores the Requests value.
	Requests int
}

// Clone returns a deep copy of the load snapshot.
func (l *Load) Clone() fwkdl.Cloneable {
	if l == nil {
		return (*Load)(nil)
	}
	cloned := *l
	return &cloned
}

// LatencyPrediction describes LatencyPrediction data used by Hermes Router.
type LatencyPrediction struct {
	// TTFTMillis stores the TTFTMillis value.
	TTFTMillis float64 `json:"ttftMillis,omitempty"`
	// TPOTMillis stores the TPOTMillis value.
	TPOTMillis float64 `json:"tpotMillis,omitempty"`
	// TotalLatencyMillis stores the TotalLatencyMillis value.
	TotalLatencyMillis float64 `json:"totalLatencyMillis,omitempty"`
}

// RequestMetaData describes RequestMetaData data used by Hermes Router.
type RequestMetaData struct {
	// SchemaVersion stores the SchemaVersion value.
	SchemaVersion int `json:"schemaVersion"`
	// RequestID stores the RequestID value.
	RequestID string `json:"requestId"`
	// Model stores the Model value.
	Model string `json:"model,omitempty"`
	// CompletedAt stores the CompletedAt value.
	CompletedAt time.Time `json:"completedAt,omitempty"`
	// InputTokenLength stores the InputTokenLength value.
	InputTokenLength int `json:"inputTokenLength,omitempty"`
	// OutputLength stores the OutputLength value.
	OutputLength int `json:"outputLength,omitempty"`
	// SelectedEndpoint stores the SelectedEndpoint value.
	SelectedEndpoint EndpointKey `json:"-"`
	// TTFTMillis stores the TTFTMillis value.
	TTFTMillis float64 `json:"ttftMillis,omitempty"`
	// TotalLatencyMillis stores the TotalLatencyMillis value.
	TotalLatencyMillis float64 `json:"totalLatencyMillis,omitempty"`
	// DispatchInflight stores the DispatchInflight value.
	DispatchInflight Load `json:"-"`
	// PrefixCacheScore stores the PrefixCacheScore value.
	PrefixCacheScore float64 `json:"prefixCacheScore,omitempty"`
	// Prediction stores the Prediction value.
	Prediction LatencyPrediction `json:"prediction,omitempty"`
	// PredictionMode stores the PredictionMode value.
	PredictionMode string `json:"predictionMode,omitempty"`
	// SelectedRoute stores the SelectedRoute value.
	SelectedRoute *SelectedRoute `json:"selectedRoute,omitempty"`
	// PredictionInputs stores the PredictionInputs value.
	PredictionInputs map[string]predictiondata.Input `json:"predictionInputs,omitempty"`
	// MissingFeatureReasons stores the MissingFeatureReasons value.
	MissingFeatureReasons map[string]string `json:"missingFeatureReasons,omitempty"`
}

// RequestStart describes RequestStart data used by Hermes Router.
type RequestStart struct {
	// RequestID stores the RequestID value.
	RequestID string
	// Model stores the Model value.
	Model string
	// InputTokenLength stores the InputTokenLength value.
	InputTokenLength int
	// SelectedEndpoint stores the SelectedEndpoint value.
	SelectedEndpoint *fwkdl.EndpointMetadata
	// TrackedEndpoints stores the TrackedEndpoints value.
	TrackedEndpoints []*fwkdl.EndpointMetadata
	// DispatchInflight stores the DispatchInflight value.
	DispatchInflight Load
	// PrefixCacheScore stores the PrefixCacheScore value.
	PrefixCacheScore float64
	// Prediction stores the Prediction value.
	Prediction LatencyPrediction
	// PredictionMode stores the PredictionMode value.
	PredictionMode string
	// SelectedRoute stores the SelectedRoute value.
	SelectedRoute *SelectedRoute
	// PredictionInputs stores the PredictionInputs value.
	PredictionInputs map[string]predictiondata.Input
	// MissingFeatureReasons stores the MissingFeatureReasons value.
	MissingFeatureReasons map[string]string
	// StartedAt stores the StartedAt value.
	StartedAt time.Time
}

// RequestCompletion describes RequestCompletion data used by Hermes Router.
type RequestCompletion struct {
	// RequestID stores the RequestID value.
	RequestID string
	// CompletedAt stores the CompletedAt value.
	CompletedAt time.Time
	// OutputLength stores the OutputLength value.
	OutputLength int
	// SelectedEndpoint stores the SelectedEndpoint value.
	SelectedEndpoint *fwkdl.EndpointMetadata
}

// PersistenceConfig describes PersistenceConfig data used by Hermes Router.
type PersistenceConfig struct {
	// Enabled stores the Enabled value.
	Enabled bool
	// FlushThreshold stores the FlushThreshold value.
	FlushThreshold int
	// FlushInterval stores the FlushInterval value.
	FlushInterval time.Duration
	// OutputPath stores the OutputPath value.
	OutputPath string
}

// completedBatchWriter writes one detached batch. On error, retryFrom is the
// first record index that should be retained for a later retry.
type completedBatchWriter interface {
	WriteCompletedBatch(cfg PersistenceConfig, records []RequestMetaData) (retryFrom int, err error)
}

type fileCompletedBatchWriter struct{}

// ActiveRequest describes ActiveRequest data used by Hermes Router.
type ActiveRequest struct {
	// MetaData stores the MetaData value.
	MetaData RequestMetaData
}

type requestState struct {
	endpointKeys []EndpointKey
	metaData     RequestMetaData
	startedAt    time.Time
	firstChunkAt time.Time
	expiresAt    time.Time
}

type endpointState struct {
	Load       Load
	Active     bool
	LastSeenAt time.Time
}

// Store tracks inflight request state shared across request-control plugins.
type Store struct {
	mu               sync.RWMutex
	flushMu          sync.Mutex
	name             string
	endpoints        map[EndpointKey]*endpointState
	requests         map[string]requestState
	completed        []RequestMetaData
	droppedCompleted uint64
	flushFailures    uint64
	persistence      PersistenceConfig
	lastFlushAt      time.Time
	lastFlushErr     error
	closed           bool
	logger           logr.Logger
	flushWriter      completedBatchWriter
	flushNotify      chan struct{}
	flushStop        chan struct{}
	flushDone        chan struct{}
	requestTimeout   time.Duration
	requestSweep     time.Duration
	now              func() time.Time
	expiryStop       chan struct{}
	expiryDone       chan struct{}
	expiryRunning    bool
}

// StoreForName returns a shared inflight store for the given logical name.
func StoreForName(name string) *Store {
	resolvedName := name
	if resolvedName == "" {
		resolvedName = DefaultStoreName
	}
	entry, _ := stores.LoadOrStore(resolvedName, newNamedStore(resolvedName, PersistenceConfig{}))
	store, ok := entry.(*Store)
	if !ok {
		panic(fmt.Sprintf("inflight store registry entry %q has unexpected type %T", resolvedName, entry))
	}
	return store
}

// StoreForNameWithPersistence returns a shared store and applies persistence settings.
func StoreForNameWithPersistence(name string, cfg PersistenceConfig) *Store {
	store := StoreForName(name)
	store.ConfigurePersistence(cfg)
	return store
}

func newStore() *Store {
	return newNamedStore("", PersistenceConfig{})
}

func newStoreWithPersistence(cfg PersistenceConfig) *Store {
	return newNamedStore("", cfg)
}

func newNamedStore(name string, cfg PersistenceConfig) *Store {
	cfg = normalizePersistenceConfig(cfg)
	now := time.Now()
	store := &Store{
		name:           name,
		endpoints:      make(map[EndpointKey]*endpointState),
		requests:       make(map[string]requestState),
		completed:      make([]RequestMetaData, 0),
		persistence:    cfg,
		lastFlushAt:    now,
		logger:         logr.Discard(),
		flushWriter:    fileCompletedBatchWriter{},
		requestTimeout: defaultRequestTimeout,
		requestSweep:   defaultRequestSweep,
		now:            time.Now,
	}
	if cfg.Enabled {
		store.ensureAsyncFlusherLocked()
	}
	return store
}

// ConfigurePersistence provides the exported ConfigurePersistence operation.
func (s *Store) ConfigurePersistence(cfg PersistenceConfig) {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.persistence = normalizePersistenceConfig(cfg)
	if !s.persistence.Enabled {
		s.lastFlushErr = nil
	} else {
		if s.lastFlushAt.IsZero() {
			s.lastFlushAt = s.now()
		}
		s.ensureAsyncFlusherLocked()
		s.signalFlushIfNeededLocked()
	}
	s.ensureExpiryLoopLocked()
}

// ConfigureLogger provides the exported ConfigureLogger operation.
func (s *Store) ConfigureLogger(logger logr.Logger) {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.logger = logger
}

// Flush writes pending completed request metadata.
func (s *Store) Flush() error {
	if s == nil {
		return nil
	}

	return s.flushCompleted(true)
}

// Close releases resources held by the receiver.
func (s *Store) Close() error {
	if s == nil {
		return nil
	}

	var (
		name       string
		err        error
		flushStop  chan struct{}
		flushDone  chan struct{}
		expiryStop chan struct{}
		expiryDone chan struct{}
	)
	func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.closed {
			return
		}
		s.closed = true
		name = s.name
		flushStop = s.flushStop
		flushDone = s.flushDone
		expiryStop = s.expiryStop
		expiryDone = s.expiryDone
	}()

	if flushStop != nil {
		close(flushStop)
	}
	if flushDone != nil {
		<-flushDone
	}
	err = s.Flush()

	if expiryStop != nil {
		close(expiryStop)
	}
	if expiryDone != nil {
		<-expiryDone
	}

	if name != "" {
		stores.Delete(name)
	}
	return err
}

// LastFlushError provides the exported LastFlushError operation.
func (s *Store) LastFlushError() error {
	if s == nil {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastFlushErr
}

// DroppedCompletedCount provides the exported DroppedCompletedCount operation.
func (s *Store) DroppedCompletedCount() uint64 {
	if s == nil {
		return 0
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.droppedCompleted
}

// FlushFailureCount provides the exported FlushFailureCount operation.
func (s *Store) FlushFailureCount() uint64 {
	if s == nil {
		return 0
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.flushFailures
}

// ActiveRequest provides the exported ActiveRequest operation.
func (s *Store) ActiveRequest(reqID string) *ActiveRequest {
	if s == nil || reqID == "" {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	state, ok := s.requests[reqID]
	if !ok {
		return nil
	}

	return &ActiveRequest{MetaData: state.metaData}
}

// StartRequest provides the exported StartRequest operation.
func (s *Store) StartRequest(start RequestStart) {
	if s == nil || start.RequestID == "" {
		return
	}
	trackedKeys, selectedKey, ok := trackedEndpointKeys(start)
	if !ok {
		return
	}
	if start.InputTokenLength < 0 {
		start.InputTokenLength = 0
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	startedAt := start.StartedAt
	if startedAt.IsZero() {
		startedAt = s.now()
	}

	if existing, ok := s.requests[start.RequestID]; ok {
		s.subtract(existing)
	}

	state := requestState{
		endpointKeys: trackedKeys,
		startedAt:    startedAt,
		expiresAt:    s.nextExpiry(startedAt),
		metaData: RequestMetaData{
			SchemaVersion:         CompletedRecordSchemaV1,
			RequestID:             start.RequestID,
			Model:                 start.Model,
			InputTokenLength:      start.InputTokenLength,
			SelectedEndpoint:      selectedKey,
			DispatchInflight:      start.DispatchInflight,
			PrefixCacheScore:      start.PrefixCacheScore,
			Prediction:            start.Prediction,
			PredictionMode:        start.PredictionMode,
			SelectedRoute:         ensureSelectedRoute(start.SelectedRoute, string(selectedKey)),
			PredictionInputs:      copyPredictionInputs(start.PredictionInputs),
			MissingFeatureReasons: copyMissingFeatureReasons(start.MissingFeatureReasons),
		},
	}
	s.requests[start.RequestID] = state
	s.add(state)
}

// MarkFirstChunk provides the exported MarkFirstChunk operation.
func (s *Store) MarkFirstChunk(reqID string, chunkAt time.Time) {
	if s == nil || reqID == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	state, ok := s.requests[reqID]
	if !ok {
		return
	}
	if state.firstChunkAt.IsZero() {
		state.firstChunkAt = chunkAt
	}
	if !state.startedAt.IsZero() && state.metaData.TTFTMillis == 0 {
		state.metaData.TTFTMillis = durationMillis(chunkAt.Sub(state.startedAt))
	}
	state.expiresAt = s.nextExpiry(chunkAt)
	s.requests[reqID] = state
}

// CompleteRequest provides the exported CompleteRequest operation.
func (s *Store) CompleteRequest(done RequestCompletion) {
	if s == nil || done.RequestID == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	state, ok := s.requests[done.RequestID]
	if !ok {
		return
	}
	if done.OutputLength > 0 {
		state.metaData.OutputLength = done.OutputLength
	}
	if state.metaData.SelectedEndpoint == "" {
		if key, ok := endpointKey(done.SelectedEndpoint); ok {
			state.metaData.SelectedEndpoint = key
		}
	}
	if !state.startedAt.IsZero() {
		state.metaData.TotalLatencyMillis = durationMillis(done.CompletedAt.Sub(state.startedAt))
	}
	state.metaData.CompletedAt = done.CompletedAt

	delete(s.requests, done.RequestID)
	s.subtract(state)
	s.completed = append(s.completed, state.metaData)
	if s.persistence.Enabled {
		s.signalFlushIfNeededLocked()
		s.limitCompletedLocked()
		return
	}
	s.limitCompletedLocked()
}

// DrainCompleted provides the exported DrainCompleted operation.
func (s *Store) DrainCompleted(limit int) []RequestMetaData {
	if s == nil {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if limit <= 0 || limit > len(s.completed) {
		limit = len(s.completed)
	}
	if limit == 0 {
		return nil
	}

	drained := append([]RequestMetaData(nil), s.completed[:limit]...)
	remaining := append([]RequestMetaData(nil), s.completed[limit:]...)
	s.completed = remaining
	return drained
}

// ApplyEndpointEvent provides the exported ApplyEndpointEvent operation.
func (s *Store) ApplyEndpointEvent(event fwkdl.EndpointEvent, at time.Time) {
	if s == nil || event.Endpoint == nil {
		return
	}
	key, ok := endpointKey(event.Endpoint.GetMetadata())
	if !ok {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if event.Type == fwkdl.EventDelete {
		s.releaseEndpointLoadLocked(key)
		return
	}

	endpoint := s.ensureEndpointLocked(key)
	endpoint.LastSeenAt = at
	endpoint.Active = true
}

// Snapshot returns the current inflight load for the given endpoint metadata.
func (s *Store) Snapshot(metadata *fwkdl.EndpointMetadata) Load {
	if s == nil {
		return Load{}
	}
	key, ok := endpointKey(metadata)
	if !ok {
		return Load{}
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	endpoint, ok := s.endpoints[key]
	if !ok {
		return Load{}
	}
	return endpoint.Load
}

func (s *Store) add(state requestState) {
	for _, key := range state.endpointKeys {
		endpoint := s.ensureEndpointLocked(key)
		load := endpoint.Load
		load.Tokens += state.metaData.InputTokenLength
		load.Requests++
		endpoint.Load = load
	}
}

func (s *Store) subtract(state requestState) {
	for _, key := range state.endpointKeys {
		endpoint := s.ensureEndpointLocked(key)
		load := endpoint.Load
		load.Tokens -= state.metaData.InputTokenLength
		load.Requests--
		if load.Tokens <= 0 {
			load.Tokens = 0
		}
		if load.Requests <= 0 {
			load.Requests = 0
		}
		endpoint.Load = load
		if !endpoint.Active && load.Tokens == 0 && load.Requests == 0 {
			delete(s.endpoints, key)
		}
	}
}

func (s *Store) ensureEndpointLocked(key EndpointKey) *endpointState {
	state, ok := s.endpoints[key]
	if !ok {
		state = &endpointState{}
		s.endpoints[key] = state
	}
	return state
}

func (s *Store) releaseEndpointLoadLocked(key EndpointKey) {
	for requestID, state := range s.requests {
		pruned := pruneEndpointKey(state.endpointKeys, key)
		if len(pruned) == len(state.endpointKeys) {
			continue
		}
		state.endpointKeys = pruned
		s.requests[requestID] = state
	}
	delete(s.endpoints, key)
}

func (s *Store) expireRequests(at time.Time) {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireRequestsLocked(at)
}

func (s *Store) expireRequestsLocked(at time.Time) {
	if s.requestTimeout <= 0 || at.IsZero() {
		return
	}

	for requestID, state := range s.requests {
		if state.expiresAt.IsZero() || at.Before(state.expiresAt) {
			continue
		}
		delete(s.requests, requestID)
		s.subtract(state)
		s.logger.V(logutil.DEBUG).Info(
			"Expired stale active request and released inflight load",
			"store", s.name,
			"requestID", requestID,
			"selectedEndpoint", state.metaData.SelectedEndpoint,
			"startedAt", state.startedAt,
			"expiredAt", state.expiresAt,
		)
	}
}

func (s *Store) ensureExpiryLoopLocked() {
	if s.expiryRunning || s.requestTimeout <= 0 || s.requestSweep <= 0 {
		return
	}
	stopCh := make(chan struct{})
	doneCh := make(chan struct{})
	s.expiryStop = stopCh
	s.expiryDone = doneCh
	s.expiryRunning = true
	go s.runExpiryLoop(stopCh, doneCh)
}

func (s *Store) runExpiryLoop(stopCh <-chan struct{}, doneCh chan<- struct{}) {
	ticker := time.NewTicker(s.requestSweep)
	defer func() {
		ticker.Stop()
		close(doneCh)
	}()

	for {
		select {
		case <-ticker.C:
			s.expireRequests(s.now())
		case <-stopCh:
			return
		}
	}
}

func (s *Store) nextExpiry(from time.Time) time.Time {
	if s.requestTimeout <= 0 || from.IsZero() {
		return time.Time{}
	}
	return from.Add(s.requestTimeout)
}

// ensureAsyncFlusherLocked is called with s.mu held after the store is shared.
// The constructor may call it without locking because the store has not escaped.
func (s *Store) ensureAsyncFlusherLocked() {
	if s.closed || s.flushNotify != nil {
		return
	}
	notify := make(chan struct{}, 1)
	stop := make(chan struct{})
	done := make(chan struct{})
	s.flushNotify = notify
	s.flushStop = stop
	s.flushDone = done
	go s.runAsyncFlusher(notify, stop, done)
}

func (s *Store) signalFlushIfThresholdLocked() {
	if !s.shouldFlushCompletedLocked() {
		return
	}
	s.signalFlushLocked()
}

func (s *Store) signalFlushIfNeededLocked() {
	if !s.persistence.Enabled || len(s.completed) == 0 {
		return
	}
	if s.shouldFlushCompletedLocked() || s.persistence.FlushInterval > 0 {
		s.signalFlushLocked()
	}
}

func (s *Store) shouldFlushCompletedLocked() bool {
	if !s.persistence.Enabled || len(s.completed) == 0 {
		return false
	}
	if s.persistence.FlushThreshold > 0 && len(s.completed) >= s.persistence.FlushThreshold {
		return true
	}
	return len(s.completed) >= defaultCompletedBufferLimit
}

func (s *Store) signalFlushLocked() {
	if s.closed {
		return
	}
	s.ensureAsyncFlusherLocked()
	select {
	case s.flushNotify <- struct{}{}:
	default:
	}
}

func (s *Store) runAsyncFlusher(notify <-chan struct{}, stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)
	for {
		timedFlush := s.timedFlushChannel()
		if timedFlush == nil {
			if !s.waitAsyncFlushSignal(notify, stop) {
				return
			}
			continue
		}
		select {
		case _, ok := <-notify:
			if !ok {
				return
			}
			s.flushCompletedFromAsync(false)
		case _, ok := <-timedFlush:
			if !ok {
				continue
			}
			s.flushCompletedFromAsync(true)
		case <-stop:
			return
		}
	}
}

func (s *Store) waitAsyncFlushSignal(notify <-chan struct{}, stop <-chan struct{}) bool {
	select {
	case _, ok := <-notify:
		if !ok {
			return false
		}
		s.flushCompletedFromAsync(false)
	case <-stop:
		return false
	}
	return true
}

func (s *Store) flushCompletedFromAsync(force bool) {
	if err := s.flushCompleted(force); err != nil {
		s.logger.V(logutil.DEBUG).Error(err, "Async completed request metadata flush failed")
	}
}

func (s *Store) timedFlushChannel() <-chan time.Time {
	if s == nil {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed || !s.persistence.Enabled || s.persistence.FlushInterval <= 0 || len(s.completed) == 0 {
		return nil
	}
	delay := s.lastFlushAt.Add(s.persistence.FlushInterval).Sub(s.now())
	if delay < 0 {
		delay = 0
	}
	return time.After(delay)
}

func (s *Store) flushCompleted(force bool) error {
	s.flushMu.Lock()
	defer s.flushMu.Unlock()

	plan, err := s.prepareCompletedFlush(force)
	if err != nil || len(plan.batch) == 0 {
		return err
	}

	retryFrom, err := plan.writer.WriteCompletedBatch(plan.cfg, plan.batch)
	if err != nil {
		s.recordFlushFailure(err, retryRecords(plan.batch, retryFrom))
		return err
	}
	s.recordFlushSuccess()
	return nil
}

type completedFlushPlan struct {
	batch  []RequestMetaData
	cfg    PersistenceConfig
	writer completedBatchWriter
}

func (s *Store) prepareCompletedFlush(force bool) (completedFlushPlan, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.completed) == 0 {
		s.lastFlushErr = nil
		return completedFlushPlan{}, nil
	}
	if !s.persistence.Enabled {
		s.limitCompletedLocked()
		return completedFlushPlan{}, nil
	}
	if s.persistence.OutputPath == "" {
		err := fmt.Errorf("persistence output path is required when persistence is enabled")
		s.setLastFlushErrorLocked(err, "failed to flush completed request metadata")
		s.limitCompletedLocked()
		return completedFlushPlan{}, err
	}
	s.limitCompletedLocked()
	if !force && !s.shouldFlushCompletedLocked() {
		return completedFlushPlan{}, nil
	}
	s.lastFlushAt = s.now()

	writer := s.flushWriter
	if writer == nil {
		writer = fileCompletedBatchWriter{}
	}
	batch := s.completed
	s.completed = make([]RequestMetaData, 0, defaultFlushThreshold)
	return completedFlushPlan{batch: batch, cfg: s.persistence, writer: writer}, nil
}

func retryRecords(batch []RequestMetaData, retryFrom int) []RequestMetaData {
	if retryFrom < 0 {
		retryFrom = 0
	}
	if retryFrom > len(batch) {
		retryFrom = len(batch)
	}
	return batch[retryFrom:]
}

func (s *Store) recordFlushFailure(err error, retry []RequestMetaData) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(retry) > 0 {
		retained := make([]RequestMetaData, 0, len(retry)+len(s.completed))
		retained = append(retained, retry...)
		retained = append(retained, s.completed...)
		s.completed = retained
		s.limitCompletedLocked()
	}
	s.setLastFlushErrorLocked(err, "failed to flush completed request metadata")
}

func (s *Store) recordFlushSuccess() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastFlushErr = nil
}

// WriteCompletedBatch provides the exported WriteCompletedBatch operation.
func (fileCompletedBatchWriter) WriteCompletedBatch(cfg PersistenceConfig, records []RequestMetaData) (int, error) {
	if len(records) == 0 {
		return 0, nil
	}
	if err := os.MkdirAll(filepath.Dir(cfg.OutputPath), completedDirPerm); err != nil {
		return 0, fmt.Errorf("create completed metadata output directory: %w", err)
	}
	file, err := os.OpenFile(cfg.OutputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, completedFilePerm)
	if err != nil {
		return 0, fmt.Errorf("open completed metadata output file: %w", err)
	}
	defer file.Close()

	written, err := writeCompletedRecords(file, records)
	if err != nil {
		return written, err
	}
	// fsync so observation-mode capture survives a host crash. This runs in the
	// async flusher instead of the request hot path.
	if err := file.Sync(); err != nil {
		// The records have already been appended to the JSONL file. Retrying them
		// would duplicate training samples, so surface the durability error but do
		// not retain the batch for replay.
		return len(records), fmt.Errorf("fsync completed metadata output file: %w", err)
	}
	return len(records), nil
}

func writeCompletedRecords(writer io.Writer, records []RequestMetaData) (int, error) {
	encoder := json.NewEncoder(writer)
	for index, item := range records {
		if err := encoder.Encode(item); err != nil {
			return index, fmt.Errorf("encode completed request metadata: %w", err)
		}
	}
	return len(records), nil
}

func (s *Store) limitCompletedLocked() {
	overflow := len(s.completed) - defaultCompletedBufferLimit
	if overflow <= 0 {
		return
	}
	s.droppedCompleted += uint64(overflow)
	s.logger.V(logutil.DEFAULT).Info(
		"Dropped oldest completed request metadata to preserve bounded in-memory buffer",
		"store", s.name,
		"dropped", overflow,
		"bufferLimit", defaultCompletedBufferLimit,
		"retained", defaultCompletedBufferLimit,
		"totalDropped", s.droppedCompleted,
	)
	copy(s.completed, s.completed[overflow:])
	s.completed = s.completed[:defaultCompletedBufferLimit]
}

func (s *Store) setLastFlushErrorLocked(err error, message string) {
	s.lastFlushErr = err
	s.flushFailures++
	s.logger.V(logutil.DEFAULT).Error(err, message,
		"store", s.name,
		"outputPath", s.persistence.OutputPath,
		"pendingCompleted", len(s.completed),
		"flushFailures", s.flushFailures,
	)
}

func durationMillis(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

func trackedEndpointKeys(start RequestStart) ([]EndpointKey, EndpointKey, bool) {
	selectedKey, selectedOK := endpointKey(start.SelectedEndpoint)
	if !selectedOK {
		return nil, "", false
	}
	trackedKeys, trackedOK := endpointKeys(start.TrackedEndpoints)
	if !trackedOK {
		return nil, "", false
	}
	if len(trackedKeys) == 0 {
		trackedKeys = []EndpointKey{selectedKey}
	}
	return trackedKeys, selectedKey, true
}

func endpointKeys(endpoints []*fwkdl.EndpointMetadata) ([]EndpointKey, bool) {
	seen := make(map[string]struct{}, len(endpoints))
	keys := make([]EndpointKey, 0, len(endpoints))
	for _, metadata := range endpoints {
		key, ok := endpointKey(metadata)
		if !ok {
			return nil, false
		}
		if _, exists := seen[string(key)]; exists {
			continue
		}
		seen[string(key)] = struct{}{}
		keys = append(keys, key)
	}
	return keys, true
}

func pruneEndpointKey(keys []EndpointKey, target EndpointKey) []EndpointKey {
	if len(keys) == 0 {
		return keys
	}
	pruned := keys[:0]
	for _, key := range keys {
		if key == target {
			continue
		}
		pruned = append(pruned, key)
	}
	return pruned
}

func endpointKey(metadata *fwkdl.EndpointMetadata) (EndpointKey, bool) {
	if metadata == nil {
		return "", false
	}
	if metadata.PodName != "" {
		return EndpointKey(fmt.Sprintf("pod://%s/%s", metadata.NamespacedName.Namespace, metadata.PodName)), true
	}
	if metadata.Address == "" || metadata.Port == "" {
		return "", false
	}
	return EndpointKey(fmt.Sprintf("addr://%s:%s", metadata.Address, metadata.Port)), true
}

func normalizePersistenceConfig(cfg PersistenceConfig) PersistenceConfig {
	if cfg.Enabled && cfg.FlushThreshold <= 0 {
		cfg.FlushThreshold = defaultFlushThreshold
	}
	if cfg.Enabled && cfg.FlushInterval <= 0 {
		cfg.FlushInterval = defaultFlushInterval
	}
	if !cfg.Enabled {
		cfg.FlushThreshold = 0
		cfg.FlushInterval = 0
		cfg.OutputPath = ""
	}
	return cfg
}
