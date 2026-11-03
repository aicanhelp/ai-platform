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

package discovery

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"gitcode.com/openFuyao/cache-indexer/pkg/config"
)

type manager struct {
	cfg        config.DiscoveryConfig
	log        logr.Logger
	client     kubernetes.Interface
	reconciler Reconciler
	registry   Registry
	segments   *segmentsClient

	mu      sync.RWMutex
	current *Snapshot
	version uint64

	cancel  context.CancelFunc
	done    chan struct{}
	stopped atomic.Bool
}

// NewManager builds a Manager bound to the given client. The
// reconciler and registry are invoked on every successful refresh.
func NewManager(
	cfg config.DiscoveryConfig,
	log logr.Logger,
	client kubernetes.Interface,
	reconciler Reconciler,
	registry Registry,
) Manager {
	cfg = normalizeDiscoveryConfig(cfg)
	return &manager{
		cfg:        cfg,
		log:        log,
		client:     client,
		reconciler: reconciler,
		registry:   registry,
		segments:   newSegmentsClient(cfg.SegmentsFetchTimeout),
		done:       make(chan struct{}),
	}
}

// Start launches the polling loop. It returns immediately; the loop runs
// until ctx is cancelled or Stop() is called.
func (m *manager) Start(ctx context.Context) error {
	if m.cfg.Namespace == "" {
		return fmt.Errorf("discovery: namespace is empty (set POD_NAMESPACE)")
	}
	if m.cfg.RefreshInterval <= 0 {
		return fmt.Errorf("discovery: refresh interval must be > 0")
	}

	loopCtx, cancel := context.WithCancel(ctx)
	m.mu.Lock()
	m.cancel = cancel
	m.mu.Unlock()

	m.log.Info("discovery started",
		"namespace", m.cfg.Namespace,
		"refreshInterval", m.cfg.RefreshInterval.String())

	// started ensures Stop() can wait on m.done after the loop is scheduled.
	// This avoids a Start() / Stop() race before the goroutine runs.
	started := make(chan struct{})
	go func() {
		close(started)
		m.loop(loopCtx)
	}()
	<-started
	return nil
}

func (m *manager) loop(ctx context.Context) {
	defer close(m.done)

	// First tick fires immediately so logs show activity right after start.
	t := time.NewTimer(0)
	defer t.Stop()

	for {
		select {
		case <-ctx.Done():
			if err := ctx.Err(); err != nil {
				m.log.Info("discovery loop stopped", "reason", err.Error())
			} else {
				m.log.Info("discovery loop stopped")
			}
			return
		case <-t.C:
			m.refreshOnce(ctx)
			t.Reset(m.cfg.RefreshInterval)
		}
	}
}

// Stop cancels the polling loop and waits for it to exit.
func (m *manager) Stop() error {
	if m.stopped.Swap(true) {
		return nil
	}
	m.mu.RLock()
	cancel := m.cancel
	m.mu.RUnlock()
	if cancel == nil {
		// Stop() called before Start(); nothing to wait for.
		return nil
	}
	cancel()
	<-m.done
	return nil
}

// Snapshot returns the most recent successful snapshot or nil if none yet.
func (m *manager) Snapshot() *Snapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.current
}

func (m *manager) refreshOnce(ctx context.Context) {
	pods, err := m.client.CoreV1().Pods(m.cfg.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		m.log.Error(err, "discovery list pods failed",
			"namespace", m.cfg.Namespace)
		return
	}

	next := m.buildSnapshot(pods.Items)

	// Resolve TransportEndpoint from Master segments for L3 routing.
	// Missing endpoints only affect L3 hit rate; L1 (ZMQ) is unaffected.
	// On vllm-ascend the AscendStoreConnector can auto-allocate runtime ports.
	if m.segments != nil && next.MooncakeMasterPod != nil && len(next.VLLMPods) > 0 {
		fctx, cancel := context.WithTimeout(ctx, m.cfg.SegmentsFetchTimeout)
		segs, ferr := m.segments.fetch(fctx, next.MooncakeMasterPod.PodIP, next.MooncakeMasterPod.HTTPPort)
		cancel()
		if ferr != nil {
			m.log.Error(ferr, "mooncake master unreachable, pods lack mooncake client, L3 skipped, L1 normal",
				"masterIP", next.MooncakeMasterPod.PodIP,
				"httpPort", next.MooncakeMasterPod.HTTPPort)
		} else {
			missing := applySegmentsToSnapshot(next, segs)
			if len(missing) > 0 {
				m.log.Error(fmt.Errorf("pods lack mooncake client"),
					"pods lack mooncake client, L3 skipped for these pods, L1 normal",
					"missingPodIPs", missing)
			}
		}
	}

	m.mu.Lock()
	prev := m.current
	m.current = next
	m.mu.Unlock()

	// Reconciler logs only snapshot changes; list failures are logged above.

	// Update registry before reconcile so callbacks see the new snapshot.
	// Skip reconcile on registry failure to avoid query/subscriber divergence.
	if m.registry != nil {
		if err := m.registry.Update(next); err != nil {
			m.log.Error(err, "registry update failed; skipping reconcile to avoid divergence",
				"version", next.Version)
			return
		}
	}
	if m.reconciler != nil {
		if err := m.reconciler.Reconcile(prev, next); err != nil {
			m.log.Error(err, "reconcile failed")
		}
	}
}

func (m *manager) buildSnapshot(items []corev1.Pod) *Snapshot {
	now := time.Now()
	m.version++
	selector := m.cfg.Selector

	snap := &Snapshot{
		Namespace:   m.cfg.Namespace,
		Version:     m.version,
		RefreshedAt: now,
		VLLMPods:    make(map[string]*VLLMPodEndpoint),
	}

	// Master candidates may be multiple; we pick the lexicographically
	// smallest ready Pod to keep the choice deterministic.
	var masterCandidates []*MooncakeMasterEndpoint

	for i := range items {
		pod := &items[i]
		if !podReady(pod) || pod.Status.Phase != corev1.PodRunning {
			continue
		}
		switch {
		case matchesVLLMPod(pod, selector):
			ep := m.buildVLLMEndpoint(pod, now)
			if ep != nil {
				snap.VLLMPods[ep.PodIP] = ep
			}
		case matchesMooncakeMasterPod(pod, selector):
			ep := m.buildMooncakeMasterEndpoint(pod, now)
			if ep != nil {
				masterCandidates = append(masterCandidates, ep)
			}
		default:
			continue
		}
	}

	if len(masterCandidates) > 0 {
		sort.SliceStable(masterCandidates, func(i, j int) bool {
			return masterCandidates[i].Identity.Name < masterCandidates[j].Identity.Name
		})
		snap.MooncakeMasterPod = masterCandidates[0]
	}
	return snap
}

func isVLLMPod(p *corev1.Pod) bool {
	return matchesVLLMPod(p, defaultDiscoverySelectorConfig())
}

func normalizeDiscoveryConfig(cfg config.DiscoveryConfig) config.DiscoveryConfig {
	if cfg.RefreshInterval <= 0 {
		cfg.RefreshInterval = config.DefaultDiscoveryRefreshInterval
	}
	if cfg.SegmentsFetchTimeout <= 0 {
		cfg.SegmentsFetchTimeout = config.DefaultDiscoverySegmentsFetchTimeout
	}
	cfg.Selector = normalizeDiscoverySelectorConfig(cfg.Selector)
	if cfg.VLLM.ZMQPortName == "" {
		cfg.VLLM.ZMQPortName = config.DefaultVLLMZMQPortName
	}
	if cfg.MooncakeMaster.HTTPPortName == "" {
		cfg.MooncakeMaster.HTTPPortName = config.DefaultMooncakeHTTPPortName
	}
	if cfg.MooncakeMaster.RPCPortName == "" {
		cfg.MooncakeMaster.RPCPortName = config.DefaultMooncakeRPCPortName
	}
	if cfg.MooncakeMaster.HTTPPort <= 0 {
		cfg.MooncakeMaster.HTTPPort = config.DefaultMooncakeMasterHTTPPort
	}
	return cfg
}

func defaultDiscoverySelectorConfig() config.DiscoverySelectorConfig {
	return normalizeDiscoverySelectorConfig(config.DiscoverySelectorConfig{})
}

func normalizeDiscoverySelectorConfig(selector config.DiscoverySelectorConfig) config.DiscoverySelectorConfig {
	if selector.EngineLabelKey == "" {
		selector.EngineLabelKey = config.DefaultDiscoveryEngineLabelKey
	}
	if selector.EngineValue == "" {
		selector.EngineValue = config.DefaultDiscoveryEngineValue
	}
	if selector.PDRoleLabelKey == "" {
		selector.PDRoleLabelKey = config.DefaultDiscoveryPDRoleLabelKey
	}
	if len(selector.PDRoleValues) == 0 {
		selector.PDRoleValues = []string{"prefill", "decode", "aggregate"}
	}
	if selector.KVManagerLabelKey == "" {
		selector.KVManagerLabelKey = config.DefaultDiscoveryKVManagerLabelKey
	}
	if selector.KVManagerValue == "" {
		selector.KVManagerValue = config.DefaultDiscoveryKVManagerValue
	}
	return selector
}

func matchesVLLMPod(p *corev1.Pod, selector config.DiscoverySelectorConfig) bool {
	if p.Labels[selector.EngineLabelKey] != selector.EngineValue {
		return false
	}
	pdRole := p.Labels[selector.PDRoleLabelKey]
	for _, value := range selector.PDRoleValues {
		if pdRole == value {
			return true
		}
	}
	return false
}

func matchesMooncakeMasterPod(p *corev1.Pod, selector config.DiscoverySelectorConfig) bool {
	return p.Labels[selector.KVManagerLabelKey] == selector.KVManagerValue
}

func podReady(p *corev1.Pod) bool {
	for _, c := range p.Status.Conditions {
		if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func (m *manager) buildVLLMEndpoint(p *corev1.Pod, now time.Time) *VLLMPodEndpoint {
	if p.Status.PodIP == "" {
		return nil
	}
	zmqPort := portByName(p, m.cfg.VLLM.ZMQPortName)

	return &VLLMPodEndpoint{
		Identity:       identityOf(p),
		PodIP:          p.Status.PodIP,
		ZMQPort:        zmqPort,
		Labels:         copyLabels(p.Labels),
		MooncakeClient: MooncakeClientEndpoint{Resolved: false, Source: "unresolved"},
		Ready:          true,
		LastSeenAt:     now,
	}
}

func (m *manager) buildMooncakeMasterEndpoint(p *corev1.Pod, now time.Time) *MooncakeMasterEndpoint {
	if p.Status.PodIP == "" {
		return nil
	}
	httpPort := portByName(p, m.cfg.MooncakeMaster.HTTPPortName)
	if httpPort == 0 {
		httpPort = m.cfg.MooncakeMaster.HTTPPort
	}
	rpcPort := portByName(p, m.cfg.MooncakeMaster.RPCPortName)
	if rpcPort == 0 {
		rpcPort = m.cfg.MooncakeMaster.RPCPort
	}
	return &MooncakeMasterEndpoint{
		Identity:   identityOf(p),
		PodIP:      p.Status.PodIP,
		HTTPPort:   httpPort,
		RPCPort:    rpcPort,
		Labels:     copyLabels(p.Labels),
		Ready:      true,
		LastSeenAt: now,
	}
}

func splitHostPort(v string) (string, int32) {
	// v can be host:port or host:port:npu_<id>; we only need host + first port.
	host, port := "", int32(0)
	first := -1
	for i := 0; i < len(v); i++ {
		if v[i] == ':' {
			first = i
			break
		}
	}
	if first < 0 {
		return v, 0
	}
	host = v[:first]
	rest := v[first+1:]
	end := len(rest)
	for i := 0; i < len(rest); i++ {
		if rest[i] == ':' {
			end = i
			break
		}
	}
	if n, ok := atoi32(rest[:end]); ok {
		port = n
	}
	return host, port
}

func atoi32(s string) (int32, bool) {
	if s == "" {
		return 0, false
	}
	const decimalBase = 10
	n := int32(0)
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*decimalBase + int32(c-'0')
	}
	return n, true
}

func portByName(p *corev1.Pod, name string) int32 {
	if name == "" {
		return 0
	}
	for ci := range p.Spec.Containers {
		for _, port := range p.Spec.Containers[ci].Ports {
			if port.Name == name {
				return port.ContainerPort
			}
		}
	}
	return 0
}

func identityOf(p *corev1.Pod) PodIdentity {
	return PodIdentity{
		Namespace:       p.Namespace,
		Name:            p.Name,
		UID:             string(p.UID),
		ResourceVersion: p.ResourceVersion,
	}
}

func copyLabels(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
