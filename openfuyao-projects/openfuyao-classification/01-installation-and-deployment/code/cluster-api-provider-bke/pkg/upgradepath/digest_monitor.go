/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package upgradepath

import (
	"context"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	upv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/oci"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	DefaultCheckInterval = 5 * time.Minute

	// DefaultUpgradePathCRName is the UpgradePath CR name when paths.yaml has no metadata.name.
	DefaultUpgradePathCRName = "openfuyao-upgrade-paths"

	// OCIDigestAnnotation stores the OCI image digest on the UpgradePath CR; the controller
	// copies it into status.lastDigest.
	OCIDigestAnnotation = annotation.UpgradePathOCIDigestAnnotationKey

	upgradePathImageName = "upgrade-path"
	ReleaseImageName     = "release-image"
	upgradePathImageTag  = "latest"
)

var digestMonitorLogger = log.With("name", "DigestMonitor")

// DefaultUpgradePathOCIRef returns the OCI reference monitored by DigestMonitor,
// built from the default fuyao image repository (see BkeConfig.ImageFuyaoRepo).
func DefaultUpgradePathOCIRef() string {
	repo := strings.TrimSuffix(bkeinit.DefaultFuyaoImageRepo, "/")
	return fmt.Sprintf("%s/%s:%s", repo, upgradePathImageName, upgradePathImageTag)
}

// DefaultReleaseImageOCIRef returns the OCI reference monitored by DigestMonitor,
// built from the default fuyao image repository (see BkeConfig.ImageFuyaoRepo).
func DefaultReleaseImageOCIRef(tag string) string {
	repo := strings.TrimSuffix(bkeinit.DefaultFuyaoImageRepo, "/")
	return fmt.Sprintf("%s/%s:%s", repo, ReleaseImageName, tag)
}

// ResolveImageOCIRefsFromRepo builds candidate OCI refs from imageRepo (domain first, then IP).
func ResolveImageOCIRefsFromRepo(repo confv1beta1.Repo, imageName, tag string) ([]string, error) {
	prefix := strings.Trim(repo.Prefix, "/")
	if prefix == "" {
		return nil, fmt.Errorf("imageRepo prefix is empty")
	}

	refs := make([]string, 0, 2)
	seen := make(map[string]struct{})

	appendRef := func(host string) {
		host = strings.TrimSpace(host)
		if host == "" {
			return
		}
		addr := host
		if repo.Port != "" && repo.Port != "443" {
			addr = net.JoinHostPort(host, repo.Port)
		}
		ref := fmt.Sprintf("%s/%s/%s:%s", addr, prefix, imageName, tag)
		if _, ok := seen[ref]; ok {
			return
		}
		seen[ref] = struct{}{}
		refs = append(refs, ref)
	}

	appendRef(repo.Domain)
	appendRef(repo.Ip)

	if len(refs) == 0 {
		return nil, fmt.Errorf("imageRepo domain and ip are empty")
	}
	return refs, nil
}

// ResolveImageOCIRefsFromCluster builds candidate OCI refs from a BKECluster imageRepo (domain first, then IP).
func ResolveImageOCIRefsFromCluster(bc *bkev1beta1.BKECluster, imageName, tag string) ([]string, error) {
	if bc.Spec.ClusterConfig == nil {
		return nil, fmt.Errorf("BKECluster %s/%s spec.clusterConfig is empty", bc.Namespace, bc.Name)
	}
	return ResolveImageOCIRefsFromRepo(bc.Spec.ClusterConfig.Cluster.ImageRepo, imageName, tag)
}

func resolveUpgradePathOCIRefs(ctx context.Context, k8sClient client.Client) ([]string, error) {
	data, err := clusterutil.GetBKEConfigCMData(ctx, k8sClient)
	if err != nil {
		return nil, fmt.Errorf("failed to get bke-config ConfigMap: %w", err)
	}

	repo, ok := RepoFromBKEConfigData(data)
	if !ok {
		return nil, fmt.Errorf("bke-config does not contain image repository settings")
	}

	refs, err := ResolveImageOCIRefsFromRepo(repo, upgradePathImageName, upgradePathImageTag)
	if err != nil {
		// Matches bkeadm prepareImageRepoConfig when only --onlineImage is set (empty prefix).
		if strings.Trim(repo.Prefix, "/") == "" {
			return []string{DefaultUpgradePathOCIRef()}, nil
		}
		return nil, fmt.Errorf("failed to resolve upgrade path OCI refs from bke-config: %w", err)
	}
	return refs, nil
}

type DigestMonitor struct {
	mu sync.RWMutex
	// ociRef is the full OCI image reference (e.g. "cr.openfuyao.cn/openfuyao/upgrade-path:latest").
	ociRef          string
	lastKnownDigest string
	lastCheckedAt   time.Time
	ociClient       *oci.Client
	k8sClient       client.Client
	checkInterval   time.Duration
	stopCh          chan struct{}
	started         bool
	failCount       int
	maxFailCount    int
}

func NewDigestMonitor(ociRef string, ociClient *oci.Client, k8sClient client.Client, interval time.Duration) *DigestMonitor {
	if interval <= 0 {
		interval = DefaultCheckInterval
	}
	return &DigestMonitor{
		ociRef:        ociRef,
		ociClient:     ociClient,
		k8sClient:     k8sClient,
		checkInterval: interval,
		stopCh:        make(chan struct{}),
		maxFailCount:  3,
	}
}

// Start launches the periodic digest check goroutine.
func (m *DigestMonitor) Start(ctx context.Context) error {
	m.mu.Lock()
	if m.started {
		m.mu.Unlock()
		return nil
	}
	m.stopCh = make(chan struct{})
	m.started = true
	m.mu.Unlock()

	if err := m.syncCRFromOCI(ctx); err != nil {
		m.recordSyncFailure("initial digest sync failed", err)
	}

	ticker := time.NewTicker(m.checkInterval)
	go m.run(ctx, ticker)

	return nil
}

func (m *DigestMonitor) run(ctx context.Context, ticker *time.Ticker) {
	defer ticker.Stop()
	logger := digestMonitorLogger.With("digestMonitor", "upgrade-path").With("ociRef", m.ociRef)
	for {
		select {
		case <-ticker.C:
			m.syncOnce(ctx)
		case <-m.stopCh:
			logger.Info("digest monitor stopped")
			return
		case <-ctx.Done():
			logger.Info("digest monitor context cancelled")
			return
		}
	}
}

func (m *DigestMonitor) syncOnce(ctx context.Context) {
	if err := m.syncCRFromOCI(ctx); err != nil {
		m.recordSyncFailure("periodic digest sync failed", err)
		return
	}
	m.mu.Lock()
	m.failCount = 0
	m.mu.Unlock()
}

func (m *DigestMonitor) recordSyncFailure(message string, err error) {
	logger := digestMonitorLogger.With("digestMonitor", "upgrade-path").With("ociRef", m.ociRef)
	logger.Error(message, err)
	m.mu.Lock()
	m.failCount++
	failCount := m.failCount
	m.mu.Unlock()
	if failCount >= m.maxFailCount {
		logger.Warn("consecutive failures reached threshold, sending alert")
	}
}

// Stop signals the periodic check goroutine to exit.
func (m *DigestMonitor) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.started {
		return
	}
	close(m.stopCh)
	m.started = false
}

// IsStarted returns whether the monitor goroutine has been launched.
func (m *DigestMonitor) IsStarted() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.started
}

func (m *DigestMonitor) syncCRFromOCI(ctx context.Context) error {
	logger := digestMonitorLogger.With("digestMonitor", "upgrade-path")

	ociRefs, err := resolveUpgradePathOCIRefs(ctx, m.k8sClient)
	if err != nil {
		return fmt.Errorf("failed to resolve upgrade path OCI refs: %w", err)
	}

	var lastErr error
	for _, ociRef := range ociRefs {
		logger.Info("pulling upgrade path OCI image", "ociRef", ociRef)

		result, err := m.trySyncOCIRef(ctx, logger, ociRef)
		if err != nil {
			return err
		}
		if result.done {
			return nil
		}
		if result.tryErr != nil {
			lastErr = result.tryErr
		}
	}

	if lastErr == nil {
		return fmt.Errorf("no available OCI refs to pull upgrade-path image")
	}
	return fmt.Errorf("all OCI refs failed: %w", lastErr)
}

type ociRefSyncResult struct {
	done   bool
	tryErr error
}

func (m *DigestMonitor) trySyncOCIRef(
	ctx context.Context,
	logger *log.Logger,
	ociRef string,
) (ociRefSyncResult, error) {
	currentDigest, err := m.ociClient.GetDigest(ctx, ociRef)
	if err != nil {
		logger.Warn("digest fetch failed, trying next OCI ref", "ociRef", ociRef, "error", err.Error())
		return ociRefSyncResult{tryErr: fmt.Errorf("failed to get digest from %s: %w", ociRef, err)}, nil
	}

	m.mu.RLock()
	digestUnchanged := currentDigest == m.lastKnownDigest
	m.mu.RUnlock()

	if digestUnchanged {
		exists, err := m.upgradePathExists(ctx)
		if err != nil {
			return ociRefSyncResult{}, err
		}
		if exists {
			m.mu.Lock()
			m.ociRef = ociRef
			m.mu.Unlock()
			logger.Info("OCI digest unchanged, skip pulling image", "ociRef", ociRef, "digest", currentDigest)
			return ociRefSyncResult{done: true}, nil
		}
		logger.Info("UpgradePath CR missing, resyncing from OCI despite unchanged digest",
			"ociRef", ociRef, "digest", currentDigest)
	}

	parsed, err := m.pullUpgradePathFromOCI(ctx, logger, ociRef)
	if err != nil {
		return ociRefSyncResult{tryErr: err}, nil
	}

	m.mu.Lock()
	m.ociRef = ociRef
	m.mu.Unlock()

	if err := m.applyToCluster(ctx, parsed, currentDigest); err != nil {
		return ociRefSyncResult{}, err
	}

	m.mu.Lock()
	m.lastKnownDigest = currentDigest
	m.lastCheckedAt = time.Now()
	m.mu.Unlock()

	logger.Info("OCI digest changed, synced UpgradePath CR", "ociRef", ociRef, "newDigest", currentDigest)
	return ociRefSyncResult{done: true}, nil
}

func (m *DigestMonitor) pullUpgradePathFromOCI(
	ctx context.Context,
	logger *log.Logger,
	ociRef string,
) (*upv1alpha1.UpgradePath, error) {
	img, err := m.ociClient.Pull(ctx, ociRef)
	if err != nil {
		logger.Warn("image pull failed, trying next OCI ref", "ociRef", ociRef, "error", err.Error())
		return nil, fmt.Errorf("failed to pull image from %s: %w", ociRef, err)
	}

	var parsed upv1alpha1.UpgradePath
	pathsLayer, err := img.GetLayerByPath("paths.yaml")
	if err != nil || pathsLayer == nil {
		logger.Warn("paths.yaml missing, trying next OCI ref", "ociRef", ociRef)
		return nil, fmt.Errorf("paths.yaml not found in OCI image %s: %w", ociRef, err)
	}
	if err := pathsLayer.UnmarshalYAML(&parsed); err != nil {
		logger.Warn("paths.yaml unmarshal failed, trying next OCI ref", "ociRef", ociRef, "error", err.Error())
		return nil, fmt.Errorf("failed to unmarshal paths.yaml from %s: %w", ociRef, err)
	}
	return &parsed, nil
}

func (m *DigestMonitor) upgradePathExists(ctx context.Context) (bool, error) {
	list := &upv1alpha1.UpgradePathList{}
	if err := m.k8sClient.List(ctx, list); err != nil {
		return false, fmt.Errorf("failed to list UpgradePath CRs: %w", err)
	}
	return len(list.Items) > 0, nil
}

func (m *DigestMonitor) applyToCluster(ctx context.Context, parsed *upv1alpha1.UpgradePath, digest string) error {
	list := &upv1alpha1.UpgradePathList{}
	if err := m.k8sClient.List(ctx, list); err != nil {
		return fmt.Errorf("failed to list UpgradePath CRs: %w", err)
	}

	if len(list.Items) == 0 {
		return m.createCR(ctx, parsed, digest)
	}
	return m.patchSpec(ctx, &list.Items[0], parsed, digest)
}

func (m *DigestMonitor) crName(parsed *upv1alpha1.UpgradePath) string {
	if parsed.Name != "" {
		return parsed.Name
	}
	return DefaultUpgradePathCRName
}

func (m *DigestMonitor) createCR(ctx context.Context, parsed *upv1alpha1.UpgradePath, digest string) error {
	name := m.crName(parsed)
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				OCIDigestAnnotation: digest,
			},
		},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths:    parsed.Spec.Paths,
			Versions: parsed.Spec.Versions,
		},
	}
	if err := m.k8sClient.Create(ctx, up); err != nil {
		return fmt.Errorf("failed to create UpgradePath CR %s: %w", name, err)
	}
	return nil
}

func (m *DigestMonitor) patchSpec(ctx context.Context, existing *upv1alpha1.UpgradePath, parsed *upv1alpha1.UpgradePath, digest string) error {
	orig := existing.DeepCopy()

	existing.Spec.Paths = parsed.Spec.Paths
	existing.Spec.Versions = parsed.Spec.Versions

	if existing.Annotations == nil {
		existing.Annotations = make(map[string]string)
	}
	existing.Annotations[OCIDigestAnnotation] = digest

	if err := m.k8sClient.Patch(ctx, existing, client.MergeFrom(orig)); err != nil {
		return fmt.Errorf("failed to patch UpgradePath CR %s: %w", existing.Name, err)
	}
	return nil
}

// LastDigest returns the most recently observed OCI image digest.
func (m *DigestMonitor) LastDigest() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.lastKnownDigest
}

// LastCheckedAt returns the timestamp of the most recent successful digest check.
func (m *DigestMonitor) LastCheckedAt() *metav1.Time {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.lastCheckedAt.IsZero() {
		return nil
	}
	t := metav1.NewTime(m.lastCheckedAt)
	return &t
}

// FailCount returns the number of consecutive sync failures since the last success.
func (m *DigestMonitor) FailCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.failCount
}

// DigestMonitorRunnable runs DigestMonitor outside UpgradePath reconciliation.
type DigestMonitorRunnable struct {
	monitor *DigestMonitor
}

// NewDigestMonitorRunnable builds a runnable that monitors the fixed upgrade-path OCI image.
func NewDigestMonitorRunnable(k8sClient client.Client, ociClient *oci.Client, interval time.Duration) *DigestMonitorRunnable {
	return &DigestMonitorRunnable{
		monitor: NewDigestMonitor("", ociClient, k8sClient, interval),
	}
}

// Start implements manager.Runnable.
func (r *DigestMonitorRunnable) Start(ctx context.Context) error {
	logger := digestMonitorLogger.With("name", "digest-monitor-runnable")
	logger.Info("starting upgrade path digest monitor", "ociRef", r.monitor.ociRef)
	return r.monitor.Start(ctx)
}

// NeedLeaderElection ensures only the elected leader polls OCI.
func (r *DigestMonitorRunnable) NeedLeaderElection() bool {
	return true
}
