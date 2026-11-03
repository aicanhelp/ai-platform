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

// Package upgradepath provides an in-memory upgrade path graph and query service.
package upgradepath

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/blang/semver/v4"

	upv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

var (
	ErrNoPath          = errors.New("no upgrade path")
	ErrVersionNotFound = errors.New("version not found in upgrade path graph")
	ErrCycleDetected   = errors.New("cycle detected in upgrade path graph")
	ErrDuplicateEdge   = errors.New("duplicate upgrade path")
	ErrSelfLoop        = errors.New("self loop detected")
	ErrEmptyCheckName  = errors.New("empty check step name")
	ErrEmptyFrom       = errors.New("path from must not be empty")
	ErrEmptyTo         = errors.New("path to must not be empty")
)

// Finder provides read-only query operations on the upgrade path graph.
// It is consumed by the controller and other components that need to query
// paths and version metadata without modifying the graph.
type Finder interface {
	// FindPath returns the shortest valid upgrade path (sequence of UpgradePathRules)
	// from one version to another, skipping blocked edges. Returns ErrNoPath or
	// ErrVersionNotFound on failure.
	FindPath(from, to string) ([]upv1alpha1.UpgradePathRule, error)
	// HasVersion checks whether a version exists as a node in the graph.
	HasVersion(version string) bool
	// AllVersions returns all version strings present in the graph, sorted by semver.
	AllVersions() []string
	// PathCount returns the total number of upgrade path rules in the graph.
	PathCount() int
	// Digest returns the SHA-256 digest of the spec that was loaded, used to detect
	// whether the graph data has changed since the last load.
	Digest() string
	// GetInstallableVersions returns all versions marked as Installable and not
	// Deprecated, sorted by semver. These are versions that can be used for fresh
	// cluster installations.
	GetInstallableVersions() []string
	// GetUpgradeableVersions returns all versions reachable from currentVersion via
	// non-blocked edges, including both direct (single-hop) and multi-hop targets.
	// Uses BFS to enumerate all reachable versions, sorted by semver.
	GetUpgradeableVersions(currentVersion string) []string
}

type Loader interface {
	Finder
	Load(paths []upv1alpha1.UpgradePathRule, versions []upv1alpha1.VersionEntry, digest string) error
	Clear()
}

type VersionNode struct {
	Version     string
	Installable bool
	Deprecated  bool
}

type Service struct {
	mu sync.RWMutex
	// adj is the adjacency list: from-version → sorted list of outgoing upgrade edges.
	adj map[string][]upv1alpha1.UpgradePathRule
	// nodes maps version strings to VersionNode metadata (installable, deprecated, etc.).
	nodes map[string]*VersionNode
	// versions tracks which version strings exist in the graph (as nodes), derived
	// from both edge endpoints and explicit VersionEntry declarations.
	versions map[string]struct{}
	digest   string
	// paths counts the total number of upgrade path rules currently loaded.
	paths int
}

func NewService() *Service {
	return &Service{
		adj:      make(map[string][]upv1alpha1.UpgradePathRule),
		nodes:    make(map[string]*VersionNode),
		versions: make(map[string]struct{}),
	}
}

// Load validates path rules (no self-loops, no duplicates, non-empty From/To and check names),
// builds the in-memory adjacency list and version nodes, detects cycles, sorts edges by target
// version, and atomically replaces the existing graph data.
func (s *Service) Load(paths []upv1alpha1.UpgradePathRule, versions []upv1alpha1.VersionEntry, digest string) error {
	if err := ValidateRules(paths); err != nil {
		return err
	}
	if err := DetectCycle(paths); err != nil {
		return err
	}

	adj, nodeSet := buildGraph(paths)
	nodes := buildNodes(versions, nodeSet)

	for from := range adj {
		sort.SliceStable(adj[from], func(i, j int) bool {
			return compareVersion(adj[from][i].To, adj[from][j].To) < 0
		})
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.adj = adj
	s.nodes = nodes
	s.versions = nodeSet
	s.digest = digest
	s.paths = len(paths)
	return nil
}

// Clear resets the Service to an empty state, removing all graph data.
// Called when all UpgradePath CRs have been deleted.
func (s *Service) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.adj = make(map[string][]upv1alpha1.UpgradePathRule)
	s.nodes = make(map[string]*VersionNode)
	s.versions = make(map[string]struct{})
	s.digest = ""
	s.paths = 0
}

// FindPath returns the shortest valid upgrade path from "from" to "to" using BFS,
// skipping blocked edges. Returns nil if from==to. Returns ErrVersionNotFound if
// either version is missing from the graph, or ErrNoPath if no reachable path exists.
type pathQueueItem struct {
	version string
	path    []upv1alpha1.UpgradePathRule
}

func (s *Service) FindPath(from, to string) ([]upv1alpha1.UpgradePathRule, error) {
	if from == to {
		return nil, nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if _, ok := s.versions[from]; !ok {
		return nil, fmt.Errorf("%w: %s", ErrVersionNotFound, from)
	}
	if _, ok := s.versions[to]; !ok {
		return nil, fmt.Errorf("%w: %s", ErrVersionNotFound, to)
	}

	visited := map[string]struct{}{from: {}}
	currentLevel := []pathQueueItem{{version: from}}

	for len(currentLevel) > 0 {
		nextLevel, path, found := s.nextPathLevel(currentLevel, to, visited)
		if found {
			return path, nil
		}
		currentLevel = nextLevel
	}

	return nil, fmt.Errorf("%w: %s -> %s", ErrNoPath, from, to)
}

func (s *Service) nextPathLevel(currentLevel []pathQueueItem, target string,
	visited map[string]struct{}) ([]pathQueueItem, []upv1alpha1.UpgradePathRule, bool) {
	if visited == nil {
		visited = make(map[string]struct{})
	}
	var nextLevel []pathQueueItem
	for _, current := range currentLevel {
		for _, edge := range s.adj[current.version] {
			if edge.Blocked {
				continue
			}
			if _, ok := visited[edge.To]; ok {
				continue
			}
			nextPath := append(copyRules(current.path), copyRule(edge))
			if edge.To == target {
				return nil, nextPath, true
			}
			visited[edge.To] = struct{}{}
			nextLevel = append(nextLevel, pathQueueItem{version: edge.To, path: nextPath})
		}
	}
	return nextLevel, nil, false
}

// HasVersion returns true if the given version exists as a node in the graph.
func (s *Service) HasVersion(version string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.versions[version]
	return ok
}

// AllVersions returns all version strings present in the graph, sorted by semver.
func (s *Service) AllVersions() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	versions := make([]string, 0, len(s.versions))
	for version := range s.versions {
		versions = append(versions, version)
	}
	sort.SliceStable(versions, func(i, j int) bool {
		return compareVersion(versions[i], versions[j]) < 0
	})
	return versions
}

// PathCount returns the total number of upgrade path rules currently loaded.
func (s *Service) PathCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.paths
}

// Digest returns the SHA-256 digest of the spec that was loaded, used to track
// whether the graph data is up-to-date with the CR spec.
func (s *Service) Digest() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.digest
}

// GetInstallableVersions returns all versions that are marked Installable and not
// Deprecated, sorted by semver. These are versions available for fresh cluster installs.
func (s *Service) GetInstallableVersions() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var result []string
	for v, n := range s.nodes {
		if n.Installable && !n.Deprecated {
			result = append(result, v)
		}
	}
	sort.SliceStable(result, func(i, j int) bool {
		return compareVersion(result[i], result[j]) < 0
	})
	return result
}

// GetUpgradeableVersions returns all versions reachable from currentVersion via
// non-blocked edges, including both direct (single-hop) and multi-hop targets.
// Uses BFS to enumerate all reachable versions, sorted by semver.
// Returns an empty slice if currentVersion is not in the graph.
func (s *Service) GetUpgradeableVersions(currentVersion string) []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if _, ok := s.versions[currentVersion]; !ok {
		return nil
	}

	visited := map[string]struct{}{currentVersion: {}}
	queue := []string{currentVersion}

	for len(queue) > 0 {
		queue = s.nextUpgradeableVersions(queue, visited)
	}

	var targets []string
	for v := range visited {
		if v != currentVersion {
			targets = append(targets, v)
		}
	}
	sort.SliceStable(targets, func(i, j int) bool {
		return compareVersion(targets[i], targets[j]) < 0
	})
	return targets
}

func (s *Service) nextUpgradeableVersions(queue []string, visited map[string]struct{}) []string {
	if visited == nil {
		visited = make(map[string]struct{})
	}
	var nextQueue []string
	for _, version := range queue {
		for _, edge := range s.adj[version] {
			if edge.Blocked {
				continue
			}
			if _, ok := visited[edge.To]; ok {
				continue
			}
			visited[edge.To] = struct{}{}
			nextQueue = append(nextQueue, edge.To)
		}
	}
	return nextQueue
}

// ValidateRules checks upgrade path rules for structural validity: non-empty From/To,
// no self-loops, no duplicate edges, and non-empty CheckStep names.
func ValidateRules(paths []upv1alpha1.UpgradePathRule) error {
	seen := make(map[string]struct{}, len(paths))
	for i, path := range paths {
		if path.From == "" {
			return fmt.Errorf("%w: path[%d]", ErrEmptyFrom, i)
		}
		if path.To == "" {
			return fmt.Errorf("%w: path[%d]", ErrEmptyTo, i)
		}
		if path.From == path.To {
			return fmt.Errorf("%w: path[%d] version %q", ErrSelfLoop, i, path.From)
		}
		for j, step := range path.PreCheck {
			if step.Name == "" {
				return fmt.Errorf("%w: path[%d].preCheck[%d]", ErrEmptyCheckName, i, j)
			}
		}
		for j, step := range path.PostCheck {
			if step.Name == "" {
				return fmt.Errorf("%w: path[%d].postCheck[%d]", ErrEmptyCheckName, i, j)
			}
		}

		key := path.From + "\x00" + path.To
		if _, ok := seen[key]; ok {
			return fmt.Errorf("%w: %s -> %s", ErrDuplicateEdge, path.From, path.To)
		}
		seen[key] = struct{}{}
	}
	return nil
}

// DetectCycle checks whether the directed graph formed by the upgrade path rules
// contains any cycle. Uses a three-color DFS (white=0, gray=1, black=2) to detect
// back edges. Returns ErrCycleDetected with the cycle path if a cycle is found.
func DetectCycle(paths []upv1alpha1.UpgradePathRule) error {
	adj, versions := buildGraph(paths)
	color := make(map[string]int, len(versions))
	stack := make([]string, 0, len(versions))

	var visit func(string) error
	visit = func(version string) error {
		color[version] = 1
		stack = append(stack, version)
		defer func() {
			stack = stack[:len(stack)-1]
			color[version] = 2
		}()

		for _, edge := range adj[version] {
			switch color[edge.To] {
			case 0:
				if err := visit(edge.To); err != nil {
					return err
				}
			case 1:
				return fmt.Errorf("%w: %s", ErrCycleDetected, formatCycle(stack, edge.To))
			default:
			}
		}
		return nil
	}

	versionsList := make([]string, 0, len(versions))
	for version := range versions {
		versionsList = append(versionsList, version)
	}
	sort.Strings(versionsList)

	for _, version := range versionsList {
		if color[version] == 0 {
			if err := visit(version); err != nil {
				return err
			}
		}
	}
	return nil
}

// buildGraph constructs an adjacency list and a version set from the given path rules.
// Each rule is deep-copied before insertion to prevent aliasing with the original slice.
// The adjacency list maps each From-version to its outgoing edges, and ensures every
// To-version also has an entry (possibly empty) so that leaf nodes are tracked.
func buildGraph(paths []upv1alpha1.UpgradePathRule) (map[string][]upv1alpha1.UpgradePathRule, map[string]struct{}) {
	adj := make(map[string][]upv1alpha1.UpgradePathRule)
	versions := make(map[string]struct{})

	for _, path := range paths {
		pathCopy := copyRule(path)
		adj[path.From] = append(adj[path.From], pathCopy)
		if _, ok := adj[path.To]; !ok {
			adj[path.To] = nil
		}
		versions[path.From] = struct{}{}
		versions[path.To] = struct{}{}
	}
	return adj, versions
}

// buildNodes merges explicit VersionEntry declarations with version strings from the
// graph edge set. Entries from the versions list provide rich metadata (Installable,
// Deprecated); versions that appear only in edges get default nodes with
// Installable=false and Deprecated=false.
func buildNodes(entries []upv1alpha1.VersionEntry, nodeSet map[string]struct{}) map[string]*VersionNode {
	nodes := make(map[string]*VersionNode)

	for _, entry := range entries {
		nodes[entry.Version] = &VersionNode{
			Version:     entry.Version,
			Installable: entry.Installable,
			Deprecated:  entry.Deprecated,
		}
	}

	for v := range nodeSet {
		if _, ok := nodes[v]; !ok {
			nodes[v] = &VersionNode{
				Version:     v,
				Installable: false,
				Deprecated:  false,
			}
		}
	}

	return nodes
}

// copyRules deep-copies a slice of UpgradePathRules, including their PreCheck/PostCheck slices.
func copyRules(paths []upv1alpha1.UpgradePathRule) []upv1alpha1.UpgradePathRule {
	if len(paths) == 0 {
		return nil
	}
	out := make([]upv1alpha1.UpgradePathRule, len(paths))
	for i := range paths {
		out[i] = copyRule(paths[i])
	}
	return out
}

// copyRule deep-copies a single UpgradePathRule, including its PreCheck and PostCheck slices,
// to prevent shared slice header aliasing when the same rule appears in multiple contexts.
func copyRule(path upv1alpha1.UpgradePathRule) upv1alpha1.UpgradePathRule {
	path.PreCheck = append([]upv1alpha1.CheckStep(nil), path.PreCheck...)
	path.PostCheck = append([]upv1alpha1.CheckStep(nil), path.PostCheck...)
	return path
}

// compareVersion compares two version strings by semver. Falls back to lexical
// comparison if either string cannot be parsed as semver. Returns -1, 0, or 1.
func compareVersion(left, right string) int {
	leftVersion, leftErr := semver.ParseTolerant(left)
	rightVersion, rightErr := semver.ParseTolerant(right)
	if leftErr == nil && rightErr == nil {
		return leftVersion.Compare(rightVersion)
	}
	if left < right {
		return -1
	}
	if left > right {
		return 1
	}
	return 0
}

// formatCycle extracts the cycle portion from a DFS stack by finding the position
// where the target version first appears, and formats it as a readable string.
func formatCycle(stack []string, target string) string {
	start := 0
	for i, version := range stack {
		if version == target {
			start = i
			break
		}
	}

	cycle := append([]string(nil), stack[start:]...)
	cycle = append(cycle, target)
	return fmt.Sprintf("%v", cycle)
}
