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

// Package l3 owns the per-Mooncake-client KVCache index.
// It uses L1's dual-index pattern and keys by TransportEndpoint.
package l3

import (
	"sync"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

// Indexer is the read/write contract for the L3 plane.
type Indexer interface {
	// ApplyReplicaSnapshot replaces one endpoint's records for a replica type.
	ApplyReplicaSnapshot(
		modelCtx apis.ModelContext,
		endpoint apis.TransportEndpoint,
		records []apis.L3ReplicaRecord,
	) error
	// RemoveReplicaKeys removes a set of keys for one endpoint.
	RemoveReplicaKeys(
		modelCtx apis.ModelContext,
		endpoint apis.TransportEndpoint,
		keys []apis.BlockHash,
	) error
	// MatchPrefix returns the longest matched key prefix for one endpoint.
	MatchPrefix(
		modelCtx apis.ModelContext,
		endpoint apis.TransportEndpoint,
		keys []apis.BlockHash,
	) (*apis.MatchResult, error)
}

type contextData struct {
	mu          sync.RWMutex
	keyIndex    map[apis.BlockHash]*apis.L3KeyEntry
	serverIndex map[apis.TransportEndpoint]map[apis.BlockHash]*apis.L3ServerState
}

func newContextData() *contextData {
	return &contextData{
		keyIndex:    map[apis.BlockHash]*apis.L3KeyEntry{},
		serverIndex: map[apis.TransportEndpoint]map[apis.BlockHash]*apis.L3ServerState{},
	}
}

type indexer struct {
	log logr.Logger

	mu        sync.RWMutex
	byContext map[apis.ModelContext]*contextData
}

// New builds a real L3 indexer.
func New(log logr.Logger) Indexer {
	idx := &indexer{log: log}
	idx.byContext = make(map[apis.ModelContext]*contextData)
	return idx
}

func (i *indexer) getOrCreateContext(mc apis.ModelContext) *contextData {
	return i.lookupContext(mc, true)
}

func (i *indexer) getContext(mc apis.ModelContext) *contextData {
	return i.lookupContext(mc, false)
}

func (i *indexer) lookupContext(mc apis.ModelContext, create bool) *contextData {
	i.mu.RLock()
	cd := i.byContext[mc]
	i.mu.RUnlock()
	if cd != nil || !create {
		return cd
	}

	i.mu.Lock()
	defer i.mu.Unlock()

	if cd = i.byContext[mc]; cd == nil {
		cd = newContextData()
		i.byContext[mc] = cd
	}
	return cd
}

func newMatchResult(mc apis.ModelContext, keys []apis.BlockHash, matched int) *apis.MatchResult {
	res := &apis.MatchResult{
		Keys:          keys,
		MatchedBlocks: matched,
	}
	if mc.BlockSize > 0 {
		res.MatchedTokens = matched * int(mc.BlockSize)
	}
	return res
}

func countMatchedPrefix(serverMap map[apis.BlockHash]*apis.L3ServerState, keys []apis.BlockHash) int {
	matched := 0
	for _, k := range keys {
		if _, ok := serverMap[k]; !ok {
			break
		}
		matched++
	}
	return matched
}

// ApplyReplicaSnapshot merges L3ReplicaRecord rows grouped by endpoint.
// Empty keys/endpoints are skipped defensively.
func (i *indexer) ApplyReplicaSnapshot(
	mc apis.ModelContext,
	endpoint apis.TransportEndpoint,
	records []apis.L3ReplicaRecord,
) error {
	if len(records) == 0 {
		return nil
	}
	cd := i.getOrCreateContext(mc)
	cd.mu.Lock()
	defer cd.mu.Unlock()

	for _, rec := range records {
		// Per-record endpoint lets one call fan out across Mooncake clients.
		ep := rec.TransportEndpoint
		if ep == "" {
			ep = endpoint
		}
		if ep == "" || rec.BlockKey == "" {
			continue
		}

		serverMap := cd.serverIndex[ep]
		if serverMap == nil {
			serverMap = map[apis.BlockHash]*apis.L3ServerState{}
			cd.serverIndex[ep] = serverMap
		}

		state := serverMap[rec.BlockKey]
		if state == nil {
			state = &apis.L3ServerState{
				ReplicaTypeSet: map[string]struct{}{},
			}
			serverMap[rec.BlockKey] = state

			entry := cd.keyIndex[rec.BlockKey]
			if entry == nil {
				entry = &apis.L3KeyEntry{
					ServerStates: map[apis.TransportEndpoint]*apis.L3ServerState{},
				}
				cd.keyIndex[rec.BlockKey] = entry
			}
			entry.ServerStates[ep] = state
		}
		if rec.ReplicaType != "" {
			state.ReplicaTypeSet[rec.ReplicaType] = struct{}{}
		}
	}
	return nil
}

// RemoveReplicaKeys removes keys from the endpoint's reverse map and
// reaps the forward index entry when it becomes empty.
func (i *indexer) RemoveReplicaKeys(
	mc apis.ModelContext,
	endpoint apis.TransportEndpoint,
	keys []apis.BlockHash,
) error {
	if endpoint == "" || len(keys) == 0 {
		return nil
	}
	cd := i.getContext(mc)
	if cd == nil {
		return nil
	}
	cd.mu.Lock()
	defer cd.mu.Unlock()

	serverMap := cd.serverIndex[endpoint]
	if serverMap == nil {
		return nil
	}
	for _, k := range keys {
		if _, ok := serverMap[k]; !ok {
			continue
		}
		delete(serverMap, k)
		i.unlinkKeyServerLocked(cd, k, endpoint)
	}
	if len(serverMap) == 0 {
		delete(cd.serverIndex, endpoint)
	}
	return nil
}

// MatchPrefix walks keys forward and stops at the first miss.
// L3 ignores medium and replica type.
func (i *indexer) MatchPrefix(
	mc apis.ModelContext,
	endpoint apis.TransportEndpoint,
	keys []apis.BlockHash,
) (*apis.MatchResult, error) {
	if endpoint == "" || len(keys) == 0 {
		return newMatchResult(mc, keys, 0), nil
	}
	cd := i.getContext(mc)
	if cd == nil {
		return newMatchResult(mc, keys, 0), nil
	}
	cd.mu.RLock()
	serverMap := cd.serverIndex[endpoint]
	if serverMap == nil {
		cd.mu.RUnlock()
		return newMatchResult(mc, keys, 0), nil
	}
	matched := countMatchedPrefix(serverMap, keys)
	cd.mu.RUnlock()
	return newMatchResult(mc, keys, matched), nil
}

// unlinkKeyServerLocked removes endpoint from keyIndex[k].ServerStates
// and reaps the entry when empty. Caller must hold cd.mu.Lock().
func (i *indexer) unlinkKeyServerLocked(cd *contextData, k apis.BlockHash, ep apis.TransportEndpoint) {
	entry := cd.keyIndex[k]
	if entry == nil {
		return
	}
	delete(entry.ServerStates, ep)
	if len(entry.ServerStates) == 0 {
		delete(cd.keyIndex, k)
	}
}
