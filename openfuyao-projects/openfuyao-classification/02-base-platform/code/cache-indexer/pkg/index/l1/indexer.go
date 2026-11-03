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

// Package l1 owns the L1 KVCache index for each vLLM server.
package l1

import (
	"sync"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

// l1HitMedium is vLLM's event medium value for local accelerator KV cache.
const l1HitMedium = "GPU"

// Indexer is the only contract the L1 ingest plane writes to and the
// query plane reads from. Implementations MUST NOT access L3.
type Indexer interface {
	// UpsertStoredEvent applies one BlockStored event into the L1 index.
	UpsertStoredEvent(event *apis.L1StoredEvent) error
	// RemoveBlocks removes one server's blocks from the L1 index.
	RemoveBlocks(event *apis.L1RemovedEvent) error
	// ClearServer deletes all blocks for one server in one model context.
	ClearServer(modelCtx apis.ModelContext, serverIP string) error
	// MatchPrefix returns the longest matched key prefix for one server.
	MatchPrefix(
		modelCtx apis.ModelContext,
		serverIP string,
		keys []apis.BlockHash,
	) (*apis.MatchResult, error)
}

// contextData holds L1 dual indexes for one ModelContext.
// Both maps share each *L1ServerState pointer.
type contextData struct {
	mu          sync.RWMutex
	keyIndex    map[apis.BlockHash]*apis.L1KeyEntry
	serverIndex map[string]map[apis.BlockHash]*apis.L1ServerState
}

func newContextData() *contextData {
	return &contextData{
		keyIndex:    map[apis.BlockHash]*apis.L1KeyEntry{},
		serverIndex: map[string]map[apis.BlockHash]*apis.L1ServerState{},
	}
}

type indexer struct {
	log logr.Logger

	mu        sync.RWMutex
	byContext map[apis.ModelContext]*contextData
}

// New builds a real L1 indexer.
func New(log logr.Logger) Indexer {
	return &indexer{
		log:       log,
		byContext: map[apis.ModelContext]*contextData{},
	}
}

func (i *indexer) getOrCreateContext(mc apis.ModelContext) *contextData {
	i.mu.RLock()
	cd, ok := i.byContext[mc]
	i.mu.RUnlock()
	if ok {
		return cd
	}
	i.mu.Lock()
	if cd, ok = i.byContext[mc]; !ok {
		cd = newContextData()
		i.byContext[mc] = cd
	}
	i.mu.Unlock()
	return cd
}

func (i *indexer) getContext(mc apis.ModelContext) *contextData {
	i.mu.RLock()
	cd := i.byContext[mc]
	i.mu.RUnlock()
	return cd
}

// UpsertStoredEvent merges a BlockStored event into both L1 indexes.
// It unions medium, dp_rank, and group_idx state.
func (i *indexer) UpsertStoredEvent(ev *apis.L1StoredEvent) error {
	if ev == nil || ev.ServerIP == "" || len(ev.BlockHashes) == 0 {
		return nil
	}
	cd := i.getOrCreateContext(ev.ModelContext)
	cd.mu.Lock()
	defer cd.mu.Unlock()

	serverMap := cd.serverIndex[ev.ServerIP]
	if serverMap == nil {
		serverMap = map[apis.BlockHash]*apis.L1ServerState{}
		cd.serverIndex[ev.ServerIP] = serverMap
	}

	for _, k := range ev.BlockHashes {
		state := serverMap[k]
		if state == nil {
			state = &apis.L1ServerState{
				MediumSet:     map[string]struct{}{},
				DPRankSet:     map[int64]struct{}{},
				GroupIndexSet: map[int64]struct{}{},
			}
			serverMap[k] = state

			entry := cd.keyIndex[k]
			if entry == nil {
				entry = &apis.L1KeyEntry{ServerStates: map[string]*apis.L1ServerState{}}
				cd.keyIndex[k] = entry
			}
			entry.ServerStates[ev.ServerIP] = state
		}
		if ev.Medium != "" {
			state.MediumSet[ev.Medium] = struct{}{}
		}
		state.DPRankSet[ev.DPRank] = struct{}{}
		if ev.GroupIndex != nil {
			state.GroupIndexSet[*ev.GroupIndex] = struct{}{}
		}
	}
	return nil
}

// RemoveBlocks removes the given block hashes from the server's reverse
// index and reaps the forward index entry when it becomes empty.
func (i *indexer) RemoveBlocks(ev *apis.L1RemovedEvent) error {
	if ev == nil || ev.ServerIP == "" || len(ev.BlockHashes) == 0 {
		return nil
	}
	cd := i.getContext(ev.ModelContext)
	if cd == nil {
		return nil
	}
	cd.mu.Lock()
	defer cd.mu.Unlock()

	serverMap := cd.serverIndex[ev.ServerIP]
	if serverMap == nil {
		return nil
	}
	for _, k := range ev.BlockHashes {
		if _, ok := serverMap[k]; !ok {
			continue
		}
		delete(serverMap, k)
		i.unlinkKeyServerLocked(cd, k, ev.ServerIP)
	}
	if len(serverMap) == 0 {
		delete(cd.serverIndex, ev.ServerIP)
	}
	return nil
}

// ClearServer drops every block held by the server in this ModelContext
// and is also called when Discovery removes a Pod.
func (i *indexer) ClearServer(mc apis.ModelContext, serverIP string) error {
	if serverIP == "" {
		return nil
	}
	cd := i.getContext(mc)
	if cd == nil {
		return nil
	}
	cd.mu.Lock()
	defer cd.mu.Unlock()

	serverMap := cd.serverIndex[serverIP]
	if serverMap == nil {
		return nil
	}
	for k := range serverMap {
		i.unlinkKeyServerLocked(cd, k, serverIP)
	}
	delete(cd.serverIndex, serverIP)
	return nil
}

// MatchPrefix walks keys forward and stops at the first miss.
// A hit requires presence in the server map and MediumSet containing l1HitMedium.
func (i *indexer) MatchPrefix(
	mc apis.ModelContext,
	serverIP string,
	keys []apis.BlockHash,
) (*apis.MatchResult, error) {
	res := &apis.MatchResult{Keys: keys}
	if serverIP == "" || len(keys) == 0 {
		return res, nil
	}
	cd := i.getContext(mc)
	if cd == nil {
		return res, nil
	}
	cd.mu.RLock()
	serverMap := cd.serverIndex[serverIP]
	if serverMap == nil {
		cd.mu.RUnlock()
		return res, nil
	}
	matched := 0
	for _, k := range keys {
		state, ok := serverMap[k]
		if !ok {
			break
		}
		if _, gpu := state.MediumSet[l1HitMedium]; !gpu {
			break
		}
		matched++
	}
	cd.mu.RUnlock()

	res.MatchedBlocks = matched
	if mc.BlockSize > 0 {
		res.MatchedTokens = matched * int(mc.BlockSize)
	}
	return res, nil
}

// unlinkKeyServerLocked removes serverIP from keyIndex[k].ServerStates
// and reaps the entry when empty. Caller must hold cd.mu.Lock().
func (i *indexer) unlinkKeyServerLocked(cd *contextData, k apis.BlockHash, serverIP string) {
	entry := cd.keyIndex[k]
	if entry == nil {
		return
	}
	delete(entry.ServerStates, serverIP)
	if len(entry.ServerStates) == 0 {
		delete(cd.keyIndex, k)
	}
}
