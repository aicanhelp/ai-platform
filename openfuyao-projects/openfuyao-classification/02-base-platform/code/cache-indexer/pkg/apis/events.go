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

package apis

// L1StoredEvent is the normalized form of a vLLM ZMQ BlockStored event.
type L1StoredEvent struct {
	ModelContext     ModelContext
	ServerIP         string
	BlockHashes      []BlockHash
	ParentBlockHash  *BlockHash
	TokenIDs         []int32
	BlockSize        int64
	Medium           string
	DPRank           int64
	GroupIndex       *int64
	EventTimestampMs int64
}

// L1RemovedEvent is the normalized form of a vLLM ZMQ BlockRemoved event.
type L1RemovedEvent struct {
	ModelContext     ModelContext
	ServerIP         string
	BlockHashes      []BlockHash
	Medium           string
	GroupIndex       *int64
	EventTimestampMs int64
}

// L3ReplicaRecord is one row in a Mooncake batch_query_keys response that
// has been normalized into (BlockKey, TransportEndpoint, ReplicaType).
type L3ReplicaRecord struct {
	ModelContext      ModelContext
	ServerIP          string
	BlockKey          BlockHash
	RawMooncakeKey    string
	ReplicaType       string
	TransportEndpoint TransportEndpoint
}

// L1ServerState is the per-(server, BlockHash) state aggregated from L1
// events. MediumSet={"GPU"} is required for L1 hit accounting.
type L1ServerState struct {
	MediumSet     map[string]struct{}
	DPRankSet     map[int64]struct{}
	GroupIndexSet map[int64]struct{}
}

// L3ServerState is the per-(transport_endpoint, BlockHash) state.
type L3ServerState struct {
	ReplicaTypeSet map[string]struct{}
}

// L1KeyEntry / L3KeyEntry are the forward-index entries shared with the
// reverse index via the same *State pointer.
type L1KeyEntry struct {
	ServerStates map[string]*L1ServerState
}

// L3KeyEntry is the L3 forward-index entry for one BlockHash.
type L3KeyEntry struct {
	ServerStates map[TransportEndpoint]*L3ServerState
}
