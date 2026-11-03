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

// Package discovery lists K8s Pods, builds snapshots, and drives ingest via
// Reconciler / EndpointSubscriber.
package discovery

import (
	"context"
	"time"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

// PodIdentity uniquely identifies a Pod observation.
type PodIdentity struct {
	Namespace, Name, UID, ResourceVersion string
}

// MooncakeClientEndpoint resolves a vLLM Pod to Mooncake transport_endpoint_.
// It is sourced from Mooncake Master /get_all_segments.
type MooncakeClientEndpoint struct {
	TransportEndpoint apis.TransportEndpoint
	Host              string
	TransferPort      int32
	Resolved          bool
	Source            string // master-segment | unresolved
}

// VLLMPodEndpoint is the discovery snapshot entry for one ready vLLM Pod.
type VLLMPodEndpoint struct {
	Identity       PodIdentity
	PodIP          string
	ZMQPort        int32
	Labels         map[string]string
	MooncakeClient MooncakeClientEndpoint
	Ready          bool
	LastSeenAt     time.Time
}

// MooncakeMasterEndpoint is the discovery snapshot entry for one Mooncake Master Pod.
type MooncakeMasterEndpoint struct {
	Identity   PodIdentity
	PodIP      string
	HTTPPort   int32
	RPCPort    int32
	Labels     map[string]string
	Ready      bool
	LastSeenAt time.Time
}

// Snapshot is the typed result of one Pod list refresh.
type Snapshot struct {
	Namespace         string
	Version           uint64
	RefreshedAt       time.Time
	VLLMPods          map[string]*VLLMPodEndpoint
	MooncakeMasterPod *MooncakeMasterEndpoint
}

// Manager periodically lists Pods and emits snapshots.
type Manager interface {
	// Start begins the background snapshot refresh loop.
	Start(ctx context.Context) error
	// Stop terminates the background refresh loop.
	Stop() error
	// Snapshot returns the latest observed discovery state.
	Snapshot() *Snapshot
}

// Reconciler turns snapshot diffs into Subscribe / Unsubscribe /
// UpdateTarget side-effects on the ingest plane via EndpointSubscriber.
type Reconciler interface {
	// Reconcile applies the delta from prev to next.
	Reconcile(prev, next *Snapshot) error
}

// EndpointSubscriber decouples discovery from concrete ingest implementations.
// Adapters in cmd/ wire L1 / L3 ingest to these methods.
type EndpointSubscriber interface {
	OnVLLMAdded(endpoint *VLLMPodEndpoint) error
	OnVLLMRemoved(podIP string) error
	OnMooncakeMasterChanged(endpoint *MooncakeMasterEndpoint) error
}

// Registry is the read-side that the query plane consults to map
// vLLM pod IP → Mooncake client transport_endpoint.
type Registry interface {
	// Update replaces the current snapshot and derived lookup maps.
	Update(snapshot *Snapshot) error
	// Snapshot returns the latest registry snapshot.
	Snapshot() *Snapshot
	// ResolveTransportEndpoint maps a vLLM Pod IP to Mooncake transport_endpoint.
	ResolveTransportEndpoint(serverIP string) (apis.TransportEndpoint, bool)
	// ResolveVLLMPod returns the discovered vLLM Pod metadata by Pod IP.
	ResolveVLLMPod(serverIP string) (*VLLMPodEndpoint, bool)
}
