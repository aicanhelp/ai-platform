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
	"github.com/go-logr/logr"
)

type reconciler struct {
	log        logr.Logger
	subscriber EndpointSubscriber
}

// NewReconciler builds a Reconciler that converts snapshot diffs into
// EndpointSubscriber callbacks. subscriber may be nil for tests.
func NewReconciler(log logr.Logger, subscriber EndpointSubscriber) Reconciler {
	return &reconciler{log: log, subscriber: subscriber}
}

func (r *reconciler) Reconcile(prev, next *Snapshot) error {
	if next == nil {
		return nil
	}
	prevPods := snapshotPods(prev)
	r.reconcileCurrentPods(prevPods, next.VLLMPods)
	r.reconcileRemovedPods(prevPods, next.VLLMPods)
	r.reconcileMooncakeMaster(prev, next)

	return nil
}

func snapshotPods(snap *Snapshot) map[string]*VLLMPodEndpoint {
	if snap == nil {
		return map[string]*VLLMPodEndpoint{}
	}
	return snap.VLLMPods
}

func (r *reconciler) reconcileCurrentPods(prevPods, nextPods map[string]*VLLMPodEndpoint) {
	for ip, ep := range nextPods {
		old, exists := prevPods[ip]
		switch {
		case !exists:
			r.handleAddedPod(ip, ep)
		case vllmPodRebuilt(old, ep):
			r.handleRebuiltPod(ip, ep)
		default:
			continue
		}
	}
}

func (r *reconciler) reconcileRemovedPods(prevPods, nextPods map[string]*VLLMPodEndpoint) {
	for ip := range prevPods {
		if _, ok := nextPods[ip]; ok {
			continue
		}
		r.handleRemovedPod(ip)
	}
}

func (r *reconciler) reconcileMooncakeMaster(prev, next *Snapshot) {
	var prevMaster *MooncakeMasterEndpoint
	if prev != nil {
		prevMaster = prev.MooncakeMasterPod
	}
	if !mooncakeMasterChanged(prevMaster, next.MooncakeMasterPod) {
		return
	}

	r.log.Info("Mooncake master changed",
		"prev", masterIP(prevMaster),
		"next", masterIP(next.MooncakeMasterPod))
	if r.subscriber == nil {
		return
	}
	if err := r.subscriber.OnMooncakeMasterChanged(next.MooncakeMasterPod); err != nil {
		r.log.Error(err, "OnMooncakeMasterChanged failed")
	}
}

func (r *reconciler) handleAddedPod(ip string, ep *VLLMPodEndpoint) {
	r.log.Info("vLLM pod added", "podIP", ip,
		"name", ep.Identity.Name,
		"zmqPort", ep.ZMQPort,
		"mooncakeResolved", ep.MooncakeClient.Resolved)
	if r.subscriber == nil {
		return
	}
	if err := r.subscriber.OnVLLMAdded(ep); err != nil {
		r.log.Error(err, "OnVLLMAdded failed", "podIP", ip)
	}
}

func (r *reconciler) handleRebuiltPod(ip string, ep *VLLMPodEndpoint) {
	r.log.Info("vLLM pod rebuilt", "podIP", ip)
	if r.subscriber == nil {
		return
	}
	if err := r.subscriber.OnVLLMRemoved(ip); err != nil {
		r.log.Error(err, "OnVLLMRemoved failed during rebuild", "podIP", ip)
	}
	if err := r.subscriber.OnVLLMAdded(ep); err != nil {
		r.log.Error(err, "OnVLLMAdded failed", "podIP", ip)
	}
}

func (r *reconciler) handleRemovedPod(ip string) {
	r.log.Info("vLLM pod removed", "podIP", ip)
	if r.subscriber == nil {
		return
	}
	if err := r.subscriber.OnVLLMRemoved(ip); err != nil {
		r.log.Error(err, "OnVLLMRemoved failed", "podIP", ip)
	}
}

func vllmPodRebuilt(a, b *VLLMPodEndpoint) bool {
	return a.Identity.UID != b.Identity.UID || a.ZMQPort != b.ZMQPort
}

func mooncakeMasterChanged(a, b *MooncakeMasterEndpoint) bool {
	if a == nil && b == nil {
		return false
	}
	if a == nil || b == nil {
		return true
	}
	return a.PodIP != b.PodIP || a.HTTPPort != b.HTTPPort || a.RPCPort != b.RPCPort || a.Identity.UID != b.Identity.UID
}

func masterIP(m *MooncakeMasterEndpoint) string {
	if m == nil {
		return ""
	}
	return m.PodIP
}
