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

package dagexec

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

// StatusComponentType is the lifecycle status-side component scope (node vs cluster).
// Distinct from executor types (inline/yaml/helm).
type StatusComponentType string

const (
	StatusComponentTypeNode    StatusComponentType = "node"
	StatusComponentTypeCluster StatusComponentType = "cluster"
)

// ComponentMarkRef identifies a component entry in clusterComponentStatuses.
type ComponentMarkRef struct {
	Cluster       *bkev1beta1.BKECluster
	Name          string
	ComponentType StatusComponentType
	NodeIP        string
}

// ComponentStatusUpdater writes component lifecycle status on BKECluster.
type ComponentStatusUpdater interface {
	MarkPending(ctx context.Context, ref ComponentMarkRef) error
	MarkInstalled(ctx context.Context, ref ComponentMarkRef, version string) error
	MarkFailed(ctx context.Context, ref ComponentMarkRef, err error) error
	MarkRollingBack(ctx context.Context, ref ComponentMarkRef) error
	MarkUninstalling(ctx context.Context, ref ComponentMarkRef) error
	MarkRemoved(ctx context.Context, ref ComponentMarkRef) error
	// ClearComponentStatus removes a component entry when execution was skipped
	// (e.g. manifest not installed) and no lifecycle record should remain.
	ClearComponentStatus(ctx context.Context, ref ComponentMarkRef) error
}

// BKEComponentStatusUpdater patches BKECluster.Status.ClusterComponentStatuses.
type BKEComponentStatusUpdater struct {
	Client client.Client
}

// NewBKEComponentStatusUpdater returns a Patch-based ComponentStatusUpdater.
func NewBKEComponentStatusUpdater(c client.Client) *BKEComponentStatusUpdater {
	return &BKEComponentStatusUpdater{Client: c}
}

func (u *BKEComponentStatusUpdater) MarkPending(ctx context.Context, ref ComponentMarkRef) error {
	return u.patch(ctx, ref, func(st *confv1beta1.ComponentLifecycleStatus) {
		st.Phase = confv1beta1.LifecyclePhasePending
		st.Message = ""
	})
}

func (u *BKEComponentStatusUpdater) MarkInstalled(ctx context.Context, ref ComponentMarkRef, version string) error {
	return u.patch(ctx, ref, func(st *confv1beta1.ComponentLifecycleStatus) {
		st.Phase = confv1beta1.LifecyclePhaseInstalled
		st.CurrentVersion = version
		st.Message = ""
	})
}

func (u *BKEComponentStatusUpdater) MarkFailed(ctx context.Context, ref ComponentMarkRef, err error) error {
	msg := ""
	if err != nil {
		msg = err.Error()
	}
	return u.patch(ctx, ref, func(st *confv1beta1.ComponentLifecycleStatus) {
		st.Phase = confv1beta1.LifecyclePhaseFailed
		st.Message = msg
	})
}

func (u *BKEComponentStatusUpdater) MarkRollingBack(ctx context.Context, ref ComponentMarkRef) error {
	return u.patch(ctx, ref, func(st *confv1beta1.ComponentLifecycleStatus) {
		st.Phase = confv1beta1.LifecyclePhaseRollingBack
	})
}

func (u *BKEComponentStatusUpdater) MarkUninstalling(ctx context.Context, ref ComponentMarkRef) error {
	return u.patch(ctx, ref, func(st *confv1beta1.ComponentLifecycleStatus) {
		st.Phase = confv1beta1.LifecyclePhaseUninstalling
	})
}

func (u *BKEComponentStatusUpdater) MarkRemoved(ctx context.Context, ref ComponentMarkRef) error {
	return u.patch(ctx, ref, func(st *confv1beta1.ComponentLifecycleStatus) {
		st.Phase = confv1beta1.LifecyclePhaseRemoved
		st.CurrentVersion = ""
	})
}

func (u *BKEComponentStatusUpdater) ClearComponentStatus(ctx context.Context, ref ComponentMarkRef) error {
	current, orig, err := u.loadClusterForStatusPatch(ctx, ref)
	if err != nil {
		return err
	}
	if current.Status.ClusterComponentStatuses == nil {
		return nil
	}
	if _, ok := current.Status.ClusterComponentStatuses[ref.Name]; !ok {
		return nil
	}
	delete(current.Status.ClusterComponentStatuses, ref.Name)
	if len(current.Status.ClusterComponentStatuses) == 0 {
		current.Status.ClusterComponentStatuses = nil
	}
	if err := u.Client.Status().Patch(ctx, current, client.MergeFrom(orig)); err != nil {
		return err
	}
	removeComponentStatusFromMemory(ref)
	return nil
}

func (u *BKEComponentStatusUpdater) patch(
	ctx context.Context,
	ref ComponentMarkRef,
	mutate func(*confv1beta1.ComponentLifecycleStatus),
) error {
	current, orig, err := u.loadClusterForStatusPatch(ctx, ref)
	if err != nil {
		return err
	}

	if current.Status.ClusterComponentStatuses == nil {
		current.Status.ClusterComponentStatuses = make(map[string]confv1beta1.ComponentLifecycleStatus)
	}
	st := current.Status.ClusterComponentStatuses[ref.Name]
	prevPhase := st.Phase

	st.Name = ref.Name
	st.NodeIP = ref.NodeIP
	st.ComponentType = toLifecycleComponentType(ref.ComponentType)
	mutate(&st)

	if st.Phase != prevPhase {
		now := metav1.Now()
		st.LastTransitionTime = &now
	}

	current.Status.ClusterComponentStatuses[ref.Name] = st
	if err := u.Client.Status().Patch(ctx, current, client.MergeFrom(orig)); err != nil {
		return err
	}

	if ref.Cluster.Status.ClusterComponentStatuses == nil {
		ref.Cluster.Status.ClusterComponentStatuses = make(map[string]confv1beta1.ComponentLifecycleStatus)
	}
	ref.Cluster.Status.ClusterComponentStatuses[ref.Name] = st
	return nil
}

func (u *BKEComponentStatusUpdater) loadClusterForStatusPatch(
	ctx context.Context,
	ref ComponentMarkRef,
) (*bkev1beta1.BKECluster, *bkev1beta1.BKECluster, error) {
	if err := u.validateUpdaterRef(ref); err != nil {
		return nil, nil, err
	}

	key := client.ObjectKeyFromObject(ref.Cluster)
	current := &bkev1beta1.BKECluster{}
	if err := u.Client.Get(ctx, key, current); err != nil {
		return nil, nil, err
	}
	return current, current.DeepCopy(), nil
}

func (u *BKEComponentStatusUpdater) validateUpdaterRef(ref ComponentMarkRef) error {
	if u == nil || u.Client == nil {
		return fmt.Errorf("component status updater client is nil")
	}
	if ref.Cluster == nil {
		return fmt.Errorf("cluster is nil")
	}
	if ref.Name == "" {
		return fmt.Errorf("component name is required")
	}
	return nil
}

func removeComponentStatusFromMemory(ref ComponentMarkRef) {
	if ref.Cluster == nil || ref.Cluster.Status.ClusterComponentStatuses == nil {
		return
	}
	delete(ref.Cluster.Status.ClusterComponentStatuses, ref.Name)
	if len(ref.Cluster.Status.ClusterComponentStatuses) == 0 {
		ref.Cluster.Status.ClusterComponentStatuses = nil
	}
}

func toLifecycleComponentType(t StatusComponentType) confv1beta1.LifecycleComponentType {
	if t == StatusComponentTypeNode {
		return confv1beta1.LifecycleComponentTypeNode
	}
	return confv1beta1.LifecycleComponentTypeCluster
}
