/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package statusmanage

import confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"

type StatusRecord struct {
	CurrentClusterState confv1beta1.ClusterStatus
	LatestFailedState   string
	LatestNormalState   string
	StatusCount         int
	NeedRequeue         bool
}

func (r *StatusRecord) Inc() {
	r.StatusCount++
}

func (r *StatusRecord) Dec() {
	r.StatusCount--
}

func (r *StatusRecord) Reset() {
	r.StatusCount = 0
	r.LatestFailedState = ""
}
func (r *StatusRecord) Equal(state string) bool {
	return r.LatestFailedState == state
}

func (r *StatusRecord) AllowFailed() bool {
	return r.StatusCount < ReconcileAllowedFailedCount
}

func (r *StatusRecord) SetLatestFailedState(state string) {
	r.LatestFailedState = state
}

func (r *StatusRecord) SetLatestNormalState(state string) {
	r.LatestNormalState = state
}

func (r *StatusRecord) SetCurrentClusterState(state confv1beta1.ClusterStatus) {
	r.CurrentClusterState = state
}
