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

	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

// patchDeclarativeUpgrade applies a narrow Status().Patch to BKECluster.status.declarativeUpgrade.
// Pattern matches BKEComponentStatusUpdater: Get → mutate one status subtree → MergeFrom patch.
func patchDeclarativeUpgrade(
	ctx context.Context,
	c client.Client,
	cluster *bkev1beta1.BKECluster,
	mutate func(*confv1beta1.DeclarativeUpgradeStatus),
) error {
	if c == nil || cluster == nil || mutate == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	key := client.ObjectKeyFromObject(cluster)
	current := &bkev1beta1.BKECluster{}
	if err := c.Get(ctx, key, current); err != nil {
		return err
	}
	// Match prior SyncStatusUntilComplete behavior: no-op when progress is unset.
	if current.Status.DeclarativeUpgrade == nil {
		return nil
	}

	orig := current.DeepCopy()
	mutate(current.Status.DeclarativeUpgrade)
	if err := c.Status().Patch(ctx, current, client.MergeFrom(orig)); err != nil {
		return err
	}

	cluster.Status.DeclarativeUpgrade = current.Status.DeclarativeUpgrade.DeepCopy()
	return nil
}
