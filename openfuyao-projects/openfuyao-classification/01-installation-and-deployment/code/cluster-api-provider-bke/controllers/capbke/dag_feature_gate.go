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

package capbke

import (
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/dagexec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/featuregate"
)

// buildDAGSchedulerConfig applies feature-gate wiring:
// Gate OFF → clear Yaml/Helm executors (Legacy for those types).
// Gate ON → keep executors from cfg (nil still means Legacy for that type).
func (r *BKEClusterReconciler) buildDAGSchedulerConfig(
	cluster *bkev1beta1.BKECluster,
	cfg dagexec.Config,
) dagexec.Config {
	return dagexec.WithFeatureGateExecutors(
		cfg,
		featuregate.HelmComponentEnabled(cluster),
		cfg.YamlExecutor,
		cfg.HelmExecutor,
	)
}
