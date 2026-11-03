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

package record

import (
	"time"

	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkemetrics "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
)

func ClusterHealthyCountRecord(obj client.Object, status confv1beta1.ClusterStatus) {
	if config.MetricsAddr == "0" {
		return
	}
	switch status {
	case bkev1beta1.ClusterStatusReady:
		bkemetrics.DefaultRegister().RecordGaugeVec(
			bkemetrics.ClusterReadyStatus,
			utils.ClientObjNS(obj),
		).Set(1)

	case bkev1beta1.ClusterUnhealthy:
		bkemetrics.DefaultRegister().RecordCounterVec(
			bkemetrics.ClusterUnhealthyCount,
			utils.ClientObjNS(obj),
		).Inc()

		bkemetrics.DefaultRegister().RecordGaugeVec(
			bkemetrics.ClusterReadyStatus,
			utils.ClientObjNS(obj),
			//bkemetrics.TimeNow(),
		).Set(0)
	default:
		// 处理未知的集群状态
		bkemetrics.DefaultRegister().RecordGaugeVec(
			bkemetrics.ClusterReadyStatus,
			utils.ClientObjNS(obj),
		).Set(0)
	}
}

func ClusterBootstrapDurationRecord(obj client.Object) {
	if config.MetricsAddr == "0" {
		return
	}

	startTime := obj.GetCreationTimestamp().Time

	bkemetrics.DefaultRegister().RecordGaugeVec(
		bkemetrics.ClusterBootstrapDurationSeconds,
		utils.ClientObjNS(obj),
		bkemetrics.FormatTime(startTime),
		bkemetrics.TimeNow(),
	).Set(time.Since(startTime).Seconds())

}
