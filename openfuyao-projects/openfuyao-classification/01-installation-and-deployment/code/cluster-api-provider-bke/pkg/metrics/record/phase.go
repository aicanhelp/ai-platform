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

	bkemetrics "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// PhaseDurationRecord record the phase run duration
func PhaseDurationRecord(obj client.Object, phaseName string, startTime time.Time, err error) {
	if config.MetricsAddr == "0" {
		return
	}
	log.Debugf("(%s)record phase %s duration, start time: %v, end time: %v", utils.ClientObjNS(obj), phaseName, bkemetrics.FormatTime(startTime), bkemetrics.TimeNow())

	describe := ""
	if err != nil {
		describe = err.Error()
	}

	key := utils.ClientObjNS(obj)

	bkemetrics.DefaultRegister().RecordGaugeVec(
		bkemetrics.PhaseDurationSeconds,
		key,
		phaseName,
		bkemetrics.FormatTime(startTime),
		bkemetrics.TimeNow(),
		describe,
	).Set(time.Since(startTime).Seconds())
}
