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
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/client"

	bkemetrics "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
)

func AddonInstallRecord(obj client.Object, addonName, addonVersion string, err error) func() {
	if config.MetricsAddr == "0" {
		return func() { return }
	}

	startTime := time.Now()

	return func() {
		bkemetrics.DefaultRegister().RecordGaugeVec(
			bkemetrics.AddonInstallDurationSeconds,
			utils.ClientObjNS(obj),
			fmt.Sprintf("%s-%s", addonName, addonVersion),
			bkemetrics.FormatTime(startTime),
			bkemetrics.TimeNow(),
		).Set(time.Since(startTime).Seconds())

		if err != nil {
			bkemetrics.DefaultRegister().RecordCounterVec(
				bkemetrics.AddonInstallFailedCount,
				utils.ClientObjNS(obj),
			).Inc()

		} else {
			bkemetrics.DefaultRegister().RecordCounterVec(
				bkemetrics.AddonInstallSuccessCount,
				utils.ClientObjNS(obj),
			).Inc()
		}

		bkemetrics.DefaultRegister().RecordCounterVec(
			bkemetrics.AddonInstallCount,
			utils.ClientObjNS(obj),
		).Inc()

	}

}
