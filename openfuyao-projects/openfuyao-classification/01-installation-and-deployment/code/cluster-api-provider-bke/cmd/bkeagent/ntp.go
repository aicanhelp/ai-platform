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

package main

import (
	"flag"
	"time"

	bkentp "gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/crontab"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// syncTime sync time with ntp, and set a coron job.
func syncTime(ntpServer string) {

	// set time zone
	loc, err := time.LoadLocation("Asia/Shanghai")
	if err == nil {
		time.Local = loc
		log.Infof("Set host timezone to Asia/Shanghai")
	}

	// try sync time
	log.Info("try sync time")
	var ntp string
	// get ntp server from flag
	if ntpServer != "" {
		server, err := utils.FormatNTPServer(ntpServer)
		if err == nil && server != "" {
			ntp = server
			log.Infof("get ntp server from flag: %s", ntp)
		} else {
			log.Warnf("get ntp server from flag failed, ntp %s err: %v", ntpServer, err)
		}
	}
	// get ntp server from Env
	if ntp == "" {
		server, err := utils.GetNTPServerEnv()
		if err == nil && server != "" {
			ntp = server
			log.Infof("get ntp server from env: %s", ntp)
		} else {
			log.Warnf("get ntp server from env failed, ntp %s err: %v", ntpServer, err)
		}
	}

	if ntp != "" {
		if err := bkentp.Date(ntp); err != nil {
			log.Warnf("sync time failed, err: %s", err.Error())
		} else {
			log.Infof("sync time success, ntp server: %s", ntp)
		}
		cron := flag.Lookup("ntpcron").Value.String()
		if err := crontab.CronSyncTime(ntp, cron); err != nil {
			log.Warnf("create sync time cron job failed, err: %s", err.Error())
		}
	} else {
		log.Warnf("No available ntp server, skipped")
	}
}
