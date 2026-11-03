/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package crontab

import (
	"time"

	"github.com/robfig/cron/v3"

	bkentp "gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

var (
	C             *cron.Cron
	SyncTimeJobId cron.EntryID
)

// EveryHour every hour
const EveryHour = "0 */1 * * *"

const SyncTimeCron = EveryHour

func init() {
	loc, err := time.LoadLocation("Asia/Shanghai")
	if err != nil {
		loc = time.UTC
		log.Warnf("failed to load location Asia/Shanghai, using UTC instead, err: %s", err.Error())
	}
	C = cron.New(cron.WithLocation(loc))
}

func CronSyncTime(server, cr string) error {
	job := func() {
		if err := bkentp.Date(server); err != nil {
			log.Warnf("sync time failed, err: %s", err.Error())
		} else {
			log.Infof("sync time from %q success", server)
		}
	}
	if cr == "" {
		cr = SyncTimeCron
	}
	jobId, err := C.AddFunc(cr, job)
	if err != nil {
		return err
	}
	SyncTimeJobId = jobId
	log.Infof("start sync time job, ntp server: %q, cron: %q", server, cr)
	// sync time right now
	C.Start()
	return nil
}

func FindSyncTimeJob() bool {
	return FindCoronJobById(SyncTimeJobId)
}
func RemoveSyncTimeJob() {
	RemoveCoronJobById(SyncTimeJobId)
}

func RemoveCoronJobById(jobId cron.EntryID) {
	C.Remove(jobId)
}

func FindCoronJobById(jobId cron.EntryID) bool {
	if len(C.Entries()) != 0 {
		job := C.Entry(jobId)
		if job.ID != 0 {
			return true
		}
		log.Warnf("job %q not found", jobId)
		return false
	}
	return false
}
