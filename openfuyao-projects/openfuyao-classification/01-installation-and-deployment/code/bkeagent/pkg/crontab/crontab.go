package crontab

import (
	"time"

	"github.com/robfig/cron/v3"
	bkentp "gopkg.openfuyao.cn/bkecommon/ntp"

	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

var (
	C             *cron.Cron
	SyncTimeJobId cron.EntryID
)

// EveryHour every hour
const EveryHour = "0 */1 * * *"

const SyncTimeCron = EveryHour

func init() {
	loc, _ := time.LoadLocation("Asia/Shanghai")
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
