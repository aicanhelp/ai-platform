package main

import (
	"flag"
	"time"

	bkentp "gopkg.openfuyao.cn/bkecommon/ntp"

	"gopkg.openfuyao.cn/bkeagent/pkg/crontab"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
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
