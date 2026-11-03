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
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/robfig/cron/v3"
	"github.com/stretchr/testify/assert"

	bkentp "gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"
)

func init() {
}

const (
	numZero       = 0
	numOne        = 1
	numTwo        = 2
	numThree      = 3
	numFour       = 4
	numFive       = 5
	numSix        = 6
	numSeven      = 7
	numEight      = 8
	numNine       = 9
	numTen        = 10
	numEleven     = 11
	numTwelve     = 12
	numFifteen    = 15
	numSixty      = 60
	numOneHundred = 100
	numTwoHundred = 200
)

const (
	testNtpServer          = "pool.ntp.org"
	testInvalidCron        = "invalid cron"
	testValidCronEveryHour = "0 */1 * * *"
	testLocationName       = "Asia/Shanghai"
)

var (
	shortWaitDuration  = 10 * time.Millisecond
	mediumWaitDuration = 50 * time.Millisecond
	longWaitDuration   = 100 * time.Millisecond
)

func setupTestCron() {
	loc, _ := time.LoadLocation(testLocationName)
	C = cron.New(cron.WithLocation(loc))
	SyncTimeJobId = numZero
}

func cleanupTestCron() {
	if C != nil {
		C.Stop()
	}
	C = nil
	SyncTimeJobId = numZero
}

func TestCronSyncTimeWithDefaultCron(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	err := CronSyncTime(testNtpServer, "")

	assert.NoError(t, err)
	assert.NotZero(t, SyncTimeJobId)
}

func TestCronSyncTimeWithCustomCron(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	customCron := "0 0 * * *"
	err := CronSyncTime(testNtpServer, customCron)

	assert.NoError(t, err)
	assert.NotZero(t, SyncTimeJobId)
}

func TestCronSyncTimeWithInvalidCron(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	err := CronSyncTime(testNtpServer, testInvalidCron)

	assert.Error(t, err)
	assert.Equal(t, cron.EntryID(numZero), SyncTimeJobId)
}

func TestCronSyncTimeWithNtpError(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		assert.AnError,
	)
	defer patches.Reset()

	err := CronSyncTime(testNtpServer, testValidCronEveryHour)

	assert.NoError(t, err)
	assert.NotZero(t, SyncTimeJobId)
}

func TestFindSyncTimeJobWhenJobExists(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	jobId, _ := C.AddFunc(testValidCronEveryHour, func() {})
	SyncTimeJobId = jobId

	result := FindSyncTimeJob()

	assert.True(t, result)
}

func TestFindSyncTimeJobWhenJobNotExists(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	SyncTimeJobId = numZero

	result := FindSyncTimeJob()

	assert.False(t, result)
}

func TestFindSyncTimeJobWhenEntriesEmpty(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	SyncTimeJobId = numOneHundred

	result := FindSyncTimeJob()

	assert.False(t, result)
}

func TestRemoveSyncTimeJob(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	jobId, _ := C.AddFunc(testValidCronEveryHour, func() {})
	SyncTimeJobId = jobId

	RemoveSyncTimeJob()

	assert.False(t, FindSyncTimeJob())
}

func TestRemoveCoronJobById(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	jobIdOne, _ := C.AddFunc(testValidCronEveryHour, func() {})
	jobIdTwo, _ := C.AddFunc(testValidCronEveryHour, func() {})

	RemoveCoronJobById(jobIdOne)

	assert.True(t, len(C.Entries()) == numOne)
	assert.Equal(t, jobIdTwo, C.Entries()[numZero].ID)
}

func TestRemoveCoronJobByIdNotExists(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	initialEntries := len(C.Entries())

	RemoveCoronJobById(numZero)

	assert.Equal(t, initialEntries, len(C.Entries()))
}

func TestFindCoronJobByIdWhenEntriesEmpty(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	result := FindCoronJobById(numOne)

	assert.False(t, result)
}

func TestFindCoronJobByIdWhenJobExists(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	jobId, _ := C.AddFunc(testValidCronEveryHour, func() {})

	result := FindCoronJobById(jobId)

	assert.True(t, result)
}

func TestFindCoronJobByIdWhenJobNotExists(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	jobId, _ := C.AddFunc(testValidCronEveryHour, func() {})

	result := FindCoronJobById(jobId + numOne)

	assert.False(t, result)
}

func TestCronSyncTimeLogsSuccess(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	err := CronSyncTime(testNtpServer, testValidCronEveryHour)

	assert.NoError(t, err)
	assert.NotZero(t, SyncTimeJobId)
}

func TestCronSyncTimeLogsWarning(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		assert.AnError,
	)
	defer patches.Reset()

	err := CronSyncTime(testNtpServer, testValidCronEveryHour)

	assert.NoError(t, err)
	assert.NotZero(t, SyncTimeJobId)
}

func TestCronSyncTimeLogsInfoOnSuccess(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	err := CronSyncTime(testNtpServer, testValidCronEveryHour)

	assert.NoError(t, err)
	assert.NotZero(t, SyncTimeJobId)
}

func TestCronSyncTimeMultipleJobs(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	err1 := CronSyncTime("server1", testValidCronEveryHour)
	err2 := CronSyncTime("server2", testValidCronEveryHour)

	assert.NoError(t, err1)
	assert.NoError(t, err2)
	assert.NotEqual(t, SyncTimeJobId, numZero)
}

func TestFindCoronJobByIdWithZeroId(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	result := FindCoronJobById(numZero)

	assert.False(t, result)
}

func TestCronSyncTimeWithDifferentCronExpressions(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	testCases := []struct {
		name string
		cron string
	}{
		{
			name: "EveryMinute",
			cron: "* * * * *",
		},
		{
			name: "EveryHour",
			cron: "0 * * * *",
		},
		{
			name: "EveryDayAtMidnight",
			cron: "0 0 * * *",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			setupTestCron()
			defer cleanupTestCron()

			patches := gomonkey.ApplyFuncReturn(
				bkentp.Date,
				nil,
			)
			defer patches.Reset()

			err := CronSyncTime(testNtpServer, tc.cron)

			assert.NoError(t, err)
		})
	}
}

func TestCronSyncTimeReAddJob(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	_ = CronSyncTime(testNtpServer, testValidCronEveryHour)
	initialJobId := SyncTimeJobId

	_ = CronSyncTime(testNtpServer, testValidCronEveryHour)

	assert.NotEqual(t, initialJobId, SyncTimeJobId)
}

func TestRemoveSyncTimeJobWhenNotExists(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	SyncTimeJobId = numZero

	RemoveSyncTimeJob()

	assert.False(t, FindSyncTimeJob())
}

func TestCronNewWithDifferentLocation(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	loc, _ := time.LoadLocation("America/New_York")
	C = cron.New(cron.WithLocation(loc))

	assert.Equal(t, "America/New_York", C.Location().String())
}

func TestCronSyncTimeWithEmptyServer(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	err := CronSyncTime("", testValidCronEveryHour)

	assert.NoError(t, err)
}

func TestFindCoronJobByIdLogsWarning(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	jobId, _ := C.AddFunc(testValidCronEveryHour, func() {})
	nonExistentJobId := jobId + numOne

	result := FindCoronJobById(nonExistentJobId)

	assert.False(t, result)
}

func TestCronSyncTimeLogsJobStart(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	err := CronSyncTime(testNtpServer, testValidCronEveryHour)

	assert.NoError(t, err)
	assert.NotZero(t, SyncTimeJobId)
}

func TestCronEntriesCount(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	assert.Equal(t, numZero, len(C.Entries()))

	C.AddFunc(testValidCronEveryHour, func() {})

	assert.Equal(t, numOne, len(C.Entries()))
}

func TestCronSyncTimeWithVeryLongCron(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	longCron := "0 0 1 * *"
	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	err := CronSyncTime(testNtpServer, longCron)

	assert.NoError(t, err)
}

func TestRemoveCoronJobByIdMultipleCalls(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	jobId, _ := C.AddFunc(testValidCronEveryHour, func() {})

	RemoveCoronJobById(jobId)
	RemoveCoronJobById(jobId)
	RemoveCoronJobById(jobId)

	assert.Equal(t, numZero, len(C.Entries()))
}

func TestCronSyncTimeConcurrentAdd(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	done := make(chan bool)
	for i := numZero; i < numFive; i++ {
		go func() {
			_ = CronSyncTime(testNtpServer, testValidCronEveryHour)
			done <- true
		}()
	}

	for i := numZero; i < numFive; i++ {
		<-done
	}

	assert.True(t, len(C.Entries()) >= numFive)
}

func TestCronSyncTimeStartIsCalled(t *testing.T) {
	setupTestCron()
	defer cleanupTestCron()

	patches := gomonkey.ApplyFuncReturn(
		bkentp.Date,
		nil,
	)
	defer patches.Reset()

	C.Start()

	initialEntries := len(C.Entries())

	_ = CronSyncTime(testNtpServer, testValidCronEveryHour)

	assert.Equal(t, initialEntries+numOne, len(C.Entries()))
}
