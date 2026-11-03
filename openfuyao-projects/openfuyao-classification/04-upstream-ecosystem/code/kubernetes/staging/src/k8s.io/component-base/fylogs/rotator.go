/*
Copyright 2018 The Kubernetes Authors.
Copyright (c) 2025 Huawei Technologies Co., Ltd.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fylogs

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	// timestampFormat is format of the timestamp suffix for rotated log.
	// See https://golang.org/pkg/time/#Time.Format.
	timestampFormat = "20060102-150405"

	defaultRotationalWaitTime = 10 * time.Second

	// compressSuffix is the suffix for compressed log.
	compressSuffix = ".gz"

	// tmpSuffix is the suffix for temporary file.
	tmpSuffix = ".tmp"

	// MB单位需要移位长度
	mbLen = 20
)

func newRotator(fileWrap *fileWrap, wrapFileConfig *WrapFileConfig, fileStat os.FileInfo) *rotator {
	return &rotator{
		fileWrap:        fileWrap,
		counter:         fileStat.Size(),
		totalCountLimit: wrapFileConfig.rotateFileConfig.TotalCountLimit,
		eachFileSize:    wrapFileConfig.rotateFileConfig.EachFileMaxSize << mbLen,
		totalSizeLimit:  wrapFileConfig.rotateFileConfig.TotalSizeLimit << mbLen,
		enableCompress:  wrapFileConfig.rotateFileConfig.EnableCompress,
		compressPerm:    wrapFileConfig.rotateFileConfig.CompressPerm,
	}
}

type rotator struct {
	*fileWrap
	counter      int64
	compressPerm uint32

	totalCountLimit int
	eachFileSize    int64
	totalSizeLimit  int64
	enableCompress  bool

	// 轮转请求计数
	rotateCounting uint
	rotateLock     sync.Mutex
}

func (r *rotator) rotate() {
	r.counter += int64(r.writtenBytes)
	if r.counter >= r.eachFileSize {
		stat, statErr := os.Stat(r.filePath)
		if statErr != nil {
			printErrWithIgnoreFailed(fmt.Errorf("stat file %s failed: %v", r.filePath, statErr))
			return
		}
		if stat.Size() < r.eachFileSize {
			r.counter = stat.Size()
			return
		}

		timestamp := time.Now().Format(timestampFormat)
		rotated := fmt.Sprintf("%s_%s", r.filePath, timestamp)
		err := os.Rename(r.filePath, rotated)
		if err != nil {
			printErrWithIgnoreFailed(fmt.Errorf("rename file %s to %s failed: %v", r.filePath, rotated, err))
			return
		}

		newTarget, err := createLogFile(r.filePath, r.filePermissions)
		if err != nil {
			printErrWithIgnoreFailed(fmt.Errorf("failed to create log file %v: %w", r.filePath, err))
			return
		}

		oldTarget := r.target
		r.target = newTarget
		closeErr := oldTarget.Close()
		if closeErr != nil {
			printErrWithIgnoreFailed(fmt.Errorf("close old log file failed:%v", closeErr))
		}
		go r.startRotateLog()
	}
}

func (r *rotator) startRotateLog() {
	r.rotateLock.Lock()
	r.rotateCounting++
	r.rotateLock.Unlock()

	// Ensure only one rotation at a time
	if r.rotateCounting != 1 {
		return
	}

	var err error
	defer r.afterRotateLog(err)
	err = r.rotateLogs()
	if err != nil {
		r.tryPrintErr(err)
	}
}

func (r *rotator) afterRotateLog(err error) {
	r.rotateLock.Lock()
	moreRotate := r.rotateCounting != 1
	r.rotateCounting = 0
	r.rotateLock.Unlock()
	if moreRotate {
		// during rotating, coming in new rotate request, start rotate log again
		go r.startRotateLog()
		return
	}
	if err != nil {
		timer := time.NewTimer(defaultRotationalWaitTime)
		<-timer.C // when failed, delay 10s retry
		timer.Stop()
		go r.startRotateLog()
	}
}

func (r *rotator) rotateLogs() error {
	pattern := fmt.Sprintf("%s_*", r.filePath)
	logs, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("failed to list log files with pattern %q: %v", pattern, err)
	}

	logs, err = r.cleanupUnusedLogs(logs)
	if err != nil {
		return fmt.Errorf("failed to cleanup logs: %v", err)
	}

	logs, err = r.removeExcessLogs(logs)
	if err != nil {
		return fmt.Errorf("failed to remove excess logs: %v", err)
	}

	if r.enableCompress {
		// Compress uncompressed log files.
		for _, log := range logs {
			if strings.HasSuffix(log, compressSuffix) {
				continue
			}
			if err := r.compressLog(log); err != nil {
				return fmt.Errorf("failed to compress log %q: %v", log, err)
			}
		}
	}
	return nil
}

// compressLog compresses a log to log.gz with gzip.
func (r *rotator) compressLog(log string) error {
	read, err := os.Open(log)
	if err != nil {
		return fmt.Errorf("failed to open log file %q: %v", log, err)
	}
	defer r.closeIgnoreErr(read, log, true)
	tmpLog := log + tmpSuffix
	tmpFile, err := os.OpenFile(tmpLog, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(r.compressPerm))
	if err != nil {
		return fmt.Errorf("failed to create temporary log file %q: %v", tmpLog, err)
	}
	defer func() {
		// Best effort cleanup of tmpLog.
		removeErr := os.Remove(tmpLog)
		if removeErr != nil {
			r.tryPrintErr(removeErr)
		}
	}()
	defer r.closeIgnoreErr(tmpFile, tmpLog, true)
	gzWriter := gzip.NewWriter(tmpFile)
	defer r.closeIgnoreErr(gzWriter, "gzWriter:"+tmpLog, true)
	if _, err := io.Copy(gzWriter, read); err != nil {
		return fmt.Errorf("failed to compress %q to %q: %v", log, tmpLog, err)
	}
	// The archive needs to be closed before renaming, otherwise an error will occur on Windows.
	r.closeIgnoreErr(gzWriter, "gzWriter:"+tmpLog, false)
	r.closeIgnoreErr(tmpFile, tmpLog, false)
	compressedLog := log + compressSuffix
	if err := os.Rename(tmpLog, compressedLog); err != nil {
		return fmt.Errorf("failed to rename %q to %q: %v", tmpLog, compressedLog, err)
	}
	// Remove old log file.
	r.closeIgnoreErr(read, log, false)
	if err := os.Remove(log); err != nil {
		return fmt.Errorf("failed to remove log %q after compress: %v", log, err)
	}
	return nil
}

func (r *rotator) closeIgnoreErr(file io.Closer, name string, ignoreClosed bool) {
	err := file.Close()
	if err != nil {
		if errors.Is(err, os.ErrClosed) && ignoreClosed {
			return
		}
		r.tryPrintErr(fmt.Errorf("close file:%s, with err: %v", name, err))
	}
}

// removeExcessLogs removes old logs to make sure there are only at most totalCountLimit and totalSizeLimit log files.
func (r *rotator) removeExcessLogs(logs []string) ([]string, error) {
	newLogs, err := r.removeExcessLogsCount(logs)
	if err != nil {
		return nil, err
	}
	return r.removeExcessLogsSize(newLogs)
}

// removeExcessLogsSize removes old logs to make sure there are only at most totalSizeLimit log files.
func (r *rotator) removeExcessLogsSize(logs []string) ([]string, error) {
	if r.totalSizeLimit == 0 {
		return logs, nil // ignore log total size
	}
	// Sort log files in oldest to newest order.
	sort.Strings(logs)

	// A log file is in writing, we can have at most totalSizeLimit - eachFileSize rotated log files.
	// Keep totalSizeLimit-eachFileSize files by removing old files.
	reserveTotalSize := r.totalSizeLimit - r.eachFileSize
	index := len(logs) - 1
	var sizeCounter int64 = 0

	for ; index >= 0; index-- {
		log := logs[index]
		stat, err := os.Stat(log)
		if err != nil {
			return nil, fmt.Errorf("stat file %s err: %v", log, err)
		}
		sizeCounter += stat.Size()
		if sizeCounter >= reserveTotalSize {
			break
		}
	}

	if sizeCounter < reserveTotalSize { // index < 0
		return logs, nil
	}

	for i := 0; i <= index; i++ {
		if err := os.Remove(logs[i]); err != nil {
			return nil, fmt.Errorf("failed to remove old log by counter %q: %v", logs[i], err)
		}
	}
	logs = logs[index:]
	return logs, nil
}

// removeExcessLogsCount removes old logs to make sure there are only at most totalCountLimit log files.
func (r *rotator) removeExcessLogsCount(logs []string) ([]string, error) {
	if r.totalCountLimit == 0 {
		return logs, nil // ignore log count
	}
	// Sort log files in oldest to newest order.
	sort.Strings(logs)

	// A log file is in writing, we can have at most totalCountLimit - 1 rotated log files.
	// Keep totalCountLimit-1 files by removing old files.
	maxRotatedFiles := r.totalCountLimit - 1
	if maxRotatedFiles < 0 {
		maxRotatedFiles = 0
	}
	i := 0
	for ; i < len(logs)-maxRotatedFiles; i++ {
		if err := os.Remove(logs[i]); err != nil {
			return nil, fmt.Errorf("failed to remove old log by counter %q: %v", logs[i], err)
		}
	}
	logs = logs[i:]
	return logs, nil
}

// cleanupUnusedLogs cleans up temporary or expectedUnused log files generated by previous log rotation failure.
func (r *rotator) cleanupUnusedLogs(logs []string) ([]string, error) {
	inuse, unused := r.filterUnusedLogs(logs)
	for _, log := range unused {
		if err := os.Remove(log); err != nil {
			return nil, fmt.Errorf("failed to remove expectedUnused log %q: %v", log, err)
		}
	}
	return inuse, nil
}

func (r *rotator) tryPrintErr(err2Print error) {
	msg := fmt.Sprintf("%s => %v\n", time.Now().Format(time.RFC3339), err2Print)
	_, err := r.fileWrap.Write([]byte(msg))
	if err != nil {
		printErrWithIgnoreFailed(fmt.Errorf("write log err: %v, with origin error : %v", err, err2Print))
	}
}

// filterUnusedLogs splits logs into 2 groups, the 1st group is in used logs,
// the second group is expectedUnused logs.
func (r *rotator) filterUnusedLogs(logs []string) ([]string, []string) {
	var inuse []string
	var unused []string
	for _, log := range logs {
		if r.isInUse(log, logs) {
			inuse = append(inuse, log)
		} else {
			unused = append(unused, log)
		}
	}
	return inuse, unused
}

// isInUse checks whether a container log file is still expectedInuse.
func (r *rotator) isInUse(fileName string, logs []string) bool {
	// All temporary files are not in use.
	if strings.HasSuffix(fileName, tmpSuffix) {
		return false
	}
	// All compressed logs are in use.
	if strings.HasSuffix(fileName, compressSuffix) {
		return true
	}
	// Files has already been compressed are not in use.
	for _, another := range logs {
		if fileName+compressSuffix == another {
			return false
		}
	}
	return true
}
