/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

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
package log

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
	"syscall"
	"time"
)

const (
	fileModeUserRoForArchiveLog = 0440
	fileModeUserRwForWritingLog = 0640
	dirModeUserRwxForLogDir     = 0750
	hourPerDay                  = 24

	compressSuffix    = ".gz"
	defaultLogPath    = "/var/log"
	defaultLogMaxSize = 20
	backupTimeFormat  = "20060102T150405.000"
)

// FileLogger implements a rotating file logger with compression and retention policies
type FileLogger struct {
	fileName            string
	maxSize             int
	maxAge              int
	maxBackups          int
	localTime           bool
	writtenLen          int64
	file                *os.File
	mutex               sync.Mutex
	staleLogCh          chan bool
	startHandleStaleLog sync.Once
}

var _ io.WriteCloser = (*FileLogger)(nil)

var (
	currentTime = time.Now
	megabyte    = 1024 * 1024
)

type logInfosWithTimestamp []logInfo

type logInfo struct {
	timestamp time.Time
	os.FileInfo
}

func (b logInfosWithTimestamp) Less(i, j int) bool {
	if i >= b.Len() || j >= b.Len() {
		return false
	}
	return b[i].timestamp.After(b[j].timestamp)
}

func (b logInfosWithTimestamp) Swap(i, j int) {
	if i >= b.Len() || j >= b.Len() {
		return
	}
	b[i], b[j] = b[j], b[i]
}

func (b logInfosWithTimestamp) Len() int {
	return len(b)
}

// openLogFile returns an opened file handle and transfers ownership to caller.
func openLogFile(name string, flag int, perm os.FileMode) (*os.File, error) {
	file, err := os.OpenFile(name, flag, perm)
	if err != nil {
		return nil, err
	}
	return file, nil
}

func (l *FileLogger) Write(bytes []byte) (int, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	writeLen := int64(len(bytes))
	if writeLen > l.maxLogContextLen() {
		return 0, fmt.Errorf("write length %d exceeds maximum file writtenLen %d", writeLen, l.maxLogContextLen())
	}

	if l.file == nil {
		if err := l.openExistingOrNew(len(bytes)); err != nil {
			return 0, err
		}
	}

	if l.writtenLen+writeLen >= l.maxLogContextLen() {
		if err := l.rotate(); err != nil {
			return 0, err
		}
	}

	n, err := l.file.Write(bytes)
	l.writtenLen += int64(n)
	return n, err
}

func (l *FileLogger) Close() error {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	return l.close()
}

func (l *FileLogger) close() error {
	if l.file == nil {
		return nil
	}
	err := l.file.Close()
	l.file = nil
	return err
}

func (l *FileLogger) Rotate() error {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	return l.rotate()
}

func (l *FileLogger) rotate() error {
	if err := l.close(); err != nil {
		return err
	}
	if err := l.openNew(); err != nil {
		return err
	}

	l.postRotateCompressLog()
	return nil
}

// openNew creates a new log file and renames the old one
func (l *FileLogger) openNew() error {
	err := os.MkdirAll(l.dir(), dirModeUserRwxForLogDir)
	if err != nil {
		return fmt.Errorf("can't make directories for new logfile: %s", err)
	}

	name := l.filename()
	info, err := os.Stat(name)
	if err == nil {
		newName := backupName(name, l.localTime)
		if err := os.Rename(name, newName); err != nil {
			return fmt.Errorf("can't rename log file: %s", err)
		}

		err = os.Chmod(newName, os.FileMode(fileModeUserRoForArchiveLog))
		if err != nil {
			return fmt.Errorf("error changing permissions: %v", err)
		}

		if err := createAndChown(name, info); err != nil {
			return err
		}
	}

	f, err := openLogFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(fileModeUserRwForWritingLog))
	if err != nil {
		return fmt.Errorf("can't open new logfile: %s", err)
	}
	l.file = f
	l.writtenLen = 0
	return nil
}

func backupName(name string, local bool) string {
	dir := filepath.Dir(name)
	fileName := filepath.Base(name)
	ext := filepath.Ext(fileName)
	prefix := fileName[:len(fileName)-len(ext)]
	t := currentTime()
	if !local {
		t = t.UTC()
	}
	timestamp := t.Format(backupTimeFormat)
	return filepath.Join(dir, fmt.Sprintf("%s-%s%s", prefix, timestamp, ext))
}

// openExistingOrNew opens existing log file or creates a new one
func (l *FileLogger) openExistingOrNew(writeLen int) error {
	l.postRotateCompressLog()

	filename := l.filename()

	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return l.openNew()
	}
	if err != nil {
		return fmt.Errorf("error getting log file info: %s", err)
	}

	if info.Size()+int64(writeLen) >= l.maxLogContextLen() {
		return l.rotate()
	}

	file, err := openLogFile(filename, os.O_APPEND|os.O_WRONLY, os.FileMode(fileModeUserRwForWritingLog))
	if err != nil {
		return l.openNew()
	}
	l.file = file
	l.writtenLen = info.Size()
	return nil
}

func (l *FileLogger) filename() string {
	l.fileName = filepath.Clean(l.fileName)
	isAbsolute := filepath.IsAbs(l.fileName)
	if !isAbsolute {
		l.fileName = filepath.Join(defaultLogPath, filepath.Base(l.fileName))
	}
	return l.fileName
}

// handleStaleLogRunOnce processes old log files for compression and deletion
func (l *FileLogger) handleStaleLogRunOnce() error {
	if l.maxBackups == 0 && l.maxAge == 0 {
		return nil
	}

	files, err := l.oldLogFiles()
	if err != nil {
		return err
	}

	preserved := make(map[string]bool)
	cutoff := time.Now().Add(-time.Duration(l.maxAge) * hourPerDay * time.Hour)
	var toCompress, toRemove []logInfo
	for _, f := range files {
		baseName := f.Name()
		isCompressed := strings.HasSuffix(baseName, compressSuffix)
		if isCompressed {
			baseName = baseName[:len(baseName)-len(compressSuffix)]
		}

		if (l.maxBackups != 0 && len(preserved) >= l.maxBackups) || (l.maxAge != 0 && f.timestamp.Before(cutoff)) {
			toRemove = append(toRemove, f)
		} else {
			preserved[baseName] = true
			if !isCompressed {
				toCompress = append(toCompress, f)
			}
		}
	}

	for _, f := range toRemove {
		if errRemove := os.Remove(filepath.Join(l.dir(), f.Name())); errRemove != nil {
			err = errRemove
		}
	}

	for _, f := range toCompress {
		fn := filepath.Join(l.dir(), f.Name())
		if errCompress := compressLogFile(fn, fn+compressSuffix); errCompress != nil {
			err = errCompress
		}
	}
	return err
}

func (l *FileLogger) handleStaleLogRun() {
	for range l.staleLogCh {
		err := l.handleStaleLogRunOnce()
		if err != nil {
			fmt.Errorf("handle stale log err: %v", err)
		}
	}
}

func (l *FileLogger) postRotateCompressLog() {
	l.startHandleStaleLog.Do(func() {
		l.staleLogCh = make(chan bool, 1)
		go l.handleStaleLogRun()
	})
	select {
	case l.staleLogCh <- true:
	default:
	}
}

// oldLogFiles returns a list of old log files sorted by timestamp
func (l *FileLogger) oldLogFiles() ([]logInfo, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	files, err := os.ReadDir(l.dir())
	if err != nil {
		return nil, fmt.Errorf("can't read log file directory: %s", err)
	}
	var logFiles []logInfo

	prefix, ext := l.getLogFilePrefixAndExt()
	for _, f := range files {
		fileInfo, fileErr := f.Info()
		if fileErr != nil {
			fmt.Printf("can't read log file info: %s\n", fileErr)
		}
		if fileInfo.IsDir() {
			continue
		}
		if t, err := l.extractTimeFromFileName(f.Name(), prefix, ext); err == nil {
			logFiles = append(logFiles, logInfo{timestamp: t, FileInfo: fileInfo})
			continue
		}

		if t, err := l.extractTimeFromFileName(f.Name(), prefix, ext+compressSuffix); err == nil {
			logFiles = append(logFiles, logInfo{timestamp: t, FileInfo: fileInfo})
			continue
		}
	}
	sort.Sort(logInfosWithTimestamp(logFiles))
	return logFiles, nil
}

// extractTimeFromFileName parses timestamp from backup filename
func (l *FileLogger) extractTimeFromFileName(filename, prefix, ext string) (time.Time, error) {
	if !strings.HasPrefix(filename, prefix) {
		return time.Time{}, errors.New("mismatched prefix")
	}
	if !strings.HasSuffix(filename, ext) {
		return time.Time{}, errors.New("mismatched extension")
	}
	ts := filename[len(prefix) : len(filename)-len(ext)]
	return time.Parse(backupTimeFormat, ts)
}

func (l *FileLogger) maxLogContextLen() int64 {
	if l.maxSize == 0 {
		return int64(defaultLogMaxSize * megabyte)
	}
	return int64(l.maxSize) * int64(megabyte)
}

func (l *FileLogger) dir() string {
	return filepath.Dir(l.filename())
}

// getLogFilePrefixAndExt extracts prefix and extension from log filename
func (l *FileLogger) getLogFilePrefixAndExt() (prefix, ext string) {
	filename := filepath.Base(l.filename())
	ext = filepath.Ext(filename)
	prefix = filename[:len(filename)-len(ext)] + "-"
	return prefix, ext
}

// createByOpen creates a file with the same mode as the original
func createByOpen(name string, info os.FileInfo) error {
	f, err := os.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode())
	if err != nil {
		return err
	}
	defer f.Close()
	return nil
}

func createAndChown(name string, info os.FileInfo) error {
	if err := createByOpen(name, info); err != nil {
		return err
	}

	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("file info does not contain syscall.Stat_t")
	}
	return os.Chown(name, int(stat.Uid), int(stat.Gid))
}

// compressLogFile compresses a log file using gzip
func compressLogFile(src, dst string) (err error) {
	srcFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("open log file failed: %v", err)
	}
	defer srcFile.Close()

	srcFileInfo, err := os.Stat(src)
	if err != nil {
		return fmt.Errorf("stat log file failed: %v", err)
	}

	if err := createAndChown(dst, srcFileInfo); err != nil {
		return fmt.Errorf("createAndChown compressed log file failed: %v", err)
	}

	gzWriter, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, srcFileInfo.Mode())
	if err != nil {
		return fmt.Errorf("open compressed log file failed: %v", err)
	}
	defer gzWriter.Close()

	gzFile := gzip.NewWriter(gzWriter)

	defer func() {
		if err != nil {
			if errRemove := os.Remove(dst); errRemove == nil {
				err = fmt.Errorf("compress log file failed: %v", err)
			} else {
				err = fmt.Errorf("compress log file failed: %v, Remove %s failed too", err, dst)
			}
		}
	}()

	if _, err := io.Copy(gzFile, srcFile); err != nil {
		return err
	}
	if err := gzFile.Close(); err != nil {
		return err
	}
	if err := gzWriter.Close(); err != nil {
		return err
	}
	if err := srcFile.Close(); err != nil {
		return err
	}
	if err := os.Remove(src); err != nil {
		return err
	}
	return nil
}
