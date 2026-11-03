/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package log

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/sirupsen/logrus"
)

const (
	byteMb      = 1024 * 1024
	delayRemove = 10
)

func cleanup(fileName string, t *testing.T) {
	<-time.After(time.Millisecond * delayRemove)
	if err := os.Remove(fileName); err != nil {
		t.Errorf("Error remove file: %v\n", err)
	} else {
		t.Log("File Removed.")
	}
}

func TestDefaultLogger(t *testing.T) {
	megabyte = byteMb
	logFile := "/tmp/devicePlugin.log"
	defer cleanup(logFile, t)

	err := InitLogging(logFile)
	assertIsNil(err, t)

	strTest := "TestErr"
	Errorln(strTest)
	info, err := os.Stat(logFile)
	assertIsNil(err, t)
	assertEquals(true, info.Size() > int64(len(strTest)), t)

	Errorf("%s", strTest)
	info1, err := os.Stat(logFile)
	assertIsNil(err, t)
	assertEquals(true, info1.Size() > (info.Size()+int64(len(strTest))), t)
}

func TestPrintLevelLogger(t *testing.T) {
	megabyte = byteMb
	logFile := "/tmp/devicePlugin1.log"
	defer cleanup(logFile, t)

	err := InitLogging(logFile)
	assertIsNil(err, t)
	*logLevel = "info"

	strTest := "TestErr"
	Debugf("%s", strTest)
	info, err := os.Stat(logFile)
	assertNotNil(err, t)

	Infof("%s", strTest)
	info, err = os.Stat(logFile)
	assertIsNil(err, t)
	assertEquals(true, info.Size() > int64(len(strTest)), t)
}

func GetFileContent(fileName string) (string, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return "", err
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

func TestConsoleLoggerByWarning(t *testing.T) {
	megabyte = byteMb
	logFile := "/tmp/devicePlugin2.log"
	defer cleanup(logFile, t)

	*loggingConsole = true
	err := InitLogging(logFile)
	assertIsNil(err, t)

	strTest := "TestErr"
	patch := gomonkey.ApplyFunc((*ConsoleHook).Fire, func(c *ConsoleHook, entry *logrus.Entry) error {
		assertEquals(strTest, entry.Message, t)
		return nil
	})
	defer patch.Reset()

	Warningln(strTest)
	info, err := os.Stat(logFile)
	assertIsNil(err, t)
	assertEquals(true, info.Size() > int64(len(strTest)), t)

	content, err := GetFileContent(logFile)
	assertIsNil(err, t)
	assertEquals(true, strings.Contains(content, strTest), t)
}

func TestConsoleLoggerOutput(t *testing.T) {
	megabyte = byteMb
	logFile := "/tmp/devicePlugin3.log"
	defer cleanup(logFile, t)

	*loggingConsole = true
	err := InitLogging(logFile)
	assertIsNil(err, t)

	tempFile, err := os.CreateTemp("", "stdout-capture-")
	if err != nil {
		t.Fatalf("Failed to create temporary file: %v", err)
	}

	defer cleanup(tempFile.Name(), t)
	oldStdout := os.Stdout
	os.Stdout = tempFile

	strTest := "TestErr111"
	Warningln(strTest)

	os.Stdout = oldStdout
	fileName := tempFile.Name()
	closeErr := tempFile.Close()
	if closeErr != nil {
		t.Logf("Failed to close temporary file: %v", err)
	}

	context, err := GetFileContent(fileName)
	assertIsNil(err, t)
	assertEquals(true, strings.Contains(context, strTest), t)

	info, err := os.Stat(logFile)
	assertIsNil(err, t)
	assertEquals(true, info.Size() > int64(len(strTest)), t)
}

func TestCheckLoggerParamValid(t *testing.T) {
	origMaxAges := *maxAges
	origMaxBackups := *maxBackups
	defer func() {
		*maxAges = origMaxAges
		*maxBackups = origMaxBackups
	}()

	*maxAges = 0
	if err := checkLoggerParamValid(); err == nil {
		t.Errorf("expected error for invalid max-ages")
	}

	*maxAges = 30
	*maxBackups = 0
	if err := checkLoggerParamValid(); err == nil {
		t.Errorf("expected error for invalid max-backups")
	}

	*maxBackups = 10
	if err := checkLoggerParamValid(); err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestParseLogLevel(t *testing.T) {
	origLogLevel := *logLevel
	defer func() { *logLevel = origLogLevel }()

	levels := map[string]logrus.Level{
		"debug":   logrus.DebugLevel,
		"info":    logrus.InfoLevel,
		"warning": logrus.WarnLevel,
		"error":   logrus.ErrorLevel,
		"fatal":   logrus.FatalLevel,
	}
	for str, expected := range levels {
		*logLevel = str
		level, err := parseLogLevel()
		if err != nil {
			t.Errorf("unexpected error for %s: %v", str, err)
		}
		if level != expected {
			t.Errorf("expected %v for %s, got %v", expected, str, level)
		}
	}

	*logLevel = "invalid"
	_, err := parseLogLevel()
	if err == nil {
		t.Errorf("expected error for invalid log level")
	}
}

func TestGetLogLevel(t *testing.T) {
	cases := []struct {
		level    logrus.Level
		expected string
	}{
		{logrus.DebugLevel, "[DEBUG]: "},
		{logrus.InfoLevel, "[INFO]: "},
		{logrus.WarnLevel, "[WARNING]: "},
		{logrus.ErrorLevel, "[ERROR]: "},
		{logrus.FatalLevel, "[FATAL]: "},
		{logrus.TraceLevel, "[UNKNOWN]: "},
	}
	for _, tc := range cases {
		result := getLogLevel(tc.level)
		if result != tc.expected {
			t.Errorf("getLogLevel(%v) = %q, want %q", tc.level, result, tc.expected)
		}
	}
}

func TestFormat(t *testing.T) {
	f := &PlainTextFormatter{
		TimestampFormat: "2006-01-02 15:04:05.000",
		pid:             12345,
	}
	entry := &logrus.Entry{
		Message: "test message",
		Level:   logrus.InfoLevel,
		Data:    logrus.Fields{"key": "value"},
	}
	var err error
	entry.Time, err = time.Parse("2006-01-02 15:04:05.000", "2025-01-01 12:00:00.000")
	if err != nil {
		t.Fatalf("parse time failed: %v", err)
	}

	buf := &bytes.Buffer{}
	entry.Buffer = buf

	data, err := f.Format(entry)
	if err != nil {
		t.Errorf("Format failed: %v", err)
	}
	output := string(data)
	if !strings.Contains(output, "test message") || !strings.Contains(output, "[key:value]") ||
		!strings.Contains(output, "[INFO]: ") {
		t.Errorf("unexpected output: %s", output)
	}
}

func TestFormat_NilBuffer(t *testing.T) {
	f := &PlainTextFormatter{
		TimestampFormat: "2006-01-02 15:04:05.000",
		pid:             12345,
	}
	entry := logrus.NewEntry(logrus.New())
	entry.Message = "test"
	entry.Level = logrus.DebugLevel
	var err error
	entry.Time, err = time.Parse("2006-01-02 15:04:05.000", "2025-01-01 12:00:00.000")
	if err != nil {
		t.Fatalf("parse time failed: %v", err)
	}

	data, err := f.Format(entry)
	if err != nil {
		t.Errorf("Format with nil buffer failed: %v", err)
	}
	if !strings.Contains(string(data), "test") {
		t.Errorf("expected output to contain message")
	}
}

func TestDebugln(t *testing.T) {
	patch := gomonkey.ApplyMethodFunc(logger, "Debugln", func(args ...interface{}) {
		// Mock implementation: silently swallow the log call to avoid side effects.
	})
	defer patch.Reset()
	Debugln("test debug")
	// 验证 patched wrapper 函数可正常调用，无需额外断言
}

func TestInfoln(t *testing.T) {
	patch := gomonkey.ApplyMethodFunc(logger, "Infoln", func(args ...interface{}) {
		// Mock implementation: silently swallow the log call to avoid side effects.
	})
	defer patch.Reset()
	Infoln("test info")
	// 验证 patched wrapper 函数可正常调用，无需额外断言
}

func TestWarningf(t *testing.T) {
	patch := gomonkey.ApplyMethodFunc(logger, "Warningf", func(format string, args ...interface{}) {
		// Mock implementation: silently swallow the format-based log call to avoid side effects.
	})
	defer patch.Reset()
	Warningf("test warning %d", 1)
	// 验证 patched wrapper 函数可正常调用，无需额外断言
}

func TestFatalln(t *testing.T) {
	patch := gomonkey.ApplyMethodFunc(logger, "Fatalln", func(args ...interface{}) {
		// Mock implementation: silently swallow the fatal-level log call to avoid os.Exit.
	})
	defer patch.Reset()
	Fatalln("test fatal")
	// 验证 patched wrapper 函数可正常调用，无需额外断言
}

func TestFatalf(t *testing.T) {
	patch := gomonkey.ApplyMethodFunc(logger, "Fatalf", func(format string, args ...interface{}) {
		// Mock implementation: silently swallow the format-based fatal log to avoid os.Exit.
	})
	defer patch.Reset()
	Fatalf("test fatal %d", 1)
	// 验证 patched wrapper 函数可正常调用，无需额外断言
}

func TestPanicln(t *testing.T) {
	patch := gomonkey.ApplyMethodFunc(logger, "Panicln", func(args ...interface{}) {
		// Mock implementation: silently swallow the panic-level log call to avoid test panic.
	})
	defer patch.Reset()
	Panicln("test panic")
	// 验证 patched wrapper 函数可正常调用，无需额外断言
}

func TestInitLoggingInvalidParam(t *testing.T) {
	oldMaxAges := *maxAges
	oldMaxBackups := *maxBackups
	defer func() {
		*maxAges = oldMaxAges
		*maxBackups = oldMaxBackups
	}()
	*maxAges = 0
	*maxBackups = 10
	fatallnCalled := false
	patchFatalln := gomonkey.ApplyFunc(Fatalln, func(_ ...interface{}) { fatallnCalled = true })
	defer patchFatalln.Reset()
	InitLogging("/tmp/test-param.log")
	if !fatallnCalled {
		t.Errorf("expected Fatalln to be called for invalid param")
	}
}

func TestInitLoggingParseLevelError(t *testing.T) {
	oldLogLevel := *logLevel
	oldMaxAges := *maxAges
	oldMaxBackups := *maxBackups
	defer func() {
		*logLevel = oldLogLevel
		*maxAges = oldMaxAges
		*maxBackups = oldMaxBackups
	}()
	*maxAges = 30
	*maxBackups = 10
	*logLevel = "invalid-level"
	err := InitLogging("/tmp/test-level.log")
	if err == nil {
		t.Errorf("expected error from parseLogLevel")
	}
}

func TestFormatTimestampError(t *testing.T) {
	f := &PlainTextFormatter{TimestampFormat: "2006-01-02 15:04:05.000", pid: 12345}
	entry := &logrus.Entry{Message: "test", Level: logrus.InfoLevel}
	var err error
	entry.Time, err = time.Parse("2006-01-02 15:04:05.000", "2025-01-01 12:00:00.000")
	if err != nil {
		t.Fatalf("parse time failed: %v", err)
	}
	entry.Buffer = &bytes.Buffer{}
	patch := gomonkey.ApplyFunc(fmt.Fprintf, func(_ io.Writer, _ string, _ ...interface{}) (int, error) {
		return 0, errors.New("mock fprintf error")
	})
	defer patch.Reset()
	_, err = f.Format(entry)
	if err == nil {
		t.Errorf("expected error from timestamp fprintf")
	}
}

func TestFormatDataError(t *testing.T) {
	f := &PlainTextFormatter{TimestampFormat: "2006-01-02 15:04:05.000", pid: 12345}
	entry := &logrus.Entry{Message: "test", Level: logrus.InfoLevel, Data: logrus.Fields{"k": "v"}}
	var err error
	entry.Time, err = time.Parse("2006-01-02 15:04:05.000", "2025-01-01 12:00:00.000")
	if err != nil {
		t.Fatalf("parse time failed: %v", err)
	}
	entry.Buffer = &bytes.Buffer{}
	calls := 0
	patch := gomonkey.ApplyFunc(fmt.Fprintf, func(_ io.Writer, _ string, _ ...interface{}) (int, error) {
		calls++
		if calls == 2 {
			return 0, errors.New("mock data fprintf error")
		}
		return 0, nil
	})
	defer patch.Reset()
	_, err = f.Format(entry)
	if err == nil {
		t.Errorf("expected error from data fprintf")
	}
}

func TestFormatMessageError(t *testing.T) {
	f := &PlainTextFormatter{TimestampFormat: "2006-01-02 15:04:05.000", pid: 12345}
	entry := &logrus.Entry{Message: "test", Level: logrus.InfoLevel}
	var err error
	entry.Time, err = time.Parse("2006-01-02 15:04:05.000", "2025-01-01 12:00:00.000")
	if err != nil {
		t.Fatalf("parse time failed: %v", err)
	}
	entry.Buffer = &bytes.Buffer{}
	calls := 0
	patch := gomonkey.ApplyFunc(fmt.Fprintf, func(_ io.Writer, _ string, _ ...interface{}) (int, error) {
		calls++
		if calls == 2 {
			return 0, errors.New("mock message fprintf error")
		}
		return 0, nil
	})
	defer patch.Reset()
	_, err = f.Format(entry)
	if err == nil {
		t.Errorf("expected error from message fprintf")
	}
}
