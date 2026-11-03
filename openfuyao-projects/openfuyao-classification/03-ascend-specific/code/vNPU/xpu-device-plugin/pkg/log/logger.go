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
	"bytes"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/sirupsen/logrus"
)

var (
	logger         = logrus.New()
	loggingConsole = flag.Bool("logging-console", false,
		"Enable console log output, default false")
	logLevel = flag.String("log-level", "info",
		"Set logging level (debug, info, error, warning, fatal)")
	maxSize = flag.Int("max-size", defaultMaxSize,
		"maximum length of a log(MB).")
	maxBackups = flag.Int("max-backups", defaultMaxBackups,
		"maximum number of backup log file")
	maxAges = flag.Int("max-ages", defaultMaxAges,
		"maximum storage duration, in days")
)

const (
	defaultMaxSize    = 20
	defaultMaxBackups = 10
	defaultMaxAges    = 30
	timestampFormat   = "2006-01-02 15:04:05.000"
)

func checkLoggerParamValid() error {
	if *maxAges <= 0 {
		return errors.New("the max-ages is invalid")
	}
	if *maxBackups <= 0 {
		return errors.New("the max-backups is invalid")
	}
	return nil
}

func InitLogging(logName string) error {
	if err := checkLoggerParamValid(); err != nil {
		Fatalln(err)
	}
	logFileOutput := FileLogger{
		fileName:   logName,
		maxSize:    *maxSize,
		maxBackups: *maxBackups,
		maxAge:     *maxAges,
	}

	logger.SetOutput(&logFileOutput)

	level, err := parseLogLevel()
	if err != nil {
		return err
	}
	logger.SetLevel(level)
	formatter := &PlainTextFormatter{TimestampFormat: timestampFormat, pid: os.Getpid()}
	logger.SetFormatter(formatter)

	if *loggingConsole {
		logConsoleHook, err := newConsoleHook(formatter)
		if err != nil {
			return fmt.Errorf("could not initialize logging to console: %v", err)
		}
		logger.AddHook(logConsoleHook)
	}
	return nil
}

func parseLogLevel() (logrus.Level, error) {
	switch *logLevel {
	case "debug":
		return logrus.DebugLevel, nil
	case "info":
		return logrus.InfoLevel, nil
	case "warning":
		return logrus.WarnLevel, nil
	case "error":
		return logrus.ErrorLevel, nil
	case "fatal":
		return logrus.FatalLevel, nil
	default:
		return logrus.FatalLevel, fmt.Errorf("invalid logging level [%v]", logLevel)
	}
}

type PlainTextFormatter struct {
	TimestampFormat string
	pid             int
}

var _ logrus.Formatter = &PlainTextFormatter{}

func (f *PlainTextFormatter) Format(entry *logrus.Entry) ([]byte, error) {
	b := entry.Buffer
	if entry.Buffer == nil {
		b = &bytes.Buffer{}
	}

	if _, err := fmt.Fprintf(b, "%s %d", entry.Time.Format(f.TimestampFormat), f.pid); err != nil {
		return nil, err
	}
	if len(entry.Data) != 0 {
		for key, value := range entry.Data {
			if _, err := fmt.Fprintf(b, "[%s:%v] ", key, value); err != nil {
				return nil, err
			}
		}
	}

	if _, err := fmt.Fprintf(b, "%s %s\n", getLogLevel(entry.Level), entry.Message); err != nil {
		return nil, err
	}

	return b.Bytes(), nil
}

func getLogLevel(level logrus.Level) string {
	switch level {
	case logrus.DebugLevel:
		return "[DEBUG]: "
	case logrus.InfoLevel:
		return "[INFO]: "
	case logrus.WarnLevel:
		return "[WARNING]: "
	case logrus.ErrorLevel:
		return "[ERROR]: "
	case logrus.FatalLevel:
		return "[FATAL]: "
	default:
		return "[UNKNOWN]: "
	}
}

func Debugf(format string, args ...interface{}) {
	logger.Debugf(format, args...)
}

func Debugln(args ...interface{}) {
	logger.Debugln(args...)
}

func Infof(format string, args ...interface{}) {
	logger.Infof(format, args...)
}

func Infoln(args ...interface{}) {
	logger.Infoln(args...)
}

func Warningf(format string, args ...interface{}) {
	logger.Warningf(format, args...)
}

func Warningln(args ...interface{}) {
	logger.Warningln(args...)
}

func Errorf(format string, args ...interface{}) {
	logger.Errorf(format, args...)
}
func Errorln(args ...interface{}) {
	logger.Errorln(args...)
}

func Fatalf(format string, args ...interface{}) {
	logger.Fatalf(format, args...)
}

func Fatalln(args ...interface{}) {
	logger.Fatalln(args...)
}

func Panicln(args ...interface{}) {
	logger.Panicln(args...)
}
