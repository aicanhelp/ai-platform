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
	"fmt"
	"io"
	"os"

	"github.com/sirupsen/logrus"
)

type ConsoleHook struct {
	formatter logrus.Formatter
}

func newConsoleHook(logFormat logrus.Formatter) (*ConsoleHook, error) {
	return &ConsoleHook{formatter: logFormat}, nil
}

func (hook *ConsoleHook) Levels() []logrus.Level {
	return logrus.AllLevels
}

func (hook *ConsoleHook) Fire(logEntry *logrus.Entry) error {
	var logWriter io.Writer
	switch logEntry.Level {
	case logrus.DebugLevel, logrus.InfoLevel, logrus.WarnLevel:
		logWriter = os.Stdout
	case logrus.ErrorLevel, logrus.FatalLevel:
		logWriter = os.Stderr
	default:
		return fmt.Errorf("unknown log level: %v", logEntry.Level)
	}

	lineBytes, err := hook.formatter.Format(logEntry)
	if err != nil {
		_, printErr := fmt.Fprintf(os.Stderr, "Unable to read logEntry, %v", err)
		if printErr != nil {
			return fmt.Errorf("print os.Stderr failed: %v", printErr)
		}
		return err
	}

	if _, err := logWriter.Write(lineBytes); err != nil {
		return err
	}

	return nil
}
