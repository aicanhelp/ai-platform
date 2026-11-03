/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/sirupsen/logrus"
)

func TestConsoleHookFire_ErrorLevel(t *testing.T) {
	hook, err := newConsoleHook(&PlainTextFormatter{})
	if err != nil {
		t.Fatal(err)
	}
	entry := &logrus.Entry{Message: "test error", Level: logrus.ErrorLevel}
	err = hook.Fire(entry)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestConsoleHookFire_UnknownLevel(t *testing.T) {
	hook, err := newConsoleHook(&PlainTextFormatter{})
	if err != nil {
		t.Fatal(err)
	}
	entry := &logrus.Entry{Message: "test", Level: logrus.TraceLevel}
	err = hook.Fire(entry)
	if err == nil {
		t.Errorf("expected error for unknown level")
	}
}

func TestConsoleHookFire_FormatError(t *testing.T) {
	formatter := &PlainTextFormatter{}
	hook, err := newConsoleHook(formatter)
	if err != nil {
		t.Fatal(err)
	}
	patch := gomonkey.ApplyMethodFunc(formatter, "Format", func(_ *logrus.Entry) ([]byte, error) {
		return nil, fmt.Errorf("mock format error")
	})
	defer patch.Reset()
	entry := &logrus.Entry{Message: "test", Level: logrus.InfoLevel}
	err = hook.Fire(entry)
	if err == nil {
		t.Errorf("expected error from format failure")
	}
}

func TestConsoleHookFire_WriteError(t *testing.T) {
	formatter := &PlainTextFormatter{}
	hook, err := newConsoleHook(formatter)
	if err != nil {
		t.Fatal(err)
	}
	patch := gomonkey.ApplyMethodFunc(formatter, "Format", func(_ *logrus.Entry) ([]byte, error) {
		return []byte("test"), nil
	})
	defer patch.Reset()
	patchWrite := gomonkey.ApplyMethodFunc(&os.File{}, "Write", func(_ []byte) (int, error) {
		return 0, errors.New("mock write error")
	})
	defer patchWrite.Reset()
	entry := &logrus.Entry{Message: "test", Level: logrus.InfoLevel}
	err = hook.Fire(entry)
	if err == nil {
		t.Errorf("expected error from write failure")
	}
}

func TestConsoleHookFire_FormatErrorWithStderrFailure(t *testing.T) {
	formatter := &PlainTextFormatter{}
	hook, err := newConsoleHook(formatter)
	if err != nil {
		t.Fatal(err)
	}
	patch := gomonkey.ApplyMethodFunc(formatter, "Format", func(_ *logrus.Entry) ([]byte, error) {
		return nil, errors.New("mock format error")
	})
	defer patch.Reset()
	patchStderr := gomonkey.ApplyMethodFunc(&os.File{}, "Write", func(_ []byte) (int, error) {
		return 0, errors.New("stderr write error")
	})
	defer patchStderr.Reset()
	entry := &logrus.Entry{Message: "test", Level: logrus.ErrorLevel}
	err = hook.Fire(entry)
	if err == nil {
		t.Errorf("expected error from stderr write failure")
	}
}

func TestConsoleHookFire_Levels(t *testing.T) {
	hook, err := newConsoleHook(&PlainTextFormatter{})
	if err != nil {
		t.Fatal(err)
	}
	levels := hook.Levels()
	if len(levels) == 0 {
		t.Errorf("expected non-empty levels")
	}
}
