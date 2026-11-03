/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1

import (
	"regexp"
	"testing"
)

var errorRegexTest = `^(` +
	`\[ERROR\]|` +
	`\bERROR\b|` +
	`\bERR\b|` +
	`<Error>|` +
	`ERROR:|` +
	`Error:|` +
	`error:|` +
	`level=Error|` +
	`level=error|` +
	`level=ERROR|` +
	`\bErr:\b|` +
	`\bERR:\b)`

var warningRegexTest = `^(` +
	`\[WARNING\]|` +
	`\bWARN\b|` +
	`\bWRN\b|` +
	`<Warning>|` +
	`WARN:|` +
	`WARNING:|` +
	`Warning:|` +
	`warning:|` +
	`Warn:|` +
	`warn:|` +
	`level=Warn|` +
	`level=warning|` +
	`level=warn|` +
	`level=WARNING|` +
	`\bWrn\b|` +
	`WRN:)`

var infoRegexTest = `^(` +
	`\[INFO\]|` +
	`\bINFO\b|` +
	`<Info>|` +
	`INFO:|` +
	`Info:|` +
	`info:|` +
	`level=Info|` +
	`level=info|` +
	`level=INFO)`

var debugRegexTest = `^(` +
	`\[DEBUG\]|` +
	`\bDEBUG\b|` +
	`\bDBG\b|` +
	`<Debug>|` +
	`DEBUG:|` +
	`Debug:|` +
	`Debugging:|` +
	`level=Debug|` +
	`level=debug|` +
	`level=DEBUG|` +
	`\bDbg\b|` +
	`\bDBG:\b)`

var criticalRegexTest = `^(` +
	`\[CRITICAL\]|` +
	`\bCRITICAL\b|` +
	`\bCRIT\b|` +
	`<Critical>|` +
	`CRITICAL:|` +
	`Critical:|` +
	`level=critical|` +
	`level=Critical|` +
	`level=CRITICAL|` +
	`\bCRIT:\b)`

var logLevelRegexTest = map[string]string{
	"error":    errorRegexTest,
	"warning":  warningRegexTest,
	"info":     infoRegexTest,
	"debug":    debugRegexTest,
	"critical": criticalRegexTest,
}

func TestLogLevelRegex(t *testing.T) {
	tests := []struct {
		level    string
		input    string
		expected bool
	}{
		// Error level detailed tests
		{"error", "[ERROR] An error occurred", true},
		{"error", "ERROR: Specific error occurred", true},
		{"error", "Unexpected ERROR in system", false},
		{"error", "level=Error detected", true},
		{"error", "Critical system error", false},
		{"error", "Handle with care: ERROR:", false},
		{"error", "System threw an ERROR:", false},
		{"error", "<Error> issue found", true},
		{"error", "An error (ERROR:) occurred", false},
		{"error", "ERROR:", true},
		{"error", "Not an error just a test", false},
		{"error", "Error-prone not an error log", false},
		{"error", "A sentence with error not logged", false},
		{"error", "level=error Simply mentioning error should not match", true},
		{"error", "Included in sentence: error does not qualify", false},
		{"error", "Looking erroneous not an error", false},
		{"error", "Warning can contain error but not an error log", false},
		{"error", "Information might have error but this is info log", false},
		{"error", "Debugging error isn't necessarily an error log", false},
		{"error", "Critically errored out still not an error log", false},

		// Info level detailed tests
		{"info", "[INFO] Information available here", true},
		{"info", "INFO: This is an info log", true},
		{"info", "System info: All systems go", false},
		{"info", "level=Info general info", true},
		{"info", "<Info> General information", true},
		{"info", "Info: your application is updated", true},
		{"info", "General information provided", false},
		{"info", "Informational [INFO] message", false},
		{"info", "INFO should match this", true},
		{"info", "Just an informational log", false},
		{"info", "Not really info just a context", false},
		{"info", "Context with info not an info log", false},
		{"info", "Mentioning INFO should not match", false},
		{"info", "Info about something else", false},
		{"info", "Random information not related", false},
		{"info", "Detailing information without context", false},
		{"info", "Error log with info not an info", false},
		{"info", "Warning with info not an info log", false},
		{"info", "Debug info not really an info", false},
		{"info", "Critical info not an info log", false},

		// Debug level detailed tests
		{"debug", "[DEBUG] Debugging started", true},
		{"debug", "DEBUG: Step into this section", true},
		{"debug", "Debugging: Check this variable", true},
		{"debug", "level=Debug detailed trace", true},
		{"debug", "<Debug> Tracing actions", true},
		{"debug", "Debug: execution point here", true},
		{"debug", "Detailed debugging session", false},
		{"debug", "Debug mode: [DEBUG] enabled", false},
		{"debug", "DEBUG should catch this", true},
		{"debug", "Just debugging the process", false},
		{"debug", "Not debugging just mentioning", false},
		{"debug", "Context with debug not a log", false},
		{"debug", "Mentioning DEBUG should not match", false},
		{"debug", "Debug about something else", false},
		{"debug", "Random debugging not related", false},
		{"debug", "Detailing debugging without context", false},
		{"debug", "Error log with debug not a debug", false},
		{"debug", "Warning with debug not a debug log", false},
		{"debug", "Info debug not really a debug", false},
		{"debug", "Critical debug not a debug log", false},

		// Critical level detailed tests
		{"critical", "[CRITICAL] System failure imminent", true},
		{"critical", "CRITICAL: Major component failure", true},
		{"critical", "Critical error detected", false},
		{"critical", "level=Critical severity high", true},
		{"critical", "<Critical> System halt", true},
		{"critical", "Critical: system is down", true},
		{"critical", "System critical condition", false},
		{"critical", "Critical issue: [CRITICAL] detected", false},
		{"critical", "CRITICAL should respond", true},
		{"critical", "Just a critical warning", false},
		{"critical", "Not critical just severe", false},
		{"critical", "Context with critical not a log", false},
		{"critical", "Mentioning CRITICAL should not match", false},
		{"critical", "Critical about something else", false},
		{"critical", "Random critical info not related", false},
		{"critical", "Detailing critical without context", false},
		{"critical", "Error log with critical not a critical", false},
		{"critical", "Warning with critical not a critical log", false},
		{"critical", "Info critical not really a critical", false},
		{"critical", "Debug critical not a critical log", false},

		// Warning level detailed tests
		{"warning", "[WARNING] Attention required", true},
		{"warning", "WARNING: Low disk space", true},
		{"warning", "System warning: High CPU usage", false},
		{"warning", "level=Warning, potential issue", true},
		{"warning", "<Warning> Low memory warning", true},
		{"warning", "Warn: This is a warning log", true},
		{"warning", "Warning state reached on device", false},
		{"warning", "level=WARNING: Threshold crossed", true},
		{"warning", "A warning about system performance", false},
		{"warning", "High temperature warning for system", false},
		{"warning", "This is not a warning just a note", false},
		{"warning", "The term warning should not match here", false},
		{"warning", "Information about system, not a warning", false},
		{"warning", "No warning just info", false},
		{"warning", "Debugging logs, not warning", false},
		{"warning", "Critical issues, but no warning here", false},
		{"warning", "Error reported, not a warning", false},
		{"warning", "Regular operation, no warnings", false},
		{"warning", "Routine update, should not trigger warnings", false},
		{"warning", "Unrelated data, warning should not match", false},
	}

	for _, test := range tests {
		pattern := logLevelRegexTest[test.level]

		r, err := regexp.Compile(pattern)
		if err != nil {
			t.Fatalf("Failed to compile regex %q: %v", pattern, err)
		}
		if match := r.MatchString(test.input); match != test.expected {
			t.Errorf("Failed %s: expected %v, got %v for input '%s'", test.level, test.expected, match, test.input)
		}
	}
}
