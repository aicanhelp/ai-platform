// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"path"
	"strings"
)

// A bounded lexer for literal simple commands. No shell is executed. Expansions,
// redirections, pipelines, conditionals and globbing are deliberately rejected.
type healthProbeShellLexer struct {
	script     string
	index      int
	quote      byte
	started    bool
	word       strings.Builder
	words      []string
	statements [][]string
}

func splitHealthProbeShell(script string) ([][]string, bool) {
	l := &healthProbeShellLexer{script: script}
	for l.index < len(script) {
		if !l.consume(script[l.index]) {
			return nil, false
		}
		l.index++
	}
	if l.quote != 0 {
		return nil, false
	}
	l.flushStatement()
	return l.statements, true
}

func (l *healthProbeShellLexer) flushWord() {
	if !l.started {
		return
	}
	l.words = append(l.words, l.word.String())
	l.word.Reset()
	l.started = false
}

func (l *healthProbeShellLexer) flushStatement() {
	l.flushWord()
	if len(l.words) == 0 {
		return
	}
	l.statements = append(l.statements, l.words)
	l.words = nil
}

func (l *healthProbeShellLexer) consume(c byte) bool {
	if c == 0 || c == '\r' {
		return false
	}
	if l.quote == '\'' {
		l.consumeQuoted(c)
		return true
	}
	if c == '\\' {
		return l.consumeEscape()
	}
	if l.quote == '"' {
		if c == '$' || c == '`' {
			return false
		}
		l.consumeQuoted(c)
		return true
	}
	return l.consumeUnquoted(c)
}

func (l *healthProbeShellLexer) consumeQuoted(c byte) {
	if c == l.quote {
		l.quote = 0
	} else {
		l.word.WriteByte(c)
	}
}

func (l *healthProbeShellLexer) consumeEscape() bool {
	if l.index+1 == len(l.script) {
		return false
	}
	next := l.script[l.index+1]
	if next == '\n' {
		l.index++
		return true
	}
	if l.quote == '"' && !strings.ContainsRune("\\\"$`", rune(next)) {
		l.word.WriteByte('\\')
		return true
	}
	l.index++
	l.word.WriteByte(next)
	l.started = true
	return true
}

func (l *healthProbeShellLexer) consumeUnquoted(c byte) bool {
	switch c {
	case '\'', '"':
		l.quote, l.started = c, true
	case ' ', '\t':
		l.flushWord()
	case ';', '\n':
		l.flushWord()
		if c == ';' && len(l.words) == 0 {
			return false
		}
		l.flushStatement()
	case '#':
		l.consumeComment()
	case '$', '`', '|', '&', '<', '>', '(', ')', '{', '}', '*', '?', '[', ']', '~', '!':
		return false
	default:
		l.word.WriteByte(c)
		l.started = true
	}
	return true
}

func (l *healthProbeShellLexer) consumeComment() {
	if l.started {
		l.word.WriteByte('#')
		return
	}
	for l.index+1 < len(l.script) && l.script[l.index+1] != '\n' {
		l.index++
	}
}

func healthProbeCommandArgs(binary string, command, args []string) ([]string, bool) {
	if len(command) == 0 {
		// The image's ENTRYPOINT cannot be inferred by the controller.
		return nil, false
	}
	argv := append(append([]string(nil), command...), args...)
	for _, arg := range argv {
		if strings.Contains(arg, "$(") {
			// Kubelet expands $(VAR) even before invoking a shell.
			return nil, false
		}
	}
	if path.Base(argv[0]) == binary {
		return argv[1:], true
	}
	script, ok := healthProbeShellScript(argv)
	if !ok {
		return nil, false
	}
	return healthProbeShellCommandArgs(binary, script)
}

func healthProbeShellScript(argv []string) (string, bool) {
	if (path.Base(argv[0]) != "bash" && path.Base(argv[0]) != "sh") || len(argv) < 3 || argv[1] != "-c" {
		return "", false
	}
	scriptArgs := argv[2:]
	if scriptArgs[0] == "--" {
		scriptArgs = scriptArgs[1:]
	}
	if len(scriptArgs) != 1 {
		return "", false
	}
	return scriptArgs[0], true
}

func healthProbeShellCommandArgs(binary, script string) ([]string, bool) {
	statements, ok := splitHealthProbeShell(script)
	if !ok || len(statements) == 0 {
		return nil, false
	}
	// Only the existing initialization prefix is supported. Arbitrary commands
	// could exit, replace the shell, define aliases/functions or change parsing.
	for _, statement := range statements[:len(statements)-1] {
		switch path.Base(statement[0]) {
		case "useradd", "groupadd", "usermod", "chown", "chmod", "mkdir", "umask":
		default:
			return nil, false
		}
	}
	last := statements[len(statements)-1]
	if last[0] == "exec" {
		last = last[1:]
	}
	if len(last) == 0 || path.Base(last[0]) != binary {
		return nil, false
	}
	return last[1:], true
}
