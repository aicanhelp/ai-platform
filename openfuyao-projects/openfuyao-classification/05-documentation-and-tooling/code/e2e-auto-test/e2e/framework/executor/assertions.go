package executor

import (
	"fmt"
	"regexp"
	"strings"
)

// AssertCommandSuccess 断言命令执行成功（退出码为0且无错误）
func AssertCommandSuccess(result *ExecResult) error {
	if result.ExitCode != 0 {
		return fmt.Errorf("command failed with exit code %d, stderr: %s", result.ExitCode, result.Stderr)
	}
	return nil
}

// AssertCommandFailed 断言命令执行失败
func AssertCommandFailed(result *ExecResult) error {
	if result.ExitCode == 0 {
		return fmt.Errorf("expected command to fail, but it succeeded")
	}
	return nil
}

// AssertExitCode 断言退出码
func AssertExitCode(result *ExecResult, expected int) error {
	if result.ExitCode != expected {
		return fmt.Errorf("expected exit code %d, got %d", expected, result.ExitCode)
	}
	return nil
}

// AssertOutputContains 断言标准输出包含指定内容
func AssertOutputContains(result *ExecResult, expected string) error {
	if !strings.Contains(result.Stdout, expected) {
		return fmt.Errorf("expected stdout to contain %q, but got: %s", expected, result.Stdout)
	}
	return nil
}

// AssertOutputNotContains 断言标准输出不包含指定内容
func AssertOutputNotContains(result *ExecResult, unexpected string) error {
	if strings.Contains(result.Stdout, unexpected) {
		return fmt.Errorf("expected stdout not to contain %q, but it does: %s", unexpected, result.Stdout)
	}
	return nil
}

// AssertOutputEquals 断言标准输出完全等于指定内容
func AssertOutputEquals(result *ExecResult, expected string) error {
	if result.Stdout != expected {
		return fmt.Errorf("expected stdout to be %q, but got: %s", expected, result.Stdout)
	}
	return nil
}

// AssertStderrContains 断言标准错误输出包含指定内容
func AssertStderrContains(result *ExecResult, expected string) error {
	if !strings.Contains(result.Stderr, expected) {
		return fmt.Errorf("expected stderr to contain %q, but got: %s", expected, result.Stderr)
	}
	return nil
}

// AssertStderrNotContains 断言标准错误输出不包含指定内容
func AssertStderrNotContains(result *ExecResult, unexpected string) error {
	if strings.Contains(result.Stderr, unexpected) {
		return fmt.Errorf("expected stderr not to contain %q, but it does: %s", unexpected, result.Stderr)
	}
	return nil
}

// AssertStderrEquals 断言标准错误输出完全等于指定内容
func AssertStderrEquals(result *ExecResult, expected string) error {
	if result.Stderr != expected {
		return fmt.Errorf("expected stderr to be %q, but got: %s", expected, result.Stderr)
	}
	return nil
}

// AssertOutputMatches 断言标准输出匹配指定模式（使用正则表达式）
func AssertOutputMatches(result *ExecResult, pattern string) error {
	matched, err := regexp.MatchString(pattern, result.Stdout)
	if err != nil {
		return fmt.Errorf("invalid regexp pattern %q: %w", pattern, err)
	}
	if !matched {
		return fmt.Errorf("expected stdout to match pattern %q, but got: %s", pattern, result.Stdout)
	}
	return nil
}

// AssertOutputEmpty 断言标准输出为空
func AssertOutputEmpty(result *ExecResult) error {
	if strings.TrimSpace(result.Stdout) != "" {
		return fmt.Errorf("expected stdout to be empty, but got: %s", result.Stdout)
	}
	return nil
}

// AssertStderrEmpty 断言标准错误输出为空
func AssertStderrEmpty(result *ExecResult) error {
	if strings.TrimSpace(result.Stderr) != "" {
		return fmt.Errorf("expected stderr to be empty, but got: %s", result.Stderr)
	}
	return nil
}

// AssertExecResult 断言命令结果满足所有条件
type ExecResultAssertion func(*ExecResult) error

// AssertExecResults 断言命令结果满足多个条件
func AssertExecResults(result *ExecResult, assertions ...ExecResultAssertion) error {
	for _, assertion := range assertions {
		if err := assertion(result); err != nil {
			return err
		}
	}
	return nil
}

// 常用的断言组合
var (
	// AssertSuccess 断言命令成功执行
	AssertSuccess = AssertExecResults
	// AssertFailure 断言命令执行失败
	AssertFailure = AssertExecResults
)

// AssertSuccessWithOutput 断言命令成功且有输出
func AssertSuccessWithOutput(expectedOutput string) ExecResultAssertion {
	return func(result *ExecResult) error {
		if err := AssertCommandSuccess(result); err != nil {
			return err
		}
		return AssertOutputContains(result, expectedOutput)
	}
}

// AssertFailureWithExitCode 断言命令失败且退出码为指定值
func AssertFailureWithExitCode(expectedCode int) ExecResultAssertion {
	return func(result *ExecResult) error {
		if err := AssertCommandFailed(result); err != nil {
			return err
		}
		return AssertExitCode(result, expectedCode)
	}
}
