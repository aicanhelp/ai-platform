package utils

import (
	"os"
	"path/filepath"
	"strings"
)

// parseEnvAssignment 解析一行 "KEY=VALUE" 的赋值语句。
// 返回 (key, value, ok)。会忽略空行/注释行/非法格式。
func parseEnvAssignment(raw string) (string, string, bool) {
	s := strings.TrimSpace(raw)
	if s == "" || strings.HasPrefix(s, "#") {
		return "", "", false
	}
	i := strings.IndexByte(s, '=')
	if i <= 0 {
		return "", "", false
	}
	k := strings.TrimSpace(s[:i])
	if k == "" {
		return "", "", false
	}
	v := strings.TrimSpace(s[i+1:])
	// 兼容常见写法：KEY="value" / KEY='value'
	v = strings.Trim(v, `"'`)
	return k, v, true
}

// LoadEnvFromFile 从指定目录加载环境变量文件
// configDir: 配置目录路径（相对于包目录）
// envFileName: 环境变量文件名（如 ".env.hermes_router"）
// 会尝试加载 envFileName 和 envFileName + ".template" 两个文件
// 如果 envFileName 存在则优先使用，否则使用 template 文件
func LoadEnvFromFile(configDir, envFileName string) error {
	// 获取包目录（utils 目录的父目录）
	dir, err := getPackageDir()
	if err != nil {
		return err
	}

	// 尝试加载两个文件：实际文件和模板文件
	for _, fileName := range []string{
		envFileName,
		envFileName + ".template",
	} {
		envPath := filepath.Join(dir, configDir, fileName)
		data, err := os.ReadFile(envPath)
		if err != nil {
			continue
		}

		// 解析并设置环境变量
		for _, rawLine := range strings.Split(string(data), "\n") {
			key, val, ok := parseEnvAssignment(rawLine)
			if !ok {
				continue
			}
			// 不覆盖已存在的环境变量，便于在命令行 export 后运行
			if _, exists := os.LookupEnv(key); exists {
				continue
			}
			_ = os.Setenv(key, val)
		}

		// 成功加载一个文件后返回
		return nil
	}

	// 两个文件都不存在时，不报错（允许通过其他方式设置环境变量）
	return nil
}
