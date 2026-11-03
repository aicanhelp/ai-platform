package executor

import (
	"net"
)

// IsValidHost 验证主机名或IP地址是否有效
func IsValidHost(host string) bool {
	// 简单的验证，实际使用时可以更严格
	if host == "" {
		return false
	}

	// 检查是否是有效的IP地址
	if net.ParseIP(host) != nil {
		return true
	}

	// 检查是否是有效的主机名
	if len(host) > 255 {
		return false
	}

	// 基本的主机名验证
	for _, char := range host {
		if !((char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') ||
			(char >= '0' && char <= '9') || char == '-' || char == '.' || char == '_') {
			return false
		}
	}

	return true
}

// IsValidPort 验证端口是否有效
func IsValidPort(port int) bool {
	return port > 0 && port <= 65535
}
