package utils

import (
	"fmt"
	"strconv"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

// ExecuteCommandOnNode 在远程节点上执行命令
// 使用 SSHExecutor 库实现
func ExecuteCommandOnNode(node config.NodeInfo, command string) (*executor.ExecResult, error) {
	// 将 Port 字符串转换为 int
	port, err := strconv.Atoi(node.Port)
	if err != nil {
		// 如果转换失败，使用默认端口 22
		port = 22
	}

	// 创建 SSH 执行器连接到目标节点
	nodeExecutor, err := executor.NewSSHExecutor(node.IP, port, node.Username, node.Password)
	if err != nil {
		return nil, fmt.Errorf("创建 SSH 连接到节点 %s 失败: %w", node.IP, err)
	}

	// 执行命令
	result, execErr := nodeExecutor.Exec(command)

	// 注意：SSHExecutor 没有公开的 Close 方法，连接会在函数返回后被垃圾回收
	// 如果需要更好的资源管理，可以考虑在 SSHExecutor 中添加 Close 方法

	if execErr != nil {
		return result, fmt.Errorf("在节点 %s 上执行命令失败: %w", node.IP, execErr)
	}

	return result, nil
}
