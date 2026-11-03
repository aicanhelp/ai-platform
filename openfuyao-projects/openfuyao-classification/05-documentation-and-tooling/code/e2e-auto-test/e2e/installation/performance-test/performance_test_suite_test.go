package performance_test

import (
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/joho/godotenv"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func init() {
	// 尝试加载 .env.performance 文件
	envFile := ".env.performance"
	if _, err := os.Stat(envFile); os.IsNotExist(err) {
		if execPath, err := os.Executable(); err == nil {
			envFile = filepath.Join(filepath.Dir(execPath), ".env.performance")
		}
	}

	if err := godotenv.Load(envFile); err != nil {
		if err := godotenv.Load(".env"); err != nil {
			log.Printf("警告: 未找到 .env.performance 或 .env 文件，将使用系统环境变量")
		}
	} else {
		log.Printf("已加载环境配置文件: %s", envFile)
	}

	// 注意：Suite timeout 需要通过命令行参数设置，例如：
	// ginkgo -v --timeout=6h ./...
	// Ginkgo v2 中没有 SetDefaultSuiteTimeout 函数
}

func TestPerformance(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "BKE Installation Performance Suite")
}