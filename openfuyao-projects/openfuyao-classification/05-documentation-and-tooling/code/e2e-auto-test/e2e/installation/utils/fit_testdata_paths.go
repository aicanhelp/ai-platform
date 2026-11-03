package utils

import (
	"path/filepath"
)

const (
	// FITTestdataRootDir 相对 installation 模块根目录：testdata/fit。
	FITTestdataRootDir = "testdata/fit"
	// FITTestdataBaselineDir vfit-pre 基线 fit-yaml 手工补导入 manifest（8 组件，升级前通用）。
	FITTestdataBaselineDir = "vfit-pre-baseline"
	// FITTestdataFIT9Dir YAML_ENGINE_FIT_9 专用预置资源（仅核心 Namespace 样本）。
	FITTestdataFIT9Dir = "yaml-engine-fit-9"
	// FITTestdataFIT9CoreNamespaceFile FIT_9 核心 Namespace fixture 文件名。
	FITTestdataFIT9CoreNamespaceFile = "fit-prune-core-namespace.yaml"
)

// FITTestdataBaselineRoot 返回 vfit-pre 基线 manifest 根目录（绝对路径）。
func FITTestdataBaselineRoot(installationModuleDir string) string {
	return filepath.Join(installationModuleDir, FITTestdataRootDir, FITTestdataBaselineDir)
}

// FITTestdataFIT9CoreNamespacePath 返回 FIT_9 核心 Namespace fixture 路径（绝对路径）。
func FITTestdataFIT9CoreNamespacePath(installationModuleDir string) string {
	return filepath.Join(installationModuleDir, FITTestdataRootDir, FITTestdataFIT9Dir, FITTestdataFIT9CoreNamespaceFile)
}
