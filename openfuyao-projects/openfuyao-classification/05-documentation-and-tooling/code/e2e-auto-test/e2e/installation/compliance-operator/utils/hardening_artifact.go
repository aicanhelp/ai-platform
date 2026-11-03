package utils

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

var hardeningBundlePaths = []string{
	"bin",
	"lib",
	"config/hardening-rules.yaml",
}

// EnsureHardeningPrerequisites audits the external commands required by the
// compliance-hardening workflow before any spec is allowed to run, installs
// only missing or incompatible dependencies, and verifies them again. The entry
// node orchestrates password-based SSH, while every cluster node executes the
// internal harden/rollback commands and therefore needs yq v4 and jq locally.
func EnsureHardeningPrerequisites(entry *executor.SSHExecutor, nodeExecs map[string]*executor.SSHExecutor, cfg *Config) ([]string, error) {
	type target struct {
		name     string
		executor *executor.SSHExecutor
		commands []string
	}

	targets := []target{{
		name:     fmt.Sprintf("entry_node %s", cfg.Entry.IP),
		executor: entry,
		commands: []string{"sshpass", "yq", "jq"},
	}}
	for _, node := range cfg.Nodes {
		if node.IP == cfg.Entry.IP {
			continue
		}
		targets = append(targets, target{
			name:     fmt.Sprintf("节点 %s(%s)", node.IP, node.Role),
			executor: nodeExecs[node.IP],
			commands: []string{"yq", "jq"},
		})
	}

	var failures []string
	var audit []string
	for _, target := range targets {
		if target.executor == nil {
			failures = append(failures, target.name+"：无可用 SSH 执行器")
			continue
		}
		result, err := target.executor.Exec(prerequisiteInstallCommand(target.commands))
		if result == nil {
			failures = append(failures, fmt.Sprintf("%s：依赖审核/安装命令无法执行: %v", target.name, err))
			continue
		}
		if result.ExitCode != 0 {
			detail := strings.TrimSpace(result.Stderr)
			if detail == "" {
				detail = strings.TrimSpace(result.Stdout)
			}
			failures = append(failures, fmt.Sprintf("%s：依赖审核、安装或验证失败（exit=%d）: %s", target.name, result.ExitCode, detail))
			continue
		}
		for _, line := range strings.Split(result.Stdout, "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "DEPENDENCY_AUDIT ") {
				audit = append(audit, target.name+"："+strings.TrimPrefix(line, "DEPENDENCY_AUDIT "))
			}
		}
	}
	if len(failures) > 0 {
		return audit, fmt.Errorf("compliance-hardening 运行依赖未就绪：\n  - %s", strings.Join(failures, "\n  - "))
	}
	return audit, nil
}

func prerequisiteInstallCommand(commands []string) string {
	script := fmt.Sprintf(`set -euo pipefail
required=(%s)
package_missing=()
preconfigured=()
installed=()

install_packages() {
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y "$@"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y "$@"
  elif command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y "$@"
  elif command -v zypper >/dev/null 2>&1; then
    zypper --non-interactive install "$@"
  else
    printf '未找到受支持的包管理器，无法安装：%%s\n' "$*" >&2
    exit 20
  fi
}

dependency_ready() {
  case "$1" in
    sshpass)
      command -v sshpass >/dev/null 2>&1 && sshpass -V >/dev/null 2>&1
      ;;
    yq)
      command -v yq >/dev/null 2>&1 &&
        yq --version 2>&1 | grep -Eq 'version v?4\.' &&
        [[ "$(printf 'value: ready\n' | yq eval -r '.value' - 2>/dev/null)" == "ready" ]]
      ;;
    jq)
      command -v jq >/dev/null 2>&1 && jq -n -e 'true' >/dev/null 2>&1
      ;;
    *) return 1 ;;
  esac
}

for command_name in "${required[@]}"; do
  if dependency_ready "$command_name"; then
    preconfigured+=("$command_name")
  elif [[ "$command_name" == "yq" ]]; then
    : # yq 使用下方官方 v4 二进制安装路径，避免发行版同名但不兼容的实现。
  else
    package_missing+=("$command_name")
  fi
done

if (( ${#package_missing[@]} > 0 )); then
  install_packages "${package_missing[@]}"
  installed+=("${package_missing[@]}")
fi

if [[ " ${required[*]} " == *" yq "* ]] && ! dependency_ready yq; then
  yq_version="v4.53.6"
  case "$(uname -m)" in
    x86_64|amd64) yq_platform="linux_amd64" ;;
    aarch64|arm64) yq_platform="linux_arm64" ;;
    *) printf '不支持为当前架构自动安装 yq：%%s\n' "$(uname -m)" >&2; exit 25 ;;
  esac
  yq_url="https://github.com/mikefarah/yq/releases/download/${yq_version}/yq_${yq_platform}"
  yq_tmp="$(mktemp /tmp/compliance-yq.XXXXXX)"
  trap 'rm -f "$yq_tmp"' EXIT
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error "$yq_url" --output "$yq_tmp"
  elif command -v wget >/dev/null 2>&1; then
    wget --quiet "$yq_url" --output-document="$yq_tmp"
  else
    install_packages curl
    curl --fail --location --silent --show-error "$yq_url" --output "$yq_tmp"
  fi
  install -m 0755 "$yq_tmp" /usr/local/bin/yq
  rm -f "$yq_tmp"
  trap - EXIT
  export PATH="/usr/local/bin:$PATH"
  installed+=("yq")
fi

for command_name in "${required[@]}"; do
  dependency_ready "$command_name" || {
    printf '依赖安装后仍不可用：%%s\n' "$command_name" >&2
    exit 21
  }
done

join_by_comma() {
  local IFS=,
  printf '%%s' "$*"
}
preconfigured_text="$(join_by_comma "${preconfigured[@]}")"
installed_text="$(join_by_comma "${installed[@]}")"
[[ -n "$preconfigured_text" ]] || preconfigured_text="none"
[[ -n "$installed_text" ]] || installed_text="none"
printf 'DEPENDENCY_AUDIT 已配置=%%s 本次安装=%%s\n' "$preconfigured_text" "$installed_text"`, strings.Join(commands, " "))
	return "bash -c " + shellQuote(script)
}

// EnsureHardeningTool synchronizes the runner's compliance-hardening fixture to
// the entry node. A remote installation is reused only when its bundle digest
// matches the current runner source. Runtime cluster.yaml files are deliberately
// excluded from the bundle and preserved because they contain environment
// credentials and are generated separately.
func EnsureHardeningTool(e *executor.SSHExecutor, cfg *Config) (bool, error) {
	if cfg.HardeningSourceDir == "" {
		return false, fmt.Errorf("Runner 上未找到 compliance-hardening 源码；请设置 %s 指向 compliance-hardening 目录或其仓库根目录", EnvHardeningSourceDir)
	}

	archive, err := buildHardeningArchive(cfg.HardeningSourceDir)
	if err != nil {
		return false, fmt.Errorf("打包 Runner 上的 compliance-hardening 工具: %w", err)
	}
	bundleDigest := fmt.Sprintf("%x", sha256.Sum256(archive))
	ready, err := remoteHardeningToolReady(e, cfg.HardeningDir, bundleDigest)
	if err != nil {
		return false, err
	}
	if ready {
		return false, nil
	}

	parentDir := path.Dir(cfg.HardeningDir)
	command := fmt.Sprintf(`set -eu
parent=%s
dest=%s
digest=%s
install -d -m 700 "$parent"
tmp="$(mktemp -d "$parent/.compliance-hardening.e2e.XXXXXX")"
old="$(mktemp -d "$parent/.compliance-hardening.old.XXXXXX")"
committed=0
cleanup() {
  if [ "$committed" -ne 1 ] && [ -d "$dest" ]; then
    rm -rf -- "$dest/bin" "$dest/lib"
    rm -f -- "$dest/config/hardening-rules.yaml" "$dest/.e2e-bundle.sha256" "$dest/.e2e-bundle.sha256.tmp"
    [ ! -e "$old/bin" ] || mv -- "$old/bin" "$dest/bin"
    [ ! -e "$old/lib" ] || mv -- "$old/lib" "$dest/lib"
    [ ! -e "$old/hardening-rules.yaml" ] || mv -- "$old/hardening-rules.yaml" "$dest/config/hardening-rules.yaml"
    [ ! -e "$old/.e2e-bundle.sha256" ] || mv -- "$old/.e2e-bundle.sha256" "$dest/.e2e-bundle.sha256"
  fi
  rm -rf -- "$tmp" "$old"
}
trap cleanup EXIT
tar -xzf - -C "$tmp"
test -x "$tmp/bin/k8s-hardening.sh"
test -d "$tmp/lib"
test -f "$tmp/config/hardening-rules.yaml"
if [ -e "$dest" ] && [ ! -d "$dest" ]; then
  echo "compliance-hardening 目标路径不是目录: $dest" >&2
  exit 1
fi
install -d -m 700 "$dest" "$dest/config"
[ ! -e "$dest/bin" ] || mv -- "$dest/bin" "$old/bin"
[ ! -e "$dest/lib" ] || mv -- "$dest/lib" "$old/lib"
[ ! -e "$dest/config/hardening-rules.yaml" ] || mv -- "$dest/config/hardening-rules.yaml" "$old/hardening-rules.yaml"
[ ! -e "$dest/.e2e-bundle.sha256" ] || mv -- "$dest/.e2e-bundle.sha256" "$old/.e2e-bundle.sha256"
mv -- "$tmp/bin" "$dest/bin"
mv -- "$tmp/lib" "$dest/lib"
mv -- "$tmp/config/hardening-rules.yaml" "$dest/config/hardening-rules.yaml"
printf '%%s\n' "$digest" > "$dest/.e2e-bundle.sha256.tmp"
chmod 600 "$dest/.e2e-bundle.sha256.tmp"
mv -- "$dest/.e2e-bundle.sha256.tmp" "$dest/.e2e-bundle.sha256"
committed=1
cleanup
trap - EXIT`, shellPath(parentDir), shellPath(cfg.HardeningDir), shellPath(bundleDigest))
	result, execErr := e.ExecWithInput(command, string(archive))
	if execErr != nil || result == nil || result.ExitCode != 0 {
		stderr := ""
		if result != nil {
			stderr = strings.TrimSpace(result.Stderr)
		}
		return false, fmt.Errorf("部署 compliance-hardening 工具到 entry_node 失败: err=%v stderr=%s", execErr, stderr)
	}

	ready, err = remoteHardeningToolReady(e, cfg.HardeningDir, bundleDigest)
	if err != nil {
		return false, err
	}
	if !ready {
		return false, fmt.Errorf("部署完成后 compliance-hardening 工具完整性检查失败: %s", cfg.HardeningDir)
	}
	return true, nil
}

func remoteHardeningToolReady(e *executor.SSHExecutor, hardeningDir, bundleDigest string) (bool, error) {
	result, err := e.Exec(fmt.Sprintf("test -d %s && test -x %s/bin/k8s-hardening.sh && test -d %s/lib && test -f %s/config/hardening-rules.yaml && test \"$(cat %s/.e2e-bundle.sha256 2>/dev/null)\" = %s",
		shellPath(hardeningDir), shellPath(hardeningDir), shellPath(hardeningDir), shellPath(hardeningDir), shellPath(hardeningDir), shellPath(bundleDigest)))
	if result == nil {
		return false, fmt.Errorf("检查 entry_node compliance-hardening 工具失败: %w", err)
	}
	if result.ExitCode == 0 {
		return true, nil
	}
	if result.ExitCode == 1 {
		return false, nil
	}
	return false, fmt.Errorf("检查 entry_node compliance-hardening 工具失败（exit=%d）: %s", result.ExitCode, strings.TrimSpace(result.Stderr))
}

func resolveHardeningSourceDir() string {
	if configured := strings.TrimSpace(os.Getenv(EnvHardeningSourceDir)); configured != "" {
		return normalizeHardeningSourceDir(configured)
	}

	workingDir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for current := filepath.Clean(workingDir); ; current = filepath.Dir(current) {
		for _, candidate := range []string{
			filepath.Join(current, "testdata", "compliance-hardening"),
			filepath.Join(current, "installation", "compliance-operator", "testdata", "compliance-hardening"),
			filepath.Join(current, "e2e", "installation", "compliance-operator", "testdata", "compliance-hardening"),
			filepath.Join(current, "compliance-operator", "compliance-hardening"),
			filepath.Join(filepath.Dir(current), "compliance-operator", "compliance-hardening"),
		} {
			if validHardeningSourceDir(candidate) {
				return candidate
			}
		}
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
	}
	return ""
}

func normalizeHardeningSourceDir(configured string) string {
	candidate := filepath.Clean(configured)
	if validHardeningSourceDir(candidate) {
		return candidate
	}
	nested := filepath.Join(candidate, "compliance-hardening")
	if validHardeningSourceDir(nested) {
		return nested
	}
	return candidate
}

func validHardeningSourceDir(sourceDir string) bool {
	for _, required := range []string{
		"bin/k8s-hardening.sh",
		"config/hardening-rules.yaml",
	} {
		info, err := os.Stat(filepath.Join(sourceDir, filepath.FromSlash(required)))
		if err != nil || !info.Mode().IsRegular() {
			return false
		}
	}
	info, err := os.Stat(filepath.Join(sourceDir, "lib"))
	return err == nil && info.IsDir()
}

func buildHardeningArchive(sourceDir string) ([]byte, error) {
	sourceDir = filepath.Clean(sourceDir)
	if !validHardeningSourceDir(sourceDir) {
		return nil, fmt.Errorf("工具目录不完整: %s", sourceDir)
	}

	var output bytes.Buffer
	gzipWriter := gzip.NewWriter(&output)
	tarWriter := tar.NewWriter(gzipWriter)
	for _, bundlePath := range hardeningBundlePaths {
		absolutePath := filepath.Join(sourceDir, filepath.FromSlash(bundlePath))
		if err := filepath.WalkDir(absolutePath, func(currentPath string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if info.Mode()&os.ModeSymlink != 0 {
				return fmt.Errorf("工具制品不允许包含符号链接: %s", currentPath)
			}
			if !info.IsDir() && !info.Mode().IsRegular() {
				return fmt.Errorf("工具制品包含不支持的文件类型: %s", currentPath)
			}

			relativePath, err := filepath.Rel(sourceDir, currentPath)
			if err != nil {
				return err
			}
			header, err := tar.FileInfoHeader(info, "")
			if err != nil {
				return err
			}
			header.Name = filepath.ToSlash(relativePath)
			header.Uid = 0
			header.Gid = 0
			header.Uname = ""
			header.Gname = ""
			header.ModTime = time.Unix(0, 0)
			header.AccessTime = time.Time{}
			header.ChangeTime = time.Time{}
			if err := tarWriter.WriteHeader(header); err != nil {
				return err
			}
			if info.IsDir() {
				return nil
			}

			file, err := os.Open(currentPath)
			if err != nil {
				return err
			}
			_, copyErr := io.Copy(tarWriter, file)
			closeErr := file.Close()
			if copyErr != nil {
				return copyErr
			}
			return closeErr
		}); err != nil {
			_ = tarWriter.Close()
			_ = gzipWriter.Close()
			return nil, err
		}
	}
	if err := tarWriter.Close(); err != nil {
		return nil, err
	}
	if err := gzipWriter.Close(); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}
