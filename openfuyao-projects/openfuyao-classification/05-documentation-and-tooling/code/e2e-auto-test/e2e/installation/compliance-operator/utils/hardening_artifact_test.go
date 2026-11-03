package utils

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrerequisiteInstallCommandSkipsInstallWhenDependenciesAreReady(t *testing.T) {
	binDir := t.TempDir()
	writeFixtureFile(t, filepath.Join(binDir, "sshpass"), []byte("#!/bin/sh\nexit 0\n"), 0o755)
	writeFixtureFile(t, filepath.Join(binDir, "jq"), []byte("#!/bin/sh\nexit 0\n"), 0o755)
	writeFixtureFile(t, filepath.Join(binDir, "yq"), []byte(`#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "yq (https://github.com/mikefarah/yq/) version v4.53.6"
else
  cat >/dev/null
  echo ready
fi
`), 0o755)
	writeFixtureFile(t, filepath.Join(binDir, "dnf"), []byte("#!/bin/sh\nprintf called >\"$INSTALL_MARKER\"\nexit 99\n"), 0o755)

	marker := filepath.Join(t.TempDir(), "package-manager-called")
	command := exec.Command("bash", "-c", prerequisiteInstallCommand([]string{"sshpass", "yq", "jq"}))
	command.Env = append(os.Environ(), "PATH="+binDir+":"+os.Getenv("PATH"), "INSTALL_MARKER="+marker)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("prerequisite audit failed: %v\n%s", err, output)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("package manager was called although all dependencies were ready")
	}
	if got := string(output); !strings.Contains(got, "DEPENDENCY_AUDIT 已配置=sshpass,yq,jq 本次安装=none") {
		t.Fatalf("audit output = %q", got)
	}
}

func TestBuildHardeningArchiveIncludesOnlyRuntimeFiles(t *testing.T) {
	sourceDir := createHardeningFixture(t)
	writeFixtureFile(t, filepath.Join(sourceDir, "config", "cluster.yaml"), []byte("password: secret\n"), 0o600)
	writeFixtureFile(t, filepath.Join(sourceDir, "README.md"), []byte("docs\n"), 0o644)

	data, err := buildHardeningArchive(sourceDir)
	if err != nil {
		t.Fatalf("buildHardeningArchive() error = %v", err)
	}
	entries := readArchiveEntries(t, data)

	for _, required := range []string{
		"bin/k8s-hardening.sh",
		"lib/common.sh",
		"config/hardening-rules.yaml",
	} {
		if _, ok := entries[required]; !ok {
			t.Errorf("archive is missing %s", required)
		}
	}
	for _, excluded := range []string{"config/cluster.yaml", "README.md"} {
		if _, ok := entries[excluded]; ok {
			t.Errorf("archive unexpectedly contains %s", excluded)
		}
	}
	if mode := entries["bin/k8s-hardening.sh"]; mode&0o111 == 0 {
		t.Errorf("entry script mode %o is not executable", mode)
	}
}

func TestBuildHardeningArchiveIsDeterministic(t *testing.T) {
	sourceDir := createHardeningFixture(t)

	first, err := buildHardeningArchive(sourceDir)
	if err != nil {
		t.Fatalf("first buildHardeningArchive() error = %v", err)
	}
	second, err := buildHardeningArchive(sourceDir)
	if err != nil {
		t.Fatalf("second buildHardeningArchive() error = %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("unchanged runtime files produced different archives; remote bundle digest would never be reusable")
	}
}

func TestBuildHardeningArchiveRejectsSymlink(t *testing.T) {
	sourceDir := createHardeningFixture(t)
	if err := os.Symlink("common.sh", filepath.Join(sourceDir, "lib", "linked.sh")); err != nil {
		t.Fatalf("create symlink: %v", err)
	}

	_, err := buildHardeningArchive(sourceDir)
	if err == nil || !strings.Contains(err.Error(), "不允许包含符号链接") {
		t.Fatalf("buildHardeningArchive() error = %v, want symlink rejection", err)
	}
}

func TestNormalizeHardeningSourceDirAcceptsRepositoryRoot(t *testing.T) {
	repositoryRoot := t.TempDir()
	sourceDir := filepath.Join(repositoryRoot, "compliance-hardening")
	createHardeningFixtureAt(t, sourceDir)

	if got := normalizeHardeningSourceDir(repositoryRoot); got != sourceDir {
		t.Fatalf("normalizeHardeningSourceDir() = %q, want %q", got, sourceDir)
	}
}

func TestResolveHardeningSourceDirFindsBundledFixture(t *testing.T) {
	t.Setenv(EnvHardeningSourceDir, "")
	repositoryRoot := t.TempDir()
	moduleRoot := filepath.Join(repositoryRoot, "e2e")
	sourceDir := filepath.Join(moduleRoot, "installation", "compliance-operator", "testdata", "compliance-hardening")
	createHardeningFixtureAt(t, sourceDir)

	previousWorkingDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("get working directory: %v", err)
	}
	if err := os.Chdir(moduleRoot); err != nil {
		t.Fatalf("change working directory: %v", err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(previousWorkingDir); err != nil {
			t.Errorf("restore working directory: %v", err)
		}
	})

	if got := resolveHardeningSourceDir(); got != sourceDir {
		t.Fatalf("resolveHardeningSourceDir() = %q, want bundled fixture %q", got, sourceDir)
	}
}

func TestResolveHardeningSourceDirPrefersExplicitOverride(t *testing.T) {
	repositoryRoot := t.TempDir()
	bundledDir := filepath.Join(repositoryRoot, "e2e", "installation", "compliance-operator", "testdata", "compliance-hardening")
	overrideDir := filepath.Join(repositoryRoot, "override", "compliance-hardening")
	createHardeningFixtureAt(t, bundledDir)
	createHardeningFixtureAt(t, overrideDir)
	t.Setenv(EnvHardeningSourceDir, overrideDir)

	previousWorkingDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("get working directory: %v", err)
	}
	if err := os.Chdir(filepath.Join(repositoryRoot, "e2e")); err != nil {
		t.Fatalf("change working directory: %v", err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(previousWorkingDir); err != nil {
			t.Errorf("restore working directory: %v", err)
		}
	})

	if got := resolveHardeningSourceDir(); got != overrideDir {
		t.Fatalf("resolveHardeningSourceDir() = %q, want explicit override %q", got, overrideDir)
	}
}

func createHardeningFixture(t *testing.T) string {
	t.Helper()
	sourceDir := t.TempDir()
	createHardeningFixtureAt(t, sourceDir)
	return sourceDir
}

func createHardeningFixtureAt(t *testing.T, sourceDir string) {
	t.Helper()
	writeFixtureFile(t, filepath.Join(sourceDir, "bin", "k8s-hardening.sh"), []byte("#!/bin/bash\n"), 0o755)
	writeFixtureFile(t, filepath.Join(sourceDir, "lib", "common.sh"), []byte("true\n"), 0o644)
	writeFixtureFile(t, filepath.Join(sourceDir, "config", "hardening-rules.yaml"), []byte("rules: []\n"), 0o644)
}

func writeFixtureFile(t *testing.T, target string, content []byte, mode os.FileMode) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatalf("create fixture directory: %v", err)
	}
	if err := os.WriteFile(target, content, mode); err != nil {
		t.Fatalf("write fixture file: %v", err)
	}
}

func readArchiveEntries(t *testing.T, data []byte) map[string]os.FileMode {
	t.Helper()
	gzipReader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("open gzip archive: %v", err)
	}
	defer gzipReader.Close()

	entries := make(map[string]os.FileMode)
	tarReader := tar.NewReader(gzipReader)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read tar archive: %v", err)
		}
		entries[header.Name] = os.FileMode(header.Mode)
	}
	return entries
}
