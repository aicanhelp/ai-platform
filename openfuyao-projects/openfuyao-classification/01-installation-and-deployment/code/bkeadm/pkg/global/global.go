/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package global

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	Docker      docker.DockerClient
	Containerd  containerd.ContainerdClient
	K8s         k8s.KubernetesClient
	Command     exec.Executor
	Workspace   string
	CustomExtra map[string]string
)

// workspaceFilePath is the conventional override file read at load time.
const workspaceFilePath = "/opt/BKE_WORKSPACE"

// workspaceEnvKey is the environment variable that overrides the workspace.
const workspaceEnvKey = "BKE_WORKSPACE"

func init() {
	Command = &exec.CommandExecutor{}
	// Package init only RESOLVES the workspace path; it MUST NOT create any
	// directories. Directory creation is an explicit, opt-in action taken by
	// the original install entry points via EnsureWorkspaceLayout. This keeps
	// read-only surfaces (preflight, help, version) side-effect-free on load.
	Workspace = ResolveWorkspace(os.Getenv(workspaceEnvKey), readWorkspaceFile(workspaceFilePath))
	CustomExtra = make(map[string]string)
}

// readWorkspaceFile returns the trimmed contents of the workspace override
// file, or "" when it is absent or unreadable. It never returns an error: the
// workspace falls back to the env value or the default when the file is
// unavailable, matching the historical precedence.
func readWorkspaceFile(path string) string {
	if !utils.IsFile(path) {
		return ""
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	s := strings.TrimSpace(string(b))
	s = strings.TrimRight(s, "\n")
	s = strings.TrimRight(s, "\r")
	s = strings.TrimRight(s, "\t")
	return s
}

// ResolveWorkspace computes the final workspace directory from the env value
// and the workspace-file value, with the precedence:
//
//	BKE_WORKSPACE env  >  /opt/BKE_WORKSPACE file  >  default "/bke"
//
// The result always appends "/bke" to a non-empty override. ResolveWorkspace
// is pure: it performs no I/O and creates nothing. Pass the current process
// env/file values explicitly; at package init they come from os.Getenv and the
// override file.
//
// The empty-string default returns "/bke" (no join) to preserve the historical
// default exactly.
func ResolveWorkspace(envValue, fileValue string) string {
	switch {
	case envValue != "":
		return filepath.Join(envValue, "bke")
	case fileValue != "":
		return filepath.Join(fileValue, "bke")
	default:
		return "/bke"
	}
}

// EnsureWorkspaceLayout explicitly creates the tmpl/volumes/mount directories
// under the given workspace. It is the ONLY place that creates these
// directories, and it is called solely by the original init/cluster-create
// execution paths. Preflight, help and version MUST NOT call it.
func EnsureWorkspaceLayout(workspace string) error {
	var failures []error
	for _, sub := range []string{"tmpl", "volumes", "mount"} {
		dir := filepath.Join(workspace, sub)
		if utils.Exists(dir) {
			continue
		}
		if err := os.MkdirAll(dir, utils.DefaultDirPermission); err != nil {
			failures = append(failures, fmt.Errorf("create %s: %w", dir, err))
		}
	}
	return errors.Join(failures...)
}

func TarGZ(prefix, target string) error {
	output, err := Command.ExecuteCommandWithOutput("sh", "-c", fmt.Sprintf("cd %s && tar --use-compress-program=pigz -cf %s .", prefix, target))
	if err != nil {
		return fmt.Errorf("tar failed, output: %s, error: %v", output, err)
	}
	return nil
}

func TarGZWithDir(prefix, dir, target string) error {
	output, err := Command.ExecuteCommandWithOutput("sh", "-c", fmt.Sprintf("cd %s && tar "+
		"--use-compress-program=pigz -cf %s ./%s", prefix, target, dir))
	if err != nil {
		return fmt.Errorf("tar failed, output: %s, error: %v", output, err)
	}
	return nil
}

// TaeGZWithoutChangeFile tar gz without change file
func TaeGZWithoutChangeFile(prefix, target string) error {
	cmd := fmt.Sprintf("cd %s && tar --use-compress-program=pigz -cf %s . --warning=no-file-changed --ignore-failed-read",
		prefix, target)
	output, err := Command.ExecuteCommandWithOutput("sh", "-c", cmd)
	if err != nil {
		return fmt.Errorf("tar failed, output: %s, error: %v", output, err)
	}
	return nil
}

func UnTarGZ(dataFile, target string) error {
	output, err := Command.ExecuteCommandWithOutput("sh", "-c", fmt.Sprintf("tar -xzf %s -C %s", dataFile, target))
	if err != nil {
		return fmt.Errorf("untar failed, output: %s, error: %v", output, err)
	}
	return nil
}

// ListK8sResources lists k8s resources by GVR and converts them to the target type
func ListK8sResources(gvr schema.GroupVersionResource, target interface{}) error {
	var workloadUnstructured *unstructured.UnstructuredList
	var err error
	dynamicClient := K8s.GetDynamicClient()
	workloadUnstructured, err = dynamicClient.Resource(gvr).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		log.Errorf("List k8s resources %s failed: %v", gvr.Resource, err)
		return err
	}
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(workloadUnstructured.UnstructuredContent(), target)
	if err != nil {
		log.Errorf("Convert k8s resources %s failed: %v", gvr.Resource, err)
		return err
	}
	return nil
}

// ExecuteManifestCommands creates a docker manifest, annotates each architecture, and pushes the final manifest.
func ExecuteManifestCommands(target, manifestCreateCmd string, manifestAnnotate []string) error {
	manifestEnv := append(os.Environ(), "DOCKER_CLI_EXPERIMENTAL=enabled")

	if err := Command.ExecuteCommandWithEnv(manifestEnv, "/bin/sh", "-c", manifestCreateCmd); err != nil {
		log.Errorf("%s, %v", manifestCreateCmd, err)
		return err
	}
	log.Infof("Execute manifest create command: %s", manifestCreateCmd)

	for _, cmd := range manifestAnnotate {
		if err := Command.ExecuteCommandWithEnv(manifestEnv, "/bin/sh", "-c", cmd); err != nil {
			log.Errorf("%s, %v", cmd, err)
			return err
		}
		log.Debugf("Execute manifest annotate command: %s", cmd)
	}

	manifestPushCmd := fmt.Sprintf("docker manifest push --purge --insecure %s", target)
	if err := Command.ExecuteCommandWithEnv(manifestEnv, "/bin/sh", "-c", manifestPushCmd); err != nil {
		log.Errorf("%s, %v", manifestPushCmd, err)
		return err
	}
	log.Infof("Execute manifest push command: %s", manifestPushCmd)
	return nil
}
