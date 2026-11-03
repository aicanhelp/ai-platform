/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package agentssh

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	agentdownload "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/download"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	DefaultBKEAgentArtifact = "bkeagent-latest-linux-{.arch}"
	defaultServiceName      = "bkeagent.service"
)

// Staging holds provider-local paths for upgrade artifacts.
type Staging struct {
	Dir         string
	ServicePath string
}

// ArtifactParams configures binary/service resolution from cluster HTTP repo.
type ArtifactParams struct {
	BaseURL           string
	BinaryArtifact    string
	TargetVersion     string
	ServiceCandidates []string
}

// ParamsFromCluster builds artifact download parameters from BKECluster and target version.
func ParamsFromCluster(bkeCluster *bkev1beta1.BKECluster, targetVersion string) ArtifactParams {
	if bkeCluster == nil || bkeCluster.Spec.ClusterConfig == nil {
		return ArtifactParams{
			BinaryArtifact:    DefaultBKEAgentArtifact,
			ServiceCandidates: defaultServiceCandidates(""),
		}
	}
	cfg := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)
	baseURL := strings.TrimSuffix(clusterutil.BuildYumRepoDownloadBaseURL(cfg), "/")
	return ArtifactParams{
		BaseURL:           baseURL,
		BinaryArtifact:    BinaryArtifactName(cfg, targetVersion),
		TargetVersion:     targetVersion,
		ServiceCandidates: ServiceCandidates(cfg, targetVersion),
	}
}

// BinaryArtifactName returns the artifact file name (may contain {.arch}).
func BinaryArtifactName(cfg bkeinit.BkeConfig, version string) string {
	if v, ok := cfg.CustomExtra["bkeagent"]; ok && strings.TrimSpace(v) != "" {
		return v
	}
	if version != "" {
		version = strings.TrimPrefix(version, "v")
		return fmt.Sprintf("bkeagent-%s-linux-{.arch}", version)
	}
	return DefaultBKEAgentArtifact
}

// ServiceCandidates returns HTTP service file names to try in order.
func ServiceCandidates(cfg bkeinit.BkeConfig, version string) []string {
	if v, ok := cfg.CustomExtra["bkeagent-service"]; ok && strings.TrimSpace(v) != "" {
		return []string{v}
	}
	return defaultServiceCandidates(version)
}

func defaultServiceCandidates(version string) []string {
	if version != "" {
		version = strings.TrimPrefix(version, "v")
		return []string{
			fmt.Sprintf("bkeagent-%s.service", version),
			defaultServiceName,
		}
	}
	return []string{defaultServiceName}
}

// BinaryURLForArch builds the full download URL for a specific architecture.
func BinaryURLForArch(params ArtifactParams, arch string) string {
	artifact := strings.ReplaceAll(params.BinaryArtifact, "{.arch}", arch)
	return fmt.Sprintf("%s/%s", params.BaseURL, strings.TrimPrefix(artifact, "/"))
}

// NewStagingDir creates a temp directory for upgrade artifacts.
func NewStagingDir(clusterName string) (string, error) {
	return os.MkdirTemp(os.TempDir(), fmt.Sprintf("bkeagent-upgrade-%s-", clusterName))
}

func removeStagingDir(dir string) error {
	if dir == "" {
		return nil
	}
	return os.RemoveAll(dir)
}

// PrepareServiceFile prefers bkeagent.service from the HTTP binary source (bootstrap node);
// on failure it falls back to the same template rendering as EnsureBKEAgent.
func PrepareServiceFile(bkeCluster *bkev1beta1.BKECluster, stagingDir string, params ArtifactParams) (string, error) {
	servicePath := filepath.Join(stagingDir, defaultServiceName)

	if params.BaseURL != "" {
		var lastErr error
		for _, candidate := range params.ServiceCandidates {
			url := fmt.Sprintf("%s/%s", params.BaseURL, strings.TrimPrefix(candidate, "/"))
			raw, err := agentdownload.DownloadBytes(url)
			if err != nil {
				lastErr = err
				continue
			}
			if err := phaseutil.WriteRenderedBKEAgentServiceFile(bkeCluster, servicePath, raw); err != nil {
				return "", err
			}
			log.Infof("prepared bkeagent.service from bootstrap binary source %s (candidate %s) with cluster template rendering",
				params.BaseURL, candidate)
			return servicePath, nil
		}
		log.Warnf("download bkeagent.service from bootstrap binary source %s failed, fallback to provider template: %v",
			params.BaseURL, lastErr)
	} else {
		log.Warnf("HTTP binary source base URL is empty, use provider template for bkeagent.service")
	}

	if err := phaseutil.RenderBKEAgentServiceFile(bkeCluster, servicePath); err != nil {
		return "", errors.Wrap(err, "render bkeagent.service from template")
	}
	return servicePath, nil
}

// DownloadBinariesForArchs downloads bkeagent binaries per architecture into stagingDir/{arch}/bkeagent.
func DownloadBinariesForArchs(stagingDir string, params ArtifactParams, archs []string) error {
	seen := make(map[string]struct{}, len(archs))
	for _, arch := range archs {
		arch = strings.TrimSpace(arch)
		if arch == "" || arch == "unknown" {
			return errors.Errorf("invalid node architecture %q", arch)
		}
		if _, ok := seen[arch]; ok {
			continue
		}
		seen[arch] = struct{}{}

		archDir := filepath.Join(stagingDir, arch)
		url := BinaryURLForArch(params, arch)
		binaryName := fmt.Sprintf("bkeagent_linux_%s", arch)
		if err := agentdownload.ExecDownloadForArch(url, archDir, binaryName, "0755", arch); err != nil {
			return errors.Wrapf(err, "download bkeagent for arch %s from %s", arch, url)
		}
	}
	if len(seen) == 0 {
		return errors.New("no architectures to download")
	}
	return nil
}

// PrepareStaging downloads service and binaries for the given node architectures.
func PrepareStaging(bkeCluster *bkev1beta1.BKECluster, params ArtifactParams, archs []string) (*Staging, error) {
	dir, err := NewStagingDir(bkeCluster.Name)
	if err != nil {
		return nil, err
	}

	staging := &Staging{Dir: dir}
	servicePath, err := PrepareServiceFile(bkeCluster, dir, params)
	if err != nil {
		if rmErr := removeStagingDir(dir); rmErr != nil {
			return nil, errors.Wrapf(err, "prepare service file (cleanup staging: %v)", rmErr)
		}
		return nil, err
	}
	staging.ServicePath = servicePath

	if err := DownloadBinariesForArchs(dir, params, archs); err != nil {
		if rmErr := removeStagingDir(dir); rmErr != nil {
			return nil, errors.Wrapf(err, "download binaries (cleanup staging: %v)", rmErr)
		}
		return nil, err
	}
	return staging, nil
}

// Cleanup removes the staging directory.
func (s *Staging) Cleanup() {
	if s == nil || s.Dir == "" {
		return
	}
	if err := removeStagingDir(s.Dir); err != nil {
		log.ControllerLogger("agentssh").Warnf("failed to remove staging dir %s: %v", s.Dir, err)
	}
	s.Dir = ""
}
