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

package main

import (
	_ "embed"
	"os"

	_ "gopkg.openfuyao.cn/bkeadm/bootstrap/preflightlogenv"
	"gopkg.openfuyao.cn/bkeadm/cmd"
	"gopkg.openfuyao.cn/bkeadm/common/cluster/initialize/versions"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
	"gopkg.openfuyao.cn/bkeadm/utils/version"
)

var (
	//go:embed configs/preflight/preflight.yaml
	embeddedPreflightPolicy []byte
)

var gitCommitId = "dev"
var architecture = ""
var timestamp = ""
var ver = "v1.0"

func main() {
	cmd.SetEmbeddedPreflightPolicy(embeddedPreflightPolicy)
	if _, err := versions.Load(); err != nil {
		log.Errorf("failed to load embedded versions manifest: %v", err)
		os.Exit(1)
	}
	log.Infof(
		"Loaded versions manifest: manifestVersion=%s, kubernetes=%s, etcd=%s, containerd=%s, openFuyao=%s, etcdImageTag=%s, pauseImageTag=%s",
		versions.ManifestVersion(),
		versions.KubernetesVersion(),
		versions.EtcdVersion(),
		versions.ContainerdVersion(),
		versions.OpenFuyaoVersion(),
		versions.EtcdImageTag(),
		versions.PauseImageTag(),
	)
	if version.Version == "" {
		version.GitCommitID = gitCommitId
		version.Version = ver
		version.Architecture = architecture
		version.Timestamp = timestamp
	}
	os.Exit(cmd.Execute())
}
