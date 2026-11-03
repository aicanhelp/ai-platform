/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package mfutil

import (
	"fmt"
	"os"
	"path"
	"path/filepath"

	"github.com/pkg/errors"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/cluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type BootScope struct {
	ClusterName      string
	ClusterNamespace string
	BkeConfig        *bkev1beta1.BKEConfig
	KubeletConfigRef *bkev1beta1.KubeletConfigRef
	HostName         string
	HostIP           string
	CurrentNode      bkenode.Node
	Extra            map[string]interface{}
	IsImmutableOS    bool // worker 节点是否使用不可变 OS（KubeOS）
}

// HasOpenFuyaoAddon is used to check if the cluster config has deployed openfuyao system controller
func (b *BootScope) HasOpenFuyaoAddon() bool {
	for _, addon := range b.BkeConfig.Addons {
		if addon.Name == utils.OpenFuyaoSystemController {
			log.Info("cluster config has deployed openfuyao system controller")
			return true
		}
	}
	log.Info("cluster config has not deployed openfuyao system controller")
	return false
}

func GenerateManifestYaml(components Components, boot *BootScope) error {
	cfg := bkeinit.BkeConfig(*boot.BkeConfig)
	log.Infof("generate %q version cluster manifests", cfg.Cluster.KubernetesVersion)
	log.Infof("generate static pod yaml, use %q image repository", cfg.ImageFuyaoRepo())
	log.Infof("generate static pod yaml, use %q certificate ", cfg.Cluster.CertificatesDir)

	nodesData, err := cluster.GetNodesData(boot.ClusterNamespace, boot.ClusterName)
	if err != nil {
		return err
	}
	nodes := bkenode.Nodes(nodesData)
	// sure that we need render etcd yaml
	flag := nodes.Filter(bkenode.FilterOptions{"Role": bkenode.EtcdNodeRole, "IP": boot.HostIP}).Length() == 1

	for _, component := range components {
		if component.Name == Etcd && !flag {
			hostName := utils.HostName()
			log.Infof("the node %q is not an etcd node, skip generate etcd static pod yaml", hostName)
			continue
		}
		if err := component.RenderFunc(component, boot); err != nil {
			return errors.Wrapf(err, "failed to render %s static pod yaml", component.Name)
		}
	}
	return nil

}

func GenerateHAManifestYaml(components HAComponents, cfg map[string]interface{}) error {
	for _, component := range components {
		if err := component.RenderFunc(component, cfg); err != nil {
			return errors.Wrapf(err, "failed to render %s static pod yaml", component.Name)
		}
	}
	return nil
}

// GenerateManifestPath 通用函数
func GenerateManifestPath(name string, mfPath *string, defaultPathFunc func() string) string {
	if *mfPath == "" {
		defaultPath := defaultPathFunc()
		log.Infof("generate %q static pod yaml using default path %q", name, defaultPath)
		*mfPath = defaultPath
	} else {
		log.Infof("generate %q static pod yaml to %q", name, *mfPath)
	}

	if err := os.MkdirAll(*mfPath, utils.RwRR); err != nil {
		return ""
	}

	return filepath.Join(*mfPath, fmt.Sprintf("%s.yaml", name))
}

func pathForManifest(c *BKEComponent) string {
	return GenerateManifestPath(c.Name, &c.MfPath, GetDefaultManifestsPath)
}

func pathForHAManifestConf(c *BKEHAComponent) string {
	if err := os.MkdirAll(c.ConfPath, utils.RwRR); err != nil {
		return ""
	}
	p := path.Join(c.ConfPath, c.ConfName)
	return p
}

func pathForHAManifestScript(c *BKEHAComponent, script string) string {
	if err := os.MkdirAll(c.ConfPath, utils.RwRR); err != nil {
		return ""
	}
	p := path.Join(c.ConfPath, script)
	return p
}

func pathForHAManifest(c *BKEHAComponent) string {
	return GenerateManifestPath(c.Name, &c.MfPath, GetDefaultManifestsPath)
}
