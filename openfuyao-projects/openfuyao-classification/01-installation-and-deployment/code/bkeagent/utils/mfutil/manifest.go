package mfutil

import (
	"fmt"
	"os"
	"path"

	"github.com/pkg/errors"
	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	bkeinit "gopkg.openfuyao.cn/bkecommon/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

type BootScope struct {
	ClusterName      string
	BkeConfig        *bkev1beta1.BKEConfig
	KubeletConfigRef *bkev1beta1.KubeletConfigRef
	HostName         string
	HostIP           string
	CurrentNode      bkenode.Node
	Extra            map[string]interface{}
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
	log.Infof("generate static pod yaml, use %q image repository", cfg.ImageRepo())
	log.Infof("generate static pod yaml, use %q certificate ", cfg.Cluster.CertificatesDir)

	nodes := bkenode.Nodes(cfg.Nodes)
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

func pathForManifest(c *BKEComponent) string {
	if c.MfPath == "" {
		log.Infof("generate %q static pod yaml using default path %q", c.Name, GetDefaultManifestsPath())
		c.MfPath = GetDefaultManifestsPath()
	} else {
		log.Infof("generate %q static pod yaml to %q", c.Name, c.MfPath)
	}
	_ = os.MkdirAll(c.MfPath, 0644)
	return path.Join(c.MfPath, fmt.Sprintf("%s.yaml", c.Name))
}

func pathForHAManifestConf(c *BKEHAComponent) string {
	_ = os.MkdirAll(c.ConfPath, 0644)
	p := path.Join(c.ConfPath, c.ConfName)
	return p
}

func pathForHAManifestScript(c *BKEHAComponent, script string) string {
	_ = os.MkdirAll(c.ConfPath, 0644)
	p := path.Join(c.ConfPath, script)
	return p
}

func pathForHAManifest(c *BKEHAComponent) string {
	if c.MfPath == "" {
		log.Infof("generate %q static pod yaml using default path %q", c.Name, GetDefaultManifestsPath())
		c.MfPath = GetDefaultManifestsPath()
	} else {
		log.Infof("generate %q static pod yaml to %q", c.Name, c.MfPath)
	}
	_ = os.MkdirAll(c.MfPath, 0644)
	return path.Join(c.MfPath, fmt.Sprintf("%s.yaml", c.Name))
}
