/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	kubedrain "k8s.io/kubectl/pkg/drain"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

type UpgradeStrategy string

const (
	UpgradePolicyRolling UpgradeStrategy = "rolling"
	UpgradePolicyDrain   UpgradeStrategy = "proportion"
	DrainTime                            = 20
)

// writer struct implements io.Writer interface as a pass-through for klog.
type writer struct {
	logFunc func(reason, msg string, args ...interface{})
}

// Write trans string(p) into writer's logFunc and returns len(p).
func (w writer) Write(p []byte) (int, error) {
	msg := string(p)
	msg = strings.TrimSuffix(msg, "\n")
	w.logFunc(constant.DrainNodeReason, msg)
	return len(p), nil
}

func NewDrainer(ctx context.Context, cs kubernetes.Interface, dynamicClient dynamic.Interface, dryRun bool, log *bkev1beta1.BKELogger) *kubedrain.Helper {
	var dryRunStrategy cmdutil.DryRunStrategy
	if dryRun {
		dryRunStrategy = cmdutil.DryRunServer
	}
	return &kubedrain.Helper{
		Client:              cs,
		Ctx:                 ctx,
		Force:               true,
		IgnoreAllDaemonSets: true,
		DeleteEmptyDirData:  true,
		GracePeriodSeconds:  -1,
		DryRunStrategy:      dryRunStrategy,
		Timeout:             DrainTime * time.Second,
		OnPodDeletedOrEvicted: func(pod *corev1.Pod, isEvicted bool) {
			vStr := "Deleted"
			if isEvicted {
				vStr = "Evicted"
			}
			log.Info(constant.DrainNodeReason, fmt.Sprintf("%s pod %q from Node", vStr, utils.ClientObjNS(pod)))
		},
		Out:    writer{log.Info},
		ErrOut: writer{log.Error},
	}
}

type ImageUpdate struct {
	ImageName string // 镜像名称（不带tag）
	PodPrefix string // Pod名称前缀
	NameSpace string // Pod归属命名空间
	NewTag    string // 新的镜像tag
}

type PatchConfig struct {
	Registry          Registry `json:"registry" yaml:"registry"`
	OpenFuyaoVersion  string   `json:"openfuyaoVersion" yaml:"openfuyaoVersion"`
	ContainerdVersion string   `json:"containerdVersion" yaml:"containerdVersion"`
	KubernetesVersion string   `json:"kubernetesVersion" yaml:"kubernetesVersion"`
	Repos             []Repo   `json:"repos" yaml:"repos"`
	Files             []File   `json:"files" yaml:"files"`
}

type Registry struct {
	ImageAddress string   `json:"imageAddress" yaml:"imageAddress"`
	Architecture []string `json:"architecture" yaml:"architecture"`
}

type Repo struct {
	Architecture []string   `json:"architecture" yaml:"architecture"`
	IsKubernetes bool       `json:"isKubernetes" yaml:"isKubernetes"`
	SubImages    []SubImage `json:"subImages" yaml:"subImages"`
}

type SubImage struct {
	SourceRepo string  `json:"sourceRepo" yaml:"sourceRepo"`
	TargetRepo string  `json:"targetRepo" yaml:"targetRepo"`
	Images     []Image `json:"images" yaml:"images"`
}

type Image struct {
	Name        string    `json:"name" yaml:"name"`
	UsedPodInfo []PodInfo `json:"usedPodInfo" yaml:"usedPodInfo"`
	Tag         []string  `json:"tag" yaml:"tag"`
}

type PodInfo struct {
	PodPrefix string `json:"podPrefix" yaml:"podPrefix"`
	NameSpace string `json:"namespace" yaml:"namespace"`
}

type File struct {
	Address string     `json:"address" yaml:"address"`
	Files   []FileInfo `json:"files" yaml:"files"`
}

type FileInfo struct {
	FileName  string `json:"fileName" yaml:"fileName"`
	FileAlias string `json:"fileAlias" yaml:"fileAlias"`
}

func GetPatchConfig(data string) (*PatchConfig, error) {
	cfg := &PatchConfig{}
	if err := yaml.Unmarshal([]byte(data), cfg); err != nil {
		return nil, errors.Errorf("Unable to serialize data %s, err %s", data, err)
	}
	return cfg, nil
}

// ErrLocalPatchKeyNotFound indicates patch.<version> is missing from the local bke-cluster ConfigMap.
var ErrLocalPatchKeyNotFound = errors.New("patch key not found in local config")

// LoadPatchConfig loads PatchConfig for openFuyaoVersion from local bke-cluster CM
// and the openfuyao-patch/cm.<version> ConfigMap.
func LoadPatchConfig(ctx context.Context, c client.Client, openFuyaoVersion string) (*PatchConfig, error) {
	bkeCmKey := fmt.Sprintf("patch.%s", openFuyaoVersion)
	patchCmKey := fmt.Sprintf("cm.%s", openFuyaoVersion)

	localConfigMap := &corev1.ConfigMap{}
	if err := c.Get(ctx, constant.GetLocalConfigMapObjectKey(), localConfigMap); err != nil {
		return nil, fmt.Errorf("get cm failed: %w", err)
	}

	if _, ok := localConfigMap.Data[bkeCmKey]; !ok {
		return nil, fmt.Errorf("%w: %s", ErrLocalPatchKeyNotFound, bkeCmKey)
	}

	cmKey := client.ObjectKey{
		Namespace: "openfuyao-patch",
		Name:      patchCmKey,
	}
	patchConfigMap := &corev1.ConfigMap{}
	if err := c.Get(ctx, cmKey, patchConfigMap); err != nil {
		return nil, fmt.Errorf("get cm failed: %w", err)
	}

	data, ok := patchConfigMap.Data[openFuyaoVersion]
	if !ok {
		return nil, fmt.Errorf("patch info %s not found in patch config", openFuyaoVersion)
	}

	return GetPatchConfig(data)
}
