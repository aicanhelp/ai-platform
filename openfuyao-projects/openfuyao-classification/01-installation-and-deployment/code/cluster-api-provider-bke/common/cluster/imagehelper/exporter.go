/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package imagehelper

import (
	"github.com/pkg/errors"

	common "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize/versions"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
)

// ImageExporter is a utility for exporting Kubernetes component images
type ImageExporter struct {
	Repo string
	// kubernetes version
	Version     string
	EtcdVersion string
	imageMap    map[string]string
	imageLi     []string
}

// NewImageExporter creates a new ImageExporter instance with the specified repository,
// kubernetes version and etcd version
func NewImageExporter(repo, version, etcdVersion string) *ImageExporter {
	return &ImageExporter{
		Repo:        repo,
		Version:     version,
		EtcdVersion: etcdVersion,
	}
}

func (e *ImageExporter) ExportImageMap() (map[string]string, error) {
	if err := e.validate(); err != nil {
		return nil, err
	}

	e.generateImageMap()

	if e.Repo != "" && e.imageMap != nil {
		e.imageMapAddRepo()
	}

	if e.imageMap == nil {
		return nil, errors.New("could not export image map, image map is nil")
	}

	return e.imageMap, nil
}

// ExportImageMapWithBootStrapPhase export image map with bootstrap phase
// phase: same as bkeagent kubeadm plugin phase
func (e *ImageExporter) ExportImageMapWithBootStrapPhase(phase string) (map[string]string, error) {
	if err := e.validate(); err != nil {
		return nil, err
	}
	e.generateImageMap()

	if e.imageMap == nil {
		return nil, errors.New("could not export image map, image map is nil")
	}

	if e.Repo != "" {
		e.imageMapAddRepo()
	}

	switch phase {

	case common.InitControlPlane:
		return e.imageMap, nil
	case common.JoinControlPlane:
		return e.imageMap, nil
	case common.JoinWorker:
		return map[string]string{
			initialize.DefaultPauseImageName: e.imageMap[initialize.DefaultPauseImageName],
		}, nil
	case common.UpgradeControlPlane:
		return e.imageMap, nil
	case common.UpgradeWorker:
		return map[string]string{
			initialize.DefaultPauseImageName: e.imageMap[initialize.DefaultPauseImageName],
		}, nil
	default:
		return nil, errors.Errorf("could not export image map, phase %q is not supported", phase)
	}
}

// ExportImageList exports a list of Kubernetes component images
func (e *ImageExporter) ExportImageList() ([]string, error) {
	if err := e.validate(); err != nil {
		return nil, err
	}

	e.generateImageMap()
	if e.Repo != "" && e.imageMap != nil {
		e.imageMapAddRepo()
	}
	e.generateImageList()
	if e.imageLi == nil || len(e.imageLi) == 0 {
		return nil, errors.New("could not export image list, image list is nil")
	}
	return e.imageLi, nil
}

func (e *ImageExporter) validate() error {
	if err := validation.ValidateK8sVersion(e.Version); err != nil {
		return err
	}
	// todo validae repo
	return nil
}

func (e *ImageExporter) generateImageMap() {
	if e.Version == "" {
		return
	}
	k8sComponentImageMapWithoutRepo := map[string]string{
		initialize.DefaultAPIServerImageName: GetImageNameWithTag(
			initialize.DefaultAPIServerImageName, e.Version),
		initialize.DefaultControllerManagerImageName: GetImageNameWithTag(
			initialize.DefaultControllerManagerImageName, e.Version),
		initialize.DefaultSchedulerImageName: GetImageNameWithTag(
			initialize.DefaultSchedulerImageName, e.Version),
		initialize.DefaultEtcdImageName: GetImageNameWithTag(
			initialize.DefaultEtcdImageName, versions.EtcdImageTag()),
	}

	if e.EtcdVersion != "" {
		k8sComponentImageMapWithoutRepo[initialize.DefaultEtcdImageName] =
			GetImageNameWithTag(initialize.DefaultEtcdImageName, e.EtcdVersion)
	}

	e.imageMap = k8sComponentImageMapWithoutRepo
}

func (e *ImageExporter) generateImageList() {
	if e.imageMap == nil {
		return
	}
	var imgLi []string
	for _, img := range e.imageMap {
		imgLi = append(imgLi, img)
	}
	e.imageLi = imgLi
}

func (e *ImageExporter) imageMapAddRepo() {
	if e.Repo == "" || e.imageMap == nil {
		return
	}
	for k, v := range e.imageMap {
		e.imageMap[k] = RepoJoinImageName(e.Repo, v)
	}
}
