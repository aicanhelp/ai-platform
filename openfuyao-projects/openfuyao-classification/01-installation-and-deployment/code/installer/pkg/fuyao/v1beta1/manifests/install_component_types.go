/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package manifests

import (
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
)

// InstallComponentManifest is installed components manifest
type InstallComponentManifest struct {
	APIVersion string        `yaml:"apiVersion"`
	Kind       string        `yaml:"kind"`
	Metadata   base.Metadata `yaml:"metadata"`
	Spec       InstallSpec   `yaml:"spec"`
}

// NewInstallComponentManifest is new InstallComponentManifest
func NewInstallComponentManifest() *InstallComponentManifest {
	installManifests := make([]*InstallManifest, 0)
	installSpec := InstallSpec{
		Manifests: installManifests,
	}
	return &InstallComponentManifest{
		APIVersion: constants.ApiVersion,
		Kind:       constants.Manifests,
		Metadata: base.Metadata{
			Name: "install-component-manifest",
		},
		Spec: installSpec,
	}
}

// InstallSpec is spec
type InstallSpec struct {
	Manifests []*InstallManifest `yaml:"manifests"`
}

// InstallManifest is node install component manifest
type InstallManifest struct {
	Node       Node                `yaml:"node"`
	Components []*InstallComponent `yaml:"components"`
}

// NewInstallManifest is new InstallManifest
func NewInstallManifest(node Node, components []*InstallComponent) *InstallManifest {
	return &InstallManifest{
		Node:       node,
		Components: components,
	}
}

// Node is node info
type Node struct {
	Ip string `yaml:"ip"`
}

// NewNode is new Node
func NewNode(ip string) Node {
	return Node{
		Ip: ip,
	}
}

// InstallComponent is install component
type InstallComponent struct {
	Name          string `yaml:"name"`
	Version       string `yaml:"version"`
	BootstrapNode string `yaml:"bootstrapNode"`
}

// NewInstallComponent is new InstallComponent
func NewInstallComponent(name, version, bootstrapNode string) *InstallComponent {
	return &InstallComponent{
		Name:          name,
		Version:       version,
		BootstrapNode: bootstrapNode,
	}
}
