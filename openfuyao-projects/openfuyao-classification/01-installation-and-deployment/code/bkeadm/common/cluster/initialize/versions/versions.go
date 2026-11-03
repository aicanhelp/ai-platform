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

// Package versions provides the embedded component version manifest.
package versions

import (
	_ "embed"
	"fmt"
	"strings"
	"sync"

	"sigs.k8s.io/yaml"
)

const (
	supportedAPIVersion      = "bke.bocloud.com/v1"
	supportedKind            = "VersionManifest"
	supportedManifestVersion = "1"
)

// Manifest is the top-level version manifest.
type Manifest struct {
	APIVersion string `json:"apiVersion" yaml:"apiVersion"`
	Kind       string `json:"kind" yaml:"kind"`
	Spec       Spec   `json:"spec" yaml:"spec"`
}

// Spec contains the schema version and component versions.
type Spec struct {
	ManifestVersion   string            `json:"manifestVersion" yaml:"manifestVersion"`
	ComponentVersions ComponentVersions `json:"componentVersions" yaml:"componentVersions"`
	ImageTags         ImageTags         `json:"imageTags" yaml:"imageTags"`
}

// ComponentVersions contains default component versions.
type ComponentVersions struct {
	Kubernetes string `json:"kubernetes" yaml:"kubernetes"`
	Etcd       string `json:"etcd" yaml:"etcd"`
	Containerd string `json:"containerd" yaml:"containerd"`
	OpenFuyao  string `json:"openFuyao" yaml:"openFuyao"`
}

// ImageTags contains default image tags.
type ImageTags struct {
	Etcd  string `json:"etcd" yaml:"etcd"`
	Pause string `json:"pause" yaml:"pause"`
}

//go:embed versions.yaml
var manifestData []byte

var (
	loadOnce       sync.Once
	cachedManifest Manifest
	cachedErr      error
)

// Load parses and validates the embedded version manifest once.
// A copy is returned so callers cannot mutate the cached manifest.
func Load() (*Manifest, error) {
	loadOnce.Do(func() {
		cachedManifest, cachedErr = parse(manifestData)
	})
	if cachedErr != nil {
		return nil, cachedErr
	}
	manifest := cachedManifest
	return &manifest, nil
}

func parse(data []byte) (Manifest, error) {
	var manifest Manifest
	if err := yaml.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, fmt.Errorf("parse embedded versions.yaml: %w", err)
	}
	if err := Validate(&manifest); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

// Validate performs only the structural and required-field checks needed to
// safely consume the manifest. Component compatibility is intentionally not
// validated here.
func Validate(manifest *Manifest) error {
	if manifest == nil {
		return fmt.Errorf("invalid versions manifest: manifest is required")
	}
	required := []struct {
		path  string
		value string
	}{
		{"apiVersion", manifest.APIVersion},
		{"kind", manifest.Kind},
		{"spec.manifestVersion", manifest.Spec.ManifestVersion},
		{"spec.componentVersions.kubernetes", manifest.Spec.ComponentVersions.Kubernetes},
		{"spec.componentVersions.etcd", manifest.Spec.ComponentVersions.Etcd},
		{"spec.componentVersions.containerd", manifest.Spec.ComponentVersions.Containerd},
		{"spec.componentVersions.openFuyao", manifest.Spec.ComponentVersions.OpenFuyao},
		{"spec.imageTags.etcd", manifest.Spec.ImageTags.Etcd},
		{"spec.imageTags.pause", manifest.Spec.ImageTags.Pause},
	}
	for _, field := range required {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("invalid versions manifest: %s is required", field.path)
		}
	}
	if manifest.APIVersion != supportedAPIVersion {
		return fmt.Errorf("invalid versions manifest: unsupported apiVersion %q", manifest.APIVersion)
	}
	if manifest.Kind != supportedKind {
		return fmt.Errorf("invalid versions manifest: unsupported kind %q", manifest.Kind)
	}
	if manifest.Spec.ManifestVersion != supportedManifestVersion {
		return fmt.Errorf("invalid versions manifest: unsupported spec.manifestVersion %q", manifest.Spec.ManifestVersion)
	}
	return nil
}

func mustLoad() *Manifest {
	manifest, err := Load()
	if err != nil {
		panic(fmt.Sprintf("versions manifest was not initialized: %v", err))
	}
	return manifest
}

// ManifestVersion returns the embedded manifest schema version.
func ManifestVersion() string { return mustLoad().Spec.ManifestVersion }

// KubernetesVersion returns the default Kubernetes version.
func KubernetesVersion() string { return mustLoad().Spec.ComponentVersions.Kubernetes }

// EtcdVersion returns the default etcd version.
func EtcdVersion() string { return mustLoad().Spec.ComponentVersions.Etcd }

// ContainerdVersion returns the default containerd version.
func ContainerdVersion() string { return mustLoad().Spec.ComponentVersions.Containerd }

// OpenFuyaoVersion returns the default openFuyao version.
func OpenFuyaoVersion() string { return mustLoad().Spec.ComponentVersions.OpenFuyao }

// EtcdImageTag returns the default etcd image tag.
func EtcdImageTag() string { return mustLoad().Spec.ImageTags.Etcd }

// PauseImageTag returns the default pause image tag.
func PauseImageTag() string { return mustLoad().Spec.ImageTags.Pause }
