/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package upgrade

// ComponentManifestVersion is the default manifest bundle version for declarative upgrade components.
const ComponentManifestVersion = "v1.0.0"

// Inline handler names match phaseframe.Phase.Name() and ComponentVersion.spec.inline.handler.
const (
	InlineHandlerPreUpgradeResources = "EnsurePreUpgradeResources"
	InlineHandlerEtcdUpgrade         = "EnsureEtcdUpgrade"
	InlineHandlerMasterUpgrade       = "EnsureMasterUpgrade"
	InlineHandlerWorkerUpgrade       = "EnsureWorkerUpgrade"
	InlineHandlerContainerdUpgrade   = "EnsureContainerdUpgrade"
	InlineHandlerAgentUpgrade        = "EnsureAgentUpgrade"
	InlineHandlerNginxUpgrade        = "EnsureNginxUpgrade"
)

// InlineHandlerVersion is the version key used in ComponentFactory.Register.
const InlineHandlerVersion = ComponentManifestVersion

// UpgradeExecutionMode describes how a component is executed in declarative upgrade.
type UpgradeExecutionMode string

const (
	UpgradeExecutionManifest UpgradeExecutionMode = "manifest"
	UpgradeExecutionInline   UpgradeExecutionMode = "inline"
)

// UpgradeComponentSpec maps legacy phases to declarative upgrade entries.
type UpgradeComponentSpec struct {
	// Name is the ReleaseImage component name (VersionContext key and DAG node name).
	Name    string
	Version string
	Mode    UpgradeExecutionMode
	// ManifestPath is set for manifest-mode components (e.g. provider/v1.0.0/component.yaml).
	ManifestPath string
	// LegacyPhase is the pre-declarative BKECluster phase name, if any.
	LegacyPhase string
	// InlineHandler is the ComponentFactory handler key for inline mode.
	InlineHandler string
}

// DeclarativeUpgradeCatalog returns the canonical upgrade component table for ReleaseImage DAG.
func DeclarativeUpgradeCatalog() []UpgradeComponentSpec {
	catalog := baseUpgradeCatalog()
	return append(catalog, nodeUpgradeCatalog()...)
}

func baseUpgradeCatalog() []UpgradeComponentSpec {
	return []UpgradeComponentSpec{
		{
			Name:          ComponentPreUpgradeResources,
			Version:       ComponentManifestVersion,
			Mode:          UpgradeExecutionInline,
			LegacyPhase:   InlineHandlerPreUpgradeResources,
			InlineHandler: InlineHandlerPreUpgradeResources,
		},
		{
			Name:         ComponentProvider,
			Version:      ComponentManifestVersion,
			Mode:         UpgradeExecutionManifest,
			ManifestPath: ManifestComponentManifestPath(ComponentProvider, ComponentManifestVersion),
			LegacyPhase:  "EnsureProviderSelfUpgrade",
		},
		{
			Name:          ComponentBKEAgent,
			Version:       ComponentManifestVersion,
			Mode:          UpgradeExecutionInline,
			LegacyPhase:   InlineHandlerAgentUpgrade,
			InlineHandler: InlineHandlerAgentUpgrade,
		},
		{
			Name:         ComponentKubeProxy,
			Version:      ComponentManifestVersion,
			Mode:         UpgradeExecutionManifest,
			ManifestPath: ManifestComponentManifestPath(ComponentKubeProxy, ComponentManifestVersion),
		},
		{
			Name:         ComponentCoreDNS,
			Version:      ComponentManifestVersion,
			Mode:         UpgradeExecutionManifest,
			ManifestPath: ManifestComponentManifestPath(ComponentCoreDNS, ComponentManifestVersion),
			LegacyPhase:  "EnsureComponentUpgrade",
		},
	}
}

func nodeUpgradeCatalog() []UpgradeComponentSpec {
	return []UpgradeComponentSpec{
		{
			Name:          ComponentEtcd,
			Version:       ComponentManifestVersion,
			Mode:          UpgradeExecutionInline,
			LegacyPhase:   InlineHandlerEtcdUpgrade,
			InlineHandler: InlineHandlerEtcdUpgrade,
		},
		{
			Name:          ComponentKubernetesMaster,
			Version:       ComponentManifestVersion,
			Mode:          UpgradeExecutionInline,
			LegacyPhase:   InlineHandlerMasterUpgrade,
			InlineHandler: InlineHandlerMasterUpgrade,
		},
		{
			Name:          ComponentKubernetesWorker,
			Version:       ComponentManifestVersion,
			Mode:          UpgradeExecutionInline,
			LegacyPhase:   InlineHandlerWorkerUpgrade,
			InlineHandler: InlineHandlerWorkerUpgrade,
		},
		{
			Name:          ComponentContainerd,
			Version:       ComponentManifestVersion,
			Mode:          UpgradeExecutionInline,
			LegacyPhase:   InlineHandlerContainerdUpgrade,
			InlineHandler: InlineHandlerContainerdUpgrade,
		},
		{
			Name:          ComponentNginxProxy,
			Version:       ComponentManifestVersion,
			Mode:          UpgradeExecutionInline,
			LegacyPhase:   InlineHandlerNginxUpgrade,
			InlineHandler: InlineHandlerNginxUpgrade,
		},
	}
}

// ManifestComponentManifestPath returns the relative manifest path under bke-manifests.
func ManifestComponentManifestPath(componentName, version string) string {
	return componentName + "/" + version + "/component.yaml"
}

// InlineUpgradeHandlers returns handler names registered in ComponentFactory.
func InlineUpgradeHandlers() []string {
	var handlers []string
	for _, spec := range DeclarativeUpgradeCatalog() {
		if spec.Mode == UpgradeExecutionInline && spec.InlineHandler != "" {
			handlers = append(handlers, spec.InlineHandler)
		}
	}
	return handlers
}
