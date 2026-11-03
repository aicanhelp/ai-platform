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

// Package bootstrap installs Bridge's default InferNexServiceConfig templates
// at controller startup. It is the OLM-bundle counterpart of the Helm chart's
// templates.installDefaultConfigs=true behavior: OLM bundles cannot ship raw CR
// instances, so the controller has to materialize them itself.
package bootstrap

import (
	"context"
	"embed"
	"fmt"
	"path"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/yaml"
)

//go:embed templates/*.yaml
var templatesFS embed.FS

const templatesDir = "templates"

// DefaultTemplatesRunnable returns a Manager-Runnable that, after leader
// election, ensures the embedded default InferNexServiceConfig templates exist
// in the given namespace. Create-if-missing only: pre-existing templates are
// left untouched so operator customizations survive controller restarts.
//
// Namespace is the controller's INFERNEX_TEMPLATE_NAMESPACE (downward-API
// injected). An empty namespace skips the bootstrap.
func DefaultTemplatesRunnable(c client.Client, namespace string) manager.Runnable {
	return &templateBootstrapper{client: c, namespace: namespace}
}

type templateBootstrapper struct {
	client    client.Client
	namespace string
}

// NeedLeaderElection ensures only the leader bootstraps templates; followers
// would race on Create and produce noisy AlreadyExists errors.
func (b *templateBootstrapper) NeedLeaderElection() bool { return true }

// Start is invoked by controller-runtime once the leader lease is held.
// It returns nil even on per-template errors so a single broken template
// never prevents the controller manager from running.
func (b *templateBootstrapper) Start(ctx context.Context) error {
	log := ctrl.LoggerFrom(ctx).WithName("default-templates")
	if b.namespace == "" {
		log.Info("template namespace empty; skipping default template bootstrap")
		return nil
	}
	entries, err := templatesFS.ReadDir(templatesDir)
	if err != nil {
		return fmt.Errorf("read embedded templates dir: %w", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		filename := e.Name()
		data, err := templatesFS.ReadFile(path.Join(templatesDir, filename))
		if err != nil {
			log.Error(err, "failed to read embedded template; skipping", "file", filename)
			continue
		}
		obj := &unstructured.Unstructured{}
		if err := yaml.Unmarshal(data, &obj.Object); err != nil {
			log.Error(err, "failed to unmarshal embedded template; skipping", "file", filename)
			continue
		}
		obj.SetNamespace(b.namespace)
		if err := b.client.Create(ctx, obj); err != nil {
			if apierrors.IsAlreadyExists(err) {
				log.V(1).Info("default template already present; leaving as-is",
					"name", obj.GetName(), "namespace", b.namespace)
				continue
			}
			log.Error(err, "failed to create default template; skipping",
				"name", obj.GetName(), "namespace", b.namespace)
			continue
		}
		log.Info("default template installed",
			"name", obj.GetName(), "namespace", b.namespace)
	}
	return nil
}
