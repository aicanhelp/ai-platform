/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package preflightbridge

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"gopkg.openfuyao.cn/preflight/api"
)

const defaultManagementTimeout = 30 * time.Second

type managementReader interface {
	ServerVersion(context.Context) (string, error)
	Discover(context.Context) ([]*metav1.APIResourceList, error)
	Secret(context.Context, string, string) (map[string][]byte, error)
	ConfigMap(context.Context, string, string) (map[string]string, error)
	ListNamespaces(context.Context) error
	ListBKE(context.Context, string) error
}

// ManagementCollector performs cached, read-only management-cluster probes.
type ManagementCollector struct {
	reader           managementReader
	versionOnce      sync.Once
	serverVersion    string
	versionErr       error
	discoveryOnce    sync.Once
	resources        []*metav1.APIResourceList
	discoveryErr     error
	secretOnce       sync.Once
	localKubeconfig  map[string][]byte
	secretErr        error
	defaultsOnce     sync.Once
	providerDefaults map[string]string
	defaultsErr      error
}

// NewManagementCollector creates GET/LIST/Discovery-only management-cluster
// probes. The returned collector has no Kubernetes mutation method.
func NewManagementCollector(kubeconfig string) (*ManagementCollector, error) {
	return NewManagementCollectorWithTimeout(kubeconfig, defaultManagementTimeout)
}

// NewManagementCollectorWithTimeout bounds client-go calls that do not expose
// a context-aware discovery method. Per-fact contexts may be shorter and are
// still honored by clientGoManagementReader.Discover.
func NewManagementCollectorWithTimeout(kubeconfig string, timeout time.Duration) (*ManagementCollector, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	if kubeconfig != "" {
		loadingRules.ExplicitPath = kubeconfig
	}
	config, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{}).
		ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("load management kubeconfig: %w", err)
	}
	if timeout > 0 {
		config.Timeout = timeout
	}
	typed, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, err
	}
	return &ManagementCollector{
		reader: &clientGoManagementReader{typed: typed, dynamic: dynamicClient, discovery: discoveryClient},
	}, nil
}

// ProviderDefaults reads the same management-cluster ConfigMap used by the
// Provider webhook. It is cached for the run and returns a copy. NotFound is
// equivalent to the webhook's built-in-default fallback; other read failures
// prevent an exact dynamic plan and are returned to the caller.
func (c *ManagementCollector) ProviderDefaults(ctx context.Context) (map[string]string, error) {
	if c == nil || c.reader == nil {
		return nil, fmt.Errorf("management reader not configured")
	}
	c.defaultsOnce.Do(func() {
		c.providerDefaults, c.defaultsErr = c.reader.ConfigMap(ctx, "cluster-system", "bke-config")
		if apierrors.IsNotFound(c.defaultsErr) {
			c.providerDefaults, c.defaultsErr = nil, nil
		}
	})
	result := make(map[string]string, len(c.providerDefaults))
	for key, value := range c.providerDefaults {
		result[key] = value
	}
	return result, c.defaultsErr
}

// ID returns the collector identifier.
func (c *ManagementCollector) ID() string { return "management-readonly" }

// Collect reads the requested management-cluster facts without mutation.
func (c *ManagementCollector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	if c.reader == nil {
		return unavailableManagement(req, api.FactErrUnsupported, "management reader not configured"), nil
	}
	if len(req.Subjects) == 0 {
		return unavailableManagement(req, api.FactErrParseError, "management subject is required"), nil
	}
	out := make([]api.Fact, 0, len(req.Keys))
	for _, key := range req.Keys {
		out = append(out, c.collect(ctx, req.Subjects[0].ID, key))
	}
	return out, nil
}

func (c *ManagementCollector) collect(ctx context.Context, subject, key string) api.Fact {
	locator := managementLocator(key)
	switch key {
	case "kubernetes.api.reachable":
		version, err := c.server(ctx)
		if err != nil {
			return c.failedFact(ctx, subject, key, locator, err)
		}
		return c.availableFact(subject, key, locator, version != "")
	case "kubernetes.discovery.complete":
		resources, err := c.discovery(ctx)
		if err != nil {
			return c.failedFact(ctx, subject, key, locator, err)
		}
		return c.availableFact(subject, key, locator, len(resources) > 0)
	case "crd.bkeclusters", "crd.bkenodes":
		return c.collectCRD(ctx, subject, key, locator)
	case "object.exists.kube-system|localkubeconfig",
		"auth.get.localkubeconfig",
		"secret.localkubeconfig.dataNonEmpty",
		"kubeconfig.parseable",
		"kubeconfig.sourceEndpoint":
		return c.collectSecret(ctx, subject, key, locator)
	case "auth.get.namespaces":
		if err := c.reader.ListNamespaces(ctx); err != nil {
			return c.failedFact(ctx, subject, key, locator, err)
		}
		return c.availableFact(subject, key, locator, true)
	case "auth.list.bkenodes":
		if err := c.reader.ListBKE(ctx, "bkenodes"); err != nil {
			return c.failedFact(ctx, subject, key, locator, err)
		}
		return c.availableFact(subject, key, locator, true)
	case "auth.list.bkeclusters":
		if err := c.reader.ListBKE(ctx, "bkeclusters"); err != nil {
			return c.failedFact(ctx, subject, key, locator, err)
		}
		return c.availableFact(subject, key, locator, true)
	default:
		return api.Fact{
			Key:       key,
			Subject:   subject,
			Available: false,
			Error:     &api.FactError{Code: api.FactErrUnsupported, Message: "unsupported management fact"},
		}
	}
}

func (c *ManagementCollector) availableFact(subject, key, locator string, value any) api.Fact {
	return api.Fact{Key: key, Subject: subject, Available: true, Value: value,
		Evidence: []api.Evidence{{Kind: "api", Source: c.ID(), Locator: locator}}}
}

func (c *ManagementCollector) failedFact(ctx context.Context, subject, key, locator string, err error) api.Fact {
	var code, message string
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded):
		code, message = api.FactErrTimeout, "management API read timed out for "+locator
	case apierrors.IsNotFound(err):
		code, message = api.FactErrNotFound, locator+" was not found"
	case apierrors.IsForbidden(err) || apierrors.IsUnauthorized(err):
		code, message = api.FactErrPermissionDenied, "permission denied reading "+locator
	default:
		code, message = api.FactErrUnreachable, "management API read failed for "+locator
	}
	return api.Fact{Key: key, Subject: subject, Available: false,
		Evidence: []api.Evidence{{Kind: "api", Source: c.ID(), Locator: locator}},
		Error:    &api.FactError{Code: code, Message: message}}
}

func (c *ManagementCollector) collectCRD(ctx context.Context, subject, key, locator string) api.Fact {
	resources, err := c.discovery(ctx)
	if err != nil {
		return c.failedFact(ctx, subject, key, locator, err)
	}
	want := strings.TrimPrefix(key, "crd.")
	for _, list := range resources {
		for _, resource := range list.APIResources {
			if resource.Name == want {
				return c.availableFact(subject, key, locator, true)
			}
		}
	}
	return c.availableFact(subject, key, locator, false)
}

func (c *ManagementCollector) collectSecret(ctx context.Context, subject, key, locator string) api.Fact {
	data, err := c.secret(ctx)
	if err != nil {
		return c.failedFact(ctx, subject, key, locator, err)
	}
	if key == "object.exists.kube-system|localkubeconfig" || key == "auth.get.localkubeconfig" {
		return c.availableFact(subject, key, locator, true)
	}
	content := data["config"]
	if key == "secret.localkubeconfig.dataNonEmpty" {
		return c.availableFact(subject, key, locator, len(content) > 0)
	}
	config, err := clientcmd.Load(content)
	if key == "kubeconfig.parseable" {
		return c.availableFact(subject, key, locator, err == nil)
	}
	if err != nil {
		return c.failedFact(ctx, subject, key, locator, fmt.Errorf("local kubeconfig is not parseable"))
	}
	return c.availableFact(subject, key, locator, len(config.Clusters) > 0)
}

func managementLocator(key string) string {
	switch key {
	case "kubernetes.api.reachable":
		return "management API /version"
	case "kubernetes.discovery.complete":
		return "management API discovery"
	case "crd.bkeclusters":
		return "CRD bkeclusters.bke.bocloud.com"
	case "crd.bkenodes":
		return "CRD bkenodes.bke.bocloud.com"
	case "object.exists.kube-system|localkubeconfig", "auth.get.localkubeconfig",
		"secret.localkubeconfig.dataNonEmpty", "kubeconfig.parseable", "kubeconfig.sourceEndpoint":
		return "Secret kube-system/localkubeconfig"
	case "auth.get.namespaces":
		return "namespaces"
	case "auth.list.bkenodes":
		return "bkenodes.bke.bocloud.com"
	case "auth.list.bkeclusters":
		return "bkeclusters.bke.bocloud.com"
	default:
		return "management API resource"
	}
}

func (c *ManagementCollector) server(ctx context.Context) (string, error) {
	c.versionOnce.Do(func() {
		c.serverVersion, c.versionErr = c.reader.ServerVersion(ctx)
	})
	return c.serverVersion, c.versionErr
}

func (c *ManagementCollector) discovery(ctx context.Context) ([]*metav1.APIResourceList, error) {
	c.discoveryOnce.Do(func() {
		c.resources, c.discoveryErr = c.reader.Discover(ctx)
	})
	return c.resources, c.discoveryErr
}

func (c *ManagementCollector) secret(ctx context.Context) (map[string][]byte, error) {
	c.secretOnce.Do(func() {
		c.localKubeconfig, c.secretErr = c.reader.Secret(ctx, "kube-system", "localkubeconfig")
	})
	return c.localKubeconfig, c.secretErr
}

type clientGoManagementReader struct {
	typed     kubernetes.Interface
	dynamic   dynamic.Interface
	discovery discovery.DiscoveryInterface
}

func (r *clientGoManagementReader) ServerVersion(ctx context.Context) (string, error) {
	type versionResult struct {
		version string
		err     error
	}
	result := make(chan versionResult, 1)
	go func() {
		version, err := r.discovery.ServerVersion()
		if err != nil {
			result <- versionResult{err: err}
			return
		}
		result <- versionResult{version: version.GitVersion}
	}()
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case resolved := <-result:
		return resolved.version, resolved.err
	}
}

func (r *clientGoManagementReader) Discover(ctx context.Context) ([]*metav1.APIResourceList, error) {
	type discoveryResult struct {
		resources []*metav1.APIResourceList
		err       error
	}
	result := make(chan discoveryResult, 1)
	go func() {
		resources, err := r.discovery.ServerPreferredResources()
		result <- discoveryResult{resources: resources, err: err}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case discovered := <-result:
		return discovered.resources, discovered.err
	}
}
func (r *clientGoManagementReader) Secret(ctx context.Context, ns, name string) (map[string][]byte, error) {
	secret, err := r.typed.CoreV1().Secrets(ns).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return secret.Data, nil
}
func (r *clientGoManagementReader) ConfigMap(ctx context.Context, ns, name string) (map[string]string, error) {
	configMap, err := r.typed.CoreV1().ConfigMaps(ns).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return configMap.Data, nil
}
func (r *clientGoManagementReader) ListNamespaces(ctx context.Context) error {
	_, err := r.typed.CoreV1().Namespaces().List(ctx, metav1.ListOptions{Limit: 1})
	return err
}
func (r *clientGoManagementReader) ListBKE(ctx context.Context, resource string) error {
	groupVersionResource := schema.GroupVersionResource{
		Group: "bke.bocloud.com", Version: "v1beta1", Resource: resource,
	}
	_, err := r.dynamic.Resource(groupVersionResource).
		List(ctx, metav1.ListOptions{Limit: 1})
	return err
}

func unavailableManagement(req api.FactRequest, code, message string) []api.Fact {
	out := make([]api.Fact, 0, len(req.Keys))
	subject := ""
	if len(req.Subjects) > 0 {
		subject = req.Subjects[0].ID
	}
	for _, key := range req.Keys {
		out = append(
			out,
			api.Fact{Key: key, Subject: subject, Available: false, Error: &api.FactError{Code: code, Message: message}},
		)
	}
	return out
}

var _ api.Collector = (*ManagementCollector)(nil)

type unavailableManagementCollector struct{ message string }

func (c unavailableManagementCollector) ID() string { return "management-unavailable" }
func (c unavailableManagementCollector) Collect(_ context.Context, req api.FactRequest) ([]api.Fact, error) {
	return unavailableManagement(req, api.FactErrUnreachable, c.message), nil
}

// NewUnavailableManagementCollector prevents management facts from falling
// through to the local collector when Kubernetes client setup failed.
func NewUnavailableManagementCollector(message string) api.Collector {
	if strings.TrimSpace(message) == "" {
		message = "management client setup failed"
	}
	return unavailableManagementCollector{message: message}
}
