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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"gopkg.openfuyao.cn/preflight/api"
)

type fakeManagementReader struct {
	versionCalls, discoveryCalls, secretCalls, configMapCalls, namespaceListCalls, bkeListCalls int
	err                                                                                         error
	discoveryErr                                                                                error
	discoveryOverride                                                                           []*metav1.APIResourceList
	secretOverride                                                                              map[string][]byte
}

func (f *fakeManagementReader) ConfigMap(context.Context, string, string) (map[string]string, error) {
	f.configMapCalls++
	return map[string]string{"host": "192.0.2.20", "imageRepoPort": "5443"}, f.err
}

func (f *fakeManagementReader) ServerVersion(context.Context) (string, error) {
	f.versionCalls++
	return "v1.28.0", f.err
}

func (f *fakeManagementReader) Discover(context.Context) ([]*metav1.APIResourceList, error) {
	f.discoveryCalls++
	if f.discoveryOverride != nil {
		return f.discoveryOverride, f.discoveryErr
	}
	return []*metav1.APIResourceList{{APIResources: []metav1.APIResource{{Name: "bkeclusters"}, {Name: "bkenodes"}}}}, f.discoveryErr
}

func TestManagementProviderDefaultsAreReadOnce(t *testing.T) {
	reader := &fakeManagementReader{}
	collector := &ManagementCollector{reader: reader}
	first, err := collector.ProviderDefaults(context.Background())
	require.NoError(t, err)
	second, err := collector.ProviderDefaults(context.Background())
	require.NoError(t, err)
	assert.Equal(t, first, second)
	assert.Equal(t, 1, reader.configMapCalls)
}
func (f *fakeManagementReader) Secret(context.Context, string, string) (map[string][]byte, error) {
	f.secretCalls++
	if f.secretOverride != nil {
		return f.secretOverride, f.err
	}
	return map[string][]byte{"config": []byte("apiVersion: v1\nkind: Config\nclusters:\n- name: local\n  cluster:\n    server: https://127.0.0.1:6443\ncontexts: []\nusers: []\n")}, f.err
}
func (f *fakeManagementReader) ListNamespaces(context.Context) error {
	f.namespaceListCalls++
	return f.err
}
func (f *fakeManagementReader) ListBKE(context.Context, string) error { f.bkeListCalls++; return f.err }

func TestManagementCollectorUsesReadOnlySurface(t *testing.T) {
	reader := &fakeManagementReader{}
	collector := &ManagementCollector{reader: reader}
	keys := []string{"kubernetes.api.reachable", "kubernetes.discovery.complete", "crd.bkeclusters", "object.exists.kube-system|localkubeconfig", "secret.localkubeconfig.dataNonEmpty", "kubeconfig.parseable", "kubeconfig.sourceEndpoint", "auth.get.namespaces", "auth.list.bkenodes", "auth.list.bkeclusters"}
	for _, key := range keys {
		facts, err := collector.Collect(context.Background(), api.FactRequest{Keys: []string{key}, Subjects: []api.Subject{{ID: "cluster"}}})
		require.NoError(t, err)
		require.Len(t, facts, 1)
		assert.True(t, facts[0].Available, key)
		assert.Equal(t, true, facts[0].Value, key)
	}
	assert.Equal(t, 1, reader.discoveryCalls)
	assert.Equal(t, 1, reader.versionCalls)
	assert.Equal(t, 1, reader.secretCalls)
	assert.Equal(t, 1, reader.namespaceListCalls)
	assert.Equal(t, 2, reader.bkeListCalls)
}

func TestManagementCollectorConvertsReadFailureToFactError(t *testing.T) {
	collector := &ManagementCollector{reader: &fakeManagementReader{err: errors.New("forbidden")}}
	facts, err := collector.Collect(context.Background(), api.FactRequest{Keys: []string{"auth.list.bkenodes"}, Subjects: []api.Subject{{ID: "cluster"}}})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
}

func TestManagementCollectorReportsMissingSecretWithLocator(t *testing.T) {
	reader := &fakeManagementReader{err: apierrors.NewNotFound(schema.GroupResource{Resource: "secrets"}, "localkubeconfig")}
	collector := &ManagementCollector{reader: reader}
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"auth.get.localkubeconfig"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.Equal(t, api.FactErrNotFound, facts[0].Error.Code)
	assert.Equal(t, "Secret kube-system/localkubeconfig was not found", facts[0].Error.Message)
	require.Len(t, facts[0].Evidence, 1)
	assert.Equal(t, "Secret kube-system/localkubeconfig", facts[0].Evidence[0].Locator)
}

func TestUnavailableManagementCollectorReturnsRootFailureInsteadOfUnsupported(t *testing.T) {
	collector := NewUnavailableManagementCollector("client setup failed")
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"kubernetes.api.reachable"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
	assert.NotEqual(t, api.FactErrUnsupported, facts[0].Error.Code)
}

type blockingManagementReader struct{ fakeManagementReader }

func (*blockingManagementReader) ServerVersion(ctx context.Context) (string, error) {
	<-ctx.Done()
	return "", ctx.Err()
}

func TestManagementDiscoveryHonorsFactContext(t *testing.T) {
	collector := &ManagementCollector{reader: &blockingManagementReader{}}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	started := time.Now()
	facts, err := collector.Collect(ctx, api.FactRequest{
		Keys: []string{"kubernetes.api.reachable"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)
	assert.Less(t, time.Since(started), time.Second)
}

func TestManagementAPIReachabilityDoesNotInheritPartialDiscoveryFailure(t *testing.T) {
	reader := &fakeManagementReader{discoveryErr: errors.New("partial discovery failure")}
	collector := &ManagementCollector{reader: reader}
	apiFacts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"kubernetes.api.reachable"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, apiFacts, 1)
	assert.True(t, apiFacts[0].Available)
	assert.Equal(t, true, apiFacts[0].Value)

	discoveryFacts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"kubernetes.discovery.complete"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, discoveryFacts, 1)
	assert.False(t, discoveryFacts[0].Available)
}

// TestManagementCollectorRejectsMissingSubjectAndReader freezes the request
// contract guards.
func TestManagementCollectorRejectsMissingSubjectAndReader(t *testing.T) {
	collector := &ManagementCollector{}
	facts, err := collector.Collect(context.Background(), api.FactRequest{Keys: []string{"kubernetes.api.reachable"}})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)

	collector = &ManagementCollector{reader: &fakeManagementReader{}}
	facts, err = collector.Collect(context.Background(), api.FactRequest{Keys: []string{"kubernetes.api.reachable"}, Subjects: nil})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.Equal(t, api.FactErrParseError, facts[0].Error.Code)
}

// TestManagementCollectorRejectsUnknownKey freezes the unsupported-fact path.
func TestManagementCollectorRejectsUnknownKey(t *testing.T) {
	collector := &ManagementCollector{reader: &fakeManagementReader{}}
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"not.a.management.fact"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
	assert.Equal(t, "management-readonly", collector.ID())
}

// TestManagementCollectorCRDMissing freezes the negative CRD discovery path.
func TestManagementCollectorCRDMissing(t *testing.T) {
	reader := &fakeManagementReader{}
	reader.discoveryOverride = []*metav1.APIResourceList{{APIResources: []metav1.APIResource{{Name: "bkeclusters"}}}}
	collector := &ManagementCollector{reader: reader}
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"crd.bkenodes"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	assert.Equal(t, false, facts[0].Value)
}

// TestManagementCollectorSecretDataEmpty freezes the empty-data branch.
func TestManagementCollectorSecretDataEmpty(t *testing.T) {
	reader := &fakeManagementReader{}
	reader.secretOverride = map[string][]byte{"config": nil}
	collector := &ManagementCollector{reader: reader}
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"secret.localkubeconfig.dataNonEmpty"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	assert.Equal(t, false, facts[0].Value)
}

// TestManagementCollectorKubeconfigUnparseable freezes the parse-failure
// branches for both parseable and sourceEndpoint facts.
func TestManagementCollectorKubeconfigUnparseable(t *testing.T) {
	reader := &fakeManagementReader{}
	reader.secretOverride = map[string][]byte{"config": []byte("not-a-kubeconfig")}
	collector := &ManagementCollector{reader: reader}
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"kubeconfig.parseable"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.True(t, facts[0].Available)
	assert.Equal(t, false, facts[0].Value)

	facts, err = collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"kubeconfig.sourceEndpoint"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
}

// TestManagementCollectorForbiddenAndTimeoutClassification freezes the
// error-code mapping for RBAC and deadline failures.
func TestManagementCollectorForbiddenAndTimeoutClassification(t *testing.T) {
	forbidden := &fakeManagementReader{}
	forbidden.err = apierrors.NewForbidden(schema.GroupResource{Resource: "secrets"}, "localkubeconfig", errors.New("denied"))
	collector := &ManagementCollector{reader: forbidden}
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{"auth.get.localkubeconfig"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	assert.Equal(t, api.FactErrPermissionDenied, facts[0].Error.Code)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	timedOut := &fakeManagementReader{}
	timedOut.err = context.DeadlineExceeded
	collector = &ManagementCollector{reader: timedOut}
	facts, err = collector.Collect(ctx, api.FactRequest{
		Keys: []string{"kubernetes.api.reachable"}, Subjects: []api.Subject{{ID: "cluster"}},
	})
	require.NoError(t, err)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)
}

// TestManagementLocatorTable freezes the locator shown in evidence.
func TestManagementLocatorTable(t *testing.T) {
	tests := map[string]string{
		"kubernetes.api.reachable":              "management API /version",
		"kubernetes.discovery.complete":         "management API discovery",
		"crd.bkeclusters":                       "CRD bkeclusters.bke.bocloud.com",
		"crd.bkenodes":                          "CRD bkenodes.bke.bocloud.com",
		"auth.get.localkubeconfig":              "Secret kube-system/localkubeconfig",
		"auth.get.namespaces":                   "namespaces",
		"auth.list.bkenodes":                    "bkenodes.bke.bocloud.com",
		"auth.list.bkeclusters":                 "bkeclusters.bke.bocloud.com",
		"anything.else":                         "management API resource",
	}
	for key, locator := range tests {
		assert.Equal(t, locator, managementLocator(key), key)
	}
}

func TestManagementProviderDefaultsNilReaderAndNotFoundFallback(t *testing.T) {
// Nil reader surfaces an actionable error instead of a silent empty map.
var nilCollector *ManagementCollector
_, err := nilCollector.ProviderDefaults(context.Background())
assert.ErrorContains(t, err, "not configured")

empty := &ManagementCollector{}
_, err = empty.ProviderDefaults(context.Background())
assert.ErrorContains(t, err, "not configured")

// ConfigMap NotFound is equivalent to built-in defaults: no error.
notFound := &fakeManagementReader{}
notFound.err = apierrors.NewNotFound(schema.GroupResource{Resource: "configmaps"}, "bke-config")
collector := &ManagementCollector{reader: notFound}
defaults, err := collector.ProviderDefaults(context.Background())
require.NoError(t, err)
assert.Empty(t, defaults)

// The unavailable collector keeps a default message when blank.
collector2 := NewUnavailableManagementCollector("   ")
facts, err := collector2.Collect(context.Background(), api.FactRequest{
Keys: []string{"kubernetes.api.reachable"}, Subjects: []api.Subject{{ID: "cluster"}},
})
require.NoError(t, err)
require.Len(t, facts, 1)
assert.Equal(t, "management-unavailable", collector2.ID())
assert.NotEmpty(t, facts[0].Error.Message)
}
