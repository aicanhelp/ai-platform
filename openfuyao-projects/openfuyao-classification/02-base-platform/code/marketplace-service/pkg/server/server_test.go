package server

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"os"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	snapshotclient "github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"marketplace-service/cmd/config"
	helmv1 "marketplace-service/pkg/api/marketplace/v1beta1"
	"marketplace-service/pkg/client/k8s"
	pb "marketplace-service/pkg/rpc/helmchart"
	"marketplace-service/pkg/server/runtime"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/zlog"
)

type mockK8sClient struct {
	cfg *rest.Config
}

func (m *mockK8sClient) ApiExtensions() apiextensionsclient.Interface { return nil }
func (m *mockK8sClient) Config() *rest.Config                         { return m.cfg }
func (m *mockK8sClient) Kubernetes() kubernetes.Interface             { return nil }
func (m *mockK8sClient) Snapshot() snapshotclient.Interface           { return nil }

func TestInitServerInsecure(t *testing.T) {
	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			InsecurePort: 9037,
			SecurePort:   0,
		},
	}
	s, err := initServer(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, s)
	assert.Equal(t, ":9037", s.Addr)
	assert.Nil(t, s.TLSConfig)
}

func TestInitServerSecureLoadPairError(t *testing.T) {
	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			InsecurePort: 9037,
			SecurePort:   9443,
			CertFile:     "not-exists-cert",
			PrivateKey:   "not-exists-key",
			CAFile:       "not-exists-ca",
		},
	}
	s, err := initServer(cfg)
	assert.Error(t, err)
	assert.Nil(t, s)
}

func TestInitServerSecureSuccess(t *testing.T) {
	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			InsecurePort: 9037,
			SecurePort:   9443,
			CertFile:     "cert",
			PrivateKey:   "key",
			CAFile:       "ca",
		},
	}

	patches := gomonkey.ApplyFunc(tls.LoadX509KeyPair, func(_, _ string) (tls.Certificate, error) {
		return tls.Certificate{}, nil
	}).ApplyFunc(os.ReadFile, func(_ string) ([]byte, error) {
		return []byte("invalid-but-accepted"), nil
	})
	defer patches.Reset()

	s, err := initServer(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, s)
	assert.Equal(t, ":9443", s.Addr)
	assert.NotNil(t, s.TLSConfig)
}

func TestGetGrpcServer(t *testing.T) {
	patchNoTLS := gomonkey.ApplyFunc(httputil.IsHttpsEnabled, func(_, _, _ string) (bool, error) {
		return false, nil
	}).ApplyFunc(httputil.GetHttpConfig, func(_, _, _ string, _ bool) (*tls.Config, error) {
		return nil, errors.New("disabled")
	})
	grpcServer := getGrpcServer()
	assert.NotNil(t, grpcServer)
	patchNoTLS.Reset()

	patchTLS := gomonkey.ApplyFunc(httputil.IsHttpsEnabled, func(_, _, _ string) (bool, error) {
		return true, nil
	}).ApplyFunc(httputil.GetHttpConfig, func(_, _, _ string, _ bool) (*tls.Config, error) {
		return &tls.Config{}, nil
	})
	grpcServer = getGrpcServer()
	assert.NotNil(t, grpcServer)
	patchTLS.Reset()
}

func TestRegisterAPI(t *testing.T) {
	ws := restful.WebService{}
	ws.Path("/rest/v1")

	patches := gomonkey.ApplyFunc(runtime.GetMarketplaceWebService, func() *restful.WebService {
		return &ws
	}).ApplyFunc(helmv1.BindMarketPlaceRoute, func(_ *restful.WebService, _ *rest.Config) *helmv1.Handler {
		return &helmv1.Handler{}
	}).ApplyFunc(initRPCServer, func(_ *helmv1.Handler) {})
	defer patches.Reset()

	s := &CServer{
		container: restful.NewContainer(),
		KubernetesClient: &mockK8sClient{
			cfg: &rest.Config{},
		},
	}
	s.registerAPI()
	assert.Len(t, s.container.RegisteredWebServices(), 1)
}

func TestRun(t *testing.T) {
	errListen := errors.New("listen err")
	ws := restful.WebService{}
	ws.Path("/rest/v1")
	patches := gomonkey.ApplyFunc(runtime.GetMarketplaceWebService, func() *restful.WebService {
		return &ws
	}).ApplyFunc(helmv1.BindMarketPlaceRoute, func(_ *restful.WebService, _ *rest.Config) *helmv1.Handler {
		return &helmv1.Handler{}
	}).ApplyFunc(initRPCServer, func(_ *helmv1.Handler) {
	}).ApplyMethod(reflect.TypeOf(&http.Server{}), "ListenAndServe", func(_ *http.Server) error {
		return errListen
	}).ApplyMethod(reflect.TypeOf(&http.Server{}), "Shutdown", func(_ *http.Server, _ context.Context) error {
		return nil
	})
	defer patches.Reset()

	s := &CServer{
		Server:           &http.Server{},
		container:        restful.NewContainer(),
		KubernetesClient: &mockK8sClient{cfg: &rest.Config{}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := s.Run(ctx)
	assert.Equal(t, errListen, err)
}

func TestRunTLS(t *testing.T) {
	errListen := errors.New("tls listen err")
	ws := restful.WebService{}
	ws.Path("/rest/v1")
	patches := gomonkey.ApplyFunc(runtime.GetMarketplaceWebService, func() *restful.WebService {
		return &ws
	}).ApplyFunc(helmv1.BindMarketPlaceRoute, func(_ *restful.WebService, _ *rest.Config) *helmv1.Handler {
		return &helmv1.Handler{}
	}).ApplyFunc(initRPCServer, func(_ *helmv1.Handler) {
	}).ApplyMethod(reflect.TypeOf(&http.Server{}), "ListenAndServeTLS", func(_ *http.Server, _, _ string) error {
		return errListen
	})
	defer patches.Reset()

	s := &CServer{
		Server: &http.Server{
			TLSConfig: &tls.Config{},
		},
		container:        restful.NewContainer(),
		KubernetesClient: &mockK8sClient{cfg: &rest.Config{}},
	}
	err := s.Run(context.Background())
	assert.Equal(t, errListen, err)
}

func TestNewServer(t *testing.T) {
	httpServer := &http.Server{Addr: ":9037"}
	mockClient := &mockK8sClient{cfg: &rest.Config{}}

	patches := gomonkey.ApplyFunc(initServer, func(_ *config.RunConfig) (*http.Server, error) {
		return httpServer, nil
	}).ApplyFunc(k8s.NewKubernetesClient, func(_ *k8s.KubernetesCfg) (k8s.Client, error) {
		return mockClient, nil
	})
	defer patches.Reset()

	cfg := &config.RunConfig{
		Server:        &runtime.ServerConfig{InsecurePort: 9037},
		KubernetesCfg: &k8s.KubernetesCfg{KubeConfig: &rest.Config{}},
	}
	s, err := NewServer(cfg, context.Background())
	assert.NoError(t, err)
	assert.NotNil(t, s)
	assert.Equal(t, httpServer, s.Server)
	assert.Equal(t, mockClient, s.KubernetesClient)
}

func TestNewServerErrors(t *testing.T) {
	patchInitErr := gomonkey.ApplyFunc(initServer, func(_ *config.RunConfig) (*http.Server, error) {
		return nil, errors.New("init failed")
	})
	s, err := NewServer(&config.RunConfig{}, context.Background())
	assert.Error(t, err)
	assert.Nil(t, s)
	patchInitErr.Reset()

	patchK8sErr := gomonkey.ApplyFunc(initServer, func(_ *config.RunConfig) (*http.Server, error) {
		return &http.Server{}, nil
	}).ApplyFunc(k8s.NewKubernetesClient, func(_ *k8s.KubernetesCfg) (k8s.Client, error) {
		return nil, errors.New("k8s failed")
	})
	s, err = NewServer(&config.RunConfig{}, context.Background())
	assert.Error(t, err)
	assert.Nil(t, s)
	patchK8sErr.Reset()
}

func TestInitRPCServerListenError(t *testing.T) {
	patches := gomonkey.ApplyFunc(net.Listen, func(_, _ string) (net.Listener, error) {
		return nil, errors.New("listen failed")
	}).ApplyFunc(zlog.Fatalf, func(_ string, _ ...interface{}) {
		panic("fatal called")
	})
	defer patches.Reset()

	assert.Panics(t, func() { initRPCServer(&helmv1.Handler{}) })
}

func TestRegisterChartManagerServerReference(t *testing.T) {
	assert.NotNil(t, pb.ChartManager_ServiceDesc.Streams)
}
