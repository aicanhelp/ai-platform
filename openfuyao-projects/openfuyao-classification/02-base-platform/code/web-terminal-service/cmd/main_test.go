package main

import (
	"context"
	"errors"
	"flag"
	"io"
	"net/http"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	faker "sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	"openfuyao.com/web-terminal-service/internal/controller"
	v1 "openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1"
	v1config "openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/config"
)

type mockTerminalManager struct {
	client    client.Client
	scheme    *runtime.Scheme
	healthErr error
	readyErr  error
	startErr  error
}

func (f *mockTerminalManager) GetHTTPClient() *http.Client {
	return &http.Client{}
}

func (f *mockTerminalManager) GetConfig() *rest.Config {
	return &rest.Config{}
}

func (f *mockTerminalManager) GetCache() cache.Cache {
	return nil
}

func (f *mockTerminalManager) GetScheme() *runtime.Scheme {
	return f.scheme
}

func (f *mockTerminalManager) GetClient() client.Client {
	return f.client
}

func (f *mockTerminalManager) GetFieldIndexer() client.FieldIndexer {
	return nil
}

func (f *mockTerminalManager) GetEventRecorderFor(name string) record.EventRecorder {
	return nil
}

func (f *mockTerminalManager) GetRESTMapper() meta.RESTMapper {
	return nil
}

func (f *mockTerminalManager) GetAPIReader() client.Reader {
	return nil
}

func (f *mockTerminalManager) Start(ctx context.Context) error {
	return f.startErr
}

func (f *mockTerminalManager) Add(runnable manager.Runnable) error {
	return nil
}

func (f *mockTerminalManager) Elected() <-chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func (f *mockTerminalManager) AddHealthzCheck(name string, check healthz.Checker) error {
	return f.healthErr
}

func (f *mockTerminalManager) AddReadyzCheck(name string, check healthz.Checker) error {
	return f.readyErr
}

func (f *mockTerminalManager) GetWebhookServer() webhook.Server {
	return nil
}

func (f *mockTerminalManager) GetLogger() logr.Logger {
	return logr.Discard()
}

func (f *mockTerminalManager) GetControllerOptions() config.Controller {
	return config.Controller{}
}

type exitPanic struct {
	code int
}

func resetFlags() {
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	flag.CommandLine.SetOutput(io.Discard)
	os.Args = []string{"cmd"}
}

func patchCommon(t *testing.T) (*gomonkey.Patches, *mockTerminalManager) {
	t.Helper()
	mgr := &mockTerminalManager{
		client: faker.NewClientBuilder().Build(),
		scheme: runtime.NewScheme(),
	}
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(ctrl.GetConfigOrDie, func() *rest.Config {
		return &rest.Config{}
	})
	patches.ApplyFunc(ctrl.NewManager, func(*rest.Config, ctrl.Options) (ctrl.Manager, error) {
		return mgr, nil
	})
	patches.ApplyFunc(ctrl.SetupSignalHandler, func() context.Context {
		return context.Background()
	})
	patches.ApplyMethod(reflect.TypeOf(&v1config.RunConfig{}), "Validate",
		func(*v1config.RunConfig) []error { return nil })
	patches.ApplyFunc(v1.StartAPIServer, func(client.Client) {})
	patches.ApplyFunc(v1.NewServer, func(*v1config.RunConfig, context.Context, client.Client) (*v1.APIServer, error) {
		return &v1.APIServer{Server: &http.Server{}}, nil
	})
	patches.ApplyMethod(reflect.TypeOf(&v1.APIServer{}), "Run",
		func(*v1.APIServer, context.Context) error { return nil })
	patches.ApplyMethod(reflect.TypeOf(&controller.WebterminalTemplateReconciler{}), "SetupWithManager",
		func(_ *controller.WebterminalTemplateReconciler, _ ctrl.Manager) error { return nil })
	t.Cleanup(patches.Reset)
	return patches, mgr
}

func TestMain_NewManagerError(t *testing.T) {
	resetFlags()
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(ctrl.GetConfigOrDie, func() *rest.Config { return &rest.Config{} })
	patches.ApplyFunc(ctrl.NewManager, func(*rest.Config, ctrl.Options) (ctrl.Manager, error) {
		return nil, errors.New("boom")
	})
	patches.ApplyFunc(os.Exit, func(code int) { panic(exitPanic{code: code}) })
	defer patches.Reset()

	assert.PanicsWithValue(t, exitPanic{code: 1}, func() { main() })
}

func TestMain_SetupWithManagerError(t *testing.T) {
	resetFlags()
	patches, _ := patchCommon(t)
	patches.ApplyMethod(reflect.TypeOf(&controller.WebterminalTemplateReconciler{}), "SetupWithManager",
		func(_ *controller.WebterminalTemplateReconciler, _ ctrl.Manager) error { return errors.New("setup") })
	patches.ApplyFunc(os.Exit, func(code int) { panic(exitPanic{code: code}) })

	assert.PanicsWithValue(t, exitPanic{code: 1}, func() { main() })
}

func TestMain_AddHealthzCheckError(t *testing.T) {
	resetFlags()
	patches, mgr := patchCommon(t)
	mgr.healthErr = errors.New("health")
	patches.ApplyFunc(os.Exit, func(code int) { panic(exitPanic{code: code}) })

	assert.PanicsWithValue(t, exitPanic{code: 1}, func() { main() })
}

func TestMain_AddReadyzCheckError(t *testing.T) {
	resetFlags()
	patches, mgr := patchCommon(t)
	mgr.readyErr = errors.New("ready")
	patches.ApplyFunc(os.Exit, func(code int) { panic(exitPanic{code: code}) })

	assert.PanicsWithValue(t, exitPanic{code: 1}, func() { main() })
}

func TestMain_StartError(t *testing.T) {
	resetFlags()
	patches, mgr := patchCommon(t)
	mgr.startErr = errors.New("start")
	patches.ApplyFunc(os.Exit, func(code int) { panic(exitPanic{code: code}) })

	assert.PanicsWithValue(t, exitPanic{code: 1}, func() { main() })
}

func TestMain_Success(t *testing.T) {
	resetFlags()
	patches, _ := patchCommon(t)
	startCalled := make(chan struct{}, 1)
	patches.ApplyFunc(v1.StartAPIServer, func(client.Client) {
		startCalled <- struct{}{}
	})
	patches.ApplyFunc(v1.NewServer, func(*v1config.RunConfig, context.Context, client.Client) (*v1.APIServer, error) {
		startCalled <- struct{}{}
		return &v1.APIServer{Server: &http.Server{}}, nil
	})
	patches.ApplyFunc(os.Exit, func(code int) { panic(exitPanic{code: code}) })

	assert.NotPanics(t, func() { main() })
	select {
	case <-startCalled:
	case <-time.After(time.Second):
		t.Fatal("expected API server startup path to be called")
	}
}
