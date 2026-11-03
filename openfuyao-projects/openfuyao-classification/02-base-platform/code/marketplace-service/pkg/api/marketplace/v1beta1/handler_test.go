package v1beta1

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"marketplace-service/pkg/helm"
)

func TestHelloHandler(t *testing.T) {
	container := restful.NewContainer()
	ws := restful.WebService{}
	ws.Path("/")
	handler := newHandler(&mockOperation{})
	bindHelmRepoGetRoutes(&ws, handler)
	container.Add(&ws)

	req, err := http.NewRequest("GET", "/hello/world", nil)
	assert.NoError(t, err)

	rr := httptest.NewRecorder()
	container.ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNotFound, rr.Code)
}

func TestNewHandler(t *testing.T) {
	var op helm.Operation
	handler := newHandler(op)
	assert.NotNil(t, handler)
	assert.Nil(t, handler.HelmHandler)
}

