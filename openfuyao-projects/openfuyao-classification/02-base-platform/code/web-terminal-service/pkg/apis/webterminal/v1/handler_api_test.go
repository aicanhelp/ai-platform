package v1

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/emicklei/go-restful/v3"
	"github.com/gorilla/websocket"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	k8stesting "k8s.io/client-go/testing"
	"sigs.k8s.io/controller-runtime/pkg/client"
	faker "sigs.k8s.io/controller-runtime/pkg/client/fake"

	apiruntime "openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/runtime"
)

const (
	userHeaderKey = "X-User"
)

func newAPIFakeMgrClient() client.Client {
	scheme := runtime.NewScheme()
	return faker.NewClientBuilder().WithScheme(scheme).Build()
}

func newAPITestServer(clientset kubernetes.Interface, mgrclient client.Client) *httptest.Server {
	handler := NewHandler(clientset, &rest.Config{}, mgrclient)
	ws := apiruntime.NewWebService()
	sayHello(ws, handler)
	terminalPod(ws, handler)
	terminalCluster(ws, handler)

	container := restful.NewContainer()
	container.Router(restful.CurlyRouter{})
	container.Filter(func(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
		user := req.Request.Header.Get(userHeaderKey)
		if user != "" {
			ctx := context.WithValue(req.Request.Context(), "user", user)
			req.Request = req.Request.WithContext(ctx)
		}
		chain.ProcessFilter(req, resp)
	})
	container.Add(ws)

	return httptest.NewServer(container)
}

func wsURL(baseURL, path string) string {
	return "ws" + strings.TrimPrefix(baseURL, "http") + path
}

func dialAndRead(url string, header http.Header) (string, *websocket.Conn, error) {
	dialer := websocket.DefaultDialer
	conn, _, err := dialer.Dial(url, header)
	if err != nil {
		return "", nil, err
	}
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, msg, err := conn.ReadMessage()
	return string(msg), conn, err
}

var _ = Describe("WebTerminal API Test", func() {
	var (
		server     *httptest.Server
		kubeClient *fake.Clientset
		mgr        client.Client
		baseURL    string
	)

	BeforeEach(func() {
		kubeClient = fake.NewSimpleClientset()
		mgr = newAPIFakeMgrClient()
		server = newAPITestServer(kubeClient, mgr)
		baseURL = server.URL
	})

	AfterEach(func() {
		server.Close()
	})

	Describe("GET /rest/webterminal/v1/hello", func() {
		It("returns a success message", func() {
			resp, err := http.Get(baseURL + apiruntime.WebTerminalBasePath + "/hello")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(body)).To(ContainSubstring("WTS API Server Working Successfully"))
		})

		It("rejects unsupported methods", func() {
			req, err := http.NewRequest(http.MethodPost, baseURL+apiruntime.WebTerminalBasePath+"/hello", nil)
			Expect(err).NotTo(HaveOccurred())

			resp, err := http.DefaultClient.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()

			Expect(resp.StatusCode).To(Equal(http.StatusMethodNotAllowed))
		})

		It("returns 404 for unknown routes", func() {
			resp, err := http.Get(baseURL + apiruntime.WebTerminalBasePath + "/unknown")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()

			Expect(resp.StatusCode).To(Equal(http.StatusNotFound))
		})
	})

	Describe("GET /rest/webterminal/v1/namespace/{namespace}/pod/{pod}/container/{container}/terminal", func() {
		It("returns 400 when not a websocket request", func() {
			resp, err := http.Get(baseURL + apiruntime.WebTerminalBasePath + "/namespace/default/pod/test-pod/container/test-container/terminal")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()

			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		})

		It("returns an error when user is missing from context", func() {
			url := wsURL(baseURL, apiruntime.WebTerminalBasePath+"/namespace/default/pod/test-pod/container/test-container/terminal")
			message, conn, err := dialAndRead(url, nil)
			Expect(err).NotTo(HaveOccurred())
			defer conn.Close()
			Expect(message).To(ContainSubstring("retrieving user information"))
		})

		It("returns an error when listing role bindings fails", func() {
			kubeClient.PrependReactor("list", "clusterrolebindings", func(action k8stesting.Action) (bool, runtime.Object, error) {
				return true, nil, errors.New("list failed")
			})

			url := wsURL(baseURL, apiruntime.WebTerminalBasePath+"/namespace/default/pod/test-pod/container/test-container/terminal")
			header := http.Header{}
			header.Set(userHeaderKey, "test-user")

			message, conn, err := dialAndRead(url, header)
			Expect(err).NotTo(HaveOccurred())
			defer conn.Close()
			Expect(message).To(ContainSubstring("retrieving cluster role bindings"))
		})

		It("returns an error when user has no access", func() {
			url := wsURL(baseURL, apiruntime.WebTerminalBasePath+"/namespace/default/pod/test-pod/container/test-container/terminal")
			header := http.Header{}
			header.Set(userHeaderKey, "test-user")

			message, conn, err := dialAndRead(url, header)
			Expect(err).NotTo(HaveOccurred())
			defer conn.Close()
			Expect(message).To(ContainSubstring("User has no access"))
		})

	})

	Describe("GET /rest/webterminal/v1/user/{user}/terminal", func() {
		It("returns 400 when not a websocket request", func() {
			resp, err := http.Get(baseURL + apiruntime.WebTerminalBasePath + "/user/test-user/terminal")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()

			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		})

		It("returns an error when user is missing from context", func() {
			url := wsURL(baseURL, apiruntime.WebTerminalBasePath+"/user/test-user/terminal")
			message, conn, err := dialAndRead(url, nil)
			Expect(err).NotTo(HaveOccurred())
			defer conn.Close()
			Expect(message).To(ContainSubstring("retrieving user information"))
		})

		It("returns an error when listing role bindings fails", func() {
			kubeClient.PrependReactor("list", "clusterrolebindings", func(action k8stesting.Action) (bool, runtime.Object, error) {
				return true, nil, errors.New("list failed")
			})

			url := wsURL(baseURL, apiruntime.WebTerminalBasePath+"/user/test-user/terminal")
			header := http.Header{}
			header.Set(userHeaderKey, "test-user")

			message, conn, err := dialAndRead(url, header)
			Expect(err).NotTo(HaveOccurred())
			defer conn.Close()
			Expect(message).To(ContainSubstring("retrieving cluster role bindings"))
		})

		It("returns an error when user has no access", func() {
			url := wsURL(baseURL, apiruntime.WebTerminalBasePath+"/user/test-user/terminal")
			header := http.Header{}
			header.Set(userHeaderKey, "test-user")

			message, conn, err := dialAndRead(url, header)
			Expect(err).NotTo(HaveOccurred())
			defer conn.Close()
			Expect(message).To(ContainSubstring("User has no access"))
		})

	})
})
