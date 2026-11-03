package console

import (
	"net/http"
	"os"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/auth"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

var (
	baseURL string
	client  *http.Client
)

var _ = BeforeSuite(func() {
	err := env.LoadEnv(".env")
	if err != nil {
		Fail("Error loading .env file: " + err.Error())
	}

	baseURL = auth.BaseURLFromEnv()
	if baseURL == "" {
		Fail("TEST_FUYAOURL or TEST_NODE2_IP is not set")
	}

	client, err = auth.NewInsecureClient()
	if err != nil {
		Fail("Error creating HTTP client: " + err.Error())
	}

	testConfig = &ConsoleTestConfig{
		BaseURL: baseURL,
		Client:  client,
	}

	Expect(auth.Login(client, baseURL, os.Getenv("TEST_USERNAME"), os.Getenv("TEST_PASSWORD"))).To(Succeed())
})

var _ = AfterSuite(func() {
	Expect(auth.Logout(client, baseURL)).To(Succeed())
})

func TestConsoleTest(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "ConsoleTest Suite")
}
