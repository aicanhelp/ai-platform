package monitoring_dashboard

import (
	"log"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func TestMonitoringDashboardSystemIntegration(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Monitoring-dashboard System Integration Suite")
}

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		log.Fatalf("Failed to load env file ../monitoring-dashboard-config/.env.monitoring-dashboard: %v", err)
	}
}
