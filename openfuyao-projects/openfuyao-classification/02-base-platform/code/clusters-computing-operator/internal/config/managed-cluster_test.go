package config_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"openfuyao.com/clusters-computing-operator/internal/config"
	"os"
)

var _ = Describe("Config", func() {
	Describe("LoadManagedClusterConfigFromEnv", func() {
		BeforeEach(func() {
			os.Unsetenv("CLUSTER_SIZE")
			os.Unsetenv("MIN_CLUSTER_SIZE")
			os.Unsetenv("NUM_MANAGEMENT_NODES")
		})

		It("returns default values when environment variables are not set", func() {
			config.LoadManagedClusterConfigFromEnv()

			cfg := config.GetManagedClusterConfig()
			Expect(cfg).NotTo(BeNil())
			Expect(cfg.ClusterSize).To(Equal(5000))
			Expect(cfg.MinClusterSize).To(Equal(100))
			Expect(cfg.NumManagementNodes).To(Equal(3))
		})

		It("uses values from environment variables when set", func() {
			os.Setenv("CLUSTER_SIZE", "3")
			os.Setenv("MIN_CLUSTER_SIZE", "1")
			os.Setenv("NUM_MANAGEMENT_NODES", "1")

			config.LoadManagedClusterConfigFromEnv()

			cfg := config.GetManagedClusterConfig()
			Expect(cfg.ClusterSize).To(Equal(3))
			Expect(cfg.MinClusterSize).To(Equal(1))
			Expect(cfg.NumManagementNodes).To(Equal(1))
		})

		It("uses default values for invalid environment variables", func() {
			os.Setenv("CLUSTER_SIZE", "invalid")
			os.Setenv("MIN_CLUSTER_SIZE", "not-a-number")
			os.Setenv("NUM_MANAGEMENT_NODES", "xyz")

			config.LoadManagedClusterConfigFromEnv()

			cfg := config.GetManagedClusterConfig()
			Expect(cfg.ClusterSize).To(Equal(5000))
			Expect(cfg.MinClusterSize).To(Equal(100))
			Expect(cfg.NumManagementNodes).To(Equal(3))
		})

		It("handles a mix of valid and invalid environment variables", func() {
			os.Setenv("CLUSTER_SIZE", "3000")
			os.Setenv("MIN_CLUSTER_SIZE", "invalid")
			os.Setenv("NUM_MANAGEMENT_NODES", "7")

			config.LoadManagedClusterConfigFromEnv()

			cfg := config.GetManagedClusterConfig()
			Expect(cfg.ClusterSize).To(Equal(3000))
			Expect(cfg.MinClusterSize).To(Equal(100))
			Expect(cfg.NumManagementNodes).To(Equal(7))
		})
	})
})
