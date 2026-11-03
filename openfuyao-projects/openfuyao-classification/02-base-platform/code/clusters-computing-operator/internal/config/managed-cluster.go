package config

import (
	"os"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"strconv"
)

type ManagedClusterConfig struct {
	ClusterSize        int
	MinClusterSize     int
	NumManagementNodes int
}

var managedClusterConfig *ManagedClusterConfig

func LoadManagedClusterConfigFromEnv() {
	managedClusterConfig = &ManagedClusterConfig{
		ClusterSize:        getEnvInt("CLUSTER_SIZE", 5000),
		MinClusterSize:     getEnvInt("MIN_CLUSTER_SIZE", 100),
		NumManagementNodes: getEnvInt("NUM_MANAGEMENT_NODES", 3),
	}
}

func GetManagedClusterConfig() *ManagedClusterConfig {
	return managedClusterConfig
}

func getEnvInt(key string, defaultValue int) int {
	valueStr := os.Getenv(key)
	if valueStr == "" {
		return defaultValue
	}

	value, err := strconv.Atoi(valueStr)
	if err != nil {
		log.Log.Error(err, "env parse error", "key", key, "value", valueStr)
		return defaultValue
	}

	return value
}
