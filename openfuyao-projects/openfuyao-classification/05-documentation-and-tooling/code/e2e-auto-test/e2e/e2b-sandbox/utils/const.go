package utils

import (
	"os"
)

const (
	// OpenSandbox Server
	ServerPodLabel  = "app=opensandbox-server"
	ServerPort      = 8080
	ServerNamespace = "flux-system"

	// SandboxGroup
	SGNamespace   = "flux-system"
	AgentPodLabel = "app=sandbox-agent-pod"

	// Test defaults
	DefaultImage   = "ubuntu-22-04-custom"
	DefaultCPU     = "10m"
	DefaultMemory  = "16Mi"
	DefaultTimeout = 3600

	// Sandbox states
	StateRunning = "Running"
	StatePaused  = "Paused"

	// Snapshot states
	SnapReady = "Ready"

	// HTTP status codes
	StatusOK                  = 200
	StatusCreated             = 201
	StatusAccepted            = 202
	StatusNoContent           = 204
	StatusBadRequest          = 400
	StatusNotFound            = 404
	StatusConflict            = 409
	StatusInternalServerError = 500
)

// EnvOr returns the value of env var key, or fallback if unset/empty.
func EnvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
