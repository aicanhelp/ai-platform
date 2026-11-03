package utils

// OptionalStackInstaller installs cluster components beyond Bridge/NATS/Prometheus.
// Replace KServeStackInstaller or GatewayStackInstaller before tests run when you have concrete helm steps.
type OptionalStackInstaller func(fitID string) error

var (
	// KServeStackInstaller clones KServe and runs llmisvc-full-install-helm.sh when SKIP_KSERVE_INSTALL is false.
	KServeStackInstaller OptionalStackInstaller = bootstrapKServeStack
	// GatewayStackInstaller installs gateway deps only when KServe install is skipped.
	GatewayStackInstaller OptionalStackInstaller = bootstrapGatewayStack
)

// InstallOptionalStackPrerequisites runs KServe/Gateway installers when not skipped.
func InstallOptionalStackPrerequisites(fitID string) error {
	if !SkipKServeInstall() {
		if err := KServeStackInstaller(fitID); err != nil {
			return err
		}
	} else {
		LogStep(fitID, "SKIP_KSERVE_INSTALL=true — skip KServe prerequisite install")
	}
	if !SkipGatewayStackInstall() {
		if err := GatewayStackInstaller(fitID); err != nil {
			return err
		}
	} else {
		LogStep(fitID, "SKIP_GATEWAY_STACK_INSTALL=true — skip Gateway API prerequisite install")
	}
	return nil
}
