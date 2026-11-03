package utils

// TeardownSuite runs AfterSuite cleanup: KServe bootstrap, then BeforeSuite helm releases.
// Safe to call from Ginkgo AfterSuite or SIGINT handler (idempotent KServe teardown).
func TeardownSuite(fitID string) {
	TeardownKServeBootstrap(fitID)
	TeardownBeforeSuitePrerequisites(fitID)
}
