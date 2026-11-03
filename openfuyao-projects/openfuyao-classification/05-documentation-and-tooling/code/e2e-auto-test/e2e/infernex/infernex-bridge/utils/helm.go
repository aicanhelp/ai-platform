package utils

import "fmt"

// helmLocalCmd prefixes KUBECONFIG for cluster-scoped helm/kubectl from the test runner.
func helmLocalCmd(helmArgs string) string {
	return fmt.Sprintf("KUBECONFIG=%q %s", localKubeconfig(), helmArgs)
}

// EnsureHelmRepos adds chart repositories used by BeforeSuite prerequisites.
func EnsureHelmRepos(fitID string) error {
	for _, repo := range []struct{ name, url string }{
		{"nats", "https://nats-io.github.io/k8s/helm/charts/"},
		{"prometheus-community", "https://prometheus-community.github.io/helm-charts"},
	} {
		cmd := helmLocalCmd(fmt.Sprintf("helm repo add %s %s 2>/dev/null || true", repo.name, repo.url))
		stdout, stderr, err := ExecLocal(cmd)
		LogStep(fitID, "helm repo add %s: stdout=%s stderr=%s err=%v", repo.name, trimForLog(stdout), trimForLog(stderr), err)
	}
	cmd := helmLocalCmd("helm repo update")
	stdout, stderr, err := ExecLocal(cmd)
	LogStep(fitID, "helm repo update: stdout=%s stderr=%s err=%v", trimForLog(stdout), trimForLog(stderr), err)
	return err
}

func runHelmUpgradeInstall(fitID, description, helmArgs string) error {
	cmd := helmLocalCmd("helm " + helmArgs)
	LogStep(fitID, "%s: %s", description, cmd)
	stdout, stderr, err := ExecLocal(cmd)
	LogStep(fitID, "helm stdout: %s", trimForLog(stdout))
	if stderr != "" {
		LogStep(fitID, "helm stderr: %s", trimForLog(stderr))
	}
	return err
}

func runHelmUninstall(fitID, description, release, namespace string) error {
	args := fmt.Sprintf("uninstall %s --namespace %s", release, namespace)
	cmd := helmLocalCmd("helm " + args)
	LogStep(fitID, "%s: %s", description, cmd)
	stdout, stderr, err := ExecLocal(cmd)
	LogStep(fitID, "helm stdout: %s", trimForLog(stdout))
	if stderr != "" {
		LogStep(fitID, "helm stderr: %s", trimForLog(stderr))
	}
	return err
}

func trimForLog(s string) string {
	const max = 4000
	if len(s) <= max {
		return s
	}
	return s[:max] + "...(truncated)"
}
