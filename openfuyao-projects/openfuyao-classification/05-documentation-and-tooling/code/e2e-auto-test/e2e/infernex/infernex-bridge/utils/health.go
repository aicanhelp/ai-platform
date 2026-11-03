package utils

import (
	"fmt"
	"strings"

	. "github.com/onsi/gomega"
)

func CheckDeploymentLogHealth(fitID, ns, deploy string, mustContain []string, mustNotContain []string) {
	LogPhaseBegin(fitID, fmt.Sprintf("log health deploy/%s", deploy))
	defer LogPhaseEnd(fitID, fmt.Sprintf("log health deploy/%s", deploy))
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl logs -n %q deploy/%q --tail=300", kc, ns, deploy)
	stdout, stderr, err := ExecLocal(cmd)
	Expect(err).NotTo(HaveOccurred(), "read logs for deploy/%s: %s", deploy, stderr)

	for _, bad := range mustNotContain {
		Expect(strings.Contains(strings.ToLower(stdout), strings.ToLower(bad))).To(BeFalse(),
			"deploy/%s log contains unhealthy keyword %q", deploy, bad)
	}
	if len(mustContain) == 0 {
		LogStep(fitID, "log health deploy/%s checked (no required keyword)", deploy)
		return
	}

	found := false
	for _, good := range mustContain {
		if strings.Contains(strings.ToLower(stdout), strings.ToLower(good)) {
			found = true
			break
		}
	}
	Expect(found).To(BeTrue(), "deploy/%s log does not contain expected keywords %v", deploy, mustContain)
	LogStep(fitID, "log health deploy/%s passed", deploy)
}
