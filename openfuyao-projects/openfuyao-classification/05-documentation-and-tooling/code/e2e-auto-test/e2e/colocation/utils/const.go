package utils

var (
	ColocationHelmConfig = map[string]string{
		"releaseName": "colocation-package",
		"helmUrl":     "oci://cr.openfuyao.cn/charts/colocation-package",
		"version":     "0.0.0-latest",
		"namespace":   "default",
	}
	NUMAAffinityHelmConfig = map[string]string{
		"releaseName": "numa-affinity-package",
		"helmUrl":     "oci://cr.openfuyao.cn/charts/numa-affinity-package",
		"version":     "0.0.0-latest",
		"namespace":   "default",
	}
)
