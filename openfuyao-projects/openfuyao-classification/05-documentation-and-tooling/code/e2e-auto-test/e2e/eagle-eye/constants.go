package eagle_eye

// EagleEyeHelmConfig holds helm install parameters for Eagle-Eye (used when not already installed).
var EagleEyeHelmConfig = struct {
	ReleaseName string
	ChartURL    string
	Version     string
	Namespace   string
}{
	ReleaseName: "eagle-eye",
	ChartURL:    "oci://cr.openfuyao.cn/charts/eagle-eye",
	Version:     "0.0.0-latest",
	Namespace:   "default",
}

// AscendChipByIP maps node InternalIP to Ascend chip type (e.g. "910A1", "910B").
var AscendChipByIP = map[string]string{
	"192.168.201.14": "Ascend910B",
	"192.168.201.15": "Ascend910B",
	// Add more: "node_ip": "910A2", etc.
}

// AscendTypesWithHBM are Ascend chip types that support real HBM metrics (not NA).
var AscendTypesWithHBM = []string{"Ascend910", "Ascend910B", "Ascend910A3", "Ascend910A5"}

// AscendTypesExcludedForDDR  are Ascend chip types that support real DDR metrics (not NA).
var AscendTypesExcludedForDDR = []string{"Ascend910B", "Ascend910A3"}

// AscendTypesExcludedForNetwork are Ascend chip types that support real network metrics (not NA).
var AscendTypesExcludedForNetwork = []string{"Ascend310", "Ascend310B", "Ascend310P"}
