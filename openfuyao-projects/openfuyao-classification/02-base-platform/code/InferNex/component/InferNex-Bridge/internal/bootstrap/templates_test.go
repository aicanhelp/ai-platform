package bootstrap

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"
)

func TestDefaultTemplatesAlignMooncakeMasterDefaults(t *testing.T) {
	t.Parallel()
	for _, filename := range []string{"aggregate.yaml", "pd.yaml"} {
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			data, err := templatesFS.ReadFile("templates/" + filename)
			if err != nil {
				t.Fatalf("read embedded template: %v", err)
			}
			obj := &unstructured.Unstructured{}
			if err := yaml.Unmarshal(data, &obj.Object); err != nil {
				t.Fatalf("unmarshal embedded template: %v", err)
			}
			labels, ok, err := unstructured.NestedStringMap(
				obj.Object,
				"spec", "components", "mooncake", "master", "template", "metadata", "labels",
			)
			if err != nil || !ok || labels["openfuyao.com/kvmanager"] != "mooncake" {
				t.Fatalf("expected mooncake kvmanager label, got labels=%v ok=%v err=%v", labels, ok, err)
			}
			containers, ok, err := unstructured.NestedSlice(
				obj.Object,
				"spec", "components", "mooncake", "master", "template", "spec", "containers",
			)
			if err != nil || !ok || len(containers) != 1 {
				t.Fatalf("expected one mooncake-master container, got containers=%#v ok=%v err=%v", containers, ok, err)
			}
			c, ok := containers[0].(map[string]interface{})
			if !ok {
				t.Fatalf("unexpected container type: %T", containers[0])
			}
			if got := c["image"]; got != "quay.io/ascend/vllm-ascend:v0.23.0" {
				t.Fatalf("unexpected mooncake image: %v", got)
			}
			args, ok := c["args"].([]interface{})
			if !ok || len(args) != 1 {
				t.Fatalf("expected one mooncake args block, got %#v", c["args"])
			}
			arg, ok := args[0].(string)
			if !ok {
				t.Fatalf("unexpected args block type: %T", args[0])
			}
			for _, want := range []string{
				"export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH",
				"--port 50051",
				"--metrics_port 9003",
			} {
				if !strings.Contains(arg, want) {
					t.Fatalf("expected args to contain %q, got:\n%s", want, arg)
				}
			}
			ports, ok := c["ports"].([]interface{})
			if !ok || len(ports) != 2 {
				t.Fatalf("expected rpc and http ports, got %#v", c["ports"])
			}
			gotPorts := map[string]int64{}
			for _, p := range ports {
				pm, ok := p.(map[string]interface{})
				if !ok {
					t.Fatalf("unexpected port type: %T", p)
				}
				name, _ := pm["name"].(string)
				gotPorts[name] = int64Value(pm["containerPort"])
			}
			if gotPorts["rpc"] != 50051 || gotPorts["http"] != 9003 {
				t.Fatalf("unexpected ports: %v", gotPorts)
			}
		})
	}
}

func int64Value(v interface{}) int64 {
	switch n := v.(type) {
	case int:
		return int64(n)
	case int64:
		return n
	case float64:
		return int64(n)
	default:
		return 0
	}
}
