package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

type NodeInfo struct {
	Hostname string   `json:"hostname"`
	IP       string   `json:"ip"`
	Password string   `json:"password"`
	Port     string   `json:"port"`
	Username string   `json:"username"`
	Role     []string `json:"role"`
}

var isDaily = true

func SystemReboot(sshClient *executor.SSHExecutor, reboot_time int) error {
	cmd := fmt.Sprintf("shutdown -r +%d", reboot_time)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("reboot command execute failed : %v", err)
	}
	return nil
}

func DoGuideInstallationAtBackground(sshClient *executor.SSHExecutor, args ...string) error {
	cmdArg := strings.Join(args, " ")
	cmd := "nohup bke init"
	if cmdArg != "" {
		cmd = fmt.Sprintf("%s %s", cmd, cmdArg)
	}
	cmd = fmt.Sprintf("%s > /dev/null 2>&1 &", cmd)
	// cmd = fmt.Sprintf("%s > /root/1.txt &", cmd)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("bke init excute failed : %v", err)
	}
	return nil
}

func DoGuideInstallation(sshClient *executor.SSHExecutor, args ...string) error {
	cmdArg := strings.Join(args, " ")
	cmd := "bke init"
	if cmdArg != "" {
		cmd = fmt.Sprintf("%s %s", cmd, cmdArg)
	}
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("bke init excute failed : %v", err)
	}
	return nil
}

func DoBkeReset(sshClient *executor.SSHExecutor) error {
	cmd := "bke reset --all --mount --confirm"
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("bke reset command failed : %v", err)
	}
	return nil
}

func ClusterReset(sshClient *executor.SSHExecutor) error {
	cmdArgs := `[{"op": "replace", "path": "/metadata/annotations/bke.bocloud.com~1ignore-target-cluster-delete", "value": "false"}, {"op": "add", "path": "/spec/reset", "value": true}]`
	cmd := fmt.Sprintf("kubectl patch bc bke-cluster -n bke-cluster --type='json' -p='%s'", cmdArgs)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("cluster reset command failed : %v", err)
	}
	return nil
}

func ClusterResetCheck(sshClient *executor.SSHExecutor, clusterName string, clusterNamespace string) (bool, error) {
	cmd := fmt.Sprintf("kubectl get bc %s -n %s", clusterName, clusterNamespace)
	res, err := sshClient.Exec(cmd)
	fullError := res.Stderr
	if err != nil {
		fullError = err.Error() + "\n" + res.Stderr
	}

	if err != nil {
		if strings.Contains(fullError, "NotFound") || strings.Contains(fullError, "not found") {
			return true, nil
		}
		return false, fmt.Errorf("cluster reset check failed: %v, stderr: %s", err, res.Stderr)
	}
	return false, nil
}

func ClusterCreateAtBackground(sshClient *executor.SSHExecutor, clusterConfigPath, nodeConfigPath string) error {
	cmd := fmt.Sprintf("nohup bke cluster create -f %s -n %s > /dev/null 2>&1 &", clusterConfigPath, nodeConfigPath)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("cluster create failed : %v", err)
	}
	return nil
}

func CopyClusterConfig(sshClient *executor.SSHExecutor, clusterYaml string, curPath string) error {
	cmd := fmt.Sprintf("cp /bke/cluster/%s %s", clusterYaml, curPath)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("copy command failed : %v", err)
	}
	return nil
}

func RemoveClusterConfig(sshClient *executor.SSHExecutor, clusterYaml string, curPath string) error {
	cmd := fmt.Sprintf("rm -rf %s%s", curPath, clusterYaml)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("remove command failed : %v", err)
	}
	return nil
}

func ClusterConfigdeletion(sshClient *executor.SSHExecutor, addons []string, clusterConfigPath string) error {
	for _, addonName := range addons {
		cmd := fmt.Sprintf(`yq -i 'del(.spec.clusterConfig.addons[] | select(.name == "%s"))' %s`, addonName, clusterConfigPath)
		_, err := sshClient.Exec(cmd)
		if err != nil {
			return fmt.Errorf("cluster config deletion failed : %v", err)
		}
	}
	return nil
}

func ClusterConfigNodeInfoAdd(sshClient *executor.SSHExecutor, nodes []NodeInfo, clusterConfigPath string) error {
	clearCmd := fmt.Sprintf(`yq -i '.spec.clusterConfig.nodes = []' %s`, clusterConfigPath)
	_, err := sshClient.Exec(clearCmd)
	if err != nil {
		return fmt.Errorf("node info clear failed : %v", err)
	}
	jsonBytes, err := json.Marshal(nodes)
	if err != nil {
		return fmt.Errorf("node info parse failed : %v", err)
	}
	nodesStr := string(jsonBytes)
	cmd := fmt.Sprintf(`yq -i '.spec.clusterConfig.nodes += %s' %s`, nodesStr, clusterConfigPath)
	_, err = sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("node info add failed : %v", err)
	}
	return nil
}

// CleanClusterConfigNodes 从集群配置文件中删除 nodes 字段（bc拆分后节点信息已移到 BKENode）
func CleanClusterConfigNodes(sshClient *executor.SSHExecutor, clusterConfigPath string) error {
	cmd := fmt.Sprintf(`yq -i 'del(.spec.clusterConfig.nodes)' %s`, clusterConfigPath)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("clean cluster config nodes failed : %v", err)
	}
	return nil
}

// GenerateBKENodeConfig 生成 BKENode YAML 文件
// bcName 是 BKECluster 名称（例如 "bke-clusterName"）
func GenerateBKENodeConfig(sshClient *executor.SSHExecutor, nodes []NodeInfo, nodeConfigPath string, bcName string) error {
	var yamlDocs []string
	for _, node := range nodes {
		roles := ""
		for _, r := range node.Role {
			roles += fmt.Sprintf("  - %s\n", r)
		}
		doc := fmt.Sprintf(`apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: %s
  namespace: %s
  labels:
    cluster.x-k8s.io/cluster-name: %s
spec:
  hostname: %s
  ip: %s
  password: '%s'
  port: "%s"
  username: %s
  role:
%sstatus: {}`, node.Hostname, bcName, bcName, node.Hostname, node.IP, node.Password, node.Port, node.Username, roles)
		yamlDocs = append(yamlDocs, doc)
	}

	content := strings.Join(yamlDocs, "\n---\n")
	writeCmd := fmt.Sprintf("cat > %s << 'BKENODE_EOF'\n%s\nBKENODE_EOF", nodeConfigPath, content)
	_, err := sshClient.Exec(writeCmd)
	if err != nil {
		return fmt.Errorf("generate BKENode config failed: %v", err)
	}
	return nil
}

func BkeClusterDeployCheck(sshClient *executor.SSHExecutor, clusterName string, clusterNamespace string) bool {
	cmd := fmt.Sprintf("kubectl get bc %s -n %s", clusterName, clusterNamespace)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return false
	}
	return true
}

func BkeClusterPhaseCheck(dynamicClient dynamic.Interface, phase string, clusterName string, clusterNamespace string) (bool, error) {

	gvr := schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1",
		Resource: "bkeclusters",
	}

	cluster, err := dynamicClient.Resource(gvr).Namespace("bke-cluster").Get(context.Background(), "bke-cluster", metav1.GetOptions{})
	if err != nil {
		return false, fmt.Errorf("cluster get failed : %v", err)
	}

	phaseStatus, found, err := unstructured.NestedSlice(cluster.Object, "status", "phaseStatus")
	if err != nil {
		return false, err
	}
	if !found {
		return false, fmt.Errorf("phaseStatus not found in bc")
	}

	for _, item := range phaseStatus {
		if phaseMap, ok := item.(map[string]interface{}); ok {
			name := getString(phaseMap, "name")
			status := getString(phaseMap, "status")
			startTime := getString(phaseMap, "startTime")
			endTime := getString(phaseMap, "endTime")
			message := getString(phaseMap, "message")

			if name == phase && status == "Running" {
				fmt.Printf("%s\n", name)
				fmt.Printf("     Status: %s\n", status)
				fmt.Printf("     Time: %s ~ %s\n", startTime, endTime)
				if message != "" {
					fmt.Printf("     Message: %s\n", message)
				}
				return true, nil
			}
		}
	}
	return false, nil
}

func getString(m map[string]interface{}, key string) string {
	if val, ok := m[key]; ok && val != nil {
		return fmt.Sprintf("%v", val)
	}
	return ""
}

func PodStatusCheck(k8sClient *k8s.K8SClient, ctx context.Context) (bool, error) {
	pods, err := k8sClient.ListPods(ctx, "", metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("get pod failed : %v", err)
	}
	if len(pods.Items) == 0 {
		return false, nil
	}
	for _, pod := range pods.Items {
		if pod.Status.Phase != "Running" && pod.Status.Phase != "Succeeded" {
			return false, nil
		}
	}
	return true, nil
}

func CommandCheck(sshClient *executor.SSHExecutor) (bool, error) {
	cmd := "kubectl get pod -A"
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return false, fmt.Errorf("command execute failed : %v", err)
	}
	return true, nil
}

func InstallationCheck(sshClient *executor.SSHExecutor, ctx context.Context) (bool, error) {
	// create k8s client
	var k8sClient *k8s.K8SClient
	var err error

	if isDaily {
		k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	} else {
		k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
	}
	if err != nil {
		return false, fmt.Errorf("k8s client create failed : %v", err)
	}

	// pod check
	podCheck, err := PodStatusCheck(k8sClient, ctx)
	if err != nil {
		return false, err
	}

	// command check
	commandCheck, err := CommandCheck(sshClient)
	if err != nil {
		return false, err
	}

	if podCheck == false || commandCheck == false {
		return false, nil
	}
	return true, nil
}

func ClusterHealthCheck(sshClient *executor.SSHExecutor, clusterName string, clusterNamespace string) (bool, error) {
	cmd := fmt.Sprintf("kubectl get bc %s -n %s -o custom-columns=STATUS:.status.clusterStatus --no-headers", clusterName, clusterNamespace)
	res, err := sshClient.Exec(cmd)
	if err != nil {
		return false, fmt.Errorf("health check command failed : %v", err)
	}
	stdout := strings.TrimSpace(strings.Trim(res.Stdout, "'"))
	if stdout == "Ready" {
		return true, nil
	}
	return false, nil
}
