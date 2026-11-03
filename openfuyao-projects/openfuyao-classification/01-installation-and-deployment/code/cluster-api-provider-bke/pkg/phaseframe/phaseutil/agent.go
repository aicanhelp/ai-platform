/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	bkessh "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/remote"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
)

const expectedSplitParts = 2

// countAvailableAgentNodes counts cluster agent availability.
// Nodes in pingNodes are judged only by this round's ping result (e.g. upgrade re-pings all nodes
// that already carry NodeAgentReadyFlag). Nodes not pinged this round fall back to NodeAgentReadyFlag.
func countAvailableAgentNodes(bkeNodes bkev1beta1.BKENodes, pingNodes bkenode.Nodes, pingSuccess []string) int {
	pingedIPs := make(map[string]struct{}, pingNodes.Length())
	for _, n := range pingNodes {
		pingedIPs[n.IP] = struct{}{}
	}
	pingSuccessIPs := make(map[string]struct{}, len(pingSuccess))
	for _, result := range pingSuccess {
		pingSuccessIPs[GetNodeIPFromCommandWaitResult(result)] = struct{}{}
	}

	available := 0
	for _, bn := range bkeNodes {
		ip := bn.Spec.IP
		if _, pinged := pingedIPs[ip]; pinged {
			if _, ok := pingSuccessIPs[ip]; ok {
				available++
			}
			continue
		}
		if bkeNodes.GetNodeStateFlag(ip, bkev1beta1.NodeAgentReadyFlag) {
			available++
		}
	}
	return available
}

// PingBKEAgent ping bkeagent(send ping command) to check if it's ready.
// Only nodes with NodeAgentPushedFlag and without NodeAgentReadyFlag are checked (install / first push).
func PingBKEAgent(ctx context.Context, c client.Client, scheme *runtime.Scheme, bkeCluster *bkev1beta1.BKECluster) (error, []string, []string) {
	tmpBKENodes, err := nodeutil.GetBKENodesFromClient(ctx, c, bkeCluster.Namespace, bkeCluster.Name)
	if err != nil {
		return err, nil, nil
	}

	bkeNodes := bkev1beta1.BKENodes(tmpBKENodes)
	pingNodes := bkenode.Nodes{}
	for _, bkenode := range bkeNodes {
		if bkeNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeAgentReadyFlag) {
			continue
		}
		if bkeNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeAgentPushedFlag) {
			pingNodes = append(pingNodes, bkenode.ToNode())
		}
	}
	if pingNodes.Length() == 0 {
		return nil, nil, nil
	}
	return pingBKEAgentOnNodes(pingBKEAgentInput{
		ctx: ctx, client: c, scheme: scheme, cluster: bkeCluster,
		bkeNodes: bkeNodes, pingNodes: pingNodes,
	})
}

// PingBKEAgentOnNodes pings the given nodes (e.g. after bkeagent upgrade on all cluster nodes).
func PingBKEAgentOnNodes(
	ctx context.Context,
	c client.Client,
	scheme *runtime.Scheme,
	bkeCluster *bkev1beta1.BKECluster,
	nodes bkenode.Nodes,
) (error, []string, []string) {
	tmpBKENodes, err := nodeutil.GetBKENodesFromClient(ctx, c, bkeCluster.Namespace, bkeCluster.Name)
	if err != nil {
		return err, nil, nil
	}
	return pingBKEAgentOnNodes(pingBKEAgentInput{
		ctx: ctx, client: c, scheme: scheme, cluster: bkeCluster,
		bkeNodes: bkev1beta1.BKENodes(tmpBKENodes), pingNodes: nodes,
	})
}

type pingBKEAgentInput struct {
	ctx       context.Context
	client    client.Client
	scheme    *runtime.Scheme
	cluster   *bkev1beta1.BKECluster
	bkeNodes  bkev1beta1.BKENodes
	pingNodes bkenode.Nodes
}

func pingBKEAgentOnNodes(input pingBKEAgentInput) (error, []string, []string) {
	ctx, c, scheme := input.ctx, input.client, input.scheme
	bkeCluster, bkeNodes, pingNodes := input.cluster, input.bkeNodes, input.pingNodes
	const nodePerTime = 5
	waitInterval := 1 * time.Second

	allTime := pingNodes.Length() * nodePerTime
	waitTimeout := time.Duration(allTime) * time.Second
	if waitTimeout < 5*time.Minute {
		waitTimeout = 5 * time.Minute
	}
	pingCommand := command.Ping{
		BaseCommand: command.BaseCommand{
			Ctx:             ctx,
			Client:          c,
			Scheme:          scheme,
			NameSpace:       bkeCluster.Namespace,
			ClusterName:     bkeCluster.Name,
			OwnerObj:        bkeCluster,
			RemoveAfterWait: true,
			Unique:          false,
			WaitInterval:    waitInterval,
			WaitTimeout:     waitTimeout,
		},
		Nodes: pingNodes,
	}

	if err := pingCommand.New(); err != nil {
		return fmt.Errorf("create ping Command failed: %w", err), nil, nil
	}
	err, successNodes, failedNodes := pingCommand.Wait()
	GenerateBKEAgentStatus(successNodes, bkeCluster, bkeNodes.ToNodes(), bkeNodes, pingNodes)
	processCommandOutput(bkeNodes, pingCommand.Command, bkeCluster)

	if err != nil && !errors.Is(err, wait.ErrWaitTimeout) {
		return err, successNodes, failedNodes
	}
	return nil, successNodes, failedNodes
}

// processCommandOutput handles the command output and sets hostnames
func processCommandOutput(bkeNodes bkev1beta1.BKENodes, cmd *agentv1beta1.Command, bkeCluster *bkev1beta1.BKECluster) {
	stdOut := GetCommandStdOut(cmd, "ping")
	// get hostName from stdOut if node.HostName == ""

	// remove already has hostname nodes
	removeNodesWithHostname(bkeNodes, stdOut)

	// get All Keys
	keys := getStdOutKeys(stdOut)

	var indicesToRemove []int

	// get hostName from stdOut if node.HostName == ""
	indicesToRemove = updateNodeHostnames(bkeNodes, bkeCluster, keys, indicesToRemove)

	cleanupKeys(keys, indicesToRemove)
}

// removeNodesWithHostname removes nodes that already have hostnames from stdOut map
func removeNodesWithHostname(bkeNodes bkev1beta1.BKENodes, stdOut map[string]map[string][]string) {
	for _, node := range bkeNodes {
		if node.Spec.Hostname != "" {
			delete(stdOut, node.Spec.Hostname+"/"+node.Spec.IP)
		}
	}
}

// getStdOutKeys extracts all keys from the stdOut map
func getStdOutKeys(stdOut map[string]map[string][]string) []string {
	var keys []string
	for k := range stdOut {
		keys = append(keys, k)
	}
	return keys
}

// updateNodeHostnames updates node hostnames based on stdOut data
func updateNodeHostnames(bkeNodes bkev1beta1.BKENodes, bkeCluster *bkev1beta1.BKECluster, keys []string, indicesToRemove []int) []int {
	for index, bkeNode := range bkeNodes {
		params := NodeHostnameUpdateParams{
			BKECluster:      bkeCluster,
			Keys:            keys,
			IndicesToRemove: indicesToRemove,
			NodeIndex:       index,
			Node:            bkenode.Node(bkeNode.ToNode()),
			BKENodes:        bkeNodes,
		}
		indicesToRemove = processNodeHostnameUpdate(params)
	}
	return indicesToRemove
}

// NodeHostnameUpdateParams holds parameters for processNodeHostnameUpdate function
type NodeHostnameUpdateParams struct {
	BKECluster      *bkev1beta1.BKECluster
	Keys            []string
	IndicesToRemove []int
	NodeIndex       int
	Node            bkenode.Node
	BKENodes        bkev1beta1.BKENodes
}

// processNodeHostnameUpdate processes hostname update for a single node
func processNodeHostnameUpdate(params NodeHostnameUpdateParams) []int {
	for i, key := range params.Keys {
		if shouldUpdateHostname(params.BKENodes, params, key, i) {
			params.IndicesToRemove = append(params.IndicesToRemove, i)
			break
		}
	}
	return params.IndicesToRemove
}

// shouldUpdateHostname determines if a node's hostname should be updated
func shouldUpdateHostname(bkeNodes bkev1beta1.BKENodes, params NodeHostnameUpdateParams, key string, index int) bool {
	v := strings.Split(key, "/")
	if !isValidSplitResult(v) {
		return false
	}

	if len(v) < expectedSplitParts {
		return false
	}

	nodeHostName := v[0]
	nodeHostIp := v[1]

	if !isMatchingNode(params.Node, nodeHostIp) {
		return false
	}

	// Do not overwrite hostname when it is already set in BKECluster; kubelet uses spec.hostname via /etc/bkeagent/node.
	if isHostnameAlreadySet(bkeNodes, params.NodeIndex) {
		return false
	}

	updateNodeHostname(bkeNodes, params.NodeIndex, nodeHostName)
	return true
}

// isValidSplitResult checks if the split result is valid
func isValidSplitResult(parts []string) bool {
	return len(parts) == expectedSplitParts
}

// isMatchingNode checks if the node IP matches the host IP
func isMatchingNode(node bkenode.Node, hostIp string) bool {
	return hostIp != "" && node.IP == hostIp
}

// isHostnameAlreadySet checks if the node already has a hostname set
func isHostnameAlreadySet(bkenodes []confv1beta1.BKENode, nodeIndex int) bool {
	return bkenodes[nodeIndex].Spec.Hostname != ""
}

// updateNodeHostname updates the node's hostname
func updateNodeHostname(bkenodes []confv1beta1.BKENode, nodeIndex int, hostname string) {
	bkenodes[nodeIndex].Spec.Hostname = hostname
}

// cleanupKeys removes processed keys from the keys slice
func cleanupKeys(keys []string, indicesToRemove []int) []string {
	for i := len(indicesToRemove) - 1; i >= 0; i-- {
		idx := indicesToRemove[i]
		keys = append(keys[:idx], keys[idx+1:]...)
	}
	return keys
}

// PushAgent push bkeagent to all Nodes,and it's the only SshClient connection
// todo 和EnsureBKEAgent逻辑一致
func PushAgent(hosts []bkessh.Host, localKubeConfig []byte, ntpServer string) []string {
	if ntpServer != "" {
		file, err := os.ReadFile("/bkeagent.service.tmpl")
		if err != nil {
			return nil
		}
		ntpServer = strings.ReplaceAll(string(file), "--ntpserver=", fmt.Sprintf("--ntpserver=%s", ntpServer))
		err = os.WriteFile("/bkeagent.service", []byte(ntpServer), 0644)
		if err != nil {
			return nil
		}
	}

	startCommand := bkessh.Command{
		FileUp: []bkessh.File{
			{Src: "/bkeagent", Dst: "/usr/local/bin/"},
			{Src: "/bkeagent.service", Dst: "/usr/lib/systemd/system"},
		},
		Cmds: bkessh.Commands{
			"sudo chmod +x /usr/local/bin/bkeagent",
			// nodeName and localKubeConfig needs pre-exist before start bkeagent
			"sudo mkdir -p /etc/openFuyao/bkeagent",

			fmt.Sprintf("sudo echo -e %q > /etc/openFuyao/bkeagent/config", localKubeConfig),
			"sudo chmod 600 /etc/openFuyao/bkeagent/config",

			"sudo systemctl enable bkeagent",
			"sudo systemctl restart bkeagent",
		},
	}

	multiCli := bkessh.NewMultiCli(context.Background())
	defer multiCli.Close()

	var failedNodes []string

	regisErrs := multiCli.RegisterHosts(hosts)
	for hostIP, _ := range regisErrs {
		failedNodes = append(failedNodes, hostIP)
	}
	if len(failedNodes) == len(hosts) {
		return failedNodes
	}

	stdErrs, _ := multiCli.Run(startCommand)

	for nodeIP, _ := range stdErrs.Out() {
		failedNodes = append(failedNodes, nodeIP)

	}
	return failedNodes

}

func GetCommandStdOut(cmd *agentv1beta1.Command, cmdId ...string) map[string]map[string][]string {
	stdOut := make(map[string]map[string][]string)
	if cmd == nil || len(cmd.Status) == 0 {
		return stdOut
	}
	for k, status := range cmd.Status {
		tmpStdOut := make(map[string][]string)
		for _, condition := range status.Conditions {
			if (cmdId != nil || len(cmdId) != 0) && utils.ContainsString(cmdId, condition.ID) {
				tmpStdOut[condition.ID] = condition.StdOut
				continue
			}
			if condition.StdOut != nil || len(condition.StdOut) != 0 {
				tmpStdOut[condition.ID] = condition.StdOut
			}
		}
		if len(tmpStdOut) != 0 {
			stdOut[k] = tmpStdOut
		}
	}
	return stdOut
}
