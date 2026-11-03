package controller

import (
	"fmt"
	"strconv"
	"strings"

	clusterv1alpha1 "openfuyao.com/clusters-computing-operator/api/v1alpha1"
)

func ExpandNode(nodeExpr string) (result []string) {
	// e.g. rack[01-10]node[1-100]
	// prefix = "rack", postfix = "node[1-100]"
	var prefix, postfix string
	// indexRange = "01-10", start = "01", end = "10"
	var indexRange, start, end string
	// result
	var nodeList []string

	// STEP 1. get prefix and post fix of the 1st range in expression
	startIdx := strings.IndexByte(nodeExpr, '[')
	if startIdx == -1 {
		// a real node name
		result = append(result, nodeExpr)
		return
	}
	// it is a node expression instead of a node
	prefix = nodeExpr[:startIdx]

	endIdx := strings.IndexByte(nodeExpr, ']')
	if endIdx == -1 {
		// it is assumed not happen because webhook should make sure the expression is valid
		result = append(result, nodeExpr)
		return
	}
	postfix = nodeExpr[endIdx+1:]

	// STEP 2. get 1st range of the expression
	indexRange = nodeExpr[startIdx+1 : endIdx]
	dashIdx := strings.IndexByte(indexRange, '-')
	if dashIdx == -1 {
		result = append(result, nodeExpr)
		return
	}

	start = indexRange[:dashIdx]
	end = indexRange[dashIdx+1:]

	// startIdx and endIdx are reused for iteration
	startIdx, err := strconv.Atoi(start)
	if err != nil {
		result = append(result, nodeExpr)
		return
	}
	endIdx, err = strconv.Atoi(end)
	if err != nil {
		result = append(result, nodeExpr)
		return
	}

	// STEP 3. expand the node range expression to node array
	format := "%s%d%s"
	if start[0] == '0' && len(start) > 1 {
		format = fmt.Sprintf("%%s%%0%dd%%s", len(start))
	}

	for i := startIdx; i <= endIdx; i++ {
		node := fmt.Sprintf(format, prefix, i, postfix)
		nodeList = append(nodeList, node)
	}
	idx := strings.IndexByte(postfix, '[')
	if idx == -1 {
		result = nodeList
		// there is no node range anymore
		return
	}

	// STEP 4: expand node range in the next level
	for _, nodeStr := range nodeList {
		node := ExpandNode(nodeStr)
		result = append(result, node...)
	}
	return
}

func generateAscendIniFile(tmc *clusterv1alpha1.TenantManagedCluster) string {
	iniContent := fmt.Sprintf(`[master]
%s

[worker]
%s

[deploy_node]
%s

[npu_node]

[large_scale]
SUB_GROUP_MAX_SIZE=5
AUTO_GROUP_BY_SUBNET=0

[all:vars]

	`,
		strings.Join(tmc.Spec.MasterList, "\n"),
		strings.Join(tmc.Spec.WorkerList, "\n"),
		strings.Join(tmc.Spec.WorkerList, "\n"),
	)

	return iniContent
}
