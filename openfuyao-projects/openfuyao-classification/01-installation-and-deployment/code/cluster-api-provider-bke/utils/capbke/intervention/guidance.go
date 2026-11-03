/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package intervention

import (
	"fmt"
	"time"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

// MasterInitTimeout returns guidance for when master initialization
// polling exceeds the timeout duration.
func MasterInitTimeout(timeout time.Duration) Guidance {
	return Guidance{
		Category: CategoryMasterInit,
		Reason:   constant.MasterInitTimeoutReason,
		Message: fmt.Sprintf(
			"Master node initialization timed out after %v. Automatic retry will continue, "+
				"but if failures persist, manual intervention is required.",
			timeout,
		),
		Actions: []string{
			"Check BKEAgent logs on the master node: /var/log/openFuyao/bkeagent.log",
			"Verify the master node is reachable and has sufficient resources (CPU, memory, disk)",
			"Check for port conflicts (6443, 10250, 2379, 2380) or residual processes",
			"If the issue is resolved, the controller will automatically retry on the next reconcile",
		},
	}
}

// NodeBootstrapTerminal returns guidance for when a node's bootstrap
// has permanently failed (NodeFailedFlag is set).
func NodeBootstrapTerminal(nodeIP string, failedNodes []string) Guidance {
	return Guidance{
		Category: CategoryNodeFailure,
		Reason:   constant.NodeBootstrapTerminalReason,
		Message: fmt.Sprintf(
			"Node %s bootstrap failed permanently (failed nodes: %v). "+
				"The node has been marked as failed and will be skipped in subsequent reconciles.",
			nodeIP, failedNodes,
		),
		Actions: []string{
			"Check BKEAgent logs on the failed node: /var/log/openFuyao/bkeagent.log",
			"Manually resolve the issue (e.g., clean residual processes, fix certificates, free disk space)",
			"After resolving, restart BKEAgent on the node to trigger a new bootstrap command",
			"If the problem cannot be resolved, delete the BKECluster resource to clean up",
		},
	}
}

// RetryExhausted returns guidance for when the StatusManager has
// exhausted the allowed failure count and set the cluster to a
// terminal *Failed health state.
func RetryExhausted(healthState string, retryCount int) Guidance {
	return Guidance{
		Category: CategoryRetryExhausted,
		Reason:   constant.RetryExhaustedReason,
		Message: fmt.Sprintf(
			"Cluster health state transitioned to %s after %d consecutive failures. "+
				"Automatic retries have been stopped. All phases are skipped until manual recovery.",
			healthState, retryCount,
		),
		Actions: []string{
			"Review the BKECluster status and identify which phase failed",
			"Check controller logs for detailed error information",
			"Resolve the root cause on the affected nodes",
			"After fixing, add the retry annotation to the BKECluster resource to reset cluster status and resume reconciliation",
		},
	}
}
