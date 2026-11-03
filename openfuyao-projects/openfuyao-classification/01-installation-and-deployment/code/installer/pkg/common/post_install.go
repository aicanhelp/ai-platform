/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package common

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	"github.com/lithammer/dedent"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/operate"
	"installer/pkg/zlog"
)

var (
	createClusterSuccessTmpl = `
		Your cluster has created successfully!

		You can run the following command on the cluster node to check the pod status:

		  kubectl get pod -A

		You can run the following command on the cluster node to check the node network status:

		  kubectl get node
		`
	podStatusAbnormalTmpl = `
		Some pod states are abnormal after the wait time!

		You can run the following command on the cluster node to check the pod status:

		  kubectl get pod -A

		You can run the following command on the cluster node to check the node network status:

		  kubectl get node
		`
	addNodesSuccessTmpl = `
		Cluster expansion succeeds!

		You can run the following command on the cluster node to check the pod status:

		  kubectl get pod -A

		You can run the following command on the cluster node to check the node network status:

		  kubectl get node
		`
)

// ClusterStatusCheck is check cluster status
type ClusterStatusCheck struct {
	Node *cluster.Node
	// 等待时间 分钟
	TimeOut time.Duration
}

// NewClusterStatusCheck is new ClusterStatusCheck
func NewClusterStatusCheck(node *cluster.Node, timeOut time.Duration) *ClusterStatusCheck {
	return &ClusterStatusCheck{
		Node:    node,
		TimeOut: timeOut * time.Minute,
	}
}

// PodStatusCheck 检查pod状态
func (c *ClusterStatusCheck) PodStatusCheck() bool {
	client := c.Node.Client()

	cmd := "kubectl get pod -A | awk 'NR > 1 {print $4}'"
	startTime := time.Now()
	endTime := startTime.Add(c.TimeOut)

	statusStrings := []string{"Pending", "ContainerCreating", "CrashLoopBackOff", "Init", "Error", "ErrImagePull",
		"ImagePullBackOff", "Terminating", "PodInitiaLizing"}

	isOk := true
	for {
		currentTime := time.Now()
		if currentTime.After(endTime) {
			isOk = false
			break
		}

		output, err := client.SudoExecCmdNoLog(cmd)

		if err != nil {
			zlog.Error("failed to obtain pod status")
			time.Sleep(1 * time.Minute)
			continue
		}

		isSleep := false
		for _, subStr := range statusStrings {
			if strings.Contains(output, subStr) {
				isSleep = true
				break
			}
		}

		if isSleep {
			zlog.Info("init container")
			time.Sleep(30 * time.Second)
		} else {
			break
		}
	}
	return isOk
}

// PostInstallCheck 安装后置检查
func PostInstallCheck(node *cluster.Node) {
	fmt.Println("[post check] Check that all Pods are normal")
	out := "[post check] This can take a few minutes or a dozen minutes because the container images need to be downloaded"
	fmt.Println(out)
	fmt.Println("[post check] You can also exit the installer by pressing ctrl+c")
	statusCheck := NewClusterStatusCheck(node, constants.CreateClusterPostCheckTimeOut)
	res := statusCheck.PodStatusCheck()
	ctx := make(map[string]interface{})

	errMsg := `
		You can run the following command on the cluster node to check the cluster status:

		  kubectl get node
		  kubectl get pod -A
`

	templates, err := template.New("create").Parse(dedent.Dedent(createClusterSuccessTmpl))
	if err != nil {
		fmt.Printf("%q\n", errMsg)
	}
	templ := template.Must(templates, err)
	if !res {
		templates, err = template.New("create").Parse(dedent.Dedent(podStatusAbnormalTmpl))
		if err != nil {
			fmt.Printf("%q\n", errMsg)
		}
		templ = template.Must(templates, err)
	}

	err = templ.Execute(os.Stdout, ctx)
	if err != nil {
		fmt.Printf("%q\n", errMsg)
	}
}

// PostAddNodeCheck 添加节点后置检查
func PostAddNodeCheck(node *cluster.Node) {
	fmt.Println("[post check] Check that all Pods are normal")
	out := "[post check] This can take a few minutes or a dozen minutes because the container images need to downloaded"
	fmt.Println(out)
	statusCheck := NewClusterStatusCheck(node, constants.AddNodePostCheckTimeOut)
	res := statusCheck.PodStatusCheck()
	ctx := make(map[string]interface{})

	errMsg := `
		You can run the following command on the cluster node to check the status of the added node:

		  kubectl get node
		  kubectl get pod -A
`

	templa, err := template.New("addNodes").Parse(dedent.Dedent(addNodesSuccessTmpl))
	if err != nil {
		fmt.Printf("%q\n", errMsg)
	}
	templ := template.Must(templa, err)
	if !res {
		templa, err = template.New("addNodes").Parse(dedent.Dedent(podStatusAbnormalTmpl))
		if err != nil {
			fmt.Printf("%q\n", errMsg)
		}
		templ = template.Must(templa, err)
	}

	err = templ.Execute(os.Stdout, ctx)
	if err != nil {
		fmt.Printf("%q\n", errMsg)
	}
}

// FinishedDestroyHandler 完成集群卸载后的处理
func FinishedDestroyHandler(nodes cluster.Nodes) {
	removeTmpFiles(nodes)

	msg := `
	Successfully completed cluster destroy
`
	fmt.Printf(msg)
}

func RemoveTmpFiles(nodes cluster.Nodes, skipRemove bool) {
	if !skipRemove {
		removeTmpFiles(nodes)
	}
}

func removeTmpFiles(nodes cluster.Nodes) {
	rmFuyaoFile := fmt.Sprintf("rm -rf %s", filepath.Join(constants.OptPath, constants.OpenFuyao))
	cmdSlice := []string{rmFuyaoFile}

	op := operate.NewNodesOperate(nodes)
	op.Cmd = cmdSlice
	op.AllNodeExecOperate(op.ExecCmd)
	op.Cache.Clear()
	zlog.Info("remove tmp files success")
}
