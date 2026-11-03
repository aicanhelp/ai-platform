/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package agent

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/root"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	// nsNamePartsCount 表示 namespace/name 格式拆分后的最小部分数
	nsNamePartsCount = 2
)

var (
	annotationKey   = "bkeagent.bocloud.com/create"
	annotationValue = "bke"
)

type Options struct {
	root.Options
	Args    []string `json:"args"`
	Name    string   `json:"name"`
	Command string   `json:"command"`
	File    string   `json:"file"`
	Nodes   string   `json:"nodes"`
}

var gvr = schema.GroupVersionResource{
	Group:    agentv1beta1.GroupVersion.Group,
	Version:  agentv1beta1.GroupVersion.Version,
	Resource: "commands",
}

func (op *Options) Exec() {
	cmd := op.buildCommand()
	if err := op.applyCommand(&cmd); err != nil {
		log.Errorf("Apply command %s failed: %v", op.Name, err)
		return
	}
	log.Infof("The execution command has been sent to the cluster, Please run the `bke command info "+
		"%s/%s` to optain the execution result", metav1.NamespaceDefault, op.Name)
}

func (op *Options) buildCommand() agentv1beta1.Command {
	cmd := agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annotationKey: annotationValue,
			},
		},
		Spec: agentv1beta1.CommandSpec{
			Suspend:  false,
			Commands: []agentv1beta1.ExecCommand{},
		},
	}
	cmd.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   agentv1beta1.GroupVersion.Group,
		Version: agentv1beta1.GroupVersion.Version,
		Kind:    "Command",
	})
	cmd.SetName(op.Name)
	cmd.SetNamespace(metav1.NamespaceDefault)
	cmd.Spec.NodeSelector = op.buildNodeSelector()
	return cmd
}

func (op *Options) buildNodeSelector() *metav1.LabelSelector {
	nodeMap := map[string]string{}
	for _, value := range strings.Split(op.Nodes, ",") {
		nodeMap[value] = value
	}
	return &metav1.LabelSelector{MatchLabels: nodeMap}
}

func (op *Options) applyCommand(cmd *agentv1beta1.Command) error {
	if op.Command != "" {
		cmd.Spec.Commands = append(cmd.Spec.Commands, agentv1beta1.ExecCommand{
			ID:            "command",
			Command:       []string{op.Command},
			Type:          agentv1beta1.CommandShell,
			BackoffIgnore: false,
			BackoffDelay:  0,
		})
	}

	if op.File != "" {
		if err := op.createConfigMapFromFile(cmd); err != nil {
			return err
		}
	}

	return op.installCommand(cmd)
}

func (op *Options) createConfigMapFromFile(cmd *agentv1beta1.Command) error {
	b1, err := os.ReadFile(op.File)
	if err != nil {
		return err
	}
	cm := corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: op.Name,
		},
		Data: map[string]string{
			"value": string(b1),
		},
	}
	_, err = global.K8s.GetClient().CoreV1().ConfigMaps(
		metav1.NamespaceDefault).Create(context.Background(), &cm, metav1.CreateOptions{})
	if err != nil {
		return err
	}
	cmd.Spec.Commands = append(cmd.Spec.Commands, agentv1beta1.ExecCommand{
		ID:            "file",
		Command:       []string{fmt.Sprintf("configmap:%s/%s:rx:shell", metav1.NamespaceDefault, op.Name)},
		Type:          agentv1beta1.CommandKubernetes,
		BackoffIgnore: false,
		BackoffDelay:  0,
	})
	return nil
}

func (op *Options) installCommand(cmd *agentv1beta1.Command) error {
	by, err := yaml2.Marshal(cmd)
	if err != nil {
		return err
	}

	cmdName := fmt.Sprintf("%s.yaml", cmd.Name)
	err = os.WriteFile(cmdName, by, utils.DefaultFilePermission)
	if err != nil {
		return fmt.Errorf("file generation failure %v", err)
	}

	return global.K8s.InstallYaml(cmdName, map[string]string{}, "")
}

func (op *Options) List() {
	commandList := &agentv1beta1.CommandList{}
	err := global.ListK8sResources(gvr, commandList)
	if err != nil {
		return
	}
	const tabPadding = 2 // 列之间的最小空格数，用于tabwriter对齐
	w := tabwriter.NewWriter(os.Stdout, 0, 0, tabPadding, ' ', 0)
	fmt.Fprintln(w, "name\tsuspend\tnode\tLastStartTime\tCompletionTime\tPhase\tStatus")
	for _, bc := range commandList.Items {
		for node, v := range bc.Status {
			completionTime := ""
			if v.CompletionTime != nil {
				completionTime = v.CompletionTime.String()
			}
			fmt.Fprintf(w, "%s/%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				bc.Namespace, bc.Name,
				strconv.FormatBool(bc.Spec.Suspend),
				node,
				v.LastStartTime.String(),
				completionTime,
				string(v.Phase),
				string(v.Status),
			)
		}
	}
	err = w.Flush()
	if err != nil {
		log.Errorf("Flush command tablewriter failed: %v", err)
	}
}

func (op *Options) Info() {
	ns := strings.Split(op.Args[0], "/")
	if len(ns) < nsNamePartsCount {
		log.Error("invalid argument format, expected namespace/name")
		return
	}
	var workloadUnstructured *unstructured.Unstructured
	var err error
	dynamicClient := global.K8s.GetDynamicClient()

	workloadUnstructured, err = dynamicClient.Resource(gvr).Namespace(ns[0]).Get(context.TODO(), ns[1], metav1.GetOptions{})
	if err != nil {
		log.Errorf("Get command %s/%s failed: %v", ns[0], ns[1], err)
		return
	}
	commands := &agentv1beta1.Command{}
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(workloadUnstructured.UnstructuredContent(), commands)
	if err != nil {
		log.Errorf("Convert command %s/%s failed: %v", ns[0], ns[1], err)
		return
	}

	const tabPadding = 2 // 列之间的最小空格数，用于tabwriter对齐
	for node, v := range commands.Status {
		for _, c := range v.Conditions {
			w := tabwriter.NewWriter(os.Stdout, 0, 0, tabPadding, ' ', 0)
			fmt.Fprintln(w, "name\tsuspend\tnode\tID\tSTATUS")
			fmt.Fprintf(w, "%s/%s\t%s\t%s\t%s\t%s\n",
				commands.Namespace, commands.Name,
				strconv.FormatBool(commands.Spec.Suspend),
				node,
				c.ID,
				string(c.Status),
			)
			err = w.Flush()
			if err != nil {
				log.Errorf("Flush command info tablewriter failed: %v", err)
			}

			output := ""
			if len(c.StdOut) > 0 {
				output = strings.Join(c.StdOut, "")
			}
			if len(c.StdErr) > 0 {
				output = strings.Join(c.StdErr, "")
			}
			fmt.Println(output)
		}
	}
}

func (op *Options) Remove() {
	ns := strings.Split(op.Args[0], "/")
	if len(ns) < nsNamePartsCount {
		log.Error("invalid argument format, expected namespace/name")
		return
	}
	var err error
	dynamicClient := global.K8s.GetDynamicClient()
	err = dynamicClient.Resource(gvr).Namespace(ns[0]).Delete(context.TODO(), ns[1], metav1.DeleteOptions{})
	if err != nil {
		log.Errorf("Delete command %s/%s failed: %v", ns[0], ns[1], err)
		return
	}
}

func (op *Options) SyncTime() {
	ntpServer := configinit.DefaultNTPServer
	if len(op.Args) >= 1 {
		ntpServer = op.Args[0]
	}
	err := ntp.Date(ntpServer)
	if err != nil {
		for i := 0; i < utils.MinManifestsImageArgs; i++ {
			time.Sleep(utils.ContainerWaitSeconds * time.Second)
			err = ntp.Date(ntpServer)
			if err == nil {
				return
			}
		}
		log.Errorf("Failed to connect to the ntp server %s", ntpServer)
	}
}
