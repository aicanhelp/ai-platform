/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	klog "k8s.io/klog/v2"

	v1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
	"openfuyao.com/colocation-service/pkg/client/clientset/versioned"
)

func main() {
	klog.InitFlags(nil)

	kubeconfigPath := loadKubeconfigPath()
	restClientConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	if err != nil {
		klog.Fatalf("failed to load kubeconfig from %s: %v", kubeconfigPath, err)
	}

	client, err := versioned.NewForConfig(restClientConfig)
	if err != nil {
		klog.Fatalf("failed to create colocation client: %v", err)
	}
	clientset, err := kubernetes.NewForConfig(restClientConfig)
	if err != nil {
		klog.Fatalf("failed to create kubernetes client: %v", err)
	}

	createTestRule(client)
	printRules(client, clientset)
}

func loadKubeconfigPath() string {
	path := os.Getenv("KUBECONFIG")
	if path != "" {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		klog.Fatalf("failed to get user home directory: %v", err)
	}
	return filepath.Join(home, ".kube", "config")
}

func createTestRule(client versioned.Interface) {
	rule := v1.ColocationWorkloadMatchRule{}
	rule.TypeMeta.APIVersion = "v1"
	rule.TypeMeta.Kind = "ColocationWorkloadMatchRule"
	rule.ObjectMeta.Namespace = os.Getenv("POD_NAMESPACE")
	rule.ObjectMeta.Name = "ab-ce.ef_okabc"
	rule.Spec.Selectors = []v1.Selector{
		{
			Namespace:     os.Getenv("POD_NAMESPACE"),
			Name:          "nginx",
			WorkloadTypes: []string{"Deployment"},
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": "nginx",
				},
			},
		},
	}

	isValid := StringValidityCheck(rule.ObjectMeta.Name, NormalMode)
	klog.Info(rule.ObjectMeta.Name, isValid)
	if !isValid {
		fmt.Println("rule name must consist of alphanumeric characters, '-', '_' or '.', " +
			"and must start and end with an alphanumeric character")
		return
	}

	savedRule, err := client.ColocationV1().ColocationWorkloadMatchRules(
		rule.ObjectMeta.Namespace).Create(context.TODO(), &rule, metav1.CreateOptions{})
	klog.Info(savedRule.ObjectMeta.Name, err)
}

func printRules(client versioned.Interface, clientset kubernetes.Interface) {
	rules, _ := client.ColocationV1().ColocationWorkloadMatchRules("").List(context.TODO(), metav1.ListOptions{})
	for _, rule := range rules.Items {
		fmt.Println("rule.name:", rule.ObjectMeta.Name)
		for _, selector := range rule.Spec.Selectors {
			printSelectorPods(clientset, selector)
		}
	}
}

func printSelectorPods(clientset kubernetes.Interface, selector v1.Selector) {
	fmt.Println("  rule.namespace:", selector.Namespace)
	labelSelector, err := metav1.LabelSelectorAsSelector(selector.Selector)
	if err != nil {
		fmt.Printf("  couldn't convert rule selector: %v\n", err)
	}
	fmt.Println("  rule.labelSelector:", labelSelector.String())
	pods, err := clientset.CoreV1().Pods(selector.Namespace).List(context.TODO(),
		metav1.ListOptions{LabelSelector: labelSelector.String()})
	if err != nil {
		fmt.Printf("  couldn't list pods: %v\n", err)
	}
	for _, pod := range pods.Items {
		fmt.Println("  rule.match.workload:", pod.Name)
	}
}

type StringCheckMode string

const (
	// 常规模式
	NormalMode StringCheckMode = "normal"
	// 自定义模式: 标签Key的格式化校验
	LabelKeyMode StringCheckMode = "label_key_mode"
)

func StringValidityCheck(input string, mode StringCheckMode) bool {
	if input == "" {
		return false
	}

	var pattern string
	if mode == NormalMode {
		pattern = "^([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]$"
	} else if mode == LabelKeyMode {
		pattern = "[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*"
	}
	re := regexp.MustCompile(pattern)

	return re.MatchString(input)
}
