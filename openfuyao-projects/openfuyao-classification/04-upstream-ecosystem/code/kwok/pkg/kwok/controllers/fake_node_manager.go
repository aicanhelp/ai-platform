package controllers

import (
	"context"
	"fmt"
	"os"
	"strings"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

var (
	fakeNodeNameFmt = "kwok%04d-in-%s"
)

func AutoCreateOrDeleteFakeNodes(ctx context.Context, labelManaged string, client kubernetes.Interface, nodeCount int) error {
	if nodeCount < 0 {
		return nil
	}

	nodeName := os.Getenv("NODENAME")
	if nodeName == "" {
		return fmt.Errorf("AutoCreateOrDeleteFakeNodes: get empty NODENAME from env")
	}
	labelParts := strings.Split(labelManaged, "=")
	if len(labelParts) != 2 {
		return fmt.Errorf("AutoCreateOrDeleteFakeNodes: labelManaged is not valid: %s", labelManaged)
	}
	labelKey := labelParts[0]
	labelValue := labelParts[1]

	// 1. 生成所有的 fake node name
	expectedNames := make(map[string]struct{})
	for i := 0; i < nodeCount; i++ {
		fakeNode := fmt.Sprintf(fakeNodeNameFmt, i, nodeName)
		expectedNames[fakeNode] = struct{}{}
	}

	// 2. 获取当前节点下已存在的fake node
	nodeList, err := client.CoreV1().Nodes().List(ctx, meta.ListOptions{
		LabelSelector: labelManaged,
	})
	if err != nil {
		return fmt.Errorf("AutoCreateOrDeleteFakeNodes: list nodes error: %w", err)
	}

	// 3. 删除多余的节点
	for i := range nodeList.Items {
		if _, ok := expectedNames[nodeList.Items[i].Name]; ok {
			delete(expectedNames, nodeList.Items[i].Name)
			continue
		}
		err := client.CoreV1().Nodes().Delete(ctx, nodeList.Items[i].Name, meta.DeleteOptions{})
		if err != nil {
			return fmt.Errorf("failed to delete node %s: %w", nodeList.Items[i].Name, err)
		}
	}

	// 4. 创建缺失的节点
	for name := range expectedNames {
		node := &corev1.Node{
			ObjectMeta: meta.ObjectMeta{
				Name: name,
				Annotations: map[string]string{
					"node.alpha.kubernetes.io/ttl": "0",
				},
				Labels: map[string]string{
					"beta.kubernetes.io/arch":       "amd64",
					"beta.kubernetes.io/os":         "linux",
					"kubernetes.io/arch":            "amd64",
					"kubernetes.io/os":              "linux",
					"kubernetes.io/hostname":        name,
					"kubernetes.io/role":            "agent",
					"node-role.kubernetes.io/agent": "",
					"type":                          "kwok",
					labelKey:                        labelValue,
				},
			},
			Spec: corev1.NodeSpec{
				Taints: []corev1.Taint{
					{
						Key:    "kwok.x-k8s.io/node",
						Value:  "fake",
						Effect: v1.TaintEffectNoSchedule,
					},
				},
			},
		}

		_, err := client.CoreV1().Nodes().Create(ctx, node, meta.CreateOptions{})
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("failed to create node %s: %w", name, err)
		}
	}

	return nil
}
