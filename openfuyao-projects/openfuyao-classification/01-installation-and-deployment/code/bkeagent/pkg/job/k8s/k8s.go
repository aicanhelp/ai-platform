package k8s

import (
	"context"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierr "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/utils"
)

type K8s interface {
	Execute(execCommands []string) ([]string, error)
}

type Task struct {
	K8sClient client.Client
	Exec      exec.Executor
}

var (
	configmap               = "configmap"
	secret                  = "secret"
	ro                      = "ro"
	rw                      = "rw"
	rx                      = "rx"
	supportResourceType     = []string{configmap, secret}
	supportResourceOperator = []string{ro, rx, rw}
	resourceLength          = 4
)

// Execute the command
// 示例[]string{"secret:ns/name:ro:/tmp/secret.json"} 获取secret/ns/name资源并写入/tmp/secret.json文件
// 示例[]string{"configmap:ns/name:rx:shell"} 获取configmap/ns/name中的资源，在agent以shell方式执行
// 示例[]string{"configmap:ns/name:rw:/tmp/iptables.rule"} 读取/tmp/iptables.rule中的内容并写入configmap/ns/name
func (t *Task) Execute(execCommands []string) ([]string, error) {
	result := []string{}
	// Support for multiple instructions
	for _, ec := range execCommands {
		// Instruction inspection
		ecList := strings.SplitN(ec, ":", resourceLength)
		if len(ecList) != resourceLength {
			return result, errors.New("Command format error")
		}
		resourceType := strings.ToLower(ecList[0])
		resourceName := strings.ToLower(ecList[1])
		resourceOperator := strings.ToLower(ecList[2])
		resourcePath := strings.ToLower(ecList[3])

		if !utils.ContainsString(supportResourceType, resourceType) {
			return result, errors.Errorf("Not with configMap or Secret resources, %s", resourceType)
		}
		if !utils.ContainsString(supportResourceOperator, resourceOperator) {
			return result, errors.Errorf("Unsupported operation types, %s", resourceOperator)
		}
		namespace, name, err := cache.SplitMetaNamespaceKey(resourceName)
		if err != nil {
			return result, errors.Errorf("The resource name is invalid %s-%s, %s", namespace, name, err.Error())
		}
		if namespace == "" {
			namespace = "default"
		}

		var obj client.Object
		switch resourceOperator {
		case ro:
			// Make sure the path exists
			if !path.IsAbs(resourcePath) {
				return result, errors.Errorf("You need to enter an absolute path, %s", resourcePath)
			}
			if !utils.Exists(resourcePath) {
				s1 := strings.Split(resourcePath, "/")
				s2 := strings.Join(s1[0:len(s1)-1], "/")
				err := os.MkdirAll(s2, os.ModePerm)
				if err != nil {
					return result, err
				}
			}

			if resourceType == configmap {
				obj = &corev1.ConfigMap{}
			}
			if resourceType == secret {
				obj = &corev1.Secret{}
			}
			// Get kubernetes resources
			err = t.K8sClient.Get(context.Background(), client.ObjectKey{Namespace: namespace, Name: name}, obj)
			if err != nil {
				return result, errors.Errorf("Unable to get the specified resource, %s-%s, %s", namespace, name, err.Error())
			}
			// Written to the file
			f, err := os.OpenFile(resourcePath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0666)
			defer f.Close()
			if err != nil {
				return result, err
			}
			switch v := obj.(type) {
			case *corev1.ConfigMap:
				for _, value := range v.Data {
					f.WriteString(value)
				}
			case *corev1.Secret:
				for _, value := range v.Data {
					f.Write(value)
				}
			}
		case rx:
			if resourceType == configmap {
				obj = &corev1.ConfigMap{}
			}
			if resourceType == secret {
				obj = &corev1.Secret{}
			}
			// Get kubernetes resources
			err = t.K8sClient.Get(context.Background(), client.ObjectKey{Namespace: namespace, Name: name}, obj)
			if err != nil {
				return result, errors.Errorf("Unable to get the specified resource, %s-%s, %s", namespace, name, err.Error())
			}
			script := []string{}
			switch v := obj.(type) {
			case *corev1.ConfigMap:
				for _, value := range v.Data {
					script = append(script, value)
				}
			case *corev1.Secret:
				for _, value := range v.Data {
					script = append(script, string(value))
				}
			}
			// Execute the script
			for _, s := range script {
				r, err := t.Exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", s)
				if err != nil {
					return result, err
				}
				result = append(result, r)
			}
			// If it is an absolute path, the file is written
			if path.IsAbs(resourcePath) {
				if !utils.Exists(resourcePath) {
					s1 := strings.Split(resourcePath, "/")
					s2 := strings.Join(s1[0:len(s1)-1], "/")
					err = os.MkdirAll(s2, os.ModePerm)
					if err != nil {
						return result, err
					}
				}
				ioutil.WriteFile(resourcePath, []byte(strings.Join(result, "\r\n")), 0644)
			}
		case rw:
			// Make sure the path exists
			if !path.IsAbs(resourcePath) {
				return result, errors.Errorf("You need to enter an absolute path, %s", resourcePath)
			}
			if !utils.IsFile(resourcePath) {
				return result, errors.Errorf("The specified file does not exist, %s", resourcePath)
			}
			content, err := ioutil.ReadFile(resourcePath)
			if err != nil {
				return result, err
			}
			// Update K8S resources
			switch resourceType {
			case configmap:
				obj = &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: namespace,
						Name:      name,
					},
					Data: map[string]string{
						"content": string(content),
					},
				}
			case secret:
				obj = &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: namespace,
						Name:      name,
					},
					Data: map[string][]byte{
						"content": content,
					},
					Type: corev1.SecretTypeOpaque,
				}
			}
			err = t.K8sClient.Update(context.Background(), obj)
			if err != nil {
				if apierr.IsNotFound(err) {
					err = t.K8sClient.Create(context.Background(), obj)
					if err != nil {
						return result, err
					}
				} else {
					return result, err
				}
			}
		}
	}
	return result, nil
}
