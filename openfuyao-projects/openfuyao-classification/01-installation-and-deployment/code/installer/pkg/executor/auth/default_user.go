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

package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"hash"
	"math/big"

	"golang.org/x/crypto/pbkdf2"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"

	"installer/pkg/clients/k8s"
	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// UserInfoConfig stores the configs for userinfo secret
type UserInfoConfig struct {
	PasswdLength  int
	SaltLength    int
	Iterations    int
	KeyLength     int
	EncryptMethod func() hash.Hash
}

// GenerateDefaultUserInfo returns the initial username and raw password which will occur on the screen
func GenerateDefaultUserInfo(k8sClient kubernetes.Interface, dynamicClient dynamic.Interface,
	config UserInfoConfig) (string, []byte, error) {
	// check if re-installed
	existed := checkDefaultUserExisted(dynamicClient, "admin")
	if existed {
		return "", nil, nil
	}

	// generate random raw password
	rawPasswd := generatePassword(config.PasswdLength)

	// encrypt raw password
	encryptedPasswd, err := encryptPassword(rawPasswd, config)
	if err != nil {
		return "", nil, err
	}

	// prepare admin user
	username := "admin"
	userCR := prepareUserInstance(username, encryptedPasswd)
	if err != nil {
		return "", nil, err
	}

	err = CreateUserInfo(dynamicClient, userCR)
	if err != nil {
		return "", nil, err
	}

	return username, rawPasswd, nil
}

func prepareUserInstance(username, encryptedPassword string) *User {
	user := &User{
		TypeMeta: metav1.TypeMeta{
			Kind:       "User",
			APIVersion: "users.openfuyao.com/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: username,
		},
		Spec: UserSpec{
			Username:          username,
			EncryptedPassword: []byte(encryptedPassword),
			Description:       "内置默认平台管理员",
			PlatformRole:      "platform-admin",
			FirstLogin:        true,
		},
	}
	return user
}

func createAdminClusterRoleBinding(k8sClient kubernetes.Interface) error {
	// 创建ClusterRoleBinding对象
	clusterRoleBinding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "admin-platform-admin",
			Labels: map[string]string{
				"openfuyao.com/clusterrole-ref": "platform-admin",
				"openfuyao.com/user-ref":        "admin",
				"role-type":                     "platform-role",
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "openfuyao-platform-admin",
		},
		Subjects: []rbacv1.Subject{
			{
				APIGroup: "rbac.authorization.k8s.io",
				Kind:     "User",
				Name:     "admin",
			},
		},
	}

	// 检查ClusterRoleBinding是否已经存在
	existingClusterRoleBinding, err := k8sClient.RbacV1().ClusterRoleBindings().Get(context.Background(),
		"admin-cluster-admin", metav1.GetOptions{})

	if err == nil {
		// 如果存在，更新ClusterRoleBinding
		clusterRoleBinding.ResourceVersion = existingClusterRoleBinding.ResourceVersion
		_, err = k8sClient.RbacV1().ClusterRoleBindings().Update(context.Background(), clusterRoleBinding,
			metav1.UpdateOptions{})
		if err != nil {
			zlog.Errorf("Error updating ClusterRoleBinding: %v", err)
			return err
		}
		zlog.Info("ClusterRoleBinding updated successfully.")
	} else {
		// 如果不存在，创建新的ClusterRoleBinding
		_, err = k8sClient.RbacV1().ClusterRoleBindings().Create(context.Background(),
			clusterRoleBinding, metav1.CreateOptions{})
		if err != nil {
			zlog.Errorf("Error creating ClusterRoleBinding: %v", err)
			return err
		}
		zlog.Info("ClusterRoleBinding created successfully.")
	}

	return nil
}

func generatePassword(length int) []byte {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()-_=+|[{}];:'\",<.>/?"
	const alphabetLength = 52
	const digitLength = 10
	const symbolLength = 31
	const half = 2
	password := make([]byte, length)

	for i := 0; i < length; i++ {
		randomIndex := generateRandomIndex(len(charset))
		password[i] = charset[randomIndex]
	}

	// 选择前中尾部三个位置保证三种要求都可满足
	alphabetIndex := generateRandomIndex(alphabetLength)
	digitIndex := generateRandomIndex(digitLength)
	symbolIndex := generateRandomIndex(symbolLength)
	password[0] = charset[alphabetIndex]
	password[length-1] = charset[alphabetLength+digitIndex]
	password[length/half] = charset[alphabetLength+digitLength+symbolIndex]

	return password
}

func generateRandomIndex(length int) int {
	bigLength := big.NewInt(int64(length))
	index, err := rand.Int(rand.Reader, bigLength)
	if err != nil {
		zlog.Errorf("cannot generate random value, err: %v", err)
		return 0
	}
	return int(index.Int64())
}

func encryptPassword(rawPassword []byte, cfg UserInfoConfig) (string, error) {
	// 生成随机的盐值
	salt := make([]byte, cfg.SaltLength)
	_, err := rand.Read(salt)
	if err != nil {
		zlog.Errorf("cannot call rand to generate salt, err: %v", err)
		return "", err
	}

	// 使用 PBKDF2 算法生成密文
	encryptedPassword := pbkdf2.Key(rawPassword, salt, cfg.Iterations, cfg.KeyLength, cfg.EncryptMethod)

	// 将盐值和密文合并并编码为 Base64 字符串
	encryptedData := append(salt, encryptedPassword...)
	encryptedPasswordBase64 := base64.StdEncoding.EncodeToString(encryptedData)

	// 返回加密后的密码
	return encryptedPasswordBase64, nil
}

func checkDefaultUserExisted(dClient dynamic.Interface, username string) bool {
	_, err := GetUserInfo(dClient, username)
	// admin secret 不存在
	if err != nil {
		return false
	}

	return true
}

// CreateDefaultUser 创建默认用户
func CreateDefaultUser(node *cluster.Node) (string, []byte, error) {
	kubeConfigPath, err := k8s.DownLoadKubeConfig(node)
	if err != nil {
		return "", nil, fmt.Errorf("failed download kube config %v", err)
	}

	k8sCfg := k8s.NewKubernetesCfg(kubeConfigPath)
	k8sClient, err := k8s.NewKubernetesClient(k8sCfg)
	if err != nil {
		return "", nil, fmt.Errorf("failed to get k8s client: %v", err)
	}
	// default configuration for userinfoconfig
	cfg := UserInfoConfig{
		PasswdLength:  12,
		SaltLength:    16,
		Iterations:    100000,
		KeyLength:     64,
		EncryptMethod: sha256.New,
	}

	username, passwd, err := GenerateDefaultUserInfo(k8sClient.Kubernetes(), k8sClient.Dynamic(), cfg)
	if err != nil {
		return "", nil, err
	}
	return username, passwd, nil
}

// PrintDefaultUserInfo 打印默认用户信息
func PrintDefaultUserInfo(cluster2 *cluster.Cluster) error {
	isContains := common.IsContainsComponentByCfg(cluster2, constants.FuyaoSystem)
	if !isContains {
		return nil
	}
	username, passwd, err := CreateDefaultUser(cluster2.Spec.Nodes.Masters[0])
	if err != nil {
		return err
	}
	defer utils.DestroyPassword(passwd)

	// 重复安装返回密码为空，则不打印登录信息
	if len(passwd) == 0 {
		zlog.Info("re-install")
		return nil
	}

	websiteIp := cluster2.Spec.Nodes.Masters[0].IP
	if len(cluster2.Spec.Nodes.Masters) > 1 {
		websiteIp = getVIP(cluster2.Spec.Components)
		if websiteIp == "" {
			return fmt.Errorf("failed to get vip")
		}
	}

	msg := `
	The website of the openFuyao is as follows:
	
		https://%s:%d

	You can login to the openFuyao using the following username and password:

		username: %s
		password: %s

`
	fmt.Println(fmt.Sprintf(msg, websiteIp, constants.FuyaoSystemPort, username, passwd))
	return nil
}

func getVIP(components []cluster.Component) string {
	for _, component := range components {
		if component.Group != constants.Ha {
			continue
		}
		for _, param := range component.Param {
			if param.Name == constants.VIP {
				return param.Value
			}
		}
	}
	return ""
}
