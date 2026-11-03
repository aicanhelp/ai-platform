/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phaseutil

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"hash"
	"math/big"

	"golang.org/x/crypto/pbkdf2"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

// ClusterList represents a list of clusters that a user is invited to.
type ClusterList struct {
	ClusterName string `json:"ClusterName,omitempty"`
}

// UserSpec defines the desired state of User
type UserSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	// Username is the name of the user.
	Username string `json:"Username,omitempty"`
	// EncryptedPassword is the user's encrypted password.
	EncryptedPassword []byte `json:"EncryptedPassword,omitempty"`
	// Description is an optional description of the user.
	Description string `json:"Description,omitempty"`
	// InvitedByClustersList contains the list of cluster names the user is invited to.
	InvitedByClustersList []string `json:"InvitedByClustersList,omitempty"`
	// PlatformRole defines the role of the user within the platform.
	PlatformRole string `json:"PlatformRole,omitempty"`
	// FailedLoginRecords is a list of timestamps for each failed login attempt.
	FailedLoginRecords []v1.Time `json:"failedLoginRecords,omitempty"`
	// FirstLogin indicates whether this is the user's first login.
	FirstLogin bool `json:"FirstLogin,omitempty"`
}

// UserStatus defines the observed state of User
type UserStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	// LockStatus indicates whether the user is locked or unlocked.
	LockStatus string `json:"lockStatus,omitempty"`
	// LockedTimestamp is the timestamp when the user was locked.
	LockedTimestamp *v1.Time `json:"lockedTimestamp,omitempty"`
	// RemainAttempts is the number of remaining login attempts before the user gets locked.
	RemainAttempts int `json:"RemainAttempts,omitempty"`
}

// User is the Schema for the users API
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Status",type="string",JSONPath=".status.lockStatus"
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=".metadata.creationTimestamp",description="Creation timestamp"
// +kubebuilder:printcolumn:name="RemainAttempts",type="integer",JSONPath="..status.RemainAttempts"
// +kubebuilder:resource:shortName=ur
// +kubebuilder:resource:scope=Cluster
type User struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty"`

	Spec   UserSpec   `json:"spec,omitempty"`
	Status UserStatus `json:"status,omitempty"`
}

// UserList contains a list of User
// +kubebuilder:object:root=true
type UserList struct {
	v1.TypeMeta `json:",inline"`
	v1.ListMeta `json:"metadata,omitempty"`
	Items       []User `json:"items"`
}

var (
	userMgmtGVR = schema.GroupVersionResource{
		Group:    "users.openfuyao.com",
		Version:  "v1alpha1",
		Resource: "users",
	}
)

// GetUserInfo returns the User with given user name
func GetUserInfo(c dynamic.Interface, name string) (*User, error) {
	dr, err := c.Resource(userMgmtGVR).Get(context.Background(), name, v1.GetOptions{})
	if err != nil {
		return nil, err
	}
	var userCR User
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(dr.Object, &userCR)
	if err != nil {
		return nil, fmt.Errorf("error converting to %s %s: %w", userMgmtGVR.Resource, name, err)
	}
	return &userCR, nil
}

// CreateUserInfo saves the User into k8s
func CreateUserInfo(c dynamic.Interface, userCR *User) error {
	obj, err := StructToUnstructured(userCR)
	if err != nil {
		return fmt.Errorf("cannot convert to unstructured object: %w", err)
	}
	_, err = c.Resource(userMgmtGVR).Create(context.Background(), obj, v1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("create user %s failed: %w", userCR.Name, err)
	}
	return nil
}

// StructToUnstructured convert from any struct to Unstructured struct
// mainly for kubernetes usage
func StructToUnstructured(v interface{}) (*unstructured.Unstructured, error) {
	UnstructuredResult, err := runtime.DefaultUnstructuredConverter.ToUnstructured(v)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{
		Object: UnstructuredResult,
	}, nil
}

// UserInfoConfig stores the configs for userinfo secret
type UserInfoConfig struct {
	PasswdLength  int
	SaltLength    int
	Iterations    int
	KeyLength     int
	EncryptMethod func() hash.Hash
}

// GenerateDefaultUserInfo returns the initial username and raw password which will occur on the screen
func GenerateDefaultUserInfo(dynamicClient dynamic.Interface, config UserInfoConfig) (string, []byte, error) {
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
		TypeMeta: v1.TypeMeta{
			Kind:       "User",
			APIVersion: "users.openfuyao.com/v1alpha1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name: username,
		},
		Spec: UserSpec{
			Username:          username,
			EncryptedPassword: []byte(encryptedPassword),
			Description:       "内置默认平台管理员\nBuilt-in default platform administrator",
			PlatformRole:      "platform-admin",
			FirstLogin:        true,
		},
	}
	return user
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
		// crypto/rand 失败时退化为 0；无法向上返回 error，由调用方依赖确定性索引
		return 0
	}
	return int(index.Int64())
}

func encryptPassword(rawPassword []byte, cfg UserInfoConfig) (string, error) {
	// 生成随机的盐值
	salt := make([]byte, cfg.SaltLength)
	_, err := rand.Read(salt)
	if err != nil {
		return "", fmt.Errorf("cannot generate salt: %w", err)
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
