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

package auth

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"console-service/pkg/constant"
	"console-service/pkg/utils/k8sutil"
	"console-service/pkg/utils/util"
	"console-service/pkg/zlog"
)

const (
	// cookie字段长度
	sessionIDByteLength = 16
	// cookie字段名称
	accessTokenName        = "AccessToken"
	refreshTokenName       = "RefreshToken"
	sessionIDName          = "SessionID"
	accessTokenExpiryName  = "AccessExpiry"
	refreshTokenExpiryName = "RefreshExpiry"
	// session储存命名空间
	sessionSecretNamespace = "session-secret"
	// other
	intBase10 = 10
	bitSize64 = 64
)

// SessionStore store the pair of a sessionID and a token
type SessionStore map[string][]byte

// GetAccessToken extract the access token from SessionStore
func (s *SessionStore) GetAccessToken() string { return string((*s)[accessTokenName]) }

// GetRefreshToken extract the refresh token from SessionStore
func (s *SessionStore) GetRefreshToken() string { return string((*s)[refreshTokenName]) }

// GetSessionID extract the sessionID from SessionStore
func (s *SessionStore) GetSessionID() string { return string((*s)[sessionIDName]) }

// GetExpiry extract the sessionID from SessionStore
func (s *SessionStore) GetExpiry() (string, string) {
	return string((*s)[accessTokenExpiryName]), string((*s)[refreshTokenExpiryName])
}

// Update updates all items except for sessionid
func (s *SessionStore) Update(accessToken, refreshToken string, accessExpiry, refreshExpiry time.Time) {
	accessExpiryTimestampStr := strconv.FormatInt(accessExpiry.Unix(), intBase10)
	refreshExpiryTimestampStr := strconv.FormatInt(refreshExpiry.Unix(), intBase10)
	(*s)[accessTokenName] = []byte(accessToken)
	(*s)[refreshTokenName] = []byte(refreshToken)
	(*s)[accessTokenExpiryName] = []byte(accessExpiryTimestampStr)
	(*s)[refreshTokenExpiryName] = []byte(refreshExpiryTimestampStr)
}

// NewStoreSession set up a new pair of token and sessionID
func NewStoreSession(token *AccessRefreshToken) (*SessionStore, error) {
	var sessionIDBytes [sessionIDByteLength]byte
	_, err := io.ReadFull(rand.Reader, sessionIDBytes[:])
	if err != nil {
		return nil, err
	}
	sessionID := hex.EncodeToString(sessionIDBytes[:])

	refreshExpiryTimestampStr := strconv.FormatInt(token.RefreshTokenExpiry.Unix(), intBase10)
	accessExpiryTimestampStr := strconv.FormatInt(token.AccessTokenExpiry.Unix(), intBase10)

	session := &SessionStore{
		accessTokenName:        []byte(token.AccessToken),
		refreshTokenName:       []byte(token.RefreshToken),
		sessionIDName:          []byte(sessionID),
		refreshTokenExpiryName: []byte(refreshExpiryTimestampStr),
		accessTokenExpiryName:  []byte(accessExpiryTimestampStr),
	}

	return session, nil
}

// GetSession looks up the session from etcd
func GetSession(clientset kubernetes.Interface, sessionID string) (*SessionStore, error) {
	key, err := util.GetSecretSymmetricEncryptKey(clientset, constant.ConsoleServiceTokenKey)
	if err != nil {
		zlog.Errorf("unable to retrieve console-service symmetric key, %v", err)
		return nil, err
	}

	secretTarget, err := getSecretFromSessionID(clientset, sessionID, key)
	if err != nil {
		zlog.Error("The session does not exist")
		return nil, err
	}

	decryptedToken, err := util.Decrypt(secretTarget.Data[accessTokenName], key)
	if err != nil {
		zlog.Errorf("unable to decrypt token, %v", err)
		return nil, err
	}
	decryptedRefreshToken, err := util.Decrypt(secretTarget.Data[refreshTokenName], key)
	if err != nil {
		zlog.Errorf("unable to decrypt refresh-token, %v", err)
		return nil, err
	}
	decryptedSessionID, err := util.Decrypt(secretTarget.Data[sessionIDName], key)
	if err != nil {
		zlog.Errorf("unable to decrypt sessionID, %v", err)
		return nil, err
	}
	util.ClearByte(key)

	return &SessionStore{
		accessTokenName:        decryptedToken,
		refreshTokenName:       decryptedRefreshToken,
		sessionIDName:          decryptedSessionID,
		accessTokenExpiryName:  secretTarget.Data[accessTokenExpiryName],
		refreshTokenExpiryName: secretTarget.Data[refreshTokenExpiryName],
	}, nil
}

func getSecretFromSessionID(clientset kubernetes.Interface, sessionID string, key []byte) (*corev1.Secret, error) {
	secretList, err := k8sutil.ListSecret(clientset, sessionSecretNamespace)
	if err != nil {
		return nil, err
	}
	var secretTarget *corev1.Secret
	for _, secret := range secretList.Items {
		secretSSID, ok := secret.Data[sessionIDName]
		if !ok {
			continue
		}
		secretSSIDDecrypted, err := util.Decrypt(secretSSID, key)
		if err != nil {
			continue
		}
		if string(secretSSIDDecrypted) == sessionID {
			secretTarget = &secret
			break
		}
	}
	if secretTarget == nil {
		return nil, errors.New("session not found")
	}
	return secretTarget, nil
}

// StoreSession stores the SessionStore as K8s secret
func StoreSession(clientset kubernetes.Interface, session *SessionStore, isUpdate bool) error {
	// set secret name
	sessionUuid := uuid.New()
	secretName := fmt.Sprintf("session-%s", sessionUuid)

	key, err := util.GetSecretSymmetricEncryptKey(clientset, constant.ConsoleServiceTokenKey)
	if err != nil {
		zlog.Errorf("unable to retrieve console-service symmetric key, %v", err)
		return err
	}

	if isUpdate {
		secretTarget, err := getSecretFromSessionID(clientset, session.GetSessionID(), key)
		if err != nil {
			zlog.Error("The session does not exist")
			return err
		}
		secretName = secretTarget.Name
		err = k8sutil.DeleteSecret(clientset, secretName, sessionSecretNamespace)
		if err != nil {
			zlog.Errorf("Failed to delete session: %v", err)
			return err
		}
	}

	encryptedToken, err := util.Encrypt((*session)[accessTokenName], key)
	if err != nil {
		zlog.Errorf("unable to encrypt token, %v", err)
		return err
	}
	encryptedRefreshToken, err := util.Encrypt((*session)[refreshTokenName], key)
	if err != nil {
		zlog.Errorf("unable to encrypt refresh-token, %v", err)
		return err
	}
	encryptedSessionID, err := util.Encrypt((*session)[sessionIDName], key)
	if err != nil {
		zlog.Errorf("unable to encrypt sessionID, %v", err)
		return err
	}
	util.ClearByte(key)

	// generate secret schema
	sessionSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: secretName, Namespace: sessionSecretNamespace},
		Data: map[string][]byte{accessTokenName: encryptedToken, refreshTokenName: encryptedRefreshToken,
			sessionIDName: encryptedSessionID, accessTokenExpiryName: (*session)[accessTokenExpiryName],
			refreshTokenExpiryName: (*session)[refreshTokenExpiryName],
		},
	}

	_, err = k8sutil.CreateSecret(clientset, sessionSecret)
	if err != nil {
		zlog.Error("Failed to store session")
		return err
	}
	zlog.Info("Session is stored to etcd")
	return nil
}

// UpdateSession updates the session-secret and extends the original session expiry
func UpdateSession(clientSet kubernetes.Interface, sessionID string, token *AccessRefreshToken) (*SessionStore, error) {
	ss, err := GetSession(clientSet, sessionID)
	if err != nil {
		return nil, err
	}
	ss.Update(token.AccessToken, token.RefreshToken, token.AccessTokenExpiry, token.RefreshTokenExpiry)

	return ss, nil
}

// GetTokenFromSessionID looks up in etcd for the token paired with sessionID
func GetTokenFromSessionID(clientset kubernetes.Interface, sessionID string) (string, string, error) {
	ss, err := GetSession(clientset, sessionID)
	if err != nil {
		return "", "", err
	}

	// check whether session has expired, delete session if expired
	accessExpiryStr, refreshExpiryStr := ss.GetExpiry()

	// if refresh-token expired, delete the whole session and login again
	if checkExpiry(refreshExpiryStr) {
		zlog.Error("The session has expired")
		err = DeleteSession(clientset, sessionID)
		if err != nil {
			zlog.Error("Error Deleting session")
			return "", "", err
		}
		return "", "", errors.New("session expired")
	}
	refreshToken := ss.GetRefreshToken()
	accessToken := ss.GetAccessToken()

	// only access-token expired, return refresh-token to update access-token
	if checkExpiry(accessExpiryStr) {
		zlog.Warn("The access-token has expired")
		return accessToken, refreshToken, errors.New(constant.OnlyAccessTokenExpiredErrorStr)
	}

	return accessToken, refreshToken, nil
}

// GetTokenFromOpenFuyaoAuthHeader looks up token in the openfuyaoauthheader
func GetTokenFromOpenFuyaoAuthHeader(req *http.Request) (string, error) {
	token := req.Header.Get(constant.OpenFuyaoAuthHeader)
	if !strings.HasPrefix(token, "Bearer ") {
		zlog.Errorf("cannot fetch token from openfuyao authentication header")
		return "", fmt.Errorf("cannot fetch token from openfuyao authentication header")
	}
	token = strings.TrimPrefix(token, "Bearer ")
	return token, nil
}

// checkExpiry returns true if the timestamp is earlier than current time or
// error occurs
func checkExpiry(timestampStr string) bool {
	timestamp, err := strconv.ParseInt(timestampStr, intBase10, bitSize64)
	if err != nil {
		fmt.Println("Error parsing timestamp:", err)
		return true
	}
	currentTime := time.Now()
	return currentTime.After(time.Unix(timestamp, 0))
}

// DeleteSession deletes the token paired with sessionID if exists
func DeleteSession(clientset kubernetes.Interface, sessionID string) error {
	zlog.Info("Deleting token and sessionID")

	key, err := util.GetSecretSymmetricEncryptKey(clientset, constant.ConsoleServiceTokenKey)
	if err != nil {
		zlog.Errorf("unable to retrieve console-service symmetric key, %v", err)
		return err
	}

	secretTarget, err := getSecretFromSessionID(clientset, sessionID, key)
	if err != nil {
		zlog.Error("The session does not exist")
		return err
	}
	util.ClearByte(key)

	err = k8sutil.DeleteSecret(clientset, secretTarget.Name, sessionSecretNamespace)
	if err != nil {
		zlog.Errorf("Failed to delete session: %v", err)
		return err
	}
	zlog.Info("Session is deleted")
	return nil
}
