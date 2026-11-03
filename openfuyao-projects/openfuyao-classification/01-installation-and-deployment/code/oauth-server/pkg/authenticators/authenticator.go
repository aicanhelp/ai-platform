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

// Package authenticators check deals with password authentication
package authenticators

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"hash"
	"regexp"
	"strconv"

	"golang.org/x/crypto/pbkdf2"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/dynamic"

	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/fuyaouser"
	"openfuyao/oauth-server/pkg/zlog"
)

// PasswordAuthenticator in an authenticator that uses username/password to verify identities
type PasswordAuthenticator interface {
	AuthenticatePassword(ctx context.Context, username string, password []byte) (*authenticator.Response, bool, error)
	ResetPassword(ctx context.Context, username string, oldPassword, newPassword []byte) error
	ConfirmPassword(ctx context.Context, username string, newPassword []byte) error
}

// FuyaoPasswordAuthenticator is the default password authenticator for openfuyao oauth-server
type FuyaoPasswordAuthenticator struct {
	k8sClient dynamic.Interface
	encryptor Encryptor
}

// NewFuyaoPasswordAuthenticator inits FuyaoPasswordAuthenticator
func NewFuyaoPasswordAuthenticator(k8sClient dynamic.Interface) *FuyaoPasswordAuthenticator {
	return &FuyaoPasswordAuthenticator{
		k8sClient: k8sClient,
		encryptor: NewPBKDF2Encryptor(),
	}
}

// AuthenticatePassword authenticates the password
func (a *FuyaoPasswordAuthenticator) AuthenticatePassword(
	ctx context.Context,
	username string, passwd []byte,
) (*authenticator.Response, bool, error) {
	// fetch old password
	userinfo, base64EncryptedPassword, err := a.fetchUserInfoAndStoredPassword(username)
	if err != nil {
		return nil, false, err
	}

	// verify the input password
	if ok, err := a.encryptor.VerifyPassword(passwd, base64EncryptedPassword); !ok || err != nil {
		return nil, false, err
	}
	zlog.LogInfo("verify password succeed")
	return &authenticator.Response{User: userinfo}, true, nil
}

func (a *FuyaoPasswordAuthenticator) checkPasswordComplexity(username string, passwd []byte) bool {
	// check password length
	if len(passwd) < constants.PasswordMinLen || len(passwd) > constants.PasswordMaxLen {
		zlog.LogError("the password length should lie between 8 and 32")
		return false
	}

	// check that the password at least contains one lowercase/uppercase letter, one number and one special character
	reUpperCase := regexp.MustCompile(`[A-Z]`)
	reLowerCase := regexp.MustCompile(`[a-z]`)
	reDigit := regexp.MustCompile(`[0-9]`)
	reSpecialChar := regexp.MustCompile(`[\x60!\"#$%&'()*+,-./:;<=>?@[\\^\]_{|}~ ]`)

	if (!reUpperCase.Match(passwd) && !reLowerCase.Match(passwd)) || !reDigit.Match(passwd) ||
		!reSpecialChar.Match(passwd) {
		zlog.LogError("password must contain at least one lowercase letter or one uppercase letter, " +
			"one number, and one special character")
		return false
	}

	// check whether the password is contained in username / reversed username
	if isByteSameAsString(passwd, username) || isByteSameAsString(passwd, reverseString(username)) {
		zlog.LogError("password cannot be the same as the account number or the reverse account number")
		return false
	}

	return true
}

func reverseString(s string) string {
	var reversed string
	for _, char := range s {
		reversed = string(char) + reversed
	}
	return reversed
}

func isByteSameAsString(passwd []byte, username string) bool {
	byteUserName := []byte(username)
	return bytes.Equal(passwd, byteUserName)
}

func (a *FuyaoPasswordAuthenticator) fetchUserInfoAndStoredPassword(username string) (user.Info, []byte, error) {
	// fetch user
	userCR, err := fuyaouser.GetUserInfo(a.k8sClient, username)
	if err != nil {
		zlog.LogErrorf("cannot get the password secret for %s", username)
		return nil, nil, fuyaoerrors.ErrPasswordAuthenticationFailed
	}

	// set to userinfo
	var userinfo user.DefaultInfo
	userinfo.Name = username
	userinfo.UID = string(userCR.UID)
	userinfo.Groups = []string{"system:authenticated"}
	encryptedPasswd := userCR.Spec.EncryptedPassword
	firstLogin := userCR.Spec.FirstLogin
	userinfo.Extra = make(map[string][]string)
	userinfo.Extra[constants.UserFirstLogin] = []string{strconv.FormatBool(firstLogin)}
	return &userinfo, encryptedPasswd, nil
}

func (a *FuyaoPasswordAuthenticator) savePassword(username string, passwd []byte, firstLogin bool) error {
	// fetch user
	userCR, err := fuyaouser.GetUserInfo(a.k8sClient, username)
	if err != nil {
		zlog.LogErrorf("cannot get the password secret for %s", username)
		return fuyaoerrors.ErrPasswordAuthenticationFailed
	}

	// check whether first login
	if firstLogin && !userCR.Spec.FirstLogin {
		zlog.LogErrorf("the user has already logged in and changed the password, cannot reconfirm it")
		return fuyaoerrors.ErrNotFirstLogin
	}

	// encrypt the password, set first login to false
	encryptedPassword, err := a.encryptor.EncryptPassword(passwd)
	if err != nil {
		return fuyaoerrors.ErrLoginServiceDown
	}

	patchData := []byte(fmt.Sprintf(`{"spec": {"EncryptedPassword": "%s", "FirstLogin": %t}}`,
		base64.StdEncoding.EncodeToString(encryptedPassword), false))
	if err = fuyaouser.PatchUserInfo(a.k8sClient, username, patchData); err != nil {
		zlog.LogErrorf("cannot save password to user cr, err: %v", err)
		return fuyaoerrors.ErrFailToPatchSecret
	}

	return nil
}

// ConfirmPassword is used when the user first logins in
func (a *FuyaoPasswordAuthenticator) ConfirmPassword(ctx context.Context, username string, newPassword []byte) error {
	// 提取旧的加密密码
	_, base64EncryptedOldPassword, err := a.fetchUserInfoAndStoredPassword(username)
	if err != nil {
		return err
	}

	// check 旧密码是否没有改
	if ok, err := a.encryptor.VerifyPassword(newPassword, base64EncryptedOldPassword); ok || err != nil {
		if err == nil {
			zlog.LogError("password verification failed, err: %v", fuyaoerrors.ErrPasswordSame)
			return fuyaoerrors.ErrPasswordSame
		}
		return err
	}

	// 校验 password 复杂度
	if ok := a.checkPasswordComplexity(username, newPassword); !ok {
		return fuyaoerrors.ErrPasswordTooWeak
	}

	// 存储新密码
	if err := a.savePassword(username, newPassword, true); err != nil {
		return err
	}

	return nil
}

// ResetPassword modifies the user password
func (a *FuyaoPasswordAuthenticator) ResetPassword(
	ctx context.Context,
	username string, oldPassword, newPassword []byte,
) error {
	// 新旧密码不能相同
	if bytes.Equal(oldPassword, newPassword) {
		return fuyaoerrors.ErrPasswordSame
	}

	// 获取加密后密码
	_, base64EncryptedOldPassword, err := a.fetchUserInfoAndStoredPassword(username)
	if err != nil {
		return err
	}

	// check 旧密码是否正确
	if ok, err := a.encryptor.VerifyPassword(oldPassword, base64EncryptedOldPassword); !ok || err != nil {
		if err == nil {
			zlog.LogError("password verification failed")
			return fuyaoerrors.ErrPasswordResetFailed
		}
		return err
	}

	// 校验 password 复杂度
	if ok := a.checkPasswordComplexity(username, newPassword); !ok {
		return fuyaoerrors.ErrPasswordTooWeak
	}

	// 存储新密码
	if err := a.savePassword(username, newPassword, false); err != nil {
		return err
	}

	return nil
}

// Encryptor manages the encrypt and decrypt funcs & config
type Encryptor interface {
	VerifyPassword(newPassword, encryptedOldPassword []byte) (bool, error)
	EncryptPassword(rawPassword []byte) ([]byte, error)
}

// PBKDF2Encryptor is the encryptor + decryptor using PBKDF2 algorithm
type PBKDF2Encryptor struct {
	saltLength    int
	iterations    int
	keyLength     int
	encryptMethod func() hash.Hash
}

// NewPBKDF2Encryptor inits PBKDF2Encryptor
func NewPBKDF2Encryptor() *PBKDF2Encryptor {
	return &PBKDF2Encryptor{
		saltLength:    16,
		iterations:    100000,
		keyLength:     64,
		encryptMethod: sha256.New,
	}
}

// EncryptPassword encrypts the password
func (e *PBKDF2Encryptor) EncryptPassword(rawPassword []byte) ([]byte, error) {
	// 生成随机的盐值
	salt := make([]byte, e.saltLength)
	_, err := rand.Read(salt)
	if err != nil {
		return nil, fuyaoerrors.ErrLoginServiceDown
	}

	// 使用 PBKDF2 算法生成密文
	encryptedPassword := pbkdf2.Key(rawPassword, salt, e.iterations, e.keyLength, e.encryptMethod)

	// 将盐值和密文合并并编码为 Base64 字符串
	encryptedData := append(salt, encryptedPassword...)
	encryptedData = []byte(base64.StdEncoding.EncodeToString(encryptedData))

	// 返回加密后的密码
	return encryptedData, nil
}

// VerifyPassword checks whether the rawPassword can be encrypted to the stored encryptedPassword
func (e *PBKDF2Encryptor) VerifyPassword(rawPassword, encryptedPassword []byte) (bool, error) {
	// decode加密后的密码
	encryptedPassword, err := base64.StdEncoding.DecodeString(string(encryptedPassword))
	if err != nil {
		return false, fuyaoerrors.ErrPasswordAuthenticationFailed
	}

	// 提取盐值和密文
	salt := encryptedPassword[:e.saltLength]
	encryptedPasswordBytes := encryptedPassword[e.saltLength:]

	// 使用相同的盐值和加密算法对原始密码进行加密
	newEncryptedPassword := pbkdf2.Key(rawPassword, salt, e.iterations, e.keyLength, e.encryptMethod)

	// 比较加密后的密码是否相同
	return bytes.Equal(encryptedPasswordBytes, newEncryptedPassword), nil
}
