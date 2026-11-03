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

package params

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"

	"installer/pkg/constants"
	"installer/pkg/utils/celutil"
	"installer/pkg/zlog"
)

// VerifyZone is verify zone
func VerifyZone(name, value string) (bool, error) {
	expression := fmt.Sprintf(`%s == "cn" || %s == "nocn"`, name, name)

	params := map[string]interface{}{
		name: value,
	}

	envOption := cel.Declarations(
		decls.NewIdent(name, decls.String, nil),
	)

	return celutil.VerifyParams(expression, params, envOption)
}

// VerifyDomain 校验域名是否合法
func VerifyDomain(domain, value string) (bool, error) {
	expression := fmt.Sprintf(`%s.matches("(http(s)?://)?([a-zA-Z0-9-]+\\.)+[a-zA-Z]{2,}$")`, domain)
	params := map[string]interface{}{
		domain: value,
	}

	envOption := cel.Declarations(
		decls.NewIdent(domain, decls.String, nil),
	)

	return celutil.VerifyParams(expression, params, envOption)
}

// VerifyPasswd is verify passwd
func VerifyPasswd(passwd []byte) (bool, error) {
	if len(passwd) < constants.PasswordMinLen || len(passwd) > constants.PasswordMaxLen {
		return false, fmt.Errorf("the password length should lie between 8 and 32")
	}

	// check that the password at least contains one lowercase/uppercase letter, one number and one special character
	reUpperCase := regexp.MustCompile(`[A-Z]`)
	reLowerCase := regexp.MustCompile(`[a-z]`)
	reDigit := regexp.MustCompile(`[0-9]`)
	reSpecialChar := regexp.MustCompile(`[!\"#$%&'()*+,-./:;<=>?@[\]^_{|}~ ]`)

	if (!reUpperCase.Match(passwd) && !reLowerCase.Match(passwd)) || !reDigit.Match(passwd) ||
		!reSpecialChar.Match(passwd) {
		return false, fmt.Errorf("password must contain at least one lowercase letter or one uppercase letter, " +
			"one number, and one special character")
	}
	return true, nil
}

// VerifyRegistryCertExpiryDays is verify registry cert expiry days
func VerifyRegistryCertExpiryDays(days string) error {
	if days == "" {
		return fmt.Errorf("registry cert expiry days is empty")
	}

	daysInt, err := strconv.Atoi(days)
	if err != nil {
		return fmt.Errorf("registry cert expiry days is not a valid integer")
	}
	if daysInt < constants.RegistryCertExpiryDaysMin || daysInt > constants.RegistryCertExpiryDaysMax {
		return fmt.Errorf("registry cert expiry days should be between %d and %d",
			constants.RegistryCertExpiryDaysMin, constants.RegistryCertExpiryDaysMax)
	}
	return nil
}

// VerifyArtifactPath is verify artifact path
func VerifyArtifactPath(artifactPath string) error {
	if artifactPath == "" {
		return fmt.Errorf("artifact path is empty")
	}

	_, err := os.Stat(artifactPath)
	if err != nil && os.IsNotExist(err) {
		return fmt.Errorf("artifact path is not exist")
	} else if err != nil {
		return fmt.Errorf("artifact path is invalid")
	} else {
		zlog.Info("artifact path is valid")
	}

	if !strings.HasSuffix(artifactPath, constants.TarGz) {
		return fmt.Errorf("artifact path is not a tar.gz file")
	}

	return nil
}
