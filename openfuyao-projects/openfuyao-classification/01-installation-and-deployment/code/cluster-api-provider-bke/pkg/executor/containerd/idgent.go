/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package containerd

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
)

const (
	// IDLength specifies the length of the hexadecimal string ID
	IDLength = 64
	// HexCharsPerByte represents the number of hexadecimal characters per byte
	HexCharsPerByte = 2
)

// GenerateID creates a secure random hexadecimal string of IDLength characters
func GenerateID() string {
	// Calculate the required number of random bytes
	// Since each hex character represents 4 bits, we need half as many bytes
	bytesLength := IDLength / HexCharsPerByte

	// Create buffer for random bytes
	randomBytes := make([]byte, bytesLength)

	// Read full random data from crypto/rand
	if _, err := rand.Read(randomBytes); err != nil {
		panic(err)
	}

	// Encode to hexadecimal string
	id := hex.EncodeToString(randomBytes)

	// Verify the length matches our expectation
	if len(id) != IDLength {
		panic(fmt.Errorf("unexpected ID length: expected %d, got %d", IDLength, len(id)))
	}

	return id
}
