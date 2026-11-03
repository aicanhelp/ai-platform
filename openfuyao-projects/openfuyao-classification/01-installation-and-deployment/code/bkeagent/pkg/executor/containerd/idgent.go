package containerd

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
)

const (
	// IDLength specifies the length of the hexadecimal string ID
	IDLength = 64
)

// GenerateID creates a secure random hexadecimal string of IDLength characters
func GenerateID() string {
	// Calculate the required number of random bytes
	// Since each hex character represents 4 bits, we need half as many bytes
	bytesLength := IDLength / 2

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
