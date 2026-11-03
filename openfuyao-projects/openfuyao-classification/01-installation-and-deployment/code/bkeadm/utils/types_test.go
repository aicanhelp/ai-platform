/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package utils

import (
	"fmt"
	"testing"
	"time"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

const (
	testZeroValue     = 0
	testOneValue      = 1
	testTwoValue      = 2
	testThreeValue    = 3
	testFourValue     = 4
	testFiveValue     = 5
	testSixValue      = 6
	testTenValue      = 10
	testSixteenValue  = 16
	testTwentyValue   = 20
	testMinusOneValue = -1
)

func TestRetryOptionsStruct(t *testing.T) {
	// Test that the RetryOptions struct has the expected fields
	opts := &RetryOptions{}

	opts.MaxRetry = testThreeValue
	opts.Delay = testFiveValue * time.Second

	assert.Equal(t, testThreeValue, opts.MaxRetry)
	assert.Equal(t, testFiveValue*time.Second, opts.Delay)
}

func TestNewRetryOptions(t *testing.T) {
	tests := []struct {
		name     string
		retry    int
		delay    time.Duration
		expected RetryOptions
	}{
		{
			name:  "normal retry options",
			retry: testThreeValue,
			delay: testFiveValue,
			expected: RetryOptions{
				MaxRetry: testThreeValue,
				Delay:    testFiveValue * time.Second,
			},
		},
		{
			name:  "zero retry and delay",
			retry: testZeroValue,
			delay: testZeroValue,
			expected: RetryOptions{
				MaxRetry: testZeroValue,
				Delay:    testZeroValue,
			},
		},
		{
			name:  "negative retry and delay",
			retry: testMinusOneValue,
			delay: testMinusOneValue,
			expected: RetryOptions{
				MaxRetry: testMinusOneValue,
				Delay:    testMinusOneValue * time.Second,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewRetryOptions(tt.retry, tt.delay)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestRetryOptionsWithDifferentValues(t *testing.T) {
	// Test with various retry and delay values
	testCases := []struct {
		maxRetry int
		delaySec time.Duration
	}{
		{testOneValue, testOneValue},
		{testFiveValue, testTenValue},
		{testTenValue, testOneValue},
		{testZeroValue, testFiveValue},
		{testThreeValue, testZeroValue},
	}

	for _, tc := range testCases {
		opts := NewRetryOptions(tc.maxRetry, tc.delaySec)

		assert.Equal(t, tc.maxRetry, opts.MaxRetry)
		assert.Equal(t, tc.delaySec*time.Second, opts.Delay)
	}
}

func TestRetryOptionsDefaultValues(t *testing.T) {
	// Test that zero values work correctly
	opts := RetryOptions{}

	assert.Equal(t, testZeroValue, opts.MaxRetry)
	assert.Equal(t, time.Duration(testZeroValue), opts.Delay)
}

func TestRetryOptionsAssignment(t *testing.T) {
	// Test assigning values to RetryOptions fields
	opts := RetryOptions{}

	opts.MaxRetry = testFiveValue
	opts.Delay = testTenValue * time.Second

	assert.Equal(t, testFiveValue, opts.MaxRetry)
	assert.Equal(t, testTenValue*time.Second, opts.Delay)

	// Test updating values
	opts.MaxRetry = testTenValue
	opts.Delay = testTwentyValue * time.Second

	assert.Equal(t, testTenValue, opts.MaxRetry)
	assert.Equal(t, testTwentyValue*time.Second, opts.Delay)
}

func TestNewRetryOptionsWithMaxValues(t *testing.T) {
	// Test with maximum possible values
	maxRetry := int(^uint(0) >> 1)             // Max int value
	maxDelay := time.Duration(^uint64(0) >> 1) // Max duration value

	opts := NewRetryOptions(maxRetry, maxDelay)

	assert.Equal(t, maxRetry, opts.MaxRetry)
	assert.Equal(t, maxDelay*time.Second, opts.Delay)
}

func TestNewRetryOptionsWithMinValues(t *testing.T) {
	// Test with minimum values (negative)
	minRetry := -100
	minDelay := time.Duration(-50)

	opts := NewRetryOptions(minRetry, minDelay)

	assert.Equal(t, minRetry, opts.MaxRetry)
	assert.Equal(t, minDelay*time.Second, opts.Delay)
}

func TestRetryOptionsEquality(t *testing.T) {
	// Test equality of RetryOptions structs
	opts1 := NewRetryOptions(testThreeValue, testFiveValue)
	opts2 := NewRetryOptions(testThreeValue, testFiveValue)

	assert.Equal(t, opts1, opts2)

	// Test inequality
	opts3 := NewRetryOptions(testFourValue, testFiveValue)
	assert.NotEqual(t, opts1, opts3)

	opts4 := NewRetryOptions(testThreeValue, testSixValue)
	assert.NotEqual(t, opts1, opts4)
}

func TestRetryOptionsAsParameter(t *testing.T) {
	// Test that RetryOptions can be used as a parameter in functions
	testFunction := func(options RetryOptions) RetryOptions {
		return options
	}

	inputOpts := NewRetryOptions(testTwoValue, testThreeValue)
	outputOpts := testFunction(inputOpts)

	assert.Equal(t, inputOpts, outputOpts)
}

func TestRetryOptionsInSlice(t *testing.T) {
	// Test that RetryOptions can be stored in a slice
	optsSlice := []RetryOptions{
		NewRetryOptions(testOneValue, testOneValue),
		NewRetryOptions(testTwoValue, testTwoValue),
		NewRetryOptions(testThreeValue, testThreeValue),
	}

	assert.Len(t, optsSlice, testThreeValue)
	assert.Equal(t, NewRetryOptions(testOneValue, testOneValue), optsSlice[testZeroValue])
	assert.Equal(t, NewRetryOptions(testTwoValue, testTwoValue), optsSlice[testOneValue])
	assert.Equal(t, NewRetryOptions(testThreeValue, testThreeValue), optsSlice[testTwoValue])
}

func TestRetryOptionsInMap(t *testing.T) {
	// Test that RetryOptions can be used as a map value
	optsMap := map[string]RetryOptions{
		"option1": NewRetryOptions(testOneValue, testOneValue),
		"option2": NewRetryOptions(testTwoValue, testTwoValue),
	}

	assert.Len(t, optsMap, testTwoValue)
	assert.Equal(t, NewRetryOptions(testOneValue, testOneValue), optsMap["option1"])
	assert.Equal(t, NewRetryOptions(testTwoValue, testTwoValue), optsMap["option2"])
}

func TestRetryOptionsZeroValueBehavior(t *testing.T) {
	// Test behavior when using zero-value RetryOptions
	var opts RetryOptions

	assert.Equal(t, testZeroValue, opts.MaxRetry)
	assert.Equal(t, time.Duration(testZeroValue), opts.Delay)

	// Verify that zero-value options can be used without causing issues
	_ = opts.MaxRetry
	_ = opts.Delay

	assert.True(t, true) // Just verify that no panic occurred
}

func TestRetryOptionsWithTimeUnitConversion(t *testing.T) {
	// Test creating RetryOptions with different time units
	opts1 := NewRetryOptions(testThreeValue, testOneValue)                // 1 second
	opts2 := NewRetryOptions(testThreeValue, time.Duration(testOneValue)) // 1 nanosecond * 1 second = 1 second

	assert.Equal(t, testThreeValue, opts1.MaxRetry)
	assert.Equal(t, time.Second, opts1.Delay)

	assert.Equal(t, testThreeValue, opts2.MaxRetry)
	assert.Equal(t, time.Second, opts2.Delay)
}

func TestRetryOptionsMemoryLayout(t *testing.T) {
	// Test that RetryOptions has the expected memory layout
	opts := NewRetryOptions(testFiveValue, testTenValue)

	// Verify that the struct has the expected size (this is implementation-dependent)
	// but we can at least verify that it's a reasonable size
	size := unsafe.Sizeof(opts)
	assert.True(t, size > testZeroValue)
	assert.True(t, size <= testSixteenValue) // Should be reasonably small (2 ints worth of data)
}

func TestRetryOptionsStringRepresentation(t *testing.T) {
	// Although RetryOptions doesn't have a String() method, we can test its representation
	opts := NewRetryOptions(testThreeValue, testFiveValue)

	// Convert to string using fmt package
	str := fmt.Sprintf("%+v", opts)

	assert.Contains(t, str, fmt.Sprintf("MaxRetry:%d", testThreeValue))
	assert.Contains(t, str, fmt.Sprintf("Delay:%ds", testFiveValue))
}

func TestRetryOptionsComparisonWithDifferentDelays(t *testing.T) {
	// Test comparing RetryOptions with different delays
	opts1 := NewRetryOptions(testThreeValue, testFiveValue)
	opts2 := NewRetryOptions(testThreeValue, testFiveValue)
	opts3 := NewRetryOptions(testThreeValue, testSixValue)

	assert.Equal(t, opts1, opts2)
	assert.NotEqual(t, opts1, opts3)
	assert.NotEqual(t, opts2, opts3)
}

func TestRetryOptionsFieldsArePublic(t *testing.T) {
	// Test that the fields of RetryOptions are accessible (public)
	opts := NewRetryOptions(testThreeValue, testFiveValue)

	// Access fields directly to verify they are public
	maxRetry := opts.MaxRetry
	delay := opts.Delay

	assert.Equal(t, testThreeValue, maxRetry)
	assert.Equal(t, testFiveValue*time.Second, delay)

	// Modify fields directly to verify they are writable
	opts.MaxRetry = testTenValue
	opts.Delay = testTwentyValue * time.Second

	assert.Equal(t, testTenValue, opts.MaxRetry)
	assert.Equal(t, testTwentyValue*time.Second, opts.Delay)
}
