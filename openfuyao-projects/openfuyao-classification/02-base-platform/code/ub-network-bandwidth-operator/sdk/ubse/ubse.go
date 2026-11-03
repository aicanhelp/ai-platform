/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package ubse implements the local UBSE IPC protocol used for bandwidth QoS.
package ubse

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"time"
)

const (
	ubseModuleCode    = 0x0005
	ubseUrmaQosCreate = 0x0001
	ubseUrmaQosDelete = 0x0002
	ubseUrmaQosGet    = 0x000A

	ubseSuccess = 0

	// BandwidthConfigExistsCode means the UBSE bandwidth configuration already exists.
	BandwidthConfigExistsCode = 11101
	// BandwidthGetOVSInterfaceMissingCode means UBSE found an invalid QoS runtime state on get.
	BandwidthGetOVSInterfaceMissingCode = 11102
	// BandwidthGetOVSInterfaceInvalidCode means UBSE found an invalid OVS interface state on get.
	BandwidthGetOVSInterfaceInvalidCode = 11103

	ubseIPCSocketPath = "/var/run/ubse/ubse.sock"
	maxMessageSize    = 10 * 1024 * 1024
	defaultTimeout    = 30 * time.Second

	bandwidthMinCount = 1
	bandwidthMaxCount = 2

	bandwidthMinPriority = 0
	bandwidthMaxPriority = 1
	bandwidthCountSize   = 4
	bandwidthItemSize    = 8
	bandwidthValueOffset = 4

	messageTypeRequest   byte = 0
	messageTypeResponse  byte = 1
	messageFlagSize           = 1
	ubseHeaderSize            = 16
	requestModuleOffset       = 1
	requestOpOffset           = 3
	requestLengthOffset       = 5
	requestBodyOffset         = 17
	responseStatusOffset      = 1
	responseLengthOffset      = 5
)

// BandwidthItem represents one URMA QoS bandwidth item in the daemon protocol.
type BandwidthItem struct {
	Priority  uint32
	Bandwidth uint32
}

// DaemonError carries a UBSE daemon status code returned in the response header.
type DaemonError struct {
	Code uint32
}

func (e *DaemonError) Error() string {
	return fmt.Sprintf("ubse daemon returned error: %d", e.Code)
}

// IsDaemonErrorCode reports whether err wraps a UBSE daemon status code.
func IsDaemonErrorCode(err error, code uint32) bool {
	var daemonErr *DaemonError
	if !errors.As(err, &daemonErr) {
		return false
	}
	if daemonErr == nil {
		return false
	}
	return daemonErr.Code == code
}

// BandwidthSet sets URMA QoS bandwidth items through the local UBSE daemon.
func BandwidthSet(items []BandwidthItem) error {
	request, err := packBandwidthItems(items)
	if err != nil {
		return err
	}
	_, err = invokeCall(ubseModuleCode, ubseUrmaQosCreate, request)
	return err
}

// BandwidthReset clears URMA QoS bandwidth items through the local UBSE daemon.
func BandwidthReset() error {
	request := packBandwidthResetRequest()
	_, err := invokeCall(ubseModuleCode, ubseUrmaQosDelete, request)
	return err
}

// BandwidthGet queries URMA QoS bandwidth items through the local UBSE daemon.
func BandwidthGet() ([]BandwidthItem, error) {
	request := packBandwidthGetRequest()
	response, err := invokeCall(ubseModuleCode, ubseUrmaQosGet, request)
	if err != nil {
		return nil, err
	}
	return unpackBandwidthItems(response)
}

func packBandwidthItems(items []BandwidthItem) ([]byte, error) {
	if err := validateBandwidthCount(len(items)); err != nil {
		return nil, err
	}

	body := make([]byte, bandwidthCountSize+len(items)*bandwidthItemSize)
	binary.LittleEndian.PutUint32(body[0:], uint32(len(items)))
	offset := bandwidthCountSize
	seenPriorities := make(map[uint32]struct{}, len(items))
	for _, item := range items {
		if err := validateBandwidthPriority(item.Priority, seenPriorities); err != nil {
			return nil, err
		}
		binary.LittleEndian.PutUint32(body[offset:], item.Priority)
		binary.LittleEndian.PutUint32(body[offset+bandwidthValueOffset:], item.Bandwidth)
		offset += bandwidthItemSize
	}
	return body, nil
}

func packBandwidthResetRequest() []byte {
	return nil
}

func packBandwidthGetRequest() []byte {
	return nil
}

func unpackBandwidthItems(body []byte) ([]BandwidthItem, error) {
	if len(body) < bandwidthCountSize {
		return nil, fmt.Errorf("bandwidth query response too short: got %d bytes", len(body))
	}

	count := binary.LittleEndian.Uint32(body[0:])
	if count > bandwidthMaxCount {
		return nil, fmt.Errorf("bandwidth item count must be <= %d, got %d", bandwidthMaxCount, count)
	}
	expectedLen := bandwidthCountSize + int(count)*bandwidthItemSize
	if len(body) < expectedLen {
		return nil, fmt.Errorf("bandwidth query response too short: got %d bytes, want %d", len(body), expectedLen)
	}

	items := make([]BandwidthItem, 0, count)
	offset := bandwidthCountSize
	seenPriorities := make(map[uint32]struct{}, count)
	for i := uint32(0); i < count; i++ {
		item := BandwidthItem{
			Priority:  binary.LittleEndian.Uint32(body[offset:]),
			Bandwidth: binary.LittleEndian.Uint32(body[offset+bandwidthValueOffset:]),
		}
		if err := validateBandwidthPriority(item.Priority, seenPriorities); err != nil {
			return nil, err
		}
		items = append(items, item)
		offset += bandwidthItemSize
	}
	return items, nil
}

func validateBandwidthCount(count int) error {
	if count < bandwidthMinCount || count > bandwidthMaxCount {
		return fmt.Errorf("bandwidth item count must be in range [%d,%d], got %d",
			bandwidthMinCount, bandwidthMaxCount, count)
	}
	return nil
}

func validateBandwidthPriority(priority uint32, seenPriorities map[uint32]struct{}) error {
	if priority < bandwidthMinPriority || priority > bandwidthMaxPriority {
		return fmt.Errorf("bandwidth priority must be 0 or 1, got %d", priority)
	}
	if seenPriorities == nil {
		return fmt.Errorf("seen priorities map must not be nil")
	}
	if _, ok := seenPriorities[priority]; ok {
		return fmt.Errorf("duplicate bandwidth priority: %d", priority)
	}
	seenPriorities[priority] = struct{}{}
	return nil
}

func invokeCall(moduleCode, opCode uint16, request []byte) ([]byte, error) {
	conn, err := connectToUnixSocket()
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	if err := sendRequest(conn, moduleCode, opCode, request); err != nil {
		return nil, err
	}
	return receiveResponse(conn)
}

func connectToUnixSocket() (net.Conn, error) {
	if _, err := os.Stat(ubseIPCSocketPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("ubse daemon not running: %w", err)
	}

	conn, err := net.Dial("unix", ubseIPCSocketPath)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to ubse daemon: %w", err)
	}

	if err := conn.SetDeadline(time.Now().Add(defaultTimeout)); err != nil {
		if closeErr := conn.Close(); closeErr != nil {
			return nil, fmt.Errorf(
				"failed to set ubse socket deadline: %w; failed to close ubse socket: %v",
				err,
				closeErr,
			)
		}
		return nil, fmt.Errorf("failed to set ubse socket deadline: %w", err)
	}
	return conn, nil
}

func sendRequest(conn net.Conn, moduleCode, opCode uint16, request []byte) error {
	message := make([]byte, messageFlagSize+ubseHeaderSize+len(request))
	message[0] = messageTypeRequest
	binary.LittleEndian.PutUint16(message[requestModuleOffset:], moduleCode)
	binary.LittleEndian.PutUint16(message[requestOpOffset:], opCode)
	binary.LittleEndian.PutUint32(message[requestLengthOffset:], uint32(len(request)))
	copy(message[requestBodyOffset:], request)

	written := 0
	for written < len(message) {
		n, err := conn.Write(message[written:])
		if err != nil {
			return fmt.Errorf("failed to send request message: %w", err)
		}
		if n == 0 {
			return fmt.Errorf("failed to send request message: %w", io.ErrShortWrite)
		}
		written += n
	}
	return nil
}

func receiveResponse(conn net.Conn) ([]byte, error) {
	header := make([]byte, messageFlagSize+ubseHeaderSize)
	if _, err := io.ReadFull(conn, header); err != nil {
		return nil, fmt.Errorf("failed to read response message header: %w", err)
	}

	if header[0] != messageTypeResponse {
		return nil, fmt.Errorf("invalid response message: isResp flag is not set")
	}

	statusCode := binary.LittleEndian.Uint32(header[responseStatusOffset:])
	responseLen := binary.LittleEndian.Uint32(header[responseLengthOffset:])
	if statusCode != ubseSuccess {
		return nil, &DaemonError{Code: statusCode}
	}
	if responseLen > maxMessageSize {
		return nil, fmt.Errorf("response body length %d exceeds maximum size %d", responseLen, maxMessageSize)
	}

	response := make([]byte, responseLen)
	if _, err := io.ReadFull(conn, response); err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	return response, nil
}
