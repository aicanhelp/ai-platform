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

package ubse

import (
	"encoding/binary"
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestPackBandwidthItems(t *testing.T) {
	items := []BandwidthItem{
		{Priority: 0, Bandwidth: 25000},
		{Priority: 1, Bandwidth: 50000},
	}

	body, err := packBandwidthItems(items)
	assert.NoError(t, err)

	assert.Len(t, body, bandwidthCountSize+len(items)*bandwidthItemSize)
	assert.Equal(t, uint32(2), binary.LittleEndian.Uint32(body[0:]))
	assert.Equal(t, uint32(0), binary.LittleEndian.Uint32(body[4:]))
	assert.Equal(t, uint32(25000), binary.LittleEndian.Uint32(body[8:]))
	assert.Equal(t, uint32(1), binary.LittleEndian.Uint32(body[12:]))
	assert.Equal(t, uint32(50000), binary.LittleEndian.Uint32(body[16:]))
}

func TestPackBandwidthItemsRejectsInvalidInput(t *testing.T) {
	cases := [][]BandwidthItem{
		nil,
		{{Priority: 2, Bandwidth: 10000}},
		{{Priority: 0, Bandwidth: 10000}, {Priority: 0, Bandwidth: 20000}},
		{{Priority: 0, Bandwidth: 10000}, {Priority: 1, Bandwidth: 20000}, {Priority: 2, Bandwidth: 30000}},
	}

	for _, items := range cases {
		_, err := packBandwidthItems(items)
		assert.Error(t, err)
	}
}

func TestPackBandwidthResetRequestIsEmpty(t *testing.T) {
	body := packBandwidthResetRequest()
	assert.Empty(t, body)
}

func TestPackBandwidthGetRequestIsEmpty(t *testing.T) {
	body := packBandwidthGetRequest()
	assert.Empty(t, body)
}

func TestUnpackBandwidthItemsParsesQueryResponse(t *testing.T) {
	body := make([]byte, bandwidthCountSize+2*bandwidthItemSize)
	binary.LittleEndian.PutUint32(body[0:], 2)
	binary.LittleEndian.PutUint32(body[4:], 1)
	binary.LittleEndian.PutUint32(body[8:], 50)
	binary.LittleEndian.PutUint32(body[12:], 0)
	binary.LittleEndian.PutUint32(body[16:], 25)

	items, err := unpackBandwidthItems(body)

	assert.NoError(t, err)
	assert.Equal(t, []BandwidthItem{
		{Priority: 1, Bandwidth: 50},
		{Priority: 0, Bandwidth: 25},
	}, items)
}

func TestUnpackBandwidthItemsAllowsEmptyQueryResponse(t *testing.T) {
	body := make([]byte, bandwidthCountSize)
	binary.LittleEndian.PutUint32(body[0:], 0)

	items, err := unpackBandwidthItems(body)

	assert.NoError(t, err)
	assert.Empty(t, items)
}

func TestUnpackBandwidthItemsRejectsShortBody(t *testing.T) {
	body := make([]byte, bandwidthCountSize+bandwidthItemSize-1)
	binary.LittleEndian.PutUint32(body[0:], 1)

	_, err := unpackBandwidthItems(body)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "too short")
}

func TestUnpackBandwidthItemsRejectsInvalidResponses(t *testing.T) {
	tests := []struct {
		name string
		body []byte
	}{
		{
			name: "count exceeds max",
			body: func() []byte {
				body := make([]byte, bandwidthCountSize)
				binary.LittleEndian.PutUint32(body[0:], 3)
				return body
			}(),
		},
		{
			name: "invalid priority",
			body: func() []byte {
				body := make([]byte, bandwidthCountSize+bandwidthItemSize)
				binary.LittleEndian.PutUint32(body[0:], 1)
				binary.LittleEndian.PutUint32(body[4:], 2)
				binary.LittleEndian.PutUint32(body[8:], 25)
				return body
			}(),
		},
		{
			name: "duplicate priority",
			body: func() []byte {
				body := make([]byte, bandwidthCountSize+2*bandwidthItemSize)
				binary.LittleEndian.PutUint32(body[0:], 2)
				binary.LittleEndian.PutUint32(body[4:], 0)
				binary.LittleEndian.PutUint32(body[8:], 25)
				binary.LittleEndian.PutUint32(body[12:], 0)
				binary.LittleEndian.PutUint32(body[16:], 50)
				return body
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := unpackBandwidthItems(tt.body)
			assert.Error(t, err)
		})
	}
}

func TestBandwidthSetReturnsDaemonErrorAfterPacking(t *testing.T) {
	err := BandwidthSet([]BandwidthItem{{Priority: 0, Bandwidth: 25000}, {Priority: 1, Bandwidth: 50000}})
	assert.Error(t, err)
}

func TestBandwidthSetRejectsInvalidItems(t *testing.T) {
	err := BandwidthSet(nil)
	assert.Error(t, err)
}

func TestBandwidthResetReturnsDaemonErrorAfterPacking(t *testing.T) {
	err := BandwidthReset()
	assert.Error(t, err)
}

func TestBandwidthGetReturnsDaemonErrorAfterPacking(t *testing.T) {
	_, err := BandwidthGet()
	assert.Error(t, err)
}

func TestInvokeCallConnectionFailure(t *testing.T) {
	_, err := invokeCall(ubseModuleCode, ubseUrmaQosCreate, []byte{1, 2, 3})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ubse daemon not running")
}

func TestConnectToUnixSocketNoDaemon(t *testing.T) {
	_, err := connectToUnixSocket()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ubse daemon not running")
}

type mockConn struct {
	readData    []byte
	readPos     int
	writeData   []byte
	writeError  bool
	readError   bool
	closeCalled bool
}

func (m *mockConn) Read(b []byte) (n int, err error) {
	if m.readError {
		return 0, errors.New("mock read error")
	}
	if m.readPos >= len(m.readData) {
		return 0, io.EOF
	}
	n = copy(b, m.readData[m.readPos:])
	m.readPos += n
	return n, nil
}

func (m *mockConn) Write(b []byte) (n int, err error) {
	if m.writeError {
		return 0, errors.New("mock write error")
	}
	m.writeData = append(m.writeData, b...)
	return len(b), nil
}

func (m *mockConn) Close() error {
	m.closeCalled = true
	return nil
}

func (m *mockConn) LocalAddr() net.Addr              { return &net.UnixAddr{Name: "/mock"} }
func (m *mockConn) RemoteAddr() net.Addr             { return &net.UnixAddr{Name: "/mock"} }
func (m *mockConn) SetDeadline(time.Time) error      { return nil }
func (m *mockConn) SetReadDeadline(time.Time) error  { return nil }
func (m *mockConn) SetWriteDeadline(time.Time) error { return nil }

func TestSendRequest(t *testing.T) {
	conn := &mockConn{}
	err := sendRequest(conn, ubseModuleCode, ubseUrmaQosCreate, []byte{1, 2, 3})
	assert.NoError(t, err)
	assert.Len(t, conn.writeData, messageFlagSize+ubseHeaderSize+3)
}

func TestSendRequestPartialWrite(t *testing.T) {
	conn := &mockConn{writeError: true}
	err := sendRequest(conn, ubseModuleCode, ubseUrmaQosCreate, []byte{1, 2, 3})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to send request")
}

func TestReceiveResponse(t *testing.T) {
	response := make([]byte, messageFlagSize+ubseHeaderSize)
	response[0] = messageTypeResponse
	binary.LittleEndian.PutUint32(response[responseStatusOffset:], ubseSuccess)
	binary.LittleEndian.PutUint32(response[responseLengthOffset:], 0)

	conn := &mockConn{readData: response}
	result, err := receiveResponse(conn)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestReceiveResponseError(t *testing.T) {
	conn := &mockConn{readError: true}
	_, err := receiveResponse(conn)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to read response")
}

func TestReceiveResponseInvalidFlag(t *testing.T) {
	response := make([]byte, messageFlagSize+ubseHeaderSize)
	response[0] = messageTypeRequest
	binary.LittleEndian.PutUint32(response[responseStatusOffset:], ubseSuccess)
	binary.LittleEndian.PutUint32(response[responseLengthOffset:], 0)

	conn := &mockConn{readData: response}
	_, err := receiveResponse(conn)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "isResp flag is not set")
}

func TestReceiveResponseErrorStatus(t *testing.T) {
	response := make([]byte, messageFlagSize+ubseHeaderSize)
	response[0] = messageTypeResponse
	binary.LittleEndian.PutUint32(response[responseStatusOffset:], 1)
	binary.LittleEndian.PutUint32(response[responseLengthOffset:], 0)

	conn := &mockConn{readData: response}
	_, err := receiveResponse(conn)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ubse daemon returned error")

	var daemonErr *DaemonError
	assert.ErrorAs(t, err, &daemonErr)
	assert.NotNil(t, daemonErr)
	assert.Equal(t, uint32(1), daemonErr.Code)
	assert.True(t, IsDaemonErrorCode(err, 1))
	assert.False(t, IsDaemonErrorCode(err, BandwidthConfigExistsCode))
}

func TestReceiveResponseExceedsMaxSize(t *testing.T) {
	response := make([]byte, messageFlagSize+ubseHeaderSize)
	response[0] = messageTypeResponse
	binary.LittleEndian.PutUint32(response[responseStatusOffset:], ubseSuccess)
	binary.LittleEndian.PutUint32(response[responseLengthOffset:], maxMessageSize+1)

	conn := &mockConn{readData: response}
	_, err := receiveResponse(conn)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds maximum size")
}

func TestReceiveResponseWithBody(t *testing.T) {
	body := []byte{1, 2, 3, 4, 5}
	response := make([]byte, messageFlagSize+ubseHeaderSize+len(body))
	response[0] = messageTypeResponse
	binary.LittleEndian.PutUint32(response[responseStatusOffset:], ubseSuccess)
	binary.LittleEndian.PutUint32(response[responseLengthOffset:], uint32(len(body)))
	copy(response[requestBodyOffset:], body)

	conn := &mockConn{readData: response}
	result, err := receiveResponse(conn)
	assert.NoError(t, err)
	assert.Equal(t, body, result)
}

func TestConstants(t *testing.T) {
	assert.Equal(t, 0x0005, int(ubseModuleCode))
	assert.Equal(t, 0x0001, int(ubseUrmaQosCreate))
	assert.Equal(t, 0x0002, int(ubseUrmaQosDelete))
	assert.Equal(t, 0x000A, int(ubseUrmaQosGet))
	assert.Equal(t, 0, int(ubseSuccess))
	assert.Equal(t, 11101, int(BandwidthConfigExistsCode))
	assert.Equal(t, 11102, int(BandwidthGetOVSInterfaceMissingCode))
	assert.Equal(t, 11103, int(BandwidthGetOVSInterfaceInvalidCode))
	assert.Equal(t, byte(0), messageTypeRequest)
	assert.Equal(t, byte(1), messageTypeResponse)
	assert.Equal(t, 4, bandwidthCountSize)
	assert.Equal(t, 8, bandwidthItemSize)
	assert.Equal(t, 17, messageFlagSize+ubseHeaderSize)
}
