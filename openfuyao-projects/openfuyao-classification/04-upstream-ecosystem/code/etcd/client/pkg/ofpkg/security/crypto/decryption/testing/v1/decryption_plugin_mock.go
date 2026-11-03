// Package v1
package v1

import (
	"context"
	"fmt"
	"net"
	"os"
	"path"
	"runtime"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	decryptapi "go.etcd.io/etcd/client/pkg/v3/ofpkg/security/crypto/decryption/grpc/v1"
)

const (
	// Now only supported unix domain socket
	unixProtocol = "unix"
)

const (
	// DefaultKey 默认的密钥文件名
	DefaultKey = "mock_file.txt"

	// DefaultValue 默认的模拟数据值
	DefaultValue = "mock_value"
)

// FileMapPlugin gRPC server for a mock Decryption provider.
// Uses file name and its content to simulate cipher text and plain text
type FileMapPlugin struct {
	PrintData  bool
	grpcServer *grpc.Server
	listener   net.Listener
	socketPath string
	dataDir    string
	keyFiles   []string
}

// NewFileMapPlugin is a constructor for FileMapPlugin.
func NewFileMapPlugin(socketPath, dataDir string) (*FileMapPlugin, error) {
	server := grpc.NewServer()
	result := &FileMapPlugin{
		grpcServer: server,
		socketPath: socketPath,
		dataDir:    dataDir,
	}

	decryptapi.RegisterDecryptionServiceServer(server, result)
	return result, nil
}

// Start starts plugin's gRPC service.
func (s *FileMapPlugin) Start(needInitData bool) error {
	var err error
	s.listener, err = net.Listen(unixProtocol, s.socketPath)
	if err != nil {
		return fmt.Errorf("failed to listen on the unix socket, error: %v", err)
	}
	fmt.Printf("Listening on %s", s.socketPath)

	if needInitData {
		err = s.PushMockData(map[string][]byte{DefaultKey: []byte(DefaultValue)})
		if err != nil {
			return fmt.Errorf("failed to init default data, error: %v", err)
		}
	}

	go s.grpcServer.Serve(s.listener)
	return nil
}

// CleanUp stops gRPC server and the underlying listener.
func (s *FileMapPlugin) CleanUp() {
	s.grpcServer.Stop()
	_ = s.listener.Close()
	for _, file := range s.keyFiles {
		_ = os.Remove(file)
	}
	if !strings.HasPrefix(s.socketPath, "@") || runtime.GOOS != "linux" {
		_ = os.Remove(s.socketPath)
	}
}

// DecryptCert performs read content from file which is the payload of DecryptRequest.
func (s *FileMapPlugin) DecryptCert(ctx context.Context,
	request *decryptapi.DecryptRequest) (*decryptapi.DecryptResponse, error) {
	return s.decrypt(request)
}

// DecryptSecret performs read content from file which is the payload of DecryptRequest.
func (s *FileMapPlugin) DecryptSecret(ctx context.Context,
	request *decryptapi.DecryptRequest) (*decryptapi.DecryptResponse, error) {
	return s.decrypt(request)
}

func (s *FileMapPlugin) decrypt(request *decryptapi.DecryptRequest) (*decryptapi.DecryptResponse, error) {
	key := string(request.Ciphertext)
	fmt.Printf("Received Decrypt Request, CipherData = %s", key)

	plainData, err := os.ReadFile(path.Join(s.dataDir, key))
	if s.PrintData {
		printRequest(request, plainData, err, path.Join(s.dataDir, key))
	}
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "decrypt failed")
	}

	return &decryptapi.DecryptResponse{
		Plaintext: plainData,
	}, nil
}

func printRequest(request *decryptapi.DecryptRequest, data []byte, err error, file string) {
	if err == nil {
		fmt.Printf("=========================[success]=============================\n")
		fmt.Printf("[time] %v\n[request] %v\n[response] %v", time.Now(), string(request.Ciphertext), string(data))
	} else {
		fmt.Printf("=========================[failed!]=============================\n")
		fmt.Printf("[time] %v\n[request] %v\n[error] open fiel failed: %v", time.Now(), string(request.Ciphertext),
			file)
	}
}

// PushMockData 推送模拟数据到文件映射插件
func (s *FileMapPlugin) PushMockData(kvData map[string][]byte) error {
	newKeyFiles := make([]string, 0, len(s.keyFiles)+len(kvData))
	for name, value := range kvData {
		dataFile := path.Join(s.dataDir, name)
		if err := os.WriteFile(dataFile, value, 0644); err != nil {
			return fmt.Errorf("error while writing data: %v", err)
		}
		newKeyFiles = append(newKeyFiles, dataFile)
	}
	s.keyFiles = newKeyFiles
	return nil
}
