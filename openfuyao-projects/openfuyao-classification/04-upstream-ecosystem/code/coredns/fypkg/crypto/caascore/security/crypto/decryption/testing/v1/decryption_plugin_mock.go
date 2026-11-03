//go:build !windows
// +build !windows

// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

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

	"github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption/grpc/v1"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"
)

const (
	// now only supported unix domain socket.
	unixProtocol = "unix"
)

var (
	DefaultKey   = "mock_file.txt"
	DefaultValue = "mock_value"
)

// FileMapPlugin gRPC sever for a mock Decryption provider.
// Uses file name and its content to simulate cipher text plain text.
type FileMapPlugin struct {
	v1.UnimplementedDecryptionServiceServer
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

	v1.RegisterDecryptionServiceServer(server, result)
	return result, nil
}

// Start starts plugins' gRPC service.
func (s *FileMapPlugin) Start(needInitData bool) error {
	var err error
	s.listener, err = net.Listen(unixProtocol, s.socketPath)
	if err != nil {
		return fmt.Errorf("failed to listen on unix socket, error: %v", err)
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
	err := s.listener.Close()
	if err != nil {
		klog.Errorf("Failed to close unix socket, error: %v", err)
	}
	for _, file := range s.keyFiles {
		err = os.Remove(file)
		if err != nil {
			klog.Errorf("Failed to remove file %s, error: %v", file, err)
		}
	}
	if runtime.GOOS != "linux" || !strings.HasPrefix(s.socketPath, "@") {
		err = os.Remove(s.socketPath)
		if err != nil {
			klog.Errorf("Failed to remove file %s, error: %v", s.socketPath, err)
		}
	}
}

// DecryptCert performs read content from file which is the payload of DecryptRequest.
func (s *FileMapPlugin) DecryptCert(ctx context.Context, request *v1.DecryptRequest) (*v1.DecryptResponse, error) {
	return s.decrypt(request)
}

// DecryptSecret performs read content from file which is the payload of DecryptRequest.
func (s *FileMapPlugin) DecryptSecret(ctx context.Context, request *v1.DecryptRequest) (*v1.DecryptResponse, error) {
	return s.decrypt(request)
}

func (s *FileMapPlugin) decrypt(request *v1.DecryptRequest) (*v1.DecryptResponse, error) {
	key := string(request.Ciphertext)
	fmt.Printf("Received Decrypt Request, CipherData = %s", key)

	plainData, err := os.ReadFile(path.Join(s.dataDir, key))
	if s.PrintData {
		println(request, plainData, err, path.Join(s.dataDir, key))
	}
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "decrypt failed")
	}

	return &v1.DecryptResponse{
		Plaintext: plainData,
	}, nil
}

func printRequest(request *v1.DecryptRequest, data []byte, err error, file string) {
	if err == nil {
		fmt.Printf("=========================[success]=============================\n")
		fmt.Printf("[time] %v\n[request] %v\n[response] %v", time.Now(), string(request.Ciphertext), string(data))
	} else {
		fmt.Printf("=========================[failed!]=============================\n")
		fmt.Printf("[time] %v\n[request] %v\n[response] %v", time.Now(), string(request.Ciphertext), string(data))
	}
}

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
