/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package server

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/collector"
	"huawei.com/xpu-exporter/common/limiter"
)

type ProtocolType int

const (
	HTTP ProtocolType = iota
	HTTPS
	portMin                = 1025
	portMax                = 40000
	timeout                = 10
	maxHeaderBytes         = 3072
	maxIPConnLimit         = 128
	maxConcurrency         = 512
	defaultShutDownTimeout = 30 * time.Second
	certFilePath           = "/opt/xpu/certs/"
)

type ExporterServer struct {
	Ip             string
	Port           int
	Concurrency    int
	LimitIPReq     string
	LimitIPConn    int
	LimitTotalConn int
	ProtocolType   ProtocolType
	collectService collector.ICollectorService
	cert           *tls.Certificate
}

func indexHandler(s *ExporterServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s.indexHandler(w, r)
	}
}

func (s *ExporterServer) indexHandler(w http.ResponseWriter, _ *http.Request) {
	var protocol = "http"
	if s.ProtocolType == HTTPS {
		protocol = "https"
	}
	_, err := w.Write([]byte(
		`<html>
			 <head><title>XPU-Exporter</title></head>
			 <body>
			 <h1 align="center">XPU-Exporter</h1>
			 <p align="center">Welcome to use XPU-Exporter,the Prometheus metrics url is ` + protocol + `://IP:` +
			strconv.Itoa(s.Port) + `/metrics: <a href="./metrics">Metrics</a></p>
			 </body>
			 </html>`))
	if err != nil {
		log.Errorf("Write to response error: %v", err)
	}
}

func (s *ExporterServer) VerifyServerParams() error {
	if s.Port < portMin || s.Port > portMax {
		return errors.New("the Port is invalid")
	}
	parsedIP := net.ParseIP(s.Ip)
	if parsedIP == nil {
		return errors.New("the listen Ip is invalid")
	}
	s.Ip = parsedIP.String()
	log.Infof("listen on: %s", s.Ip)

	reg := regexp.MustCompile(limiter.IPReqLimitReg)
	if !reg.Match([]byte(s.LimitIPReq)) {
		return errors.New("limitIPReq format error")
	}
	if s.LimitIPConn < 1 || s.LimitIPConn > maxIPConnLimit {
		return errors.New("limitIPConn is invalid")
	}
	if s.LimitTotalConn < 1 || s.LimitTotalConn > maxConcurrency {
		return errors.New("limitTotalConn is invalid")
	}
	if s.Concurrency < 1 || s.Concurrency > maxConcurrency {
		return errors.New("concurrency is invalid")
	}

	if os.Getenv("HTTPS_ENABLE") == "on" {
		s.ProtocolType = HTTPS
		certPath, err := getCertsFile(".crt")
		if err != nil {
			return fmt.Errorf("HTTPS certificate error: %v", err)
		}
		keyPath, err := getCertsFile(".key")
		if err != nil {
			return fmt.Errorf("HTTPS key error: %v", err)
		}
		cert, err := tls.LoadX509KeyPair(certPath, keyPath)
		if err != nil {
			return fmt.Errorf("HTTPS key pair error: %v", err)
		}
		s.cert = &cert
	} else {
		s.ProtocolType = HTTP
		log.Infoln("The HTTPS service is not enabled, using HTTP")
	}

	return nil
}

func getTLSConfig() *tls.Config {
	return &tls.Config{
		MinVersion:               tls.VersionTLS13,
		PreferServerCipherSuites: true,
		InsecureSkipVerify:       false,
	}
}

func getCertsFile(suffix string) (string, error) {
	files, err := os.ReadDir(certFilePath)
	if err != nil {
		return "", err
	}
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		if strings.HasSuffix(file.Name(), suffix) {
			return filepath.Join(certFilePath, file.Name()), nil
		}
	}
	return "", fmt.Errorf("no files with the suffix '%s' found in directory: %s", suffix, certFilePath)
}

func (s *ExporterServer) initConfig() *limiter.HandlerConfig {
	conf := &limiter.HandlerConfig{
		PrintLog:         true,
		Method:           http.MethodGet,
		LimitBytes:       limiter.DefaultDataLimit,
		TotalConCurrency: s.Concurrency,
		IPConCurrency:    s.LimitIPReq,
		CacheSize:        limiter.DefaultCacheSize,
	}
	return conf
}

func (s *ExporterServer) newServer(conf *limiter.HandlerConfig) (*http.Server, net.Listener, error) {
	handler, err := limiter.NewLimitHandler(http.DefaultServeMux, conf)
	if err != nil {
		return nil, nil, err
	}

	server := &http.Server{
		Addr:           s.Ip + ":" + strconv.Itoa(s.Port),
		Handler:        handler,
		ReadTimeout:    timeout * time.Second,
		WriteTimeout:   timeout * time.Second,
		MaxHeaderBytes: maxHeaderBytes,
	}

	if s.ProtocolType == HTTPS {
		server.TLSConfig = getTLSConfig()
		server.TLSConfig.GetCertificate = func(*tls.ClientHelloInfo) (*tls.Certificate, error) {
			if s.cert == nil {
				return nil, fmt.Errorf("certificate not loaded")
			}
			return s.cert, nil
		}
	}

	ln, err := net.Listen("tcp", server.Addr)
	if err != nil {
		return nil, nil, fmt.Errorf("listen IP and port error: %v", err)
	}

	limitLs, err := limiter.LimitListener(ln, s.LimitTotalConn, s.LimitIPConn, limiter.DefaultCacheSize)
	if err != nil {
		return nil, nil, fmt.Errorf("limit listener error: %v", err)
	}

	return server, limitLs, nil
}

// StartServe initializes and starts the HTTP/HTTPS metrics server
func (s *ExporterServer) StartServe(ctx context.Context, cancel context.CancelFunc, reg *prometheus.Registry) {
	http.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{ErrorHandling: promhttp.ContinueOnError}))
	http.Handle("/", indexHandler(s))

	conf := s.initConfig()
	server, listener, err := s.newServer(conf)
	if err != nil {
		log.Errorf("Failed to create server: %v", err)
		cancel()
		return
	}

	go func() {
		var err error
		if s.ProtocolType == HTTPS {
			log.Infof("Starting HTTPS server on %s", server.Addr)
			err = server.ServeTLS(listener, "", "")
		} else {
			log.Infof("Starting HTTP server on %s", server.Addr)
			err = server.Serve(listener)
		}

		if err != nil && err != http.ErrServerClosed {
			log.Errorf("Server error: %v and stopped", err)
			cancel()
		}
	}()

	<-ctx.Done()
	log.Infoln("Received stop signal, shutting down server")

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), defaultShutDownTimeout)
	defer cancelShutdown()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Errorf("Server shutdown error: %v", err)
	}
}

// RegisterCollectorService registers a collector service with the exporter server
func (s *ExporterServer) RegisterCollectorService(c collector.ICollectorService) error {
	if s == nil {
		return fmt.Errorf("s is nil")
	}
	if c == nil {
		return fmt.Errorf("collector service is nil")
	}
	s.collectService = c
	log.Infof("Collect service[%s] registered.", c.GetName())
	return nil
}

// CreateCollector creates a new prometheus collector with specified timing parameters
func (s *ExporterServer) CreateCollector(
	cacheTime time.Duration, updateTime time.Duration) prometheus.Collector {
	return s.collectService.CreateCollector(cacheTime, updateTime)
}

// StartCollect begins the data collection process in a background goroutine
func (s *ExporterServer) StartCollect(ctx context.Context, fn context.CancelFunc) {
	s.collectService.Start(ctx, fn)
}
