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
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"sync"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/collector/npuservice"
	"huawei.com/xpu-exporter/server"
	"huawei.com/xpu-exporter/versions"
)

var (
	updateTime int
	xpuType    string
)

const (
	exporterServerPort = 8082
	updateTimeConst    = 5
	oneMinute          = 60
	cacheTime          = 65 * time.Second
	defaultConcurrency = 5
	defaultConnection  = 20
	defaultLogDir      = "/var/log/xpu/xpu-exporter"
)

var serverHandler *server.ExporterServer

func init() {
	serverHandler = &server.ExporterServer{}
	flag.StringVar(&xpuType, "type", "", "Set xpu type,range[gpu,npu],can not be empty")
	flag.IntVar(&updateTime, "updateTime", updateTimeConst,
		"Interval (seconds) to update the npu metrics cache,range[1-60]")
	flag.IntVar(&serverHandler.Port, "port", exporterServerPort,
		"The serverHandler port of the http service,range[1025-40000]")
	flag.StringVar(&serverHandler.Ip, "ip", os.Getenv("HOST_IP"),
		"The listen ip of the service,0.0.0.0 is not recommended when install on Multi-NIC host")
	flag.IntVar(&serverHandler.Concurrency, "concurrency", defaultConcurrency,
		"The max concurrency of the http serverHandler, range is [1-512]")
	flag.IntVar(&serverHandler.LimitIPConn, "limitIPConn", defaultConcurrency,
		"the tcp connection limit for each Ip, range  is [1,128]")
	flag.IntVar(&serverHandler.LimitTotalConn, "limitTotalConn", defaultConnection,
		"the tcp connection limit for all request,range  is [1,512]")
	flag.StringVar(&serverHandler.LimitIPReq, "limitIPReq", "20/1",
		"the http request limit counts for each Ip,20/1 means allow 20 request in 1 seconds")
}

func checkCommonParamValid() error {
	if updateTime > oneMinute || updateTime < 1 {
		return errors.New("the updateTime is invalid")
	}
	return nil
}

func loadCollectorService() error {
	var err error
	switch xpuType {
	case npuservice.CollectorName:
		err = serverHandler.RegisterCollectorService(npuservice.New(npuservice.CollectorName))
	default:
		err = fmt.Errorf("the mandatory parameter type=npu or type=npu is missing or value[%s] error", xpuType)
	}
	return err
}

func main() {
	flag.Parse()

	syscall.Umask(0)
	logFileName := path.Join(defaultLogDir, "xpu-exporter.log")
	if err := log.InitLogging(logFileName); err != nil {
		log.Fatalln(err)
	}

	log.Infof("npu exporter starting and the version is %s", versions.BuildVersion)
	if err := loadCollectorService(); err != nil {
		log.Fatalln(err)
	}

	if err := checkCommonParamValid(); err != nil {
		log.Fatalln(err)
	}

	if err := serverHandler.VerifyServerParams(); err != nil {
		log.Fatalln(err)
	}

	c := serverHandler.CreateCollector(cacheTime, time.Duration(updateTime)*time.Second)
	reg := prometheus.NewRegistry()
	reg.MustRegister(c)

	ctx, cancel := context.WithCancel(context.Background())
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		serverHandler.StartCollect(ctx, cancel)
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		serverHandler.StartServe(ctx, cancel, reg)
	}()
	wg.Wait()
}
