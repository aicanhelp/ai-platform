/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package options describe the config for main.go
package options

import "flag"

// Flags describes opts usage
func Flags(options *Opts) {
	// get command line parameters
	flag.BoolVar(&options.DryRun, "dryRun", false, "When present,"+
		" indicates that modifications should not be persisted")
	flag.BoolVar(&options.Healthz, "health", false, "Describes a health check to be performed"+
		" against a container to determine whether it is alive or ready to receive traffic")
	flag.IntVar(&options.Port, "port", 443, "Webhook nriServer port.")
	flag.StringVar(&options.CertFile, "tlsCertFile", "/etc/webhook/certs/server.pem",
		"File containing the x509 Certificate for HTTPS.")
	flag.StringVar(&options.KeyFile, "tlsKeyFile", "/etc/webhook/certs/key.pem",
		"File containing the x509 private key to --tlsCertFile.")
	flag.StringVar(&options.SvcNamespace, "svcNamespace", "volcano-system",
		"The namespace of the service ")
	flag.StringVar(&options.SvcName, "svcName", "numa-affinity-webhook-svc",
		"The name of the service")

	flag.Parse()
}
