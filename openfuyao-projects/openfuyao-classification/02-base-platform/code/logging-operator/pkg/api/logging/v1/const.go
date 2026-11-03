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

// Package v1 provides the HTTP request handling and interaction logic for
// a logging server designed to work within Kubernetes environments. This package
// encapsulates functionalities for making and processing HTTP requests
// to Loki, a horizontally-scalable, highly-available, multi-tenant log aggregation
// system. It aims to facilitate logging data retrieval and analysis by
// providing structured access to logs stored in Loki.
//
// The package defines the APIServer and Handler types that manage the HTTP server
// and request routing respectively. The APIServer serves as the entry point for
// incoming HTTP requests, whereas Handler implements the logic for processing
// these requests, including querying Loki for log data based on various
// parameters such as filename, time range, and log level.
//
// Key Components:
//
//   - HTTPRequest: A struct that encapsulates details of an HTTP request, including
//     URL, method, headers, and body. It serves as a foundational element for
//     constructing requests to Loki.
//
//   - Handler: The core struct that handles incoming requests, routes them to
//     appropriate processing functions, and interacts with Loki to fetch or
//     manipulate log data. It maintains a map of predefined Loki query configurations
//     and a dictionary mapping human-readable time units to their respective
//     durations for flexible time range queries.
//
//   - LogData: A struct designed to unmarshal JSON responses from Loki, specifically
//     targeting the data structure returned by Loki's series and query_range APIs.
//
// Functionalities:
//
//   - Dynamic Loki Query Construction: Allows building and executing complex Loki
//     queries based on request parameters, enabling clients to perform detailed log
//     searches and analysis.
//
//   - Time Range Calculation: Offers utilities to calculate start and end times for
//     queries based on either specific timestamps or relative time units (e.g., "last 5 minutes").
//
//   - API Endpoints: Defines RESTful endpoints for interacting with the logging server,
//     including endpoints for listing available log sources and querying logs based
//     on various filters.
//
// Usage:
// To use this package, create an instance of the Handler struct via the New function,
// then call its methods to handle incoming requests or interact with Loki. The
// package's functionality is encapsulated in the Handler type, making it the primary
// interface for clients.
//
// Note:
// This package assumes that Loki is accessible via HTTP and that clients have
// necessary permissions to query Loki. It also relies on external packages such
// as "github.com/emicklei/go-restful/v3" for RESTful routing and
// "k8s.io/apimachinery/pkg/util/runtime" for Kubernetes-related utilities.
//
// The package is designed with extensibility in mind, allowing for future
// enhancements such as additional Loki query types, improved error handling,
// and richer response parsing.
package v1

const (
	// ContentTypeJSON specifies the MIME type for JSON, used in content type headers.
	ContentTypeJSON = "application/json"
	// ContentType is a header field name used to specify the MIME type of the content being sent.
	ContentType = "Content-Type"
)

// logLevelRegex contains regular expressions for matching log messages by severity levels.
// It maps each log level to a regex pattern that identifies strings commonly associated
// with each log level in log files or streams. This allows for filtering and categorization
// of log messages based on their indicated severity level in various log formats.
//
// Patterns:
// "error"    - Matches strings that denote an error log level. The pattern captures various
// formats commonly used to indicate errors, such as tags or prefixes in logs.
// "warning"  - Matches strings that denote a warning log level. It includes common indicators
// of warnings in log messages.
// "info"     - Matches strings representing an informational log level. This is typically used
// for regular messages that provide runtime information without indicating errors.
// "debug"    - Matches debug-level messages, which are usually intended for development and
// debugging purposes, providing detailed contextual information.
// "critical" - Matches strings that indicate a critical severity level, often used for logs
// that describe severe conditions that may require immediate attention.
var logLevelRegex = map[string]string{
	"error":    errorRegex,
	"warning":  warningRegex,
	"info":     infoRegex,
	"debug":    debugRegex,
	"critical": criticalRegex,
}

var errorRegex = `(` +
	`\\[ERROR\\]|` +
	`\\bERROR\\b|` +
	`\\bERR\\b|` +
	`\\<Error\\>|` +
	`ERROR\\:|` +
	`Error\\:|` +
	`error\\:|` +
	`level=Error|` +
	`level=error|` +
	`level=ERROR|` +
	`\\x22level\\x22:\\x22error\\x22|` +
	`\\x22level\\x22:\\x22Error\\x22|` +
	`\\x22level\\x22:\\x22ERROR\\x22|` +
	`\\bErr:\\b|` +
	`\\bERR:\\b)`

var warningRegex = `(` +
	`\\[WARNING\\]|` +
	`\\bWARN\\b|` +
	`\\bWRN\\b|` +
	`\\<Warning\\>|` +
	`WARN\\:|` +
	`WARNING\\:|` +
	`Warning\\:|` +
	`warning:|` +
	`Warn\\:|` +
	`warn\\:|` +
	`level=Warn|` +
	`level=warning|` +
	`level=warn|` +
	`level=WARNING|` +
	`\\x22level\\x22:\\x22warn\\x22|` +
	`\\x22level\\x22:\\x22Warn\\x22|` +
	`\\x22level\\x22:\\x22warning\\x22|` +
	`\\x22level\\x22:\\x22Warning\\x22|` +
	`\\x22level\\x22:\\x22WARNING\\x22|` +
	`\\bWrn\\b|` +
	`\\bWRN:\\b)`

var infoRegex = `(` +
	`\\[INFO\\]|` +
	`\\bINFO\\b|` +
	`\\<Info\\>|` +
	`INFO\\:|` +
	`Info\\:|` +
	`info\\:|` +
	`level=Info|` +
	`level=info|` +
	`level=INFO|` +
	`\\x22level\\x22:\\x22info\\x22|` +
	`\\x22level\\x22:\\x22Info\\x22|` +
	`\\x22level\\x22:\\x22INFO\\x22)`

var debugRegex = `(` +
	`\\[DEBUG\\]|` +
	`\\bDEBUG\\b|` +
	`\\bDBG\\b|` +
	`\\<Debug\\>|` +
	`DEBUG\\:|` +
	`Debug\\:|` +
	`Debugging\\:|` +
	`level=Debug|` +
	`level=debug|` +
	`level=DEBUG|` +
	`\\x22level\\x22:\\x22debug\\x22|` +
	`\\x22level\\x22:\\x22Debug\\x22|` +
	`\\x22level\\x22:\\x22DEBUG\\x22|` +
	`\\bDbg\\b|` +
	`\\bDBG:\\b)`

var criticalRegex = `(` +
	`\\[CRITICAL\\]|` +
	`\\bCRITICAL\\b|` +
	`\\bCRIT\\b|` +
	`\\<Critical\\>|` +
	`CRITICAL\\:|` +
	`Critical\\:|` +
	`level=critical|` +
	`level=Critical|` +
	`level=CRITICAL|` +
	`\\x22level\\x22:\\x22critical\\x22|` +
	`\\x22level\\x22:\\x22Critical\\x22|` +
	`\\x22level\\x22:\\x22CRITICAL\\x22|` +
	`\\bCRIT:\\b)`
