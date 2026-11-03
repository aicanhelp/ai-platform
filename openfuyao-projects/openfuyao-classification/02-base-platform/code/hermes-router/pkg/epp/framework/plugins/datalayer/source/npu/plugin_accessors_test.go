/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *         http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package npu

import (
	"testing"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

func TestDataSourceTypeAccessors(t *testing.T) {
	source := newDataSourceWithResolver("npu-source", validParameters(), &resolverStub{targetURL: "http://127.0.0.1:9090/metrics"})

	typedName := source.TypedName()
	if typedName.Name != "npu-source" {
		t.Fatalf("expected plugin name npu-source, got %q", typedName.Name)
	}
	if typedName.Type != NPUExporterSourceType {
		t.Fatalf("expected plugin type %q, got %q", NPUExporterSourceType, typedName.Type)
	}
	if got := source.OutputType(); got != PodType() {
		t.Fatalf("expected output type %v, got %v", PodType(), got)
	}
	if got := source.ExtractorType(); got != fwkdl.ExtractorType {
		t.Fatalf("expected extractor type %v, got %v", fwkdl.ExtractorType, got)
	}
}

func TestPodTypeReturnsPodStructType(t *testing.T) {
	podType := PodType()
	if podType == nil {
		t.Fatal("expected non-nil pod type")
	}
	if got := podType.Name(); got != "Pod" {
		t.Fatalf("expected pod type name Pod, got %q", got)
	}
}
