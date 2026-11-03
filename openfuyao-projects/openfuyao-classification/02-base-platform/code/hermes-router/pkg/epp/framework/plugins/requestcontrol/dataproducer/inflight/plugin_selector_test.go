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

package inflight

import (
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	internalinflight "hermes-router/pkg/epp/internal/inflight"
)

func TestNewLoadProducerDefaultsEmptyAttributeKey(t *testing.T) {
	producer := NewLoadProducer("producer", internalinflight.StoreForName("default-key-test"), "")

	produced := producer.Produces()
	if _, ok := produced[internalinflight.DefaultAttributeKey]; !ok {
		t.Fatalf("expected default attribute key in declaration, got %+v", produced)
	}
}

func TestPrepareRequestDataHandlesNilStoreAndPods(t *testing.T) {
	var nilProducer *LoadProducer
	if err := nilProducer.PrepareRequestData(nil, nil, nil); err != nil {
		t.Fatalf("expected nil producer to be a no-op, got %v", err)
	}

	producer := NewLoadProducer("producer", nil, internalinflight.DefaultAttributeKey)
	if err := producer.PrepareRequestData(nil, nil, nil); err != nil {
		t.Fatalf("expected nil store to be a no-op, got %v", err)
	}

	store := internalinflight.StoreForName("nil-pods-test")
	producer = NewLoadProducer("producer", store, internalinflight.DefaultAttributeKey)
	if err := producer.PrepareRequestData(nil, nil, []fwkscheduling.Endpoint{nil}); err != nil {
		t.Fatalf("expected nil pod entry to be skipped, got %v", err)
	}
}

func TestParseParametersRejectsUnknownFields(t *testing.T) {
	if _, err := parseParameters([]byte(`{"unknown":1}`)); err == nil {
		t.Fatal("expected unknown field to be rejected")
	}
	if _, err := parseParameters([]byte(`{invalid`)); err == nil {
		t.Fatal("expected invalid JSON to be rejected")
	}
	if params, err := parseParameters(nil); err != nil || params.StoreName != internalinflight.DefaultStoreName {
		t.Fatalf("expected defaults for empty parameters, got %+v, %v", params, err)
	}
	if params, err := parseParameters([]byte(`{}`)); err != nil || params.AttributeKey != internalinflight.DefaultAttributeKey {
		t.Fatalf("expected default attribute key for empty object, got %+v, %v", params, err)
	}
}
