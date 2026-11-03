package main

import (
	"testing"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/discovery"
	ingestl1 "gitcode.com/openFuyao/cache-indexer/pkg/ingest/l1"
	ingestl3 "gitcode.com/openFuyao/cache-indexer/pkg/ingest/l3"
)

const (
	testPodIP   = "10.0.0.1"
	testPodName = "pod-a"
	testZMQPort = int32(5557)
	testHTTP    = int32(9001)
	testRPC     = int32(9002)
)

type fakeL1Subscriber struct {
	target       ingestl1.VLLMSubscribeTarget
	unsubscribed string
}

func (f *fakeL1Subscriber) Subscribe(target ingestl1.VLLMSubscribeTarget) error {
	f.target = target
	return nil
}

func (f *fakeL1Subscriber) Unsubscribe(podIP string) error {
	f.unsubscribed = podIP
	return nil
}

func (f *fakeL1Subscriber) Stop() {}

type fakeL3Subscriber struct {
	target *ingestl3.MooncakePollTarget
}

func (f *fakeL3Subscriber) UpdateTarget(target *ingestl3.MooncakePollTarget) error {
	f.target = target
	return nil
}

func (f *fakeL3Subscriber) Stop() {}

func TestSubscriberAdapter_OnVLLMAddedAndRemoved(t *testing.T) {
	l1 := &fakeL1Subscriber{}
	l3 := &fakeL3Subscriber{}
	adapter := newSubscriberAdapter(logr.Discard(), l1, l3)

	err := adapter.OnVLLMAdded(&discovery.VLLMPodEndpoint{
		PodIP:   testPodIP,
		ZMQPort: testZMQPort,
		Identity: discovery.PodIdentity{
			Name: testPodName,
		},
	})
	if err != nil {
		t.Fatalf("OnVLLMAdded: %v", err)
	}
	if l1.target.PodIP != testPodIP || l1.target.ZMQPort != testZMQPort || l1.target.PodName != testPodName {
		t.Fatalf("target=%+v", l1.target)
	}

	if err := adapter.OnVLLMRemoved(testPodIP); err != nil {
		t.Fatalf("OnVLLMRemoved: %v", err)
	}
	if l1.unsubscribed != testPodIP {
		t.Fatalf("unsubscribed=%q want=%q", l1.unsubscribed, testPodIP)
	}
}

func TestSubscriberAdapter_OnMooncakeMasterChanged(t *testing.T) {
	l1 := &fakeL1Subscriber{}
	l3 := &fakeL3Subscriber{}
	adapter := newSubscriberAdapter(logr.Discard(), l1, l3)

	err := adapter.OnMooncakeMasterChanged(&discovery.MooncakeMasterEndpoint{
		PodIP:    testPodIP,
		HTTPPort: testHTTP,
		RPCPort:  testRPC,
		Identity: discovery.PodIdentity{Name: testPodName},
	})
	if err != nil {
		t.Fatalf("OnMooncakeMasterChanged: %v", err)
	}
	if l3.target == nil || l3.target.PodIP != testPodIP || l3.target.HTTPPort != testHTTP || l3.target.RPCPort != testRPC || l3.target.PodName != testPodName {
		t.Fatalf("target=%+v", l3.target)
	}

	if err := adapter.OnMooncakeMasterChanged(nil); err != nil {
		t.Fatalf("OnMooncakeMasterChanged nil: %v", err)
	}
	if l3.target != nil {
		t.Fatalf("target should be nil after clear, got %+v", l3.target)
	}
}
