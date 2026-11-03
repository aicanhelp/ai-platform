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

package main

import (
	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/discovery"
	ingestl1 "gitcode.com/openFuyao/cache-indexer/pkg/ingest/l1"
	ingestl3 "gitcode.com/openFuyao/cache-indexer/pkg/ingest/l3"
)

// subscriberAdapter bridges discovery to concrete ingest packages.
// Keeping it in cmd avoids discovery <-> ingest imports.
type subscriberAdapter struct {
	log   logr.Logger
	l1Sub ingestl1.SubscriptionManager
	l3Sub ingestl3.MooncakePollTargetUpdater
}

func newSubscriberAdapter(
	log logr.Logger,
	l1Sub ingestl1.SubscriptionManager,
	l3Sub ingestl3.MooncakePollTargetUpdater,
) discovery.EndpointSubscriber {
	return &subscriberAdapter{log: log, l1Sub: l1Sub, l3Sub: l3Sub}
}

func (a *subscriberAdapter) OnVLLMAdded(ep *discovery.VLLMPodEndpoint) error {
	return a.l1Sub.Subscribe(ingestl1.VLLMSubscribeTarget{
		PodIP:   ep.PodIP,
		ZMQPort: ep.ZMQPort,
		PodName: ep.Identity.Name,
	})
}

func (a *subscriberAdapter) OnVLLMRemoved(podIP string) error {
	return a.l1Sub.Unsubscribe(podIP)
}

func (a *subscriberAdapter) OnMooncakeMasterChanged(ep *discovery.MooncakeMasterEndpoint) error {
	if ep == nil {
		return a.l3Sub.UpdateTarget(nil)
	}
	return a.l3Sub.UpdateTarget(&ingestl3.MooncakePollTarget{
		PodIP:    ep.PodIP,
		HTTPPort: ep.HTTPPort,
		RPCPort:  ep.RPCPort,
		PodName:  ep.Identity.Name,
	})
}
