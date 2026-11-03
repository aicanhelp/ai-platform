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

// Package epp provides the factory functions of all plugins in this repository.
package epp

import (
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"

	extractorNPU "hermes-router/pkg/epp/framework/plugins/datalayer/extractor/npu"
	sourceNPU "hermes-router/pkg/epp/framework/plugins/datalayer/source/npu"
	inflightproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflight"
	pdgroupproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/pdgroup"
	predictionfeatureextractor "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictionfeature"
	prefixcacheproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixcache"
	tokenizerproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	requestlifecycle "hermes-router/pkg/epp/framework/plugins/requestcontrol/lifecycle"
	requestprerequest "hermes-router/pkg/epp/framework/plugins/requestcontrol/prerequest"
	pdgroupfilter "hermes-router/pkg/epp/framework/plugins/scheduling/filter/pdgroup"
	prefixcachefilter "hermes-router/pkg/epp/framework/plugins/scheduling/filter/prefixcache"
	maxscorepicker "hermes-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	minscorepicker "hermes-router/pkg/epp/framework/plugins/scheduling/picker/minscore"
	randompicker "hermes-router/pkg/epp/framework/plugins/scheduling/picker/random"
	softmaxpotcpicker "hermes-router/pkg/epp/framework/plugins/scheduling/picker/softmaxpotc"
	alwaysdisaggdecider "hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider/alwaysdisagg"
	prefixbaseddecider "hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider/prefixbased"
	disaggprofilehandler "hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	autokvawarescorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/autokvaware"
	bucketscorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/bucket"
	decodekvawarescorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/decodekvaware"
	kvawarescorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/kvaware"
	pdkvawarescorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/pdkvaware"
	predictionscorer "hermes-router/pkg/epp/framework/plugins/scheduling/scorer/prediction"
)

// RegisterAllPlugins registers the factory functions of all plugins in this repository.
func RegisterAllPlugins() {
	// Scheduling plugins.
	fwkplugin.Register(kvawarescorer.PluginType, kvawarescorer.Factory)
	fwkplugin.Register(pdkvawarescorer.PluginType, pdkvawarescorer.Factory)
	fwkplugin.Register(decodekvawarescorer.PluginType, decodekvawarescorer.Factory)
	fwkplugin.Register(autokvawarescorer.PluginType, autokvawarescorer.Factory)
	fwkplugin.Register(pdgroupfilter.PluginType, pdgroupfilter.Factory)
	fwkplugin.Register(prefixcachefilter.PluginType, prefixcachefilter.Factory)
	fwkplugin.Register(bucketscorer.PDBucketType, bucketscorer.PDBucketFactory)
	fwkplugin.Register(predictionscorer.PluginType, predictionscorer.Factory)
	fwkplugin.Register(randompicker.PluginType, randompicker.Factory)
	fwkplugin.Register(minscorepicker.PluginType, minscorepicker.Factory)
	fwkplugin.Register(maxscorepicker.PluginType, maxscorepicker.Factory)
	fwkplugin.Register(softmaxpotcpicker.PluginType, softmaxpotcpicker.Factory)
	fwkplugin.Register(alwaysdisaggdecider.PluginType, alwaysdisaggdecider.Factory)
	fwkplugin.Register(prefixbaseddecider.PluginType, prefixbaseddecider.Factory)
	fwkplugin.Register(disaggprofilehandler.PluginType, disaggprofilehandler.Factory)

	// Request control plugins.
	fwkplugin.Register(inflightproducer.PluginType, inflightproducer.Factory)
	fwkplugin.Register(pdgroupproducer.PluginType, pdgroupproducer.Factory)
	fwkplugin.Register(predictionfeatureextractor.PluginType, predictionfeatureextractor.Factory)
	fwkplugin.Register(prefixcacheproducer.PluginType, prefixcacheproducer.Factory)
	fwkplugin.Register(tokenizerproducer.PluginType, tokenizerproducer.Factory)
	fwkplugin.Register(requestlifecycle.PluginType, requestlifecycle.Factory)
	fwkplugin.Register(requestprerequest.HeaderHandlerType, requestprerequest.HeaderHandlerFactory)

	// Datalayer plugins.
	fwkplugin.Register(sourceNPU.NPUExporterSourceType, sourceNPU.Factory)
	fwkplugin.Register(extractorNPU.NPUExporterExtractorType, extractorNPU.Factory)
}
