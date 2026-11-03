package plugins

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"

	"hermes-router/pkg/plugins/filter"
	"hermes-router/pkg/plugins/picker"
	"hermes-router/pkg/plugins/prerequest"
	"hermes-router/pkg/plugins/scorer"
)

// RegisterAllPlugins registers the factory functions of all plugins in this repository.
func RegisterAllPlugins() {
	plugins.Register(picker.PickerMinRandomType, picker.PickerMinRandomFactory)
	plugins.Register(scorer.AggKVCacheAwareType, scorer.AggKVCacheAwareFactory)
	plugins.Register(scorer.PDKVCacheAwareType, scorer.PDKVCacheAwareFactory)
	plugins.Register(picker.PickerPDKVCacheAwareType, picker.PickerPDKVCacheAwareFactory)
	plugins.Register(filter.FilterByPDLabelType, filter.FilterByPDLabelFactory)
	plugins.Register(prerequest.PDHeaderHandlerType, prerequest.PDHeaderHandlerFactory)
	plugins.Register(scorer.PDBucketType, scorer.PDBucketFactory)
	plugins.Register(picker.RandomPDBucketType, picker.RandomPDBucketFactory)
}
