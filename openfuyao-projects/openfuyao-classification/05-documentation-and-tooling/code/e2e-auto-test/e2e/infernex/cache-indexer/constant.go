package cacheindexer

const (
	// standalone_test.go
	cacheIndexerLogContainer     = "cache-indexer"
	cacheIndexerRequestBlockSize = 128
)

const (
	// aggregate_single_backend_test.go
	aggregatedPromptRepeatCount        = 80
	aggregatedPromptMinTokens          = 256
	aggregatedPromptBaseTextA          = "The quick brown fox jumps over the lazy dog. "
	aggregatedPromptBaseTextCold       = "Sphinx of black quartz, judge my vow. "
	aggregatedPromptPrefixBlocks       = 2
	aggregatedPromptTotalBlocks        = 3
	aggregatedPartialHitRatio          = 2.0 / 3.0
	aggregatedHitRatioTolerance        = 1e-8
	aggregatedMutatedTokenIndex        = 0
	aggregatedMultiBackendReplicaCount = 2
	aggregatedMultiBackendQueryBlocks  = 2
	pdSingleGroupPrefillRole           = "prefill"
	pdSingleGroupDecodeRole            = "decode"
	pdSingleGroupQueryBlocks           = 2
	pdMultiGroupBackendCount           = 2
)
