package infernexbridge

const (
	// Substrings from vinfernexservice-v1alpha1.kb.io denial messages (see validating apply logs).
	ErrMergedEngine           = "merged spec must define spec.engine"
	ErrCacheIndexerEnabled    = "spec.components.cacheIndexer.enabled is required"
	ErrMooncakeEnabled        = "spec.components.mooncake.enabled is required"
	ErrTidalEnabled           = "spec.components.pdOrchestrator.tidal.enabled is required"
	ErrEagleMonitorEnabled    = "spec.components.eagleEye.hardwareMonitor.enabled is required"
	ErrNPENetworkPerfEnabled  = "spec.components.eagleEye.networkPerformanceExporter.enabled is required"
	ErrRouterEnabled          = "spec.intelligentGatewayRouting.router.enabled is required"
	ErrRouterTemplate         = "spec.intelligentGatewayRouting.router.template is required"
	ErrHTTPRouteExclusive     = "spec.intelligentGatewayRouting.httpRoute.ref and spec.intelligentGatewayRouting.httpRoute.spec are mutually exclusive"
	ErrGatewayExclusive       = "spec.intelligentGatewayRouting.gateway.ref and spec.intelligentGatewayRouting.gateway.spec are mutually exclusive"
	ErrPoolExclusive          = "spec.intelligentGatewayRouting.inferencePool.ref and spec.intelligentGatewayRouting.inferencePool.spec are mutually exclusive"
	ErrImageRequired          = "spec.engine.template.spec.containers[0].image is required"
	ErrRouterMainContainer    = `container named "main"`
	ErrRouterGRPCPort         = `named port "grpc"`
	ErrDataParallelDivisible  = "must be divisible by dataParallelSizeLocal"
	ErrWorkerWhenGroupSizeOne = "worker must not be set when groupSize is 1"
)
