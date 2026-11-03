package npu_dra_plugin

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
)

const (
	topologyHCCSResourceAttribute     = "topoHccsRingID"
	topologySIOResourceAttribute      = "topoSioID"
	topologyNUMAResourceAttribute     = "numaNode"
	topologyPCIBusIDResourceAttribute = "resource.kubernetes.io/pciBusID"
	topologyPCIeRootResourceAttribute = "resource.kubernetes.io/pcieRoot"
	topologyPhysicalIDAttribute       = "physicalID"
	topologySeries910B                = "910B"
	topologySeries910C                = "910C"
)

var (
	topologyHCCSAttribute = topologyFullyQualifiedName(driverName + "/" + topologyHCCSResourceAttribute)
	topologySIOAttribute  = topologyFullyQualifiedName(driverName + "/" + topologySIOResourceAttribute)
	topologyNUMAAttribute = topologyFullyQualifiedName(driverName + "/" + topologyNUMAResourceAttribute)
	topologyPCIeAttribute = topologyFullyQualifiedName(topologyPCIeRootResourceAttribute)
	topologyPCIBusIDRe    = regexp.MustCompile(`^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$`)
	topologyPCIeRootRe    = regexp.MustCompile(`^pci[0-9a-fA-F]{4}:[0-9a-fA-F]{2}$`)
)

type topologyInventory struct {
	series  string
	devices map[int]resourcev1.Device
}

type topologySysfsDevice struct {
	pcieRoot string
	numaNode int64
	series   string
}

// 本文件为真实集群 E2E 用例，绝不解析或 mock `npu-smi info -t topo`
// fixture：硬件判定来自 PCI sysfs，拓扑上报以 ResourceSlice 为准，分配
// 结果来自真实 ResourceClaim。插件 Pod 内实时 `npu-smi info` 与
// `npu-smi info -t topo` 输出仅作为独立交叉证据，且仅当从 "Chip Phy-ID"
// 双 ID 列推导出的 NPU ID -> physicalID 映射无歧义时才使用。生产 Issue #40
// 的根因正是 NPU ID 与 physicalID 混淆，因此映射歧义或重复时成员级交叉
// 比较必须跳过，而不是误判插件失败。
//
// 拓扑发布契约（Pass/Fail 唯一 oracle）：HAL/DCMI 连通分量按最小
// physicalID 升序从 0 编号，HCCS_SW 归一化为 HCCS，无关系时不发布属性。
var _ = Describe("NPU DRA Plugin 拓扑亲和测试", Label("npu-dra-plugin", "26.09", "topology-affinity"), Serial, func() {
	var inventory topologyInventory
	var pluginPod *corev1.Pod

	BeforeEach(func() {
		ensureNamespace()
		pluginPod = topologyPluginPodOnNode()
		series, deviceCount := requireTargetTopologyHardware(pluginPod)
		requireTopologyDeviceClass()
		inventory = requireTargetTopologyInventory(pluginPod, series, deviceCount)
	})

	It("昇腾-npu-dra-FIT_1：DRA Driver部署与拓扑属性上报验证", Label("npu-dra-affinity-fit-1", "P0"), func() {
		devices := startTopologyWorkload("fit1", []resourcev1.DeviceRequest{topologyExactRequest("npu", 1, "")}, nil)
		Expect(devices).To(HaveLen(1))
		Expect(topologyStringAttribute(devices[0].device, "type")).To(Equal(inventory.series))
		GinkgoWriter.Printf("FIT_1 验证通过: %d 卡 %s ResourceSlice 的真实整卡分配\n", len(inventory.devices), inventory.series)
	})

	It("昇腾-npu-dra-FIT_2：910B4双HCCS环拓扑属性准确性验证", Label("npu-dra-affinity-fit-2", "P0"), func() {
		if inventory.series != topologySeries910B || len(inventory.devices) != 16 {
			Skip(fmt.Sprintf("本用例需 910B 16 卡硬件；实际: series=%s 设备数=%d", inventory.series, len(inventory.devices)))
		}
		groups := topologyIntegerGroups(inventory.devices, topologyHCCSResourceAttribute)
		Expect(sortedTopologyGroupIDs(groups)).To(Equal([]int64{0, 1}), "16 卡 910B 应发布两个连续编号的 HCCS 环；实际各分量大小=%v", topologyGroupSizes(groups))
		Expect(groups[0]).To(HaveLen(8), "HCCS 环 0 应包含八卡")
		Expect(groups[1]).To(HaveLen(8), "HCCS 环 1 应包含八卡")
	})

	It("昇腾-npu-dra-FIT_5：910C超节点SIO拓扑属性验证", Label("npu-dra-affinity-fit-5", "P1"), func() {
		if inventory.series != topologySeries910C || len(inventory.devices) != 8 {
			Skip(fmt.Sprintf("本用例需 910C 8 卡超节点硬件；实际: series=%s 设备数=%d", inventory.series, len(inventory.devices)))
		}
		hccsGroups := topologyIntegerGroups(inventory.devices, topologyHCCSResourceAttribute)
		sioGroups := topologyIntegerGroups(inventory.devices, topologySIOResourceAttribute)

		Expect(sortedTopologyGroupIDs(hccsGroups)).To(Equal([]int64{0}), "八卡 910C 应发布一个 HCCS_SW 连通分组；实际 HCCS各分量大小=%v", topologyGroupSizes(hccsGroups))
		Expect(hccsGroups[0]).To(HaveLen(8))

		Expect(sortedTopologyGroupIDs(sioGroups)).To(Equal([]int64{0, 1, 2, 3}), "八卡 910C 应发布四个 SIO 分组；实际 SIO各分量大小=%v", topologyGroupSizes(sioGroups))
		for groupID := int64(0); groupID < 4; groupID++ {
			Expect(sioGroups[groupID]).To(HaveLen(2), "SIO 分组 %d 应包含两卡", groupID)
		}
	})

	It("昇腾-npu-dra-FIT_6：单HCCS环8卡约束调度验证", Label("npu-dra-affinity-fit-6", "P0"), func() {
		rings := topologyIntegerGroups(inventory.devices, topologyHCCSResourceAttribute)
		if !hasTopologyGroupOfSize(rings, 8) {
			Skip(fmt.Sprintf("本用例需至少 8 卡的 HCCS 环；实际: HCCS分量数=%d 各分量大小=%v",
				len(rings), topologyGroupSizes(rings)))
		}
		devices := startTopologyWorkload("fit6", []resourcev1.DeviceRequest{topologyExactRequest("npu", 8, "")}, []resourcev1.DeviceConstraint{topologyMatch("npu", topologyHCCSAttribute)})
		Expect(devices).To(HaveLen(8))
		assertSameTopologyAttribute(devices, topologyHCCSAttribute)
	})

	It("昇腾-npu-dra-FIT_7：同HCCS环+同NUMA 4卡约束调度验证", Label("npu-dra-affinity-fit-7", "P0"), func() {
		if !hasHCCSNUMAGroup(inventory.devices, 4) {
			Skip(fmt.Sprintf("本用例需至少 4 卡同属一个 HCCS 环且同 NUMA 节点；实际 HCCS各分量大小=%v HCCS-NUMA交集=%v",
				topologyGroupSizes(topologyIntegerGroups(inventory.devices, topologyHCCSResourceAttribute)), hccsNUMAIntersectionSizes(inventory.devices)))
		}
		devices := startTopologyWorkload("fit7", []resourcev1.DeviceRequest{topologyExactRequest("npu", 4, "")}, []resourcev1.DeviceConstraint{
			topologyMatch("npu", topologyHCCSAttribute),
			topologyMatch("npu", topologyNUMAAttribute),
		})
		Expect(devices).To(HaveLen(4))
		assertSameTopologyAttribute(devices, topologyHCCSAttribute)
		assertSameTopologyAttribute(devices, topologyNUMAAttribute)
	})

	It("昇腾-npu-dra-FIT_8：同PCIe Root（PIX连接）2卡约束调度验证", Label("npu-dra-affinity-fit-8", "P1"), func() {
		roots := topologyStringGroupSizes(inventory.devices, topologyPCIeRootResourceAttribute)
		if !hasStringTopologyGroup(inventory.devices, topologyPCIeRootResourceAttribute, 2) {
			Skip(fmt.Sprintf("本用例需至少 2 卡同属一个 PCIe Root；实际 PCIe Root分组=%v", roots))
		}
		devices := startTopologyWorkload("fit8", []resourcev1.DeviceRequest{topologyExactRequest("npu", 2, "")}, []resourcev1.DeviceConstraint{topologyMatch("npu", topologyPCIeAttribute)})
		Expect(devices).To(HaveLen(2))
		assertSameTopologyAttribute(devices, topologyPCIeAttribute)
	})

	It("昇腾-npu-dra-FIT_9：优先级调度降级验证（HCCS不可用降级到PCIe Root）", Label("npu-dra-affinity-fit-9", "P0"), func() {
		rings := topologyIntegerGroups(inventory.devices, topologyHCCSResourceAttribute)
		// 降级场景需要两个 HCCS 环之间一对同 PCIe Root 的跨环设备对：
		// 占位 Pod 占用其余全部设备后，仅剩的空闲两卡不同环（同环优先级
		// 真实不可满足），但同 PCIe Root（第二优先级可满足）。单环节点
		// （如 8 卡 910B）不存在跨环对，按前置条件跳过。
		pairs, found := findSymmetricHCCSPairs(inventory.devices, 1)
		if !found {
			Skip(fmt.Sprintf("本用例需两个 HCCS 环之间存在同 PCIe Root 的跨环设备对；实际: 设备数=%d HCCS分量数=%d 各分量大小=%v",
				len(inventory.devices), len(rings), topologyGroupSizes(rings)))
		}
		pair := pairs[0]

		occupied := make([]int, 0, len(inventory.devices)-2)
		pairSet := map[int]struct{}{pair[0]: {}, pair[1]: {}}
		for _, physicalID := range sortedTopologyPhysicalIDs(inventory.devices) {
			if _, reserved := pairSet[physicalID]; !reserved {
				occupied = append(occupied, physicalID)
			}
		}

		By(fmt.Sprintf("用占位 Claim/Pod 占用跨环同 Root 候选对 %v 之外的全部 %d 张卡", pair, len(occupied)))
		holderClaim, holderPod := createTopologyClaimAndPod("fit9-holder",
			[]resourcev1.DeviceRequest{topologyExactRequest("holder", int64(len(occupied)), topologyRingOccupationExpression(occupied))},
			nil)
		DeferCleanup(func() { cleanupTopologyClaimAndPod(holderClaim, holderPod) })
		waitForPodRunning(testNamespace, holderPod)
		holderDevices := topologyAllocatedDevices(holderClaim)
		Expect(topologyPhysicalIDs(holderDevices)).To(Equal(occupied), "占位 Claim 应恰好占用降级候选对之外的全部设备")
		GinkgoWriter.Printf("占位 Pod %s 已占用 %d 张卡，仅留跨环同 Root 候选对 %v 空闲\n", holderPod, len(occupied), pair)

		By("节点仅剩不同环的两张空闲卡，创建 firstAvailable 被测 Claim/Pod")
		// 两个子请求都按 CSV 原语义（无 CEL 钉死）：第一优先级约束到同
		// HCCS 环，第二优先级约束到同 PCIe Root。占位后仅剩的两张空闲卡
		// 不同环，第一优先级因真实容量不足不可满足，调度器必须降级。
		devices := startTopologyWorkload("fit9", []resourcev1.DeviceRequest{topologyFirstAvailableRequest()}, []resourcev1.DeviceConstraint{
			topologyMatch("npu/npu-same-hccs-ring", topologyHCCSAttribute),
			topologyMatch("npu/npu-same-pcieroot", topologyPCIeAttribute),
		})
		Expect(devices).To(HaveLen(2))
		for _, device := range devices {
			Expect(device.request).To(Equal("npu/npu-same-pcieroot"), "FirstAvailable 应在 HCCS 候选不可满足后选择 PCIe Root 候选")
		}
		allocated := topologyPhysicalIDs(devices)
		expectedPair := []int{pair[0], pair[1]}
		sort.Ints(expectedPair)
		Expect(allocated).To(Equal(expectedPair), "降级分配应恰好落在仅剩的两张空闲卡上")
		assertSameTopologyAttribute(devices, topologyPCIeAttribute)
		Expect(topologyIntegerAttribute(devices[0], topologyHCCSAttribute)).NotTo(Equal(topologyIntegerAttribute(devices[1], topologyHCCSAttribute)), "降级到 PCIe Root 的两卡应来自不同 HCCS 环")
		GinkgoWriter.Printf("降级分配 physicalIDs=%v（跨环同 Root 候选对为 %v）\n", allocated, expectedPair)
	})

	It("昇腾-npu-dra-FIT_10：跨环对称12卡约束调度验证", Label("npu-dra-affinity-fit-10", "P1"), func() {
		pairs, found := findSymmetricHCCSPairs(inventory.devices, 6)
		if !found {
			Skip(fmt.Sprintf("本用例需两个 HCCS 环之间存在六组同 PCIe Root 的跨环设备对；实际 HCCS各分量大小=%v",
				topologyGroupSizes(topologyIntegerGroups(inventory.devices, topologyHCCSResourceAttribute))))
		}

		requests, constraints := topologySymmetricTwelveCardRequest(pairs)
		devices := startTopologyWorkload("fit10", requests, constraints)
		Expect(devices).To(HaveLen(12))

		leftRequests := []string{"npu1", "npu2", "npu3", "npu4", "npu5", "npu6"}
		rightRequests := []string{"npu7", "npu8", "npu9", "npu10", "npu11", "npu12"}
		assertSameTopologyAttribute(topologyDevicesForRequests(devices, leftRequests), topologyHCCSAttribute)
		assertSameTopologyAttribute(topologyDevicesForRequests(devices, rightRequests), topologyHCCSAttribute)
		leftRing := topologyIntegerAttribute(topologyDevicesForRequests(devices, []string{"npu1"})[0], topologyHCCSAttribute)
		rightRing := topologyIntegerAttribute(topologyDevicesForRequests(devices, []string{"npu7"})[0], topologyHCCSAttribute)
		Expect(leftRing).NotTo(Equal(rightRing), "两个六卡组应位于不同 HCCS 环")
		for i := 1; i <= 6; i++ {
			pair := topologyDevicesForRequests(devices, []string{fmt.Sprintf("npu%d", i), fmt.Sprintf("npu%d", i+6)})
			assertSameTopologyAttribute(pair, topologyPCIeAttribute)
			expectedPhysicalIDs := []int{pairs[i-1][0], pairs[i-1][1]}
			sort.Ints(expectedPhysicalIDs)
			Expect(topologyPhysicalIDs(pair)).To(Equal(expectedPhysicalIDs))
		}
	})

	It("昇腾-npu-dra-FIT_12：拓扑约束无法满足调度失败验证", Label("npu-dra-affinity-fit-12", "P1"), func() {
		groups := topologyIntegerGroups(inventory.devices, topologyHCCSResourceAttribute)
		// 本用例需要一个恰好 8 卡的 HCCS 分量，使 9 卡同环请求真正
		// 不可满足。
		if !hasTopologyGroupOfSize(groups, 8) || len(inventory.devices) != 8 {
			Skip(fmt.Sprintf("本用例需节点恰好只有一个 8 卡 HCCS 分量，使 9 卡同环请求不可满足；实际: 设备数=%d HCCS分量数=%d 各分量大小=%v",
				len(inventory.devices), len(groups), topologyGroupSizes(groups)))
		}

		claimName, podName := createTopologyClaimAndPod("fit12", []resourcev1.DeviceRequest{topologyExactRequest("npu", 9, "")}, []resourcev1.DeviceConstraint{topologyMatch("npu", topologyHCCSAttribute)})
		DeferCleanup(func() { cleanupTopologyClaimAndPod(claimName, podName) })
		assertTopologyRequestRemainsPending(claimName, podName)
		assertUnschedulableEventMentionsAllocation(podName)
	})

	It("昇腾-npu-dra-FIT_24：拓扑约束Pod删除后资源回收验证", Label("npu-dra-affinity-fit-24", "P1"), func() {
		rings := topologyIntegerGroups(inventory.devices, topologyHCCSResourceAttribute)
		if !hasTopologyGroupOfSize(rings, 8) {
			Skip(fmt.Sprintf("本用例需至少 8 卡的 HCCS 环以验证回收；实际 HCCS各分量大小=%v",
				topologyGroupSizes(rings)))
		}
		claimName, podName := createTopologyClaimAndPod("fit24-release", []resourcev1.DeviceRequest{topologyExactRequest("npu", 8, "")}, []resourcev1.DeviceConstraint{topologyMatch("npu", topologyHCCSAttribute)})
		DeferCleanup(func() { cleanupTopologyClaimAndPod(claimName, podName) })
		waitForPodRunning(testNamespace, podName)
		Expect(topologyAllocatedDevices(claimName)).To(HaveLen(8))
		cleanupTopologyClaimAndPod(claimName, podName)

		devices := startTopologyWorkload("fit24-reclaim", []resourcev1.DeviceRequest{topologyExactRequest("npu", 8, "")}, []resourcev1.DeviceConstraint{topologyMatch("npu", topologyHCCSAttribute)})
		Expect(devices).To(HaveLen(8), "删除 Pod 和 ResourceClaim 后应能再次取得八卡")
	})
})

func requireTopologyDeviceClass() {
	_, err := k8sClient.Clientset.ResourceV1().DeviceClasses().Get(ctx, deviceClassFull, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "真机拓扑用例需要 DeviceClass %s", deviceClassFull)
}

func requireTargetTopologyHardware(pluginPod *corev1.Pod) (string, int) {
	const command = `for path in /sys/bus/pci/devices/*; do
	[ -r "$path/vendor" ] && [ -r "$path/device" ] || continue
	vendor=$(cat "$path/vendor")
	device=$(cat "$path/device")
	case "$vendor:$device" in
		0x19e5:0xd802) printf '910B\n' ;;
		0x19e5:0xd803) printf '910C\n' ;;
	esac
	done`
	stdout, stderr, err := execCommandInPod(pluginNamespace, pluginPod.Name, topologyPluginContainer(pluginPod), command)
	Expect(err).NotTo(HaveOccurred(), "从目标节点 PCI sysfs 识别 NPU 型号失败: %s", stderr)
	seriesSet := make(map[string]struct{})
	for _, series := range strings.Fields(stdout) {
		seriesSet[series] = struct{}{}
	}
	if len(seriesSet) == 0 {
		Skip("节点 PCI sysfs 未发现 910B(0xd802) 或 910C(0xd803)，跳过真实拓扑 E2E")
	}
	Expect(seriesSet).To(HaveLen(1), "单节点拓扑 E2E 不支持混合 910B/910C")
	for series := range seriesSet {
		return series, len(strings.Fields(stdout))
	}
	return "", 0
}

func requireTargetTopologyInventory(pluginPod *corev1.Pod, expectedSeries string, expectedDeviceCount int) topologyInventory {
	devices := topologyDevicesFromResourceSlice()
	Expect(devices).To(HaveLen(expectedDeviceCount), "ResourceSlice 设备数应与目标节点 PCI sysfs 中的 %s 设备数一致", expectedSeries)
	result := topologyInventory{series: expectedSeries, devices: devices}
	busIDs := make(map[string]struct{}, len(devices))

	for _, physicalID := range sortedTopologyPhysicalIDs(devices) {
		device := devices[physicalID]
		Expect(device.Name).To(Equal(fmt.Sprintf("npu-%d", physicalID)), "physicalID 与 ResourceSlice 设备名应一致")
		Expect(device.Attributes).NotTo(HaveKey(resourcev1.QualifiedName("physicalId")), "仅允许字段 physicalID")
		Expect(device.Attributes).NotTo(HaveKey(resourcev1.QualifiedName("busId")), "仅允许字段 resource.kubernetes.io/pciBusID")
		Expect(topologyStringAttribute(device, "chipName")).NotTo(BeEmpty())
		Expect(topologyDeviceIntegerAttribute(device, resourcev1.FullyQualifiedName("memoryTotal"))).To(BeNumerically(">", 0))
		memCapacity, found := device.Capacity[resourcev1.QualifiedName("memCapacity")]
		Expect(found).To(BeTrue(), "设备 %s 缺少 memCapacity", device.Name)
		Expect(memCapacity.Value.Sign()).To(BeNumerically(">", 0), "设备 %s 的 memCapacity 应为正数", device.Name)

		sysfs := readTopologySysfsDevice(pluginPod, device)
		Expect(sysfs.series).To(Equal(expectedSeries), "目标 ResourceSlice 不应混入非 %s 设备", expectedSeries)
		busID := strings.ToLower(topologyStringAttribute(device, topologyPCIBusIDResourceAttribute))
		_, duplicateBusID := busIDs[busID]
		Expect(duplicateBusID).To(BeFalse(), "ResourceSlice 存在重复 pciBusID=%s", busID)
		busIDs[busID] = struct{}{}
		Expect(topologyStringAttribute(device, "type")).To(Equal(expectedSeries), "设备 %s 的 type 应与 PCI device ID 一致", device.Name)
		assertDevicePCIeAndNUMAFromSysfs(device, sysfs)
		GinkgoWriter.Printf("ResourceSlice 设备 physicalID=%d: %s\n", physicalID, topologyDeviceAttributeSummary(device))
	}

	requireHALTopologySource(pluginPod)
	assertTopologyGroupContract(devices, topologyHCCSResourceAttribute)
	assertTopologyGroupContract(devices, topologySIOResourceAttribute)
	return result
}

// topologyPluginPodOnNode 返回目标节点上的插件 Pod；环境不满足（无插件
// Pod、目标节点未部署、或集群无任何 ResourceSlice）时 Skip，与其他
// 用例的 findNPUNode Skip 风格一致，使纯 CI 集群可整套运行。
func topologyPluginPodOnNode() *corev1.Pod {
	pods, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).List(ctx, metav1.ListOptions{LabelSelector: "app=npu-dra-plugin"})
	Expect(err).NotTo(HaveOccurred(), "列出 npu-dra-plugin Pod")
	if len(pods.Items) == 0 {
		Skip("未找到 npu-dra-plugin Pod，插件未部署，跳过真实拓扑 E2E")
	}
	for i := range pods.Items {
		if pods.Items[i].Spec.NodeName == localNodeName {
			return &pods.Items[i]
		}
	}
	Skip(fmt.Sprintf("节点 %s 未部署 npu-dra-plugin Pod，跳过真实拓扑 E2E（可用节点: %s）", localNodeName, topologyPluginPodNodes(pods.Items)))
	return nil
}

func topologyPluginPodNodes(pods []corev1.Pod) string {
	nodes := make([]string, 0, len(pods))
	for i := range pods {
		nodes = append(nodes, pods[i].Spec.NodeName)
	}
	sort.Strings(nodes)
	return strings.Join(nodes, ",")
}

func topologyPluginContainer(pod *corev1.Pod) string {
	for _, container := range pod.Spec.Containers {
		if container.Name == "dra-plugin" {
			return container.Name
		}
	}
	Expect(pod.Spec.Containers).NotTo(BeEmpty(), "插件 Pod 应至少有一个容器")
	return pod.Spec.Containers[0].Name
}

func requireHALTopologySource(pluginPod *corev1.Pod) {
	container := topologyPluginContainer(pluginPod)
	var latest, latestError string
	Eventually(func() string {
		logs, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).GetLogs(
			pluginPod.Name, &corev1.PodLogOptions{Container: container}).Do(ctx).Raw()
		if err != nil {
			latestError = err.Error()
			return ""
		}
		latest = latestHALTopologyStatus(string(logs))
		return latest
	}, 60*time.Second, pollInterval).ShouldNot(BeEmpty(), "插件日志应包含 topology source=hal 状态，最后一次读取错误: %s", latestError)
	Expect(latest).To(Equal("topology source=hal"), "目标节点最新一轮 HAL 拓扑发现必须成功，不能以 unavailable 或 skipped 冒充成功")
	GinkgoWriter.Printf("插件拓扑状态: %s\n", latest)
}

func latestHALTopologyStatus(logs string) string {
	lines := strings.Split(logs, "\n")
	for index := len(lines) - 1; index >= 0; index-- {
		if offset := strings.Index(lines[index], "topology source=hal"); offset >= 0 {
			return strings.TrimSpace(lines[index][offset:])
		}
	}
	return ""
}

func readTopologySysfsDevice(pluginPod *corev1.Pod, device resourcev1.Device) topologySysfsDevice {
	busID := topologyStringAttribute(device, topologyPCIBusIDResourceAttribute)
	Expect(topologyPCIBusIDRe.MatchString(busID)).To(BeTrue(), "ResourceSlice 的 pciBusID 非法: %q", busID)
	command := fmt.Sprintf("path=$(readlink -f /sys/bus/pci/devices/%s) && printf '%%s\\n' \"$path\" && cat \"$path/vendor\" \"$path/device\" \"$path/numa_node\"", strings.ToLower(busID))
	stdout, stderr, err := execCommandInPod(pluginNamespace, pluginPod.Name, topologyPluginContainer(pluginPod), command)
	Expect(err).NotTo(HaveOccurred(), "读取 NPU %s 的 PCI sysfs 失败: %s", device.Name, stderr)
	fields := strings.Fields(stdout)
	Expect(fields).To(HaveLen(4), "PCI sysfs 输出应包含真实路径、vendor、device 和 NUMA: %q", stdout)
	Expect(strings.ToLower(fields[1])).To(Equal("0x19e5"), "设备 %s 应为 Huawei PCI 设备", device.Name)

	series := ""
	switch strings.ToLower(fields[2]) {
	case "0xd802":
		series = topologySeries910B
	case "0xd803":
		series = topologySeries910C
	}
	numaNode, err := strconv.ParseInt(fields[3], 10, 64)
	Expect(err).NotTo(HaveOccurred(), "解析设备 %s 的 sysfs NUMA: %q", device.Name, fields[3])

	pcieRoot := ""
	for _, component := range strings.Split(fields[0], "/") {
		if topologyPCIeRootRe.MatchString(component) {
			pcieRoot = strings.ToLower(component)
			break
		}
	}
	Expect(pcieRoot).NotTo(BeEmpty(), "PCI 路径应包含 pcie root: %s", fields[0])
	return topologySysfsDevice{pcieRoot: pcieRoot, numaNode: numaNode, series: series}
}

func assertDevicePCIeAndNUMAFromSysfs(device resourcev1.Device, sysfs topologySysfsDevice) {
	Expect(strings.ToLower(topologyStringAttribute(device, topologyPCIeRootResourceAttribute))).To(Equal(sysfs.pcieRoot), "设备 %s 的 pcieRoot", device.Name)
	attribute, reported := device.Attributes[resourcev1.QualifiedName(topologyNUMAResourceAttribute)]
	if sysfs.numaNode < 0 {
		Expect(reported).To(BeFalse(), "sysfs NUMA=-1 时不应上报 numaNode")
		return
	}
	Expect(reported).To(BeTrue(), "sysfs NUMA 可用时必须上报 numaNode")
	Expect(attribute.IntValue).NotTo(BeNil(), "numaNode 应为整数")
	Expect(*attribute.IntValue).To(Equal(sysfs.numaNode), "设备 %s 的 numaNode", device.Name)
}

func topologyDevicesFromResourceSlice() map[int]resourcev1.Device {
	all, err := k8sClient.Clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "列出 ResourceSlice")

	matching := make([]resourcev1.ResourceSlice, 0)
	for _, slice := range all.Items {
		if slice.Spec.Driver == driverName && slice.Spec.NodeName != nil && *slice.Spec.NodeName == localNodeName {
			matching = append(matching, slice)
		}
	}
	Expect(matching).NotTo(BeEmpty(), "节点 %s 应有 npu.huawei.com ResourceSlice", localNodeName)

	poolName := matching[0].Spec.Pool.Name
	latestGeneration := matching[0].Spec.Pool.Generation
	for _, slice := range matching[1:] {
		Expect(slice.Spec.Pool.Name).To(Equal(poolName), "目标节点应只发布一个 NPU 资源池")
		if slice.Spec.Pool.Generation > latestGeneration {
			latestGeneration = slice.Spec.Pool.Generation
		}
	}

	current := make([]resourcev1.ResourceSlice, 0)
	for _, slice := range matching {
		if slice.Spec.Pool.Generation == latestGeneration {
			current = append(current, slice)
		}
	}
	Expect(current).To(HaveLen(int(current[0].Spec.Pool.ResourceSliceCount)), "ResourceSlice 当前 generation 必须完整")

	devices := make(map[int]resourcev1.Device)
	names := make(map[string]struct{})
	for _, slice := range current {
		for _, device := range slice.Spec.Devices {
			idAttribute, found := device.Attributes[resourcev1.QualifiedName(topologyPhysicalIDAttribute)]
			Expect(found).To(BeTrue(), "设备 %s 缺少 physicalID", device.Name)
			Expect(idAttribute.IntValue).NotTo(BeNil(), "设备 %s 的 physicalID 应为整数", device.Name)
			physicalID := int(*idAttribute.IntValue)
			_, duplicateID := devices[physicalID]
			Expect(duplicateID).To(BeFalse(), "ResourceSlice 存在重复 physicalID=%d", physicalID)
			_, duplicateName := names[device.Name]
			Expect(duplicateName).To(BeFalse(), "ResourceSlice 存在重复设备名 %s", device.Name)
			devices[physicalID] = device
			names[device.Name] = struct{}{}
		}
	}
	Expect(devices).NotTo(BeEmpty(), "节点 %s 的当前 ResourceSlice 应包含设备", localNodeName)
	return devices
}

func assertTopologyGroupContract(devices map[int]resourcev1.Device, attribute string) {
	groups := topologyIntegerGroups(devices, attribute)
	ids := sortedTopologyGroupIDs(groups)
	for index, groupID := range ids {
		Expect(groupID).To(Equal(int64(index)), "%s 分组 ID 应从 0 连续编号", attribute)
		Expect(len(groups[groupID])).To(BeNumerically(">=", 2), "%s 不应发布无拓扑边的单设备分组", attribute)
		if index > 0 {
			Expect(groups[ids[index-1]][0]).To(BeNumerically("<", groups[groupID][0]), "%s 应按分组最小 physicalID 排序", attribute)
		}
	}
}

func topologyIntegerGroups(devices map[int]resourcev1.Device, attribute string) map[int64][]int {
	groups := make(map[int64][]int)
	for physicalID, device := range devices {
		value, found := device.Attributes[resourcev1.QualifiedName(attribute)]
		if !found {
			continue
		}
		Expect(value.IntValue).NotTo(BeNil(), "设备 %s 的 %s 应为整数", device.Name, attribute)
		Expect(*value.IntValue).To(BeNumerically(">=", 0), "设备 %s 的 %s 不得为负数", device.Name, attribute)
		groups[*value.IntValue] = append(groups[*value.IntValue], physicalID)
	}
	for groupID := range groups {
		sort.Ints(groups[groupID])
	}
	return groups
}

func sortedTopologyGroupIDs(groups map[int64][]int) []int64 {
	ids := make([]int64, 0, len(groups))
	for groupID := range groups {
		ids = append(ids, groupID)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

func sortedTopologyPhysicalIDs(devices map[int]resourcev1.Device) []int {
	ids := make([]int, 0, len(devices))
	for physicalID := range devices {
		ids = append(ids, physicalID)
	}
	sort.Ints(ids)
	return ids
}

func hasTopologyGroupOfSize(groups map[int64][]int, count int) bool {
	for _, members := range groups {
		if len(members) >= count {
			return true
		}
	}
	return false
}

func topologyGroupSizes(groups map[int64][]int) []int {
	sizes := make([]int, 0, len(groups))
	for _, groupID := range sortedTopologyGroupIDs(groups) {
		sizes = append(sizes, len(groups[groupID]))
	}
	return sizes
}

func topologyStringGroupSizes(devices map[int]resourcev1.Device, attribute string) map[string]int {
	groups := make(map[string]int)
	for _, device := range devices {
		value, found := device.Attributes[resourcev1.QualifiedName(attribute)]
		if !found || value.StringValue == nil {
			continue
		}
		groups[*value.StringValue]++
	}
	return groups
}

func hccsNUMAIntersectionSizes(devices map[int]resourcev1.Device) map[string]int {
	groups := make(map[string]int)
	for _, device := range devices {
		hccs, hccsFound := topologyOptionalIntegerAttribute(device, topologyHCCSResourceAttribute)
		numa, numaFound := topologyOptionalIntegerAttribute(device, topologyNUMAResourceAttribute)
		if !hccsFound || !numaFound {
			continue
		}
		key := fmt.Sprintf("%d/%d", hccs, numa)
		groups[key]++
	}
	return groups
}

func hasHCCSNUMAGroup(devices map[int]resourcev1.Device, count int) bool {
	for _, size := range hccsNUMAIntersectionSizes(devices) {
		if size >= count {
			return true
		}
	}
	return false
}

func hasStringTopologyGroup(devices map[int]resourcev1.Device, attribute string, count int) bool {
	groups := make(map[string]int)
	for _, device := range devices {
		value := topologyStringAttribute(device, attribute)
		groups[value]++
		if groups[value] >= count {
			return true
		}
	}
	return false
}

// topologyRingOccupationExpression 构造精确匹配给定环成员的 CEL 选择器，
// 供 FIT_9 的占位 Claim 使用。
func topologyRingOccupationExpression(members []int) string {
	parts := make([]string, 0, len(members))
	for _, member := range members {
		parts = append(parts, topologyPhysicalIDExpression(member))
	}
	return strings.Join(parts, " || ")
}

func findSymmetricHCCSPairs(devices map[int]resourcev1.Device, count int) ([][2]int, bool) {
	ringRoots := make(map[int64]map[string][]int)
	for physicalID, device := range devices {
		ring, found := topologyOptionalIntegerAttribute(device, topologyHCCSResourceAttribute)
		if !found {
			continue
		}
		if _, found := ringRoots[ring]; !found {
			ringRoots[ring] = make(map[string][]int)
		}
		root := topologyStringAttribute(device, topologyPCIeRootResourceAttribute)
		ringRoots[ring][root] = append(ringRoots[ring][root], physicalID)
	}

	ringIDs := make([]int64, 0, len(ringRoots))
	for ringID := range ringRoots {
		ringIDs = append(ringIDs, ringID)
	}
	sort.Slice(ringIDs, func(i, j int) bool { return ringIDs[i] < ringIDs[j] })
	for leftIndex, leftRing := range ringIDs {
		for _, rightRing := range ringIDs[leftIndex+1:] {
			pairs := make([][2]int, 0)
			roots := make([]string, 0)
			for root := range ringRoots[leftRing] {
				if _, found := ringRoots[rightRing][root]; found {
					roots = append(roots, root)
				}
			}
			sort.Strings(roots)
			for _, root := range roots {
				leftMembers := ringRoots[leftRing][root]
				rightMembers := ringRoots[rightRing][root]
				sort.Ints(leftMembers)
				sort.Ints(rightMembers)
				pairCount := min(len(leftMembers), len(rightMembers))
				for index := 0; index < pairCount; index++ {
					pairs = append(pairs, [2]int{leftMembers[index], rightMembers[index]})
				}
			}
			if len(pairs) >= count {
				return pairs[:count], true
			}
		}
	}
	return nil, false
}

func topologyOptionalIntegerAttribute(device resourcev1.Device, attribute string) (int64, bool) {
	value, found := device.Attributes[resourcev1.QualifiedName(attribute)]
	if !found {
		return 0, false
	}
	Expect(value.IntValue).NotTo(BeNil(), "设备 %s 的 %s 应为整数", device.Name, attribute)
	return *value.IntValue, true
}

func topologyExactRequest(name string, count int64, expression string) resourcev1.DeviceRequest {
	request := resourcev1.DeviceRequest{Name: name, Exactly: &resourcev1.ExactDeviceRequest{
		DeviceClassName: deviceClassFull,
		AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
		Count:           count,
	}}
	if expression != "" {
		request.Exactly.Selectors = []resourcev1.DeviceSelector{{CEL: &resourcev1.CELDeviceSelector{Expression: expression}}}
	}
	return request
}

// topologyFirstAvailableRequest 构造 FIT_9 的 FirstAvailable 请求，按 CSV
// 原语义：第一优先级选 2 张同 HCCS 环卡，第二优先级降级选 2 张同 PCIe
// Root 卡。子请求级约束（引用 "npu/npu-same-hccs-ring" 等）由调度器只
// 作用于对应子请求的设备，两个优先级的语义天然区分。
func topologyFirstAvailableRequest() resourcev1.DeviceRequest {
	subrequest := func(name string) resourcev1.DeviceSubRequest {
		return resourcev1.DeviceSubRequest{Name: name, DeviceClassName: deviceClassFull,
			AllocationMode: resourcev1.DeviceAllocationModeExactCount, Count: 2}
	}
	return resourcev1.DeviceRequest{Name: "npu", FirstAvailable: []resourcev1.DeviceSubRequest{
		subrequest("npu-same-hccs-ring"),
		subrequest("npu-same-pcieroot"),
	}}
}

func topologyMatch(requests string, attribute resourcev1.FullyQualifiedName) resourcev1.DeviceConstraint {
	return resourcev1.DeviceConstraint{Requests: strings.Split(requests, ","), MatchAttribute: &attribute}
}

func topologyFullyQualifiedName(value string) resourcev1.FullyQualifiedName {
	return resourcev1.FullyQualifiedName(value)
}

func topologyPhysicalIDExpression(physicalID int) string {
	return fmt.Sprintf(`device.attributes["npu.huawei.com"].physicalID == %d`, physicalID)
}

// assertUnschedulableEventMentionsAllocation 验证调度器发出了 FailedScheduling
// 事件且消息为 DRA 分配失败（K8s 1.34 dynamicresources 插件对此只输出固定
// 消息 "cannot allocate all claims"，不携带具体拓扑属性名——真机验证见
// Issue #3，因此只断言 "cannot allocate" 子串）。
func assertUnschedulableEventMentionsAllocation(podName string) {
	// 扫描全部 FailedScheduling 事件：Pod 可能先收到非 DRA 原因的调度
	// 失败事件（如瞬时污点），只看第一条会饿死等待。
	Eventually(func() bool {
		events, err := k8sClient.Clientset.CoreV1().Events(testNamespace).List(ctx, metav1.ListOptions{
			FieldSelector: "involvedObject.name=" + podName,
		})
		if err != nil {
			return false
		}
		for _, event := range events.Items {
			if event.Reason == "FailedScheduling" && strings.Contains(event.Message, "cannot allocate") {
				return true
			}
		}
		return false
	}, 120*time.Second, pollInterval).Should(BeTrue(),
		"九卡同环请求的调度失败事件应表明 DRA 无法完成分配（K8s 调度器对此类失败仅输出通用消息，不含拓扑属性名）")
}

func topologySymmetricTwelveCardRequest(pairs [][2]int) ([]resourcev1.DeviceRequest, []resourcev1.DeviceConstraint) {
	Expect(pairs).To(HaveLen(6))
	requests := make([]resourcev1.DeviceRequest, 0, 12)
	leftRequests, rightRequests := make([]string, 0, 6), make([]string, 0, 6)
	for index, pair := range pairs {
		leftName := fmt.Sprintf("npu%d", index+1)
		rightName := fmt.Sprintf("npu%d", index+7)
		requests = append(requests,
			topologyExactRequest(leftName, 1, topologyPhysicalIDExpression(pair[0])),
			topologyExactRequest(rightName, 1, topologyPhysicalIDExpression(pair[1])),
		)
		leftRequests = append(leftRequests, leftName)
		rightRequests = append(rightRequests, rightName)
	}
	constraints := []resourcev1.DeviceConstraint{
		{Requests: leftRequests, MatchAttribute: &topologyHCCSAttribute},
		{Requests: rightRequests, MatchAttribute: &topologyHCCSAttribute},
	}
	for index := 0; index < 6; index++ {
		constraints = append(constraints, resourcev1.DeviceConstraint{Requests: []string{leftRequests[index], rightRequests[index]}, MatchAttribute: &topologyPCIeAttribute})
	}
	return requests, constraints
}

func startTopologyWorkload(caseName string, requests []resourcev1.DeviceRequest, constraints []resourcev1.DeviceConstraint) []topologyAllocatedDevice {
	claimName, podName := createTopologyClaimAndPod(caseName, requests, constraints)
	DeferCleanup(func() { cleanupTopologyClaimAndPod(claimName, podName) })
	waitForPodRunning(testNamespace, podName)
	Expect(verifyNPUSMIInPod(podName)).NotTo(BeEmpty(), "测试 Pod 内应可以执行 npu-smi info")
	GinkgoWriter.Printf("测试 Pod %s 可访问其分配到的 NPU 设备\n", podName)
	return topologyAllocatedDevices(claimName)
}

func createTopologyClaimAndPod(caseName string, requests []resourcev1.DeviceRequest, constraints []resourcev1.DeviceConstraint) (string, string) {
	name := fmt.Sprintf("topology-%s-%s", caseName, strconv.FormatInt(time.Now().UnixNano(), 36))
	claimName, podName := name, name
	claim := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: claimName, Namespace: testNamespace},
		Spec:       resourcev1.ResourceClaimSpec{Devices: resourcev1.DeviceClaim{Requests: requests, Constraints: constraints}},
	}
	_, err := k8sClient.Clientset.ResourceV1().ResourceClaims(testNamespace).Create(ctx, claim, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred(), "创建 ResourceClaim %s", claimName)

	pod := buildFullNPUPod(podName, "")
	pod.Spec.ResourceClaims = []corev1.PodResourceClaim{{Name: "npu", ResourceClaimName: ptrStr(claimName)}}
	_, err = k8sClient.Clientset.CoreV1().Pods(testNamespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		_ = k8sClient.Clientset.ResourceV1().ResourceClaims(testNamespace).Delete(ctx, claimName, metav1.DeleteOptions{})
	}
	Expect(err).NotTo(HaveOccurred(), "创建测试 Pod %s", podName)
	return claimName, podName
}

func cleanupTopologyClaimAndPod(claimName, podName string) {
	deletePodAndWait(testNamespace, podName)
	_ = k8sClient.Clientset.ResourceV1().ResourceClaims(testNamespace).Delete(ctx, claimName, metav1.DeleteOptions{})
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(ctx context.Context) (bool, error) {
		_, err := k8sClient.Clientset.ResourceV1().ResourceClaims(testNamespace).Get(ctx, claimName, metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	})
}

func assertTopologyRequestRemainsPending(claimName, podName string) {
	Eventually(func() bool {
		claim, claimErr := k8sClient.Clientset.ResourceV1().ResourceClaims(testNamespace).Get(ctx, claimName, metav1.GetOptions{})
		pod, podErr := k8sClient.Clientset.CoreV1().Pods(testNamespace).Get(ctx, podName, metav1.GetOptions{})
		if claimErr != nil || podErr != nil || claim.Status.Allocation != nil {
			return false
		}
		for _, condition := range pod.Status.Conditions {
			if condition.Type == corev1.PodScheduled && condition.Status == corev1.ConditionFalse && condition.Reason == corev1.PodReasonUnschedulable {
				return true
			}
		}
		return false
	}, 120*time.Second, pollInterval).Should(BeTrue(), "九卡同环请求应被调度器判定为不可满足")

	Consistently(func() bool {
		claim, claimErr := k8sClient.Clientset.ResourceV1().ResourceClaims(testNamespace).Get(ctx, claimName, metav1.GetOptions{})
		pod, podErr := k8sClient.Clientset.CoreV1().Pods(testNamespace).Get(ctx, podName, metav1.GetOptions{})
		return claimErr == nil && podErr == nil && claim.Status.Allocation == nil && pod.Status.Phase == corev1.PodPending
	}, 30*time.Second, pollInterval).Should(BeTrue(), "不可满足的九卡请求应持续 Pending 且不得产生分配")
}

type topologyAllocatedDevice struct {
	request string
	device  resourcev1.Device
}

func topologyAllocatedDevices(claimName string) []topologyAllocatedDevice {
	var claim *resourcev1.ResourceClaim
	Eventually(func() int {
		var err error
		claim, err = k8sClient.Clientset.ResourceV1().ResourceClaims(testNamespace).Get(ctx, claimName, metav1.GetOptions{})
		if err != nil || claim.Status.Allocation == nil {
			return 0
		}
		return len(claim.Status.Allocation.Devices.Results)
	}, pollTimeout, pollInterval).Should(BeNumerically(">", 0), "ResourceClaim %s 应完成分配", claimName)

	byName := topologyDevicesByName()
	allocated := make([]topologyAllocatedDevice, 0, len(claim.Status.Allocation.Devices.Results))
	for _, result := range claim.Status.Allocation.Devices.Results {
		if result.Driver != driverName {
			continue
		}
		device, found := byName[result.Device]
		Expect(found).To(BeTrue(), "分配设备 %s 应出现在 ResourceSlice", result.Device)
		GinkgoWriter.Printf("ResourceClaim %s 分配结果: request=%s device=%s 属性=%s\n", claimName, result.Request, result.Device, topologyDeviceAttributeSummary(device))
		allocated = append(allocated, topologyAllocatedDevice{request: result.Request, device: device})
	}
	return allocated
}

func topologyDevicesByName() map[string]resourcev1.Device {
	devices := make(map[string]resourcev1.Device)
	for _, device := range topologyDevicesFromResourceSlice() {
		devices[device.Name] = device
	}
	return devices
}

func assertSameTopologyAttribute(devices []topologyAllocatedDevice, attribute resourcev1.FullyQualifiedName) {
	Expect(devices).NotTo(BeEmpty())
	want := topologyAttributeValue(devices[0].device, attribute)
	for _, item := range devices[1:] {
		Expect(topologyAttributeValue(item.device, attribute)).To(Equal(want), "请求 %s 的 %s 应与同组一致", item.request, attribute)
	}
}

func topologyAttributeValue(device resourcev1.Device, attribute resourcev1.FullyQualifiedName) string {
	name := topologyResourceSliceAttributeName(attribute)
	value, found := device.Attributes[name]
	Expect(found).To(BeTrue(), "设备 %s 缺少属性 %s", device.Name, attribute)
	if value.IntValue != nil {
		return fmt.Sprintf("int:%d", *value.IntValue)
	}
	if value.StringValue != nil {
		return "string:" + *value.StringValue
	}
	Fail(fmt.Sprintf("设备 %s 的属性 %s 没有可比较值", device.Name, attribute))
	return ""
}

func topologyResourceSliceAttributeName(attribute resourcev1.FullyQualifiedName) resourcev1.QualifiedName {
	value := string(attribute)
	if strings.HasPrefix(value, driverName+"/") {
		value = strings.TrimPrefix(value, driverName+"/")
	}
	return resourcev1.QualifiedName(value)
}

func topologyStringAttribute(device resourcev1.Device, attribute string) string {
	value, found := device.Attributes[resourcev1.QualifiedName(attribute)]
	Expect(found).To(BeTrue(), "设备 %s 缺少属性 %s", device.Name, attribute)
	Expect(value.StringValue).NotTo(BeNil(), "设备 %s 的属性 %s 应为字符串", device.Name, attribute)
	return *value.StringValue
}

func topologyDeviceAttributeSummary(device resourcev1.Device) string {
	keys := make([]string, 0, len(device.Attributes))
	for key := range device.Attributes {
		keys = append(keys, string(key))
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		value := device.Attributes[resourcev1.QualifiedName(key)]
		switch {
		case value.IntValue != nil:
			parts = append(parts, fmt.Sprintf("%s=%d", key, *value.IntValue))
		case value.StringValue != nil:
			parts = append(parts, fmt.Sprintf("%s=%s", key, *value.StringValue))
		}
	}
	return strings.Join(parts, ", ")
}

func topologyIntegerAttribute(item topologyAllocatedDevice, attribute resourcev1.FullyQualifiedName) int64 {
	return topologyDeviceIntegerAttribute(item.device, attribute)
}

func topologyDeviceIntegerAttribute(device resourcev1.Device, attribute resourcev1.FullyQualifiedName) int64 {
	name := topologyResourceSliceAttributeName(attribute)
	value, found := device.Attributes[name]
	Expect(found).To(BeTrue(), "设备 %s 缺少属性 %s", device.Name, attribute)
	Expect(value.IntValue).NotTo(BeNil(), "设备 %s 的属性 %s 应为整数", device.Name, attribute)
	return *value.IntValue
}

func topologyDevicesForRequests(devices []topologyAllocatedDevice, requests []string) []topologyAllocatedDevice {
	byRequest := make(map[string]topologyAllocatedDevice, len(devices))
	for _, item := range devices {
		byRequest[item.request] = item
	}
	result := make([]topologyAllocatedDevice, 0, len(requests))
	for _, request := range requests {
		item, found := byRequest[request]
		Expect(found).To(BeTrue(), "请求 %s 应有分配结果", request)
		result = append(result, item)
	}
	return result
}

func topologyPhysicalIDs(devices []topologyAllocatedDevice) []int {
	ids := make([]int, 0, len(devices))
	for _, item := range devices {
		value, found := item.device.Attributes[resourcev1.QualifiedName(topologyPhysicalIDAttribute)]
		Expect(found).To(BeTrue(), "设备 %s 缺少 physicalID", item.device.Name)
		Expect(value.IntValue).NotTo(BeNil(), "设备 %s 的 physicalID 应为整数", item.device.Name)
		ids = append(ids, int(*value.IntValue))
	}
	sort.Ints(ids)
	return ids
}
