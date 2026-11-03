/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Tabs } from 'antd';
import NumaDetail from '@/pages/NUMA/NumaDetail';
import NumaAllocate from '@/pages/NUMA/NumaAllocate';
import NumaAffinity from '@/pages/NUMA/NumaAffinity';
import { useLocation, useParams } from 'inula-router';
import '@/styles/numa.less';
import { getCR, normalizeKeys } from '@/api/numaApi';
import { initializeNumaNodeData, parseAllocatable } from '@/utils/numa/common';
import { useEffect, useState } from 'openinula';
import { convertToRanges } from '@/utils/numa/common';
import { getDatArr, processAssignmentData, processContainerAssignments } from '@/utils/numa/prometheus';
import { useIntl } from 'inula-intl';

export default function NumaDetails() {
  const intl = useIntl();
  const {
    state: { numaSum },
  } = useLocation();

  const routerParam = useParams();

  const nodeName = routerParam.numaDetailName;

  const [dataSource, setDataSource] = useState([]);

  const [cpuRange, setCpuRange] = useState('');

  const getNUMANodeCpuMapping = async (resourceName) => {
    try {
      const crDetailsNumaDetails = await getCR(resourceName);
      if (!crDetailsNumaDetails?.data?.data) {
        return null;
      }

      crDetailsNumaDetails.data.data = normalizeKeys(crDetailsNumaDetails.data.data);

      const cpuDetailNumaDetails = crDetailsNumaDetails.data.data?.spec?.cpudetail;
      const allocatableNumaDetails = crDetailsNumaDetails.data.data?.spec?.numares?.cpu?.allocatable;
      const numaNodeCpuMapping = {};

      if (cpuDetailNumaDetails) {
        Object.keys(cpuDetailNumaDetails).forEach((cpuId) => {
          const cpuInfo = cpuDetailNumaDetails[cpuId];
          const numaId = cpuInfo.numa || '0';

          if (!numaNodeCpuMapping[numaId]) {
            numaNodeCpuMapping[numaId] = {
              cpus: [],
              allocatableCpus: [],
            };
          }
          numaNodeCpuMapping[numaId].cpus.push(cpuId);
        });
      }
      if (allocatableNumaDetails) {
        updateAllocatableCpuMapping(numaNodeCpuMapping, allocatableNumaDetails);
      }
      return numaNodeCpuMapping;
    } catch (error) {
      return null;
    }
  };

  const updateAllocatableCpuMapping = (numaNodeCpuMapping, allocatableCpus) => {
    const nodeDistancesNumaDetails = parseAllocatable(allocatableCpus);

    Object.keys(numaNodeCpuMapping).forEach((numaId) => {
      const allNumaCpus = numaNodeCpuMapping[numaId].cpus;
      numaNodeCpuMapping[numaId].allocatableCpus = allNumaCpus.filter((cpuId) => nodeDistancesNumaDetails.has(cpuId));
    });
  };

  const createQueryCheckArrNumaDetails = (numaNodeName) => {
    return [
      `numa_node_memory_MemTotal{node="${nodeName}", numa_node="${numaNodeName}"}`,
      `numa_node_memory_MemUsed{node="${nodeName}", numa_node="${numaNodeName}"}`,
      `numa_node_distance{node="${nodeName}", source_numa_node="${numaNodeName}"}`,
    ];
  };

  const createAssignmentQueryCheckArr = () => {
    return [`container_numa_node_assignment{node='${nodeName}'}`];
  };

  const processPrometheusResults = (prometheusResults, numaNodeCpuMapping, initializedNodeData) => {
    const numaNodeDistances = new Map();
    if (prometheusResults) {
      prometheusResults.forEach((result, index) => {
        const numaNodeNameNumaDetails = Object.keys(numaNodeCpuMapping)[Math.floor(index / 3)];
        const queryTypeNumaDetails = index % 3;

        const metricValue = result?.result?.[0]?.value?.[1] ?? 'N/A';
        if (queryTypeNumaDetails === 0) {
          initializedNodeData[numaNodeNameNumaDetails].numaNodeMemoryTotal = metricValue;
        } else if (queryTypeNumaDetails === 1) {
          initializedNodeData[numaNodeNameNumaDetails].numaNodeMemoryUsed = metricValue;
        } else {
          processDistanceMetrics(result, numaNodeNameNumaDetails, numaNodeDistances);
        }
      });
    }
    return numaNodeDistances;
  };

  const processDistanceMetrics = (result, numaNodeName, numaNodeDistances) => {
    const resultsNumaDetails = result?.result ?? [];
    resultsNumaDetails.forEach((entry) => {
      const sourceNumaNodeNumaDetails = numaNodeName;
      const targetNumaNodeNumaDetails = entry?.metric?.target_numa_node ?? 'N/A';
      const distanceValueNumaDetails = entry?.value?.[1] ?? 'N/A';

      if (!numaNodeDistances.has(sourceNumaNodeNumaDetails)) {
        numaNodeDistances.set(sourceNumaNodeNumaDetails, new Map());
      }
      numaNodeDistances.get(sourceNumaNodeNumaDetails).set(targetNumaNodeNumaDetails, distanceValueNumaDetails);
    });
  };

  const createDataSourceArray = (initializedNodeData, numaNodeDistances) => {
    return Object.keys(initializedNodeData).map((numaNodeName) => {
      const nodeDistancesNumaDetails = numaNodeDistances.get(numaNodeName) || new Map();
      const distancesArrayNumaDetails = Array.from(nodeDistancesNumaDetails.entries()).map(
        ([targetNode, distance]) => ({
          targetNode,
          distance,
        }),
      );
      return {
        name: numaNodeName,
        ...initializedNodeData[numaNodeName],
        distances: distancesArrayNumaDetails,
      };
    });
  };

  const handleData = async () => {
    const numaNodeCpuMapping = await getNUMANodeCpuMapping(nodeName);

    const initializedNodeData = initializeNumaNodeData(numaNodeCpuMapping);

    const queryCheckArr = Object.keys(numaNodeCpuMapping).flatMap((numaNodeName) =>
      createQueryCheckArrNumaDetails(numaNodeName),
    );

    const prometheusResults = await getDatArr(queryCheckArr);

    const numaNodeDistances = processPrometheusResults(prometheusResults, numaNodeCpuMapping, initializedNodeData);

    const prometheusAssignmentResults = await getDatArr(createAssignmentQueryCheckArr());
    const containerAssignments = prometheusAssignmentResults[0]?.result ?? [];

    const { cpuOccupied } = await processContainerAssignments(containerAssignments, processAssignmentData);

    let dataSourceArray = createDataSourceArray(initializedNodeData, numaNodeDistances);

    const allCpus = dataSourceArray.flatMap((numaNodeData) => numaNodeData.cpus);

    const cpuOccupiedSet = new Set(cpuOccupied);

    const newAllocatableCpus = allCpus.filter((occupiedCpu) => !cpuOccupiedSet.has(occupiedCpu));

    dataSourceArray = dataSourceArray.map((nodeData) => {
      const updatedAllocatableCpus = nodeData.cpus.filter((cpu) => newAllocatableCpus.includes(cpu));
      return { ...nodeData, allocatableCpus: updatedAllocatableCpus };
    });

    setDataSource(dataSourceArray);

    const uniqueCpus = [...new Set(allCpus)].sort((a, b) => a - b);
    setCpuRange(convertToRanges(uniqueCpus));
  };

  useEffect(() => {
    if (nodeName) {
      handleData();
    }
  }, [nodeName]);

  const items = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'numa.numaInfo' }),
      children: <NumaDetail nodeName={nodeName} />,
    },
    {
      key: '2',
      label: intl.formatMessage({ id: 'numa.containerNumaAllocation' }),
      children: <NumaAllocate nodeName={nodeName} dataSourceInfo={dataSource} cpuRange={cpuRange} />,
    },
    {
      key: '3',
      label: intl.formatMessage({ id: 'numa.affinityOptimizedPods' }),
      children: <NumaAffinity nodeName={nodeName} numaSum={numaSum} />,
    },
  ];
  return (
    <div className="numa_container_details">
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'common.numaAffinityScheduling' }), disabled: true },
          { title: intl.formatMessage({ id: 'numa.clusterNumaMonitor' }), path: '/numaMonitor' },
          { title: intl.formatMessage({ id: 'numa.nodeNumaDetails' }), path: '/numaMonitor/numaDetails' },
        ]}
      />
      <div className="node_name">{nodeName}</div>
      <Tabs items={items} style={{ backgroundColor: '#fff' }} />
    </div>
  );
}
