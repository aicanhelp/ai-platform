/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import cpu from '@/assets/images/cpu.png';
import numaNode from '@/assets/images/numaNode.png';
import memory from '@/assets/images/memory.png';
import cpuAllocationRate from '@/assets/images/cpuAllocationRate.png';
import memoryAllocationRate from '@/assets/images/memoryAllocationRate.png';
import topologyStrategy from '@/assets/images/topologyStrategy.png';
import { Table, Tooltip } from 'antd';

import { getCR, normalizeKeys } from '@/api/numaApi';
import { useEffect, useState, useRef } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/numa.less';
import {
  convertToRanges,
  initializeNodeData,
  formatBytes,
  transformPercent,
  getNodeListInfo,
  initializeNumaNodeData,
  parseAllocatable,
} from '@/utils/numa/common';
import { sorterFirstAlphabet } from '@/tools/utils';
import {
  getDatArr,
  processAssignmentData,
  processContainerAssignments,
  mergePrometheusData,
} from '@/utils/numa/prometheus';
import { percenteHeight } from '@/tools/utils';
import { QuestionCircleOutlined } from '@ant-design/icons';

export default function NumaDetail({ nodeName, cpuRange }) {
  const intl = useIntl();
  const [dataSource, setDataSource] = useState([]);

  const [dataList, setDataList] = useState();

  const [cpuOccupied, setCpuOccupied] = useState(null);

  const [current, setCurrent] = useState(1);

  const [tableData, setTableData] = useState([]);

  const [headers, setHeaders] = useState([]);

  const queryCheckArr = [
    'numa_nodes_count',
    'sum(numa_node_cpus_count) by (node)',
    'sum(numa_node_memory_MemTotal) by (node)',
    'sum(numa_node_memory_MemUsed) by (node) / sum(numa_node_memory_MemTotal) by (node)',
  ];

  const createAssignmentQueryCheckArr = () => {
    return [`container_numa_node_assignment{node='${nodeName}'}`];
  };

  const infoArr = [
    {
      img: numaNode,
      num: dataList?.metricValues[0] === 'N/A' ? '0' : dataList?.metricValues[0],
      title: intl.formatMessage({ id: 'numa.totalNumaNodes' }),
    },
    {
      img: cpu,
      num: dataList?.metricValues[1] === 'N/A' ? '0' : dataList?.metricValues[1],
      title: intl.formatMessage({ id: 'numa.totalCpu' }),
    },
    { img: memory, num: formatBytes(dataList?.metricValues[2]), title: intl.formatMessage({ id: 'numa.totalMemory' }) },
    {
      img: cpuAllocationRate,
      num: dataList?.cpuAllocation,
      title: intl.formatMessage({ id: 'numa.cpuAllocationRate' }),
    },
    {
      img: memoryAllocationRate,
      num: dataList?.metricValues[3],
      title: intl.formatMessage({ id: 'numa.memoryAllocationRate' }),
    },
    { img: topologyStrategy, num: dataList?.topologyPolicy, title: intl.formatMessage({ id: 'numa.topologyPolicy' }) },
  ];

  const columns = [
    {
      title: intl.formatMessage({ id: 'numa.numaNodeName' }),
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (name) => `NUMA Node ${name}`,
    },
    {
      title: intl.formatMessage({ id: 'numa.cpuList' }),
      dataIndex: 'cpus',
      key: 'cpus',
      render: (cpus) => convertToRanges(cpus),
    },
    {
      title: intl.formatMessage({ id: 'numa.freeCpuList' }),
      dataIndex: 'allocatableCpus',
      key: 'allocatableCpus',
      render: (allocatableCpus) => convertToRanges(allocatableCpus),
    },
    {
      title: intl.formatMessage({ id: 'numa.totalMemory' }),
      dataIndex: 'numaNodeMemoryTotal',
      key: 'numaNodeMemoryTotal',
      render: (numaNodeMemoryTotal) => formatBytes(numaNodeMemoryTotal),
    },
    {
      title: intl.formatMessage({ id: 'numa.usedMemory' }),
      dataIndex: 'numaNodeMemoryUsed',
      key: 'numaNodeMemoryUsed',
      render: (numaNodeMemoryUsed) => formatBytes(numaNodeMemoryUsed),
    },
  ];

  const columns1 = headers.map((header, index) => {
    const colConfig = {
      title: header,
      dataIndex: `col${index}`, // 使用动态字段名
      key: `col${index}`,
    };

    // 对第一列做特别处理，加粗显示
    if (index === 0) {
      colConfig.render = (text) => <strong>{text}</strong>;
    }
    return colConfig;
  });

  // 将表数据转换成对象格式，以匹配 columns 的 dataIndex
  const dataSourceList = tableData.map((row, rowIndex) => {
    const rowObj = {};
    row.forEach((cell, cellIndex) => {
      rowObj[`col${cellIndex}`] = cell; // 使用动态字段名
    });
    return {
      key: rowIndex, // 为每行指定唯一键
      ...rowObj,
    };
  });

  const getNUMANodeCpuMapping = async (resourceName) => {
    try {
      const crDetailsNumaDetail = await getCR(resourceName);
      if (!crDetailsNumaDetail?.data?.data) {
        return null;
      }

      crDetailsNumaDetail.data.data = normalizeKeys(crDetailsNumaDetail.data.data);

      const cpuDetailNumaDetail = crDetailsNumaDetail.data.data?.spec?.cpudetail;
      const allocatableNumaDetail = crDetailsNumaDetail.data.data?.spec?.numares?.cpu?.allocatable;
      const numaNodeCpuMapping = {};

      if (cpuDetailNumaDetail) {
        Object.keys(cpuDetailNumaDetail).forEach((cpuId) => {
          const cpuInfo = cpuDetailNumaDetail[cpuId];
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
      if (allocatableNumaDetail) {
        updateAllocatableCpuMapping(numaNodeCpuMapping, allocatableNumaDetail);
      }
      return numaNodeCpuMapping;
    } catch (error) {
      return null;
    }
  };

  const updateAllocatableCpuMapping = (numaNodeCpuMapping, allocatableCpus) => {
    const allocatableCpuSetNumaDetailNumaDetail = parseAllocatable(allocatableCpus);

    Object.keys(numaNodeCpuMapping).forEach((numaId) => {
      const allNumaCpus = numaNodeCpuMapping[numaId].cpus;
      numaNodeCpuMapping[numaId].allocatableCpus = allNumaCpus.filter((cpuId) =>
        allocatableCpuSetNumaDetailNumaDetail.has(cpuId),
      );
    });
  };

  const createQueryCheckArrNumaDetail = (numaNodeName) => {
    return [
      `numa_node_memory_MemTotal{node="${nodeName}", numa_node="${numaNodeName}"}`,
      `numa_node_memory_MemUsed{node="${nodeName}", numa_node="${numaNodeName}"}`,
      `numa_node_distance{node="${nodeName}", source_numa_node="${numaNodeName}"}`,
    ];
  };

  const processPrometheusResults = (prometheusResults, numaNodeCpuMapping, initializedNodeData) => {
    const numaNodeDistances = new Map();
    if (prometheusResults) {
      prometheusResults.forEach((result, index) => {
        const numaNodeNameNumaDetail = Object.keys(numaNodeCpuMapping)[Math.floor(index / 3)];
        const queryTypeNumaDetail = index % 3;

        const metricValue = result?.result?.[0]?.value?.[1] ?? 'N/A';
        if (queryTypeNumaDetail === 0) {
          initializedNodeData[numaNodeNameNumaDetail].numaNodeMemoryTotal = metricValue;
        } else if (queryTypeNumaDetail === 1) {
          initializedNodeData[numaNodeNameNumaDetail].numaNodeMemoryUsed = metricValue;
        } else {
          processDistanceMetrics(result, numaNodeNameNumaDetail, numaNodeDistances);
        }
      });
    }
    return numaNodeDistances;
  };

  const processDistanceMetrics = (result, numaNodeName, numaNodeDistances) => {
    const resultsNumaDetail = result?.result ?? [];
    resultsNumaDetail.forEach((entry) => {
      const sourceNumaNodeNumaDetail = numaNodeName;
      const targetNumaNodeNumaDetail = entry?.metric?.target_numa_node ?? 'N/A';
      const distanceValueNumaDetail = entry?.value?.[1] ?? 'N/A';

      if (!numaNodeDistances.has(sourceNumaNodeNumaDetail)) {
        numaNodeDistances.set(sourceNumaNodeNumaDetail, new Map());
      }
      numaNodeDistances.get(sourceNumaNodeNumaDetail).set(targetNumaNodeNumaDetail, distanceValueNumaDetail);
    });
  };

  const createDataSourceArray = (initializedNodeData, numaNodeDistances) => {
    return Object.keys(initializedNodeData).map((numaNodeName) => {
      const nodeDistancesNumaDetail = numaNodeDistances.get(numaNodeName) || new Map();
      const distancesArrayNumaDetail = Array.from(nodeDistancesNumaDetail.entries()).map(([targetNode, distance]) => ({
        targetNode,
        distance,
      }));
      return {
        name: numaNodeName,
        ...initializedNodeData[numaNodeName],
        distances: distancesArrayNumaDetail,
      };
    });
  };

  const handleDataList = async () => {
    const nodes = await getNodeListInfo();
    if (!nodes) {
      return;
    }

    const defaultValue = 'N/A';
    const initializedNodeData = initializeNodeData(nodes, 7, defaultValue);
    let mergedMetrics = initializedNodeData;

    const prometheusResults = await getDatArr(queryCheckArr);

    if (prometheusResults) {
      mergedMetrics = mergePrometheusData(prometheusResults, initializedNodeData);
    }

    const mergedData = await Promise.all(
      nodes.map(async (node) => {
        const nodeMetadataName = node.metadata.name;
        const metricValues = mergedMetrics[nodeMetadataName];

        const crDetailsNumaDetail = await getCR(nodeMetadataName);
        if (crDetailsNumaDetail?.data) {
          crDetailsNumaDetail.data = normalizeKeys(crDetailsNumaDetail.data);
        }

        const prometheusAssignmentResults = await getDatArr(createAssignmentQueryCheckArr());
        const containerAssignments = prometheusAssignmentResults[0]?.result ?? [];

        const {
          containerNum,
          crossNumaContainerNum,
          cpuOccupied: proccessedCpuOccupied,
          cpuOccupiedNum,
        } = await processContainerAssignments(containerAssignments, processAssignmentData);

        return {
          node: nodeMetadataName,
          topologyPolicy: crDetailsNumaDetail?.data?.data?.spec?.policies?.topologymanagerpolicy || 'N/A',
          containerNum,
          cpuOccupied: proccessedCpuOccupied,
          cpuAllocation: metricValues[1] !== 'N/A' && metricValues[1] > 0 ? cpuOccupiedNum / metricValues[1] : 'N/A',
          crossNumaContainerNum,
          metricValues,
        };
      }),
    );

    const result = mergedData.find((item) => item.node === nodeName);
    setDataList(result);
    setCpuOccupied(result?.cpuOccupied);
  };

  const handleData = async () => {
    const numaNodeCpuMappingNumaDetail = await getNUMANodeCpuMapping(nodeName);

    const initializedNodeDataNumaDetail = initializeNumaNodeData(numaNodeCpuMappingNumaDetail);

    const newQueryCheckArr = Object.keys(numaNodeCpuMappingNumaDetail).flatMap((numaNodeName) =>
      createQueryCheckArrNumaDetail(numaNodeName),
    );

    const prometheusResults = await getDatArr(newQueryCheckArr);

    const numaNodeDistances = processPrometheusResults(
      prometheusResults,
      numaNodeCpuMappingNumaDetail,
      initializedNodeDataNumaDetail,
    );
    let dataSourceArray = createDataSourceArray(initializedNodeDataNumaDetail, numaNodeDistances);
    const allCpus = dataSourceArray.flatMap((numaNodeData) => numaNodeData.cpus);

    const cpuOccupiedSet = new Set(cpuOccupied);

    const newAllocatableCpus = allCpus.filter((occupiedCpu) => !cpuOccupiedSet.has(occupiedCpu));

    dataSourceArray = dataSourceArray.map((nodeData) => {
      const updatedAllocatableCpusNumaDetail = nodeData.cpus.filter((availableCpu) =>
        newAllocatableCpus.includes(availableCpu),
      );
      return { ...nodeData, allocatableCpus: updatedAllocatableCpusNumaDetail };
    });

    const headersList = [''].concat(dataSourceArray.map((item) => `Numa Node ${item.name}`));

    const tableDataListNumaDetail = dataSourceArray.map((row) => {
      const rowData = [`Numa Node ${row.name}`];
      headersList.slice(1).forEach((targetNode) => {
        const [, , targetId] = targetNode.split(' ');
        const distance = row.distances.find((d) => d.targetNode === targetId)?.distance || 'NA';
        rowData.push(distance);
      });
      return rowData;
    });
    setHeaders(headersList);
    setTableData(tableDataListNumaDetail);
    setDataSource(dataSourceArray);
  };

  const findDistance = (dataSourceInfo, targetName, targetNode) => {
    // 遍历 dataSource 数组
    const element = dataSourceInfo.find((item) => item.name === targetName);
    if (element) {
      // 在找到的元素的 distances 数组中查找 targetNode 对应的 distance
      const distanceItem = element.distances.find((item) => item.targetNode === targetNode);
      return distanceItem ? distanceItem.distance : '--';
    }
    return '--';
  };

  const previousCpuOccupied = useRef(null);

  const handlePageChange = (page, pageSize) => {
    setCurrent(page); // 更新当前页码
  };

  useEffect(() => {
    const fetchDataList = async () => {
      if (nodeName) {
        await handleDataList(nodeName);
      }
    };

    fetchDataList();
  }, [nodeName]);

  useEffect(() => {
    const fetchData = async () => {
      if (cpuOccupied !== null && JSON.stringify(cpuOccupied) !== JSON.stringify(previousCpuOccupied.current)) {
        previousCpuOccupied.current = cpuOccupied;
        handleData(nodeName);
      }
    };

    fetchData();
  }, [cpuOccupied, nodeName]);

  return (
    <div>
      <div className="numa_indicator">
        <div style={{ display: 'flex', justifyContent: 'space-evenly', position: 'relative' }}>
          {infoArr.map((items) => (
            <div className="numa_list" style={{ display: 'flex', gap: '20px' }}>
              <div>
                <img className="numa_img_display" src={items.img} />
              </div>
              <div>
                <div className="numa_title_bold">
                  {items.title === intl.formatMessage({ id: 'numa.cpuAllocationRate' }) ||
                  items.title === intl.formatMessage({ id: 'numa.memoryAllocationRate' })
                    ? transformPercent(items.num)
                    : items.num}
                </div>
                <div className="numa_title_normal">{items.title}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="numa_info_node">
        <div>
          <div className="numa_node_header">
            <div className="numa_node_title">{intl.formatMessage({ id: 'numa.numaNodeInfoChart' })}</div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <div>CPU: {cpuRange}</div>
              <div
                className="numa_node_number_free_item"
                style={{ width: 'auto', minWidth: '66px', height: '24px', padding: '0 8px' }}
              >
                {intl.formatMessage({ id: 'numa.notOccupied' })}
              </div>
              <div
                className="numa_node_number_occupy_item"
                style={{ width: 'auto', minWidth: '66px', height: '24px', padding: '0 8px' }}
              >
                {intl.formatMessage({ id: 'numa.occupied' })}
              </div>
              <Tooltip title={intl.formatMessage({ id: 'numa.containerDisplayInfo' })} placement="top">
                <QuestionCircleOutlined className="question-icon-hover large-icon" />
              </Tooltip>
            </div>
          </div>
          <div className="numa_topo_container">
            <div className={`flex_container outdoor_flex_container_detail center_flex`}>
              {!!dataSource.length &&
                dataSource.map((item) => (
                  <div className={`numa_node_container`}>
                    <div style={{ display: 'flex', justifyContent: 'center', flexDirection: 'column' }}>
                      <div className="numa_node_memory_container">
                        <div className="numa_node_memory_free">
                          {intl.formatMessage({ id: 'numa.free' })}：
                          {formatBytes(item.numaNodeMemoryTotal - item.numaNodeMemoryUsed)}
                        </div>
                        <div
                          className="numa_node_memory_occupy"
                          style={{ height: percenteHeight(item?.numaNodeMemoryUsed, item?.numaNodeMemoryTotal) }}
                        ></div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        {intl.formatMessage({ id: 'numa.totalMemory' })}: {formatBytes(item.numaNodeMemoryTotal)}
                      </div>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <div style={{ textAlign: 'center', fontWeight: 'bold' }}></div>
                      <div className="liner_length_10"></div>
                    </div>
                    <div className={'numa_node_number_container'} style={{ overflowY: 'auto' }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '18px' }}>NUMA Node {item.name}</div>
                      <div className="grid_template_columns_5">
                        {item.cpus.map((cpuItem) => {
                          return (
                            <div
                              className={
                                item.allocatableCpus && item.allocatableCpus.includes(cpuItem)
                                  ? 'numa_node_number_free_item'
                                  : 'numa_node_number_occupy_item'
                              }
                              key={cpuItem}
                            >
                              {cpuItem < 10 ? `0${cpuItem}` : cpuItem}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
            <div style={{ fontSize: '16px', margin: '40px 0 20px 0', fontWeight: 'bold' }}>
              distance
              <Tooltip title={intl.formatMessage({ id: 'numa.distanceTooltip' })} placement="right">
                <QuestionCircleOutlined className="question-icon-hover" style={{ marginLeft: '7px' }} />
              </Tooltip>
            </div>
            <Table
              locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
              columns={columns1}
              dataSource={dataSourceList}
              pagination={false} // 可以根据需要启用分页
              scroll={{ x: true }}
            />
          </div>
        </div>
      </div>
      <div className="volcano-table">
        <Table
          locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
          dataSource={dataSource}
          columns={columns}
          pagination={{
            className: 'volcano-page',
            current,
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
            showQuickJumper: true,
            onChange: handlePageChange,
            showTotal: (total) => intl.formatMessage({ id: 'common.totalItems' }, { total: dataSource.length }),
          }}
          style={{ marginBottom: '20px' }}
        />
      </div>
    </div>
  );
}
