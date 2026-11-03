/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import crossContainer from '@/assets/images/crossContainer.png';
import containerNum from '@/assets/images/containerNum.png';
import cpu from '@/assets/images/cpu.png';
import memory from '@/assets/images/memory.png';
import { Table, Input, Modal, Tooltip } from 'antd';

import { useEffect, useState, useRef } from 'openinula';
import { convertToRanges, formatBytes, initializeNumaNodeData } from '@/utils/numa/common';
import topologyStrategy from '@/assets/images/topologyStrategy.png';
import { sorterFirstAlphabet } from '@/tools/utils';
import '@/styles/numa.less';
import { getCR, normalizeKeys } from '@/api/numaApi';
import { getDatArr, processAssignmentData, processContainerAssignments } from '@/utils/numa/prometheus';
import { parseAllocatable } from '@/utils/numa/common';
import { percenteHeight } from '@/tools/utils';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { useIntl } from 'inula-intl';

export default function NumaAllocate({ nodeName, cpuRange }) {
  const intl = useIntl();
  const { Search } = Input;
  const [infoValues, setInfoValues] = useState({
    containerNum: 'N/A',
    crossNumaContainerNum: 'N/A',
    cpuOccupiedNum: [],
    cpuOccupied: 'N/A',
    memoryOccupied: 'N/A',
  });

  const [dataSource, setDataSource] = useState([]);
  const [dataSourceInfo, setDataSourceInfo] = useState([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);

  const [containerInfo, setContainerInfo] = useState();

  const [current, setCurrent] = useState(1);

  const [cpuOccupied, setCpuOccupied] = useState(null);

  const [allCpuList, setAllCpuList] = useState([]);

  const [tableData, setTableData] = useState([]);

  const [headers, setHeaders] = useState([]);

  const queryCheckArr = [
    `container_numa_node_assignment{node='${nodeName}'}`,
    `sum(numa_node_memory_MemUsed{node='${nodeName}'})`,
  ];

  const infoArr = [
    { img: containerNum, num: infoValues.containerNum, title: intl.formatMessage({ id: 'numa.containerCount' }) },
    {
      img: crossContainer,
      num: infoValues.crossNumaContainerNum,
      title: intl.formatMessage({ id: 'numa.crossNumaContainerCount' }),
    },
    { img: cpu, num: infoValues.cpuOccupiedNum, title: intl.formatMessage({ id: 'numa.occupiedCpu' }) },
    {
      img: memory,
      num: formatBytes(infoValues.memoryOccupied),
      title: intl.formatMessage({ id: 'numa.occupiedMemory' }),
    },
  ];

  const columns = [
    {
      title: intl.formatMessage({ id: 'numa.containerName' }),
      dataIndex: 'container',
      key: 'container',
      sorter: (a, b) => sorterFirstAlphabet(a.container, b.container),
    },
    {
      title: intl.formatMessage({ id: 'numa.containerPod' }),
      dataIndex: 'podName',
      key: 'podName',
    },
    {
      title: intl.formatMessage({ id: 'numa.containerCpuMapping' }),
      dataIndex: 'cpuList',
      key: 'cpuList',
      render: (cpuList) => convertToRanges(cpuList),
    },
    {
      title: intl.formatMessage({ id: 'numa.isCrossNuma' }),
      dataIndex: 'crossNuma',
      key: 'crossNuma',
      filters: [
        {
          text: intl.formatMessage({ id: 'common.yes' }),
          value: intl.formatMessage({ id: 'common.yes' }),
        },
        {
          text: intl.formatMessage({ id: 'common.no' }),
          value: intl.formatMessage({ id: 'common.no' }),
        },
      ],
      onFilter: (value, record) => record.crossNuma === value,
    },
    {
      title: intl.formatMessage({ id: 'numa.containerCpuNuma' }),
      dataIndex: 'cpuNumaNode',
      key: 'cpuNumaNode',
      render: (cpuNumaNode) => {
        if (Array.isArray(cpuNumaNode)) {
          return cpuNumaNode.map((node) => `NUMA Node ${node}`).join(', ');
        }
        return cpuNumaNode;
      },
    },
    {
      title: intl.formatMessage({ id: 'numa.topology' }),
      dataIndex: 'topologyStrategy',
      render: (_, record) => (
        <img src={topologyStrategy} style={{ width: '20px', cursor: 'point' }} onClick={() => showModal(record)} />
      ),
    },
  ];

  const columns1 = headers.map((header, index) => ({
    title: header,
    dataIndex: `col${index}`, // 使用动态字段名
    key: `col${index}`,
  }));

  // 将表数据转换成对象格式，以匹配 columns 的 dataIndex
  const dataSourceListNumaAllocate = tableData.map((row, rowIndex) => {
    const rowObjNumaAllocate = {};
    row.forEach((cell, cellIndex) => {
      rowObjNumaAllocate[`col${cellIndex}`] = cell; // 使用动态字段名
    });
    return {
      key: rowIndex, // 为每行指定唯一键
      ...rowObjNumaAllocate,
    };
  });

  const showModal = (record) => {
    setContainerInfo(record);
    setOpen(true);
  };

  const handleData = async (question) => {
    setLoading(true);
    const prometheusResults = await getDatArr(queryCheckArr);
    const containerAssignments = prometheusResults[0]?.result ?? [];

    const {
      containerNum: processedContainerNum,
      cpuOccupied,
      cpuOccupiedNum,
      crossNumaContainerNum,
      dataSourceArray,
    } = await processContainerAssignments(containerAssignments, processAssignmentData);

    setInfoValues({
      containerNum: processedContainerNum,
      crossNumaContainerNum,
      cpuOccupied,
      cpuOccupiedNum,
      memoryOccupied: prometheusResults[1]?.result?.[0]?.value?.[1] || 'N/A',
    });

    let needCpuList = [];
    if (!question) {
      dataSourceArray.map((item) => {
        needCpuList = needCpuList.concat(item.cpuList);
      });
      setDataSource(dataSourceArray);
      setLoading(false);
    } else {
      const filelteredNodes = dataSourceArray.filter(
        (item) => item.podName && item.podName.toLowerCase().includes(question.toLowerCase()),
      );
      filelteredNodes.map((item) => {
        needCpuList = needCpuList.concat(item.cpuList);
      });
      setDataSource(filelteredNodes);
      setLoading(false);
    }
    setAllCpuList(needCpuList);
  };

  const handleSearch = (e) => {
    handleData(e);
  };
  const handleOk = () => {
    setConfirmLoading(true);
    setTimeout(() => {
      setOpen(false);
      setConfirmLoading(false);
    }, 2000);
  };

  const handleCancel = () => {
    setOpen(false);
  };

  const findDistance = (dataSourceInfoList, targetName, targetNode) => {
    // 遍历 dataSource 数组
    const element = dataSourceInfoList.find((item) => item.name === targetName);
    if (element) {
      // 在找到的元素的 distances 数组中查找 targetNode 对应的 distance
      const distanceItem = element.distances.find((item) => item.targetNode === targetNode);
      return distanceItem ? distanceItem.distance : '--';
    }
    return '--';
  };

  const getCpuItemClassName = (cpuElement, allocatableCpus, cpuList, cpuNumaNode) => {
    let baseClassName = '';
    if (allocatableCpus.includes(cpuElement)) {
      baseClassName = 'numa_node_number_occupy_item';
      if (cpuList.includes(cpuElement)) {
        baseClassName = 'container_occupy_item';
      }
    } else {
      baseClassName = 'numa_node_number_free_item';
    }
    return baseClassName;
  };

  const handlePageChange = (page, pageSize) => {
    setCurrent(page); // 更新当前页码
  };

  const createDataSourceArray = (initializedNodeData, numaNodeDistances) => {
    return Object.keys(initializedNodeData).map((numaNodeName) => {
      const nodeDistancesNumaAllocate = numaNodeDistances.get(numaNodeName) || new Map();
      const distancesArrayNumaAllocate = Array.from(nodeDistancesNumaAllocate.entries()).map(
        ([targetNode, distance]) => ({
          targetNode,
          distance,
        }),
      );
      return {
        name: numaNodeName,
        ...initializedNodeData[numaNodeName],
        distances: distancesArrayNumaAllocate,
      };
    });
  };

  const updateAllocatableCpuMapping = (numaNodeCpuMapping, allocatableCpus) => {
    const allocatableCpuSetNumaAllocate = parseAllocatable(allocatableCpus);

    Object.keys(numaNodeCpuMapping).forEach((numaId) => {
      const allNumaCpus = numaNodeCpuMapping[numaId].cpus;
      numaNodeCpuMapping[numaId].allocatableCpus = allNumaCpus.filter((cpuId) =>
        allocatableCpuSetNumaAllocate.has(cpuId),
      );
    });
  };

  const createQueryCheckArrNumaAllocate = (numaNodeName) => {
    return [
      `numa_node_memory_MemTotal{node="${nodeName}", numa_node="${numaNodeName}"}`,
      `numa_node_memory_MemUsed{node="${nodeName}", numa_node="${numaNodeName}"}`,
      `numa_node_distance{node="${nodeName}", source_numa_node="${numaNodeName}"}`,
    ];
  };

  const processDistanceMetrics = (result, numaNodeName, numaNodeDistances) => {
    const resultsNumaAllocate = result?.result ?? [];
    resultsNumaAllocate.forEach((entry) => {
      const sourceNumaNodeNumaAllocate = numaNodeName;
      const targetNumaNodeNumaAllocate = entry?.metric?.target_numa_node ?? 'N/A';
      const distanceValueNumaAllocate = entry?.value?.[1] ?? 'N/A';

      if (!numaNodeDistances.has(sourceNumaNodeNumaAllocate)) {
        numaNodeDistances.set(sourceNumaNodeNumaAllocate, new Map());
      }
      numaNodeDistances.get(sourceNumaNodeNumaAllocate).set(targetNumaNodeNumaAllocate, distanceValueNumaAllocate);
    });
  };

  const processPrometheusResults = (prometheusResults, numaNodeCpuMapping, initializedNodeData) => {
    const numaNodeDistances = new Map();
    if (prometheusResults) {
      prometheusResults.forEach((result, index) => {
        const numaNodeNameNumaAllocate = Object.keys(numaNodeCpuMapping)[Math.floor(index / 3)];
        const queryTypeNumaAllocate = index % 3;

        const metricValue = result?.result?.[0]?.value?.[1] ?? 'N/A';
        if (queryTypeNumaAllocate === 0) {
          initializedNodeData[numaNodeNameNumaAllocate].numaNodeMemoryTotal = metricValue;
        } else if (queryTypeNumaAllocate === 1) {
          initializedNodeData[numaNodeNameNumaAllocate].numaNodeMemoryUsed = metricValue;
        } else {
          processDistanceMetrics(result, numaNodeNameNumaAllocate, numaNodeDistances);
        }
      });
    }
    return numaNodeDistances;
  };
  const getNUMANodeCpuMapping = async (resourceName) => {
    try {
      const crDetailsNumaAllocate = await getCR(resourceName);
      if (!crDetailsNumaAllocate?.data?.data) {
        return null;
      }

      crDetailsNumaAllocate.data.data = normalizeKeys(crDetailsNumaAllocate.data.data);

      const cpuDetailNumaAllocate = crDetailsNumaAllocate.data.data?.spec?.cpudetail;
      const allocatableNumaAllocate = crDetailsNumaAllocate.data.data?.spec?.numares?.cpu?.allocatable;
      const numaNodeCpuMapping = {};

      if (cpuDetailNumaAllocate) {
        Object.keys(cpuDetailNumaAllocate).forEach((cpuId) => {
          const cpuInfo = cpuDetailNumaAllocate[cpuId];
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
      if (allocatableNumaAllocate) {
        updateAllocatableCpuMapping(numaNodeCpuMapping, allocatableNumaAllocate);
      }
      return numaNodeCpuMapping;
    } catch (error) {
      return null;
    }
  };

  const handleDataList = async () => {
    const numaNodeCpuMappingNumaAllocate = await getNUMANodeCpuMapping(nodeName);

    const initializedNodeDataNumaAllocate = initializeNumaNodeData(numaNodeCpuMappingNumaAllocate);

    const newQueryCheckArr = Object.keys(numaNodeCpuMappingNumaAllocate).flatMap((numaNodeName) =>
      createQueryCheckArrNumaAllocate(numaNodeName),
    );

    const prometheusResults = await getDatArr(newQueryCheckArr);

    const numaNodeDistances = processPrometheusResults(
      prometheusResults,
      numaNodeCpuMappingNumaAllocate,
      initializedNodeDataNumaAllocate,
    );
    let dataSourceArray = createDataSourceArray(initializedNodeDataNumaAllocate, numaNodeDistances);
    const allCpusNumaAllocate = dataSourceArray.flatMap((numaNodeData) => numaNodeData.cpus);

    const cpuOccupiedSet = new Set(cpuOccupied);

    const newAllocatableCpus = allCpusNumaAllocate.filter((occupiedCpu) => !cpuOccupiedSet.has(occupiedCpu));

    dataSourceArray = dataSourceArray.map((nodeData) => {
      const updatedAllocatableCpusNumaAllocate = nodeData.cpus.filter((availableCpu) =>
        newAllocatableCpus.includes(availableCpu),
      );
      return { ...nodeData, allocatableCpus: updatedAllocatableCpusNumaAllocate };
    });
    const headersList = [''].concat(dataSourceArray.map((item) => `Numa Node ${item.name}`));

    const tableDataListNumaAllocate = dataSourceArray.map((row) => {
      const rowData = [`Numa Node ${row.name}`];
      headersList.slice(1).forEach((targetNode) => {
        const [, , targetId] = targetNode.split(' ');
        const distance = row.distances.find((d) => d.targetNode === targetId)?.distance || 'NA';
        rowData.push(distance);
      });
      return rowData;
    });
    setHeaders(headersList);
    setTableData(tableDataListNumaAllocate);
    setDataSourceInfo(dataSourceArray);
  };

  useEffect(() => {
    if (nodeName) {
      handleData();
    }
  }, [nodeName]);

  useEffect(() => {
    handleDataList();
  }, []);

  return (
    <div>
      <Modal
        title={intl.formatMessage({ id: 'numa.topology' })}
        width={1347}
        className="numa_modal_allocate"
        getContainer={false}
        centered
        open={open}
        forceRender={true}
        footer={null}
        onOk={handleOk}
        confirmLoading={confirmLoading}
        onCancel={handleCancel}
      >
        <div className="numa_node_header">
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
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
            <div>
              {intl.formatMessage({ id: 'numa.container' })}: {containerInfo?.container}
            </div>
            <div className="container_occupy">{intl.formatMessage({ id: 'numa.containerOccupied' })}</div>
          </div>
        </div>
        <div className="numa_info_node_indoor">
          <div>
            <div className={`flex_container indoor_flex_container center_flex`}>
              {!!dataSourceInfo.length &&
                dataSourceInfo.map((item) => (
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
                          const baseClassName = getCpuItemClassName(
                            cpuItem,
                            allCpuList,
                            containerInfo?.cpuList || [],
                            containerInfo?.cpuNumaNode || [],
                          );
                          return (
                            <div className={baseClassName} key={cpuItem}>
                              {cpuItem < 10 ? `0${cpuItem}` : cpuItem}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
          <div style={{ fontSize: '16px', marginBottom: '20px', marginTop: '32px', fontWeight: 'bold' }}>
            Distance
            <Tooltip title={intl.formatMessage({ id: 'numa.distanceTooltip' })} placement="right">
              <QuestionCircleOutlined className="question-icon-hover" style={{ marginLeft: '7px' }} />
            </Tooltip>
          </div>
          <Table
            style={{ marginBottom: '60px' }}
            columns={columns1}
            dataSource={dataSourceListNumaAllocate}
            pagination={false} // 可以根据需要启用分页
            scroll={{ x: true }}
          />
        </div>
      </Modal>
      <div className="numa_indicator">
        <div style={{ display: 'flex', justifyContent: 'space-evenly' }}>
          {infoArr.map((items) => (
            <div style={{ display: 'flex', gap: '20px' }}>
              <div>
                <img src={items.img} style={{ width: '56px' }} />
              </div>
              <div>
                <div style={{ fontSize: '24px', fontWeight: '700' }}>{items.num === 'N/A' ? '0' : items.num}</div>
                <div style={{ color: '#89939b', fontSize: '16px' }}>{items.title}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="volcano-table">
        <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
          <Search
            placeholder={intl.formatMessage({ id: 'numa.searchPodName' })}
            style={{ width: '300px' }}
            onSearch={handleSearch}
          />
          <Tooltip title={intl.formatMessage({ id: 'numa.containerDisplayInfo' })} placement="right">
            <QuestionCircleOutlined className="question-icon-hover large-icon" />
          </Tooltip>
        </div>
        <Table
          locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
          dataSource={dataSource}
          columns={columns}
          loading={loading}
          className="numaAllocateTable"
          style={{ marginBottom: '20px' }}
          pagination={{
            className: 'volcano-page',
            current,
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
            showQuickJumper: true,
            onChange: handlePageChange,
            showTotal: (total) => intl.formatMessage({ id: 'common.totalItems' }, { total: dataSource.length }),
          }}
        />
      </div>
    </div>
  );
}
