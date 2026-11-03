/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import '@/styles/numa.less';
import { Input, Pagination, Table } from 'antd';

import { Link } from 'inula-router';
import { getCR, getNodeList, normalizeKeys } from '@/api/numaApi';
import { useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import { initializeNodeData, formatBytes, transformPercent, getNodeListInfo } from '@/utils/numa/common';
import { sortNaN, sorterFirstAlphabet } from '@/tools/utils';
import {
  getDatArr,
  mergePrometheusData,
  processAssignmentData,
  processContainerAssignments,
} from '@/utils/numa/prometheus';
import BreadCrumbCom from '@/components/BreadCrumbCom';

export default function NumaList() {
  const intl = useIntl();
  const { Search } = Input;
  const [dataSource, setDataSource] = useState([]);
  const [loading, setLoading] = useState(false);
  const [current, setCurrent] = useState(1);
  const [datas, setDatas] = useState();

  const queryCheckArr = [
    'numa_nodes_count',
    'sum(numa_node_cpus_count) by (node)',
    'sum(numa_node_memory_MemTotal) by (node)',
    'sum(numa_node_memory_MemUsed) by (node) / sum(numa_node_memory_MemTotal) by (node)',
  ];

  const createAssignmentQueryCheckArr = (nodeName) => {
    return [`container_numa_node_assignment{node='${nodeName}'}`];
  };

  const columns = [
    {
      title: intl.formatMessage({ id: 'numa.nodeName' }),
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.node, b.node),
      render: (_, record) => (
        <div>
          <Link
            to={{
              pathname: `/numaMonitor/numaDetails/${record.node}`,
              state: { numaSum: record.metricValues[0] === 'N/A' ? 0 : record.metricValues[0] },
            }}
          >
            {record.node}
          </Link>
        </div>
      ),
    },
    {
      title: intl.formatMessage({ id: 'numa.totalNumaNodes' }),
      dataIndex: 'metricValues',
      key: 'metricValues_0',
      sorter: (a, b, sorter) => sortNaN(a.metricValues[0], b.metricValues[0], sorter),
      render: (_, record) => (record.metricValues[0] === 'N/A' ? '0' : record.metricValues[0]),
    },
    {
      title: intl.formatMessage({ id: 'numa.totalCpu' }),
      dataIndex: 'metricValues',
      key: 'metricValues_1',
      sorter: (a, b, sorter) => sortNaN(a.metricValues[1], b.metricValues[1], sorter),
      render: (_, record) => (record.metricValues[1] === 'N/A' ? '0' : record.metricValues[1]),
    },
    {
      title: intl.formatMessage({ id: 'numa.totalMemory' }),
      dataIndex: 'metricValues',
      key: 'metricValues_2',
      sorter: (a, b, sorter) => sortNaN(a.metricValues[2], b.metricValues[2], sorter),
      render: (_, record) => formatBytes(record.metricValues[2]),
    },
    {
      title: intl.formatMessage({ id: 'numa.cpuAllocationRate' }),
      dataIndex: 'cpuAllocation',
      key: 'cpuAllocation',
      sorter: (a, b, sorter) => sortNaN(a.cpuAllocation, b.cpuAllocation, sorter),
      render: (_, record) => transformPercent(record.cpuAllocation),
    },
    {
      title: intl.formatMessage({ id: 'numa.memoryAllocationRate' }),
      dataIndex: 'metricValues',
      key: 'metricValues_3',
      sorter: (a, b, sorter) => sortNaN(a.metricValues[3], b.metricValues[3], sorter),
      render: (_, record) => transformPercent(record.metricValues[3]),
    },
    {
      title: intl.formatMessage({ id: 'numa.topologyPolicy' }),
      dataIndex: 'topologyPolicy',
      key: 'topologyPolicy',
      sorter: (a, b) => sorterFirstAlphabet(a.topologyPolicy, b.topologyPolicy),
      render: (_, record) => record.topologyPolicy,
    },
    {
      title: intl.formatMessage({ id: 'numa.containerCount' }),
      dataIndex: 'containerNum',
      key: 'containerNum',
      sorter: (a, b, sorter) => sortNaN(a.containerNum, b.containerNum, sorter),
      render: (_, record) => record.containerNum,
    },
    {
      title: intl.formatMessage({ id: 'numa.crossNumaContainerCount' }),
      dataIndex: 'crossNumaContainerNum',
      key: 'crossNumaContainerNum',
      sorter: (a, b, sorter) => sortNaN(a.crossNumaContainerNum, b.crossNumaContainerNum, sorter),
      render: (_, record) => record.crossNumaContainerNum,
    },
  ];

  const handleData = async (question) => {
    setLoading(true);

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
        const nodeName = node.metadata.name;
        const metricValues = mergedMetrics[nodeName];

        const crDetails = await getCR(nodeName);
        if (crDetails?.data) {
          crDetails.data = normalizeKeys(crDetails.data);
        }

        const prometheusAssignmentResults = await getDatArr(createAssignmentQueryCheckArr(nodeName));
        const containerAssignments = prometheusAssignmentResults[0]?.result ?? [];

        const { containerNum, crossNumaContainerNum, cpuOccupiedNum } = await processContainerAssignments(
          containerAssignments,
          processAssignmentData,
        );

        return {
          node: nodeName,
          topologyPolicy: crDetails?.data?.data?.spec?.policies?.topologymanagerpolicy || 'N/A',
          containerNum,
          cpuAllocation: metricValues[1] !== 'N/A' && metricValues[1] > 0 ? cpuOccupiedNum / metricValues[1] : 'N/A',
          crossNumaContainerNum,
          metricValues,
        };
      }),
    );
    setDatas(mergedData);
    if (!question) {
      setDataSource(mergedData);
      setLoading(false);
    } else {
      const filteredNodes = mergedData.filter(
        (item) => item.node && item.node.toLowerCase().includes(question.toLowerCase()),
      );
      setDataSource(filteredNodes);
      setLoading(false);
    }
  };

  useEffect(() => {
    handleData();
  }, []);

  const handleSearch = async (e) => {
    if (!e) {
      setDataSource(datas);
      setLoading(false);
    } else {
      const filteredNodes = datas.filter((item) => item.node && item.node.toLowerCase().includes(e.toLowerCase()));
      setDataSource(filteredNodes);
      setLoading(false);
    }
  };

  const handlePageChange = (page, pageSize) => {
    setCurrent(page); // 更新当前页码
  };
  return (
    <div className="out-table-list">
      <div style={{ marginBottom: '20px' }}>
        <BreadCrumbCom
          items={[
            { title: intl.formatMessage({ id: 'common.numaAffinityScheduling' }), disabled: true },
            { title: intl.formatMessage({ id: 'numa.clusterNumaMonitor' }), path: '/numaMonitor' },
          ]}
        />
      </div>
      <div className="numa_container">
        <div style={{ marginBottom: '20px' }}>
          <Search
            placeholder={intl.formatMessage({ id: 'numa.searchNodeName' })}
            style={{ width: '300px' }}
            onSearch={handleSearch}
          />
        </div>
        <div className="tab_table_flex">
          <Table
            locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
            dataSource={dataSource}
            columns={columns}
            loading={loading}
            className="numaListTable"
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
    </div>
  );
}
