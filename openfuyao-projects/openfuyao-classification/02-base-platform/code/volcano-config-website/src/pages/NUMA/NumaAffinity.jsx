/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import { Table, Space } from 'antd';

import { useEffect, useState, useCallback } from 'openinula';
import '@/styles/numa.less';
import { getPodAffinityList } from '@/api/schedulingApi';
import { ResponseCode } from '@/common/constants';
import { convertToRanges } from '@/utils/numa/common';
import { sorterFirstAlphabet } from '@/tools/utils';
import { useIntl } from 'inula-intl';

export default function NumaDetail({ nodeName, numaSum }) {
  const intl = useIntl();
  const [dataSource, setDataSource] = useState([]);
  const [current, setCurrent] = useState(1);
  const columns = [
    {
      title: intl.formatMessage({ id: 'numa.numaName' }),
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <Space size="middle" className="">
          {record.name}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'numa.affinityPodCount' }),
      dataIndex: 'sum',
      key: 'sum',
      sorter: (a, b) => sorterFirstAlphabet(a.sum, b.sum),
      render: (_, record) => (
        <Space size="middle" className="">
          {record.sum}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'numa.networkAffinityPodName' }),
      dataIndex: 'netName',
      key: 'netName',
      render: (_, record) => <div style={{ wordBreak: 'break-all' }}>{record.netName}</div>,
    },
  ];

  const handlePageChange = (page, pageSize) => {
    setCurrent(page); // 更新当前页码
  };

  const getDataSource = useCallback(async () => {
    const res = await getPodAffinityList();
    if (res.status === ResponseCode.OK) {
      const [filterCurrentNodeData] = res.data.spec.node.filter((item) => item.name === nodeName);
      if (filterCurrentNodeData) {
        const podOriginalData = filterCurrentNodeData.podAffinity;
        const dataList = [];
        for (let i = 0; i < numaSum; i++) {
          const filterData = podOriginalData.filter((item) => item.numaNum === i);
          const sumWithInitial = filterData.length;
          const netName = [];
          filterData.map((item) => {
            netName.push(item.podName);
          });
          dataList.push({
            name: `Numa ${i}`,
            sum: sumWithInitial,
            netName: netName.join('，'),
          });
        }
        setDataSource([...dataList]);
      }
    }
  }, [numaSum]);

  useEffect(() => {
    getDataSource();
  }, [getDataSource]);

  return (
    <div>
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
