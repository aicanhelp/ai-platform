/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import '@/styles/PriorityClass.less';
import { MoreOutlined } from '@ant-design/icons';
import { Input, Table, Space, Popover, Button, message } from 'antd';

import { Link, useHistory } from 'inula-router';
import { useEffect, useState } from 'openinula';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { getPriorityClassesList, deleteSpecifiedPriorityClasses } from '@/api/schedulingApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { sortNaN, sorterFirstAlphabet } from '@/tools/utils';
import { useIntl } from 'inula-intl';

export default function PriorityClass() {
  const intl = useIntl();
  const [messageApi, contextHolder] = message.useMessage();
  const { Search } = Input;
  const [popOpen, setPopOpen] = useState('');
  const history = useHistory();
  const [dataSource, setDatatSource] = useState([]);
  const [priorityModalOpen, setPriorityModalOpen] = useState(false); // 删除对话框展示
  const [isPriorityDelCheck, setIsPriorityDelCheck] = useState(false); // 是否选中
  const [priorityName, setPriorityDelName] = useState(''); // 删除的名称
  const [current, setCurrent] = useState(1);
  const columns = [
    {
      title: intl.formatMessage({ id: 'priority.priorityClassName' }),
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => (
        <div>
          <Link
            to={`/SchedulingPolicyConfiguration/PriorityClassDetails/?priorityName=${record.metadata.name}&isEdit=${false}`}
          >
            {record.metadata.name}
          </Link>
        </div>
      ),
    },
    {
      title: intl.formatMessage({ id: 'priority.priorityValue' }),
      dataIndex: 'priority',
      key: 'priority',
      sorter: (a, b) => a.value - b.value,
      render: (_, record) => record.value,
    },
    {
      title: intl.formatMessage({ id: 'priority.isDefault' }),
      dataIndex: 'globalDefault',
      key: 'globalDefault',
      filters: [
        { text: intl.formatMessage({ id: 'common.yes' }), value: intl.formatMessage({ id: 'common.yes' }) },
        { text: intl.formatMessage({ id: 'common.no' }), value: intl.formatMessage({ id: 'common.no' }) },
      ],
      onFilter: (value, record) => {
        const isDefault = record.globalDefault
          ? intl.formatMessage({ id: 'common.yes' })
          : intl.formatMessage({ id: 'common.no' });
        return isDefault === value;
      },
      render: (_, record) =>
        record.globalDefault ? intl.formatMessage({ id: 'common.yes' }) : intl.formatMessage({ id: 'common.no' }),
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      key: 'handle',
      fixed: 'right',
      width: 100,
      render: (_, record) => (
        <Space>
          <Popover
            getPopupContainer={() => document.getElementById('priorityList')}
            placement="bottom"
            className="warehouseConfiguration"
            content={
              <div className="pop_modal">
                <Button type="link">
                  <Link
                    to={`/SchedulingPolicyConfiguration/PriorityClassUpdate/?priorityName=${record.metadata.name}&isEdit=true`}
                  >
                    {intl.formatMessage({ id: 'priority.modify' })}
                  </Link>
                </Button>
                <Button type="text" onClick={() => handleDeletePriorityIndex(record.metadata.name)}>
                  {intl.formatMessage({ id: 'common.delete' })}
                </Button>
              </div>
            }
            trigger="click"
            btn_
            open={popOpen === `${record.metadata.name}`}
            onOpenChange={(newOpen) => (newOpen ? setPopOpen(record.metadata.name) : setPopOpen(''))}
          >
            <MoreOutlined className="common_antd_icon primary_color warehouseConfiguration" />
          </Popover>
        </Space>
      ),
    },
  ];

  const getPriorityClassesListInfo = async (name) => {
    const res = await getPriorityClassesList();
    if (name) {
      const filteredData = res.data.items.filter((item) =>
        item.metadata.name.toLowerCase().includes(name.toLowerCase()),
      );
      setDatatSource(filteredData);
    } else {
      setDatatSource(res.data.items);
    }
  };

  const handlePriorityCheckFn = (e) => {
    setIsPriorityDelCheck(e.target.checked);
  };

  // 取消删除
  const handleDelCancelPriorityIndex = () => {
    setPriorityModalOpen(false);
    setPriorityDelName('');
  };

  const handleDelPriorityConfirm = async () => {
    try {
      const resPriorityClass = await deleteSpecifiedPriorityClasses(priorityName);
      if (resPriorityClass.status === ResponseCode.OK) {
        message.success(intl.formatMessage({ id: 'priority.deleteSuccess' }));
        setPriorityModalOpen(false);
        setIsPriorityDelCheck(false);
        setPriorityDelName('');
        getPriorityClassesListInfo();
      }
    } catch (error) {
      if (error.response.status === 403) {
        message.error(intl.formatMessage({ id: 'common.permissionDenied' }));
      } else {
        message.error(error.response?.data.message);
      }
    }
  };

  const handleSearch = (e) => {
    getPriorityClassesListInfo(e);
  };

  // 删除按钮
  const handleDeletePriorityIndex = (name) => {
    // 隐藏气泡框
    setPopOpen('');
    setPriorityModalOpen(true); // 打开弹窗
    setPriorityDelName(name);
    setIsPriorityDelCheck(false);
  };

  const handlePageChange = (page, pageSize) => {
    setCurrent(page); // 更新当前页码
  };

  useEffect(() => {
    getPriorityClassesListInfo();
  }, []);
  return (
    <div className="priority_class_container" id="priorityList">
      <div style={{ marginBottom: '20px' }}>
        <BreadCrumbCom
          items={[
            { title: intl.formatMessage({ id: 'scheduling.schedulingPolicyConfig' }), disabled: true },
            { title: 'PriorityClass', path: '/SchedulingPolicyConfiguration/PriorityClass' },
          ]}
        />
      </div>
      <div
        className="priority-out-box"
        style={{
          display: 'flex',
          flexDirection: 'column',
          padding: '32px 32px 16px 32px',
          height: 'calc(100vh - 64px - 54px - 20px - 32px - 48px - 2px)',
          margin: '0px 32px 20px 20px',
        }}
      >
        <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between' }}>
          <Search
            placeholder={intl.formatMessage({ id: 'priority.searchPriorityClassName' })}
            style={{ width: '300px' }}
            onSearch={handleSearch}
          />
          <Button className="primary_btn" onClick={() => history.push('/SchedulingPolicyConfiguration/createPriority')}>
            {intl.formatMessage({ id: 'priority.create' })}
          </Button>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', flex: '1' }}>
          <Table
            locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
            className="tab_table_flex"
            dataSource={dataSource}
            pagination={{
              className: 'volcano-page',
              current,
              pageSizeOptions: [10, 20, 50],
              showSizeChanger: true,
              showQuickJumper: true,
              onChange: handlePageChange,
              hideOnSinglePage: false, // 添加这一行来强制显示分页
              showTotal: () => intl.formatMessage({ id: 'common.totalItems' }, { total: dataSource.length }),
            }}
            columns={columns}
          />
        </div>
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'priority.deletePriorityClass' })}
        className="PriorityClass"
        open={priorityModalOpen}
        cancelFn={handleDelCancelPriorityIndex}
        content={[
          intl.formatMessage({ id: 'priority.deleteWarning' }),
          intl.formatMessage({ id: 'priority.confirmDelete' }, { name: priorityName }),
        ]}
        showCheck={true}
        checkFn={handlePriorityCheckFn}
        isCheck={isPriorityDelCheck}
        confirmFn={handleDelPriorityConfirm}
      />
    </div>
  );
}
