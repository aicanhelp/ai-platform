/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useState, useCallback } from 'openinula';
import { useIntl } from 'inula-intl';
import { SyncOutlined, MoreOutlined } from '@ant-design/icons';
import { Form, Table, Space, Input, Button, Popover, Select, message } from 'antd';

import '../../styles/ColocationWorkloadManagementPage.less';
import { filterStatusName, compareObjKeys } from '../../utils/common';
import {
  filterRepeat,
  sortWorkLoadByNameAscend,
  sortWorkLoadByNameDescend,
  sortWorkLoadByTimeDescend,
  sortWorkLoadByTimeAscend,
} from '../../tools/utils';
import ColocationStateModal from '../../components/ColocationStateModal';
import ColocationDeleteModal from '../../components/ColocationDeleteModal';
import Dayjs from 'dayjs';
import { containerRouterPrefix, ResponseCode } from '../../common/constants';
import {
  getColocationDaemonSetsListData,
  getNonColocationDaemonSetsListData,
  addColocationDaemonSets,
  updateDaemonSetsExpectStatusToOnline,
  updateDaemonSetsExpectStatusToOffline,
  deleteColocationDaemonSetsById,
} from '../../api/colocationApi';

export default function DaemonSetPage() {
  const intl = useIntl();
  const [daemonSetForm] = Form.useForm();
  const [popOpen, setPopOpen] = useState('');
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [colShow, setColShow] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  // table部分

  // 非混部table
  const [daemonSetSortObj, setDaemonSetSortObj] = useState({}); // 排序
  const [daemonSetDetailFilterObj, setDaemonSetDetailFilterObj] = useState({}); // 筛选
  const [daemonSetDetailTotal, setDaemonSetDetailTotal] = useState(0);
  const [daemonSetPage, setDaemonSetPage] = useState(1);
  const [daemonSetDetailPageSize, setDaemonSetDetailPageSize] = useState(10);
  const [originalList, setOriginalList] = useState([]); // 非混部原始数据
  const [daemonSetLoading, setDaemonSetLoading] = useState(false);
  const [daemonSetList, setDaemonSetList] = useState([]); // 非混部daemonSet table数据集
  const [selectionType, setSelectionType] = useState(false);
  const [filterDaemonSetNamespace, setFilterDaemonSetNamespace] = useState([]); // namespace赋值筛选项
  const [currentDaemonSetNamespace, setCurrentDaemonSetNamespace] = useState('');
  // 混部table
  const [colocationDaemonSetSortObj, setColocationDaemonSetSortObj] = useState({}); // 排序
  const [colocationDaemonSetDetailFilterObj, setColocationDaemonSetDetailFilterObj] = useState({}); // 筛选
  const [colocationDaemonSetDetailTotal, setColocationDaemonSetDetailTotal] = useState(0);
  const [colocationDaemonSetPage, setColocationDaemonSetPage] = useState(1);
  const [colocationDaemonSetDetailPageSize, setColocationDaemonSetDetailPageSize] = useState(10);
  const [colocationOriginalList, setColocationOriginalList] = useState([]); // 混部原始数据
  const [colocationDaemonSetList, setColocationDaemonSetList] = useState([]); // 混部daemonSet table数据集
  const [colocationFilterDaemonSetNamespace, setColocationFilterDaemonSetNamespace] = useState([]); // namespace赋值筛选项
  const [currentColocationDaemonSetNamespace, setCurrentColocationDaemonSetNamespace] = useState('');

  // 混部状态弹窗
  const [colocationDaemonSetState, setColocationDaemonSetState] = useState(false);
  const [choseState, setChoseState] = useState(intl.formatMessage({ id: 'workload.status.online' }));
  // 删除提醒弹窗
  const [deleteDaemonSetModal, setDeleteDaemonSetModal] = useState(false);
  const [deleteDaemonSetName, setDeleteDaemonSetName] = useState('');
  const [deleteId, setDeleteId] = useState('');

  // 期望混部属性下拉项
  const [daemonSetStatusOption, setDaemonSetStatusOption] = useState([
    {
      value: intl.formatMessage({ id: 'workload.status.online' }),
      label: intl.formatMessage({ id: 'workload.status.online' }),
    },
    {
      value: intl.formatMessage({ id: 'workload.status.offline' }),
      label: intl.formatMessage({ id: 'workload.status.offline' }),
    },
  ]);

  // 混部节点移除
  const handleDeleteDaemonSet = async (record) => {
    setPopOpen('');
    setDeleteDaemonSetModal(true);
    setDeleteDaemonSetName(record.name);
    setDeleteId(record.key);
  };

  // 设置非混部的确定按钮
  const handleDeleteDaemonSetConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationDaemonSetsById(deleteId);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'message.nonColocationSetSuccess' }));
        setTimeout(() => {
          getColocationDaemonSetList();
        }, 1000);
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.nonColocationSetFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setDeleteDaemonSetModal(false);
  };

  // 设置非混部的取消按钮
  const handleDeleteDaemonSetCancel = () => {
    setDeleteDaemonSetModal(false);
  };

  // 混部daemonSet列表项
  const colocationDaemonSetColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'colocationDaemonSetName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: colocationDaemonSetSortObj.key === 'colocationDaemonSetName' ? colocationDaemonSetSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'colocationDaemonSetNamespace',
      filters: colocationFilterDaemonSetNamespace,
      filteredValue: currentColocationDaemonSetNamespace ? [currentColocationDaemonSetNamespace] : null,
      filterMultiple: false,
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'colocation_daemonSet_createTime',
      sorter: true,
      sortOrder:
        colocationDaemonSetSortObj.key === 'colocation_daemonSet_createTime' ? colocationDaemonSetSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.colocationAttribute' }),
      key: 'colocation_daemonSet_status',
      render: (_, record) => (
        <div className={`status_group`}>
          <span className={`${record.colocationType.toLowerCase()}_circle`}></span>
          <span>
            {record.colocationType === 'offline' || record.colocationType === 'online'
              ? filterStatusName(record.colocationType, intl)
              : '--'}
          </span>
        </div>
      ),
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      key: 'colocation_handle',
      render: (_, record) => (
        <Space>
          <Popover
            placement="bottom"
            content={
              <div className="colocation_rules_col_popver">
                {record.colocationType === 'online' ? (
                  <Button type="link" onClick={() => changeColocationDaemonSetState(record)}>
                    {' '}
                    {intl.formatMessage({ id: 'button.changeToOffline' })}
                  </Button>
                ) : (
                  <></>
                )}
                <Button
                  type="link"
                  onClick={() => {
                    handleDeleteDaemonSet(record);
                  }}
                >
                  {' '}
                  {intl.formatMessage({ id: 'button.setNonColocation' })}
                </Button>
              </div>
            }
            trigger="click"
            open={popOpen === `${record.name}_${record.namespace}`}
            onOpenChange={(newOpen) => (newOpen ? setPopOpen(`${record.name}_${record.namespace}`) : setPopOpen(''))}
          >
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover>
        </Space>
      ),
    },
  ];

  // 非混部daemonSet列表项
  const daemonSetColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'daemonSetName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: daemonSetSortObj.key === 'daemonSetName' ? daemonSetSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'daemonSetNamespace',
      filters: filterDaemonSetNamespace,
      filteredValue: currentDaemonSetNamespace ? [currentDaemonSetNamespace] : null,
      filterMultiple: false,
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'daemonSet_createTime',
      sorter: true,
      sortOrder: daemonSetSortObj.key === 'daemonSet_createTime' ? daemonSetSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.expectedColocationAttr' }),
      key: 'daemonSet_status',
      width: 160,
      render: (_, record) => (
        <div>
          <Select
            className="table_hope_status"
            defaultValue={filterStatusName(record.colocationType, intl)}
            onChange={() => handleHopeChange(record)}
            options={daemonSetStatusOption}
          />
        </div>
      ),
    },
  ];

  // 获取非混部DaemonSetList
  const getDaemonSetList = useCallback(async () => {
    setDaemonSetLoading(true);
    try {
      const res = await getNonColocationDaemonSetsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          item.colocationType = 'online';
          return {
            ...item,
            key: item.uid,
          };
        });
        setDaemonSetList(arr);
        setOriginalList(arr);
        setDaemonSetDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setFilterDaemonSetNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setDaemonSetList([]); // 数组为空
    }
    setDaemonSetLoading(false);
  }, []);

  // 获取混部DaemonSetList
  const getColocationDaemonSetList = useCallback(async () => {
    setDaemonSetLoading(true);
    try {
      const res = await getColocationDaemonSetsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        setColocationDaemonSetList(arr);
        setColocationOriginalList(arr);
        setColocationDaemonSetDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setColocationFilterDaemonSetNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setColocationDaemonSetList([]); // 数组为空
    }
    setDaemonSetLoading(false);
  }, []);

  // 混部具体表格操作
  const getColocationDaemonSetResource = () => {
    let filtertArr = [];
    if (daemonSetForm.getFieldsValue().colocationDaemonSetName) {
      filtertArr = colocationOriginalList.filter((item) =>
        item.name.toLowerCase().includes(daemonSetForm.getFieldsValue().colocationDaemonSetName),
      );
    } else {
      filtertArr = JSON.parse(JSON.stringify(colocationOriginalList || []));
    }
    if (Object.keys(colocationDaemonSetDetailFilterObj).length !== 0) {
      if (!colocationDaemonSetDetailFilterObj.colocationDaemonSetNamespace) {
        setCurrentColocationDaemonSetNamespace('');
      } else {
        // 获取筛选框
        let [temporyStatus, ...reset] = colocationDaemonSetDetailFilterObj.colocationDaemonSetNamespace;
        setCurrentColocationDaemonSetNamespace(temporyStatus);
        filtertArr = filtertArr.filter((item) => item.namespace === temporyStatus);
      }
    }
    if (Object.keys(colocationDaemonSetSortObj).length !== 0) {
      if (colocationDaemonSetSortObj.key === 'colocationDaemonSetName') {
        if (colocationDaemonSetSortObj.order === 'ascend') {
          sortWorkLoadByNameAscend('daemonSet', filtertArr);
        } else {
          if (colocationDaemonSetSortObj.order === 'descend') {
            sortWorkLoadByNameDescend('daemonSet', filtertArr);
          }
        }
      } else {
        if (colocationDaemonSetSortObj.order === 'ascend') {
          sortWorkLoadByTimeAscend('daemonSet', filtertArr);
        } else {
          if (colocationDaemonSetSortObj.order === 'descend') {
            sortWorkLoadByTimeDescend('daemonSet', filtertArr);
          }
        }
      }
    }
    setColocationDaemonSetDetailTotal(filtertArr.length);
    setColocationDaemonSetList([...filtertArr]);
  };

  // 非混部具体表格操作
  const getDaemonSetResource = () => {
    let filtertArr = [];
    if (daemonSetForm.getFieldsValue().daemonSetName) {
      daemonSetForm.setFieldsValue({ daemonSetKeyValue: '' });
      filtertArr = originalList.filter((item) =>
        item.name.toLowerCase().includes(daemonSetForm.getFieldsValue().daemonSetName),
      );
    } else if (daemonSetForm.getFieldsValue().daemonSetKeyValue) {
      daemonSetForm.setFieldsValue({ daemonSetName: '' });
      filtertArr = originalList.filter((item) => {
        return compareObjKeys(item, daemonSetForm.getFieldsValue().daemonSetKeyValue);
      });
    } else {
      filtertArr = JSON.parse(JSON.stringify(originalList || []));
    }
    if (Object.keys(daemonSetDetailFilterObj).length !== 0) {
      if (!daemonSetDetailFilterObj.daemonSetNamespace) {
        setCurrentDaemonSetNamespace('');
      } else {
        // 获取筛选框
        let [temporyStatus, ...reset] = daemonSetDetailFilterObj.daemonSetNamespace;
        setCurrentDaemonSetNamespace(temporyStatus);
        filtertArr = filtertArr.filter((item) => item.namespace === temporyStatus);
      }
    }
    if (Object.keys(daemonSetSortObj).length !== 0) {
      if (daemonSetSortObj.key === 'daemonSetName') {
        if (daemonSetSortObj.order === 'ascend') {
          sortWorkLoadByNameAscend('daemonSet', filtertArr);
        } else {
          if (daemonSetSortObj.order === 'descend') {
            sortWorkLoadByNameDescend('daemonSet', filtertArr);
          }
        }
      } else {
        if (daemonSetSortObj.order === 'ascend') {
          sortWorkLoadByTimeAscend('daemonSet', filtertArr);
        } else {
          if (daemonSetSortObj.order === 'descend') {
            sortWorkLoadByTimeDescend('daemonSet', filtertArr);
          }
        }
      }
    }
    setDaemonSetDetailTotal(filtertArr.length);
    setDaemonSetList([...filtertArr]);
  };

  // 混部表格操作
  const handleColocationDaemonSetDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setColocationDaemonSetPage(pagination.current || 1);
      setColocationDaemonSetDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setColocationDaemonSetDetailFilterObj(filter);
    }
    if (extra.action === 'sort') {
      setColocationDaemonSetSortObj({ key: _sorter.columnKey, order: _sorter.order });
    }
  };

  // 非混部表格操作
  const handleDaemonSetDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setDaemonSetPage(pagination.current || 1);
      setDaemonSetDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setDaemonSetDetailFilterObj(filter);
    }
    if (extra.action === 'sort') {
      setDaemonSetSortObj({ key: _sorter.columnKey, order: _sorter.order });
    }
  };

  // 改变期望选择
  const handleHopeChange = async (data) => {
    if (data.colocationType === 'online') {
      data.colocationType = 'offline';
    } else {
      data.colocationType = 'online';
    }
  };

  // 点击出现改变混部属性的弹窗
  const changeColocationDaemonSetState = (record) => {
    setPopOpen('');
    setChoseState(filterStatusName(record.colocationType, intl));
    setColocationDaemonSetState(true);
    setDeleteId(record.key);
  };

  const openDaemonSetButton = () => {
    setSelectionType(true);
    setColShow(true);
    getDaemonSetList();
  };

  // 混部选择
  const onDaemonSetSelectChange = (newSelectedRowKeys, newSelectedRows) => {
    setSelectedRowKeys(newSelectedRowKeys);
    setSelectedRows(newSelectedRows);
  };

  // 混部model的确定按钮
  const handleDaemonSetStateModelConfirm = async (type) => {
    if (type === intl.formatMessage({ id: 'workload.status.offline' })) {
      // 离线代接口
      try {
        const res = await updateDaemonSetsExpectStatusToOffline(deleteId);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
          getColocationDaemonSetList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.modifyFailed' }));
      }
    }
    setColocationDaemonSetState(false);
  };

  // 混部model的取消按钮
  const handleDaemonSetStateModelCancel = async () => {
    setColocationDaemonSetState(false);
  };

  // 重置按钮
  const handleDaemonSetReset = async () => {
    await getDaemonSetList();
    getDaemonSetResource();
  };

  // 重置按钮
  const handleColocationDaemonSetReset = async () => {
    await getColocationDaemonSetList();
    getColocationDaemonSetResource();
  };

  // table的取消 重置 确定
  const onDaemonSetCancel = () => {
    setColShow(false);
    setSelectionType(false);
    daemonSetForm.setFieldsValue({ daemonSetName: '' });
    daemonSetForm.setFieldsValue({ daemonSetKeyValue: '' });
    getDaemonSetResource();
    setCurrentDaemonSetNamespace(''); // 重置筛选条件
    setDaemonSetSortObj({});
    setDaemonSetDetailFilterObj({});
    setDaemonSetPage(1);
  };

  const onDaemonSetRest = () => {
    setSelectedRowKeys([]);
    setSelectedRows([]);
  };

  const onDaemonSetSure = async () => {
    if (selectedRows.length === 0) {
      messageApi.error(intl.formatMessage({ id: 'message.selectAtLeastOneWorkload' }));
    } else if (selectedRows.length > 10) {
      messageApi.error(intl.formatMessage({ id: 'message.maxTenWorkloads' }));
    } else {
      // 代接口 selectedRowKeys, selectedRows
      try {
        const res = await addColocationDaemonSets(selectedRows);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.addColocationSuccess' }, { type: 'DaemonSet' }));
          setColShow(false);
          daemonSetForm.resetFields();
          setSelectedRowKeys([]);
          setSelectedRows([]);

          getColocationDaemonSetResource();
          setCurrentColocationDaemonSetNamespace(''); // 重置筛选条件
          setColocationDaemonSetSortObj({});
          setColocationDaemonSetDetailFilterObj({});
          setColocationDaemonSetPage(1);

          getDaemonSetResource();
          setCurrentDaemonSetNamespace(''); // 重置筛选条件
          setDaemonSetSortObj({});
          setDaemonSetDetailFilterObj({});
          setDaemonSetPage(1);

          getColocationDaemonSetList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.addColocationFailed' }, { type: 'DaemonSet' }));
      }
    }
  };

  useEffect(() => {
    getColocationDaemonSetResource();
  }, [colocationDaemonSetDetailFilterObj, colocationDaemonSetSortObj]);

  useEffect(() => {
    getDaemonSetResource();
  }, [daemonSetDetailFilterObj, daemonSetSortObj]);

  useEffect(() => {
    getColocationDaemonSetList();
  }, [getColocationDaemonSetList]);

  return (
    <div className="workload_content">
      <div
        className={
          colShow && daemonSetDetailTotal >= 7
            ? 'workload_pageBottom  workload_detail_content'
            : 'workload_detail_content'
        }
      >
        <div className="workload_detail_content_box">
          <div className="workload_detail_form">
            <Form className="workload-searchForm form_padding_bottom" form={daemonSetForm}>
              {colShow ? (
                <div className="workload-searchForm-box">
                  <Form.Item name="daemonSetName" className="workload-search-input" style={{ marginRight: '20px' }}>
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                      onSearch={() => getDaemonSetResource()}
                      autoComplete="off"
                      maxLength={53}
                    />
                  </Form.Item>
                  <Form.Item name="daemonSetKeyValue" className="workload-search-input">
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.byLabel' })}
                      onSearch={() => getDaemonSetResource()}
                      autoComplete="off"
                    />
                  </Form.Item>
                </div>
              ) : (
                <Form.Item
                  name="colocationDaemonSetName"
                  className="workload-search-input"
                  style={{ marginRight: '20px' }}
                >
                  <Input.Search
                    placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                    onSearch={() => getColocationDaemonSetResource()}
                    autoComplete="off"
                    maxLength={53}
                  />
                </Form.Item>
              )}
              <Form.Item>
                {!colShow ? (
                  <Space>
                    <Button
                      className="workload_detail_form_button"
                      style={{ width: '180px' }}
                      onClick={() => openDaemonSetButton()}
                    >
                      {intl.formatMessage({ id: 'button.addColocation' }, { type: 'DaemonSet' })}
                    </Button>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleColocationDaemonSetReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                ) : (
                  <Space>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleDaemonSetReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                )}
              </Form.Item>
            </Form>
          </div>
          <div className="workload_detail_content_table">
            {contextHolder}
            {colShow ? (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={daemonSetLoading}
                columns={daemonSetColumns}
                dataSource={daemonSetList}
                rowSelection={
                  selectionType
                    ? {
                        type: 'checkbox',
                        selectedRowKeys,
                        preserveSelectedRowKeys: true,
                        onChange: onDaemonSetSelectChange,
                      }
                    : null
                }
                pagination={{
                  className: 'colocation-website-page',
                  current: daemonSetPage,
                  pageSize: daemonSetDetailPageSize,
                  total: daemonSetDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleDaemonSetDetailTableChange}
              ></Table>
            ) : (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={daemonSetLoading}
                columns={colocationDaemonSetColumns}
                dataSource={colocationDaemonSetList}
                pagination={{
                  className: 'colocation-website-page',
                  current: colocationDaemonSetPage,
                  pageSize: colocationDaemonSetDetailPageSize,
                  total: colocationDaemonSetDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleColocationDaemonSetDetailTableChange}
              ></Table>
            )}
          </div>
        </div>
      </div>
      <ColocationStateModal
        title={intl.formatMessage({ id: 'message.changeState' }, { state: choseState })}
        open={colocationDaemonSetState}
        cancelFn={handleDaemonSetStateModelCancel}
        confirmFn={() => handleDaemonSetStateModelConfirm(choseState)}
        state={choseState}
      />
      <ColocationDeleteModal
        title={intl.formatMessage({ id: 'message.setNonColocation' })}
        open={deleteDaemonSetModal}
        cancelFn={handleDeleteDaemonSetCancel}
        content={[intl.formatMessage({ id: 'message.confirmDelete' }, { name: deleteDaemonSetName })]}
        confirmFn={handleDeleteDaemonSetConfirmBack}
      />
      {colShow ? (
        <div className="child_content_workload_footer">
          <div className="workload_btn_footer_text">
            <p>{intl.formatMessage({ id: 'message.selectedWorkloads' }, { count: selectedRows.length })}</p>
          </div>
          <div className="workload_btn_footer_button">
            <Button className="cancel_btn" onClick={onDaemonSetCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={onDaemonSetRest}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={onDaemonSetSure}>
              {intl.formatMessage({ id: 'common.confirm' })}
            </Button>
          </div>
        </div>
      ) : (
        <></>
      )}
    </div>
  );
}
