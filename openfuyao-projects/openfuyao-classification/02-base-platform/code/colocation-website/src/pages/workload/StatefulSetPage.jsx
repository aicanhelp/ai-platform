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
import { SyncOutlined, MoreOutlined, CloseOutlined } from '@ant-design/icons';
import { Form, Table, Space, Input, Button, Popover, Select, message } from 'antd';

import '../../styles/ColocationWorkloadManagementPage.less';
import { filterStatusName, reseverStatusName, compareObjKeys } from '../../utils/common';
import {
  filterRepeat,
  sortWorkLoadByNameAscend,
  sortWorkLoadByNameDescend,
  sortWorkLoadByTimeDescend,
  sortWorkLoadByTimeAscend,
  sortWorkLoadByNamespaceAscend,
  sortWorkLoadByNamespaceDescend,
  sortWorkLoadByStatusAscend,
  sortWorkLoadByStatusDescend,
} from '../../tools/utils';
import ColocationStateModal from '../../components/ColocationStateModal';
import ColocationDeleteModal from '../../components/ColocationDeleteModal';
import Dayjs from 'dayjs';
import { getWorkloadStatusOptions, ResponseCode } from '../../common/constants';
import {
  getColocationStatefulSetsListData,
  getNonColocationStatefulSetsListData,
  addColocationStatefulSets,
  updateStatefulSetsExpectStatusToOnline,
  updateStatefulSetsExpectStatusToOffline,
  deleteColocationStatefulSetsById,
} from '../../api/colocationApi';
import ToolTipComponent from '@/components/ToolTipComponent';

export default function StatefulSetPage() {
  const intl = useIntl();
  const [statefulSetForm] = Form.useForm();
  const [popOpen, setPopOpen] = useState('');
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [colShow, setColShow] = useState(false);
  const [isShow, setIsShow] = useState('');
  const [messageApi, contextHolder] = message.useMessage();
  // table部分

  // 非混部table
  const [statefulSetSortObj, setStatefulSetSortObj] = useState({}); // 排序
  const [statefulSetDetailFilterObj, setStatefulSetDetailFilterObj] = useState({}); // 筛选
  const [statefulSetDetailTotal, setStatefulSetDetailTotal] = useState(0);
  const [statefulSetPage, setStatefulSetPage] = useState(1);
  const [statefulSetDetailPageSize, setStatefulSetDetailPageSize] = useState(10);
  const [originalList, setOriginalList] = useState([]); // 非混部原始数据
  const [statefulSetLoading, setStatefulSetLoading] = useState(false);
  const [statefulSetList, setStatefulSetList] = useState([]); // 非混部statefulSet table数据集
  const [selectionType, setSelectionType] = useState(false);
  const [filterStatefulSetNamespace, setFilterStatefulSetNamespace] = useState([]); // namespace赋值筛选项
  const [currentStatefulSetNamespace, setCurrentStatefulSetNamespace] = useState('');
  const [attTypeList, setAttTypeList] = useState([]); // 超级原始table数据集
  // 混部table
  const [colocationStatefulSetSortObj, setColocationStatefulSetSortObj] = useState({}); // 排序
  const [colocationStatefulSetDetailFilterObj, setColocationStatefulSetDetailFilterObj] = useState({}); // 筛选
  const [colocationStatefulSetDetailTotal, setColocationStatefulSetDetailTotal] = useState(0);
  const [colocationStatefulSetPage, setColocationStatefulSetPage] = useState(1);
  const [colocationStatefulSetDetailPageSize, setColocationStatefulSetDetailPageSize] = useState(10);
  const [colocationOriginalList, setColocationOriginalList] = useState([]); // 混部原始数据
  const [colocationStatefulSetList, setColocationStatefulSetList] = useState([]); // 混部statefulSet table数据集
  const [colocationFilterStatefulSetNamespace, setColocationFilterStatefulSetNamespace] = useState([]); // namespace赋值筛选项
  const [currentColocationStatefulSetNamespace, setCurrentColocationStatefulSetNamespace] = useState('');
  const [colocationFilterStatefulSetStatus, setColocationFilterStatefulSetStatus] = useState([
    { text: intl.formatMessage({ id: 'workload.status.online' }), value: '在线' },
    { text: intl.formatMessage({ id: 'workload.status.offline' }), value: '离线' },
    { text: intl.formatMessage({ id: 'workload.status.oversold' }), value: '超卖' },
  ]);
  const [currentColocationStatefulSetStatus, setCurrentColocationStatefulSetStatus] = useState('');
  const [isStatefulSetWithoutPaddingBottom, setIsStatefulSetWithoutPaddingBottom] = useState('no-padding');
  // 混部状态弹窗
  const [colocationStatefulSetState, setColocationStatefulSetState] = useState(false);
  const [choseState, setChoseState] = useState(intl.formatMessage({ id: 'workload.status.online' }));
  // 删除提醒弹窗
  const [deleteStatefulSetModal, setDeleteStatefulSetModal] = useState(false);
  const [deleteStatefulSetName, setDeleteStatefulSetName] = useState('');
  const [deleteId, setDeleteId] = useState('');

  const handleStatefulSetShowChange = () => {
    setIsShow('cant_show');
  };

  // 混部节点移除
  const handleDeleteStatefulSet = async (record) => {
    setPopOpen('');
    setDeleteStatefulSetModal(true);
    setDeleteStatefulSetName(record.name);
    setDeleteId(record.key);
  };

  // 设置非混部的确定按钮
  const handleDeleteStatefulSetConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationStatefulSetsById(deleteId);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'message.nonColocationSetSuccess' }));
        setTimeout(() => {
          getColocationStatefulSetList();
        }, 1000);
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.nonColocationSetFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setDeleteStatefulSetModal(false);
  };

  // 设置非混部的取消按钮
  const handleDeleteStatefulSetCancel = () => {
    setDeleteStatefulSetModal(false);
  };

  // 混部statefulSet列表项
  const colocationStatefulSetColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'colocationStatefulSetName',
      dataIndex: 'name',
      sorter: true,
      sortOrder:
        colocationStatefulSetSortObj.key === 'colocationStatefulSetName' ? colocationStatefulSetSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'colocationStatefulSetNamespace',
      filters: colocationFilterStatefulSetNamespace,
      filteredValue: currentColocationStatefulSetNamespace ? [currentColocationStatefulSetNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder:
        colocationStatefulSetSortObj.key === 'colocationStatefulSetNamespace' ? colocationStatefulSetSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'colocation_statefulSet_createTime',
      sorter: true,
      sortOrder:
        colocationStatefulSetSortObj.key === 'colocation_statefulSet_createTime'
          ? colocationStatefulSetSortObj.order
          : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.businessAttr' }),
      key: 'colocation_statefulSet_status',
      filters: colocationFilterStatefulSetStatus,
      filteredValue: currentColocationStatefulSetStatus ? [currentColocationStatefulSetStatus] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder:
        colocationStatefulSetSortObj.key === 'colocation_statefulSet_status' ? colocationStatefulSetSortObj.order : '',
      render: (_, record) => (
        <div className={`status_group`}>
          <span className={`${record.colocationType.toLowerCase()}_circle`}></span>
          <span>
            {record.colocationType === 'offline' ||
            record.colocationType === 'online' ||
            record.colocationType === 'oversold'
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
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <Button type="link" onClick={() => changeColocationStatefulSetState(record, 'off')}>
                      {' '}
                      {intl.formatMessage({ id: 'action.changeToOffline' })}
                    </Button>
                  </div>
                ) : (
                  <></>
                )}
                {record.colocationType === 'offline' ? (
                  <div style={{ display: 'flex', flexDirection: 'column' }}></div>
                ) : (
                  <></>
                )}
                <Button
                  type="link"
                  onClick={() => {
                    handleDeleteStatefulSet(record);
                  }}
                >
                  {' '}
                  {intl.formatMessage({ id: 'action.setNonColocation' })}
                </Button>
              </div>
            }
            trigger="click"
            open={popOpen === `${record.name}_${record.namespace}`}
            onOpenChange={(newOpen) => (newOpen ? setPopOpen(`${record.name}_${record.namespace}`) : setPopOpen(''))}
          >
            {record.colocationType === 'oversold' ? (
              <Button className="workload-table-opt" type="text" disabled>
                <MoreOutlined className="common_antd_icon" />
              </Button>
            ) : (
              <Button className="workload-table-opt" type="text">
                <MoreOutlined className="common_antd_icon primary_color" />
              </Button>
            )}
          </Popover>
        </Space>
      ),
    },
  ];

  // 非混部statefulSet列表项
  const statefulSetColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'statefulSetName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: statefulSetSortObj.key === 'statefulSetName' ? statefulSetSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'statefulSetNamespace',
      filters: filterStatefulSetNamespace,
      filteredValue: currentStatefulSetNamespace ? [currentStatefulSetNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: statefulSetSortObj.key === 'statefulSetNamespace' ? statefulSetSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'statefulSet_createTime',
      sorter: true,
      sortOrder: statefulSetSortObj.key === 'statefulSet_createTime' ? statefulSetSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.expectedBusinessAttr' }),
      key: 'statefulSet_status',
      width: 160,
      render: (_, record) => (
        <div>
          <Select
            className="table_hope_status"
            value={filterStatusName(record.colocationType, intl)}
            onChange={(e) => handleHopeChange(e, record)}
            options={getWorkloadStatusOptions(intl)}
          />
        </div>
      ),
    },
  ];

  // 获取非混部StatefulSetList
  const getStatefulSetList = useCallback(async () => {
    setStatefulSetLoading(true);
    try {
      const res = await getNonColocationStatefulSetsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          item.colocationType = 'online';
          return {
            ...item,
            key: item.uid,
          };
        });
        setStatefulSetList(arr);
        setOriginalList(JSON.parse(JSON.stringify(arr)));
        setAttTypeList(JSON.parse(JSON.stringify(JSON.parse(JSON.stringify(arr)))));
        setStatefulSetDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setFilterStatefulSetNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setStatefulSetList([]); // 数组为空
    }
    setStatefulSetLoading(false);
  }, []);

  // 获取混部StatefulSetList
  const getColocationStatefulSetList = useCallback(async () => {
    setStatefulSetLoading(true);
    try {
      const res = await getColocationStatefulSetsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        setColocationStatefulSetList(arr);
        setColocationOriginalList(arr);
        setColocationStatefulSetDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setColocationFilterStatefulSetNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setColocationStatefulSetList([]); // 数组为空
    }
    setStatefulSetLoading(false);
  }, []);

  // 混部具体表格操作
  const getColocationStatefulSetResource = () => {
    let filtertArr = [];
    if (statefulSetForm.getFieldsValue().colocationStatefulSetName) {
      filtertArr = colocationOriginalList.filter((item) =>
        item.name.toLowerCase().includes(statefulSetForm.getFieldsValue().colocationStatefulSetName.toLowerCase()),
      );
    } else {
      filtertArr = JSON.parse(JSON.stringify(colocationOriginalList || []));
    }
    if (Object.keys(colocationStatefulSetDetailFilterObj).length !== 0) {
      if (
        colocationStatefulSetDetailFilterObj.colocationStatefulSetNamespace &&
        colocationStatefulSetDetailFilterObj.colocation_statefulSet_status
      ) {
        // 获取筛选框
        let [temporyNamespace, ...reset] = colocationStatefulSetDetailFilterObj.colocationStatefulSetNamespace;
        let [temporyStatus, ...reset1] = colocationStatefulSetDetailFilterObj.colocation_statefulSet_status;
        setCurrentColocationStatefulSetNamespace(temporyNamespace);
        setCurrentColocationStatefulSetStatus(temporyStatus);
        filtertArr = filtertArr.filter(
          (item) => item.namespace === temporyNamespace && filterStatusName(item.colocationType) === temporyStatus,
        );
      } else {
        if (colocationStatefulSetDetailFilterObj.colocationStatefulSetNamespace) {
          setCurrentColocationStatefulSetStatus('');
          let [temporyNamespace, ...reset] = colocationStatefulSetDetailFilterObj.colocationStatefulSetNamespace;
          setCurrentColocationStatefulSetNamespace(temporyNamespace);
          filtertArr = filtertArr.filter((item) => item.namespace === temporyNamespace);
        } else if (colocationStatefulSetDetailFilterObj.colocation_statefulSet_status) {
          setCurrentColocationStatefulSetNamespace('');
          let [temporyStatus, ...reset1] = colocationStatefulSetDetailFilterObj.colocation_statefulSet_status;
          setCurrentColocationStatefulSetStatus(temporyStatus);
          filtertArr = filtertArr.filter((item) => filterStatusName(item.colocationType) === temporyStatus);
        } else {
          setCurrentColocationStatefulSetNamespace('');
          setCurrentColocationStatefulSetStatus('');
        }
      }
    }
    if (Object.keys(colocationStatefulSetSortObj).length !== 0) {
      if (colocationStatefulSetSortObj.key === 'colocationStatefulSetName') {
        if (colocationStatefulSetSortObj.order) {
          colocationStatefulSetSortObj.order === 'ascend'
            ? sortWorkLoadByNameAscend('statefulSet', filtertArr)
            : sortWorkLoadByNameDescend('statefulSet', filtertArr);
        }
      } else if (colocationStatefulSetSortObj.key === 'colocationStatefulSetNamespace') {
        if (colocationStatefulSetSortObj.order) {
          colocationStatefulSetSortObj.order === 'ascend'
            ? sortWorkLoadByNamespaceAscend('statefulSet', filtertArr)
            : sortWorkLoadByNamespaceDescend('statefulSet', filtertArr);
        }
      } else if (colocationStatefulSetSortObj.key === 'colocation_statefulSet_createTime') {
        if (colocationStatefulSetSortObj.order) {
          colocationStatefulSetSortObj.order === 'ascend'
            ? sortWorkLoadByTimeAscend('statefulSet', filtertArr)
            : sortWorkLoadByTimeDescend('statefulSet', filtertArr);
        }
      } else {
        if (colocationStatefulSetSortObj.order) {
          colocationStatefulSetSortObj.order === 'ascend'
            ? sortWorkLoadByStatusAscend('statefulSet', filtertArr)
            : sortWorkLoadByStatusDescend('statefulSet', filtertArr);
        }
      }
    }
    setColocationStatefulSetDetailTotal(filtertArr.length);
    setColocationStatefulSetList([...filtertArr]);
  };

  // 非混部具体表格操作
  const getStatefulSetResource = () => {
    let filtertArr = [];
    if (statefulSetForm.getFieldsValue().statefulSetName) {
      filtertArr = originalList.filter((item) =>
        item.name.toLowerCase().includes(statefulSetForm.getFieldsValue().statefulSetName.toLowerCase()),
      );
    } else if (statefulSetForm.getFieldsValue().statefulSetKeyValue) {
      filtertArr = originalList.filter((item) => {
        return compareObjKeys(item, statefulSetForm.getFieldsValue().statefulSetKeyValue);
      });
    } else {
      filtertArr = JSON.parse(JSON.stringify(originalList || []));
    }
    if (Object.keys(statefulSetDetailFilterObj).length !== 0) {
      if (!statefulSetDetailFilterObj.statefulSetNamespace) {
        setCurrentStatefulSetNamespace('');
      } else {
        // 获取筛选框
        let [temporyStatus, ...reset] = statefulSetDetailFilterObj.statefulSetNamespace;
        setCurrentStatefulSetNamespace(temporyStatus);
        filtertArr = filtertArr.filter((item) => item.namespace === temporyStatus);
      }
    }
    if (Object.keys(statefulSetSortObj).length !== 0) {
      if (statefulSetSortObj.key === 'statefulSetName') {
        if (statefulSetSortObj.order) {
          statefulSetSortObj.order === 'ascend'
            ? sortWorkLoadByNameAscend('statefulSet', filtertArr)
            : sortWorkLoadByNameDescend('statefulSet', filtertArr);
        }
      } else if (statefulSetSortObj.key === 'statefulSetNamespace') {
        if (statefulSetSortObj.order) {
          statefulSetSortObj.order === 'ascend'
            ? sortWorkLoadByNamespaceAscend('statefulSet', filtertArr)
            : sortWorkLoadByNamespaceDescend('statefulSet', filtertArr);
        }
      } else {
        if (statefulSetSortObj.order) {
          statefulSetSortObj.order === 'ascend'
            ? sortWorkLoadByTimeAscend('statefulSet', filtertArr)
            : sortWorkLoadByTimeDescend('statefulSet', filtertArr);
        }
      }
    }
    setStatefulSetDetailTotal(filtertArr.length);
    setStatefulSetList([...filtertArr]);
  };

  // 非混部table通过name查找
  const getStatefulSetResourceTableByName = () => {
    statefulSetForm.setFieldsValue({ statefulSetKeyValue: '' });
    getStatefulSetResource();
  };

  // 非混部table通过name查找
  const getStatefulSetResourceTableByLabel = () => {
    statefulSetForm.setFieldsValue({ statefulSetName: '' });
    getStatefulSetResource();
  };

  // 混部表格操作
  const handleColocationStatefulSetDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setColocationStatefulSetPage(pagination.current || 1);
      setColocationStatefulSetDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setColocationStatefulSetDetailFilterObj(filter);
      setColocationStatefulSetPage(1);
      setColocationStatefulSetDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setColocationStatefulSetSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setColocationStatefulSetPage(1);
      setColocationStatefulSetDetailPageSize(10);
    }
  };

  // 非混部表格操作
  const handleStatefulSetDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setStatefulSetPage(pagination.current || 1);
      setStatefulSetDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setStatefulSetDetailFilterObj(filter);
      setStatefulSetPage(1);
      setStatefulSetDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setStatefulSetSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setStatefulSetPage(1);
      setStatefulSetDetailPageSize(10);
    }
  };

  // 改变期望选择
  const handleHopeChange = async (checked, data) => {
    let newtablelist = JSON.parse(JSON.stringify(statefulSetList));
    newtablelist.forEach((item) => {
      if (item.name === data.name) {
        item.colocationType = reseverStatusName(checked);
      }
    });
    let newOrgtablelist = originalList;
    newOrgtablelist.forEach((item) => {
      if (item.name === data.name) {
        item.colocationType = reseverStatusName(checked);
      }
    });
    const intersection = newtablelist.filter((obj1) => selectedRows.some((obj2) => obj2.name === obj1.name));
    setStatefulSetList(newtablelist);
    setSelectedRows(intersection);
  };

  // 点击出现改变混部属性的弹窗
  const changeColocationStatefulSetState = (record, type) => {
    setPopOpen('');
    if (type === 'off') {
      setChoseState(intl.formatMessage({ id: 'workload.status.offline' }));
    } else {
      setChoseState(intl.formatMessage({ id: 'workload.status.oversold' }));
    }
    setColocationStatefulSetState(true);
    setDeleteId(record.key);
  };

  const openStatefulSetButton = () => {
    setSelectionType(true);
    setColShow(true);
    setIsStatefulSetWithoutPaddingBottom('has-padding');
    getStatefulSetList();
  };

  // 混部选择
  const onStatefulSetSelectChange = (newSelectedRowKeys, newSelectedRows) => {
    setSelectedRowKeys(newSelectedRowKeys);
    setSelectedRows(newSelectedRows);
  };

  // 混部model的确定按钮
  const handleStatefulSetStateModelConfirm = async (type) => {
    if (type === intl.formatMessage({ id: 'workload.status.offline' })) {
      // 离线代接口
      try {
        const res = await updateStatefulSetsExpectStatusToOffline(deleteId);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
          getColocationStatefulSetList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.modifyFailed' }));
      }
    } else {
      // 超卖代接口
      messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
      getColocationStatefulSetList();
    }
    setColocationStatefulSetState(false);
  };

  // 混部model的取消按钮
  const handleStatefulSetStateModelCancel = async () => {
    setColocationStatefulSetState(false);
  };

  // 重置按钮
  const handleStatefulSetReset = async () => {
    setStatefulSetLoading(true);
    await getNonColocationStatefulSetsListData();
    getStatefulSetResource();
    setStatefulSetLoading(false);
  };

  // 重置按钮
  const handleColocationStatefulSetReset = async () => {
    await getColocationStatefulSetList();
    getColocationStatefulSetResource();
  };

  // 条件重置的公共部分
  const restFilter = () => {
    statefulSetForm.setFieldsValue({ statefulSetName: '' });
    statefulSetForm.setFieldsValue({ statefulSetKeyValue: '' });
    setCurrentStatefulSetNamespace(''); // 重置筛选条件
    setStatefulSetSortObj({});
    setStatefulSetDetailFilterObj({});
    setSelectedRowKeys([]);
    setSelectedRows([]);
    setStatefulSetPage(1);
    setStatefulSetDetailPageSize(10);
  };

  // table的取消 重置 确定
  const onStatefulSetCancel = () => {
    setColShow(false);
    setIsStatefulSetWithoutPaddingBottom('no-padding');
    setSelectionType(false);
    getStatefulSetResource();
    restFilter();
  };

  const onStatefulSetRest = () => {
    restFilter();
    setStatefulSetList(JSON.parse(JSON.stringify(attTypeList)));
    setOriginalList(JSON.parse(JSON.stringify(attTypeList)));
  };

  const onStatefulSetSure = async () => {
    if (selectedRows.length === 0) {
      messageApi.error(intl.formatMessage({ id: 'message.selectAtLeastOneWorkload' }));
    } else if (selectedRows.length > 10) {
      messageApi.error(intl.formatMessage({ id: 'message.maxTenWorkloads' }));
    } else {
      // 代接口 selectedRowKeys, selectedRows
      try {
        const res = await addColocationStatefulSets(selectedRows);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.addColocationSuccess' }, { type: 'StatefulSet' }));
          setColShow(false);
          setIsStatefulSetWithoutPaddingBottom('no-padding');
          statefulSetForm.resetFields();
          setSelectedRowKeys([]);
          setSelectedRows([]);

          getColocationStatefulSetResource();
          setCurrentColocationStatefulSetNamespace(''); // 重置筛选条件
          setColocationStatefulSetSortObj({});
          setColocationStatefulSetDetailFilterObj({});
          setColocationStatefulSetPage(1);

          getStatefulSetResource();
          setCurrentStatefulSetNamespace(''); // 重置筛选条件
          setStatefulSetSortObj({});
          setStatefulSetDetailFilterObj({});
          setStatefulSetPage(1);

          getColocationStatefulSetList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.addColocationFailed' }, { type: 'StatefulSet' }));
      }
    }
  };

  useEffect(() => {
    getColocationStatefulSetResource();
  }, [colocationStatefulSetDetailFilterObj, colocationStatefulSetSortObj]);

  useEffect(() => {
    getStatefulSetResource();
  }, [statefulSetDetailFilterObj, statefulSetSortObj]);

  useEffect(() => {
    getColocationStatefulSetList();
  }, [getColocationStatefulSetList]);

  return (
    <div className="workload_content">
      <div className="workload_detail_content">
        <div className={`${isShow}`} style={{ marginTop: '20px' }}>
          <ToolTipComponent>
            <div
              className="statefulSetDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'message.qosTip' })}</div>
              <CloseOutlined onClick={handleStatefulSetShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className={`${isShow}workload_detail_content_box`}>
          <div className="workload_detail_form">
            <Form className="workload-searchForm form_padding_bottom" form={statefulSetForm}>
              {colShow ? (
                <div className="workload-searchForm-box">
                  <Form.Item name="statefulSetName" className="workload-search-input" style={{ marginRight: '20px' }}>
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                      onSearch={() => getStatefulSetResourceTableByName()}
                      autoComplete="off"
                      maxLength={53}
                    />
                  </Form.Item>
                  <Form.Item name="statefulSetKeyValue" className="workload-search-input">
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.byLabel' })}
                      onSearch={() => getStatefulSetResourceTableByLabel()}
                      autoComplete="off"
                    />
                  </Form.Item>
                </div>
              ) : (
                <Form.Item
                  name="colocationStatefulSetName"
                  className="workload-search-input"
                  style={{ marginRight: '20px' }}
                >
                  <Input.Search
                    placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                    onSearch={() => getColocationStatefulSetResource()}
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
                      onClick={() => openStatefulSetButton()}
                    >
                      {intl.formatMessage({ id: 'button.addColocation' }, { type: 'StatefulSet' })}
                    </Button>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleColocationStatefulSetReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                ) : (
                  <Space>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleStatefulSetReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                )}
              </Form.Item>
            </Form>
          </div>
          <div
            className={`workLoad-colocation-table-${isStatefulSetWithoutPaddingBottom}  workload_detail_content_table`}
          >
            {contextHolder}
            {colShow ? (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={statefulSetLoading}
                columns={statefulSetColumns}
                dataSource={statefulSetList}
                rowSelection={
                  selectionType
                    ? {
                        type: 'checkbox',
                        selectedRowKeys,
                        preserveSelectedRowKeys: true,
                        onChange: onStatefulSetSelectChange,
                      }
                    : null
                }
                pagination={{
                  className: 'colocation-website-page',
                  current: statefulSetPage,
                  pageSize: statefulSetDetailPageSize,
                  total: statefulSetDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleStatefulSetDetailTableChange}
              ></Table>
            ) : (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={statefulSetLoading}
                columns={colocationStatefulSetColumns}
                dataSource={colocationStatefulSetList}
                pagination={{
                  className: 'colocation-website-page',
                  current: colocationStatefulSetPage,
                  pageSize: colocationStatefulSetDetailPageSize,
                  total: colocationStatefulSetDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleColocationStatefulSetDetailTableChange}
              ></Table>
            )}
          </div>
        </div>
      </div>
      <ColocationStateModal
        title={intl.formatMessage({ id: 'message.changeToState' }, { state: choseState })}
        open={colocationStatefulSetState}
        cancelFn={handleStatefulSetStateModelCancel}
        confirmFn={() => handleStatefulSetStateModelConfirm(choseState)}
        state={choseState}
      />
      <ColocationDeleteModal
        title={intl.formatMessage({ id: 'action.setNonColocationWorkload' })}
        open={deleteStatefulSetModal}
        cancelFn={handleDeleteStatefulSetCancel}
        content={[intl.formatMessage({ id: 'message.confirmSetNonColocation' }, { name: deleteStatefulSetName })]}
        confirmFn={handleDeleteStatefulSetConfirmBack}
      />
      {colShow ? (
        <div className="child_content_workload_footer">
          <div className="workload_btn_footer_text">
            <p>{intl.formatMessage({ id: 'message.selectedWorkloads' }, { count: selectedRows.length })}</p>
          </div>
          <div className="workload_btn_footer_button">
            <Button className="cancel_btn" onClick={onStatefulSetCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={onStatefulSetRest}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={onStatefulSetSure}>
              {intl.formatMessage({ id: 'button.ok' })}
            </Button>
          </div>
        </div>
      ) : (
        <></>
      )}
    </div>
  );
}
