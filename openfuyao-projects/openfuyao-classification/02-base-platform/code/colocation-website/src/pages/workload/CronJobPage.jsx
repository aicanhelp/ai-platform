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
  getColocationCronJobsListData,
  getNonColocationCronJobsListData,
  addColocationCronJobs,
  updateCronJobsExpectStatusToOnline,
  updateCronJobsExpectStatusToOffline,
  deleteColocationCronJobsById,
} from '../../api/colocationApi';
import ToolTipComponent from '@/components/ToolTipComponent';

export default function CronJobPage() {
  const intl = useIntl();
  const [cronJobForm] = Form.useForm();
  const [popOpen, setPopOpen] = useState('');
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [colShow, setColShow] = useState(false);
  const [isShow, setIsShow] = useState('');
  const [messageApi, contextHolder] = message.useMessage();
  // table部分

  // 非混部table
  const [cronJobSortObj, setCronJobSortObj] = useState({}); // 排序
  const [cronJobDetailFilterObj, setCronJobDetailFilterObj] = useState({}); // 筛选
  const [cronJobDetailTotal, setCronJobDetailTotal] = useState(0);
  const [cronJobPage, setCronJobPage] = useState(1);
  const [cronJobDetailPageSize, setCronJobDetailPageSize] = useState(10);
  const [originalList, setOriginalList] = useState([]); // 非混部原始数据
  const [cronJobLoading, setCronJobLoading] = useState(false);
  const [cronJobList, setCronJobList] = useState([]); // 非混部cronJob table数据集
  const [selectionType, setSelectionType] = useState(false);
  const [filterCronJobNamespace, setFilterCronJobNamespace] = useState([]); // namespace赋值筛选项
  const [currentCronJobNamespace, setCurrentCronJobNamespace] = useState('');
  const [attTypeList, setAttTypeList] = useState([]); // 超级原始table数据集
  // 混部table
  const [colocationCronJobSortObj, setColocationCronJobSortObj] = useState({}); // 排序
  const [colocationCronJobDetailFilterObj, setColocationCronJobDetailFilterObj] = useState({}); // 筛选
  const [colocationCronJobDetailTotal, setColocationCronJobDetailTotal] = useState(0);
  const [colocationCronJobPage, setColocationCronJobPage] = useState(1);
  const [colocationCronJobDetailPageSize, setColocationCronJobDetailPageSize] = useState(10);
  const [colocationOriginalList, setColocationOriginalList] = useState([]); // 混部原始数据
  const [colocationCronJobList, setColocationCronJobList] = useState([]); // 混部cronJob table数据集
  const [colocationFilterCronJobNamespace, setColocationFilterCronJobNamespace] = useState([]); // namespace赋值筛选项
  const [currentColocationCronJobNamespace, setCurrentColocationCronJobNamespace] = useState('');
  const [colocationFilterCronJobStatus, setColocationFilterCronJobStatus] = useState([
    { text: intl.formatMessage({ id: 'workload.status.online' }), value: '在线' },
    { text: intl.formatMessage({ id: 'workload.status.offline' }), value: '离线' },
    { text: intl.formatMessage({ id: 'workload.status.oversold' }), value: '超卖' },
  ]);
  const [currentColocationCronJobStatus, setCurrentColocationCronJobStatus] = useState('');
  const [isCronJobWithoutPaddingBottom, setIsCronJobWithoutPaddingBottom] = useState('no-padding');
  // 混部状态弹窗
  const [colocationCronJobState, setColocationCronJobState] = useState(false);
  const [choseState, setChoseState] = useState(intl.formatMessage({ id: 'workload.status.online' }));
  // 删除提醒弹窗
  const [deleteCronJobModal, setDeleteCronJobModal] = useState(false);
  const [deleteCronJobName, setDeleteCronJobName] = useState('');
  const [deleteId, setDeleteId] = useState('');

  const handleCronJobShowChange = () => {
    setIsShow('cant_show');
  };

  // 混部节点移除
  const handleDeleteCronJob = async (record) => {
    setPopOpen('');
    setDeleteCronJobModal(true);
    setDeleteCronJobName(record.name);
    setDeleteId(record.key);
  };

  // 设置非混部的确定按钮
  const handleDeleteCronJobConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationCronJobsById(deleteId);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'message.nonColocationSetSuccess' }));
        setTimeout(() => {
          getColocationCronJobList();
        }, 1000);
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.nonColocationSetFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setDeleteCronJobModal(false);
  };

  // 设置非混部的取消按钮
  const handleDeleteCronJobCancel = () => {
    setDeleteCronJobModal(false);
  };

  // 混部cronJob列表项
  const colocationCronJobColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'colocationCronJobName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: colocationCronJobSortObj.key === 'colocationCronJobName' ? colocationCronJobSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'colocationCronJobNamespace',
      filters: colocationFilterCronJobNamespace,
      filteredValue: currentColocationCronJobNamespace ? [currentColocationCronJobNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: colocationCronJobSortObj.key === 'colocationCronJobNamespace' ? colocationCronJobSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'colocation_cronJob_createTime',
      sorter: true,
      sortOrder: colocationCronJobSortObj.key === 'colocation_cronJob_createTime' ? colocationCronJobSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.businessAttr' }),
      key: 'colocation_cronJob_status',
      filters: colocationFilterCronJobStatus,
      filteredValue: currentColocationCronJobStatus ? [currentColocationCronJobStatus] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: colocationCronJobSortObj.key === 'colocation_cronJob_status' ? colocationCronJobSortObj.order : '',
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
                    <Button type="link" onClick={() => changeColocationCronJobState(record, 'off')}>
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
                    handleDeleteCronJob(record);
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

  // 非混部cronJob列表项
  const cronJobColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'cronJobName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: cronJobSortObj.key === 'cronJobName' ? cronJobSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'cronJobNamespace',
      filters: filterCronJobNamespace,
      filteredValue: currentCronJobNamespace ? [currentCronJobNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: cronJobSortObj.key === 'cronJobNamespace' ? cronJobSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'cronJob_createTime',
      sorter: true,
      sortOrder: cronJobSortObj.key === 'cronJob_createTime' ? cronJobSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.expectedBusinessAttr' }),
      key: 'cronJob_status',
      width: 160,
      render: (_, record) => (
        <div>
          <Select
            className="table_hope_status"
            value={filterStatusName(record.colocationType, intl)}
            onChange={(e) => handleCronJobHopeChange(e, record)}
            options={getWorkloadStatusOptions(intl)}
          />
        </div>
      ),
    },
  ];

  // 获取非混部CronJobList
  const getCronJobList = useCallback(async () => {
    setCronJobLoading(true);
    try {
      const res = await getNonColocationCronJobsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          item.colocationType = 'online';
          return {
            ...item,
            key: item.uid,
          };
        });
        setCronJobList(arr);
        setOriginalList(JSON.parse(JSON.stringify(arr)));
        setAttTypeList(JSON.parse(JSON.stringify(JSON.parse(JSON.stringify(arr)))));
        setCronJobDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setFilterCronJobNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setCronJobList([]); // 数组为空
    }
    setCronJobLoading(false);
  }, []);

  // 获取混部CronJobList
  const getColocationCronJobList = useCallback(async () => {
    setCronJobLoading(true);
    try {
      const res = await getColocationCronJobsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        setColocationCronJobList(arr);
        setColocationOriginalList(arr);
        setColocationCronJobDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setColocationFilterCronJobNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setColocationCronJobList([]); // 数组为空
    }
    setCronJobLoading(false);
  }, []);

  // 混部具体表格操作
  const getColocationCronJobResource = () => {
    let filtertArr = [];
    if (cronJobForm.getFieldsValue().colocationCronJobName) {
      filtertArr = colocationOriginalList.filter((item) =>
        item.name.toLowerCase().includes(cronJobForm.getFieldsValue().colocationCronJobName.toLowerCase()),
      );
    } else {
      filtertArr = JSON.parse(JSON.stringify(colocationOriginalList || []));
    }
    if (Object.keys(colocationCronJobDetailFilterObj).length !== 0) {
      if (
        colocationCronJobDetailFilterObj.colocationCronJobNamespace &&
        colocationCronJobDetailFilterObj.colocation_cronJob_status
      ) {
        let [temporyNamespace, ...reset] = colocationCronJobDetailFilterObj.colocationCronJobNamespace;
        let [temporyStatus, ...reset1] = colocationCronJobDetailFilterObj.colocation_cronJob_status;
        setCurrentColocationCronJobNamespace(temporyNamespace);
        setCurrentColocationCronJobStatus(temporyStatus);
        filtertArr = filtertArr.filter(
          (item) =>
            item.namespace === temporyNamespace && filterStatusName(item.colocationType, intl) === temporyStatus,
        );
      } else {
        if (colocationCronJobDetailFilterObj.colocationCronJobNamespace) {
          setCurrentColocationCronJobStatus('');
          let [temporyNamespace, ...reset] = colocationCronJobDetailFilterObj.colocationCronJobNamespace;
          setCurrentColocationCronJobNamespace(temporyNamespace);
          filtertArr = filtertArr.filter((item) => item.namespace === temporyNamespace);
        } else if (colocationCronJobDetailFilterObj.colocation_cronJob_status) {
          setCurrentColocationCronJobNamespace('');
          let [temporyStatus, ...reset1] = colocationCronJobDetailFilterObj.colocation_cronJob_status;
          setCurrentColocationCronJobStatus(temporyStatus);
          filtertArr = filtertArr.filter((item) => filterStatusName(item.colocationType, intl) === temporyStatus);
        } else {
          setCurrentColocationCronJobNamespace('');
          setCurrentColocationCronJobStatus('');
        }
      }
    }
    if (Object.keys(colocationCronJobSortObj).length !== 0) {
      if (colocationCronJobSortObj.key === 'colocationCronJobName') {
        if (colocationCronJobSortObj.order) {
          colocationCronJobSortObj.order === 'ascend'
            ? sortWorkLoadByNameAscend('cronJob', filtertArr)
            : sortWorkLoadByNameDescend('cronJob', filtertArr);
        }
      } else if (colocationCronJobSortObj.key === 'colocationCronJobNamespace') {
        if (colocationCronJobSortObj.order) {
          colocationCronJobSortObj.order === 'ascend'
            ? sortWorkLoadByNamespaceAscend('cronJob', filtertArr)
            : sortWorkLoadByNamespaceDescend('cronJob', filtertArr);
        }
      } else if (colocationCronJobSortObj.key === 'colocation_cronJob_createTime') {
        if (colocationCronJobSortObj.order) {
          colocationCronJobSortObj.order === 'ascend'
            ? sortWorkLoadByTimeAscend('cronJob', filtertArr)
            : sortWorkLoadByTimeDescend('cronJob', filtertArr);
        }
      } else {
        if (colocationCronJobSortObj.order) {
          colocationCronJobSortObj.order === 'ascend'
            ? sortWorkLoadByStatusAscend('cronJob', filtertArr)
            : sortWorkLoadByStatusDescend('cronJob', filtertArr);
        }
      }
    }
    setColocationCronJobDetailTotal(filtertArr.length);
    setColocationCronJobList([...filtertArr]);
  };

  // 非混部具体表格操作
  const getCronJobResource = () => {
    let filtertArr = [];
    if (cronJobForm.getFieldsValue().cronJobName) {
      filtertArr = originalList.filter((item) =>
        item.name.toLowerCase().includes(cronJobForm.getFieldsValue().cronJobName.toLowerCase()),
      );
    } else if (cronJobForm.getFieldsValue().cronJobKeyValue) {
      filtertArr = originalList.filter((item) => {
        return compareObjKeys(item, cronJobForm.getFieldsValue().cronJobKeyValue);
      });
    } else {
      filtertArr = JSON.parse(JSON.stringify(originalList || []));
    }
    if (Object.keys(cronJobDetailFilterObj).length !== 0) {
      if (!cronJobDetailFilterObj.cronJobNamespace) {
        setCurrentCronJobNamespace('');
      } else {
        // 获取筛选框
        let [temporyStatus, ...reset] = cronJobDetailFilterObj.cronJobNamespace;
        setCurrentCronJobNamespace(temporyStatus);
        filtertArr = filtertArr.filter((item) => item.namespace === temporyStatus);
      }
    }
    if (Object.keys(cronJobSortObj).length !== 0) {
      if (cronJobSortObj.key === 'cronJobName') {
        if (cronJobSortObj.order) {
          cronJobSortObj.order === 'ascend'
            ? sortWorkLoadByNameAscend('cronJob', filtertArr)
            : sortWorkLoadByNameDescend('cronJob', filtertArr);
        }
      } else if (cronJobSortObj.key === 'cronJobNamespace') {
        if (cronJobSortObj.order) {
          cronJobSortObj.order === 'ascend'
            ? sortWorkLoadByNamespaceAscend('cronJob', filtertArr)
            : sortWorkLoadByNamespaceDescend('cronJob', filtertArr);
        }
      } else {
        if (cronJobSortObj.order) {
          cronJobSortObj.order === 'ascend'
            ? sortWorkLoadByTimeAscend('cronJob', filtertArr)
            : sortWorkLoadByTimeDescend('cronJob', filtertArr);
        }
      }
    }
    setCronJobDetailTotal(filtertArr.length);
    setCronJobList([...filtertArr]);
  };

  // 非混部table通过name查找
  const getCronJobResourceTableByName = () => {
    cronJobForm.setFieldsValue({ cronJobKeyValue: '' });
    getCronJobResource();
  };

  // 非混部table通过name查找
  const getCronJobResourceTableByLabel = () => {
    cronJobForm.setFieldsValue({ cronJobName: '' });
    getCronJobResource();
  };

  // 混部表格操作
  const handleColocationCronJobDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setColocationCronJobPage(pagination.current || 1);
      setColocationCronJobDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setColocationCronJobDetailFilterObj(filter);
      setColocationCronJobPage(1);
      setColocationCronJobDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setColocationCronJobSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setColocationCronJobPage(1);
      setColocationCronJobDetailPageSize(10);
    }
  };

  // 非混部表格操作
  const handleCronJobDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setCronJobPage(pagination.current || 1);
      setCronJobDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setCronJobDetailFilterObj(filter);
      setCronJobPage(1);
      setCronJobDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setCronJobSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setCronJobPage(1);
      setCronJobDetailPageSize(10);
    }
  };

  // 改变期望选择
  const handleCronJobHopeChange = async (checked, data) => {
    let newtablelist = JSON.parse(JSON.stringify(cronJobList));
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
    setCronJobList(newtablelist);
    setOriginalList(newOrgtablelist);
    setSelectedRows(intersection);
  };

  // 点击出现改变混部属性的弹窗
  const changeColocationCronJobState = (record, type) => {
    setPopOpen('');
    if (type === 'off') {
      setChoseState(intl.formatMessage({ id: 'workload.status.offline' }));
    } else {
      setChoseState(intl.formatMessage({ id: 'workload.status.oversold' }));
    }
    setColocationCronJobState(true);
    setDeleteId(record.key);
  };

  const openCronJobButton = () => {
    setSelectionType(true);
    setColShow(true);
    setIsCronJobWithoutPaddingBottom('has-padding');
    getCronJobList();
  };

  // 混部选择
  const onCronJobSelectChange = (newSelectedRowKeys, newSelectedRows) => {
    setSelectedRowKeys(newSelectedRowKeys);
    setSelectedRows(newSelectedRows);
  };

  // 混部model的确定按钮
  const handleCronJobStateModelConfirm = async (type) => {
    if (type === intl.formatMessage({ id: 'workload.status.offline' })) {
      // 离线代接口
      try {
        const res = await updateCronJobsExpectStatusToOffline(deleteId);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
          getColocationCronJobList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.modifyFailed' }));
      }
    } else {
      // 超卖代接口
      messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
      getColocationCronJobList();
    }
    setColocationCronJobState(false);
  };

  // 混部model的取消按钮
  const handleCronJobStateModelCancel = async () => {
    setColocationCronJobState(false);
  };

  // 重置按钮
  const handleCronJobReset = async () => {
    setCronJobLoading(true);
    await getNonColocationCronJobsListData();
    getCronJobResource();
    setCronJobLoading(false);
  };

  // 重置按钮
  const handleColocationCronJobReset = async () => {
    await getColocationCronJobList();
    getColocationCronJobResource();
  };

  // 条件重置的公共部分
  const restFilter = () => {
    cronJobForm.setFieldsValue({ cronJobName: '' });
    cronJobForm.setFieldsValue({ cronJobKeyValue: '' });
    setCurrentCronJobNamespace(''); // 重置筛选条件
    setCronJobSortObj({});
    setCronJobDetailFilterObj({});
    setSelectedRowKeys([]);
    setSelectedRows([]);
    setCronJobPage(1);
    setCronJobDetailPageSize(10);
  };

  // table的取消 重置 确定
  const onCronJobCancel = () => {
    setColShow(false);
    setIsCronJobWithoutPaddingBottom('no-padding');
    setSelectionType(false);
    getCronJobResource();
    restFilter();
  };

  const onCronJobRest = () => {
    restFilter();
    setCronJobList(JSON.parse(JSON.stringify(attTypeList)));
    setOriginalList(JSON.parse(JSON.stringify(attTypeList)));
  };

  const onCronJobSure = async () => {
    if (selectedRows.length === 0) {
      messageApi.error(intl.formatMessage({ id: 'message.selectAtLeastOneWorkload' }));
    } else if (selectedRows.length > 10) {
      messageApi.error(intl.formatMessage({ id: 'message.maxTenWorkloads' }));
    } else {
      // 代接口 selectedRowKeys, selectedRows
      try {
        const res = await addColocationCronJobs(selectedRows);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.addColocationSuccess' }, { type: 'CronJob' }));
          setColShow(false);
          setIsCronJobWithoutPaddingBottom('no-padding');
          cronJobForm.resetFields();
          setSelectedRowKeys([]);
          setSelectedRows([]);

          getColocationCronJobResource();
          setCurrentColocationCronJobNamespace(''); // 重置筛选条件
          setColocationCronJobSortObj({});
          setColocationCronJobDetailFilterObj({});
          setColocationCronJobPage(1);

          getCronJobResource();
          setCurrentCronJobNamespace(''); // 重置筛选条件
          setCronJobSortObj({});
          setCronJobDetailFilterObj({});
          setCronJobPage(1);

          getColocationCronJobList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.addColocationFailed' }, { type: 'CronJob' }));
      }
    }
  };

  useEffect(() => {
    getColocationCronJobResource();
  }, [colocationCronJobDetailFilterObj, colocationCronJobSortObj]);

  useEffect(() => {
    getCronJobResource();
  }, [cronJobDetailFilterObj, cronJobSortObj]);

  useEffect(() => {
    getColocationCronJobList();
  }, [getColocationCronJobList]);

  return (
    <div className="workload_content">
      <div className="workload_detail_content">
        <div className={`${isShow}`} style={{ marginTop: '20px' }}>
          <ToolTipComponent>
            <div
              className="cronJobDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'message.qosTip' })}</div>
              <CloseOutlined onClick={handleCronJobShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className={`${isShow}workload_detail_content_box`}>
          <div className="workload_detail_form">
            <Form className="workload-searchForm form_padding_bottom" form={cronJobForm}>
              {colShow ? (
                <div className="workload-searchForm-box">
                  <Form.Item name="cronJobName" className="workload-search-input" style={{ marginRight: '20px' }}>
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                      onSearch={() => getCronJobResourceTableByName()}
                      autoComplete="off"
                      maxLength={53}
                    />
                  </Form.Item>
                  <Form.Item name="cronJobKeyValue" className="workload-search-input">
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.byLabel' })}
                      onSearch={() => getCronJobResourceTableByLabel()}
                      autoComplete="off"
                    />
                  </Form.Item>
                </div>
              ) : (
                <Form.Item
                  name="colocationCronJobName"
                  className="workload-search-input"
                  style={{ marginRight: '20px' }}
                >
                  <Input.Search
                    placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                    onSearch={() => getColocationCronJobResource()}
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
                      onClick={() => openCronJobButton()}
                    >
                      {intl.formatMessage({ id: 'button.addColocation' }, { type: 'CronJob' })}
                    </Button>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleColocationCronJobReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                ) : (
                  <Space>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleCronJobReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                )}
              </Form.Item>
            </Form>
          </div>
          <div className={`workLoad-colocation-table-${isCronJobWithoutPaddingBottom}  workload_detail_content_table`}>
            {contextHolder}
            {colShow ? (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={cronJobLoading}
                columns={cronJobColumns}
                dataSource={cronJobList}
                rowSelection={
                  selectionType
                    ? {
                        type: 'checkbox',
                        selectedRowKeys,
                        preserveSelectedRowKeys: true,
                        onChange: onCronJobSelectChange,
                      }
                    : null
                }
                pagination={{
                  className: 'colocation-website-page',
                  current: cronJobPage,
                  pageSize: cronJobDetailPageSize,
                  total: cronJobDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleCronJobDetailTableChange}
              ></Table>
            ) : (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={cronJobLoading}
                columns={colocationCronJobColumns}
                dataSource={colocationCronJobList}
                pagination={{
                  className: 'colocation-website-page',
                  current: colocationCronJobPage,
                  pageSize: colocationCronJobDetailPageSize,
                  total: colocationCronJobDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleColocationCronJobDetailTableChange}
              ></Table>
            )}
          </div>
        </div>
      </div>
      <ColocationStateModal
        title={intl.formatMessage({ id: 'message.changeToState' }, { state: choseState })}
        open={colocationCronJobState}
        cancelFn={handleCronJobStateModelCancel}
        confirmFn={() => handleCronJobStateModelConfirm(choseState)}
        state={choseState}
      />
      <ColocationDeleteModal
        title={intl.formatMessage({ id: 'action.setNonColocationWorkload' })}
        open={deleteCronJobModal}
        cancelFn={handleDeleteCronJobCancel}
        content={[intl.formatMessage({ id: 'message.confirmSetNonColocation' }, { name: deleteCronJobName })]}
        confirmFn={handleDeleteCronJobConfirmBack}
      />
      {colShow ? (
        <div className="child_content_workload_footer">
          <div className="workload_btn_footer_text">
            <p>{intl.formatMessage({ id: 'message.selectedWorkloads' }, { count: selectedRows.length })}</p>
          </div>
          <div className="workload_btn_footer_button">
            <Button className="cancel_btn" onClick={onCronJobCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={onCronJobRest}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={onCronJobSure}>
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
