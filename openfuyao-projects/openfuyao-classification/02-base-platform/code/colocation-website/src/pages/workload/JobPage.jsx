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
  getColocationJobsListData,
  getNonColocationJobsListData,
  addColocationJobs,
  updateJobsExpectStatusToOnline,
  updateJobsExpectStatusToOffline,
  deleteColocationJobsById,
} from '../../api/colocationApi';
import ToolTipComponent from '@/components/ToolTipComponent';

export default function JobPage() {
  const intl = useIntl();
  const [jobForm] = Form.useForm();
  const [popOpen, setPopOpen] = useState('');
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [colShow, setColShow] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  const [isShow, setIsShow] = useState('');
  // table部分

  // 非混部table
  const [jobSortObj, setJobSortObj] = useState({}); // 排序
  const [jobDetailFilterObj, setJobDetailFilterObj] = useState({}); // 筛选
  const [jobDetailTotal, setJobDetailTotal] = useState(0);
  const [jobPage, setJobPage] = useState(1);
  const [jobDetailPageSize, setJobDetailPageSize] = useState(10);
  const [originalList, setOriginalList] = useState([]); // 非混部原始数据
  const [jobLoading, setJobLoading] = useState(false);
  const [jobList, setJobList] = useState([]); // 非混部job table数据集
  const [selectionType, setSelectionType] = useState(false);
  const [filterJobNamespace, setFilterJobNamespace] = useState([]); // namespace赋值筛选项
  const [currentJobNamespace, setCurrentJobNamespace] = useState('');
  const [attTypeList, setAttTypeList] = useState([]); // 超级原始table数据集
  // 混部table
  const [colocationJobSortObj, setColocationJobSortObj] = useState({}); // 排序
  const [colocationJobDetailFilterObj, setColocationJobDetailFilterObj] = useState({}); // 筛选
  const [colocationJobDetailTotal, setColocationJobDetailTotal] = useState(0);
  const [colocationJobPage, setColocationJobPage] = useState(1);
  const [colocationJobDetailPageSize, setColocationJobDetailPageSize] = useState(10);
  const [colocationOriginalList, setColocationOriginalList] = useState([]); // 混部原始数据
  const [colocationJobList, setColocationJobList] = useState([]); // 混部job table数据集
  const [colocationFilterJobNamespace, setColocationFilterJobNamespace] = useState([]); // namespace赋值筛选项
  const [currentColocationJobNamespace, setCurrentColocationJobNamespace] = useState('');
  const [colocationFilterJobStatus, setColocationFilterJobStatus] = useState([
    { text: intl.formatMessage({ id: 'workload.status.online' }), value: '在线' },
    { text: intl.formatMessage({ id: 'workload.status.offline' }), value: '离线' },
    { text: intl.formatMessage({ id: 'workload.status.oversold' }), value: '超卖' },
  ]); // 混部属性赋值筛选项
  const [currentColocationJobStatus, setCurrentColocationJobStatus] = useState('');
  const [isJobWithoutPaddingBottom, setIsJobWithoutPaddingBottom] = useState('no-padding');
  // 混部状态弹窗
  const [colocationJobState, setColocationJobState] = useState(false);
  const [choseState, setChoseState] = useState(intl.formatMessage({ id: 'workload.status.online' }));
  // 删除提醒弹窗
  const [deleteJobModal, setDeleteJobModal] = useState(false);
  const [deleteJobName, setDeleteJobName] = useState('');
  const [deleteId, setDeleteId] = useState('');

  const handleJobShowChange = () => {
    setIsShow('cant_show');
  };

  // 混部节点移除
  const handleDeleteJob = async (record) => {
    setPopOpen('');
    setDeleteJobModal(true);
    setDeleteJobName(record.name);
    setDeleteId(record.key);
  };

  // 设置非混部的确定按钮
  const handleDeleteJobConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationJobsById(deleteId);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'message.nonColocationSetSuccess' }));
        setTimeout(() => {
          getColocationJobList();
        }, 1000);
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.nonColocationSetFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setDeleteJobModal(false);
  };

  // 设置非混部的取消按钮
  const handleDeleteJobCancel = () => {
    setDeleteJobModal(false);
  };

  // 混部job列表项
  const colocationJobColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'colocationJobName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: colocationJobSortObj.key === 'colocationJobName' ? colocationJobSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'colocationJobNamespace',
      filters: colocationFilterJobNamespace,
      filteredValue: currentColocationJobNamespace ? [currentColocationJobNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: colocationJobSortObj.key === 'colocationJobNamespace' ? colocationJobSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'colocation_job_createTime',
      sorter: true,
      sortOrder: colocationJobSortObj.key === 'colocation_job_createTime' ? colocationJobSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.businessAttr' }),
      key: 'colocation_job_status',
      filters: colocationFilterJobStatus,
      filteredValue: currentColocationJobStatus ? [currentColocationJobStatus] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: colocationJobSortObj.key === 'colocation_job_status' ? colocationJobSortObj.order : '',
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
                    <Button type="link" onClick={() => changeColocationJobState(record, 'off')}>
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
                    handleDeleteJob(record);
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

  // 非混部job列表项
  const jobColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'jobName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: jobSortObj.key === 'jobName' ? jobSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'jobNamespace',
      filters: filterJobNamespace,
      filteredValue: currentJobNamespace ? [currentJobNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: jobSortObj.key === 'jobNamespace' ? jobSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'job_createTime',
      sorter: true,
      sortOrder: jobSortObj.key === 'job_createTime' ? jobSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.expectedBusinessAttr' }),
      key: 'job_status',
      width: 160,
      render: (_, record) => (
        <div>
          <Select
            className="table_hope_status"
            value={filterStatusName(record.colocationType, intl)}
            onChange={(e) => handleJobHopeChange(e, record)}
            options={getWorkloadStatusOptions(intl)}
          />
        </div>
      ),
    },
  ];

  // 获取非混部JobList
  const getJobList = useCallback(async () => {
    setJobLoading(true);
    try {
      const res = await getNonColocationJobsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          item.colocationType = 'online';
          return {
            ...item,
            key: item.uid,
          };
        });
        setJobList(arr);
        setOriginalList(JSON.parse(JSON.stringify(arr)));
        setAttTypeList(JSON.parse(JSON.stringify(JSON.parse(JSON.stringify(arr)))));
        setJobDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setFilterJobNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setJobList([]); // 数组为空
    }
    setJobLoading(false);
  }, []);

  // 获取混部JobList
  const getColocationJobList = useCallback(async () => {
    setJobLoading(true);
    try {
      const res = await getColocationJobsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        setColocationJobList(arr);
        setColocationOriginalList(arr);
        setColocationJobDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setColocationFilterJobNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setColocationJobList([]); // 数组为空
    }
    setJobLoading(false);
  }, []);

  // 混部具体表格操作
  const getColocationJobResource = () => {
    let filtertArr = [];
    if (jobForm.getFieldsValue().colocationJobName) {
      filtertArr = colocationOriginalList.filter((item) =>
        item.name.toLowerCase().includes(jobForm.getFieldsValue().colocationJobName.toLowerCase()),
      );
    } else {
      filtertArr = JSON.parse(JSON.stringify(colocationOriginalList || []));
    }
    if (Object.keys(colocationJobDetailFilterObj).length !== 0) {
      if (colocationJobDetailFilterObj.colocationJobNamespace && colocationJobDetailFilterObj.colocation_job_status) {
        let [temporyNamespace, ...reset] = colocationJobDetailFilterObj.colocationJobNamespace;
        let [temporyStatus, ...reset1] = colocationJobDetailFilterObj.colocation_job_status;
        setCurrentColocationJobNamespace(temporyNamespace);
        setCurrentColocationJobStatus(temporyStatus);
        filtertArr = filtertArr.filter(
          (item) =>
            item.namespace === temporyNamespace && filterStatusName(item.colocationType, intl) === temporyStatus,
        );
      } else {
        if (colocationJobDetailFilterObj.colocationJobNamespace) {
          setCurrentColocationJobStatus('');
          let [temporyNamespace, ...reset] = colocationJobDetailFilterObj.colocationJobNamespace;
          setCurrentColocationJobNamespace(temporyNamespace);
          filtertArr = filtertArr.filter((item) => item.namespace === temporyNamespace);
        } else if (colocationJobDetailFilterObj.colocation_job_status) {
          setCurrentColocationJobNamespace('');
          let [temporyStatus, ...reset1] = colocationJobDetailFilterObj.colocation_job_status;
          setCurrentColocationJobStatus(temporyStatus);
          filtertArr = filtertArr.filter((item) => filterStatusName(item.colocationType, intl) === temporyStatus);
        } else {
          setCurrentColocationJobNamespace('');
          setCurrentColocationJobStatus('');
        }
      }
    }
    if (Object.keys(colocationJobSortObj).length !== 0) {
      if (colocationJobSortObj.key === 'colocationJobName') {
        if (colocationJobSortObj.order) {
          colocationJobSortObj.order === 'ascend'
            ? sortWorkLoadByNameAscend('job', filtertArr)
            : sortWorkLoadByNameDescend('job', filtertArr);
        }
      } else if (colocationJobSortObj.key === 'colocationJobNamespace') {
        if (colocationJobSortObj.order) {
          colocationJobSortObj.order === 'ascend'
            ? sortWorkLoadByNamespaceAscend('job', filtertArr)
            : sortWorkLoadByNamespaceDescend('job', filtertArr);
        }
      } else if (colocationJobSortObj.key === 'colocation_job_createTime') {
        if (colocationJobSortObj.order) {
          colocationJobSortObj.order === 'ascend'
            ? sortWorkLoadByTimeAscend('job', filtertArr)
            : sortWorkLoadByTimeDescend('job', filtertArr);
        }
      } else {
        if (colocationJobSortObj.order) {
          colocationJobSortObj.order === 'ascend'
            ? sortWorkLoadByStatusAscend('job', filtertArr)
            : sortWorkLoadByStatusDescend('job', filtertArr);
        }
      }
    }
    setColocationJobDetailTotal(filtertArr.length);
    setColocationJobList([...filtertArr]);
  };

  // 非混部具体表格操作
  const getJobResource = () => {
    let filtertArr = [];
    if (jobForm.getFieldsValue().jobName) {
      filtertArr = originalList.filter((item) =>
        item.name.toLowerCase().includes(jobForm.getFieldsValue().jobName.toLowerCase()),
      );
    } else if (jobForm.getFieldsValue().jobKeyValue) {
      filtertArr = originalList.filter((item) => {
        return compareObjKeys(item, jobForm.getFieldsValue().jobKeyValue);
      });
    } else {
      filtertArr = JSON.parse(JSON.stringify(originalList || []));
    }
    if (Object.keys(jobDetailFilterObj).length !== 0) {
      if (!jobDetailFilterObj.jobNamespace) {
        setCurrentJobNamespace('');
      } else {
        // 获取筛选框
        let [temporyStatus, ...reset] = jobDetailFilterObj.jobNamespace;
        setCurrentJobNamespace(temporyStatus);
        filtertArr = filtertArr.filter((item) => item.namespace === temporyStatus);
      }
    }
    if (Object.keys(jobSortObj).length !== 0) {
      if (jobSortObj.key === 'jobName') {
        if (jobSortObj.order) {
          jobSortObj.order === 'ascend'
            ? sortWorkLoadByNameAscend('job', filtertArr)
            : sortWorkLoadByNameDescend('job', filtertArr);
        }
      } else if (jobSortObj.key === 'jobNamespace') {
        if (jobSortObj.order) {
          jobSortObj.order === 'ascend'
            ? sortWorkLoadByNamespaceAscend('job', filtertArr)
            : sortWorkLoadByNamespaceDescend('job', filtertArr);
        }
      } else {
        if (jobSortObj.order) {
          jobSortObj.order === 'ascend'
            ? sortWorkLoadByTimeAscend('job', filtertArr)
            : sortWorkLoadByTimeDescend('job', filtertArr);
        }
      }
    }
    setJobDetailTotal(filtertArr.length);
    setJobList([...filtertArr]);
  };

  // 非混部table通过name查找
  const getJobResourceTableByName = () => {
    jobForm.setFieldsValue({ jobKeyValue: '' });
    getJobResource();
  };

  // 非混部table通过name查找
  const getJobResourceTableByLabel = () => {
    jobForm.setFieldsValue({ jobName: '' });
    getJobResource();
  };

  // 混部表格操作
  const handleColocationJobDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setColocationJobPage(pagination.current || 1);
      setColocationJobDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setColocationJobDetailFilterObj(filter);
      setColocationJobPage(1);
      setColocationJobDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setColocationJobSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setColocationJobPage(1);
      setColocationJobDetailPageSize(10);
    }
  };

  // 非混部表格操作
  const handleJobDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setJobPage(pagination.current || 1);
      setJobDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setJobDetailFilterObj(filter);
      setJobPage(1);
      setJobDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setJobSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setJobPage(1);
      setJobDetailPageSize(10);
    }
  };

  // 改变期望选择
  const handleJobHopeChange = async (checked, data) => {
    let newtablelist = JSON.parse(JSON.stringify(jobList));
    newtablelist.forEach((item) => {
      if (item.name === data.name) {
        item.colocationType = reseverStatusName(checked, intl);
      }
    });
    let newOrgtablelist = originalList;
    newOrgtablelist.forEach((item) => {
      if (item.name === data.name) {
        item.colocationType = reseverStatusName(checked, intl);
      }
    });
    const intersection = newtablelist.filter((obj1) => selectedRows.some((obj2) => obj2.name === obj1.name));
    setJobList(newtablelist);
    setOriginalList(newOrgtablelist);
    setSelectedRows(intersection);
  };

  // 点击出现改变混部属性的弹窗
  const changeColocationJobState = (record, type) => {
    setPopOpen('');
    if (type === 'off') {
      setChoseState(intl.formatMessage({ id: 'workload.status.offline' }));
    } else {
      setChoseState(intl.formatMessage({ id: 'workload.status.oversold' }));
    }
    setColocationJobState(true);
    setDeleteId(record.key);
  };

  const openJobButton = () => {
    setSelectionType(true);
    setColShow(true);
    setIsJobWithoutPaddingBottom('has-padding');
    getJobList();
  };

  // 混部选择
  const onJobSelectChange = (newSelectedRowKeys, newSelectedRows) => {
    setSelectedRowKeys(newSelectedRowKeys);
    setSelectedRows(newSelectedRows);
  };

  // 混部model的确定按钮
  const handleJobStateModelConfirm = async (type) => {
    if (type === intl.formatMessage({ id: 'workload.status.offline' })) {
      // 离线代接口
      try {
        const res = await updateJobsExpectStatusToOffline(deleteId);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
          getColocationJobList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.modifyFailed' }));
      }
    } else {
      // 超卖代接口
      messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
      getColocationJobList();
    }
    setColocationJobState(false);
  };

  // 混部model的取消按钮
  const handleJobStateModelCancel = async () => {
    setColocationJobState(false);
  };

  // 重置按钮
  const handleJobReset = async () => {
    setJobLoading(true);
    await getNonColocationJobsListData();
    getJobResource();
    setJobLoading(false);
  };

  // 重置按钮
  const handleColocationJobReset = async () => {
    await getColocationJobList();
    getColocationJobResource();
  };

  // 条件重置的公共部分
  const restFilter = () => {
    jobForm.setFieldsValue({ jobName: '' });
    jobForm.setFieldsValue({ jobKeyValue: '' });
    setCurrentJobNamespace(''); // 重置筛选条件
    setJobSortObj({});
    setJobDetailFilterObj({});
    setSelectedRowKeys([]);
    setSelectedRows([]);
    setJobPage(1);
    setJobDetailPageSize(10);
  };

  // table的取消 重置 确定
  const onJobCancel = () => {
    setColShow(false);
    setIsJobWithoutPaddingBottom('no-padding');
    setSelectionType(false);
    getJobResource();
    restFilter();
  };

  const onJobRest = () => {
    restFilter();
    setJobList(JSON.parse(JSON.stringify(attTypeList)));
    setOriginalList(JSON.parse(JSON.stringify(attTypeList)));
  };

  const onJobSure = async () => {
    if (selectedRows.length === 0) {
      messageApi.error(intl.formatMessage({ id: 'message.selectAtLeastOneWorkload' }));
    } else if (selectedRows.length > 10) {
      messageApi.error(intl.formatMessage({ id: 'message.maxTenWorkloads' }));
    } else {
      // 代接口 selectedRowKeys, selectedRows
      try {
        const res = await addColocationJobs(selectedRows);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.addColocationSuccess' }, { type: 'Job' }));
          setColShow(false);
          setIsJobWithoutPaddingBottom('no-padding');
          jobForm.resetFields();
          setSelectedRowKeys([]);
          setSelectedRows([]);

          getColocationJobResource();
          setCurrentColocationJobNamespace(''); // 重置筛选条件
          setColocationJobSortObj({});
          setColocationJobDetailFilterObj({});
          setColocationJobPage(1);

          getJobResource();
          setCurrentJobNamespace(''); // 重置筛选条件
          setJobSortObj({});
          setJobDetailFilterObj({});
          setJobPage(1);

          getColocationJobList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.addColocationFailed' }, { type: 'Job' }));
      }
    }
  };

  useEffect(() => {
    getColocationJobResource();
  }, [colocationJobDetailFilterObj, colocationJobSortObj]);

  useEffect(() => {
    getJobResource();
  }, [jobDetailFilterObj, jobSortObj]);

  useEffect(() => {
    getColocationJobList();
  }, [getColocationJobList]);

  return (
    <div className="workload_content">
      <div className="workload_detail_content">
        <div className={`${isShow}`} style={{ marginTop: '20px' }}>
          <ToolTipComponent>
            <div
              className="jobDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'message.qosTip' })}</div>
              <CloseOutlined onClick={handleJobShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className={`${isShow}workload_detail_content_box`}>
          <div className="workload_detail_form">
            <Form className="workload-searchForm form_padding_bottom" form={jobForm}>
              {colShow ? (
                <div className="workload-searchForm-box">
                  <Form.Item name="jobName" className="workload-search-input" style={{ marginRight: '20px' }}>
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                      onSearch={() => getJobResourceTableByName()}
                      autoComplete="off"
                      maxLength={53}
                    />
                  </Form.Item>
                  <Form.Item name="jobKeyValue" className="workload-search-input">
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.byLabel' })}
                      onSearch={() => getJobResourceTableByLabel()}
                      autoComplete="off"
                    />
                  </Form.Item>
                </div>
              ) : (
                <Form.Item name="colocationJobName" className="workload-search-input" style={{ marginRight: '20px' }}>
                  <Input.Search
                    placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                    onSearch={() => getColocationJobResource()}
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
                      onClick={() => openJobButton()}
                    >
                      {intl.formatMessage({ id: 'button.addColocation' }, { type: 'Job' })}
                    </Button>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleColocationJobReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                ) : (
                  <Space>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleJobReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                )}
              </Form.Item>
            </Form>
          </div>
          <div className={`workLoad-colocation-table-${isJobWithoutPaddingBottom}  workload_detail_content_table`}>
            {contextHolder}
            {colShow ? (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={jobLoading}
                columns={jobColumns}
                dataSource={jobList}
                rowSelection={
                  selectionType
                    ? {
                        type: 'checkbox',
                        selectedRowKeys,
                        preserveSelectedRowKeys: true,
                        onChange: onJobSelectChange,
                      }
                    : null
                }
                pagination={{
                  className: 'colocation-website-page',
                  current: jobPage,
                  pageSize: jobDetailPageSize,
                  total: jobDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleJobDetailTableChange}
              ></Table>
            ) : (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={jobLoading}
                columns={colocationJobColumns}
                dataSource={colocationJobList}
                pagination={{
                  className: 'colocation-website-page',
                  current: colocationJobPage,
                  pageSize: colocationJobDetailPageSize,
                  total: colocationJobDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleColocationJobDetailTableChange}
              ></Table>
            )}
          </div>
        </div>
      </div>
      <ColocationStateModal
        title={intl.formatMessage({ id: 'message.changeToState' }, { state: choseState })}
        open={colocationJobState}
        cancelFn={handleJobStateModelCancel}
        confirmFn={() => handleJobStateModelConfirm(choseState)}
        state={choseState}
      />
      <ColocationDeleteModal
        title={intl.formatMessage({ id: 'action.setNonColocationWorkload' })}
        open={deleteJobModal}
        cancelFn={handleDeleteJobCancel}
        content={[intl.formatMessage({ id: 'message.confirmSetNonColocation' }, { name: deleteJobName })]}
        confirmFn={handleDeleteJobConfirmBack}
      />
      {colShow ? (
        <div className="child_content_workload_footer">
          <div className="workload_btn_footer_text">
            <p>{intl.formatMessage({ id: 'message.selectedWorkloads' }, { count: selectedRows.length })}</p>
          </div>
          <div className="workload_btn_footer_button">
            <Button className="cancel_btn" onClick={onJobCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={onJobRest}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={onJobSure}>
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
