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
  getColocationDeploymentsListData,
  getNonColocationDeploymentsListData,
  addColocationDeployments,
  updateDeploymentsExpectStatusToOffline,
  deleteColocationDeploymentsById,
} from '../../api/colocationApi';
import ToolTipComponent from '@/components/ToolTipComponent';

export default function DeploymentPage() {
  const intl = useIntl();
  const [deploymentForm] = Form.useForm();
  const [popOpen, setPopOpen] = useState('');
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [colShow, setColShow] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  const [isShow, setIsShow] = useState('');
  // table部分

  // 非混部table
  const [deploymentSortObj, setDeploymentSortObj] = useState({}); // 排序
  const [deploymentDetailFilterObj, setDeploymentDetailFilterObj] = useState({}); // 筛选
  const [deploymentDetailTotal, setDeploymentDetailTotal] = useState(0);
  const [deploymentPage, setDeploymentPage] = useState(1);
  const [deploymentDetailPageSize, setDeploymentDetailPageSize] = useState(10);
  const [originalList, setOriginalList] = useState([]); // 非混部原始数据
  const [deploymentLoading, setDeploymentLoading] = useState(false);
  const [deploymentList, setDeploymentList] = useState([]); // 非混部deployment table数据集
  const [selectionType, setSelectionType] = useState(false);
  const [filterDeploymentNamespace, setFilterDeploymentNamespace] = useState([]); // namespace赋值筛选项
  const [currentDeploymentNamespace, setCurrentDeploymentNamespace] = useState('');
  const [attTypeList, setAttTypeList] = useState([]); // 超级原始table数据集
  // 混部table
  const [colocationDeploymentSortObj, setColocationDeploymentSortObj] = useState({}); // 排序
  const [colocationDeploymentDetailFilterObj, setColocationDeploymentDetailFilterObj] = useState({}); // 筛选
  const [colocationDeploymentDetailTotal, setColocationDeploymentDetailTotal] = useState(0);
  const [colocationDeploymentPage, setColocationDeploymentPage] = useState(1);
  const [colocationDeploymentDetailPageSize, setColocationDeploymentDetailPageSize] = useState(10);
  const [colocationOriginalList, setColocationOriginalList] = useState([]); // 混部原始数据
  const [colocationDeploymentList, setColocationDeploymentList] = useState([]); // 混部deployment table数据集
  const [colocationFilterDeploymentNamespace, setColocationFilterDeploymentNamespace] = useState([]); // namespace赋值筛选项
  const [currentColocationDeploymentNamespace, setCurrentColocationDeploymentNamespace] = useState('');
  const [colocationFilterDeploymentStatus, setColocationFilterDeploymentStatus] = useState([
    { text: intl.formatMessage({ id: 'workload.status.online' }), value: '在线' },
    { text: intl.formatMessage({ id: 'workload.status.offline' }), value: '离线' },
    { text: intl.formatMessage({ id: 'workload.status.oversold' }), value: '超卖' },
  ]);
  const [currentColocationDeploymentStatus, setCurrentColocationDeploymentStatus] = useState('');
  const [isDeploymentWithoutPaddingBottom, setIsDeploymentWithoutPaddingBottom] = useState('no-padding');

  // 混部状态弹窗
  const [colocationDeploymentState, setColocationDeploymentState] = useState(false);
  const [choseState, setChoseState] = useState(intl.formatMessage({ id: 'workload.status.online' }));
  // 删除提醒弹窗
  const [deleteDeploymentModal, setDeleteDeploymentModal] = useState(false);
  const [deleteDeploymentName, setDeleteDeploymentName] = useState('');
  const [deleteId, setDeleteId] = useState('');

  const handleDeploymentShowChange = () => {
    setIsShow('cant_show');
  };

  // 混部节点移除
  const handleDeleteDeployment = async (record) => {
    setPopOpen('');
    setDeleteDeploymentModal(true);
    setDeleteDeploymentName(record.name);
    setDeleteId(record.key);
  };

  // 设置非混部的确定按钮
  const handleDeleteDeploymentConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationDeploymentsById(deleteId);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'message.nonColocationSetSuccess' }));
        setTimeout(() => {
          getColocationDeploymentList();
        }, 1000);
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.nonColocationSetFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setDeleteDeploymentModal(false);
  };

  // 设置非混部的取消按钮
  const handleDeleteDeploymentCancel = () => {
    setDeleteDeploymentModal(false);
  };

  // 混部deployment列表项
  const colocationDeploymentColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'colocationDeploymentName',
      dataIndex: 'name',
      sorter: true,
      sortOrder:
        colocationDeploymentSortObj.key === 'colocationDeploymentName' ? colocationDeploymentSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'colocationDeploymentNamespace',
      filters: colocationFilterDeploymentNamespace,
      filteredValue: currentColocationDeploymentNamespace ? [currentColocationDeploymentNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder:
        colocationDeploymentSortObj.key === 'colocationDeploymentNamespace' ? colocationDeploymentSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'colocation_deployment_createTime',
      sorter: true,
      sortOrder:
        colocationDeploymentSortObj.key === 'colocation_deployment_createTime' ? colocationDeploymentSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.businessAttr' }),
      key: 'colocation_deployment_status',
      filters: colocationFilterDeploymentStatus,
      filteredValue: currentColocationDeploymentStatus ? [currentColocationDeploymentStatus] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder:
        colocationDeploymentSortObj.key === 'colocation_deployment_status' ? colocationDeploymentSortObj.order : '',
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
                    <Button type="link" onClick={() => changeColocationDeploymentState(record, 'off')}>
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
                    handleDeleteDeployment(record);
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

  // 非混部deployment列表项
  const deploymentColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.workloadName' }),
      key: 'deploymentName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: deploymentSortObj.key === 'deploymentName' ? deploymentSortObj.order : '',
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'deploymentNamespace',
      filters: filterDeploymentNamespace,
      filteredValue: currentDeploymentNamespace ? [currentDeploymentNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: deploymentSortObj.key === 'deploymentNamespace' ? deploymentSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.createTime' }),
      key: 'deployment_createTime',
      sorter: true,
      sortOrder: deploymentSortObj.key === 'deployment_createTime' ? deploymentSortObj.order : '',
      render: (_, record) => (
        <Space>{Dayjs(record.creationTimestamp ? record.creationTimestamp : 'none').format('YYYY-MM-DD HH:mm')}</Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.expectedBusinessAttr' }),
      key: 'deployment_status',
      width: 160,
      render: (_, record) => (
        <div>
          <Select
            className="table_hope_status"
            value={filterStatusName(record.colocationType, intl)}
            onChange={(e) => handleHopeDeploymentChange(e, record)}
            options={getWorkloadStatusOptions(intl)}
          />
        </div>
      ),
    },
  ];

  // 获取非混部DeploymentList
  const getDeploymentList = useCallback(async () => {
    setDeploymentLoading(true);
    try {
      const res = await getNonColocationDeploymentsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          item.colocationType = 'online';
          return {
            ...item,
            key: item.uid,
          };
        });
        setDeploymentList(arr);
        setOriginalList(JSON.parse(JSON.stringify(arr)));
        setAttTypeList(JSON.parse(JSON.stringify(JSON.parse(JSON.stringify(arr)))));
        setDeploymentDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setFilterDeploymentNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setDeploymentList([]); // 数组为空
    }
    setDeploymentLoading(false);
  }, []);

  // 获取混部DeploymentList
  const getColocationDeploymentList = useCallback(async () => {
    setDeploymentLoading(true);
    try {
      const res = await getColocationDeploymentsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        setColocationDeploymentList(arr);
        setColocationOriginalList(arr);
        setColocationDeploymentDetailTotal(arr?.length || 0);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setColocationFilterDeploymentNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setColocationDeploymentList([]); // 数组为空
    }
    setDeploymentLoading(false);
  }, []);

  // 混部具体表格操作
  const getColocationDeploymentResource = () => {
    let filtertArr = [];
    if (deploymentForm.getFieldsValue().colocationDeploymentName) {
      filtertArr = colocationOriginalList.filter((item) =>
        item.name.toLowerCase().includes(deploymentForm.getFieldsValue().colocationDeploymentName.toLowerCase()),
      );
    } else {
      filtertArr = JSON.parse(JSON.stringify(colocationOriginalList || []));
    }
    if (Object.keys(colocationDeploymentDetailFilterObj).length !== 0) {
      if (
        colocationDeploymentDetailFilterObj.colocationDeploymentNamespace &&
        colocationDeploymentDetailFilterObj.colocation_deployment_status
      ) {
        // 获取筛选框
        let [temporyNamespace, ...reset] = colocationDeploymentDetailFilterObj.colocationDeploymentNamespace;
        let [temporyStatus, ...reset1] = colocationDeploymentDetailFilterObj.colocation_deployment_status;
        setCurrentColocationDeploymentNamespace(temporyNamespace);
        setCurrentColocationDeploymentStatus(temporyStatus);
        filtertArr = filtertArr.filter(
          (item) =>
            item.namespace === temporyNamespace && filterStatusName(item.colocationType, intl) === temporyStatus,
        );
      } else {
        if (colocationDeploymentDetailFilterObj.colocationDeploymentNamespace) {
          setCurrentColocationDeploymentStatus('');
          let [temporyNamespace, ...reset] = colocationDeploymentDetailFilterObj.colocationDeploymentNamespace;
          setCurrentColocationDeploymentNamespace(temporyNamespace);
          filtertArr = filtertArr.filter((item) => item.namespace === temporyNamespace);
        } else if (colocationDeploymentDetailFilterObj.colocation_deployment_status) {
          setCurrentColocationDeploymentNamespace('');
          let [temporyStatus, ...reset1] = colocationDeploymentDetailFilterObj.colocation_deployment_status;
          setCurrentColocationDeploymentStatus(temporyStatus);
          filtertArr = filtertArr.filter((item) => filterStatusName(item.colocationType, intl) === temporyStatus);
        } else {
          setCurrentColocationDeploymentNamespace('');
          setCurrentColocationDeploymentStatus('');
        }
      }
    }
    if (Object.keys(colocationDeploymentSortObj).length !== 0) {
      if (colocationDeploymentSortObj.key === 'colocationDeploymentName') {
        if (colocationDeploymentSortObj.order) {
          colocationDeploymentSortObj.order === 'ascend'
            ? sortWorkLoadByNameAscend('deployment', filtertArr)
            : sortWorkLoadByNameDescend('deployment', filtertArr);
        }
      } else if (colocationDeploymentSortObj.key === 'colocationDeploymentNamespace') {
        if (colocationDeploymentSortObj.order) {
          colocationDeploymentSortObj.order === 'ascend'
            ? sortWorkLoadByNamespaceAscend('deployment', filtertArr)
            : sortWorkLoadByNamespaceDescend('deployment', filtertArr);
        }
      } else if (colocationDeploymentSortObj.key === 'colocation_deployment_createTime') {
        if (colocationDeploymentSortObj.order) {
          colocationDeploymentSortObj.order === 'ascend'
            ? sortWorkLoadByTimeAscend('deployment', filtertArr)
            : sortWorkLoadByTimeDescend('deployment', filtertArr);
        }
      } else {
        if (colocationDeploymentSortObj.order) {
          colocationDeploymentSortObj.order === 'ascend'
            ? sortWorkLoadByStatusAscend('deployment', filtertArr)
            : sortWorkLoadByStatusDescend('deployment', filtertArr);
        }
      }
    }
    setColocationDeploymentDetailTotal(filtertArr.length);
    setColocationDeploymentList([...filtertArr]);
  };

  // 非混部具体表格操作
  const getDeploymentResource = () => {
    let filtertArr = [];
    if (deploymentForm.getFieldsValue().deploymentName) {
      filtertArr = originalList.filter((item) =>
        item.name.toLowerCase().includes(deploymentForm.getFieldsValue().deploymentName.toLowerCase()),
      );
    } else if (deploymentForm.getFieldsValue().deploymentKeyValue) {
      filtertArr = originalList.filter((item) => {
        return compareObjKeys(item, deploymentForm.getFieldsValue().deploymentKeyValue);
      });
    } else {
      filtertArr = JSON.parse(JSON.stringify(originalList || []));
    }
    if (Object.keys(deploymentDetailFilterObj).length !== 0) {
      if (!deploymentDetailFilterObj.deploymentNamespace) {
        setCurrentDeploymentNamespace('');
      } else {
        // 获取筛选框
        let [temporyStatus, ...reset] = deploymentDetailFilterObj.deploymentNamespace;
        setCurrentDeploymentNamespace(temporyStatus);
        filtertArr = filtertArr.filter((item) => item.namespace === temporyStatus);
      }
    }
    if (Object.keys(deploymentSortObj).length !== 0) {
      if (deploymentSortObj.key === 'deploymentName') {
        if (deploymentSortObj.order) {
          deploymentSortObj.order === 'ascend'
            ? sortWorkLoadByNameAscend('deployment', filtertArr)
            : sortWorkLoadByNameDescend('deployment', filtertArr);
        }
      } else if (deploymentSortObj.key === 'deploymentNamespace') {
        if (deploymentSortObj.order) {
          deploymentSortObj.order === 'ascend'
            ? sortWorkLoadByNamespaceAscend('deployment', filtertArr)
            : sortWorkLoadByNamespaceDescend('deployment', filtertArr);
        }
      } else {
        if (deploymentSortObj.order) {
          deploymentSortObj.order === 'ascend'
            ? sortWorkLoadByTimeAscend('deployment', filtertArr)
            : sortWorkLoadByTimeDescend('deployment', filtertArr);
        }
      }
    }
    setDeploymentDetailTotal(filtertArr.length);
    setDeploymentList([...filtertArr]);
  };

  // 非混部table通过name查找
  const getDeploymentResourceTableByName = () => {
    deploymentForm.setFieldsValue({ deploymentKeyValue: '' });
    getDeploymentResource();
  };

  // 非混部table通过name查找
  const getDeploymentResourceTableByLabel = () => {
    deploymentForm.setFieldsValue({ deploymentName: '' });
    getDeploymentResource();
  };

  // 混部表格操作
  const handleColocationDeploymentDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setColocationDeploymentPage(pagination.current || 1);
      setColocationDeploymentDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setColocationDeploymentDetailFilterObj(filter);
      setColocationDeploymentPage(1);
      setColocationDeploymentDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setColocationDeploymentSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setColocationDeploymentPage(1);
      setColocationDeploymentDetailPageSize(10);
    }
  };

  // 非混部表格操作
  const handleDeploymentDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setDeploymentPage(pagination.current || 1);
      setDeploymentDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setDeploymentDetailFilterObj(filter);
      setDeploymentPage(1);
      setDeploymentDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setDeploymentSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setDeploymentPage(1);
      setDeploymentDetailPageSize(10);
    }
  };

  // 改变期望选择
  const handleHopeDeploymentChange = async (checked, data) => {
    let newtablelist = JSON.parse(JSON.stringify(deploymentList));
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
    setDeploymentList(newtablelist);
    setOriginalList(newOrgtablelist);
    setSelectedRows(intersection);
  };

  // 点击出现改变混部属性的弹窗
  const changeColocationDeploymentState = (record, type) => {
    setPopOpen('');
    if (type === 'off') {
      setChoseState(intl.formatMessage({ id: 'workload.status.offline' }));
    } else {
      setChoseState(intl.formatMessage({ id: 'workload.status.oversold' }));
    }
    setColocationDeploymentState(true);
    setDeleteId(record.key);
  };

  const openDeploymentButton = () => {
    setSelectionType(true);
    setColShow(true);
    setIsDeploymentWithoutPaddingBottom('has-padding');
    getDeploymentList();
  };

  // 混部选择
  const onDeploymentSelectChange = (newSelectedRowKeys, newSelectedRows) => {
    setSelectedRowKeys(newSelectedRowKeys);
    setSelectedRows(newSelectedRows);
  };

  // 混部model的确定按钮
  const handleDeploymentStateModelConfirm = async (type) => {
    if (type === intl.formatMessage({ id: 'workload.status.offline' })) {
      // 离线代接口
      try {
        const res = await updateDeploymentsExpectStatusToOffline(deleteId);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
          getColocationDeploymentList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.modifyFailed' }));
      }
    } else {
      // 超卖代接口
      messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
      getColocationDeploymentList();
    }
    setColocationDeploymentState(false);
  };

  // 混部model的取消按钮
  const handleDeploymentStateModelCancel = async () => {
    setColocationDeploymentState(false);
  };

  // 重置按钮
  const handleDeploymentReset = async () => {
    setDeploymentLoading(true);
    await getNonColocationDeploymentsListData();
    getDeploymentResource();
    setDeploymentLoading(false);
  };

  // 重置按钮
  const handleColocationDeploymentReset = async () => {
    await getColocationDeploymentList();
    getColocationDeploymentResource();
  };

  // 条件重置的公共部分
  const restFilter = () => {
    deploymentForm.setFieldsValue({ deploymentName: '' });
    deploymentForm.setFieldsValue({ deploymentKeyValue: '' });
    setCurrentDeploymentNamespace(''); // 重置筛选条件
    setDeploymentSortObj({});
    setDeploymentDetailFilterObj({});
    setSelectedRowKeys([]);
    setSelectedRows([]);
    setDeploymentPage(1);
    setDeploymentDetailPageSize(10);
  };

  // table的取消 重置 确定
  const onDeploymentCancel = () => {
    setColShow(false);
    setIsDeploymentWithoutPaddingBottom('no-padding');
    setSelectionType(false);
    getDeploymentResource();
    restFilter();
  };

  const onDeploymentRest = () => {
    restFilter();
    setDeploymentList(JSON.parse(JSON.stringify(attTypeList)));
    setOriginalList(JSON.parse(JSON.stringify(attTypeList)));
  };

  const onDeploymentSure = async () => {
    if (selectedRows.length === 0) {
      messageApi.error(intl.formatMessage({ id: 'message.selectAtLeastOneWorkload' }));
    } else if (selectedRows.length > 10) {
      messageApi.error(intl.formatMessage({ id: 'message.maxTenWorkloads' }));
    } else {
      // 代接口 selectedRowKeys, selectedRows
      try {
        const res = await addColocationDeployments(selectedRows);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.addColocationSuccess' }, { type: 'Deployment' }));
          setColShow(false);
          setIsDeploymentWithoutPaddingBottom('no-padding');
          deploymentForm.resetFields();
          setSelectedRowKeys([]);
          setSelectedRows([]);

          getColocationDeploymentResource();
          setCurrentColocationDeploymentNamespace(''); // 重置筛选条件
          setColocationDeploymentSortObj({});
          setColocationDeploymentDetailFilterObj({});
          setColocationDeploymentPage(1);

          getDeploymentResource();
          setCurrentDeploymentNamespace(''); // 重置筛选条件
          setDeploymentSortObj({});
          setDeploymentDetailFilterObj({});
          setDeploymentPage(1);

          getColocationDeploymentList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.addColocationFailed' }, { type: 'Deployment' }));
      }
    }
  };

  useEffect(() => {
    getColocationDeploymentResource();
  }, [colocationDeploymentDetailFilterObj, colocationDeploymentSortObj]);

  useEffect(() => {
    getDeploymentResource();
  }, [deploymentDetailFilterObj, deploymentSortObj]);

  useEffect(() => {
    getColocationDeploymentList();
  }, [getColocationDeploymentList]);

  return (
    <div className="workload_content">
      <div className="workload_detail_content">
        <div className={`${isShow}`} style={{ marginTop: '20px' }}>
          <ToolTipComponent>
            <div
              className="deploymentDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'message.qosTip' })}</div>
              <CloseOutlined onClick={handleDeploymentShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className={`${isShow}workload_detail_content_box`}>
          <div className="workload_detail_form">
            <Form className="workload-searchForm form_padding_bottom" form={deploymentForm}>
              {colShow ? (
                <div className="workload-searchForm-box">
                  <Form.Item name="deploymentName" className="workload-search-input" style={{ marginRight: '20px' }}>
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                      onSearch={() => getDeploymentResourceTableByName()}
                      autoComplete="off"
                      maxLength={53}
                    />
                  </Form.Item>
                  <Form.Item name="deploymentKeyValue" className="workload-search-input">
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.byLabel' })}
                      onSearch={() => getDeploymentResourceTableByLabel()}
                      autoComplete="off"
                    />
                  </Form.Item>
                </div>
              ) : (
                <Form.Item
                  name="colocationDeploymentName"
                  className="workload-search-input"
                  style={{ marginRight: '20px' }}
                >
                  <Input.Search
                    placeholder={intl.formatMessage({ id: 'search.workloadName' })}
                    onSearch={() => getColocationDeploymentResource()}
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
                      onClick={() => openDeploymentButton()}
                    >
                      {intl.formatMessage({ id: 'button.addColocation' }, { type: 'Deployment' })}
                    </Button>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleColocationDeploymentReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                ) : (
                  <Space>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleDeploymentReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                )}
              </Form.Item>
            </Form>
          </div>
          <div
            className={`workLoad-colocation-table-${isDeploymentWithoutPaddingBottom}  workload_detail_content_table`}
          >
            {contextHolder}
            {colShow ? (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={deploymentLoading}
                columns={deploymentColumns}
                dataSource={deploymentList}
                rowSelection={
                  selectionType
                    ? {
                        type: 'checkbox',
                        selectedRowKeys,
                        preserveSelectedRowKeys: true,
                        onChange: onDeploymentSelectChange,
                      }
                    : null
                }
                pagination={{
                  className: 'colocation-website-page',
                  current: deploymentPage,
                  pageSize: deploymentDetailPageSize,
                  total: deploymentDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleDeploymentDetailTableChange}
              ></Table>
            ) : (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={deploymentLoading}
                columns={colocationDeploymentColumns}
                dataSource={colocationDeploymentList}
                pagination={{
                  className: 'colocation-website-page',
                  current: colocationDeploymentPage,
                  pageSize: colocationDeploymentDetailPageSize,
                  total: colocationDeploymentDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleColocationDeploymentDetailTableChange}
              ></Table>
            )}
          </div>
        </div>
      </div>
      <ColocationStateModal
        title={intl.formatMessage({ id: 'message.changeToState' }, { state: choseState })}
        open={colocationDeploymentState}
        cancelFn={handleDeploymentStateModelCancel}
        confirmFn={() => handleDeploymentStateModelConfirm(choseState)}
        state={choseState}
      />
      <ColocationDeleteModal
        title={intl.formatMessage({ id: 'action.setNonColocationWorkload' })}
        open={deleteDeploymentModal}
        cancelFn={handleDeleteDeploymentCancel}
        content={[intl.formatMessage({ id: 'message.confirmSetNonColocation' }, { name: deleteDeploymentName })]}
        confirmFn={handleDeleteDeploymentConfirmBack}
      />
      {colShow ? (
        <div className="child_content_workload_footer">
          <div className="workload_btn_footer_text">
            <p>{intl.formatMessage({ id: 'message.selectedWorkloads' }, { count: selectedRows.length })}</p>
          </div>
          <div className="workload_btn_footer_button">
            <Button className="cancel_btn" onClick={onDeploymentCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={onDeploymentRest}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={onDeploymentSure}>
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
