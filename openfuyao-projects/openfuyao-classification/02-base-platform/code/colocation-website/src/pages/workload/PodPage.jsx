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
import { SyncOutlined, MoreOutlined, CloseOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { Form, Table, Space, Input, Button, Popover, Select, message, Tooltip } from 'antd';

import '../../styles/ColocationWorkloadManagementPage.less';
import { filterStatusName, reseverStatusName, compareObjKeys } from '../../utils/common';
import {
  filterRepeat,
  sortWorkLoadByNameAscend,
  sortWorkLoadByNameDescend,
  sortWorkLoadByNodeDescend,
  sortWorkLoadByNodeAscend,
  sortWorkLoadByNamespaceAscend,
  sortWorkLoadByNamespaceDescend,
  sortWorkLoadByStatusAscend,
  sortWorkLoadByStatusDescend,
  sortWorkLoadByIpAscend,
  sortWorkLoadByIpDescend,
} from '../../tools/utils';
import ColocationStateModal from '../../components/ColocationStateModal';
import ColocationDeleteModal from '../../components/ColocationDeleteModal';
import { getWorkloadStatusOptions, ResponseCode } from '../../common/constants';
import {
  getColocationPodsListData,
  getNonColocationPodsListData,
  addColocationPods,
  updatePodsExpectStatusToOnline,
  updatePodsExpectStatusToOffline,
  deleteColocationPodsById,
} from '../../api/colocationApi';
import ToolTipComponent from '@/components/ToolTipComponent';

export default function PodPage() {
  const intl = useIntl();
  const [podForm] = Form.useForm();
  const [popOpen, setPopOpen] = useState('');
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [colShow, setColShow] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  const [isShow, setIsShow] = useState('');
  // table部分

  // 非混部table
  const [podSortObj, setPodSortObj] = useState({}); // 排序
  const [podDetailFilterObj, setPodDetailFilterObj] = useState({}); // 筛选
  const [podDetailTotal, setPodDetailTotal] = useState(0);
  const [podPage, setPodPage] = useState(1);
  const [podDetailPageSize, setPodDetailPageSize] = useState(10);
  const [originalList, setOriginalList] = useState([]); // 非混部原始数据
  const [podLoading, setPodLoading] = useState(false);
  const [podList, setPodList] = useState([]); // 非混部pod table数据集
  const [selectionType, setSelectionType] = useState(false);
  const [filterPodNamespace, setFilterPodNamespace] = useState([]); // namespace赋值筛选项
  const [currentPodNamespace, setCurrentPodNamespace] = useState('');
  const [attTypeList, setAttTypeList] = useState([]); // 超级原始table数据集
  // 混部table
  const [colocationPodSortObj, setColocationPodSortObj] = useState({}); // 排序
  const [colocationPodDetailFilterObj, setColocationPodDetailFilterObj] = useState({}); // 筛选
  const [colocationPodDetailTotal, setColocationPodDetailTotal] = useState(0);
  const [colocationPodPage, setColocationPodPage] = useState(1);
  const [colocationPodDetailPageSize, setColocationPodDetailPageSize] = useState(10);
  const [colocationOriginalList, setColocationOriginalList] = useState([]); // 混部原始数据
  const [colocationPodList, setColocationPodList] = useState([]); // 混部pod table数据集
  const [colocationFilterPodNamespace, setColocationFilterPodNamespace] = useState([]); // namespace赋值筛选项
  const [currentColocationPodNamespace, setCurrentColocationPodNamespace] = useState('');
  const [colocationFilterPodStatus, setColocationFilterPodStatus] = useState([
    { text: intl.formatMessage({ id: 'workload.status.online' }), value: '在线' },
    { text: intl.formatMessage({ id: 'workload.status.offline' }), value: '离线' },
    { text: intl.formatMessage({ id: 'workload.status.oversold' }), value: '超卖' },
  ]); // 混部属性赋值筛选项
  const [currentColocationPodStatus, setCurrentColocationPodStatus] = useState('');
  const [isPodWithoutPaddingBottom, setIsPodWithoutPaddingBottom] = useState('no-padding');

  // 混部状态弹窗
  const [colocationState, setColocationState] = useState(false);
  const [choseState, setChoseState] = useState(intl.formatMessage({ id: 'workload.status.online' }));
  // 删除提醒弹窗
  const [deleteModal, setDeleteModal] = useState(false);
  const [deleteName, setDeleteName] = useState('');
  const [deleteId, setDeleteId] = useState('');

  const handlePodShowChange = () => {
    setIsShow('cant_show');
  };

  // 混部节点移除
  const handleDeleteNode = async (record) => {
    setPopOpen('');
    setDeleteModal(true);
    setDeleteName(record.name);
    setDeleteId(record.key);
  };

  // 设置非混部的确定按钮
  const handleDeleteConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationPodsById(deleteId);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'message.nonColocationSetSuccess' }));
        getColocationPodList();
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.nonColocationSetFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setDeleteModal(false);
  };

  // 设置非混部的取消按钮
  const handleDeleteCancel = () => {
    setDeleteModal(false);
  };

  // 混部pod列表项
  const colocationPodColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.podName' }),
      key: 'colocationPodName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: colocationPodSortObj.key === 'colocationPodName' ? colocationPodSortObj.order : '',
      render: (_, record) => (
        <Space>
          {record.hasController === true ? (
            <div style={{ display: 'flex' }}>
              <div style={{ marginRight: '8px' }}>{record.name}</div>
              <Tooltip title={intl.formatMessage({ id: 'message.higherControllerTip' })}>
                <QuestionCircleOutlined style={{ color: '#89939B' }} />
              </Tooltip>
            </div>
          ) : (
            <div>{record.name}</div>
          )}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'colocationPodNamespace',
      filters: colocationFilterPodNamespace,
      filteredValue: currentColocationPodNamespace ? [currentColocationPodNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: colocationPodSortObj.key === 'colocationPodNamespace' ? colocationPodSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.colocationNode' }),
      key: 'colocation_pod_belong',
      sorter: true,
      sortOrder: colocationPodSortObj.key === 'colocation_pod_belong' ? colocationPodSortObj.order : '',
      render: (_, record) => <>{record.hostIP ? record.hostIP : '--'}</>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.podIP' }),
      key: 'colocation_pod_ip',
      sorter: true,
      sortOrder: colocationPodSortObj.key === 'colocation_pod_ip' ? colocationPodSortObj.order : '',
      render: (_, record) => (
        <>
          {record.podIPs
            ? record.podIPs.map((item) => {
                return (
                  <div className="table_tuning_box">
                    <p>{item}</p>
                  </div>
                );
              })
            : '--'}
        </>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.businessAttr' }),
      key: 'colocation_pod_status',
      filters: colocationFilterPodStatus,
      filteredValue: currentColocationPodStatus ? [currentColocationPodStatus] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: colocationPodSortObj.key === 'colocation_pod_status' ? colocationPodSortObj.order : '',
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
                    <Button type="link" onClick={() => changeColocationState(record, 'off')}>
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
                    handleDeleteNode(record);
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
            {record.colocationType === 'oversold' || record.hasController === true ? (
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

  // 非混部pod列表项
  const podColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.podName' }),
      key: 'podName',
      dataIndex: 'name',
      sorter: true,
      sortOrder: podSortObj.key === 'podName' ? podSortObj.order : '',
      render: (_, record) => (
        <Space>
          {record.hasController === true ? (
            <div style={{ display: 'flex' }}>
              <div style={{ marginRight: '8px' }}>{record.name}</div>
              <Tooltip title={intl.formatMessage({ id: 'message.higherControllerTip' })}>
                <QuestionCircleOutlined style={{ color: '#89939B' }} />
              </Tooltip>
            </div>
          ) : (
            <div>{record.name}</div>
          )}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.namespace' }),
      key: 'podNamespace',
      filters: filterPodNamespace,
      filteredValue: currentPodNamespace ? [currentPodNamespace] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: podSortObj.key === 'podNamespace' ? podSortObj.order : '',
      render: (_, record) => <Space>{record.namespace}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.colocationNode' }),
      key: 'pod_belong',
      sorter: true,
      sortOrder: podSortObj.key === 'pod_belong' ? podSortObj.order : '',
      render: (_, record) => <>{record.hostIP ? record.hostIP : '--'}</>,
    },
    {
      title: intl.formatMessage({ id: 'table.column.podIP' }),
      key: 'pod_ip',
      sorter: true,
      sortOrder: podSortObj.key === 'pod_ip' ? podSortObj.order : '',
      render: (_, record) => (
        <>
          {record.podIPs
            ? record.podIPs.map((item) => {
                return (
                  <div className="table_tuning_box">
                    <p>{item}</p>
                  </div>
                );
              })
            : '--'}
        </>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.expectedBusinessAttr' }),
      key: 'pod_status',
      width: 160,
      render: (_, record) => (
        <div>
          <Select
            className="table_hope_status"
            value={filterStatusName(record.colocationType, intl)}
            onChange={(e) => handlePodHopeChange(e, record)}
            options={getWorkloadStatusOptions(intl)}
            disabled={record.hasController === true}
          />
        </div>
      ),
    },
  ];

  // 获取非混部PodList
  const getPodList = useCallback(async () => {
    setPodLoading(true);
    try {
      const res = await getNonColocationPodsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          item.colocationType = 'online';
          return {
            ...item,
            key: item.uid,
          };
        });
        arr.sort((a, b) => {
          // 如果a的hasController为true而b的hasController为false，则a排在b前面
          if (a.hasController && !b.hasController) {
            return 1;
          }
          // 如果b的hasController为true而a的hasController为false，则b排在a前面
          if (!a.hasController && b.hasController) {
            return -1;
          }
          // 如果两者都为true或都为false，则保持原有顺序或者根据其他条件排序
          return 0;
        });
        setPodList(arr);
        setOriginalList(JSON.parse(JSON.stringify(arr)));
        setAttTypeList(JSON.parse(JSON.stringify(JSON.parse(JSON.stringify(arr)))));
        setPodDetailTotal(arr.length);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setFilterPodNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setPodList([]); // 数组为空
    }
    setPodLoading(false);
  }, []);

  // 获取混部PodList
  const getColocationPodList = useCallback(async () => {
    setPodLoading(true);
    try {
      const res = await getColocationPodsListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data.data.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        arr.sort((a, b) => {
          if (a.hasController && !b.hasController) {
            return 1;
          }
          if (!a.hasController && b.hasController) {
            return -1;
          }
          return 0;
        });
        setColocationPodList(arr);
        setColocationOriginalList(arr);
        setColocationPodDetailTotal(arr.length);
        // 赋值命名空间
        if (res.data.data) {
          let temporyNamespaceList = [];
          res.data.data.map((item) => {
            temporyNamespaceList.push({ text: item.namespace, value: item.namespace });
          });
          setColocationFilterPodNamespace([...filterRepeat(temporyNamespaceList)]);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setColocationPodList([]); // 数组为空
    }
    setPodLoading(false);
  }, []);

  // 混部具体表格操作
  const getColocationPodResource = () => {
    let filtertArr = [];
    if (podForm.getFieldsValue().colocationPodName) {
      filtertArr = colocationOriginalList.filter((item) =>
        item.name.toLowerCase().includes(podForm.getFieldsValue().colocationPodName.toLowerCase()),
      );
    } else {
      filtertArr = JSON.parse(JSON.stringify(colocationOriginalList));
    }
    if (colocationPodDetailFilterObj.colocationPodNamespace && colocationPodDetailFilterObj.colocation_pod_status) {
      // 获取筛选框
      let [temporyNamespace, ...reset] = colocationPodDetailFilterObj.colocationPodNamespace;
      let [temporyStatus, ...reset1] = colocationPodDetailFilterObj.colocation_pod_status;
      setCurrentColocationPodNamespace(temporyNamespace);
      setCurrentColocationPodStatus(temporyStatus);
      filtertArr = filtertArr.filter(
        (item) => item.namespace === temporyNamespace && filterStatusName(item.colocationType) === temporyStatus,
      );
    } else {
      if (colocationPodDetailFilterObj.colocationPodNamespace) {
        // 获取筛选框
        setCurrentColocationPodStatus('');
        let [temporyNamespace, ...reset] = colocationPodDetailFilterObj.colocationPodNamespace;
        setCurrentColocationPodNamespace(temporyNamespace);
        filtertArr = filtertArr.filter((item) => item.namespace === temporyNamespace);
      } else if (colocationPodDetailFilterObj.colocation_pod_status) {
        // 获取筛选框
        setCurrentColocationPodNamespace('');
        let [temporyStatus, ...reset1] = colocationPodDetailFilterObj.colocation_pod_status;
        setCurrentColocationPodStatus(temporyStatus);
        filtertArr = filtertArr.filter((item) => filterStatusName(item.colocationType) === temporyStatus);
      } else {
        setCurrentColocationPodNamespace('');
        setCurrentColocationPodStatus('');
      }
    }
    if (colocationPodSortObj.key === 'colocationPodName') {
      if (colocationPodSortObj.order) {
        colocationPodSortObj.order === 'ascend'
          ? sortWorkLoadByNameAscend('pod', filtertArr)
          : sortWorkLoadByNameDescend('pod', filtertArr);
      }
    } else if (colocationPodSortObj.key === 'colocationPodNamespace') {
      if (colocationPodSortObj.order) {
        colocationPodSortObj.order === 'ascend'
          ? sortWorkLoadByNamespaceAscend('pod', filtertArr)
          : sortWorkLoadByNamespaceDescend('pod', filtertArr);
      }
    } else if (colocationPodSortObj.key === 'colocation_pod_belong') {
      if (colocationPodSortObj.order) {
        colocationPodSortObj.order === 'ascend'
          ? sortWorkLoadByNodeAscend('pod', filtertArr)
          : sortWorkLoadByNodeDescend('pod', filtertArr);
      }
    } else if (colocationPodSortObj.key === 'colocation_pod_ip') {
      if (colocationPodSortObj.order) {
        colocationPodSortObj.order === 'ascend'
          ? sortWorkLoadByIpAscend('pod', filtertArr)
          : sortWorkLoadByIpDescend('pod', filtertArr);
      }
    } else {
      if (colocationPodSortObj.order) {
        colocationPodSortObj.order === 'ascend'
          ? sortWorkLoadByStatusAscend('pod', filtertArr)
          : sortWorkLoadByStatusDescend('pod', filtertArr);
      }
    }
    setColocationPodDetailTotal(filtertArr.length);
    setColocationPodList([...filtertArr]);
  };

  // 非混部具体表格操作
  const getPodResource = () => {
    let filtertArr = originalList;
    if (podForm.getFieldsValue().podName) {
      filtertArr = originalList.filter((item) =>
        item.name.toLowerCase().includes(podForm.getFieldsValue().podName.toLowerCase()),
      );
    } else if (podForm.getFieldsValue().podKeyValue) {
      filtertArr = originalList.filter((item) => {
        return compareObjKeys(item, podForm.getFieldsValue().podKeyValue);
      });
    } else {
      filtertArr = JSON.parse(JSON.stringify(originalList));
    }
    if (!podDetailFilterObj?.podNamespace) {
      setCurrentPodNamespace('');
    } else {
      // 获取筛选框
      let [temporyStatus, ...reset] = podDetailFilterObj.podNamespace;
      setCurrentPodNamespace(temporyStatus);
      filtertArr = filtertArr.filter((item) => item.namespace === temporyStatus);
    }
    if (podSortObj?.key === 'podName') {
      if (podSortObj.order) {
        podSortObj.order === 'ascend'
          ? sortWorkLoadByNameAscend('pod', filtertArr)
          : sortWorkLoadByNameDescend('pod', filtertArr);
      }
    } else if (podSortObj?.key === 'podNamespace') {
      if (podSortObj.order) {
        podSortObj.order === 'ascend'
          ? sortWorkLoadByNamespaceAscend('pod', filtertArr)
          : sortWorkLoadByNamespaceDescend('pod', filtertArr);
      }
    } else if (podSortObj?.key === 'pod_belong') {
      if (podSortObj.order) {
        podSortObj.order === 'ascend'
          ? sortWorkLoadByNodeAscend('pod', filtertArr)
          : sortWorkLoadByNodeDescend('pod', filtertArr);
      }
    } else {
      if (podSortObj.order) {
        podSortObj.order === 'ascend'
          ? sortWorkLoadByIpAscend('pod', filtertArr)
          : sortWorkLoadByIpDescend('pod', filtertArr);
      }
    }

    setPodDetailTotal(filtertArr.length);
    setPodList([...filtertArr]);
  };

  // 非混部table通过name查找
  const getPodResourceTableByName = () => {
    podForm.setFieldsValue({ podKeyValue: '' });
    getPodResource();
  };

  // 非混部table通过name查找
  const getPodResourceTableByLabel = () => {
    podForm.setFieldsValue({ podName: '' });
    getPodResource();
  };

  // 混部表格操作
  const handleColocationPodDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setColocationPodPage(pagination.current || 1);
      setColocationPodDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setColocationPodDetailFilterObj(filter);
      setColocationPodPage(1);
      setColocationPodDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setColocationPodSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setColocationPodPage(1);
      setColocationPodDetailPageSize(10);
    }
  };

  // 非混部表格操作
  const handlePodDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setPodPage(pagination.current || 1);
      setPodDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setPodDetailFilterObj(filter);
      setPodPage(1);
      setPodDetailPageSize(10);
    }
    if (extra.action === 'sort') {
      setPodSortObj({ key: _sorter.columnKey, order: _sorter.order });
      setPodPage(1);
      setPodDetailPageSize(10);
    }
  };

  // 改变期望选择
  const handlePodHopeChange = (checked, data) => {
    let newtablelist = JSON.parse(JSON.stringify(podList));
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
    setPodList(newtablelist);
    setOriginalList(newOrgtablelist);
    setSelectedRows(intersection);
  };

  // 点击出现改变混部属性的弹窗
  const changeColocationState = (record, type) => {
    setPopOpen('');
    if (type === 'off') {
      setChoseState(intl.formatMessage({ id: 'workload.status.offline' }));
    } else {
      setChoseState(intl.formatMessage({ id: 'workload.status.oversold' }));
    }
    setColocationState(true);
    setDeleteId(record.key);
  };

  const openButton = () => {
    setSelectionType(true);
    setColShow(true);
    setIsPodWithoutPaddingBottom('has-padding');
    getPodList();
  };

  // 混部选择
  const onSelectChange = (newSelectedRowKeys, newSelectedRows) => {
    setSelectedRowKeys(newSelectedRowKeys);
    setSelectedRows(newSelectedRows);
  };

  // 混部model的确定按钮
  const handleStateModelConfirm = async (type) => {
    if (type === intl.formatMessage({ id: 'workload.status.offline' })) {
      // 离线代接口
      try {
        const res = await updatePodsExpectStatusToOffline(deleteId);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
          getColocationPodList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.modifyFailed' }));
      }
    } else {
      // 超卖代接口
      messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }));
      getColocationPodList();
    }
    setColocationState(false);
  };

  // 混部model的取消按钮
  const handleStateModelCancel = async () => {
    setColocationState(false);
  };

  // 重置按钮
  const handlePodReset = async () => {
    setPodLoading(true);
    await getNonColocationPodsListData();
    getPodResource();
    setPodLoading(false);
  };

  // 重置按钮
  const handleColocationPodReset = async () => {
    await getColocationPodList();
    getColocationPodResource();
  };

  // 条件重置的公共部分
  const restFilter = () => {
    podForm.setFieldsValue({ podName: '' });
    podForm.setFieldsValue({ podKeyValue: '' });
    setCurrentPodNamespace(''); // 重置筛选条件
    setPodSortObj({});
    setPodDetailFilterObj({});
    setSelectedRowKeys([]);
    setSelectedRows([]);
    setPodPage(1);
    setPodDetailPageSize(10);
  };

  // table的取消 重置 确定
  const onCancel = () => {
    setColShow(false);
    setIsPodWithoutPaddingBottom('no-padding');
    setSelectionType(false);
    getPodResource();
    restFilter();
  };

  const onRest = () => {
    restFilter();
    setPodList(JSON.parse(JSON.stringify(attTypeList)));
    setOriginalList(JSON.parse(JSON.stringify(attTypeList)));
  };

  const onSure = async () => {
    if (selectedRows.length === 0) {
      messageApi.error(intl.formatMessage({ id: 'message.selectAtLeastOneWorkload' }));
    } else if (selectedRows.length > 10) {
      messageApi.error(intl.formatMessage({ id: 'message.maxTenWorkloads' }));
    } else {
      try {
        const res = await addColocationPods(selectedRows);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'message.addColocationSuccess' }, { type: 'Pod' }));
          setColShow(false);
          setIsPodWithoutPaddingBottom('no-padding');
          podForm.resetFields();
          setSelectedRowKeys([]);
          setSelectedRows([]);

          getColocationPodResource();
          setCurrentColocationPodNamespace(''); // 重置筛选条件
          setColocationPodSortObj({});
          setColocationPodDetailFilterObj({});
          setColocationPodPage(1);

          getPodResource();
          setCurrentPodNamespace(''); // 重置筛选条件
          setPodSortObj({});
          setPodDetailFilterObj({});
          setPodPage(1);

          getColocationPodList();
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'message.addColocationFailed' }, { type: 'Pod' }));
      }
    }
  };

  const rowClassName = (record) => {
    if (record.hasController === true) {
      return 'disabled-node-row';
    }
    return '';
  };

  useEffect(() => {
    getColocationPodResource();
  }, [colocationPodDetailFilterObj, colocationPodSortObj]);

  useEffect(() => {
    getPodResource();
  }, [podDetailFilterObj, podSortObj]);

  useEffect(() => {
    getColocationPodList();
  }, [getColocationPodList]);

  return (
    <div className="workload_content">
      <div className="workload_detail_content">
        <div className={`${isShow}`} style={{ marginTop: '20px' }}>
          <ToolTipComponent>
            <div
              className="podDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'message.qosTip' })}</div>
              <CloseOutlined onClick={handlePodShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className={`${isShow}workload_detail_content_box`}>
          <div className="workload_detail_form">
            <Form className="workload-searchForm form_padding_bottom" form={podForm}>
              {colShow ? (
                <div className="workload-searchForm-box">
                  <Form.Item name="podName" className="workload-search-input" style={{ marginRight: '20px' }}>
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.podName' })}
                      onSearch={() => getPodResourceTableByName()}
                      autoComplete="off"
                      maxLength={53}
                    />
                  </Form.Item>
                  <Form.Item name="podKeyValue" className="workload-search-input">
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.byLabel' })}
                      onSearch={() => getPodResourceTableByLabel()}
                      autoComplete="off"
                    />
                  </Form.Item>
                </div>
              ) : (
                <Form.Item name="colocationPodName" className="workload-search-input" style={{ marginRight: '20px' }}>
                  <Input.Search
                    placeholder={intl.formatMessage({ id: 'search.podName' })}
                    onSearch={() => getColocationPodResource()}
                    autoComplete="off"
                    maxLength={53}
                  />
                </Form.Item>
              )}
              <Form.Item>
                {!colShow ? (
                  <Space>
                    <Button className="workload_detail_form_button" onClick={() => openButton()}>
                      {intl.formatMessage({ id: 'button.addColocation' }, { type: 'Pod' })}
                    </Button>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleColocationPodReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                ) : (
                  <Space>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handlePodReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                )}
              </Form.Item>
            </Form>
          </div>
          <div className={`workLoad-colocation-table-${isPodWithoutPaddingBottom}  workload_detail_content_table`}>
            {contextHolder}
            {colShow ? (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={podLoading}
                columns={podColumns}
                dataSource={podList}
                rowClassName={rowClassName}
                rowSelection={
                  selectionType
                    ? {
                        type: 'checkbox',
                        selectedRowKeys,
                        preserveSelectedRowKeys: true,
                        onChange: onSelectChange,
                        getCheckboxProps: (record) => ({
                          name: record.name,
                          disabled: record.hasController === true,
                        }),
                      }
                    : null
                }
                pagination={{
                  className: 'colocation-website-page',
                  current: podPage,
                  pageSize: podDetailPageSize,
                  total: podDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handlePodDetailTableChange}
              ></Table>
            ) : (
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={podLoading}
                columns={colocationPodColumns}
                dataSource={colocationPodList}
                rowClassName={rowClassName}
                pagination={{
                  className: 'colocation-website-page',
                  current: colocationPodPage,
                  pageSize: colocationPodDetailPageSize,
                  total: colocationPodDetailTotal || 0,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleColocationPodDetailTableChange}
              ></Table>
            )}
          </div>
        </div>
      </div>
      <ColocationStateModal
        title={intl.formatMessage({ id: 'message.changeToState' }, { state: choseState })}
        open={colocationState}
        cancelFn={handleStateModelCancel}
        confirmFn={() => handleStateModelConfirm(choseState)}
        state={choseState}
      />
      <ColocationDeleteModal
        title={intl.formatMessage({ id: 'action.setNonColocationWorkload' })}
        open={deleteModal}
        cancelFn={handleDeleteCancel}
        content={[intl.formatMessage({ id: 'message.confirmSetNonColocation' }, { name: deleteName })]}
        confirmFn={handleDeleteConfirmBack}
      />
      {colShow ? (
        <div className="child_content_workload_footer">
          <div className="workload_btn_footer_text">
            <p>{intl.formatMessage({ id: 'message.selectedWorkloads' }, { count: selectedRows.length })}</p>
          </div>
          <div className="workload_btn_footer_button">
            <Button className="cancel_btn" onClick={onCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={onRest}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={onSure}>
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
