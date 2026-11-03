/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { clusterScaleDown, clusterScaleUp, validateNodes } from '@/api/clusterApi';
import { DEFAULT_CURRENT_PAGE, ResponseCode, nodeStatus } from '@/common/constants';
import { useState, useCallback, useEffect, useRef, useMemo } from 'openinula';
import { PlusOutlined, CheckCircleFilled, ExclamationCircleFilled, DeleteOutlined, LoadingOutlined } from '@ant-design/icons';
import AddNode from '@/pages/clusterManage/AddNode';
import useLocalStorage from '@/hooks/useLocalStorage';
import { Button, Form, message, ConfigProvider, Table, Input, Space } from 'antd';
import CreateClusterLog from '@/pages/clusterManage/CreateClusterLog';
import { forbiddenMsg, sorterFirstAlphabet, disabledHandle, extractIPv4, nodeStatusValueToText } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useIntl } from '../../../i18n';

export default function NodeDetail({ clusterDetail, nodeLoading }) {
  const intl = useIntl();
  const [clusterName, setClusterName] = useState('');
  const [clusterStatus, setClusterStatus] = useState('');
  const [clusterNodes, setClusterNodes] = useState([]);

  const [clusterIsHA, setClusterIsHA] = useState(false);
  const [clusterBalanceIp, setClusterBalanceIp] = useState('');

  const [nodeForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();

  const [nodeListFilerted, setNodeListFiltered] = useState([]); // 数据集
  const [nodeDelModalOpen, setNodeDelModalOpen] = useState(false); // 删除对话框展示
  const [isSpecialCheck, setIsSpecialCheck] = useState(false);
  const [isNodeDelCheck, setIsNodeDelCheck] = useState(false); // 是否选中

  const [filterNodeValue, setFilterNodeValue] = useState();
  const [selectedNodes, setSelectedNodes] = useState([]);

  const [nodePage, setNodePage] = useState(DEFAULT_CURRENT_PAGE); // 页码
  const [nodePageSize, setNodePageSize] = useState(10); // 分页大小
  const currentPageRef = useRef(1); // 防止在异步回调中 unmount

  const [addNodeModalOpen, setAddNodeModalOpen] = useState(false);
  const [logNodeOpen, setLogNodeOpen] = useState(false);

  const [checkLoading, setCheckLoading] = useState(false);
  const [isCheckPass, setIsCheckPass] = useState(false);
  const [errorIP, setErrorIP] = useState('');
  const [isDisabled, setIsDisabled] = useState(disabledHandle(clusterStatus));

  useEffect(() => {
    setClusterName(clusterDetail?.clusterName || '');
    setClusterStatus(clusterDetail?.clusterStatus || '');
    setClusterNodes(clusterDetail?.nodes || []);
    setClusterIsHA(clusterDetail?.clusterIsHA || false);
    setClusterBalanceIp(clusterDetail?.Host || '');
  }, [clusterDetail]);

  const handleLogExit = () => {
    setLogNodeOpen(false);
    getNodeList();
  };

  // 检索
  const handleSearchNode = (totalData = clusterNodes, isChange = true) => {
    const nodeFormName = nodeForm.getFieldValue('node_name');
    let temporyList = totalData;
    if (nodeFormName) {
      temporyList = temporyList.filter((item) => item.hostname.toLowerCase().includes(nodeFormName.toLowerCase()));
    }
    setNodeListFiltered(temporyList);
    if (isChange) {
      currentPageRef.current = DEFAULT_CURRENT_PAGE;
      setNodePage(DEFAULT_CURRENT_PAGE);
    }
  };

  const updateDisplayData = () => {
    const maxPage = Math.ceil(nodeListFilerted.length / nodePageSize) || 1;
    if (currentPageRef.current > maxPage) {
      currentPageRef.current = maxPage;
    }
    const start = (currentPageRef.current - 1) * nodePageSize;
    const end = start + nodePageSize;
    return nodeListFilerted.slice(start, end);
  };

  const handleDeleleCancel = () => {
    setNodeDelModalOpen(false);
  };

  const handleDeleteNodeConfirm = async (temporyNodeName = selectedNodes) => {
    const deleteNodeData = {
      nodes: temporyNodeName,
    };
    try {
      const res = await clusterScaleDown(clusterName, deleteNodeData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'node.detail.scaleDownInProgress' }));
        setIsNodeDelCheck(false);
        setNodeDelModalOpen(false);
        setSelectedNodes([]);
        getNodeList(true);
        setLogNodeOpen(true);
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error, intl);
      } else {
        messageApi.error(
          intl.formatMessage({ id: 'node.detail.scaleDownFailed' }, { error: error.response.data.message }),
        );
      }
    }
  };

  const addNodeNameMap = (role) =>
    role.includes('master')
      ? intl.formatMessage({ id: 'node.detail.masterNode' })
      : intl.formatMessage({ id: 'node.detail.workerNode' });

  const handleClusterCheckFn = (e) => {
    setIsNodeDelCheck(e.target.checked);
  };

  const handleSpecialCheckFn = (e) => {
    setIsSpecialCheck(e.target.checked);
  };

  const handleAddNodeOK = async (form) => {
    await form.validateFields();
    let nodeList = form.getFieldValue('node');
    // 判断节点名称和IP是否有重复，否则报错
    let nodeNameList = nodeList.map((item) => item.hostname);
    let nodeIPList = nodeList.map((item) => item.ip);
    let nodeNameSet = [...new Set(nodeNameList)];
    let nodeIPSet = [...new Set(nodeIPList)];
    if (nodeNameSet.length !== nodeNameList.length) {
      return messageApi.error(intl.formatMessage({ id: 'node.duplicateName' }));
    }
    if (nodeIPSet.length !== nodeIPList.length) {
      return messageApi.error(intl.formatMessage({ id: 'node.duplicateIp' }));
    }
    const scaleUpData = {
      nodes: nodeList.map((item) => ({
        hostname: item.hostname,
        ip: item.ip,
        port: String(item.port),
        role: ['node'],
        username: item.userName,
        password: item.password,
      })),
    };
    try {
      const res = await clusterScaleUp(clusterName, scaleUpData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'node.detail.scaleUpInProgress' }));
        setAddNodeModalOpen(false);
        setLogNodeOpen(true);
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error, intl);
      } else {
        messageApi.error(
          intl.formatMessage({ id: 'node.detail.scaleUpFailed' }, { error: error.response.data.message }),
        );
      }
    }
  };

  const cancelAddNode = (form) => {
    form.resetFields();
    setAddNodeModalOpen(false);
  };

  const birthDifferentStatusSvg = (status) => {
    const colorClass = nodeStatus[status] ? `${nodeStatus[status]}_color` : 'unknown_color';
    switch (status) {
      case 'Ready':
        return (
          <span className={colorClass}>
            <CheckCircleFilled />
          </span>
        );
      case 'Pending':
      case 'Provisioned':
      case 'Initializing':
      case 'BootStrapping':
      case 'Upgrading':
      case 'Deleting':
      case 'Managing':
        return (
          <span className={colorClass}>
            <LoadingOutlined />
          </span>
        );
      case 'NotReady':
      case 'Unknown':
      case 'Failed':
      case 'InitFailed':
      case 'BootStrapFailed':
      case 'UpgradeFailed':
      case 'DeleteFailed':
      case 'ManageFailed':
        return (
          <span className={colorClass}>
            <ExclamationCircleFilled />
          </span>
        );
      default:
        return (
          <span className="unknown_color">
            <ExclamationCircleFilled />
          </span>
        );
    }
  };

  const solveArchitecture = (cpu) => {
    return cpu.replace('amd', 'x86_');
  };

  const handleNodeValueChange = (changedValues, _allValues) => {
    const deleteList = changedValues.node.some((item) => item && Object.keys(item).length > 1); // 删除
    if (!deleteList) {
      setIsCheckPass(false);
    }
  };

  const nodeStatusOptions = useMemo(() => {
    return clusterNodes.map((item) => ({ text: nodeStatusValueToText(item.status, intl), value: item.status }));
  }, [clusterNodes]);

  // 列表项
  const nodeColumns = [
    {
      title: intl.formatMessage({ id: 'node.name' }),
      key: 'node_name',
      fixed: 'left',
      width: 210,
      sorter: (a, b) => sorterFirstAlphabet(a.hostname, b.hostname),
      render: (_, record) => record.hostname,
    },
    {
      title: intl.formatMessage({ id: 'node.detail.status' }),
      filters: nodeStatusOptions,
      filterMultiple: false,
      width: 200,
      filteredValue: filterNodeValue ? [filterNodeValue] : [],
      onFilter: (value, record) => record.status.toLowerCase() === value.toLowerCase(),
      sorter: (a, b) => sorterFirstAlphabet(a.status, b.status),
      key: 'podStatus',
      render: (_, record) =>
        record.status ? (
          <p className="resource_status" data-name={record.hostname}>
            {birthDifferentStatusSvg(record.status)}
            {nodeStatusValueToText(record.status, intl)}
          </p>
        ) : (
          <p>--</p>
        ),
    },
    {
      title: intl.formatMessage({ id: 'node.ip' }),
      width: 210,
      key: 'node_ip',
      sorter: (a, b) => sorterFirstAlphabet(a.ip, b.ip),
      render: (_, record) => record.ip,
    },
    {
      title: intl.formatMessage({ id: 'node.detail.type' }),
      width: 200,
      key: 'node_type',
      sorter: (a, b) => sorterFirstAlphabet(addNodeNameMap(a.role), addNodeNameMap(b.role)),
      render: (_, record) => (record.role ? addNodeNameMap(record.role) : '--'),
    },
    {
      title: intl.formatMessage({ id: 'node.detail.cpuArch' }),
      width: 200,
      key: 'node_cpu',
      sorter: (a, b) => sorterFirstAlphabet(a.architecture, b.architecture),
      render: (_, record) => (record.architecture ? solveArchitecture(record.architecture) : '--'),
    },
    {
      title: intl.formatMessage({ id: 'node.detail.osVersion' }),
      width: 200,
      key: 'os_version',
      sorter: (a, b) => sorterFirstAlphabet(a.os, b.os),
      render: (_, record) => record.os || '--',
    },
    {
      title: intl.formatMessage({ id: 'node.detail.cpuCount' }),
      width: 200,
      key: 'cpu_num',
      sorter: (a, b) => sorterFirstAlphabet(a.cpu, b.cpu),
      render: (_, record) => record.cpu || '--',
    },
    {
      title: intl.formatMessage({ id: 'node.detail.memory' }),
      width: 200,
      key: 'node_memory',
      sorter: (a, b) => sorterFirstAlphabet(a.memory, b.memory),
      render: (_, record) => record.memory || '--',
    },
    {
      title: intl.formatMessage({ id: 'node.action' }),
      width: 96,
      key: 'handle',
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <DeleteOutlined
            className={`common_antd_icon ${record.role?.includes('master') || isDisabled ? 'disabled_color' : 'primary_color'}`}
            style={record.role?.includes('master') || isDisabled ? { cursor: 'not-allowed' } : { cursor: 'pointer' }}
            onClick={(e) => handleDeleteSingleNode(e, record)}
          />
        </Space>
      ),
    },
  ];

  const handleDeleteSingleNode = (e, record) => {
    if (record.role?.includes('master') || isDisabled) {
      e.preventDefault();
    } else {
      setNodeDelModalOpen(true);
      setSelectedNodes([record.hostname]);
      setIsSpecialCheck(false);
      setIsNodeDelCheck(false);
    }
  };

  const handleCreateNodeCheck = async (form) => {
    setIsCheckPass(false);
    await form.validateFields();
    const formObj = form.getFieldsValue();
    try {
      const nodeList = formObj.node;
      const passNodeList = [];
      nodeList?.map((item) => {
        passNodeList.push({
          hostname: item.hostname,
          ip: item.ip,
          port: String(item.port),
          username: item.userName,
          password: item.password,
        });
      });
      const jsonData = {
        namespace: clusterName,
        nodes: [...passNodeList],
        balanceIp: clusterIsHA ? clusterBalanceIp : '',
      };
      setCheckLoading(true);
      const res = await validateNodes(jsonData);
      if (res.data.code === ResponseCode.OK) {
        setCheckLoading(false);
        setIsCheckPass(true);
        messageApi.success(intl.formatMessage({ id: 'node.validateSuccess' }));
        setErrorIP('');
      }
    } catch (e) {
      setCheckLoading(false);
      messageApi.error(intl.formatMessage({ id: 'node.validateFailed' }, { error: e.response.data.message }));
      const [ip, ...resets] = extractIPv4(e.response.data.message);
      setErrorIP(ip);
    }
  };

  const getNodeList = useCallback(
    async (isChange = false) => {
      handleSearchNode(clusterNodes, isChange); // 先搜索
      // 过滤selectedNodeKeys，只保留存在于validIPs中的hostname
      let validNames = clusterNodes.map((item) => item.hostname);
      setSelectedNodes((prevKeys) => prevKeys.filter((key) => validNames.includes(key)));
    },
    [clusterNodes],
  );

  useEffect(() => {
    getNodeList();
  }, [getNodeList]);

  const handleNodeTableChange = useCallback((pagination, filter, _sorter, extra) => {
    if (extra.action === 'filter') {
      setFilterNodeValue(filter.podStatus);
    }
    if (extra.action === 'paginate') {
      currentPageRef.current = pagination.current;
      setNodePage(pagination.current);
      setNodePageSize(pagination.pageSize);
    }
  }, []);

  const rowSelection = {
    onChange: (selectedRowKeys, selectedRows) => {
      setSelectedNodes(selectedRowKeys);
    },
    getCheckboxProps: (record) => ({
      disabled: record.role?.includes('master') || isDisabled,
      name: record.hostname,
    }),
  };

  const deleteNodeHandle = () => {
    setNodeDelModalOpen(true);
    setIsNodeDelCheck(false);
    setIsSpecialCheck(false);
  };

  const handleAddNode = () => {
    setAddNodeModalOpen(true);
  };

  useEffect(() => {
    setIsDisabled(disabledHandle(clusterStatus));
  }, [clusterStatus]);

  const ifDeletingGuideNodes = () =>
    selectedNodes.some((item) => {
      const targetNode = clusterNodes.find((node) => node.hostname === item);
      return targetNode.ip === window.location.hostname;
    });

  return (
    <div className="tab_container container_margin_box">
      <ToastMsg contextHolder={contextHolder} />
      <Form className="pod_searchForm form_padding_bottom" form={nodeForm}>
        <Form.Item name="node_name" className="pod_search_input">
          <Input.Search
            placeholder={intl.formatMessage({ id: 'node.searchPlaceholder' })}
            onSearch={() => handleSearchNode()}
            autoComplete="off"
          />
        </Form.Item>
        <Form.Item>
          <Space>
            <Button
              className={!isDisabled && 'cluster_primary_btn'}
              disabled={isDisabled}
              icon={<PlusOutlined />}
              onClick={handleAddNode}
            >
              {intl.formatMessage({ id: 'node.addNode' })}
            </Button>
            <Button disabled={!selectedNodes.length || isDisabled} className="cancel_btn" onClick={deleteNodeHandle}>
              {intl.formatMessage({ id: 'node.delete' })}
            </Button>
          </Space>
        </Form.Item>
      </Form>
      <div className="tab_table_flex">
        <ConfigProvider>
          <Table
            rowKey="hostname"
            rowSelection={{ ...rowSelection }}
            className="table_padding"
            loading={nodeLoading}
            columns={nodeColumns}
            dataSource={updateDisplayData()}
            onChange={handleNodeTableChange}
            pagination={{
              className: 'page',
              current: nodePage,
              pageSize: nodePageSize,
              total: nodeListFilerted.length,
              showTotal: (total) => intl.formatMessage({ id: 'node.totalItems' }, { total }),
              showSizeChanger: true,
              showQuickJumper: true,
              pageSizeOptions: [10, 20, 50],
              onChange: (page) => setNodePage(page),
            }}
            scroll={{ x: 1572 }}
          />
        </ConfigProvider>
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'node.deleteNode' })}
        open={nodeDelModalOpen}
        cancelFn={handleDeleleCancel}
        content={[
          intl.formatMessage({ id: 'node.deleteWarning' }),
          intl.formatMessage({ id: 'node.deleteConfirm' }, { nodes: selectedNodes.join(', ') }),
        ]}
        checkText={intl.formatMessage({ id: 'node.deleteAcknowledge' })}
        specialText={intl.formatMessage({ id: 'node.deleteSpecialWarning' })}
        showSpecialCheck={ifDeletingGuideNodes()}
        isSpecialCheck={isSpecialCheck}
        specialCheckFn={handleSpecialCheckFn}
        isCheck={isNodeDelCheck}
        showCheck={true}
        checkFn={handleClusterCheckFn}
        confirmFn={() => handleDeleteNodeConfirm()}
      />
      <AddNode
        open={addNodeModalOpen}
        okFn={handleAddNodeOK}
        cancelFn={cancelAddNode}
        handleCheck={handleCreateNodeCheck}
        checkLoading={checkLoading}
        errorIP={errorIP}
        handleNodeValueChange={handleNodeValueChange}
        isCheckPass={isCheckPass}
      />
      {logNodeOpen && (
        <CreateClusterLog clusterName={clusterName} open={logNodeOpen} okFn={handleLogExit} cancelFn={handleLogExit} />
      )}
    </div>
  );
}
