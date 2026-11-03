/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Button, Form, Space, Input, Table, ConfigProvider, Popover, message, Modal } from 'antd';
import {
  PlusOutlined,
  MoreOutlined,
  LoadingOutlined,
  CheckCircleFilled,
  ExclamationCircleFilled,
} from '@ant-design/icons';
import { useCallback, useEffect, useState, useRef, useMemo } from 'openinula';
import { DEFAULT_CURRENT_PAGE, ResponseCode, clusterStatus } from '@/common/constants';
import { listcluster, deleteCluster, upgradeCluster, autoUpgrade, getCluster, createPatch } from '@/api/clusterApi';
import { containerRouterPrefix } from '@/common/constants.js';
import { Link, useHistory } from 'inula-router';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import {
  sorterFirstAlphabet,
  forbiddenMsg,
  requestResource,
  disabledHandle,
  isAdminCluster,
  clusterStatusValueToText,
} from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import '@/styles/pages/podDetail.less';
import UpgradeClusterModal from '@/components/UpgradeClusterModal';
import CreateClusterLog from '@/pages/clusterManage/CreateClusterLog';
import '@/styles/components/codeMirrorEditor.less';
import CodeMirror from '@uiw/react-codemirror';
import { yaml } from '@codemirror/lang-yaml';
import UpgradePrepareModal from '@/components/UpgradePrepareModal';
import useButtonLoading from '@/hooks/useButtonLoading';
import ToastMsg from '@/components/ToastMsg';
import { useIntl } from '../../i18n';

let timeOutId = 0;
export default function ClusterManageList() {
  const intl = useIntl();
  const history = useHistory();
  const [clusterForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [clusterListFilterd, setClusterListFilterd] = useState([]); // 数据集
  const [clusterLoading, setClusterLoading] = useState(false); // 加载中
  const [clusterOpen, setClusterOpen] = useState(''); // 悬浮框是否展示
  const [clusterDelModalOpen, setClusterDelModalOpen] = useState(false); // 删除对话框展示
  const [clusterDelName, setClusterDelName] = useState(''); // 删除的名称
  const [isClusterDelCheck, setIsClusterDelCheck] = useState(false); // 是否选中

  const [clusterList, setClusterList] = useState([]); // 原始数据
  const [filterClusterValue, setFilterClusterValue] = useState([]);
  const [upgradeModalOpen, setUpgradeModalOpen] = useState(false);
  const [upgradeClusterName, setUpgradeClusterName] = useState('');
  const [clusterPage, setClusterPage] = useState(DEFAULT_CURRENT_PAGE);
  const [clusterPageSize, setClusterPageSize] = useState(10);
  const currentPageRef = useRef(1);

  const [logOpen, setLogOpen] = useState(false);
  const [logClusterName, setLogClusterName] = useState('');
  const [isShowSpecialCheck, setIsShowSpecialCheck] = useState(false);
  const [isSpecialCheck, setIsSpecialCheck] = useState(false);
  const [uploadPatchModalOpen, setUploadPatchModalOpen] = useState(false);
  const [prepareModalOpen, setPrepareModalopen] = useState(false);
  const [clusterPrepareName, setClusterPrepareName] = useState(''); // 删除的名称
  const [prepareConfirmLoading, setPrepareConfirmLoading] = useState(false);
  const [uploadLoading, setUploadLoading, uploadPending] = useButtonLoading();

  const judgeCpu = (isAmd, isArm) => {
    if (isAmd && isArm) {
      return 'x86/arm';
    } else if (isAmd) {
      return 'x86';
    } else if (isArm) {
      return 'arm';
    } else {
      return '-';
    }
  };
  const birthDifferentStatusSvg = (status) => {
    let svg;
    switch (status) {
      case 'Healthy':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <CheckCircleFilled />
          </span>
        );
        break;
      case 'Deleting':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'Deploying':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'Upgrading':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'ScalingWorkerNodesUp':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'ScalingWorkerNodesDown':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'ScalingMasterNodesUp':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'ScalingMasterNodesDown':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'DeleteFailed':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <ExclamationCircleFilled />
          </span>
        );
        break;
      case 'ScaleFailed':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <ExclamationCircleFilled />
          </span>
        );
        break;
      case 'UpgradeFailed':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <ExclamationCircleFilled />
          </span>
        );
        break;
      case 'DeployFailed':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <ExclamationCircleFilled />
          </span>
        );
        break;
      case 'Unhealthy':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <ExclamationCircleFilled />
          </span>
        );
        break;
      case 'Checking':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'DryRun':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <LoadingOutlined />
          </span>
        );
        break;
      case 'Paused':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <ExclamationCircleFilled />
          </span>
        );
        break;
      case 'PauseFailed':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <ExclamationCircleFilled />
          </span>
        );
        break;
      case 'DryRunFailed':
        svg = (
          <span className={`${clusterStatus[status]}_color`}>
            <ExclamationCircleFilled />
          </span>
        );
        break;
      default:
        svg = (
          <span className="unknown_color">
            <ExclamationCircleFilled />
          </span>
        );
        break;
    }
    return svg;
  };

  const handleLogExit = () => {
    setLogOpen(false);
    getClusterList();
  };

  const handleOpenPrepareModal = async (record) => {
    setClusterOpen('');
    setPrepareModalopen(true);
    setClusterPrepareName(record.name);
  };

  const handlePrepare = async (autoUpgradeForm) => {
    const values = await autoUpgradeForm.validateFields();
    setPrepareConfirmLoading(true);
    const requestBody = {
      patchDir: values.path,
      patchName: values.filename,
    };
    try {
      const res = await autoUpgrade(clusterPrepareName, requestBody);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'cluster.prepareSuccess' }));
        setPrepareModalopen(false);
        autoUpgradeForm.resetFields();
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'cluster.prepareFailed' }, { error: e.response.data.message }));
    }
    setPrepareConfirmLoading(false);
  };

  const handleCancelPrepare = (autoUpgradeForm) => {
    autoUpgradeForm.resetFields();
    setPrepareModalopen(false);
  };

  const handleUpgrade = async (clusterUpgradeForm) => {
    await clusterUpgradeForm.validateFields();
    const name = clusterUpgradeForm.getFieldValue('clusterName');
    const version = clusterUpgradeForm.getFieldValue('openFuyaoUpgradeVersion');
    const [upgradeError, upgradeRes] = await requestResource(upgradeCluster(name, { version }));
    if (upgradeError) {
      messageApi.error(intl.formatMessage({ id: 'cluster.upgradeFailed' }));
    } else if (upgradeRes) {
      if (upgradeRes.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'cluster.upgrading' }));
        setUpgradeModalOpen(false);
        getClusterList(true);
      }
    }
  };

  const handleIndexCancelUpgrade = (clusterUpgradeForm) => {
    clusterUpgradeForm.resetFields();
    setUpgradeModalOpen(false);
  };

  // 升级按钮
  const handleOpenUpgradeModal = (record) => {
    setClusterOpen('');
    setUpgradeModalOpen(true);
    setUpgradeClusterName(record.name);
  };
  // 删除按钮
  const handleDeleteClusterWorkload = async (record) => {
    // 隐藏气泡框
    const [error, res] = await requestResource(getCluster(record.name));
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'cluster.getInfoFailed' }));
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        const nodes = res.data.data.nodes;
        const result = nodes?.find((item) => window.location.hostname.includes(item.ip));
        if (result) {
          setIsShowSpecialCheck(true);
        } else {
          setIsShowSpecialCheck(false);
        }
        setClusterOpen('');
        setClusterDelModalOpen(true); // 打开弹窗
        setClusterDelName(record.name);
        setIsClusterDelCheck(false);
        setIsSpecialCheck(false);
      }
    }
  };

  const handleDeleletCluster = () => {
    setClusterDelModalOpen(false);
    setClusterDelName('');
  };

  const handleDeleletClusterConfirm = async () => {
    try {
      const res = await deleteCluster(clusterDelName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'cluster.deleting' }));
        setIsClusterDelCheck(false);
        setClusterDelModalOpen(false);
        setClusterDelName('');
        getClusterList(true);
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error, intl);
      } else {
        messageApi.error(
          intl.formatMessage({ id: 'cluster.deleteFailed' }, { error: error.response.data.message || '' }),
        );
      }
    }
  };

  const handleClusterCheckFn = (e) => {
    setIsClusterDelCheck(e.target.checked);
  };

  const handleSpecialCheckFn = (e) => {
    setIsSpecialCheck(e.target.checked);
  };

  const handleClusterOpenChange = (newOpen, record) => {
    if (newOpen) {
      setClusterOpen(record.name);
    } else {
      setClusterOpen('');
    }
  };

  const sortCluster = (a, b) => {
    let firstTime = a;
    let endTime = b;
    if (firstTime > endTime) {
      return 1;
    }
    if (firstTime < endTime) {
      return -1;
    }
    return 0;
  };

  const clusterStatusOptions = useMemo(() => {
    return Object.keys(clusterStatus).map((item) => ({ text: clusterStatusValueToText(item, intl), value: item }));
  }, [intl]);

  // 列表项
  const clusterColumns = [
    {
      title: intl.formatMessage({ id: 'cluster.col.name' }),
      key: 'clusterName',
      fixed: 'left',
      width: 240,
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <div style={{ display: 'flex', gap: '16px' }}>
          <Link
            to={{
              pathname: `/cluster/${record.name}`,
              state: { openFuyaoVersion: record.openFuyaoVersion },
            }}
          >
            {record.name}
          </Link>
          <p className="installer_cluster_left_role" style={{ display: isAdminCluster(record) ? 'block' : 'none' }}>
            {intl.formatMessage({ id: 'cluster.manageCluster' })}
          </p>
        </div>
      ),
    },
    {
      title: intl.formatMessage({ id: 'cluster.col.status' }),
      width: 200,
      filters: clusterStatusOptions,
      sorter: (a, b) => sorterFirstAlphabet(a.status, b.status),
      filterSearch: true,
      filterMultiple: true,
      filteredValue: filterClusterValue && filterClusterValue.length ? filterClusterValue : [],
      onFilter: (value, record) => record.status.toLowerCase() === value.toLowerCase(),
      key: 'clusterStatus',
      render: (_, record) => (
        <p
          className={`resource_status ${(record.status || 'Unhealthy').toLowerCase()}_circle`}
          data-status={record.status}
          data-name={record.name}
          style={{ cursor: 'pointer' }}
        >
          {birthDifferentStatusSvg(record.status)}
          <Button
            type="link"
            onClick={() => {
              setLogOpen(true);
              setLogClusterName(record.name);
            }}
          >
            {clusterStatusValueToText(record.status, intl)}
          </Button>
        </p>
      ),
    },
    {
      title: intl.formatMessage({ id: 'cluster.col.nodeCount' }),
      width: 200,
      key: 'node_count',
      sorter: (a, b) => sorterFirstAlphabet(a.nodeSum, b.nodeSum),
      render: (_, record) => record.nodeSum,
    },
    {
      title: intl.formatMessage({ id: 'cluster.col.openFuyaoVersion' }),
      width: 200,
      key: 'openFuyaoVersion',
      sorter: (a, b) => sorterFirstAlphabet(a.openFuyaoVersion, b.openFuyaoVersion),
      render: (_, record) => record.openFuyaoVersion || '--',
    },
    {
      title: intl.formatMessage({ id: 'cluster.col.k8sVersion' }),
      width: 200,
      key: 'k8sVersion',
      sorter: (a, b) => sorterFirstAlphabet(a.kubernetesVersion, b.kubernetesVersion),
      render: (_, record) => record.kubernetesVersion || '--',
    },
    {
      title: intl.formatMessage({ id: 'cluster.col.cpuArch' }),
      width: 200,
      key: 'node_cpu',
      sorter: (a, b) => sorterFirstAlphabet(judgeCpu(a.isAmd, a.isArm), judgeCpu(b.isAmd, b.isArm)),
      render: (_, record) => judgeCpu(record.isAmd, record.isArm),
    },

    {
      title: 'OS',
      width: 200,
      key: 'node_os',
      sorter: (a, b) => sorterFirstAlphabet(a.osList.join(','), b.osList.join(',')),
      render: (_, record) => (record.osList && record.osList.length ? record.osList.join(',') : '--'),
    },
    {
      title: intl.formatMessage({ id: 'cluster.col.runtime' }),
      width: 200,
      key: 'node_kind',
      render: (_, record) => 'containerd',
    },
    {
      title: intl.formatMessage({ id: 'cluster.col.createTime' }),
      key: 'cluster_createTime',
      width: 200,
      sorter: (a, b) => sortCluster(a.createTime, b.createTime),
      render: (_, record) => record.createTime || '--',
    },
    {
      title: intl.formatMessage({ id: 'cluster.col.actions' }),
      width: 96,
      key: 'handle',
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Popover
            className="podTab"
            placement="bottom"
            getPopupContainer={() => document.querySelector('.tab_table_flex')}
            content={
              <div className="pop_modal">
                <Button
                  type="link"
                  onClick={() => handleOpenPrepareModal(record)}
                  disabled={record.status !== 'Healthy' || disabledHandle(record.status)}
                >
                  {intl.formatMessage({ id: 'cluster.upgradePrepare' })}
                </Button>
                <Button
                  type="link"
                  onClick={() => handleOpenUpgradeModal(record)}
                  disabled={record.status !== 'Healthy' || disabledHandle(record.status)}
                >
                  {intl.formatMessage({ id: 'cluster.upgrade' })}
                </Button>
                <Button
                  type="link"
                  onClick={() => handleDeleteClusterWorkload(record)}
                  disabled={disabledHandle(record.status)}
                >
                  {intl.formatMessage({ id: 'cluster.delete' })}
                </Button>
              </div>
            }
            trigger="click"
            open={clusterOpen === record.name}
            onOpenChange={(e) => handleClusterOpenChange(e, record)}
          >
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover>
        </Space>
      ),
    },
  ];

  // 获取clusterList
  const getClusterList = useCallback(async (isChange = false) => {
    setClusterLoading(true);
    const [error, res] = await requestResource(listcluster());
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'cluster.getListFailed' }));
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        const clusterList = res.data.data.items.sort((a, b) => {
          const timeA = new Date(a.createTime).getTime();
          const timeB = new Date(b.createTime).getTime();
          return timeB - timeA; // 降序排序
        });
        setClusterList(clusterList);
        handleSearchCluster(clusterList, isChange); // 先搜索
      }
    }
    setClusterLoading(false);
  }, []);

  // 检索
  const handleSearchCluster = (totalData = clusterList, isChange = true) => {
    const clusterFormName = clusterForm.getFieldValue('cluster_name');
    if (clusterFormName) {
      setClusterListFilterd(
        totalData.filter((item) => item.name.toLowerCase().includes(clusterFormName.toLowerCase())),
      );
    } else {
      setClusterListFilterd(totalData);
    }
    if (isChange) {
      currentPageRef.current = 1; // 页码重置
      setClusterPage(1);
    }
  };

  const updateDisplayData = () => {
    const maxPage = Math.ceil(clusterListFilterd.length / clusterPageSize) || 1; // 最大页数
    // 如果数据不够分页(即分页后数据长度为空)
    if (currentPageRef.current > maxPage) {
      currentPageRef.current = maxPage;
    }
    const start = (currentPageRef.current - 1) * clusterPageSize;
    const end = start + clusterPageSize;
    return clusterListFilterd.slice(start, end);
  };

  const handleClusterTableChange = useCallback((pagination, filter, _sorter, extra) => {
    if (extra.action === 'filter') {
      setFilterClusterValue(filter.clusterStatus);
    }
    if (extra.action === 'paginate') {
      currentPageRef.current = pagination.current;
      setClusterPage(pagination.current);
      setClusterPageSize(pagination.pageSize);
    }
  }, []);

  const timeSearch = () => {
    localStorage.setItem('clusterTimeManageId', String(timeOutId));
    window.clearTimeout(timeOutId);
    getClusterList();
    const id = window.setTimeout(() => {
      timeSearch();
    }, 10000);
    timeOutId = id;
  };

  useEffect(() => {
    if (timeOutId) {
      window.clearTimeout(timeOutId);
    }
    timeSearch();
    return () => {
      localStorage.removeItem('clusterTimeManageId');
      window.clearTimeout(timeOutId);
    };
  }, []);

  const [patchYaml, setPatchYaml] = useState('');

  const handleUploadPatch = async () => {
    if (uploadPending) return;
    if (patchYaml) {
      const singleLinePatchYaml = patchYaml.replace(/\n/g, '\n');
      const match = singleLinePatchYaml.match(/openFuyaoVersion\s*:\s*("([^"]*)"|'([^']*)'|([^\s\r\n]+))/);
      if (match === null) {
        messageApi.error(intl.formatMessage({ id: 'cluster.noVersionField' }));
        return;
      }
      const filename = match[1] || match[2] || match[3] || null;
      const body = {
        PatchFileName: `${filename}.yaml`,
        PatchFileContent: singleLinePatchYaml,
      };
      setUploadLoading(true);
      try {
        const res = await createPatch(body);
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'cluster.uploadSuccess' }));
          setUploadPatchModalOpen(false);
          setPatchYaml('');
        } else {
          messageApi.error(intl.formatMessage({ id: 'cluster.uploadFailed' }));
        }
      } catch (e) {
        messageApi.error(intl.formatMessage({ id: 'cluster.uploadFailedMsg' }, { message: e.message }));
      } finally {
        setUploadLoading(false);
      }
    } else {
      messageApi.error(intl.formatMessage({ id: 'cluster.inputPatchContent' }));
    }
  };

  const handlerCancelUploadPatch = () => {
    setUploadPatchModalOpen(false);
    setPatchYaml('');
  };
  return (
    <>
      <BreadCrumbCom
        items={[
          {
            title: intl.formatMessage({ id: 'cluster.breadcrumb' }),
            path: `/${containerRouterPrefix}/cluster`,
            disabled: true,
          },
        ]}
      />
      <div className="tab_container container_margin_box">
        <ToastMsg contextHolder={contextHolder} />
        <Form className="pod_searchForm form_padding_bottom" form={clusterForm}>
          <Form.Item name="cluster_name" className="pod_search_input">
            <Input.Search
              placeholder={intl.formatMessage({ id: 'cluster.searchName' })}
              onSearch={() => handleSearchCluster()}
              autoComplete="off"
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button className="cluster_primary_btn" onClick={() => setUploadPatchModalOpen(true)}>
                {intl.formatMessage({ id: 'cluster.patchUpload' })}
              </Button>
              <Button
                className="cluster_primary_btn"
                icon={<PlusOutlined />}
                onClick={() => history.push(`/cluster/create`)}
              >
                {intl.formatMessage({ id: 'cluster.addCluster' })}
              </Button>
            </Space>
          </Form.Item>
        </Form>
        <div className="tab_table_flex">
          <ConfigProvider>
            <Table
              className="table_padding"
              loading={clusterLoading}
              columns={clusterColumns}
              dataSource={updateDisplayData()}
              onChange={handleClusterTableChange}
              pagination={{
                className: 'page',
                pageSize: clusterPageSize,
                current: clusterPage,
                total: clusterListFilterd.length,
                showTotal: (total) => intl.formatMessage({ id: 'cluster.totalItems' }, { total }),
                showSizeChanger: true,
                showQuickJumper: true,
                pageSizeOptions: [10, 20, 50],
              }}
              scroll={{ x: 1572 }}
            />
          </ConfigProvider>
        </div>
        <DeleteInfoModal
          title={intl.formatMessage({ id: 'cluster.deleteCluster' })}
          open={clusterDelModalOpen}
          cancelFn={handleDeleletCluster}
          content={[
            intl.formatMessage({ id: 'cluster.deleteWarning' }),
            intl.formatMessage({ id: 'cluster.confirmDelete' }, { name: clusterDelName }),
          ]}
          checkText={intl.formatMessage({ id: 'cluster.deleteCheckText' })}
          specialText={intl.formatMessage({ id: 'cluster.deleteSpecialText' })}
          showSpecialCheck={isShowSpecialCheck}
          isSpecialCheck={isSpecialCheck}
          specialCheckFn={handleSpecialCheckFn}
          isCheck={isClusterDelCheck}
          showCheck={true}
          checkFn={handleClusterCheckFn}
          confirmFn={handleDeleletClusterConfirm}
        />
        {upgradeModalOpen && (
          <UpgradeClusterModal
            open={upgradeModalOpen}
            clusterInfo={clusterList.find((item) => item.name === upgradeClusterName)}
            okFn={handleUpgrade}
            cancelFn={handleIndexCancelUpgrade}
          />
        )}
        {logOpen && (
          <CreateClusterLog clusterName={logClusterName} open={logOpen} okFn={handleLogExit} cancelFn={handleLogExit} />
        )}
        <Modal
          width={900}
          title={intl.formatMessage({ id: 'cluster.uploadPatchTitle' })}
          open={uploadPatchModalOpen}
          onOk={() => handleUploadPatch()}
          okButtonProps={{ loading: uploadLoading, disabled: uploadPending }}
          onCancel={() => handlerCancelUploadPatch()}
          getContainer={false}
        >
          <div className="upload-patch">
            <p className="upload-patch-desc">{intl.formatMessage({ id: 'cluster.uploadPatchDesc' })}</p>
            <div className="code_mirror_box">
              <CodeMirror
                value={patchYaml}
                onChange={(value, _) => setPatchYaml(value)}
                extensions={[
                  yaml({
                    closeBrackets: true,
                    closeBracketsKeymap: true,
                    closeBracketsWhitelist: '()[]{}""\'',
                  }),
                ]}
              />
            </div>
          </div>
        </Modal>
        {prepareModalOpen && (
          <UpgradePrepareModal
            open={prepareModalOpen}
            loading={prepareConfirmLoading}
            okFn={handlePrepare}
            cancelFn={handleCancelPrepare}
          />
        )}
      </div>
    </>
  );
}
