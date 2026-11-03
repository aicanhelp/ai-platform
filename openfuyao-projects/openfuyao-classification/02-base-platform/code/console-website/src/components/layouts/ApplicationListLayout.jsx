/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
 *       http://license.coscl.org.cn/MulanPSL2
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Button, Form, Space, Input, Table, Popover, message, Tooltip } from 'antd';
import { SyncOutlined, MoreOutlined, CloseCircleFilled, CheckCircleFilled, LoadingOutlined, QuestionCircleFilled, ExclamationCircleFilled } from '@ant-design/icons';
import { useCallback, useStore, useEffect, useState } from 'openinula';
import { useTranslation } from 'react-i18next';
import { DEFAULT_CURRENT_PAGE } from '@/common/constants';
import { getHelmsData, getHelmHistoryVersionData, rollBackHelmVersion, deleteRelease } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import { Link, useHistory, useLocation } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import ApplicationBackModal from '@/components/layouts/ApplicationBackModal';
import { getHelmManageStateKey } from '@/utils/common';
import defaultIcon from '@/assets/images/helmIcon.png';
import '@/styles/pages/helm.less';
import Dayjs from 'dayjs';
import ToolTipComponent from '@/components/ToolTipComponent';
import { sorterFirstAlphabet, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import useButtonLoading from '@/hooks/useButtonLoading';

/**
 * Application/Extension List Layout Component
 *
 * @param {Object} props
 * @param {'application'|'extension'} props.type - Type: 'application' or 'extension'
 * @param {Array} props.extraColumns - Extra columns for extension (e.g., plugin toggle column)
 * @param {React.ReactNode} props.pluginStore - Plugin store for extension
 */
export default function ApplicationListLayout({ type = 'application', extraColumns = [] }) {
  const isExtension = type === 'extension';
  const { t } = useTranslation('application');

  // Text configuration based on type
  const textConfig = {
    breadcrumbTitle: isExtension ? t('extension.title') : t('module.title'),
    searchPlaceholder: t('searchPlaceholder', {
      ns: 'common',
      name: t(isExtension ? 'searchName.extension' : 'searchName.app', { ns: 'common' }),
    }),
    tableName: isExtension ? t('table.extensionName') : t('table.name'),
    tooltipDesc: isExtension
      ? t('extension.description')
      : t('module.description'),
    uninstallTitle: isExtension ? t('modal.uninstallExtensionTitle') : t('modal.uninstallTitle'),
  };

  // Route configuration based on type
  const routePrefix = isExtension ? 'extendManage' : 'applicationManageHelm';

  const [form] = Form.useForm();
  const history = useHistory();
  const location = useLocation();
  const [originalList, setOriginalList] = useState([]);
  const [filterValue, setFilterValue] = useState();
  const [filterStatus, setFilterStatus] = useState([]);

  const [messageApi, contextHolder] = message.useMessage();
  const [page, setPage] = useState(DEFAULT_CURRENT_PAGE);
  const [list, setList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [popOpen, setPopOpen] = useState('');

  // Uninstall modal state
  const [uninstallModal, setUninstallModal] = useState(false);
  const [uninstallName, setUninstallName] = useState('');
  const [uninstallNamespace, setUninstallNamespace] = useState('');
  const [isUninstallCheck, setIsUninstallCheck] = useState(false);

  // Rollback modal state
  const [backModal, setBackModal] = useState(false);
  const [backName, setBackName] = useState('');
  const [backAppVersion, setBackAppVersion] = useState('');
  const [currentVersion, setCurrentVersion] = useState('');
  const [releaseHistoryList, setReleaseHistoryList] = useState([]);
  const [checkedNameSpace, setCheckedNameSpace] = useState('');
  const [checkedName, setCheckedName] = useState('');
  const [checkedVersion, setCheckedVersion] = useState(1);

  // Extension specific state
  const [, setUninstallLoading] = useState(false);
  const [uninstallStatusList, setUninstallStatusList] = useState({});
  const [refreshLoading, setRefreshLoading] = useButtonLoading();
  const [checkList, setCheckList] = useState({});

  const themeStore = useStore('theme');

  // Reset button
  const handleReset = async () => {
    setRefreshLoading(true);
    try {
      await getList(false);
    } finally {
      setRefreshLoading(false);
    }
  };

  // Upgrade button
  const handleUpgrade = (record) => {
    history.push(`/${containerRouterPrefix}/${routePrefix}/${record.namespace}/${record.name}/upgrade`);
  };

  // Uninstall button
  const handleUninstall = (record) => {
    setPopOpen('');
    setUninstallModal(true);
    setUninstallName(record.name);
    setUninstallNamespace(record.namespace);
  };

  // Confirm uninstall
  const handleConfirmUninstall = async () => {
    if (isExtension) {
      setUninstallLoading(true);
      setUninstallStatusList(prev => ({ ...prev, [uninstallName]: true }));
    }

    try {
      const res = await deleteRelease(uninstallNamespace, uninstallName);
      if (res.status === ResponseCode.OK) {
        if (isExtension) {
          // 豁免：扩展卸载特殊 toast（duration 10 + 点击刷新），保留原时长与交互
          messageApi.open({
            duration: 10,
            content: (
              <div className='extend_customize_message'>
                <CheckCircleFilled className='extend_customize_message_icon' />
                <p className='extend_customize_context'>{t('plugin.uninstallSuccess', { name: uninstallName })}</p>
                <p className='extend_customize_operate' onClick={() => window.location.reload()}>{t('plugin.clickRefresh')}</p>
              </div>
            ),
          });
          // 关窗/刷新延时 1 秒与普通应用分支对齐；状态复位同样延后，避免等待期重复提交
          await new Promise((resolve) => {
            setTimeout(() => {
              setUninstallLoading(false);
              setUninstallStatusList(prev => ({ ...prev, [uninstallName]: false }));
              setUninstallModal(false);
              setIsUninstallCheck(false);
              getList();
              resolve();
            }, 1000);
          });
        } else {
          await showSuccessThen(messageApi, t('modal.uninstallSuccess'), () => {
            setUninstallModal(false);
            setIsUninstallCheck(false);
            getList();
          });
        }
      }
    } catch (error) {
      if (isExtension) {
        setUninstallLoading(false);
        setUninstallStatusList(prev => ({ ...prev, [uninstallName]: false }));
      }
      if (error.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      }
    }
  };

  // Cancel uninstall modal
  const handleCancelModal = () => {
    if (isExtension) {
      if (uninstallStatusList[uninstallName]) {
        messageApi.loading(t('status.uninstalling'));
      } else {
        setIsUninstallCheck(false);
        setUninstallStatusList(prev => ({ ...prev, [uninstallName]: false }));
      }
    } else {
      setIsUninstallCheck(false);
    }
    setUninstallModal(false);
  };

  // Uninstall checkbox
  const handleUninstallCheckFn = (e) => {
    setIsUninstallCheck(e.target.checked);
    if (isExtension) {
      setCheckList(prev => ({ ...prev, [uninstallName]: e.target.checked }));
    }
  };

  // Rollback button
  const handleRollback = (record) => {
    setPopOpen('');
    setBackModal(true);
    setBackName(record.name);
    setBackAppVersion(record.chart.metadata.appVersion);
    setCurrentVersion(record.version);
    getHistoryVersionList(record.namespace, record.name);
  };

  // Confirm rollback
  const handleConfirmRollback = async () => {
    try {
      const res = await rollBackHelmVersion(checkedNameSpace, checkedName, checkedVersion);
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('modal.rollbackSuccess'), () => {
          setBackModal(false);
          getList();
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(t('modal.rollbackFailed', { message: error.response?.data?.msg }));
      }
    }
  };

  // Cancel rollback modal
  const handleCancelRollbackModal = () => {
    setBackModal(false);
  };

  // Get history version list
  const getHistoryVersionList = useCallback(async (namespace, releaseName) => {
    try {
      const res = await getHelmHistoryVersionData(namespace, releaseName);
      if (res.status === ResponseCode.OK) {
        const arrList = res.data.data.map((item, index) => ({
          ...item,
          key: index + 1,
        }));
        setReleaseHistoryList(arrList);
      }
    } catch {
      setReleaseHistoryList([]);
    }
  }, []);

  // Row selection for rollback
  const rowSelection = {
    onChange: (_selectedRowKeys, selectedRows) => {
      setCheckedNameSpace(selectedRows[0].namespace);
      setCheckedVersion(selectedRows[0].version);
      setCheckedName(selectedRows[0].name);
    },
    getCheckboxProps: (record) => ({
      name: record.version,
      disabled: currentVersion === record.version,
    }),
  };

  // Status icon by stable phase key (matches status.* i18n)
  const stateIconFilter = (stateKey) => {
    if (stateKey === 'deployed') {
      return <CheckCircleFilled className={`${isExtension ? 'extend' : 'helm'}_state_successful`} />;
    }
    if (stateKey === 'failed') {
      return <CloseCircleFilled className={`${isExtension ? 'extend' : 'helm'}_state_failed`} />;
    }
    if (stateKey === 'unknown') {
      return <QuestionCircleFilled className={`${isExtension ? 'extend' : 'helm'}_state_unknown`} />;
    }
    if (stateKey === 'uninstalling') {
      return <ExclamationCircleFilled className={`${isExtension ? 'extend' : 'helm'}_state_unknown`} />;
    }
    if (stateKey === 'pending') {
      return <LoadingOutlined className={`${isExtension ? 'extend' : 'helm'}_state_pending`} />;
    }
    return <QuestionCircleFilled className={`${isExtension ? 'extend' : 'helm'}_state_unknown`} />;
  };

  // Table columns
  const columns = [
    {
      title: textConfig.tableName,
      key: 'name',
      dataIndex: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <Space size="middle">
          <div className={`${isExtension ? 'extend' : 'helm'}_name`}>
            {record.chart.metadata.icon ? (
              <img
                src={record.chart.metadata.icon}
                alt=""
                style={{ height: '34px', width: '34px', marginRight: '8px' }}
                onError={(e) => e.target.src = defaultIcon}
              />
            ) : (
              <img src={defaultIcon} alt="" style={{ height: '34px', width: '34px', marginRight: '8px' }} />
            )}
            <Link to={`/${containerRouterPrefix}/${routePrefix}/${record.namespace}/${record.name}`}>
              {record.name}
            </Link>
          </div>
        </Space>
      ),
    },
    {
      title: t('table.status'),
      key: 'state',
      filters: filterStatus,
      filteredValue: filterValue || [],
      filterMultiple: true,
      sorter: (a, b) => sorterFirstAlphabet(
        t(`status.${getHelmManageStateKey(a.info.status)}`),
        t(`status.${getHelmManageStateKey(b.info.status)}`),
      ),
      onFilter: (value, record) => getHelmManageStateKey(record.info.status) === value,
      render: (_, record) => {
        const stateKey = getHelmManageStateKey(record.info.status);
        return (
          <Space size="middle">
            <span>{stateIconFilter(stateKey)}</span>
            <span className={`${isExtension ? 'extend' : 'helm'}_state_margin`}>
              {t(`status.${stateKey}`)}
            </span>
          </Space>
        );
      },
    },
    ...extraColumns,
    {
      title: t('table.updateTime'),
      key: 'update_time',
      sorter: (a, b) => Dayjs(a.info.lastDeployed) - Dayjs(b.info.lastDeployed),
      render: (_, record) => (
        <Space size="middle">
          {Dayjs(record.info.lastDeployed).format('YYYY-MM-DD HH:mm')}
        </Space>
      ),
    },
    {
      title: t('actions.operation'),
      key: 'handle',
      render: (_, record) => (
        <Space>
          <Popover
            placement="bottom"
            content={
              <div className="pop_modal">
                <Button type="link" onClick={() => handleUpgrade(record)}>{t('actions.upgrade')}</Button>
                <Button type="link" onClick={() => handleRollback(record)}>{t('actions.rollback')}</Button>
                <Button type="link" onClick={() => handleUninstall(record)}>{t('actions.uninstall')}</Button>
              </div>
            }
            trigger="click"
            open={popOpen === `${record.name}_${record.namespace}`}
            onOpenChange={(newOpen) => newOpen ? setPopOpen(`${record.name}_${record.namespace}`) : setPopOpen('')}
          >
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover>
        </Space>
      ),
    },
  ];

  // Rollback table columns
  const backColumns = [
    {
      title: t('rollback.chartTemplate'),
      width: '20%',
      key: 'name',
      ellipsis: true,
      render: (_, record) => (
        <Space size="middle">
          <Tooltip placement="bottom" title={record.chartName}>
            <p className={currentVersion === record.version ? 'now_version' : ''}>{record.chartName}</p>
          </Tooltip>
        </Space>
      ),
    },
    {
      title: t('rollback.sequence'),
      key: 'version',
      ellipsis: true,
      render: (_, record) => (
        <Space size="middle" className={`${isExtension ? 'defaultExtendIndexClass' : 'defaultHelmIndexClass'}`}>
          <p className={currentVersion === record.version ? 'now_version' : ''}>{record.version}</p>
          {currentVersion === record.version && (
            <div className="back_version_disable">
              <span className={`back_version_disable_font ${isExtension ? 'defaultExtendIndexClass' : 'defaultHelmIndexClass'}`}>{t('rollback.currentVersion')}</span>
            </div>
          )}
        </Space>
      ),
    },
    {
      title: t('rollback.chart'),
      ellipsis: true,
      width: '25%',
      key: 'chart',
      render: (_, record) => (
        <Space size="middle" className={`${isExtension ? 'defaultExtendIndexClass' : 'defaultHelmIndexClass'}`}>
          <Tooltip placement="bottom" title={`${record.chartName}-${record.chartVersion}`}>
            <p className={currentVersion === record.version ? 'now_version' : ''}>
              {record.chartName}-{record.chartVersion}
            </p>
          </Tooltip>
        </Space>
      ),
    },
    {
      title: t('table.updateTime'),
      dataIndex: 'createTime',
      width: '25%',
      ellipsis: true,
      render: (_, record) => (
        <Space size="middle" className={`${isExtension ? 'defaultExtendIndexClass' : 'defaultHelmIndexClass'}`}>
          <p className={currentVersion === record.version ? 'now_version' : ''}>
            {Dayjs(record.lastDeployed).format('YYYY-MM-DD HH:mm')}
          </p>
        </Space>
      ),
    },
  ];

  // Get list data
  const getList = useCallback(async (isChange = true) => {
    setLoading(true);
    try {
      // Extension uses true, Application uses false
      const res = await getHelmsData(isExtension);
      if (res.status === ResponseCode.OK) {
        setOriginalList(res.data?.data?.items ? [...res.data.data.items] : []);
        handleSearch(res.data?.data?.items ?? [], isChange);
      }
    } catch (e) {
      if (e.response?.data?.code === ResponseCode.NotFound) {
        setList([]);
      }
    }
    setLoading(false);
  }, [isExtension]);

  // Search handler
  const handleSearch = (totalData = originalList, isChange = true) => {
    const searchName = form.getFieldValue(isExtension ? 'extend_name' : 'helm_name');
    let tempList = totalData || [];
    if (searchName) {
      tempList = tempList.filter(item =>
        (item.name).toLowerCase().includes(searchName.toLowerCase()),
      );
    }
    setList([...tempList]);
    if (isChange) { setPage(DEFAULT_CURRENT_PAGE); }
  };

  // Table change handler
  const handleTableChange = useCallback(
    (_pagination, filter, _sorter, extra) => {
      if (extra.action === 'filter') {
        setFilterValue(filter.state);
      }
    },
    [],
  );

  useEffect(() => {
    getList();
  }, [getList]);

  useEffect(() => {
    if (location?.state?.status) {
      const raw = location.state.status;
      const lower = String(raw).toLowerCase();
      let keys = [];
      // Direct match: overview chart passes 'Deployed'/'Pending'/'Failed', or URL carries filter values
      if (lower === 'deployed') {
        keys = ['deployed'];
      } else if (lower === 'failed') {
        keys = ['failed'];
      } else if (lower === 'pending') {
        // Overview chart's "Pending" aggregates pending-* and uninstalling statuses
        keys = ['pending', 'uninstalling'];
      }
      if (keys.length > 0) {
        setFilterValue(keys);
      }
      window.history.replaceState(null, '');
    }
  }, [location]);

  useEffect(() => {
    const statusKeys = ['deployed', 'failed', 'pending', 'uninstalling'];
    setFilterStatus(statusKeys.map((k) => ({ text: t(`status.${k}`), value: k })));
  }, [t]);

  // CSS class names based on type
  const containerClass = isExtension ? 'extend_all' : '';
  const tabTopClass = isExtension ? 'extend-tab-top' : 'helm-tab-top';
  const tabContainerClass = isExtension ? 'extend-tab-container' : 'helm-tab-container';
  const searchFormClass = isExtension ? 'extend-searchForm' : 'helm-searchForm';
  const searchInputClass = isExtension ? 'extend-search-input' : 'helm-search-input';

  return (
    <div className={`child_content withBread_content overview ${containerClass}`}>
      <ToastMsg contextHolder={contextHolder} />
      <div className={tabTopClass}>
        <BreadCrumbCom
          className="create_bread"
          items={[{ title: textConfig.breadcrumbTitle, path: `/${containerRouterPrefix}/${routePrefix}` }]}
        />
      </div>
      <ToolTipComponent>
        <span>{textConfig.tooltipDesc}</span>
      </ToolTipComponent>
      <div
        className={tabContainerClass}
        style={isExtension ? { backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' } : {}}
      >
        <Form className={`${searchFormClass} form_padding_bottom`} form={form}>
          <Form.Item name={isExtension ? 'extend_name' : 'helm_name'} className={searchInputClass}>
            <Input.Search
              placeholder={textConfig.searchPlaceholder}
              onSearch={() => handleSearch()}
              autoComplete="off"
              maxLength={53}
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button
                icon={<SyncOutlined />}
                onClick={handleReset}
                loading={refreshLoading}
                className="reset_btn"
                style={{ marginLeft: '16px' }}
              />
            </Space>
          </Form.Item>
        </Form>
        <div
          className="tab_table_flex"
          style={isExtension ? { backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' } : {}}
        >
          <Table
            className="table_padding"
            loading={loading}
            columns={columns}
            dataSource={list}
            onChange={handleTableChange}
            pagination={{
              className: 'page',
              current: page,
              showTotal: (total) => t('list.total', { total }),
              showSizeChanger: true,
              showQuickJumper: true,
              pageSizeOptions: [10, 20, 50],
              onChange: (p) => setPage(p),
            }}
          />
        </div>
        <DeleteInfoModal
          title={textConfig.uninstallTitle}
          open={uninstallModal}
          cancelFn={handleCancelModal}
          confirmText={t('actions.uninstall')}
          content={[
            t('modal.uninstallWarning'),
            isExtension ? t('modal.uninstallExtensionConfirm', { name: uninstallName }) : t('modal.uninstallConfirm', { name: uninstallName }),
          ]}
          isCheck={isExtension ? checkList[uninstallName] : isUninstallCheck}
          showCheck={true}
          confirmBtnStatus={isExtension ? uninstallStatusList[uninstallName] : undefined}
          checkFn={handleUninstallCheckFn}
          confirmFn={handleConfirmUninstall}
        />
        <ApplicationBackModal
          title={t('rollback.title')}
          open={backModal}
          name={backName}
          version={backAppVersion}
          dataSource={releaseHistoryList}
          cancelFn={handleCancelRollbackModal}
          tableColumns={backColumns}
          confirmFn={handleConfirmRollback}
          rowSelection={rowSelection}
        />
      </div>
    </div>
  );
}