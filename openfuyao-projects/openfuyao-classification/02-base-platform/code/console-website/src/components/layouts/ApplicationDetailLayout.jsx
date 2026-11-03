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

import { Tabs, Button, Popover, Space, message, Tooltip } from 'antd';
import { useEffect, useState, useCallback, useMemo, useStore, cloneElement } from 'openinula';
import { useTranslation } from 'react-i18next';
import { useParams, useHistory } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import { containerRouterPrefix } from '@/constant.js';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { getHelmDetailDescriptionData, getHelmHistoryVersionData, rollBackHelmVersion, deleteRelease } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { showSuccessThen } from '@/tools/utils';
import ApplicationBackModal from '@/components/layouts/ApplicationBackModal';
import Dayjs from 'dayjs';
import defaultIcon from '@/assets/images/helmIcon.png';
import ToastMsg from '@/components/ToastMsg';
import useButtonLoading from '@/hooks/useButtonLoading';

/**
 * Application/Extension Detail Layout Component
 *
 * @param {Object} props
 * @param {'application'|'extension'} props.type - Type: 'application' or 'extension'
 * @param {React.ReactNode} props.infoComponent - Info tab component
 * @param {React.ReactNode} props.yamlComponent - YAML tab component
 * @param {React.ReactNode} props.resourceComponent - Resource tab component
 * @param {React.ReactNode} props.logComponent - Log tab component
 * @param {React.ReactNode} props.eventComponent - Event tab component
 * @param {React.ReactNode} props.monitorComponent - Monitor tab component
 * @param {Function} props.onPluginToggle - Plugin toggle handler for extension
 * @param {boolean} props.isViewMode - Whether view mode is enabled for extension
 * @param {boolean} props.pluginShow - Whether plugin is shown for extension
 */
export default function ApplicationDetailLayout({
  type = 'application',
  infoComponent,
  yamlComponent,
  resourceComponent,
  logComponent,
  eventComponent,
  monitorComponent,
  onPluginToggle,
  isViewMode = true,
  pluginShow = false,
}) {
  const isExtension = type === 'extension';
  const { t } = useTranslation('application');
  const routePrefix = isExtension ? 'extendManage' : 'applicationManageHelm';
  const paramName = isExtension ? 'extend_name' : 'helm_name';
  const paramNamespace = isExtension ? 'extend_namespace' : 'helm_namespace';

  const param = useParams();
  const themeStore = useStore('theme');
  const history = useHistory();

  const [messageApi, contextHolder] = message.useMessage();
  const [tabKey, setTabKey] = useState('1');
  const [popOpen, setPopOpen] = useState(false);
  const [detailData, setDetailData] = useState({});
  const [description, setDescription] = useState({});
  const [icon, setIcon] = useState('');
  const [detailLoaded, setDetailLoaded] = useState(false);
  const [currentVersion, setCurrentVersion] = useState('');

  // Rollback state
  const [releaseHistoryList, setReleaseHistoryList] = useState([]);
  const [checkedNameSpace, setCheckedNameSpace] = useState('');
  const [checkedName, setCheckedName] = useState('');
  const [checkedVersion, setCheckedVersion] = useState(1);
  const [backModal, setBackModal] = useState(false);
  const [backName, setBackName] = useState('');
  const [backAppVersion, setBackAppVersion] = useState('');

  // Uninstall state
  const [uninstallModal, setUninstallModal] = useState(false);
  const [isUninstallCheck, setIsUninstallCheck] = useState(false);
  const [uninstallLoading, setUninstallLoading, uninstallPending] = useButtonLoading();

  const name = param[paramName];
  const namespace = param[paramNamespace];
  const actionEnabled = isExtension ? isViewMode : true;

  // Text configuration
  const textConfig = {
    breadcrumbTitle: isExtension ? t('extension.title') : t('module.title'),
    detailTitle: isExtension ? t('module.extensionDetail') : t('module.helmDetail'),
  };

  const detailDataPropName = isExtension ? 'extendDetailDataProps' : 'helmDetailDataProps';
  const injectDetailData = useCallback((component) => {
    if (!component) { return component; }
    return cloneElement(component, {
      [detailDataPropName]: detailData,
    });
  }, [detailDataPropName, detailData]);

  // Tab items
  const items = useMemo(() => [
    { key: '1', label: t('detail.tabInfo'), children: injectDetailData(infoComponent) },
    { key: '2', label: t('detail.tabYaml'), children: yamlComponent },
    { key: '3', label: t('detail.tabResource'), children: injectDetailData(resourceComponent) },
    { key: '4', label: t('detail.tabLog'), children: injectDetailData(logComponent) },
    { key: '5', label: t('detail.tabEvent'), children: injectDetailData(eventComponent) },
    { key: '6', label: t('detail.tabMonitor'), children: injectDetailData(monitorComponent) },
  ], [t, injectDetailData, infoComponent, yamlComponent, resourceComponent, logComponent, eventComponent, monitorComponent]);

  // Rollback columns
  const backColumns = [
    {
      title: t('rollback.chartTemplate'),
      width: '20%',
      ellipsis: true,
      key: 'name',
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
      ellipsis: true,
      key: 'version',
      render: (_, record) => (
        <Space size="middle" className={isExtension ? 'defaultExtendDetailClass' : 'defaultHelmDetailClass'}>
          <p className={currentVersion === record.version ? 'now_version' : ''}>{record.version}</p>
          {currentVersion === record.version && (
            <div className="back_version_disable">
              <span className={`back_version_disable_font ${isExtension ? 'defaultExtendDetailClass' : 'defaultHelmDetailClass'}`}>{t('rollback.currentVersion')}</span>
            </div>
          )}
        </Space>
      ),
    },
    {
      width: '25%',
      ellipsis: true,
      title: t('rollback.chart'),
      key: 'chart',
      render: (_, record) => (
        <Space size="middle" className={isExtension ? 'defaultExtendDetailClass' : 'defaultHelmDetailClass'}>
          <Tooltip placement="bottom" title={`${record.chartName}-${record.chartVersion}`}>
            <p className={currentVersion === record.version ? 'now_version' : ''}>
              {record.chartName}-{record.chartVersion}
            </p>
          </Tooltip>
        </Space>
      ),
    },
    {
      width: '25%',
      ellipsis: true,
      dataIndex: 'createTime',
      title: t('table.updateTime'),
      render: (_, record) => (
        <Space size="middle" className={isExtension ? 'defaultExtendIndexClass' : 'defaultHelmDetailClass'}>
          <p className={currentVersion === record.version ? 'now_version' : ''}>
            {Dayjs(record.lastDeployed).format('YYYY-MM-DD HH:mm')}
          </p>
        </Space>
      ),
    },
  ];

  // Handlers
  const handlePopOpenChange = (open) => setPopOpen(open);

  const handleUpgrade = () => {
    history.push(`/${containerRouterPrefix}/${routePrefix}/${namespace}/${name}/upgrade`);
  };

  const handleUninstall = () => {
    setPopOpen(false);
    setUninstallModal(true);
  };

  const handleUninstallCheckFn = (e) => setIsUninstallCheck(e.target.checked);

  const handleConfirmUninstall = async () => {
    if (isExtension) { setUninstallLoading(true); }

    try {
      const res = await deleteRelease(namespace, name);
      if (res.status === ResponseCode.OK) {
        if (isExtension) {
          // 关窗/跳转延时 1 秒与普通应用分支对齐；状态复位同样延后，避免等待期重复提交
          await new Promise((resolve) => {
            setTimeout(() => {
              setUninstallLoading(false);
              setUninstallModal(false);
              setIsUninstallCheck(false);
              history.push(`/${containerRouterPrefix}/${routePrefix}`);
              resolve();
            }, 1000);
          });
        } else {
          await showSuccessThen(messageApi, t('modal.uninstallSuccess'), () => {
            setUninstallModal(false);
            setIsUninstallCheck(false);
            history.push(`/${containerRouterPrefix}/${routePrefix}`);
          });
        }
      }
    } catch (error) {
      if (isExtension) { setUninstallLoading(false); }
      if (error.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(t('modal.uninstallFailed', { message: error.response?.data?.msg }));
      }
    }
  };

  const handleCancelUninstall = () => {
    if (isExtension && uninstallPending) {
      messageApi.loading(t('status.uninstalling'));
    }
    setUninstallModal(false);
    setIsUninstallCheck(false);
  };

  const handleRollback = () => {
    getHistoryVersionList();
    setPopOpen(false);
    setBackModal(true);
  };

  const handleConfirmRollback = async () => {
    try {
      const res = await rollBackHelmVersion(checkedNameSpace, checkedName, checkedVersion);
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('modal.rollbackSuccess'), () => {
          setBackModal(false);
          getDetailList();
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(t('modal.rollbackFailed', { message: error.response?.data?.msg }));
      }
      // 失败保持弹窗打开，可重试
    }
  };

  const handleCancelRollback = () => setBackModal(false);

  const handlePluginToggle = () => {
    setPopOpen(false);
    if (onPluginToggle) { onPluginToggle(false); }
    setTabKey('1');
  };

  const handleTabChange = (key) => {
    if (isExtension && !isViewMode && onPluginToggle) {
      onPluginToggle(true);
    }
    setTabKey(key);
  };

  const getHistoryVersionList = useCallback(async () => {
    try {
      const res = await getHelmHistoryVersionData(namespace, name);
      if (res.status === ResponseCode.OK) {
        const arr = res.data.data.map((item, index) => ({
          ...item,
          key: index + 1,
        }));
        setReleaseHistoryList(arr);
      }
    } catch {
      setReleaseHistoryList([]);
    }
  }, [namespace, name]);

  const getDetailList = useCallback(async () => {
    if (name) {
      setDetailLoaded(false);
      try {
        const res = await getHelmDetailDescriptionData(namespace, name);
        if (res.status === ResponseCode.OK) {
          setCurrentVersion(res.data.data.version);
          setBackAppVersion(res.data.data.chart.metadata.appVersion);
          setBackName(res.data.data.name);
          setDescription(res.data.data.chart.metadata.description);
          setIcon(res.data.data.chart.metadata.icon);
          setDetailData(res.data.data);
        }
      } catch {
        // Handle error
      }
      setDetailLoaded(true);
    }
  }, [name, namespace]);

  const rowSelection = {
    onChange: (_selectedRowKeys, selectedRows) => {
      if (!selectedRows.length) { return; }
      setCheckedNameSpace(selectedRows[0].namespace);
      setCheckedName(selectedRows[0].name);
      setCheckedVersion(selectedRows[0].version);
    },
    getCheckboxProps: (record) => ({
      disabled: currentVersion === record.version,
      name: record.version,
    }),
  };

  useEffect(() => {
    getDetailList();
  }, [getDetailList]);

  // CSS class names based on type
  const containerClass = isExtension ? 'extend_all' : '';
  const detailTitleClass = isExtension ? 'extend_detail_title' : 'helm_detail_title';

  return (
    <div className={`child_content withBread_content ${containerClass}`}>
      <ToastMsg contextHolder={contextHolder} />
      <BreadCrumbCom items={[
        { title: textConfig.breadcrumbTitle, path: `/${containerRouterPrefix}/${routePrefix}` },
        { title: textConfig.detailTitle, path: `/${containerRouterPrefix}/${routePrefix}/${namespace}/${name}` },
      ]} />
      <div
        className={detailTitleClass}
        style={isExtension ? { backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' } : {}}
      >
        <div style={{ display: 'flex' }}>
          <div>
            {icon ? (
              <img
                src={icon}
                alt=""
                style={{ height: '30px', width: '30px', marginRight: '8px' }}
                onError={(e) => e.target.src = defaultIcon}
                className='title_image'
              />
            ) : (
              <img src={defaultIcon} alt="" style={{ height: '30px', width: '30px', marginRight: '8px' }} className='title_image' />
            )}
          </div>
          <div className={isExtension ? 'extend_descript_group' : 'descript_group'}>
            <div style={{ marginRight: '64px' }}>
              <h3 className='descript_group_name'>{name}</h3>
            </div>
            <div style={{ marginRight: '64px' }}>
              <p className='descript_group_description'>{description}</p>
            </div>
          </div>
        </div>
        <div style={{ display: 'flex' }}>
          <Popover
            placement='bottom'
            content={
              <Space className='column_pop'>
                <Button type="link" onClick={handleUpgrade}>{t('actions.upgrade')}</Button>
                <Button type="link" onClick={handleRollback}>{t('actions.rollback')}</Button>
                {isExtension && (
                  <Button
                    disabled={(detailData.info?.status === 'failed' || !pluginShow)}
                    type="link"
                    onClick={handlePluginToggle}
                  >
                    {t('actions.pluginToggle')}
                  </Button>
                )}
                <Button type="link" onClick={handleUninstall}>{t('actions.uninstall')}</Button>
              </Space>
            }
            open={popOpen && actionEnabled}
            onOpenChange={handlePopOpenChange}
          >
            <Button className={actionEnabled ? 'primary_btn' : 'disable_btn'} disabled={!actionEnabled}>
              {t('actions.operation')} <DownOutlined className='small_margin_adjust' />
            </Button>
          </Popover>
        </div>
      </div>
      {detailLoaded && (
        <Tabs
          items={items}
          onChange={handleTabChange}
          activeKey={tabKey}
          destroyInactiveTabPane={true}
          size={isExtension ? 'small' : undefined}
        />
      )}
      <DeleteInfoModal
        title={isExtension ? t('modal.uninstallExtensionTitle') : t('modal.uninstallTitle')}
        open={uninstallModal}
        cancelFn={handleCancelUninstall}
        content={[
          t('modal.uninstallWarning'),
          isExtension ? t('modal.uninstallExtensionConfirm', { name }) : t('modal.uninstallConfirm', { name }),
        ]}
        confirmBtnStatus={isExtension ? uninstallLoading : undefined}
        confirmText={t('actions.uninstall')}
        isCheck={isUninstallCheck}
        showCheck={true}
        checkFn={handleUninstallCheckFn}
        confirmFn={handleConfirmUninstall}
      />
      <ApplicationBackModal
        title={t('rollback.title')}
        name={backName}
        version={backAppVersion}
        open={backModal}
        cancelFn={handleCancelRollback}
        tableColumns={backColumns}
        dataSource={releaseHistoryList}
        rowSelection={rowSelection}
        confirmFn={handleConfirmRollback}
      />
    </div>
  );
}