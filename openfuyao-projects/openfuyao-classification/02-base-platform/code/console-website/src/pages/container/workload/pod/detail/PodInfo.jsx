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
import { Link } from 'inula-router';
import { containerRouterPrefix } from '@/constant';
import { Fragment, useCallback, useEffect, useMemo, useState, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import { CloseOutlined, CodeOutlined } from '@ant-design/icons';
import { ResponseCode } from '@/common/constants';
import { editAnnotationsOrLabels, getPodLog } from '@/api/containerApi';
import Dayjs from 'dayjs';
import { EditOutlined, CalendarOutlined } from '@ant-design/icons';
import { Table, message, Modal, Spin } from 'antd';
import AnnotationModal from '@/components/AnnotationModal';
import {
  firstAlphabetUp,
  solveAnnotationOrLabelDiff,
  forbiddenMsg,
  showSuccessThen,
  solveHostIpAddress,
  solvePodIpAddress,
} from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import ToolTipComponent from '@/components/ToolTipComponent';
import NoPermissionsModal from '@/components/NoPermissionsModal';
import TerminalNewModal from '@/components/webTerminal/TerminalNewModal';
import { checkTerminalAccess } from '@/api/terminalApi';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function PodInfo({ podName, podNamespace, podDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation('common');
  const [isShow, setIsShow] = useState(true);
  const [containerTableTotalData, setContainerTableTotalData] = useState([]); // 全部数据
  const [messageApi, contextHolder] = message.useMessage();

  const oldAnnotations = podDetailDataProps.metadata.annotations;
  const oldLabels = podDetailDataProps.metadata.labels;

  const [podContainerTableData, setPodContainerTableData] = useState([]); // 容器列表数据
  const [filterContainerStatus, setFilterContainerStatus] = useState([]); // 赋值筛选项

  const [containerLoading, setContainerLoading] = useState(false); // 加载中
  const [containerFilterObj, setContainerFilterObj] = useState({});
  const [logContent, setLogContent] = useState('');
  const [logName, setLogName] = useState('');
  const [logOpen, setLogOpen] = useState(false);
  const [logLoaded, setLogLoaded] = useState(true);
  const [noPermissionsModalOpen, setNoPermissionsModalOpen] = useState(false); // 无权限弹窗
  const [newTerminalModal, setNewTerminalModal] = useState(false); // 终端重复

  const [containerName, setContainerName] = useState('');

  // 对话框展示
  const [isPodAnnotationModalOpen, setIsPodAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);

  const themeStore = useStore('theme');
  const terminalStore = useStore('terminal');

  const handleShowChange = () => {
    setIsShow(false);
  };

  const getContainerTableData = useCallback(async () => {
    setContainerLoading(true);
    let _status = '';
    let containerData;
    // 筛选状态
    if (Object.keys(containerFilterObj).length !== 0) {
      _status = containerFilterObj.container_status ? containerFilterObj.container_status[0] : ''; // 获取筛选框
    }
    if (_status) {
      containerData = containerTableTotalData.filter(item => (item.containerReason).toLowerCase() === _status.toLowerCase()); // 相等项
    } else {
      containerData = containerTableTotalData;
    }
    setPodContainerTableData(containerData);
    setContainerLoading(false);
  }, [containerFilterObj, containerTableTotalData]);

  // 表格变化
  const handleContainerTableChange = useCallback(
    (
      _pagination,
      filter,
      _sorter,
      extra,
    ) => {
      if (extra.action === 'filter') {
        setContainerFilterObj(filter);
      }
    },
    [],
  );

  // 请求日志
  const handleRequestLog = async (name) => {
    if (!logLoaded) { return; } // 请求进行中防重复点击
    setLogName(name);
    setLogLoaded(false);
    try {
      const res = await getPodLog(podNamespace, podName, name);
      if (res.status === ResponseCode.OK) {
        setLogContent(res.data);
        setLogOpen(true);
        setLogLoaded(true);
      }
    } catch (e) {
      setLogLoaded(true);
      if (e?.response?.data?.code === ResponseCode.BadRequest) {
        messageApi.error(e.response.data.message);
      } else {
        messageApi.error(t('resourceDetail.logFetchError'));
      }
    }
  };

  const handleTerminalConsole = async (name) => {
    // 检查权限
    setContainerName(name);
    terminalStore.$a.setTerminalPodDescription({ namespace: podNamespace, pod: podName });
    try {
      const permissionsRes = await checkTerminalAccess(
        `/namespace/${terminalStore.$s.podDescription.namespace}/pod/${terminalStore.$s.podDescription.pod}/container/${name}/terminal`,
      );
      if (permissionsRes.data.Code === ResponseCode.OK) {
        if (terminalStore.$s.isOpen) {
          setNewTerminalModal(true);
        } else {
          terminalStore.$a.setTerminalInit({ type: 'container', name });
          terminalStore.$a.setTerminalScreenSize('middle');
          terminalStore.$a.setTerminalOpen(true);
        }
      }
      if (permissionsRes.data.Code === ResponseCode.Forbidden) {
        setNoPermissionsModalOpen(true);
      }
      if (permissionsRes.data.Code === ResponseCode.NotFound) {
        messageApi.error(permissionsRes.data.data);
      }
      if (permissionsRes.data.Code === ResponseCode.InternalServerError) {
        messageApi.error(t('resourceDetail.serviceError'));
      }
    } catch (e) {
      if (e?.response?.data?.Code === ResponseCode.Forbidden) {
        setNoPermissionsModalOpen(true);
      } else if (e?.response?.data?.Code === ResponseCode.InternalServerError) {
        messageApi.error(t('resourceDetail.serviceError'));
      } else {
        messageApi.error(t('resourceDetail.terminalCheckFailed'));
      }
    }
  };

  const podDetailColumns = useMemo(() => [
    {
      title: t('resourceDetail.containerName'),
      key: 'container_name',
      render: (_, record) => <span>{record.containerName}</span>,
    },
    {
      title: t('common.status'),
      filters: filterContainerStatus,
      filterMultiple: false,
      key: 'container_status',
      render: (_, record) => <p className={`resource_status ${(record.containerStatus).toLowerCase()}_circle`}>
        {firstAlphabetUp(record.containerReason)}
      </p>,
    },
    {
      title: t('resourceDetail.image'),
      width: '25%',
      key: 'container_mirror',
      ellipsis: true,
      render: (_, record) => record.containerMirror,
    },
    {
      title: t('resourceDetail.memory'),
      width: '15%',
      key: 'container_memory',
      render: (_, record) => record.containerMemory ? record.containerMemory : '--',
    },
    {
      title: t('resourceDetail.log'),
      key: 'container_log',
      render: (_, record) =>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <Spin spinning={!logLoaded && logName === record.containerName} />
          <CalendarOutlined className="primary_icon common_antd_icon" onClick={() => handleRequestLog(record.containerName)} />
        </div>,
    },
    {
      title: t('resourceDetail.terminal'),
      key: 'terminal_console',
      render: (_, record) => <CodeOutlined className="primary_icon common_antd_icon" onClick={() => handleTerminalConsole(record.containerName)} />,
    },
  ], [t, i18n.language, filterContainerStatus, logLoaded, logName, handleRequestLog, handleTerminalConsole]);

  const handleClosePermissionModal = () => {
    setNoPermissionsModalOpen(false);
  };

  const handleCancelNewTerminal = () => {
    setNewTerminalModal(false);
  };

  const handleOkNewTerminal = () => {
    terminalStore.$a.setTerminalInit({ type: 'container', name: containerName });
    terminalStore.$a.setTerminalScreenSize('middle');
    terminalStore.$a.setTerminalPodDescription({ namespace: podNamespace, pod: podName });
    terminalStore.$a.setTerminalOpen(false);
    setTimeout(() => {
      terminalStore.$a.setTerminalOpen(true);
    });
    setNewTerminalModal(false);
  };

  // 注解成功回调
  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.annotationUnchanged', { ns: 'common' }));
      setIsPodAnnotationModalOpen(false);
    } else {
      const podAnnotationkeyArr = [];
      data.map(item => podAnnotationkeyArr.push(item.key));
      if (podAnnotationkeyArr.filter((item, index) => podAnnotationkeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const resPodAnnotation = await editAnnotationsOrLabels('pod', podNamespace, podName, addAnnotationList);
          if (resPodAnnotation.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn(); // 先刷新防止页面抖动
              setIsPodAnnotationModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: error.response?.data?.message }));
          }
        }
      }
    }
  };

  // 注解失败回调
  const handleAnnotationCancel = () => {
    setIsPodAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.labelUnchanged', { ns: 'common' }));
      setIsLabelModalOpen(false);
    } else {
      const podLabelkeyArr = [];
      data.map(item => podLabelkeyArr.push(item.key));
      if (podLabelkeyArr.filter((item, index) => podLabelkeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的标签
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const resPodLabel = await editAnnotationsOrLabels('pod', podNamespace, podName, addLabelList);
          if (resPodLabel.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsLabelModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: error.response?.data?.message }));
          }
        }
      }
    }
  };

  // 标签失败回调
  const handleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handleLogOk = () => {
    setLogOpen(false);
    setLogName('');
    setLogContent('');
  };

  useEffect(() => {
    // 获取详情
    getContainerTableData();
  }, [getContainerTableData]);

  useEffect(() => {
    let total = [];
    let totalStatusList = [];
    let filterContainerList = [];
    podDetailDataProps?.spec?.containers?.map(item => {
      total.push({
        containerName: item.name,
        containerMemory: item.resources?.requests?.memory || '',
      });
    });
    podDetailDataProps.status?.containerStatuses?.map(item => {
      total.forEach((containerItem) => {
        if (item.name === containerItem.containerName) { // 判断是否存在name
          containerItem.containerStatus = Object.keys(item.state)[0] || 'Unknown'; // 默认取数组第一项状态key
          containerItem.containerReason = (Object.prototype.hasOwnProperty.call(item.state, containerItem.containerStatus) &&
            item.state[containerItem.containerStatus].reason) ||
            containerItem.containerStatus;
          containerItem.containerMirror = item.image || '--';
          totalStatusList.push(containerItem.containerReason);
        }
      });
    });
    totalStatusList = [...new Set(totalStatusList)];
    totalStatusList.map(statusItem => {
      filterContainerList.push({ text: firstAlphabetUp(statusItem), value: firstAlphabetUp(statusItem) });
    });
    setFilterContainerStatus([...filterContainerList]);
    setContainerTableTotalData([...total]);
  }, [podDetailDataProps]);

  const podBaseInfoItems = useMemo(() => {
    const d = podDetailDataProps;
    const phase = d?.status?.phase;
    return [
      { key: t('resourceDetail.podName'), value: d?.metadata?.name },
      { key: t('resourceDetail.nodeName'), value: d?.spec?.nodeName },
      {
        key: t('resourceDetail.podStatus'),
        value: phase && <span className={`${phase.toLowerCase()}_circle`}>{phase}</span>,
      },
      { key: t('resourceDetail.nodeIP'), value: solveHostIpAddress(d?.status) },
      { key: t('resourceDetail.podIP'), value: solvePodIpAddress(d?.status) },
      {
        key: t('resourceDetail.restartTotal'),
        value: d?.status?.containerStatuses?.reduce((acc, item) => acc + item.restartCount, 0),
      },
      { key: t('resourceDetail.namespace'), value: d?.metadata?.namespace },
      {
        key: t('resourceDetail.createdAt'),
        value: d?.metadata?.creationTimestamp && Dayjs(d.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
      },
    ];
  }, [podDetailDataProps, t, i18n.language]);

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={`${isShow ? '' : 'cant_show'}`}>
      <ToolTipComponent>
        <div className="podDetail" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className='promot-box'>
            {t('resourceDetail.monitorBanner')}
            <Link to={`/${containerRouterPrefix}/monitor/monitorDashboard`}>{t('resourceDetail.monitorLink')}</Link>
            {t('resourceDetail.monitorSuffix')}
          </div>
          <CloseOutlined onClick={handleShowChange} />
        </div>
      </ToolTipComponent>
    </div>
    <div className={`tab_container container_margin_box ${isShow ? 'tooltip_container_height' : 'normal_container_height'}`}>
      <div className="detail_card">
        <h3>{t('resourceDetail.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={podBaseInfoItems} columns={2} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.labelsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={podDetailDataProps?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {podDetailDataProps.metadata?.labels?.length ?
                podDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.annotationsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsPodAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isPodAnnotationModalOpen} type="annotation" dataList={podDetailDataProps?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {podDetailDataProps.metadata?.annotations?.length ?
                podDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
      <div className="detail_card container_card_add">
        <h3>{t('resourceDetail.containers')}</h3>
        <Table className="table_padding" style={{ paddingRight: '0px' }} loading={containerLoading} dataSource={podContainerTableData} rowKey={(record) => record.name ?? record.image} columns={podDetailColumns}
          pagination={{
            className: 'page',
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
            showTotal: total => t('resourceDetail.paginationTotal', { total }),
          }}
          onChange={handleContainerTableChange} />
      </div>
      <NoPermissionsModal open={noPermissionsModalOpen} handleOpen={handleClosePermissionModal} />
      <TerminalNewModal open={newTerminalModal} handleOk={handleOkNewTerminal} cancelFn={handleCancelNewTerminal} />
      <Modal
        className="pod_log_container"
        width={800}
        title={t('resourceDetail.containerLogTitle', { name: logName })}
        open={logOpen}
        onOk={handleLogOk}
        onCancel={handleLogOk}
        footer={(_, { OkBtn }) => (
          <>
            <OkBtn />
          </>
        )}
      >
        <div className="pod_log_content">
          <pre>{logContent}</pre>
        </div>
      </Modal>
    </div>
  </Fragment>;
}