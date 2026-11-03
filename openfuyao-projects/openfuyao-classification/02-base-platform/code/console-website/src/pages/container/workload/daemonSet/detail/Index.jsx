/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix } from '@/constant.js';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import DaemonSetEvents from '@/pages/container/workload/daemonSet/detail/DaemonSetEvents';
import DaemonSetInfo from '@/pages/container/workload/daemonSet/detail/DaemonSetInfo';
import DaemonSetYaml from '@/pages/container/workload/daemonSet/detail/DaemonSetYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useStore, useRef } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { useTranslation } from 'react-i18next';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getDaemonSetDetailDescription, editAnnotationsOrLabels, deleteDaemonSet } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';

export default function DaemonSetDetail() {
  const { t } = useTranslation(['workload', 'common']);
  const { name, namespace, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [daemonSetDelModalOpen, setDaemonSetDelModalOpen] = useState(false); // 删除对话框展示
  const [isDaemonSetDelCheck, setIsDaemonSetDelCheck] = useState(false); // 是否选中
  const [detailLoaded, setDetailLoded] = useState(false); // 加载完成
  const detailReqSeqRef = useRef(0);

  const [daemonSetDetailTabKey, setDaemonSetDetailTabKey] = useState(activeKey || 'info');

  const [daemonSetDetailData, setDaemonSetDetailData] = useState({}); // 详情数据

  const [daemonSetPopOpen, setDaemonSetPopOpen] = useState(false); // 气泡悬浮
  const [isReadyOnly, setIsReadyOnly] = useState(true); // 是否只读

  const [isDaemonSetAnnotationModalOpen, setIsDaemonSetAnnotationModalOpen] = useState(false);
  const [isDaemonSetLabelModalOpen, setIsDaemonSetLabelModalOpen] = useState(false);
  const [oldDaemonSetAnnotations, setOldDaemonsetAnnotations] = useState();
  const [oldDaemonSetLabels, setOldDaemonSetLabels] = useState();

  const [daemonSetYaml, setDaemonSetYaml] = useState(''); // 传递yaml

  const themeStore = useStore('theme');

  // 处理只读
  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handleModify = () => {
    setDaemonSetDetailTabKey('yaml'); // 跳向yaml
    setIsReadyOnly(false); // 退出只读模式
  };

  // 注解成功回调
  const handleAnnotationOk = async (daemonSetData) => {
    if (JSON.stringify(oldDaemonSetAnnotations) === JSON.stringify(daemonSetData)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsDaemonSetAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      daemonSetData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        const addAnnotationList = solveAnnotationOrLabelDiff(oldDaemonSetAnnotations, daemonSetData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('daemonset', namespace, name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getDaemonSetDetailInfo();
              setIsDaemonSetAnnotationModalOpen(false);
            });
          }
        } catch (error) {
          if (error?.response?.status === ResponseCode.Forbidden) {
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
    setIsDaemonSetAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsDaemonSetAnnotationModalOpen(true);
    setDaemonSetPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsDaemonSetLabelModalOpen(true);
    setDaemonSetPopOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (daemonSetData) => {
    if (JSON.stringify(oldDaemonSetLabels) === JSON.stringify(daemonSetData)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsDaemonSetLabelModalOpen(false);
    } else {
      const keyArr = [];
      daemonSetData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldDaemonSetLabels, daemonSetData, 'label');
        try {
          const res = await editAnnotationsOrLabels('daemonset', namespace, name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getDaemonSetDetailInfo();
              setIsDaemonSetLabelModalOpen(false);
            });
          }
        } catch (error) {
          if (error?.response?.status === ResponseCode.Forbidden) {
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
    setIsDaemonSetLabelModalOpen(false);
  };

  // 气泡
  const handleDaemonSetPopOpenChange = (open) => {
    setDaemonSetPopOpen(open);
  };

  const handleSetDaemonSetDetailTabKey = (key) => {
    setDaemonSetDetailTabKey(key);
    setIsReadyOnly(true);
  };

  // 删除按钮
  const handleDeleteDaemonSet = () => {
    setDaemonSetPopOpen(false); // 气泡框
    setDaemonSetDelModalOpen(true); // 打开弹窗
  };

  const handleDeleteDaemonSetCancel = () => {
    setDaemonSetDelModalOpen(false);
  };

  const handleDeleteDaemonSetConfirm = async () => {
    try {
      const res = await deleteDaemonSet(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setDaemonSetDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/workload/daemonSet`);
        });
      }
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };

  const handleDaemonSetDetailCheckFn = (e) => {
    setIsDaemonSetDelCheck(e.target.checked);
  };

  const getDaemonSetDetailInfo = useCallback(async () => {
    if (name && namespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getDaemonSetDetailDescription(namespace, name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setDaemonSetYaml(jsonToYaml(JSON.stringify(res.data))); // 先传递保持元数据clean
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldDaemonsetAnnotations([...res.data.metadata.annotations]);
          setOldDaemonSetLabels([...res.data.metadata.labels]);
          setDaemonSetDetailData(res.data);
        }
      } catch (error) {
        if (error.response?.status !== ResponseCode.OK) {
          messageApi.error(t('messages.fetchFailed', { ns: 'common', message: error.response?.data?.message }));
        }
      } finally {
        if (seq === detailReqSeqRef.current) {
          setDetailLoded(true);
        }
      }
    }
  }, [name, namespace]);

  useEffect(() => {
    if (daemonSetDetailTabKey === 'info' || activeKey) {
      getDaemonSetDetailInfo();
    }
  }, [daemonSetDetailTabKey, getDaemonSetDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <DaemonSetInfo
        daemonSetName={name}
        daemonSetNamespace={namespace}
        daemonSetDetailDataProps={daemonSetDetailData}
        refreshFn={getDaemonSetDetailInfo}
      />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <DaemonSetYaml
        daemonSetYamlProps={daemonSetYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getDaemonSetDetailInfo}
      />,
    },
    {
      key: 'events',
      label: t('actions.events'),
      children: <DaemonSetEvents
        daemonSetName={name}
        daemonSetNamespace={namespace} />,
    },
  ];

  return <div className="child_content withBread_content">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/workload`, disabled: true },
      { title: t('daemonSet.title'), path: '/daemonSet' },
      { title: t('actions.detail'), path: '/detail' },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
      <h3>{name}</h3>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleModify}>{t('actions.modify')}</Button>
            <Button type="link" onClick={handleEditLabel}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={handleEditAnnotation}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteDaemonSet}>{t('actions.delete')}</Button>
          </Space>
        }
        open={daemonSetPopOpen}
        onOpenChange={handleDaemonSetPopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isDaemonSetAnnotationModalOpen} type="annotation" dataList={daemonSetDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isDaemonSetLabelModalOpen} type="label" dataList={daemonSetDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: 'DaemonSet' })}
        open={daemonSetDelModalOpen}
        cancelFn={handleDeleteDaemonSetCancel}
        content={[
          t('actions.deleteWarning', { resource: 'DaemonSet' }),
          t('actions.deleteConfirmMessage', { resource: 'DaemonSet', name }),
        ]}
        isCheck={isDaemonSetDelCheck}
        showCheck={true}
        checkFn={handleDaemonSetDetailCheckFn}
        confirmFn={handleDeleteDaemonSetConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetDaemonSetDetailTabKey} activeKey={daemonSetDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}