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
import StatefulSetEvents from '@/pages/container/workload/statefulSet/detail/StatefulSetEvents';
import StatefulSetInfo from '@/pages/container/workload/statefulSet/detail/StatefulSetInfo';
import StatefulSetYaml from '@/pages/container/workload/statefulSet/detail/StatefulSetYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useStore, useRef } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getStatefulSet, editAnnotationsOrLabels, deleteStatefulSet } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { useTranslation } from 'react-i18next';

export default function StatefulSetDetail() {
  const { t } = useTranslation(['workload', 'common']);
  const { name, namespace, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [statefulSetDelModalOpen, setStatefulSetDelModalOpen] = useState(false); // 删除对话框展示
  const [isStatefulSetDelCheck, setIsStatefulSetDelCheck] = useState(false); // 是否选中
  const [detailLoaded, setDetailLoded] = useState(false); // 加载完成
  const detailReqSeqRef = useRef(0);

  const [statefulSetDetailTabKey, setStatefulSetDetailTabKey] = useState(activeKey || 'info');

  const [statefulSetDetailData, setStatefulSetDetailData] = useState({}); // 详情数据

  const [statefulSetPopOpen, setStatefulSetPopOpen] = useState(false); // 气泡悬浮

  const [isReadyOnly, setIsReadyOnly] = useState(true); // 是否只读

  const [isStatefulSetAnnotationModalOpen, setIsStatefulSetAnnotationModalOpen] = useState(false);
  const [isStatefulSetLabelModalOpen, setIsStatefulSetLabelModalOpen] = useState(false);
  const [statefulSetYaml, setStatefulSetYaml] = useState('');
  const [oldStatefulSetAnnotations, setOldStatefulSetAnnotations] = useState();
  const [oldStatefulSetLabels, setOldStatefulSetLabels] = useState();
  const themeStore = useStore('theme');

  // 气泡
  const handleStatefulSetPopOpenChange = (open) => {
    setStatefulSetPopOpen(open);
  };

  const handleSetStatefulSetDetailTabKey = (key) => {
    setStatefulSetDetailTabKey(key);
    setIsReadyOnly(true);
  };

  // 删除按钮
  const handleDeleteStatefulSet = () => {
    setStatefulSetPopOpen(false); // 气泡框
    setStatefulSetDelModalOpen(true); // 打开弹窗
  };

  const handleDeleteStatefulSetCancel = () => {
    setStatefulSetDelModalOpen(false);
  };

  const handleDeleteStatefulSetConfirm = async () => {
    try {
      const res = await deleteStatefulSet(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setStatefulSetDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/workload/statefulSet`);
        });
      }
    } catch (e) {
      if (e?.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, e);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: e.response?.data?.message }));
      }
    }
  };

  const handleStatefulSetCheckFn = (e) => {
    setIsStatefulSetDelCheck(e.target.checked);
  };

  const getStatefulSetDetailInfo = useCallback(async () => {
    if (name && namespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getStatefulSet(namespace, name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setStatefulSetYaml(jsonToYaml(JSON.stringify(res.data))); // 先传递保持元数据clean
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldStatefulSetAnnotations([...res.data.metadata.annotations]);
          setOldStatefulSetLabels([...res.data.metadata.labels]);
          setStatefulSetDetailData(res.data);
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

  const handleModify = () => {
    setStatefulSetDetailTabKey('yaml'); // 跳向yaml
    setIsReadyOnly(false); // 退出只读模式
  };

  // 注解成功回调
  const handleAnnotationOk = async (statefulSetData) => {
    if (JSON.stringify(oldStatefulSetAnnotations) === JSON.stringify(statefulSetData)) {
      messageApi.info(t('messages.annotation', { ns: 'common' }) + t('messages.noChanges', { ns: 'common' }));
      setIsStatefulSetAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      statefulSetData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldStatefulSetAnnotations, statefulSetData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('statefulset', namespace, name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getStatefulSetDetailInfo();
              setIsStatefulSetAnnotationModalOpen(false);
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
    setIsStatefulSetAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsStatefulSetAnnotationModalOpen(true);
    setStatefulSetPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsStatefulSetLabelModalOpen(true);
    setStatefulSetPopOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (statefulSetData) => {
    if (JSON.stringify(oldStatefulSetLabels) === JSON.stringify(statefulSetData)) {
      messageApi.info(t('messages.label', { ns: 'common' }) + t('messages.noChanges', { ns: 'common' }));
      setIsStatefulSetLabelModalOpen(false);
    } else {
      const keyArr = [];
      statefulSetData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldStatefulSetLabels, statefulSetData, 'label');
        try {
          const res = await editAnnotationsOrLabels('statefulset', namespace, name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getStatefulSetDetailInfo();
              setIsStatefulSetLabelModalOpen(false);
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
    setIsStatefulSetLabelModalOpen(false);
  };

  // 处理只读
  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  useEffect(() => {
    if (statefulSetDetailTabKey === 'info' || activeKey) {
      getStatefulSetDetailInfo();
    }
  }, [statefulSetDetailTabKey, getStatefulSetDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <StatefulSetInfo
        statefulSetName={name}
        statefulSetNamespace={namespace}
        statefulSetDetailDataProps={statefulSetDetailData}
        refreshFn={getStatefulSetDetailInfo}
      />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <StatefulSetYaml
        statefulSetYamlProps={statefulSetYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getStatefulSetDetailInfo} />,
    },
    {
      key: 'events',
      label: t('actions.events'),
      children: <StatefulSetEvents
        statefulSetName={name}
        statefulSetNamespace={namespace} />,
    },
  ];

  return <div className="child_content withBread_content">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/workload`, disabled: true },
      { title: t('statefulSet.title'), path: '/statefulSet' },
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
            <Button type="link" onClick={handleDeleteStatefulSet}>{t('actions.delete')}</Button>
          </Space>
        }
        open={statefulSetPopOpen}
        onOpenChange={handleStatefulSetPopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isStatefulSetAnnotationModalOpen} type="annotation" dataList={statefulSetDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isStatefulSetLabelModalOpen} type="label" dataList={statefulSetDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: 'StatefulSet' })}
        open={statefulSetDelModalOpen}
        cancelFn={handleDeleteStatefulSetCancel}
        content={[
          t('actions.deleteWarning', { resource: 'StatefulSet' }),
          t('actions.deleteConfirmMessage', { resource: 'StatefulSet', name }),
        ]}
        isCheck={isStatefulSetDelCheck}
        showCheck={true}
        checkFn={handleStatefulSetCheckFn}
        confirmFn={handleDeleteStatefulSetConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetStatefulSetDetailTabKey} activeKey={statefulSetDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}