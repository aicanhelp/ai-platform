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
import PodEvents from '@/pages/container/workload/pod/detail/PodEvents';
import PodInfo from '@/pages/container/workload/pod/detail/PodInfo';
import PodYaml from '@/pages/container/workload/pod/detail/PodYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useStore, useRef } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { useTranslation } from 'react-i18next';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getPod, editAnnotationsOrLabels, deletePod } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotation, jsonToYaml, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';

export default function PodDetail() {
  const { t } = useTranslation(['workload', 'common']);
  const { name, namespace, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [podDelModalOpen, setPodDelModalOpen] = useState(false); // 删除对话框展示
  const [isPodDelCheck, setIsPodDelCheck] = useState(false); // 是否选中
  const [detailLoaded, setDetailLoded] = useState(false); // 加载完成
  const detailReqSeqRef = useRef(0);

  const [oldAnnotations, setOldAnnotations] = useState([]); // 未修改前注解
  const [oldLabels, setOldLabels] = useState([]);

  const [podDetailTabKey, setPodDetailTabKey] = useState(activeKey || 'info');

  const [podDetailData, setPodDetailData] = useState({}); // 详情数据

  const [podPopOpen, setPodPopOpen] = useState(false); // 气泡悬浮
  // 对话框展示
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const [isReadyOnly, setIsReadyOnly] = useState(true); // 是否只读

  const [podYaml, setPodYaml] = useState('');

  const themeStore = useStore('theme');

  // 气泡
  const handlePodPopOpenChange = (open) => {
    setPodPopOpen(open);
  };

  const handleSetPodDetailTabKey = (key) => {
    setPodDetailTabKey(key);
    setIsReadyOnly(true);
  };

  // 删除按钮
  const handleDeletePod = () => {
    setPodPopOpen(false); // 气泡框
    setPodDelModalOpen(true); // 打开弹窗
  };

  const handleDeletePodCancel = () => {
    setPodDelModalOpen(false);
  };

  const handleDeletePodConfirm = async () => {
    try {
      const res = await deletePod(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setPodDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/workload/pod`);
        });
      }
    } catch (podDetailError) {
      if (podDetailError.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, podDetailError);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: podDetailError.response?.data?.message }));
      }
    }
  };

  const handlePodDetailCheckFn = (e) => {
    setIsPodDelCheck(e.target.checked);
  };

  const getPodDetailInfo = useCallback(async () => {
    if (name && namespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getPod(namespace, name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setPodYaml(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          // 备份一份annotations
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setPodDetailData(res.data);
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

  // 注解成功回调
  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsAnnotationModalOpen(false);
    } else {
      const podDetailAnnotationkeyArr = [];
      data.map(item => podDetailAnnotationkeyArr.push(item.key));
      if (podDetailAnnotationkeyArr.filter((item, index) => podDetailAnnotationkeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const resPodDetailAnnotation = await editAnnotationsOrLabels('pod', namespace, name, addAnnotationList);
          if (resPodDetailAnnotation.status === ResponseCode.OK) {
            showSuccessThen(messageApi, `${t('messages.editAnnotationSuccess', { ns: 'common' })}`, () => {
              getPodDetailInfo();
              setIsAnnotationModalOpen(false);
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
    setIsAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsAnnotationModalOpen(true);
    setPodPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsLabelModalOpen(true);
    setPodPopOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      const podDetailLabelkeyArr = [];
      data.map(item => podDetailLabelkeyArr.push(item.key));
      if (podDetailLabelkeyArr.filter((item, index) => podDetailLabelkeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const resPodDetailLabel = await editAnnotationsOrLabels('pod', namespace, name, addLabelList);
          if (resPodDetailLabel.status === ResponseCode.OK) {
            showSuccessThen(messageApi, `${t('messages.editLabelSuccess', { ns: 'common' })}`, () => {
              getPodDetailInfo();
              setIsLabelModalOpen(false);
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

  // 处理只读
  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  // 标签失败回调
  const handleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handleModify = () => {
    setPodDetailTabKey('yaml'); // 跳向yaml
    setIsReadyOnly(false); // 退出只读模式
  };

  useEffect(() => {
    if (podDetailTabKey === 'info' || activeKey) {
      getPodDetailInfo();
    }
  }, [podDetailTabKey, getPodDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <PodInfo
        podName={name}
        podNamespace={namespace}
        podDetailDataProps={podDetailData}
        refreshFn={getPodDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <PodYaml
        podYamlProps={podYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getPodDetailInfo} />,
    },
    {
      key: 'events',
      label: t('actions.events'),
      children: <PodEvents
        podName={name}
        podNamespace={namespace}
      />,
    },
  ];

  return <div className="child_content withBread_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/workload`, disabled: true },
      { title: t('pod.title'), path: '/pod' },
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
            <Button type="link" onClick={handleDeletePod}>{t('actions.delete')}</Button>
          </Space>
        }
        open={podPopOpen}
        onOpenChange={handlePodPopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal
        open={isAnnotationModalOpen}
        type="annotation"
        dataList={podDetailData?.metadata?.annotations}
        callbackOk={handleAnnotationOk}
        callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal
        open={isLabelModalOpen}
        type="label"
        dataList={podDetailData?.metadata?.labels}
        callbackOk={handleLabelOk}
        callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: 'Pod' })}
        open={podDelModalOpen}
        cancelFn={handleDeletePodCancel}
        content={[
          t('actions.deleteWarning', { resource: 'Pod' }),
          t('actions.deleteConfirmMessage', { resource: 'Pod', name }),
        ]}
        isCheck={isPodDelCheck}
        showCheck={true}
        checkFn={handlePodDetailCheckFn}
        confirmFn={handleDeletePodConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetPodDetailTabKey} activeKey={podDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}