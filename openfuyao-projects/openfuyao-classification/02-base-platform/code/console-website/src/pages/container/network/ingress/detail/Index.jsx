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
import IngressInfo from '@/pages/container/network/ingress/detail/IngressInfo';
import IngressYaml from '@/pages/container/network/ingress/detail/IngressYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useEffect, useState, useCallback, useRef, useStore } from 'openinula';
import { useParams, useHistory } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import { getIngressDetailDescription, deleteIngress, editIngressLabelOrAnnotation } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotation, jsonToYaml, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function IngressDetail() {
  const { t } = useTranslation(['network', 'common']);
  const themeStore = useStore('theme');
  const { name, namespace, activeKey } = useParams();

  const history = useHistory();

  const [messageApi, contextHolder] = message.useMessage();

  const [ingressTabDeleteModal, setIngressTabDeleteModal] = useState(false);

  const [isIngressDelCheck, setisIngressDelCheck] = useState(false);

  const [ingressDetailTabKey, setIngressDetailTabKey] = useState(activeKey || 'detail');

  const [ingressDetailData, setIngressDetailData] = useState({});

  const [ingressYamlData, setIngressYamlData] = useState('');

  const [ingressPopOpen, setIngressPopOpen] = useState(false);

  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);

  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);

  const [detailLoaded, setDetailLoded] = useState(false);

  const [isIngressReadyOnly, setIsIngressReadyOnly] = useState(true);

  const [oldAnnotations, setOldAnnotations] = useState([]);

  const [oldLabels, setOldLabels] = useState([]);

  const handleSetIngressDetailTabKey = (key) => {
    setIngressDetailTabKey(key);
    setIsIngressReadyOnly(true);
  };

  const handleIngressPopOpenChange = (open) => {
    setIngressPopOpen(open);
  };

  const handleEditIngressYaml = () => {
    setIngressDetailTabKey('yaml');
    setIsIngressReadyOnly(false);
  };

  const handleReadyOnly = (bool) => {
    setIsIngressReadyOnly(bool);
  };

  const handleDeleteIngress = () => {
    setIngressPopOpen(false);
    setIngressTabDeleteModal(true);
  };

  const handleDeleteIngressCancel = () => {
    setIngressTabDeleteModal(false);
  };

  const handleDeleteIngressConfirm = async () => {
    try {
      const res = await deleteIngress(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setIngressTabDeleteModal(false);
          history.push(`/${containerRouterPrefix}/network/ingress`);
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };

  const handleIngressCheckFn = (e) => {
    setisIngressDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getIngressDetailInfo = useCallback(async () => {
    if (namespace && name) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getIngressDetailDescription(namespace, name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setIngressYamlData(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setIngressDetailData(res.data);
        }
      } catch (error) {
        if (seq !== detailReqSeqRef.current) { return; }
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

  const handleIngressLabelOk = async (data) => {
    const ingressLabKeyArr = [];
    data.map(item => ingressLabKeyArr.push(item.key));
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      if (ingressLabKeyArr.filter((item, index) => ingressLabKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addIngressLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editIngressLabelOrAnnotation(namespace, name, addIngressLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              setIsLabelModalOpen(false);
              getIngressDetailInfo();
            });
          }
        } catch (error) {
          if (error.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: error.response?.data?.message }));
          }
        }
      }
    }
  };

  const handleIngressLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handleIngressAnnotationOk = async (data) => {
    const ingressAnnKeyArr = [];
    data.map(item => ingressAnnKeyArr.push(item.key));
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsAnnotationModalOpen(false);
    } else {
      if (ingressAnnKeyArr.filter((item, index) => ingressAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addIngressAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editIngressLabelOrAnnotation(namespace, name, addIngressAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              setIsAnnotationModalOpen(false);
              getIngressDetailInfo();
            });
          }
        } catch (error) {
          if (error.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: error.response?.data?.message }));
          }
        }
      }
    }
  };

  const handleIngressAnnotationCancel = () => {
    setIsAnnotationModalOpen(false);
  };

  const items = [
    {
      key: 'detail',
      label: t('actions.detail'),
      children: <IngressInfo ingressDetailDataProps={ingressDetailData} refreshFn={getIngressDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <IngressYaml
        ingressYamlProps={ingressYamlData}
        isIngressReadyOnly={isIngressReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getIngressDetailInfo} />,
    },
  ];

  useEffect(() => {
    if (ingressDetailTabKey === 'detail' || activeKey) {
      getIngressDetailInfo();
    }
  }, [ingressDetailTabKey, getIngressDetailInfo, activeKey]);

  useEffect(() => {
    if (activeKey) {
      setIsIngressReadyOnly(false);
    }
  }, [activeKey]);

  return <div className="child_content withBread_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/network`, disabled: true },
      { title: t('ingress.title'), path: `/${containerRouterPrefix}/network/ingress` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
      <div style={{ marginRight: '64px' }}>
        <h3>{name}</h3>
      </div>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleEditIngressYaml}>{t('actions.modify')}</Button>
            <Button type="link" onClick={() => setIsLabelModalOpen(true)}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={() => setIsAnnotationModalOpen(true)}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteIngress}>{t('actions.delete')}</Button>
          </Space>
        }
        open={ingressPopOpen}
        onOpenChange={handleIngressPopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isLabelModalOpen} type="label" dataList={ingressDetailData?.metadata?.labels} callbackOk={handleIngressLabelOk} callbackCancel={handleIngressLabelCancel} />}
      {detailLoaded && <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={ingressDetailData?.metadata?.annotations} callbackOk={handleIngressAnnotationOk} callbackCancel={handleIngressAnnotationCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('ingress.title') })}
        open={ingressTabDeleteModal}
        cancelFn={handleDeleteIngressCancel}
        content={[
          t('actions.deleteWarning', { resource: t('ingress.title') }),
          t('actions.deleteConfirmMessage', { resource: t('ingress.title'), name }),
        ]}
        isCheck={isIngressDelCheck}
        showCheck={true}
        checkFn={handleIngressCheckFn}
        confirmFn={handleDeleteIngressConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetIngressDetailTabKey} activeKey={ingressDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}
