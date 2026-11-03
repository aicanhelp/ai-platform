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
import ServiceInfo from '@/pages/container/network/service/detail/ServiceInfo';
import ServiceYaml from '@/pages/container/network/service/detail/ServiceYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useEffect, useState, useCallback, useRef, useStore } from 'openinula';
import { useParams, useHistory } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import { getServiceDetailDescription, deleteService, editServiceLabelOrAnnotation } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotation, jsonToYaml, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function ServiceDetail() {
  const { t } = useTranslation(['network', 'common']);
  const { name, namespace, activeKey } = useParams();
  const themeStore = useStore('theme');
  const history = useHistory();

  const [messageApi, contextHolder] = message.useMessage();

  const [serviceTabDeleteModal, setServiceTabDeleteModal] = useState(false);

  const [isServiceDelCheck, setisServiceDelCheck] = useState(false);

  const [serviceDetailTabKey, setServiceDetailTabKey] = useState(activeKey || 'detail');

  const [serviceDetailData, setServiceDetailData] = useState({});

  const [serviceYamlData, setServiceYamlData] = useState('');

  const [servicePopOpen, setServicePopOpen] = useState(false);

  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);

  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);

  const [detailLoaded, setDetailLoded] = useState(false);

  const [isServiceReadyOnly, setIsServiceReadyOnly] = useState(true);

  const [oldAnnotations, setOldAnnotations] = useState([]);

  const [oldLabels, setOldLabels] = useState([]);

  const handleSetServiceDetailTabKey = (key) => {
    setServiceDetailTabKey(key);
    setIsServiceReadyOnly(true);
  };

  const handleServicePopOpenChange = (open) => {
    setServicePopOpen(open);
  };

  const handleEditServiceYaml = () => {
    setServiceDetailTabKey('yaml');
    setIsServiceReadyOnly(false);
  };

  const handleReadyOnly = (bool) => {
    setIsServiceReadyOnly(bool);
  };

  const handleDeleteService = () => {
    setServicePopOpen(false);
    setServiceTabDeleteModal(true);
  };

  const handleDeleteServiceCancel = () => {
    setServiceTabDeleteModal(false);
  };

  const handleDeleteServiceConfirm = async () => {
    try {
      const res = await deleteService(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setServiceTabDeleteModal(false);
          history.push(`/${containerRouterPrefix}/network/service`);
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

  const handleServiceCheckFn = (e) => {
    setisServiceDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getServiceDetailInfo = useCallback(async () => {
    if (namespace && name) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getServiceDetailDescription(namespace, name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setServiceYamlData(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setServiceDetailData(res.data);
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

  const handleServiceLabelOk = async (data) => {
    const serviceLabKeyArr = [];
    data.map(item => serviceLabKeyArr.push(item.key));
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      if (serviceLabKeyArr.filter((item, index) => serviceLabKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addServiceLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editServiceLabelOrAnnotation(namespace, name, addServiceLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              setIsLabelModalOpen(false);
              getServiceDetailInfo();
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

  const handleServiceLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handleServiceAnnotationOk = async (data) => {
    const serviceAnnKeyArr = [];
    data.map(item => serviceAnnKeyArr.push(item.key));
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsAnnotationModalOpen(false);
    } else {
      if (serviceAnnKeyArr.filter((item, index) => serviceAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addServiceAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editServiceLabelOrAnnotation(namespace, name, addServiceAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              setIsAnnotationModalOpen(false);
              getServiceDetailInfo();
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

  const handleServiceAnnotationCancel = () => {
    setIsAnnotationModalOpen(false);
  };

  const items = [
    {
      key: 'detail',
      label: t('actions.detail'),
      children: <ServiceInfo serviceDetailDataProps={serviceDetailData} refreshFn={getServiceDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <ServiceYaml
        serviceYamlProps={serviceYamlData}
        isServiceReadyOnly={isServiceReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getServiceDetailInfo} />,
    },
  ];

  useEffect(() => {
    if (serviceDetailTabKey === 'detail' || activeKey) {
      getServiceDetailInfo();
    }
  }, [serviceDetailTabKey, getServiceDetailInfo, activeKey]);

  useEffect(() => {
    if (activeKey) {
      setIsServiceReadyOnly(false);
    }
  }, [activeKey]);

  return <div className="child_content withBread_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/network`, disabled: true },
      { title: t('service.title'), path: `/${containerRouterPrefix}/network/service` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
      <div style={{ marginRight: '64px' }}>
        <h3>{name}</h3>
      </div>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleEditServiceYaml}>{t('actions.modify')}</Button>
            <Button type="link" onClick={() => setIsLabelModalOpen(true)}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={() => setIsAnnotationModalOpen(true)}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteService}>{t('actions.delete')}</Button>
          </Space>
        }
        open={servicePopOpen}
        onOpenChange={handleServicePopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isLabelModalOpen} type="label" dataList={serviceDetailData?.metadata?.labels} callbackOk={handleServiceLabelOk} callbackCancel={handleServiceLabelCancel} />}
      {detailLoaded && <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={serviceDetailData?.metadata?.annotations} callbackOk={handleServiceAnnotationOk} callbackCancel={handleServiceAnnotationCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('service.title') })}
        open={serviceTabDeleteModal}
        cancelFn={handleDeleteServiceCancel}
        content={[
          t('actions.deleteWarning', { resource: t('service.title') }),
          t('actions.deleteConfirmMessage', { resource: t('service.title'), name }),
        ]}
        isCheck={isServiceDelCheck}
        showCheck={true}
        checkFn={handleServiceCheckFn}
        confirmFn={handleDeleteServiceConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetServiceDetailTabKey} activeKey={serviceDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}
