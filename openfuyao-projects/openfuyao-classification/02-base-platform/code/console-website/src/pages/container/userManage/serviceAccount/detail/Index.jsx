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
import ServiceAccountInfo from '@/pages/container/userManage/serviceAccount/detail/ServiceAccountInfo';
import ServiceAccountYaml from '@/pages/container/userManage/serviceAccount/detail/ServiceAccountYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useRef, useStore } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getServiceAccountDetailDescription, editAnnotationsOrLabels, deleteServiceAccount } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function ServiceAccountDetail() {
  const { t } = useTranslation(['rbac', 'common']);
  const { name, namespace, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [serviceAccountDelModalOpen, setServiceAccountDelModalOpen] = useState(false);
  const [isServiceAccountDelCheck, setIsServiceAccountDelCheck] = useState(false);
  const [detailLoaded, setDetailLoded] = useState(false);

  const [serviceAccountDetailTabKey, setServiceAccountDetailTabKey] = useState(activeKey || 'info');

  const [serviceAccountDetailData, setServiceAccountDetailData] = useState({});

  const [serviceAccountPopOpen, setServiceAccountPopOpen] = useState(false);

  const [isReadyOnly, setIsReadyOnly] = useState(true);

  const [isServiceAccountAnnotationModalOpen, setIsServiceAccountAnnotationModalOpen] = useState(false);
  const [isServiceAccountLabelModalOpen, setIsServiceAccountLabelModalOpen] = useState(false);
  const [oldServiceAccountAnnotations, setOldServiceAccountAnnotations] = useState();
  const [oldServiceAccountLabels, setOldServiceAccountLabels] = useState();
  const themeStore = useStore('theme');

  const [serviceAccountYaml, setServiceAccountYaml] = useState('');

  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handleServiceAccountPopOpenChange = (open) => {
    setServiceAccountPopOpen(open);
  };

  const handleSetServiceAccountDetailTabKey = (key) => {
    setServiceAccountDetailTabKey(key);
    setIsReadyOnly(true);
  };

  const handleDeleteServiceAccount = () => {
    setServiceAccountPopOpen(false);
    setServiceAccountDelModalOpen(true);
  };

  const handleDeleteServiceAccountCancel = () => {
    setServiceAccountDelModalOpen(false);
  };

  const handleDeleteServiceAccountConfirm = async () => {
    try {
      const res = await deleteServiceAccount(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setServiceAccountDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/userManage/serviceAccount`);
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

  const handleServiceAccountDetailCheckFn = (e) => {
    setIsServiceAccountDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getServiceAccountDetailInfo = useCallback(async () => {
    if (name && namespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getServiceAccountDetailDescription(namespace, name);
        if (res.status === ResponseCode.OK) {
          setServiceAccountYaml(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldServiceAccountAnnotations([...res.data.metadata.annotations]);
          setOldServiceAccountLabels([...res.data.metadata.labels]);
          setServiceAccountDetailData(res.data);
        }
      } catch (error) {
        if (seq !== detailReqSeqRef.current) {
          return;
        }
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
    setServiceAccountDetailTabKey('yaml');
    setIsReadyOnly(false);
  };

  const handleAnnotationOk = async (serviceAccountData) => {
    if (JSON.stringify(oldServiceAccountAnnotations) === JSON.stringify(serviceAccountData)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsServiceAccountAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      serviceAccountData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldServiceAccountAnnotations, serviceAccountData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('serviceaccount', namespace, name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getServiceAccountDetailInfo();
              setIsServiceAccountAnnotationModalOpen(false);
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

  const handleAnnotationCancel = () => {
    setIsServiceAccountAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsServiceAccountAnnotationModalOpen(true);
    setServiceAccountPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsServiceAccountLabelModalOpen(true);
    setServiceAccountPopOpen(false);
  };

  const handleLabelOk = async (serviceAccountData) => {
    if (JSON.stringify(oldServiceAccountLabels) === JSON.stringify(serviceAccountData)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsServiceAccountLabelModalOpen(false);
    } else {
      const keyArr = [];
      serviceAccountData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldServiceAccountLabels, serviceAccountData, 'label');
        try {
          const res = await editAnnotationsOrLabels('serviceaccount', namespace, name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getServiceAccountDetailInfo();
              setIsServiceAccountLabelModalOpen(false);
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

  const handleLabelCancel = () => {
    setIsServiceAccountLabelModalOpen(false);
  };

  const isInitialMountRef = useRef(true);

  useEffect(() => {
    if (serviceAccountDetailTabKey === 'info' || (activeKey && isInitialMountRef.current)) {
      getServiceAccountDetailInfo();
    }
    isInitialMountRef.current = false;
  }, [serviceAccountDetailTabKey, getServiceAccountDetailInfo, activeKey]);

  useEffect(() => {
    if (activeKey === 'yaml') {
      setServiceAccountDetailTabKey('yaml');
      setIsReadyOnly(false);
    } else if (activeKey === 'info') {
      setServiceAccountDetailTabKey('info');
      setIsReadyOnly(true);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <ServiceAccountInfo
        serviceAccountName={name}
        serviceAccountNamespace={namespace}
        serviceAccountDetailDataProps={serviceAccountDetailData}
        refreshFn={getServiceAccountDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <ServiceAccountYaml
        serviceAccountYamlProps={serviceAccountYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getServiceAccountDetailInfo} />,
    },
  ];

  return <div className="child_content withBread_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title'), disabled: true },
      { title: t('serviceAccount.title'), path: `/${containerRouterPrefix}/userManage/serviceAccount` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
      <h3>{name}</h3>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleModify}>{t('actions.modify')}</Button>
            <Button type="link" onClick={handleEditLabel}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={handleEditAnnotation}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteServiceAccount}>{t('actions.delete')}</Button>
          </Space>
        }
        open={serviceAccountPopOpen}
        onOpenChange={handleServiceAccountPopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isServiceAccountAnnotationModalOpen} type="annotation" dataList={serviceAccountDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isServiceAccountLabelModalOpen} type="label" dataList={serviceAccountDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('serviceAccount.title') })}
        open={serviceAccountDelModalOpen}
        cancelFn={handleDeleteServiceAccountCancel}
        content={[
          t('actions.deleteWarning', { resource: t('serviceAccount.title') }),
          t('actions.deleteConfirmMessage', { resource: t('serviceAccount.title'), name }),
        ]}
        isCheck={isServiceAccountDelCheck}
        showCheck={true}
        checkFn={handleServiceAccountDetailCheckFn}
        confirmFn={handleDeleteServiceAccountConfirm} />
    </div>
    {detailLoaded && <Tabs
      items={items}
      onChange={handleSetServiceAccountDetailTabKey}
      activeKey={serviceAccountDetailTabKey}
      destroyInactiveTabPane={true}>
    </Tabs>
    }
  </div>;
}
