/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix } from '@/constant.js';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import ResourceQuotaInfo from '@/pages/container/platformManage/resourceQuota/detail/ResourceQuotaInfo';
import ResourceQuotaYaml from '@/pages/container/platformManage/resourceQuota/detail/ResourceQuotaYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useRef, useStore } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getResourceQuota, deleteResourceQuota, editAnnotationsOrLabels } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, jsonToYaml, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function ResourceQuotaDetail() {
  const { t } = useTranslation(['namespace', 'common']);
  const { namespace, name, activeKey } = useParams();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [resourceQuotaDelModalOpen, setResourceQuotaDelModalOpen] = useState(false);
  const [isResourceQuotaDelCheck, setIsResourceQuotaDelCheck] = useState(false);
  const [detailLoaded, setDetailLoded] = useState(false);
  const [resourceQuotaDetailTabKey, setResourceQuotaDetailTabKey] = useState(activeKey || 'info');
  const [resourceQuotaDetailData, setResourceQuotaDetailData] = useState({});
  const [resourceQuotaPopOpen, setResourceQuotaPopOpen] = useState(false);
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const [oldAnnotations, setOldAnnotations] = useState();
  const [oldLabels, setOldLabels] = useState();
  const [resourceQuotaYaml, setResourceQuotaYaml] = useState('');
  const [isReadyOnly, setIsReadyOnly] = useState(true);

  const themeStore = useStore('theme');

  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handleResourceQuotaPopOpenChange = (open) => {
    setResourceQuotaPopOpen(open);
  };

  const handleSetResourceQuotaDetailTabKey = (key) => {
    setResourceQuotaDetailTabKey(key);
    setIsReadyOnly(true);
  };

  const handleDeleteResourceQuota = () => {
    setResourceQuotaPopOpen(false);
    setResourceQuotaDelModalOpen(true);
  };

  const handleDeleteResourceQuotaCancel = () => {
    setResourceQuotaDelModalOpen(false);
  };

  const handleDelResourceQuotaConfirm = async () => {
    try {
      const res = await deleteResourceQuota(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setResourceQuotaDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/namespace/resourceQuota`);
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

  const handleResourceQuotaCheckFn = (e) => {
    setIsResourceQuotaDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getResourceQuotaDetailInfo = useCallback(async () => {
    if (name && namespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getResourceQuota(namespace, name);
        if (seq !== detailReqSeqRef.current) {
          return;
        }
        if (res.status === ResponseCode.OK) {
          setResourceQuotaYaml(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setResourceQuotaDetailData(res.data);
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

  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      data.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('resourcequota', namespace, name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getResourceQuotaDetailInfo();
              setIsAnnotationModalOpen(false);
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
    setIsAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsAnnotationModalOpen(true);
    setResourceQuotaPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsLabelModalOpen(true);
    setResourceQuotaPopOpen(false);
  };

  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      const keyArr = [];
      data.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('resourcequota', namespace, name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getResourceQuotaDetailInfo();
              setIsLabelModalOpen(false);
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
    setIsLabelModalOpen(false);
  };

  const handleModify = () => {
    setResourceQuotaDetailTabKey('yaml');
    setIsReadyOnly(false);
  };

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <ResourceQuotaInfo
        resourceQuotaName={name}
        resourceQuotaNamespace={namespace}
        resourceQuotaDetailDataProps={resourceQuotaDetailData}
        refreshFn={getResourceQuotaDetailInfo}
      />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <ResourceQuotaYaml
        resourceQuotaYamlProps={resourceQuotaYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getResourceQuotaDetailInfo}
      />,
    },
  ];

  const isInitialMountRef = useRef(true);

  useEffect(() => {
    if (resourceQuotaDetailTabKey === 'info' || (activeKey && isInitialMountRef.current)) {
      getResourceQuotaDetailInfo();
    }
    isInitialMountRef.current = false;
  }, [resourceQuotaDetailTabKey, getResourceQuotaDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  return (
    <div className="child_content withBread_content">
      <ToastMsg contextHolder={contextHolder} />
      <BreadCrumbCom items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/namespace/resourceQuota`, disabled: true },
        { title: t('resourceQuota.title'), path: `/${containerRouterPrefix}/namespace/resourceQuota` },
        { title: t('actions.detail'), path: '/detail' },
      ]}
      />
      <div className="pod_title" style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
        <h3>{name}</h3>
        <Popover placement="bottom"
          content={
            <Space className="column_pop">
              <Button type="link" onClick={handleModify}>{t('actions.modify')}</Button>
              <Button type="link" onClick={handleEditLabel}>{t('actions.editLabel')}</Button>
              <Button type="link" onClick={handleEditAnnotation}>{t('actions.editAnnotation')}</Button>
              <Button type="link" onClick={handleDeleteResourceQuota}>{t('actions.delete')}</Button>
            </Space>
          }
          open={resourceQuotaPopOpen}
          onOpenChange={handleResourceQuotaPopOpenChange}>
          <Button className="primary_btn">{t('actions.operation')} <DownOutlined className="small_margin_adjust" /></Button>
        </Popover>
        {detailLoaded && <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={resourceQuotaDetailData?.metadata.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
        {detailLoaded && <AnnotationModal open={isLabelModalOpen} type="label" dataList={resourceQuotaDetailData?.metadata.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
        <DeleteInfoModal
          title={t('actions.deleteConfirm', { resource: t('resourceQuota.title') })}
          open={resourceQuotaDelModalOpen}
          cancelFn={handleDeleteResourceQuotaCancel}
          content={[
            t('actions.deleteWarning', { resource: t('resourceQuota.title') }),
            t('actions.deleteConfirmMessage', { resource: t('resourceQuota.title'), name }),
          ]}
          isCheck={isResourceQuotaDelCheck}
          showCheck={true}
          checkFn={handleResourceQuotaCheckFn}
          confirmFn={handleDelResourceQuotaConfirm} />
      </div>
      {detailLoaded && <Tabs items={items} onChange={handleSetResourceQuotaDetailTabKey} activeKey={resourceQuotaDetailTabKey} destroyInactiveTabPane={true} />}
    </div>
  );
}
