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
import NamespaceInfo from '@/pages/container/platformManage/namespace/detail/NamespaceInfo';
import NamespaceYaml from '@/pages/container/platformManage/namespace/detail/NamespaceYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useRef, useStore } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getNamespace, deleteNamespace, editAnnotationsOrLabels } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotation, jsonToYaml, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function NamespaceDetail() {
  const { t } = useTranslation(['namespace', 'common']);
  const { name, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [namespaceDelModalOpen, setNamespaceDelModalOpen] = useState(false);
  const [isNamespaceDelCheck, setIsNamespaceDelCheck] = useState(false);
  const [detailLoaded, setDetailLoded] = useState(false);

  const [namespaceDetailTabKey, setNamespaceDetailTabKey] = useState(activeKey || 'info');

  const [namespaceDetailData, setNamespaceDetailData] = useState({});

  const [namespacePopOpen, setNamespacePopOpen] = useState(false);
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const [isReadyOnly, setIsReadyOnly] = useState(true);
  const [oldAnnotations, setOldAnnotations] = useState();
  const [oldLabels, setOldLabels] = useState();
  const [namespaceYaml, setNamespaceYaml] = useState('');
  const themeStore = useStore('theme');

  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handleNamespacePopOpenChange = (open) => {
    setNamespacePopOpen(open);
  };

  const handleSetNamespaceDetailTabKey = (key) => {
    setNamespaceDetailTabKey(key);
    setIsReadyOnly(true);
  };

  const handleDeleteNamespace = () => {
    setNamespacePopOpen(false);
    setNamespaceDelModalOpen(true);
  };

  const handleDeleteNamespaceCancel = () => {
    setNamespaceDelModalOpen(false);
  };

  const handleDelNamespaceConfirm = async () => {
    try {
      const res = await deleteNamespace(name);
      if (res.status === ResponseCode.OK) {
        setNamespaceDelModalOpen(false);
        showSuccessThen(messageApi, t('messages.deleteSuccess'), () => {
          history.push(`/${containerRouterPrefix}/namespace/namespaceManage`);
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

  const handleNamespaceCheckFn = (e) => {
    setIsNamespaceDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getNamespaceDetailInfo = useCallback(async () => {
    if (name) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getNamespace(name);
        if (seq !== detailReqSeqRef.current) {
          return;
        }
        if (res.status === ResponseCode.OK) {
          setNamespaceYaml(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setNamespaceDetailData(res.data);
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
  }, [name]);

  const handleNamespaceDetailAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsAnnotationModalOpen(false);
    } else {
      const namespaceDetailAnnKeyArr = [];
      data.map(item => namespaceDetailAnnKeyArr.push(item.key));
      if (namespaceDetailAnnKeyArr.filter((item, index) => namespaceDetailAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addNamespaceAnnList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('namespace', '', name, addNamespaceAnnList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getNamespaceDetailInfo();
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

  const handleNamespaceDetailAnnotationCancel = () => {
    setIsAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsAnnotationModalOpen(true);
    setNamespacePopOpen(false);
  };

  const handleEditLabel = () => {
    setIsLabelModalOpen(true);
    setNamespacePopOpen(false);
  };

  const handleNamespaceDetailLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handleNamespaceDetailLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      const namespaceDetailLabKeyArr = [];
      data.map(item => namespaceDetailLabKeyArr.push(item.key));
      if (namespaceDetailLabKeyArr.filter((item, index) => namespaceDetailLabKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addNamespaceLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('namespace', '', name, addNamespaceLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getNamespaceDetailInfo();
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

  const handleModify = () => {
    setNamespaceDetailTabKey('yaml');
    setIsReadyOnly(false);
  };

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <NamespaceInfo
        namespaceName={name}
        namespaceDetailDataProps={namespaceDetailData}
        refreshFn={getNamespaceDetailInfo}
      />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <NamespaceYaml
        namespaceYamlProps={namespaceYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getNamespaceDetailInfo}
      />,
    },
  ];

  const isInitialMountRef = useRef(true);

  useEffect(() => {
    if (namespaceDetailTabKey === 'info' || (activeKey && isInitialMountRef.current)) {
      getNamespaceDetailInfo();
    }
    isInitialMountRef.current = false;
  }, [namespaceDetailTabKey, getNamespaceDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  return (
    <div className="child_content withBread_content">
      <ToastMsg contextHolder={contextHolder} />
      <BreadCrumbCom items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/namespace/namespaceManage`, disabled: true },
        { title: t('namespace.title'), path: `/${containerRouterPrefix}/namespace/namespaceManage` },
        { title: t('actions.detail'), path: '/detail' },
      ]}
      />
      <div className="pod_title" style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
        <h3>{name}</h3>
        <Popover placement='bottom'
          content={
            <Space className='column_pop'>
              <Button type='link' onClick={handleModify}>{t('actions.modify')}</Button>
              <Button type='link' onClick={handleEditLabel}>{t('actions.editLabel')}</Button>
              <Button type='link' onClick={handleEditAnnotation}>{t('actions.editAnnotation')}</Button>
              <Button type='link' onClick={handleDeleteNamespace}>{t('actions.delete')}</Button>
            </Space>
          }
          open={namespacePopOpen}
          onOpenChange={handleNamespacePopOpenChange}>
          <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
        </Popover>
        {detailLoaded && <AnnotationModal open={isAnnotationModalOpen} type='annotation' dataList={namespaceDetailData?.metadata.annotations} callbackOk={handleNamespaceDetailAnnotationOk} callbackCancel={handleNamespaceDetailAnnotationCancel} />}
        {detailLoaded && <AnnotationModal open={isLabelModalOpen} type='label' dataList={namespaceDetailData?.metadata.labels} callbackOk={handleNamespaceDetailLabelOk} callbackCancel={handleNamespaceDetailLabelCancel} />}
        <DeleteInfoModal
          title={t('actions.deleteConfirm', { resource: t('namespace.title') })}
          open={namespaceDelModalOpen}
          cancelFn={handleDeleteNamespaceCancel}
          content={[
            t('actions.deleteNamespaceWarning', { resource: t('namespace.title') }),
            t('actions.deleteConfirmMessage', { resource: t('namespace.title'), name }),
          ]}
          isCheck={isNamespaceDelCheck}
          showCheck={true}
          checkFn={handleNamespaceCheckFn}
          confirmFn={handleDelNamespaceConfirm} />
      </div>
      {detailLoaded && <Tabs items={items} onChange={handleSetNamespaceDetailTabKey} activeKey={namespaceDetailTabKey} destroyInactiveTabPane={true} />}
    </div>
  );
}
