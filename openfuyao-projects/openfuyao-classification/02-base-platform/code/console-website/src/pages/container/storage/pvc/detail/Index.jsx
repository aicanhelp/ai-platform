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
import PvcInfo from '@/pages/container/storage/pvc/detail/PvcInfo';
import PvcYaml from '@/pages/container/storage/pvc/detail/PvcYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useEffect, useState, useCallback, useRef, useStore } from 'openinula';
import { useParams, useHistory } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getPvcDetailDescription, deletePvc, editPvcLabelOrAnnotation } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotation, jsonToYaml, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function PvcDetail() {
  const { t } = useTranslation(['storage', 'common']);
  const { name, namespace, activeKey } = useParams();

  const history = useHistory();

  const [messageApi, contextHolder] = message.useMessage();

  const themeStore = useStore('theme');

  const [pvcTabDeleteModal, setPvcTabDeleteModal] = useState(false);

  const [isPvcDelCheck, setisPvcDelCheck] = useState(false);

  const [pvcDetailTabKey, setPvcDetailTabKey] = useState(activeKey || 'info');

  const [pvcDetailData, setPvcDetailData] = useState({});

  const [pvcYamlData, setPvcYamlData] = useState('');

  const [pvcPopOpen, setPvcPopOpen] = useState(false);

  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);

  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);

  const [detailLoaded, setDetailLoded] = useState(false);

  const [isPvcReadyOnly, setIsPvcReadyOnly] = useState(true);

  const [oldAnnotations, setOldAnnotations] = useState([]);

  const [oldLabels, setOldLabels] = useState([]);

  const handleSetPvcDetailTabKey = (key) => {
    setPvcDetailTabKey(key);
    setIsPvcReadyOnly(true);
  };

  const handlePvcPopOpenChange = (open) => {
    setPvcPopOpen(open);
  };

  const handleEditPvcYaml = () => {
    setPvcDetailTabKey('yaml');
    setIsPvcReadyOnly(false);
  };

  const handleReadyOnly = (bool) => {
    setIsPvcReadyOnly(bool);
  };

  const handleDeletePvc = () => {
    setPvcPopOpen(false);
    setPvcTabDeleteModal(true);
  };

  const handleDeletePvcCancel = () => {
    setPvcTabDeleteModal(false);
  };

  const handleDeletePvcConfirm = async () => {
    try {
      const res = await deletePvc(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setPvcTabDeleteModal(false);
          history.push(`/${containerRouterPrefix}/storage/pvc`);
        });
      }
    } catch (pvcError) {
      if (pvcError.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, pvcError);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: pvcError.response?.data?.message }));
      }
    }
  };

  const handlePvcCheckFn = (e) => {
    setisPvcDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getPvcDetailInfo = useCallback(async () => {
    if (namespace && name) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getPvcDetailDescription(namespace, name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setPvcYamlData(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setPvcDetailData(res.data);
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

  const handleLabelOk = async (data) => {
    const keyArr = [];
    data.map(item => keyArr.push(item.key));
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editPvcLabelOrAnnotation(namespace, name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              setIsLabelModalOpen(false);
              getPvcDetailInfo();
            });
          }
        } catch (e) {
          if (e.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, e);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: e.response?.data?.message }));
          }
        }
      }
    }
  };

  const handleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handleAnnotationOk = async (data) => {
    const keyArr = [];
    data.map(item => keyArr.push(item.key));
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsAnnotationModalOpen(false);
    } else {
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editPvcLabelOrAnnotation(namespace, name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              setIsAnnotationModalOpen(false);
              getPvcDetailInfo();
            });
          }
        } catch (pvce) {
          if (pvce.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, pvce);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: pvce.response?.data?.message }));
          }
        }
      }
    }
  };

  const handleAnnotationCancel = () => {
    setIsAnnotationModalOpen(false);
  };

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <PvcInfo pvcDetailDataProps={pvcDetailData} refreshFn={getPvcDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <PvcYaml
        pvcYamlProps={pvcYamlData}
        isPvcReadyOnly={isPvcReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getPvcDetailInfo} />,
    },
  ];

  useEffect(() => {
    getPvcDetailInfo();
  }, [getPvcDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsPvcReadyOnly(false);
    }
  }, [activeKey]);

  return <div className="child_content withBread_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title') },
      { title: t('pvc.title'), path: `/${containerRouterPrefix}/storage/pvc` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' ? 'none' : undefined, backgroundColor: themeStore.$s.theme !== 'light' ? '#2a2d34' : undefined }}>
      <div style={{ marginRight: '64px' }}>
        <h3>{name}</h3>
      </div>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleEditPvcYaml}>{t('actions.modify')}</Button>
            <Button type="link" onClick={() => setIsLabelModalOpen(true)}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={() => setIsAnnotationModalOpen(true)}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeletePvc}>{t('actions.delete')}</Button>
          </Space>
        }
        open={pvcPopOpen}
        onOpenChange={handlePvcPopOpenChange}>
        <Button className='primary_btn detail_title_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isLabelModalOpen} type="label" dataList={pvcDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      {detailLoaded && <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={pvcDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('pvc.title') })}
        open={pvcTabDeleteModal}
        cancelFn={handleDeletePvcCancel}
        content={[
          t('actions.deleteWarning', { resource: t('pvc.title') }),
          t('actions.deleteConfirmMessage', { resource: t('pvc.title'), name }),
        ]}
        isCheck={isPvcDelCheck}
        showCheck={true}
        checkFn={handlePvcCheckFn}
        confirmFn={handleDeletePvcConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetPvcDetailTabKey} activeKey={pvcDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}
