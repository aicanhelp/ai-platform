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
import PvInfo from '@/pages/container/storage/pv/detail/PvInfo';
import PvYaml from '@/pages/container/storage/pv/detail/PvYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useEffect, useState, useCallback, useRef, useStore } from 'openinula';
import { useParams, useHistory } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getPvDetailDescription, deletePv, editPvLabelOrAnnotation } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotation, jsonToYaml, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function PvDetail() {
  const { t } = useTranslation(['storage', 'common']);
  const { name, activeKey } = useParams();

  const history = useHistory();

  const [messageApi, contextHolder] = message.useMessage();

  const themeStore = useStore('theme');

  const [pvTabDeleteModal, setPvTabDeleteModal] = useState(false);

  const [isPvDelCheck, setisPvDelCheck] = useState(false);

  const [pvDetailTabKey, setPvDetailTabKey] = useState(activeKey || 'info');

  const [pvDetailData, setPvDetailData] = useState({});

  const [pvYamlData, setPvYamlData] = useState('');

  const [pvPopOpen, setPvPopOpen] = useState(false);

  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);

  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);

  const [detailLoaded, setDetailLoded] = useState(false);

  const [isPvReadyOnly, setIsPvReadyOnly] = useState(true);

  const [oldAnnotations, setOldAnnotations] = useState([]);

  const [oldLabels, setOldLabels] = useState([]);

  const handleSetPvDetailTabKey = (key) => {
    setPvDetailTabKey(key);
    setIsPvReadyOnly(true);
  };

  const handlePvPopOpenChange = (open) => {
    setPvPopOpen(open);
  };

  const handleEditPvYaml = () => {
    setPvDetailTabKey('yaml');
    setIsPvReadyOnly(false);
  };

  const handleReadyOnly = (bool) => {
    setIsPvReadyOnly(bool);
  };

  const handleDeletePv = () => {
    setPvPopOpen(false);
    setPvTabDeleteModal(true);
  };

  const handleDeletePvCancel = () => {
    setPvTabDeleteModal(false);
  };

  const handleDeletePvConfirm = async () => {
    try {
      const res = await deletePv(name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setPvTabDeleteModal(false);
          history.push(`/${containerRouterPrefix}/storage/pv`);
        });
      }
    } catch (e) {
      if (e.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, e);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: e.response?.data?.message }));
      }
    }
  };

  const handlePvCheckFn = (e) => {
    setisPvDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getPvDetailInfo = useCallback(async () => {
    if (name) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getPvDetailDescription(name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setPvYamlData(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setPvDetailData(res.data);
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
  }, [name]);

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
          const res = await editPvLabelOrAnnotation(name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              setIsLabelModalOpen(false);
              getPvDetailInfo();
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
          const res = await editPvLabelOrAnnotation(name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              setIsAnnotationModalOpen(false);
              getPvDetailInfo();
            });
          }
        } catch (e) {
          if (e.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, e);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: e.response?.data?.message }));
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
      children: <PvInfo pvDetailDataProps={pvDetailData} refreshFn={getPvDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <PvYaml
        pvYamlProps={pvYamlData}
        isPvReadyOnly={isPvReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getPvDetailInfo} />,
    },
  ];

  useEffect(() => {
    getPvDetailInfo();
  }, [getPvDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsPvReadyOnly(false);
    }
  }, [activeKey]);

  return <div className="child_content withBread_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title') },
      { title: t('pv.title'), path: `/${containerRouterPrefix}/storage/pv` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' ? 'none' : undefined, backgroundColor: themeStore.$s.theme !== 'light' ? '#2a2d34' : undefined }}>
      <div style={{ marginRight: '64px' }}>
        <h3>{name}</h3>
      </div>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleEditPvYaml}>{t('actions.modify')}</Button>
            <Button type="link" onClick={() => setIsLabelModalOpen(true)}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={() => setIsAnnotationModalOpen(true)}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeletePv}>{t('actions.delete')}</Button>
          </Space>
        }
        open={pvPopOpen}
        onOpenChange={handlePvPopOpenChange}>
        <Button className='primary_btn detail_title_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isLabelModalOpen} type="label" dataList={pvDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      {detailLoaded && <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={pvDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('pv.title') })}
        open={pvTabDeleteModal}
        cancelFn={handleDeletePvCancel}
        content={[
          t('actions.deleteWarning', { resource: t('pv.title') }),
          t('actions.deleteConfirmMessage', { resource: t('pv.title'), name }),
        ]}
        isCheck={isPvDelCheck}
        showCheck={true}
        checkFn={handlePvCheckFn}
        confirmFn={handleDeletePvConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetPvDetailTabKey} activeKey={pvDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}
