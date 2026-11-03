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
import ScInfo from '@/pages/container/storage/sc/detail/ScInfo';
import ScYaml from '@/pages/container/storage/sc/detail/ScYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useEffect, useState, useCallback, useRef, useStore } from 'openinula';
import { useParams, useHistory } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getScDetailDescription, deleteSc, editScLabelOrAnnotation } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotation, jsonToYaml, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function ScDetail() {
  const { t } = useTranslation(['storage', 'common']);
  const { name, activeKey } = useParams();

  const history = useHistory();

  const [messageApi, contextHolder] = message.useMessage();

  const themeStore = useStore('theme');

  const [scTabDeleteModal, setScTabDeleteModal] = useState(false);

  const [isScDelCheck, setisScDelCheck] = useState(false);

  const [scDetailTabKey, setScDetailTabKey] = useState(activeKey || 'info');

  const [scDetailData, setScDetailData] = useState({});

  const [scYamlData, setScYamlData] = useState('');

  const [scPopOpen, setScPopOpen] = useState(false);

  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);

  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);

  const [detailLoaded, setDetailLoded] = useState(false);

  const [isScReadyOnly, setIsScReadyOnly] = useState(true);

  const [oldAnnotations, setOldAnnotations] = useState([]);

  const [oldLabels, setOldLabels] = useState([]);

  const handleSetScDetailTabKey = (key) => {
    setScDetailTabKey(key);
    setIsScReadyOnly(true);
  };

  const handleScPopOpenChange = (open) => {
    setScPopOpen(open);
  };

  const handleEditScYaml = () => {
    setScDetailTabKey('yaml');
    setIsScReadyOnly(false);
  };

  const handleReadyOnly = (bool) => {
    setIsScReadyOnly(bool);
  };

  const handleDeleteSc = () => {
    setScPopOpen(false);
    setScTabDeleteModal(true);
  };

  const handleDeleteScCancel = () => {
    setScTabDeleteModal(false);
  };

  const handleDeleteScConfirm = async () => {
    try {
      const res = await deleteSc(name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setScTabDeleteModal(false);
          history.push(`/${containerRouterPrefix}/storage/sc`);
        });
      }
    } catch (scError) {
      if (scError.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, scError);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: scError.response?.data?.message }));
      }
    }
  };

  const handleScCheckFn = (e) => {
    setisScDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getScDetailInfo = useCallback(async () => {
    if (name) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getScDetailDescription(name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setScYamlData(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setScDetailData(res.data);
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
          const res = await editScLabelOrAnnotation(name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              setIsLabelModalOpen(false);
              getScDetailInfo();
            });
          }
        } catch (scError) {
          if (scError.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, scError);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: scError.response?.data?.message }));
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
          const res = await editScLabelOrAnnotation(name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              setIsAnnotationModalOpen(false);
              getScDetailInfo();
            });
          }
        } catch (scError) {
          if (scError.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, scError);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: scError.response?.data?.message }));
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
      children: <ScInfo scDetailDataProps={scDetailData} refreshFn={getScDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <ScYaml
        scYamlProps={scYamlData}
        isScReadyOnly={isScReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getScDetailInfo} />,
    },
  ];

  useEffect(() => {
    getScDetailInfo();
  }, [getScDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsScReadyOnly(false);
    }
  }, [activeKey]);

  return <div className="child_content withBread_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title') },
      { title: t('sc.title'), path: `/${containerRouterPrefix}/storage/sc` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' ? 'none' : undefined, backgroundColor: themeStore.$s.theme !== 'light' ? '#2a2d34' : undefined }}>
      <div style={{ marginRight: '64px' }}>
        <h3>{name}</h3>
      </div>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleEditScYaml}>{t('actions.modify')}</Button>
            <Button type="link" onClick={() => setIsLabelModalOpen(true)}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={() => setIsAnnotationModalOpen(true)}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteSc}>{t('actions.delete')}</Button>
          </Space>
        }
        open={scPopOpen}
        onOpenChange={handleScPopOpenChange}>
        <Button className='primary_btn detail_title_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isLabelModalOpen} type="label" dataList={scDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      {detailLoaded && <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={scDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('sc.title') })}
        open={scTabDeleteModal}
        cancelFn={handleDeleteScCancel}
        content={[
          t('actions.deleteWarning', { resource: t('sc.title') }),
          t('actions.deleteConfirmMessage', { resource: t('sc.title'), name }),
        ]}
        isCheck={isScDelCheck}
        showCheck={true}
        checkFn={handleScCheckFn}
        confirmFn={handleDeleteScConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetScDetailTabKey} activeKey={scDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}
