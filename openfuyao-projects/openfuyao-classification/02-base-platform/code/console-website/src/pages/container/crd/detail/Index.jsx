/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix } from '@/constant';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useStore } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import '@/styles/pages/podDetail.less';
import { getCustomResourceDefinitionDetailDescription, deleteCustomResourceDefinition, editAnnotationsOrLabels } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, solveDecodePath, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import CrdInfo from './CrdInfo';
import CrdYaml from './CrdYaml';
import CrInstances from './CrInstances';

export default function CrdDetail() {
  const { t } = useTranslation(['crd', 'common']);
  const { customResourceName, activeKey } = useParams();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');

  const [crdName, setCrdName] = useState('');
  const [delModalOpen, setDelModalOpen] = useState(false);
  const [isDelCheck, setIsDelCheck] = useState(false);
  const [detailLoaded, setDetailLoaded] = useState(false);
  const [tabKey, setTabKey] = useState(activeKey || 'detail');
  const [detailData, setDetailData] = useState({});
  const [popOpen, setPopOpen] = useState(false);
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const [oldAnnotations, setOldAnnotations] = useState();
  const [oldLabels, setOldLabels] = useState();
  const [crdYaml, setCrdYaml] = useState('');
  const [isReadyOnly, setIsReadyOnly] = useState(true);

  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handlePopOpenChange = (open) => {
    setPopOpen(open);
  };

  const handleSetTabKey = (key) => {
    setTabKey(key);
    setIsReadyOnly(true);
  };

  const handleDelete = () => {
    setPopOpen(false);
    setDelModalOpen(true);
  };

  const handleDelCancel = () => {
    setDelModalOpen(false);
  };

  const handleDelConfirm = async () => {
    try {
      const res = await deleteCustomResourceDefinition(crdName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/customResourceDefinition`);
        });
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };

  const handleDelCheckFn = (e) => {
    setIsDelCheck(e.target.checked);
  };

  const getDetailInfo = useCallback(async () => {
    if (crdName) {
      setDetailLoaded(false);
      try {
        const res = await getCustomResourceDefinitionDetailDescription(crdName);
        if (res.status === ResponseCode.OK) {
          setCrdYaml(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setDetailData(res.data);
        }
      } catch (error) {
        if (error.response?.status !== ResponseCode.OK) {
          messageApi.error(t('messages.fetchFailed', { ns: 'common', message: error.response?.data?.message }));
        }
      } finally {
        setDetailLoaded(true);
      }
    }
  }, [crdName]);

  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.annotationUnchanged'));
      setIsAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      data.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('customresourcedefinition', '', crdName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getDetailInfo();
              setIsAnnotationModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
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
    setPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsLabelModalOpen(true);
    setPopOpen(false);
  };

  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.labelUnchanged'));
      setIsLabelModalOpen(false);
    } else {
      const keyArr = [];
      data.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('customresourcedefinition', '', crdName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getDetailInfo();
              setIsLabelModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
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
    setTabKey('yaml');
    setIsReadyOnly(false);
  };

  const items = [
    {
      key: 'detail',
      label: t('detail.tabInfo'),
      children: detailLoaded && (
        <CrdInfo
          detailData={detailData}
          isLabelModalOpen={isLabelModalOpen}
          setIsLabelModalOpen={setIsLabelModalOpen}
          handleLabelOk={handleLabelOk}
          handleLabelCancel={handleLabelCancel}
          isAnnotationModalOpen={isAnnotationModalOpen}
          setIsAnnotationModalOpen={setIsAnnotationModalOpen}
          handleAnnotationOk={handleAnnotationOk}
          handleAnnotationCancel={handleAnnotationCancel}
        />
      ),
    },
    {
      key: 'yaml',
      label: t('detail.tabYaml'),
      children: detailLoaded && (
        <CrdYaml
          crdYaml={crdYaml}
          isReadyOnly={isReadyOnly}
          handleReadyOnly={handleReadyOnly}
          getDetailInfo={getDetailInfo}
        />
      ),
    },
    {
      key: 'example',
      label: t('detail.tabInstances'),
      children: detailLoaded && (
        <CrInstances crdDetailData={detailData} />
      ),
    },
  ];

  useEffect(() => {
    if (activeKey) {
      setTabKey(activeKey);
      history.replace(history.location.pathname.slice(0, history.location.pathname.lastIndexOf('/')));
      setIsReadyOnly(false);
    }
  }, [activeKey, history]);

  useEffect(() => {
    getDetailInfo();
  }, [getDetailInfo]);

  useEffect(() => {
    setCrdName(solveDecodePath(customResourceName));
  }, [customResourceName]);

  return (
    <div className="child_content">
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <BreadCrumbCom items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/customResourceDefinition` },
        { title: t('module.crdDetail'), path: '/detail' },
      ]} />
      <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
        <h3>{crdName}</h3>
        <Popover placement='bottom'
          content={
            <Space className='column_pop'>
              <Button type="link" onClick={handleModify}>{t('actions.modify')}</Button>
              <Button type="link" onClick={handleEditLabel}>{t('actions.editLabel')}</Button>
              <Button type="link" onClick={handleEditAnnotation}>{t('actions.editAnnotation')}</Button>
              <Button type="link" onClick={handleDelete}>{t('actions.delete')}</Button>
            </Space>
          }
          open={popOpen}
          onOpenChange={handlePopOpenChange}>
          <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
        </Popover>
        <DeleteInfoModal
          title={t('modal.deleteTitle')}
          open={delModalOpen}
          cancelFn={handleDelCancel}
          content={[
            t('modal.deleteWarning'),
            t('modal.deleteConfirm', { name: crdName }),
          ]}
          isCheck={isDelCheck}
          showCheck={true}
          checkFn={handleDelCheckFn}
          confirmFn={handleDelConfirm} />
      </div>
      {detailLoaded && <Tabs items={items} onChange={handleSetTabKey} activeKey={tabKey}></Tabs>}
    </div>
  );
}
