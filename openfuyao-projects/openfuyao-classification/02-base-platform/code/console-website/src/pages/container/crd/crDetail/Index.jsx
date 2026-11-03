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
import ToastMsg from '@/components/ToastMsg';
import { useState, useEffect, useCallback, useMemo, useStore } from 'openinula';
import { useHistory, useLocation, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import '@/styles/pages/podDetail.less';
import {
  getResourceInstanceDetailDescription,
  deleteResourceInstance,
  editResourceInstanceAnnotationOrlabels,
  getCustomResourceDefinitionDetailDescription,
} from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, solveDecodePath, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import CrInfo from './CrInfo';
import CrYaml from './CrYaml';

export default function CrDetail() {
  const { t } = useTranslation(['crd', 'common']);
  const { customResourceName, activeKey, instanceName, namespace } = useParams();
  const { state } = useLocation();
  const [prefixObj, setPrefixObj] = useState(state ?? null);
  const activePrefixObj = useMemo(() => {
    if (!prefixObj) { return null; }
    return { ...prefixObj, namespace: prefixObj.namespace || namespace || '' };
  }, [prefixObj, namespace]);
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');

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
  const [yamlData, setYamlData] = useState('');
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
    if (!activePrefixObj) { return; }
    try {
      const res = await deleteResourceInstance(activePrefixObj, instanceName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/customResourceDefinition/${customResourceName}/example`);
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
    if (instanceName && activePrefixObj) {
      setDetailLoaded(false);
      try {
        const res = await getResourceInstanceDetailDescription(activePrefixObj, instanceName);
        if (res.status === ResponseCode.OK) {
          setYamlData(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setDetailData(res.data);
        }
      } catch (e) {
        messageApi.error(t('messages.fetchFailed', { ns: 'common', message: e.response?.data?.message }));
      } finally {
        setDetailLoaded(true);
      }
    }
  }, [instanceName, activePrefixObj]);

  // 刷新/直链时 location.state 丢失，从 CRD API 重建 prefixObj
  useEffect(() => {
    if (!prefixObj && customResourceName) {
      getCustomResourceDefinitionDetailDescription(solveDecodePath(customResourceName))
        .then(res => {
          if (res.status === ResponseCode.OK) {
            setPrefixObj({
              group: res.data?.spec?.group,
              version: res.data?.spec?.versions?.[0]?.name,
              plural: res.data?.spec?.names?.plural,
              scopeNamespace: res.data?.spec?.scope,
              namespace,
            });
          }
        })
        .catch(e => messageApi.error(t('messages.fetchFailed', { ns: 'common', message: e.response?.data?.message })));
    }
  }, [prefixObj, customResourceName, namespace]);

  const handleAnnotationOk = async (data) => {
    if (!activePrefixObj) { return; }
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
          const res = await editResourceInstanceAnnotationOrlabels(activePrefixObj, instanceName, addAnnotationList);
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
    if (!activePrefixObj) { return; }
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
          const res = await editResourceInstanceAnnotationOrlabels(activePrefixObj, instanceName, addLabelList);
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
        <CrInfo
          customResourceName={customResourceName}
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
        <CrYaml
          yamlData={yamlData}
          isReadyOnly={isReadyOnly}
          handleReadyOnly={handleReadyOnly}
          getDetailInfo={getDetailInfo}
          prefixObj={activePrefixObj}
        />
      ),
    },
  ];

  useEffect(() => {
    if (activeKey) {
      setTabKey(activeKey);
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  useEffect(() => {
    if (tabKey === 'detail' || activeKey) {
      getDetailInfo();
    }
  }, [tabKey, getDetailInfo]);

  return (
    <div className="child_content">
      <ToastMsg contextHolder={contextHolder} />
      <BreadCrumbCom items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/customResourceDefinition` },
        { title: t('module.crdDetail'), path: `${customResourceName}/example` },
        { title: t('module.crInstanceDetail') },
      ]} />
      <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
        <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
          <span className='label_title'>{t('detail.instanceLabel')}</span>
          <h3>{instanceName}</h3>
        </div>
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
          title={t('modal.deleteInstanceTitle')}
          open={delModalOpen}
          cancelFn={handleDelCancel}
          content={[
            t('modal.deleteInstanceWarning'),
            t('modal.deleteInstanceConfirm', { name: instanceName }),
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
