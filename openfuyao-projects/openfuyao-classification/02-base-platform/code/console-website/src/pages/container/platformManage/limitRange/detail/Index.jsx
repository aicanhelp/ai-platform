/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
 *
 *       http://license.coscl.org.cn/MulanPSL2
 *
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import { containerRouterPrefix } from '@/constant.js';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import LimitRangeYaml from '@/pages/container/platformManage/limitRange/detail/LimitRangeYaml';
import LimitRangeInfo from '@/pages/container/platformManage/limitRange/detail/LimitRangeInfo';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useRef, useStore } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getLimitRange, deleteLimitRange, editAnnotationsOrLabels } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function LimitRangeDetail() {
  const { t } = useTranslation(['namespace', 'common']);
  const { namespace, name, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [limitRangeDelModalOpen, setLimitRangeDelModalOpen] = useState(false);
  const [isLimitRangeDelCheck, setIsLimitRangeDelCheck] = useState(false);
  const [detailLoaded, setDetailLoded] = useState(false);

  const [limitRangeDetailTabKey, setLimitRangeDetailTabKey] = useState(activeKey || 'info');

  const [limitRangeDetailData, setLimitRangeDetailData] = useState({});

  const [limitRangePopOpen, setLimitRangePopOpen] = useState(false);
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);

  const [oldAnnotations, setOldAnnotations] = useState();
  const [oldLabels, setOldLabels] = useState();

  const [limitRangeYaml, setLimitRangeYaml] = useState('');
  const [isReadyOnly, setIsReadyOnly] = useState(true);

  const themeStore = useStore('theme');

  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handleLimitRangePopOpenChange = (open) => {
    setLimitRangePopOpen(open);
  };

  const handleSetLimitRangeDetailTabKey = (key) => {
    setLimitRangeDetailTabKey(key);
    setIsReadyOnly(true);
  };

  const handleDeleteLimitRange = () => {
    setLimitRangePopOpen(false);
    setLimitRangeDelModalOpen(true);
  };

  const handleDeleteLimitRangeCancel = () => {
    setLimitRangeDelModalOpen(false);
  };

  const handleDelLimitRangeConfirm = async () => {
    try {
      const res = await deleteLimitRange(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setLimitRangeDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/namespace/limitRange`);
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

  const handleLimitRangeCheckFn = (e) => {
    setIsLimitRangeDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getLimitRangeDetailInfo = useCallback(async () => {
    if (name && namespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getLimitRange(namespace, name);
        if (seq !== detailReqSeqRef.current) {
          return;
        }
        if (res.status === ResponseCode.OK) {
          setLimitRangeYaml(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldAnnotations([...res.data.metadata.annotations]);
          setOldLabels([...res.data.metadata.labels]);
          setLimitRangeDetailData(res.data);
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

  const handleLimitRangeAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsAnnotationModalOpen(false);
    } else {
      const limitRangeAnnKeyArr = [];
      data.map(item => limitRangeAnnKeyArr.push(item.key));
      if (limitRangeAnnKeyArr.filter((item, index) => limitRangeAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('limitrange', namespace, name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getLimitRangeDetailInfo();
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

  const handleLimitRangeAnnotationCancel = () => {
    setIsAnnotationModalOpen(false);
  };

  const handleEditLimitRangeAnnotation = () => {
    setIsAnnotationModalOpen(true);
    setLimitRangePopOpen(false);
  };

  const handleEditLimitRangeLabel = () => {
    setIsLabelModalOpen(true);
    setLimitRangePopOpen(false);
  };

  const handleLimitRangeLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      const limitRangeLabelKeyArr = [];
      data.map(item => limitRangeLabelKeyArr.push(item.key));
      if (limitRangeLabelKeyArr.filter((item, index) => limitRangeLabelKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('limitrange', namespace, name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getLimitRangeDetailInfo();
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

  const handleLimitRangeLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handleModify = () => {
    setLimitRangeDetailTabKey('yaml');
    setIsReadyOnly(false);
  };

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <LimitRangeInfo
        limitRangeName={name}
        limitRangeNamespace={namespace}
        limitRangeDetailDataProps={limitRangeDetailData}
        refreshFn={getLimitRangeDetailInfo}
      />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <LimitRangeYaml
        limitRangeYamlProps={limitRangeYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getLimitRangeDetailInfo}
      />,
    },
  ];

  const isInitialMountRef = useRef(true);

  useEffect(() => {
    if (limitRangeDetailTabKey === 'info' || (activeKey && isInitialMountRef.current)) {
      getLimitRangeDetailInfo();
    }
    isInitialMountRef.current = false;
  }, [limitRangeDetailTabKey, getLimitRangeDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  return (
    <div className="child_content withBread_content">
      <ToastMsg contextHolder={contextHolder} />
      <BreadCrumbCom items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/namespace/limitRange`, disabled: true },
        { title: t('limitRange.title'), path: `/${containerRouterPrefix}/namespace/limitRange` },
        { title: t('actions.detail'), path: '/detail' },
      ]}
      />
      <div className="pod_title" style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
        <h3>{name}</h3>
        <Popover placement="bottom"
          content={
            <Space className="column_pop">
              <Button type="link" onClick={handleModify}>{t('actions.modify')}</Button>
              <Button type="link" onClick={handleEditLimitRangeLabel}>{t('actions.editLabel')}</Button>
              <Button type="link" onClick={handleEditLimitRangeAnnotation}>{t('actions.editAnnotation')}</Button>
              <Button type="link" onClick={handleDeleteLimitRange}>{t('actions.delete')}</Button>
            </Space>
          }
          open={limitRangePopOpen}
          onOpenChange={handleLimitRangePopOpenChange}>
          <Button className="primary_btn">{t('actions.operation')} <DownOutlined className="small_margin_adjust" /></Button>
        </Popover>
        {detailLoaded && <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={limitRangeDetailData?.metadata.annotations} callbackOk={handleLimitRangeAnnotationOk} callbackCancel={handleLimitRangeAnnotationCancel} />}
        {detailLoaded && <AnnotationModal open={isLabelModalOpen} type="label" dataList={limitRangeDetailData?.metadata.labels} callbackOk={handleLimitRangeLabelOk} callbackCancel={handleLimitRangeLabelCancel} />}
        <DeleteInfoModal
          title={t('actions.deleteConfirm', { resource: t('limitRange.title') })}
          open={limitRangeDelModalOpen}
          cancelFn={handleDeleteLimitRangeCancel}
          content={[
            t('actions.deleteWarning', { resource: t('limitRange.title') }),
            t('actions.deleteConfirmMessage', { resource: t('limitRange.title'), name }),
          ]}
          isCheck={isLimitRangeDelCheck}
          showCheck={true}
          checkFn={handleLimitRangeCheckFn}
          confirmFn={handleDelLimitRangeConfirm} />
      </div>
      {detailLoaded && <Tabs items={items} onChange={handleSetLimitRangeDetailTabKey} activeKey={limitRangeDetailTabKey} destroyInactiveTabPane={true} />}
    </div>
  );
}
