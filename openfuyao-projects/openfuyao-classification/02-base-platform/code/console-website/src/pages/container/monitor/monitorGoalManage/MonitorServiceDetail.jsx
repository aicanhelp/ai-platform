/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:

 *       http://license.coscl.org.cn/MulanPSL2

 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import { containerRouterPrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import Detail from '@/pages/container/monitor/monitorGoalManage/detail/Detail';
import Yaml from '@/pages/container/monitor/monitorGoalManage/detail/Yaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useStore } from 'openinula';
import { useHistory, useParams, useLocation } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getResourceInstanceDetailDescription, editResourceInstanceAnnotationOrlabels, deleteResourceInstance } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import { disabledModifyMonitorServiceCr } from '@/common/constants';

export default function MonitorServiceDetail() {
  const { exampleName, namespace, activeKey } = useParams();
  const { state: prefixObj } = useLocation();
  const history = useHistory();
  const { t } = useTranslation('monitoring');
  const [messageApi, contextHolder] = message.useMessage();

  const [serviceMonitorDelModalOpen, setServiceMonitorDelModalOpen] = useState(false); // 删除对话框展示
  const [isServiceMonitorDelCheck, setIsServiceMonitorDelCheck] = useState(false); // 是否选中
  const [detailLoaded, setDetailLoded] = useState(false); // 加载完成

  const [serviceMonitorDetailTabKey, setServiceMonitorDetailTabKey] = useState(activeKey || 'detail');

  const [serviceMonitorDetailData, setServiceMonitorDetailData] = useState({}); // 详情数据

  const [serviceMonitorPopOpen, setServiceMonitorPopOpen] = useState(false); // 气泡悬浮

  const [isReadyOnly, setIsReadyOnly] = useState(true); // 是否只读

  const [isMonitorAnnotationModalOpen, setIsMonitorAnnotationModalOpen] = useState(false);
  const [isMonitorLabelModalOpen, setIsMonitorLabelModalOpen] = useState(false);
  const [oldMonitorAnnotations, setOldMonitorAnnotations] = useState();
  const [oldMonitorLabels, setOldMonitorLabels] = useState();

  const [serviceMonitorYaml, setServiceMonitorYaml] = useState(''); // 传递yaml

  const themeStore = useStore('theme');

  // 处理只读
  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  // 气泡
  const handleServiceMonitorPopOpenChange = (open) => {
    setServiceMonitorPopOpen(open);
  };

  const handleSetServiceMonitorDetailTabKey = (key) => {
    setServiceMonitorDetailTabKey(key);
  };

  // 删除按钮
  const handleDeleteServiceMonitor = () => {
    setServiceMonitorPopOpen(false); // 气泡框
    setIsServiceMonitorDelCheck(false);
    setServiceMonitorDelModalOpen(true); // 打开弹窗
  };

  const handleDeleteServiceMonitorCancel = () => {
    setServiceMonitorDelModalOpen(false);
    setIsServiceMonitorDelCheck(false);
  };

  const handleDeleteServiceMonitorConfirm = async () => {
    try {
      const res = await deleteResourceInstance({ ...prefixObj, namespace }, exampleName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess'), () => {
          setServiceMonitorDelModalOpen(false);
          setIsServiceMonitorDelCheck(false);
          history.push(`/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor`);
        });
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      }
    }
  };

  const handleServiceMonitorCheckFn = (e) => {
    setIsServiceMonitorDelCheck(e.target.checked);
  };

  const getServiceMonitorDetailInfo = useCallback(async () => {
    if (exampleName && namespace) {
      setDetailLoded(false);
      const res = await getResourceInstanceDetailDescription({ ...prefixObj, namespace }, exampleName);
      if (res.status === ResponseCode.OK) {
        setServiceMonitorYaml(jsonToYaml(JSON.stringify(res.data))); // 先传递保持元数据clean
        res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
        res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
        setOldMonitorAnnotations([...res.data.metadata.annotations]);
        setOldMonitorLabels([...res.data.metadata.labels]);
        setServiceMonitorDetailData(res.data);
      }
      setDetailLoded(true);
    }
  }, [exampleName, namespace]);

  const handleModify = () => {
    setServiceMonitorDetailTabKey('yaml'); // 跳向yaml
    setIsReadyOnly(false); // 退出只读模式
  };

  // 注解成功回调
  const handleMonitorAnnotationOk = async (monitorServiceData) => {
    if (JSON.stringify(oldMonitorAnnotations) === JSON.stringify(monitorServiceData)) {
      messageApi.info(t('messages.annotationNotModified'));
      setIsMonitorAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      monitorServiceData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey'));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldMonitorAnnotations, monitorServiceData, 'annotation');
        try {
          const res = await editResourceInstanceAnnotationOrlabels({ ...prefixObj, namespace }, exampleName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.annotationEditSuccess'), () => {
              getServiceMonitorDetailInfo();
              setIsMonitorAnnotationModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(`${t('messages.annotationEditFailed')}${error.response.data.message}`);
          }
        }
      }
    }
  };
  // 注解失败回调
  const handleAnnotationCancel = () => {
    setIsMonitorAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsMonitorAnnotationModalOpen(true);
    setServiceMonitorPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsMonitorLabelModalOpen(true);
    setServiceMonitorPopOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (monitorServiceData) => {
    if (JSON.stringify(oldMonitorLabels) === JSON.stringify(monitorServiceData)) {
      messageApi.info(t('messages.labelNotModified'));
      setIsMonitorLabelModalOpen(false);
      return;
    }
    const keyArr = [];
    monitorServiceData.map(item => keyArr.push(item.key));
    if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
      messageApi.error(t('messages.duplicateKey'));
    } else {
      // 请求接口添加
      // 比较前后是否一致 返回处理后的标签
      const addLabelList = solveAnnotationOrLabelDiff(oldMonitorLabels, monitorServiceData, 'label');
      try {
        const res = await editResourceInstanceAnnotationOrlabels({ ...prefixObj, namespace }, exampleName, addLabelList);
        if (res.status === ResponseCode.OK) {
          showSuccessThen(messageApi, t('messages.labelEditSuccess'), () => {
            getServiceMonitorDetailInfo();
            setIsMonitorLabelModalOpen(false);
          });
        }
      } catch (error) {
        if (error.response.status === ResponseCode.Forbidden) {
          forbiddenMsg(messageApi, error);
        } else {
          messageApi.error(`${t('messages.labelEditFailed')}${error.response.data.message}`);
        }
      }
    }
  };

  // 标签失败回调
  const handleLabelCancel = () => {
    setIsMonitorLabelModalOpen(false);
  };

  useEffect(() => {
    if (serviceMonitorDetailTabKey === 'detail' || activeKey) {
      getServiceMonitorDetailInfo();
    }
  }, [serviceMonitorDetailTabKey, getServiceMonitorDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'detail',
      label: t('serviceMonitor.detail.title'),
      children: <Detail
        prefixObjProps={prefixObj}
        serviceMonitorName={exampleName}
        serviceMonitorNamespace={namespace}
        serviceMonitorDetailDataProps={serviceMonitorDetailData}
        refreshFn={getServiceMonitorDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('serviceMonitor.yaml.title'),
      children: <Yaml
        serviceMonitorYamlProps={serviceMonitorYaml}
        prefixObjProps={prefixObj}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getServiceMonitorDetailInfo} />,
    },
  ];

  return <div className="child_content">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/monitor`, disabled: true },
      { title: t('module.targetList'), path: '/monitorGoalManage' },
      { title: t('serviceMonitor.title'), path: '/serviceMonitor' },
      { title: t('serviceMonitor.detail.title'), path: '/detail' },
    ]} />
    <div className='pod_title' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', border: themeStore.$s.theme !== 'light' && 'none' }}>
      <h3>{exampleName}</h3>
      {!disabledModifyMonitorServiceCr.includes(exampleName) && <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button
              type="link"
              onClick={handleModify}
            >
              {t('serviceMonitor.actions.modify')}
            </Button>
            <Button type="link" onClick={handleEditLabel}>{t('serviceMonitor.actions.editLabel')}</Button>
            <Button type="link" onClick={handleEditAnnotation}>{t('serviceMonitor.actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteServiceMonitor}>{t('serviceMonitor.actions.delete')}</Button>
          </Space>
        }
        open={serviceMonitorPopOpen}
        onOpenChange={handleServiceMonitorPopOpenChange}>
        <Button className='primary_btn'>{t('serviceMonitor.actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      }
      {detailLoaded && <AnnotationModal open={isMonitorAnnotationModalOpen} type="annotation" dataList={serviceMonitorDetailData?.metadata?.annotations} callbackOk={handleMonitorAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isMonitorLabelModalOpen} type="label" dataList={serviceMonitorDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('serviceMonitor.modal.deleteServiceMonitorTitle')}
        open={serviceMonitorDelModalOpen}
        cancelFn={handleDeleteServiceMonitorCancel}
        content={[
          t('serviceMonitor.modal.deleteServiceMonitorWarning'),
          t('serviceMonitor.modal.deleteServiceMonitorConfirm', { name: exampleName }),
        ]}
        isCheck={isServiceMonitorDelCheck}
        showCheck={true}
        checkFn={handleServiceMonitorCheckFn}
        confirmFn={handleDeleteServiceMonitorConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetServiceMonitorDetailTabKey} activeKey={serviceMonitorDetailTabKey}></Tabs>}
  </div>;
}
