/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Link } from 'inula-router';
import { containerRouterPrefix } from '@/constant';
import { Fragment, useState, useStore, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { CloseOutlined, EditOutlined } from '@ant-design/icons';
import Dayjs from 'dayjs';
import { message } from 'antd';
import { getWorkloadStatusJudge, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import CpuIcon from '@/assets/images/cpuIcon.png';
import MemoryIcon from '@/assets/images/memoryIcon.png';
import AnnotationModal from '@/components/AnnotationModal';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import LabelTag from '@/components/LabelTag';
import ToolTipComponent from '@/components/ToolTipComponent';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function DeploymentInfo({ deploymentName, deploymentNamespace, deploymentDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation('common');
  const [isShow, setIsShow] = useState(true);
  const [messageApi, contextHolder] = message.useMessage();
  const [isDeploymentAnnotationModalOpen, setIsDeploymentAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const oldAnnotations = deploymentDetailDataProps.metadata.annotations;
  const oldLabels = deploymentDetailDataProps.spec.template.metadata.labels;
  const themeStore = useStore('theme');

  const deploymentBaseInfoItems = useMemo(() => {
    const d = deploymentDetailDataProps;
    const statusText = getWorkloadStatusJudge(d.status);
    return [
      { key: t('resourceDetail.workloadName'), value: d.metadata.name, descriptionClassName: 'deploymentDetail' },
      { key: t('resourceDetail.workloadType'), value: d.kind },
      { key: t('resourceDetail.workloadStatus'), value: <span className={`${statusText.toLowerCase()}_circle`}>{statusText}</span> },
      { key: t('resourceDetail.createdAt'), value: Dayjs(d.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm') },
      { key: t('resourceDetail.namespace'), value: d.metadata.namespace },
      { key: t('resourceDetail.instances'), value: `${d.status.readyReplicas || 0}/${d.status.replicas || 0}` },
      { key: t('resourceDetail.containerImage'), value: d.spec?.template?.spec?.containers[0]?.image },
    ];
  }, [deploymentDetailDataProps, t, i18n.language]);

  // 注解成功回调
  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.annotationUnchanged', { ns: 'common' }));
      setIsDeploymentAnnotationModalOpen(false);
    } else {
      const deploymentKeyArr = [];
      data.map(item => deploymentKeyArr.push(item.key));
      if (deploymentKeyArr.filter((item, index) => deploymentKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('deployment', deploymentNamespace, deploymentName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsDeploymentAnnotationModalOpen(false);
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
  // 注解失败回调
  const handleAnnotationCancel = () => {
    setIsDeploymentAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.labelUnchanged', { ns: 'common' }));
      setIsLabelModalOpen(false);
    } else {
      const keyArr = [];
      data.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label', true);
        try {
          const res = await editAnnotationsOrLabels('deployment', deploymentNamespace, deploymentName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
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

  // 标签失败回调
  const handleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handleShowChange = () => {
    setIsShow(false);
  };

  return <Fragment>
    <div className="deploymentMessage" style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={`${isShow ? '' : 'cant_show'}`}>
      <ToolTipComponent>
        <div className="deploymentDetail" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className='promot-box'>
            {t('resourceDetail.monitorBanner')}
            <Link to={`/${containerRouterPrefix}/monitor/monitorDashboard`}>{t('resourceDetail.monitorLink')}</Link>
            {t('resourceDetail.monitorSuffix')}
          </div>
          <CloseOutlined onClick={handleShowChange} />
        </div>
      </ToolTipComponent>
    </div>
    <div className={`deploymentDetail tab_container container_margin_box ${isShow ? 'tooltip_container_height' : 'normal_container_height'}`}>
      <div className="detail_card">
        <h3>{t('resourceDetail.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={deploymentBaseInfoItems} columns={2} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.labelsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={deploymentDetailDataProps?.spec.template.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {deploymentDetailDataProps.spec.template.metadata?.labels?.length ?
                deploymentDetailDataProps.spec.template.metadata.labels.map(item =>
                  <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.annotationsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsDeploymentAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isDeploymentAnnotationModalOpen} type="annotation" dataList={deploymentDetailDataProps?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {deploymentDetailDataProps.metadata?.annotations?.length ?
                deploymentDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
      <div className="detail_card container_card_add deploymentDetail">
        <h3>{t('resourceDetail.resourceUsage')}</h3>
        <div className="resource_list">
          <div className="resource_item">
            <img className="resource_icon" src={CpuIcon} />
            <div className="resource_word">
              <p>{t('resourceDetail.cpuCore')}</p>
              <p>{deploymentDetailDataProps.spec?.template?.spec?.containers[0]?.resources?.requests?.cpu || '--'}</p>
            </div>
          </div>

          <div className="resource_item">
            <img className="resource_icon" src={MemoryIcon} />
            <div className="resource_word">
              <p>{t('resourceDetail.memoryMB')}</p>
              <p>{deploymentDetailDataProps.spec?.template?.spec?.containers[0]?.resources?.requests?.memory || '--'}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Fragment>;
}