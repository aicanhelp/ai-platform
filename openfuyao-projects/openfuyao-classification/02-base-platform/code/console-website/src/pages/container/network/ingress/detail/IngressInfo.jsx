/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Fragment, useCallback, useEffect, useMemo, useState, useStore } from 'openinula';
import { ResponseCode } from '@/common/constants';
import Dayjs from 'dayjs';
import { EditOutlined, InfoCircleFilled } from '@ant-design/icons';
import { message, Table, Space } from 'antd';
import AnnotationModal from '@/components/AnnotationModal';
import { editIngressLabelOrAnnotation } from '@/api/containerApi';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function Detail({ ingressDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['network', 'common']);

  const [messageApi, contextHolder] = message.useMessage();

  const oldAnnotations = ingressDetailDataProps?.metadata?.annotations; // 未修改前注解

  const oldLabels = ingressDetailDataProps?.metadata?.labels;

  const [ingressRuleTabelData, setIngressRuleTabelData] = useState([]); // 规则表

  const [ingressDetailInfoLoading, setIngressDetailInfoLoading] = useState(false); // 加载中

  // 标签对话框展示
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  // 注解对话框展示
  const [isIngressAnnotationModalOpen, setIsIngressAnnotationModalOpen] = useState(false);

  const themeStore = useStore('theme');

  // 标签成功回调
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
          const res = await editIngressLabelOrAnnotation(ingressDetailDataProps?.metadata?.namespace ?? '', ingressDetailDataProps?.metadata?.name ?? '', addLabelList);
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

  // 注解成功回调
  const handleAnnotationOk = async (data) => {
    const keyArr = [];
    data.map(item => keyArr.push(item.key));
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsIngressAnnotationModalOpen(false);
    } else {
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editIngressLabelOrAnnotation(ingressDetailDataProps?.metadata?.namespace ?? '', ingressDetailDataProps?.metadata?.name ?? '', addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsIngressAnnotationModalOpen(false);
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
    setIsIngressAnnotationModalOpen(false);
  };

  const ingressDetailInfoColumns = useMemo(() => [
    {
      title: t('table.host'),
      key: 'host',
      // sorter: true,
      render: (_, record) => (
        <Space size="middle">
          {record.host ? record.host : '--'}
        </Space>
      ),
    },
    {
      title: t('table.path'),
      key: 'path',
      render: (_, record) => (
        <Space size="middle">
          {record.http.paths ? record.http.paths[0].path : '--'}
        </Space>
      ),
    },
    {
      title: t('table.pathType'),
      key: 'type',
      render: (_, record) => (
        <Space size="middle">
          {record.http.paths ? record.http.paths[0].pathType : '--'}
        </Space>
      ),
    },
    {
      title: t('table.service'),
      key: 'service',
      render: (_, record) => (
        <Space size="middle">
          {record.http.paths ? record.http.paths[0].backend.service.name : '--'}
        </Space>
      ),
    },
    {
      title: t('table.servicePort'),
      key: 'port',
      render: (_, record) => (
        <Space size="middle">
          {record.http.paths ? record.http.paths[0].backend.service.port.number : '--'}
        </Space>
      ),
    },
  ], [t, i18n.language]);

  const getIngressDetailInfoTableList = useCallback(() => {
    setIngressDetailInfoLoading(true);
    const data = ingressDetailDataProps?.spec?.rules ?? [];
    setIngressRuleTabelData(data);
    setIngressDetailInfoLoading(false);
  }, [ingressDetailDataProps]);

  useEffect(() => {
    // 获取详情
    getIngressDetailInfoTableList();
  }, [getIngressDetailInfoTableList]);

  const ingressBaseInfoItems = useMemo(() => [
    {
      key: 'ingressName',
      value: ingressDetailDataProps?.metadata.name,
      description: t('ingress.title'),
    },
    {
      key: 'namespace',
      value: ingressDetailDataProps?.metadata.namespace,
      description: t('ingress.namespace'),
    },
    {
      key: 'creationTimestamp',
      value: Dayjs(ingressDetailDataProps?.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
      description: t('ingress.created'),
    },
  ], [ingressDetailDataProps, t, i18n.language]);

  return <Fragment>
    <ToastMsg contextHolder={contextHolder} />
    <div className="net_tab_container container_margin_box net_tooltip_container_height">
      <div className="detail_card net-detail-card" style={{ padding: '32px 32px 0 32px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div>
          <h3>{t('info.basicInfo')}</h3>
          <div className="detail_info_box">
            <DetailBaseInfoList items={ingressBaseInfoItems} columns={1} />
            <div className="annotation">
              <div className="ann_title">
                <p>{t('info.labels')}</p>
                <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
              </div>
              <AnnotationModal open={isLabelModalOpen} type="label" dataList={ingressDetailDataProps?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
              <div className="key_value">
                {ingressDetailDataProps.metadata?.labels?.length ?
                  ingressDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                  <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
              </div>
            </div>
            <div className="annotation">
              <div className="ann_title">
                <p>{t('info.annotations')}</p>
                <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsIngressAnnotationModalOpen(true)} />
              </div>
              <AnnotationModal open={isIngressAnnotationModalOpen} type="annotation" dataList={ingressDetailDataProps?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
              <div className="key_value">
                {ingressDetailDataProps.metadata?.annotations?.length ?
                  ingressDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                  <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
              </div>
            </div>
          </div>
        </div>
        <div className="net-detail-table-box">
          <h3>{t('info.ingressRules')}</h3>
          <div className='hint'>
            <InfoCircleFilled className='hint_icon' />
            <span className='hint_text'>{t('info.ingressRulesDescription')}</span>
          </div>
          <div className="net-detail-table">
            <Table style={{ paddingRight: '0px' }} loading={ingressDetailInfoLoading} dataSource={ingressRuleTabelData} columns={ingressDetailInfoColumns}
              rowKey={(record) => `${record.host ?? ''}_${record.http?.paths?.[0]?.path ?? ''}`}
              pagination={{
                className: 'page',
                pageSizeOptions: [10, 20, 50],
              }} />
          </div>
        </div>
      </div>
    </div>
  </Fragment>;
}