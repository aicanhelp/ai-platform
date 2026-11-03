/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Fragment, useMemo, useState, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import Dayjs from 'dayjs';
import AnnotationModal from '@/components/AnnotationModal';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import { EditOutlined } from '@ant-design/icons';
import { Table, message } from 'antd';
import { ResponseCode } from '@/common/constants';
import { solveAnnotationOrLabelDiff, solveLimitTableList, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { solveLimitRangeRate } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function LimiRangeInfo({ limitRangeName, limitRangeNamespace, limitRangeDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['namespace', 'common']);
  const [messageApi, contextHolder] = message.useMessage();
  const [isLimitDetailAnnotationModalOpen, setIsLimitDetailAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const limitRangeTabelData = solveLimitTableList(limitRangeDetailDataProps?.spec?.limits); // 表数据
  const oldAnnotations = limitRangeDetailDataProps?.metadata?.annotations || [];
  const oldLabels = limitRangeDetailDataProps?.metadata?.labels || [];
  const themeStore = useStore('theme');

  // 注解成功回调
  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsLimitDetailAnnotationModalOpen(false);
    } else {
      const limitRangeDetailkeyArr = [];
      data.map(item => limitRangeDetailkeyArr.push(item.key));
      if (limitRangeDetailkeyArr.filter((item, index) => limitRangeDetailkeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('limitrange', limitRangeNamespace, limitRangeName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsLimitDetailAnnotationModalOpen(false);
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
    setIsLimitDetailAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsLabelModalOpen(false);
    } else {
      const keyArr = [];
      data.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('limitrange', limitRangeNamespace, limitRangeName, addLabelList);
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

  const limitRangeDetailColumns = useMemo(() => [
    {
      title: t('table.type'),
      key: 'limit_type',
      render: (_, record) => record.type,
    },
    {
      title: t('table.resource'),
      key: 'limit_resource',
      ellipsis: true,
      render: (_, record) => record.resource,
    },
    {
      title: t('table.min'),
      width: '15%',
      key: 'limit_smallest',
      render: (_, record) => record.min ? record.min : '--',
    },
    {
      title: t('table.max'),
      width: '15%',
      key: 'limit_biggest',
      render: (_, record) => record.max ? record.max : '--',
    },
    {
      title: t('table.defaultRequest'),
      key: 'limit_request',
      render: (_, record) => record.defaultRequest || '--',
    },
    {
      title: t('table.defaultLimit'),
      key: 'limit',
      render: (_, record) => record.default || '--',
    },
    {
      title: t('table.ratio'),
      key: 'biggeset_limitOrRequest',
      render: (_, record) => solveLimitRangeRate(record.default, record.defaultRequest) || '--',
    },
  ], [t, i18n.language]);

  const limitRangeBaseInfoItems = useMemo(() => [
    { key: `${t('fields.name')}：`, value: limitRangeDetailDataProps?.metadata?.name },
    { key: `${t('fields.namespace')}：`, value: limitRangeDetailDataProps?.metadata?.namespace },
    { key: `${t('fields.createdAt')}：`, value: Dayjs(limitRangeDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
  ], [limitRangeDetailDataProps, t, i18n.language]);

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={'tab_container container_margin_box normal_container_height'}>
      <div className="detail_card">
        <h3>{t('info.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={limitRangeBaseInfoItems} columns={1} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.labels')}：</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={oldLabels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {oldLabels?.length ? oldLabels.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.annotations')}：</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLimitDetailAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isLimitDetailAnnotationModalOpen} type="annotation" dataList={oldAnnotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {oldAnnotations?.length ? oldAnnotations.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
      <div className="detail_card container_card_add">
        <h3>{t('info.limits')}</h3>
        <Table className="table_padding"
          style={{ paddingRight: '0px' }}
          dataSource={limitRangeTabelData}
          columns={limitRangeDetailColumns}
          pagination={{
            className: 'page',
            pageSizeOptions: [10, 20, 50],
            style: { paddingRight: '0px' },
            showSizeChanger: true,
            showTotal: total => t('pagination.total', { ns: 'common', total }),
          }} />
      </div>
    </div>
  </Fragment>;
}
