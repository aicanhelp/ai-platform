/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Table } from 'antd';
import { useMemo } from 'openinula';
import { CheckCircleFilled, ExclamationCircleFilled, EditOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import Dayjs from 'dayjs';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import AnnotationModal from '@/components/AnnotationModal';
import LabelTag from '@/components/LabelTag';
import { solveCustomResourceStatus } from '@/tools/utils';

export default function CrdInfo({
  detailData,
  isLabelModalOpen,
  setIsLabelModalOpen,
  handleLabelOk,
  handleLabelCancel,
  isAnnotationModalOpen,
  setIsAnnotationModalOpen,
  handleAnnotationOk,
  handleAnnotationCancel,
}) {
  const { t } = useTranslation(['crd', 'common']);

  const baseInfoItems = useMemo(() => {
    if (!detailData?.metadata) { return []; }
    return [
      { key: t('detail.name'), value: detailData.metadata.name },
      { key: t('detail.version'), value: detailData.spec.versions?.[0]?.name || '--' },
      { key: t('detail.status'), value: solveCustomResourceStatus(detailData.status?.conditions) ? <CheckCircleFilled style={{ color: '#09aa71' }} /> : <ExclamationCircleFilled style={{ color: '#e7434a' }} /> },
      { key: t('detail.scope'), value: detailData.spec.scope || '--' },
      { key: t('detail.createTime'), value: Dayjs(detailData.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm') },
    ];
  }, [detailData, t]);

  const statusColumns = useMemo(() => [
    {
      title: t('statusTable.type'),
      key: 'type',
      sorter: (a, b) => a.type.localeCompare(b.type),
      render: (_, record) => record.type,
    },
    {
      title: t('statusTable.status'),
      key: 'status',
      render: (_, record) => (
        <div className="status_group">
          <span className={record.status === 'True' ? 'running_circle' : 'failed_circle'}></span>
          <span>{record.status}</span>
        </div>
      ),
    },
    {
      title: t('statusTable.updateTime'),
      width: '15%',
      key: 'updateTime',
      sorter: (a, b) => Dayjs(a.lastTransitionTime) - Dayjs(b.lastTransitionTime),
      render: (_, record) => Dayjs(record.lastTransitionTime).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: t('statusTable.reason'),
      key: 'reason',
      ellipsis: true,
      render: (_, record) => record.reason || '--',
    },
    {
      title: t('statusTable.message'),
      key: 'message',
      ellipsis: true,
      render: (_, record) => record.message || '--',
    },
  ], [t]);

  const versionColumns = useMemo(() => [
    {
      title: t('versionTable.name'),
      key: 'name',
      sorter: (a, b) => a.name.length - b.name.length,
      render: (_, record) => record.name,
    },
    {
      title: t('versionTable.served'),
      key: 'served',
      render: (_, record) => <span>{record.served ? 'True' : 'False'}</span>,
    },
    {
      title: t('versionTable.storage'),
      key: 'storage',
      render: (_, record) => <span>{record.storage ? 'True' : 'False'}</span>,
    },
  ], [t]);

  return (
    <div className="tab_container container_margin_box normal_container_height">
      <div className="detail_card">
        <h3>{t('detail.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={baseInfoItems} columns={2} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('detail.labels')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={detailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {detailData.metadata?.labels?.length ?
                detailData.metadata.labels.map(item =>
                  <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('detail.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('detail.annotations')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={detailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {detailData.metadata?.annotations?.length ?
                detailData.metadata.annotations.map(item =>
                  <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('detail.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
      <div className="detail_card container_card_add">
        <h3>{t('statusTable.title')}</h3>
        <Table
          className="table_padding"
          style={{ paddingRight: '0px' }}
          dataSource={detailData.status?.conditions || []}
          rowKey={(record) => record.type ?? record.reason}
          columns={statusColumns}
          pagination={{
            className: 'page',
            style: { paddingRight: '0px' },
            showTotal: total => t('pagination.total', { ns: 'common', total }),
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
          }}
        />
      </div>
      <div className="detail_card container_card_add">
        <h3>{t('versionTable.title')}</h3>
        <Table
          className="table_padding"
          style={{ paddingRight: '0px' }}
          dataSource={detailData.spec?.versions || []}
          rowKey={(record) => record.name}
          columns={versionColumns}
          pagination={{
            className: 'page',
            style: { paddingRight: '0px' },
            showTotal: total => t('pagination.total', { ns: 'common', total }),
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
          }}
        />
      </div>
    </div>
  );
}
