/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { clusterStatus, DEFAULT_REPO } from '@/common/constants';
import { Tooltip, message } from 'antd';
import { useState, useEffect } from 'openinula';
import { CheckCircleFilled, ExclamationCircleFilled, QuestionCircleOutlined } from '@ant-design/icons';
import ToastMsg from '@/components/ToastMsg';
import { clusterStatusValueToText } from '@/tools/utils';
import { useIntl } from '../../../i18n';
export default function ClusterDetail({ clusterDetail }) {
  const intl = useIntl();
  const [status, setStatus] = useState(clusterDetail?.clusterStatus || '');
  const [clusterName, setClusterName] = useState(clusterDetail?.clusterName || '');
  const [clusterDetailData, setClusterDetailData] = useState(clusterDetail); // 详情数据
  const [messageApi, contextHolder] = message.useMessage();

  useEffect(() => {
    setStatus(clusterDetail?.clusterStatus || '');
    setClusterName(clusterDetail?.clusterName || '');
    setClusterDetailData(clusterDetail);
  }, [clusterDetail]);

  return (
    <>
      <ToastMsg contextHolder={contextHolder} />
      <div className="container_margin_box" style={{ margin: '32px' }}>
        <div className="detail_card">
          <h3>{intl.formatMessage({ id: 'cluster.detail.basicInfo' })}</h3>
          <div className="detail_info_box" style={{ padding: 0 }}>
            <div className="base_info_list">
              <div className="flex_item_opt">
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'cluster.detail.clusterName' })}:</p>
                  <p className="base_value">{clusterName}</p>
                </div>
                <div className="base_description">
                  <p className="base_key" style={{ width: 'auto' }}>
                    {intl.formatMessage({ id: 'cluster.detail.controlPlaneEndpoint' })}
                    <Tooltip title={intl.formatMessage({ id: 'cluster.detail.controlPlaneTooltip' })}>
                      <QuestionCircleOutlined className="warn_icon" style={{ marginRight: '5px' }} />
                    </Tooltip>
                    ：
                  </p>
                  <p className="base_value">{clusterDetailData?.Host ?? '--'}</p>
                </div>
              </div>
              <div className="flex_item_opt">
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'cluster.detail.status' })}:</p>
                  <p className="base_value icon_status">
                    <span className={`${clusterStatus[status]}_color`}>
                      {clusterStatus[status] === 'healthy' ? <CheckCircleFilled /> : <ExclamationCircleFilled />}
                    </span>
                    <span>{clusterStatusValueToText(status, intl)}</span>
                  </p>
                </div>
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'cluster.detail.createdAt' })}:</p>
                  <p className="base_value">{clusterDetailData?.createTime ? clusterDetailData.createTime : '--'}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="container_margin_box">
        <div className="detail_card">
          <h3>{intl.formatMessage({ id: 'cluster.detail.resourceInfo' })}</h3>
          <div className="detail_info_box" style={{ padding: 0 }}>
            <div className="base_info_list">
              <div className="flex_item_opt resource_flex_gap">
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'cluster.detail.registryAddress' })}:</p>
                  <p className="base_value">{clusterDetailData?.imageRepo.domain}</p>
                </div>
              </div>
              <div className="flex_item_opt resource_flex_gap">
                <div className="base_description">
                  <p className="base_key">{intl.formatMessage({ id: 'cluster.detail.chartRepoAddress' })}:</p>
                  <p className="base_value">
                    {clusterDetailData?.imageRepo.domain.includes(DEFAULT_REPO.imageRepo)
                      ? DEFAULT_REPO.helmRepo
                      : `http://${clusterDetailData?.ip}:38080`}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
