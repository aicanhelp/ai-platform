/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/log.less';
import { useIntl } from 'inula-intl';
import { Tooltip } from 'antd';
import { ResponseCode } from '@/common/constants';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { getLogCollectionData, getLogJobTime } from '@/api/containerLogApi';
import { useEffect, useState } from 'openinula';

export default function LogCollecting() {
  const intl = useIntl();
  const [collectData, setCollectData] = useState([]);
  const [collectAge, setCollectAge] = useState('');
  const [collectTime, setCollectTime] = useState('');
  const getLogCollection = async () => {
    const res = await getLogCollectionData();
    if (res.status === ResponseCode.OK) {
      setCollectData(res.data.data[Object.keys(res.data.data)[0]]);
    }
  };
  const getLogCollectionTime = async () => {
    const res = await getLogJobTime();
    if (res.status === ResponseCode.OK) {
      if (!res.data.data.TableManager.RetentionDeletesEnabled) {
        setCollectTime(intl.formatMessage({ id: 'collection.unlimited' }));
      } else {
        setCollectTime(res.data.data.TableManager.RetentionPeriod);
      }
      setCollectAge(res.data.data.LimitsConfig.RejectOldSamplesMaxAge);
    }
  };
  useEffect(() => {
    getLogCollection();
    getLogCollectionTime();
  }, []);
  return (
    <div className="logCollecting">
      <p className="logCollecting_title">{intl.formatMessage({ id: 'collection.sourceTitle' })}</p>
      {collectData.map((item) => (
        <div className="logCollecting_info">
          <p className="logCollecting_info_title">{item.name}</p>
          <p className="logCollecting_info_path">
            {intl.formatMessage({ id: 'collection.path' })}{' '}
            <span className="logCollecting_info_address">{item.mountPath}</span>
          </p>
        </div>
      ))}
      <p className="logCollecting_title logCollecting_title_time">
        {intl.formatMessage({ id: 'collection.retentionTitle' })}
      </p>
      <div className="logCollecting_time">
        <p className="logCollecting_time_title">{intl.formatMessage({ id: 'collection.retentionLabel' })}</p>
        <Tooltip title={intl.formatMessage({ id: 'collection.retentionTooltip' })}>
          <QuestionCircleOutlined style={{ color: '#89939B' }} />
        </Tooltip>
        <p className="logCollecting_time_num">{collectTime}</p>
      </div>
      <div className="logCollecting_time">
        <p className="logCollecting_time_title">{intl.formatMessage({ id: 'collection.maxAgeLabel' })}</p>
        <Tooltip title={intl.formatMessage({ id: 'collection.maxAgeTooltip' })}>
          <QuestionCircleOutlined style={{ color: '#89939B' }} />
        </Tooltip>
        <p className="logCollecting_time_num">{collectAge}</p>
      </div>
    </div>
  );
}
