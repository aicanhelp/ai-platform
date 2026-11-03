/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { ResponseCode } from '@/common/constants';
import { useLocation } from 'inula-router';
import { useEffect, useState } from 'openinula';
import { message } from 'antd';
import { ExportOutlined, RightOutlined, DownOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import '@/styles/rayLog.less';
import { getServiceDetail } from '@/api/rayApi.js';
import { exportYamlOutPutLog } from '@/tools/utils';
import { useIntl } from 'inula-intl';
export default function Log() {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [messageApi, contextHolder] = message.useMessage();
  const location = useLocation();
  const serviceName = location.state.name;
  const serviceNameSpace = location.state.namespace;
  const [serviceDetailLogData, setServiceDetailLogData] = useState([]);
  const [serviceOpenLog, setServiceOpenLog] = useState();
  const [log, setLog] = useState();
  const getServiceInfo = async () => {
    try {
      const res = await getServiceDetail(serviceNameSpace, serviceName);
      if (res.data.code === ResponseCode.OK) {
        const arr = res.data.data.ray_head_logs.split('\n');
        setLog(res.data.data.ray_head_logs);
        arr.pop();
        setServiceDetailLogData(arr);
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(intl.formatMessage({ id: 'ray.service.detailError' }, { name: serviceName }) + ',' + msg);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(intl.formatMessage({ id: 'ray.service.detailError' }, { name: serviceName }) + ',' + msg);
      }
    }
  };
  const exportYaml = () => {
    exportYamlOutPutLog(`${serviceName}`, log);
    messageApi.success(intl.formatMessage({ id: 'log.exportSuccess' }));
  };
  useEffect(() => {
    getServiceInfo();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="log">
      {contextHolder}
      <div className="log_top">
        <p className="log_top_title">{intl.formatMessage({ id: 'log.info' })}</p>
        <p className="log_top_copy" onClick={exportYaml}>
          <ExportOutlined style={{ color: '#3f66f5', marginRight: '4px' }} />
          {intl.formatMessage({ id: 'log.export' })}
        </p>
      </div>
      <div className="log_contents">
        <div className="log_content">
          {serviceDetailLogData && serviceDetailLogData.length > 0 ? (
            serviceDetailLogData.map((item, index) => (
              <div className="log_content_single">
                <p className="log_content_single_line"></p>
                <RightOutlined
                  className="log_content_single_arrow"
                  style={{ display: serviceOpenLog === index + 1 ? 'none' : 'block' }}
                  onClick={() => (serviceOpenLog ? setServiceOpenLog() : setServiceOpenLog(index + 1))}
                />
                <DownOutlined
                  className="log_content_single_arrow"
                  style={{ display: serviceOpenLog === index + 1 ? 'block' : 'none' }}
                  onClick={() => (serviceOpenLog ? setServiceOpenLog() : setServiceOpenLog(index + 1))}
                />
                <p
                  className={`log_content_single_log ${serviceOpenLog === index + 1 ? 'log_content_single_log_select' : ''}`}
                >
                  {item}
                </p>
              </div>
            ))
          ) : (
            <p className="log_content_nolog">{intl.formatMessage({ id: 'log.noLog' })}</p>
          )}
        </div>
      </div>
    </div>
  );
}
