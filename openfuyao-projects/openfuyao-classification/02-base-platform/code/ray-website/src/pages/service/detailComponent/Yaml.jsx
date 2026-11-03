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
import useLocalStorage from '@/hooks/useLocalStorage';
import '@/styles/rayServiceDetail.less';
import AddYaml from '@/components/AddYaml';
import { message } from 'antd';
import { getServiceDetail } from '@/api/rayApi.js';
import { useIntl } from 'inula-intl';
export default function Yaml() {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [messageApi, contextHolder] = message.useMessage();
  const location = useLocation();
  const serviceName = location.state.name;
  const serviceNameSpace = location.state.namespace;
  const [serviceDetailYamlData, setServiceDetailYamlData] = useState();
  const getServiceInfo = async () => {
    try {
      const res = await getServiceDetail(serviceNameSpace, serviceName);
      if (res.data.code === ResponseCode.OK) {
        setServiceDetailYamlData(res.data.data.config_detail);
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
  useEffect(() => {
    getServiceInfo();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div>
      {contextHolder}
      {serviceDetailYamlData ? (
        <AddYaml isShowBtn={false} ifEditCode={false} parentYaml={serviceDetailYamlData} />
      ) : (
        intl.formatMessage({ id: 'common.noData' })
      )}
    </div>
  );
}
