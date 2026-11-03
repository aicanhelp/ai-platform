/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix, ResponseCode } from '@/common/constants';
import { useHistory } from 'inula-router';
import { useEffect } from 'openinula';
import { message } from 'antd';
import useLocalStorage from '@/hooks/useLocalStorage';
import AddYaml from '@/components/AddYaml';
import { addCluster } from '@/api/rayApi.js';
import { useIntl } from 'inula-intl';
export default function RayClusterAddYaml() {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [messageApi, contextHolder] = message.useMessage();
  const creator = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).name : '';
  const history = useHistory();
  const getChildYaml = async (isStart, yaml) => {
    const appFormData = new FormData();
    appFormData.append('config', yaml);
    appFormData.append('createBy', creator);
    appFormData.append('isStart', isStart);
    try {
      const res = await addCluster(appFormData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'yaml.createSuccessMsg' }, { type: 'RayCluster' }));
        setTimeout(() => {
          history.push('/RayCluster');
        }, 1500);
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'yaml.createFailMsg' }, { type: 'RayCluster' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'yaml.createFailMsg' }, { type: 'RayCluster' })},${msg}`);
      }
    }
  };
  useEffect(() => {}, [currentTheme]);
  return (
    <div>
      {contextHolder}
      <AddYaml
        path={'/RayCluster'}
        isShowBtn={true}
        ifEditCode={true}
        parentYaml={''}
        yamlName={'RayCluster'}
        getYaml={getChildYaml}
        cancelFn={oncancel}
      />
    </div>
  );
}
