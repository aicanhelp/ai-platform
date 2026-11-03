/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { useEffect, useState } from 'openinula';
import { Radio } from 'antd';
import useLocalStorage from '@/hooks/useLocalStorage';
import '@/styles/rayServiceAdd.less';
import RayJobAddForm from './RayJobAddForm';
import RayJobAddYaml from './RayJobAddYaml';
import { useIntl } from 'inula-intl';
export default function RayJobAddConfig() {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [mode, setMode] = useState('form');
  const handleJobModeChange = (e) => {
    setMode(e.target.value);
  };
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: 'openFuyao Ray', disabled: true },
          { title: 'RayJob', path: `/RayJob` },
          { title: intl.formatMessage({ id: 'config.createTitle' }), path: `/addConfig` },
        ]}
      />
      <div className="rayServiceAdd">
        <Radio.Group onChange={handleJobModeChange} value={mode}>
          <Radio.Button value="form">{intl.formatMessage({ id: 'config.formCreate' })}</Radio.Button>
          <Radio.Button value="yaml">{intl.formatMessage({ id: 'config.yamlCreate' })}</Radio.Button>
        </Radio.Group>
        {mode === 'form' ? <RayJobAddForm /> : <RayJobAddYaml />}
      </div>
    </div>
  );
}
