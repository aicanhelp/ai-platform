/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
 *       http://license.coscl.org.cn/MulanPSL2
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import { useState, useStore } from 'openinula';
import { useParams } from 'inula-router';
import { CheckCircleFilled } from '@ant-design/icons';
import { message } from 'antd';
import { useTranslation } from 'react-i18next';
import ApplicationDetailLayout from '@/components/layouts/ApplicationDetailLayout';
import ExtendDetailInfo from '@/pages/container/extendManage/extend/detail/ExtendDetailInfo';
import ExtendDetailYaml from '@/pages/container/extendManage/extend/detail/ExtendDetailYaml';
import ExtendDetailResource from '@/pages/container/extendManage/extend/detail/ExtendDetailResource';
import ExtendDetailLog from '@/pages/container/extendManage/extend/detail/ExtendDetailLog';
import ExtendDetailEvent from '@/pages/container/extendManage/extend/detail/ExtendDetailEvent';
import ExtendDetailMonitor from '@/pages/container/extendManage/extend/detail/ExtendDetailMonitor';

export default function ExtendDetail() {
  const param = useParams();
  const { t } = useTranslation('application');
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();

  const extendName = param.extend_name;
  const extendNamespace = param.extend_namespace;

  // Extension specific state for plugin toggle
  const [isViewMode, setIsEditNow] = useState(true);
  const [pluginShow, setPluginShow] = useState(false);

  // Refresh page
  const refreshAllPage = () => {
    window.location.reload();
  };

  // Handle plugin toggle edit
  const handlePluginToggle = (bool) => {
    setIsEditNow(bool);
  };

  // Handle save refresh
  const handleOkRefresh = () => {
    messageApi.open({
      duration: 10,
      content: (
        <div className='extend_customize_message'>
          <CheckCircleFilled className='extend_customize_message_icon' />
          <p className='extend_customize_context' style={{ color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
            {t('plugin.extensionUpdated')}
          </p>
          <p className='extend_customize_operate' onClick={refreshAllPage}>{t('plugin.clickRefresh')}</p>
        </div>
      ),
    });
  };

  // Handle plugin show change
  const handlePluginShowChange = (value) => {
    setPluginShow(value);
  };

  // Info component with props
  const infoComponent = (
    <ExtendDetailInfo
      extendName={extendName}
      isViewMode={isViewMode}
      handleEditFn={handlePluginToggle}
      handleOkRefreshFn={handleOkRefresh}
      handlePluginShow={handlePluginShowChange}
    />
  );

  // YAML component with props
  const yamlComponent = (
    <ExtendDetailYaml
      extendName={extendName}
      extendNamespace={extendNamespace}
    />
  );

  // Resource component with props
  const resourceComponent = (
    <ExtendDetailResource />
  );

  // Log component with props
  const logComponent = (
    <ExtendDetailLog
      extendName={extendName}
      extendNamespace={extendNamespace}
    />
  );

  // Event component with props
  const eventComponent = (
    <ExtendDetailEvent
      extendName={extendName}
      extendNamespace={extendNamespace}
    />
  );

  // Monitor component with props
  const monitorComponent = (
    <ExtendDetailMonitor />
  );

  return (
    <>
      <div style={{
        background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff',
        color: themeStore.$s.theme === 'dark' ? '#fff' : '#333',
      }}>
        {contextHolder}
      </div>
      <ApplicationDetailLayout
        type="extension"
        infoComponent={infoComponent}
        yamlComponent={yamlComponent}
        resourceComponent={resourceComponent}
        logComponent={logComponent}
        eventComponent={eventComponent}
        monitorComponent={monitorComponent}
        onPluginToggle={handlePluginToggle}
        isViewMode={isViewMode}
        pluginShow={pluginShow}
      />
    </>
  );
}