/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import MarketCodeMirrorEditor from '@/components/MarketCodeMirrorEditor';
import { exportYamlOutPut } from '@/tools/utils';
import { ExportOutlined, CopyOutlined } from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import { message, Skeleton } from 'antd';
import { useStore } from 'openinula';
import NoContent from '@/assets/images/noContent.png';
import { useTranslation } from 'react-i18next';

export default function DefaultParameter({ defaultParams, chart, loading }) {
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const { t } = useTranslation(['appMarket', 'common']);

  const exportYaml = () => {
    try {
      exportYamlOutPut(chart, defaultParams);
      messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
    } catch {
      messageApi.error(t('messages.exportFailed'));
    }
  };

  const handleCopyYaml = () => {
    try {
      copy(defaultParams);
      messageApi.success(t('messages.copySuccess', { ns: 'common' }));
    } catch {
      messageApi.error(t('messages.copyFailed'));
    }
  };

  if (!defaultParams) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', height: '100%', padding: '1px 32px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', alignItems: 'center', minHeight: '400px', justifyContent: 'center' }}>
          <div><img src={NoContent} alt={t('messages.noDefaultParams')} /></div>
          <div style={{ color: '#89939b', fontSize: '14px' }}>{t('messages.noDefaultParams')}</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '400px', padding: '32px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className="helm_yaml_card">
        <div className="yaml_flex_box">
          <h3>YAML</h3>
          <div className="yaml_tools">
            <div className="tool_word_group defaultParameterExport" onClick={exportYaml}>
              <ExportOutlined className="common_antd_icon primary_color" />
              <span>{t('deploy.export')}</span>
            </div>
            <div className="tool_word_group defaultParameterCopy" onClick={handleCopyYaml}>
              <CopyOutlined className="common_antd_icon primary_color" />
              <span>{t('deploy.copy')}</span>
            </div>
          </div>
        </div>
      </div>
      {loading ? (
        <Skeleton active paragraph={{ rows: 10 }} />
      ) : (
        <MarketCodeMirrorEditor yamlData={defaultParams} className='fit_content' isEdit={false} />
      )}
    </div>
  );
}
