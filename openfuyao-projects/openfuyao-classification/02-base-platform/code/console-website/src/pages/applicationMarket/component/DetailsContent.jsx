/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import copy from 'copy-to-clipboard';
import { useStore, useEffect, useRef } from 'openinula';
import { CopyOutlined } from '@ant-design/icons';
import { message, Tooltip } from 'antd';
import { useTranslation } from 'react-i18next';
export default function DetailsContent({ contentArray, title }) {
  const themeStore = useStore('theme');
  const copiedTimerRef = useRef(null);
  const { t } = useTranslation('appMarket');
  const [messageApi, contextHolder] = message.useMessage();
  useEffect(() => () => clearTimeout(copiedTimerRef.current), []);
  const handleCopy = () => {
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
    clearTimeout(copiedTimerRef.current);
    copiedTimerRef.current = setTimeout(() => {
    }, 1500);
  };

  return (
    <div style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div style={{ padding: '32px 0 0 32px', color: themeStore.$s.theme === 'light' ? '#333' : '#fff', fontSize: '16px', fontWeight: 'bold' }}>{title}</div>
      <div style={{ padding: '20px 0px 0px 64px', display: 'flex' }}>
        <div className='tips_color' style={{ minWidth: '120px', color: themeStore.$s.theme !== 'light' && '#89939b' }}>{`${contentArray[0].title}：`}</div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <Tooltip title={contentArray[0].value}>
            <div className='clamp_3' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{contentArray[0].value}</div>
          </Tooltip>
          <CopyOutlined
            style={{
              color: '#3f66f5ff',
              border: '1px solid rgba(0,0,0,0)',
              cursor: 'pointer',
            }}
            onClick={() => {
              copy(contentArray[0].value);
              handleCopy();
            }}
          />
        </div>
      </div>
      <div className='details_content' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        {contentArray.slice(1, 5).map((item) => (
          <div style={{ display: 'flex' }}>
            <div style={{ display: 'flex', marginBottom: '16px' }}>
              <div className='tips_color' style={{ minWidth: '120px', color: themeStore.$s.theme !== 'light' && '#89939b' }}>{`${item.title}：`}</div>
              <div style={{ display: 'flex', gap: '16px' }}>
                <Tooltip title={item.value}>
                  <div className='clamp_1' style={{ width: '200px', color: themeStore.$s.theme !== 'light' && '#fff' }}>{item.value}</div>
                </Tooltip>
                {item.copyable && (
                  <CopyOutlined
                    style={{
                      color: '#3f66f5ff',
                      border: '1px solid rgba(0,0,0,0)',
                      cursor: 'pointer',
                    }}
                    onClick={() => {
                      copy(item.value);
                      handleCopy();
                    }}
                  />
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
