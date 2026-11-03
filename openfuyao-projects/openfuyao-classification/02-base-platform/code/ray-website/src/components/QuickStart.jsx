/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Button, Tabs } from 'antd';
import '@/styles/components/quickstartModal.less';
import { Fragment } from 'openinula';
import useLocalStorage from '@/hooks/useLocalStorage';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useIntl } from 'inula-intl';
/**
 *
 * @param isQuickOpen 控制对话框打开
 * @param onCancelQuick 取消函数回调
 * @return
 */
export default function QuickStart({ isQuickOpen, onCancelQuick }) {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const paramsYamlData1 = `  pip install -U "ray"`;
  const paramsYamlData2 = `  import ray
  ray.init()

  @ray.remote
  def f(x):
      return x * x

  futures = [f.remote(i) for i in range(4)]
  print(ray.get(futures)) # [0,1,4,9]  `;
  const items = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'quickstart.whatIsRay' }),
      children: (
        <div>
          <p>{intl.formatMessage({ id: 'quickstart.rayDescription' })}</p>
        </div>
      ),
    },
    {
      key: '2',
      label: intl.formatMessage({ id: 'quickstart.howToUseRay' }),
      children: (
        <div>
          <p className="quickstart_text">{intl.formatMessage({ id: 'quickstart.installInstruction' })}</p>
          <SyntaxHighlighter language="javascript" style={dark}>
            {paramsYamlData1}
          </SyntaxHighlighter>
          <p className="quickstart_text">{intl.formatMessage({ id: 'quickstart.usageInstruction' })}</p>
          <SyntaxHighlighter language="javascript" style={dark}>
            {paramsYamlData2}
          </SyntaxHighlighter>
        </div>
      ),
    },
  ];
  return (
    <Modal
      getContainer={false}
      width={682}
      height={560}
      title={intl.formatMessage({ id: 'quickstart.welcome' })}
      className={`modal_flex_quickstart ${currentTheme === 'dark' ? 'dark_box' : ''}`}
      open={isQuickOpen}
      onCancel={onCancelQuick}
      destroyOnClose
      footer={[
        <Fragment>
          <Button onClick={onCancelQuick} className="cancel_btn">
            {intl.formatMessage({ id: 'common.close' })}
          </Button>
        </Fragment>,
      ]}
    >
      <Tabs defaultActiveKey="1" items={items} />
    </Modal>
  );
}
