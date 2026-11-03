// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.
import { Form, Input, Select } from 'antd';
import { useEffect } from 'openinula';

export function CollectionCreateForm({ initialValues, onFormInstanceReady }) {
  const [form] = Form.useForm();

  useEffect(() => {
    onFormInstanceReady(form);
  }, []);

  return (
    <Form
      layout='vertical'
      form={form}
      name='form_in_modal'
      initialValues={initialValues}
    >
      <div className='form_container'>
        <div
          style={{
            display: 'flex',
            gap: '20px',
            justifyContent: 'space-between',
          }}
        >
          <Form.Item
            className='log_search'
            name='title'
            label='告警ID'
            rules={[
              {
                required: true,
                message: 'Please input the title of collection!',
              },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item name='name' label='告警名称' className='log_search'>
            <Input />
          </Form.Item>
        </div>
        <Form.Item name='description' label='告警表达式'>
          <Input.TextArea />
        </Form.Item>

        <div>
          <Form.Item
            label='持续时间'
            style={{ width: '50%' }}
            className='collection-create-form_last-form-item'
          >
            <div style={{ display: 'flex', gap: '8px' }}>
              <Input style={{ width: '46%' }} />
              <Select
                style={{ width: '46%' }}
                options={[
                  {
                    value: 'second',
                    label: '秒',
                  },
                  {
                    value: 'minutes',
                    label: '分',
                  },
                  {
                    value: 'hour',
                    label: '时',
                  },
                ]}
              />
            </div>
          </Form.Item>
        </div>

        <div className='flex_between'>
          <Form.Item
            className='log_search'
            name='title'
            label='告警标签'
            rules={[
              {
                required: true,
                message: 'Please input the title of collection!',
              },
            ]}
          >
            <Input style={{ width: '320px', height: '32px' }} />
          </Form.Item>
          <Form.Item name='name' label='告警注解' className='log_search'>
            <Input style={{ width: '320px', height: '32px' }} />
          </Form.Item>
        </div>
      </div>
    </Form>
  );
}
