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

import { Modal, Button } from 'antd';
import { ExclamationCircleFilled } from '@ant-design/icons';
import { Fragment } from 'openinula';
import { useIntl } from 'inula-intl';

/**
 *
 * @param title 对话框标题
 * @param open 控制对话框打开
 * @param cancelFn 取消函数回调
 * @param content 对话框内容 数组！
 * @param confirmFn 确认回调
 * @returns
 */
export default function ColocationDeleteModal({ title, open, cancelFn, content, confirmFn }) {
  const intl = useIntl();
  return (
    <Modal
      className="modal_flex_delete"
      open={open}
      title={title}
      onCancel={cancelFn}
      getContainer={false}
      destroyOnClose
      footer={[
        <Fragment>
          <Button className="cancel_btn" onClick={confirmFn}>
            {intl.formatMessage({ id: 'common.confirm' })}
          </Button>
          <Button className="cancel_btn" onClick={cancelFn}>
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
        </Fragment>,
      ]}
    >
      <div className="modal_delete_content">
        <ExclamationCircleFilled className="warn_icon" />
        <div className="word_tograry">
          <div className="word_tograry_alarm">
            {content.map((item) => (
              <p>{item}</p>
            ))}
          </div>
        </div>
      </div>
    </Modal>
  );
}
