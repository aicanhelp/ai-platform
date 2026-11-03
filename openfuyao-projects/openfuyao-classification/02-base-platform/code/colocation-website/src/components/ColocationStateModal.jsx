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
import { useIntl } from 'inula-intl';
/**
 *
 * @param open 控制对话框打开
 * @param cancelFn 取消函数回调
 * @returns
 */
export default function ColocationStateModal({ title, open, cancelFn, confirmFn, state }) {
  const intl = useIntl();
  return (
    <>
      <Modal
        className="modal_flex_colocation"
        title={title}
        open={open}
        onOk={confirmFn}
        onCancel={cancelFn}
        getContainer={false}
        destroyOnClose
        width="452px"
        style={{
          top: '220px',
        }}
        footer={[
          <div className="modal_flex_colocation_state_bottom">
            <div>
              <Button className="delete_btn" style={{ marginRight: '8px' }} onClick={confirmFn}>
                {intl.formatMessage({ id: 'common.confirm' })}
              </Button>
              <Button className="cancel_btn" onClick={cancelFn}>
                {intl.formatMessage({ id: 'common.cancel' })}
              </Button>
            </div>
          </div>,
        ]}
      >
        <div className="modal_flex_colocation_state_content">
          <ExclamationCircleFilled className="delete_icon" />
          <div className="word_tograry">
            <p className="tograry_text">{intl.formatMessage({ id: 'message.confirmChangeState' }, { state })}</p>
          </div>
        </div>
      </Modal>
    </>
  );
}
