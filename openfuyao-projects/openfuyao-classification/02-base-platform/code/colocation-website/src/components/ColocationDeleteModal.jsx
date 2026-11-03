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

import { Modal, Button, Checkbox } from 'antd';
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
export default function ColocationDeleteModal({
  title,
  open,
  cancelFn,
  content,
  confirmFn,
  leftText,
  showCheck = false,
  isCheck = false,
  checkFn = () => {},
  checkboxText,
}) {
  const intl = useIntl();
  const defaultLeftText = leftText || intl.formatMessage({ id: 'common.confirm' });
  const defaultCheckboxText = checkboxText || intl.formatMessage({ id: 'message.understandRisk' });
  return (
    <Modal
      className="modal_flex_delete"
      open={open}
      title={title}
      onCancel={cancelFn}
      getContainer={false}
      destroyOnClose
      style={{
        top: '220px',
      }}
      footer={[
        <Fragment>
          <Button
            className={`${!isCheck && showCheck ? 'disabled_btn' : 'delete_btn'}`}
            onClick={confirmFn}
            disabled={!isCheck && showCheck}
          >
            {defaultLeftText}
          </Button>
          <Button className="cancel_btn" onClick={cancelFn}>
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
        </Fragment>,
      ]}
    >
      <div className="modal_delete_content">
        <ExclamationCircleFilled className="warn_icon" />
        <div className="word_format">
          <div className="word_format_alarm">
            {content.map((item) => (
              <p className="format_text">{item}</p>
            ))}
            {!!showCheck && (
              <>
                <Checkbox
                  className="multi_checkbox"
                  checked={isCheck}
                  onChange={(e) => {
                    checkFn(e);
                  }}
                >
                  {defaultCheckboxText}
                </Checkbox>
                <div className="multi_checkbox" style={{ marginLeft: '24px' }}>
                  {intl.formatMessage({ id: 'message.caution' })}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </Modal>
  );
}
