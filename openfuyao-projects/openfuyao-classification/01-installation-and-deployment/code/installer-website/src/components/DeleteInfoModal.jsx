/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Button, Checkbox } from 'antd';
import { ExclamationCircleFilled } from '@ant-design/icons';
import '@/styles/components/deleteInfoModal.less';
import { Fragment } from 'openinula';
import { useIntl } from '../i18n';
import useButtonLoading from '../hooks/useButtonLoading';

/**
 *
 * @param title 对话框标题
 * @param open 控制对话框打开
 * @param cancelFn 取消函数回调
 * @param content 对话框内容 数组！
 * @param confirmFn 确认回调
 * @param confirmText 确认按钮文本 默认可不填
 * @param showCheck 是否展示检查框(默认不展示)
 * @param isCheck 是否选中(默认不展示)
 * @param checkFn 选中回调(默认不展示)
 * @returns
 */
export default function DeleteInfoModal({
  title,
  open,
  cancelFn,
  content,
  confirmFn,
  cancelText,
  confirmBtnID = '',
  cancelBtnID = '',
  confirmText,
  showCheck = false,
  isCheck = false,
  checkText,
  checkFn = () => {},
  specialText = '',
  isSpecialCheck = false,
  specialCheckFn = () => {},
  showSpecialCheck = false,
}) {
  const intl = useIntl();
  const [confirmLoading, setConfirmLoading, confirmPending] = useButtonLoading();

  const handleConfirm = async () => {
    if (confirmPending) return;
    setConfirmLoading(true);
    try {
      await confirmFn();
    } catch (error) {
      // 调用方未自行捕获异常时兜底，避免 unhandled rejection 导致按钮卡死
    } finally {
      setConfirmLoading(false);
    }
  };

  // Apply i18n defaults if not provided
  const finalCancelText = cancelText ?? intl.formatMessage({ id: 'delete.cancelText' });
  const finalConfirmText = confirmText ?? intl.formatMessage({ id: 'delete.confirmText' });
  const finalCheckText = checkText ?? intl.formatMessage({ id: 'delete.checkText' });
  const judgeIsDisable = () => {
    return showSpecialCheck ? !(isCheck && isSpecialCheck) : !isCheck;
  };

  return (
    <Modal
      className="modal_flex_delete"
      open={open}
      title={title}
      onCancel={cancelFn}
      destroyOnHidden
      getContainer={false}
      footer={[
        <Fragment>
          <Button
            id={confirmBtnID}
            className={`${judgeIsDisable() || confirmPending ? 'disabled_btn' : 'delete_btn'}`}
            style={{ width: '96px' }}
            onClick={handleConfirm}
            loading={confirmLoading}
            disabled={judgeIsDisable() || confirmPending}
          >
            {finalConfirmText}
          </Button>
          <Button id={cancelBtnID} className={`cancel_btn`} style={{ width: '96px' }} onClick={cancelFn} disabled={confirmPending}>
            {finalCancelText}
          </Button>
        </Fragment>,
      ]}
    >
      <div className="modal_delete_content">
        <ExclamationCircleFilled className="warn_icon" />
        <div className="word_tograry">
          <div className="word_tograry_alarm">
            {content.map((item) => (
              <p className="bread_word_delete">{item}</p>
            ))}
          </div>
          {!!showCheck && (
            <Checkbox
              className="multi_checkbox"
              checked={isCheck}
              onChange={(e) => {
                checkFn(e);
              }}
            >
              <span>{finalCheckText}</span>
            </Checkbox>
          )}
          {!!showSpecialCheck && (
            <Checkbox
              className="multi_checkbox"
              checked={isSpecialCheck}
              onChange={(e) => {
                specialCheckFn(e);
              }}
            >
              <span>{specialText}</span>
            </Checkbox>
          )}
        </div>
      </div>
    </Modal>
  );
}
