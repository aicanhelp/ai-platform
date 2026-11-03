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

import { Modal, Button, Table, Input, message, Form } from 'antd';
import { useEffect, useState, useCallback } from 'openinula';

import { useIntl } from 'inula-intl';
/**
 *
 * @param open 控制对话框打开
 * @param cancelFn 取消函数回调
 * @returns
 */
export default function colocationTableModal({
  title,
  open,
  cancelFn,
  tableColumns,
  dataSource,
  rowSelection,
  confirmFn,
  selectedRows,
  refresh,
  // search,
}) {
  const intl = useIntl();
  const [finalList, setFinalList] = useState([]);
  const [messageApi, contextHolder] = message.useMessage();
  const [hybridInnerForm] = Form.useForm();
  const beforeConfirmFn = (rowSelectedData) => {
    if (selectedRows.length > 10) {
      messageApi.error(intl.formatMessage({ id: 'colocation.maxNodes' }));
    } else {
      hybridInnerForm.setFieldValue('hybrid_inner_name', '');
      confirmFn(rowSelectedData);
    }
  };
  const beforeCancelFn = () => {
    hybridInnerForm.setFieldValue('hybrid_inner_name', '');
    cancelFn();
  };
  // 内部检索
  const handleSearchPod = (totalData = dataSource) => {
    const hybridFormName = hybridInnerForm.getFieldValue('hybrid_inner_name');
    let temporyList = totalData;
    if (hybridFormName) {
      temporyList = temporyList.filter((item) => item.name.toLowerCase().includes(hybridFormName.toLowerCase()));
    }
    setFinalList([...temporyList]);
  };

  // 初始化
  useEffect(() => {
    setFinalList(dataSource);
  }, [dataSource]);

  return (
    <>
      <Modal
        className="modal_flex_colocation"
        title={title}
        open={open}
        onOk={beforeConfirmFn}
        onCancel={beforeCancelFn}
        getContainer={false}
        width="720px"
        footer={[
          <div className="modal_flex_colocation_bottom">
            <div>
              <p className="modal_flex_colocation_bottom_p">
                {intl.formatMessage({ id: 'colocation.selectedNodes' }, { count: selectedRows.length })}
              </p>
            </div>
            <div>
              <Button className="back_cancel_btn" onClick={beforeCancelFn}>
                {intl.formatMessage({ id: 'common.cancel' })}
              </Button>
              <Button className="back_config_btn" onClick={() => beforeConfirmFn(selectedRows)}>
                {intl.formatMessage({ id: 'common.confirm' })}
              </Button>
            </div>
          </div>,
        ]}
      >
        {contextHolder}
        <Form form={hybridInnerForm}>
          <Form.Item name="hybrid_inner_name" className="modal_flex_colocation_form">
            <Input.Search
              placeholder={intl.formatMessage({ id: 'colocation.searchNodeName' })}
              className="modal_flex_search"
              onSearch={() => handleSearchPod()}
              autoComplete="off"
              maxLength={53}
            />
          </Form.Item>
        </Form>
        <Table
          locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
          scroll={{ y: 450 }}
          columns={tableColumns}
          dataSource={finalList}
          pagination={false}
          rowkey={(source) => dataSource.chart}
          rowSelection={{
            type: 'checkbox',
            preserveSelectedRowKeys: true,
            ...rowSelection,
          }}
        />
      </Modal>
    </>
  );
}
