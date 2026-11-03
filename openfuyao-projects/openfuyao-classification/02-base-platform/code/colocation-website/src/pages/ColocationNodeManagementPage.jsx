/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '../components/BreadCrumbCom';
import ColocationTableModal from '../components/ColocationTableModal';
import ColocationDeleteModal from '../components/ColocationDeleteModal';
import { containerRouterPrefix, ResponseCode } from '../common/constants';
import { useParams } from 'inula-router';
import { useEffect, useState, useCallback } from 'openinula';
import { useIntl } from 'inula-intl';
import { SyncOutlined, MinusCircleOutlined, CloseOutlined } from '@ant-design/icons';
import { Form, Table, Space, Input, Button, message, Switch } from 'antd';

import {
  getNodeListData,
  getNodesAndPodsCount,
  deleteColocationNodesById,
  addColocationNodes,
  getNonColocationNodeListData,
  setOneNodeOversold,
} from '../api/colocationApi';
import '../styles/colocationNodeManagementPage.less';
import { sorterFirstAlphabe } from '../tools/utils';
import useLocalStorage from '@/hooks/useLocalStorage';
import ToolTipComponent from '@/components/ToolTipComponent';
import { filterStatusName } from '../utils/common';

export default function ColocationNodeManagementPage() {
  const intl = useIntl();
  const [currentTheme] = useLocalStorage('theme', 'light');
  const [isShow, setIsShow] = useState(true);
  const [colocationNodeForm] = Form.useForm();
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [messageApi, contextHolder] = message.useMessage();
  // table部分
  const [originalList, setOriginalList] = useState([]); // 原始数据
  const [colocationNodeLoading, setColocationNodeLoading] = useState(false);
  const [colocationNodeList, setColocationNodeList] = useState([]); // 工作负载table数据集
  const [colocationNodePage, setColocationNodePage] = useState(1);
  const [filterNodeStatus, setFilterNodeStatus] = useState([
    { text: intl.formatMessage({ id: 'node.status.normal' }), value: '正常' },
    { text: intl.formatMessage({ id: 'node.status.abnormal' }), value: '异常' },
  ]); // 筛选项
  const [filterNodeValue, setFilterNodeValue] = useState(); // 筛选值
  // 添加弹窗
  const [colocationTableModal, setColocationTableModal] = useState(false);
  const [modelColocationNodelist, setModelColocationNodelist] = useState([]);
  // 删除提醒弹窗
  const [deleteModal, setDeleteModal] = useState(false);
  const [deleteName, setDeleteName] = useState('');
  // 删除勾选
  const [isNodeDelCheck, setIsNodeDelCheck] = useState(false); // 是否选中
  // 变混部删除勾选
  const [isColocationNodeDelCheck, setIsColocationNodeDelCheck] = useState(false); // 是否选中
  // 列表项
  const workLoadColumns = [
    {
      title: intl.formatMessage({ id: 'node.name' }),
      key: 'node_name',
      dataIndex: 'name',
      width: 300,
      sorter: (a, b) => sorterFirstAlphabe(a.name, b.name),
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'node.status' }),
      key: 'node_status',
      width: 200,
      filters: filterNodeStatus,
      filterMultiple: false,
      filteredValue: filterNodeValue ? [filterNodeValue] : [],
      onFilter: (value, record) => filterStatusName(record.nodeStatus, intl) === value,
      sorter: (a, b) => sorterFirstAlphabe(a.osImage, b.osImage),
      render: (_, record) => (
        <Space>
          <div className="table_single_box">
            <div className="table_single_box_status_box">
              <span className={`${record.nodeStatus.toLowerCase()}_circle`}></span>
              <span>
                {record.nodeStatus === 'active' || record.nodeStatus === 'inactive'
                  ? filterStatusName(record.nodeStatus, intl)
                  : '--'}
              </span>
            </div>
          </div>
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'node.ipAddress' }),
      key: 'node_tune',
      width: 200,
      sorter: (a, b) => sorterFirstAlphabe(a.addresses, b.addresses),
      render: (_, record) => (
        <>
          {record.addresses?.map((item) => {
            return (
              <div className="table_tuning_box">
                <p>{item}</p>
              </div>
            );
          })}
        </>
      ),
    },
    {
      title: 'OS',
      key: 'node_cup',
      width: 200,
      sorter: (a, b) => sorterFirstAlphabe(a.osImage, b.osImage),
      render: (_, record) => (
        <Space>
          <div className="table_single_box">
            <div className="table_single_box_number_format">
              <span>{record.osImage}</span>
            </div>
          </div>
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'node.setOversold' }),
      key: 'node_setting',
      width: 100,
      sorter: (a, b) => sorterFirstAlphabe(a.isOversold, b.isOversold),
      render: (_, record) => (
        <Switch
          checked={record.isOversold}
          key={record.id}
          onChange={() => handleSwitchChange(record.uid, record['oversubscription-status'])}
        />
      ),
    },
    {
      title: intl.formatMessage({ id: 'node.setNonColocation' }),
      key: 'handle',
      width: 150,
      render: (_, record) => (
        <Space>
          <MinusCircleOutlined
            className="table_handle"
            onClick={() => {
              handleDeleteNode(record);
            }}
          />
        </Space>
      ),
    },
  ];

  // 弹窗列表项
  const modelColumns = [
    {
      title: intl.formatMessage({ id: 'node.name' }),
      key: 'node_name',
      dataIndex: 'name',
      sorter: (a, b) => sorterFirstAlphabe(a.name, b.name),
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'node.ipAddress' }),
      key: 'node_tune',
      sorter: (a, b) => sorterFirstAlphabe(a.addresses, b.addresses),
      render: (_, record) => (
        <>
          {record.addresses?.map((item) => {
            return (
              <div className="table_tuning_box">
                <p>{item}</p>
              </div>
            );
          })}
        </>
      ),
    },
    {
      title: 'OS',
      key: 'node_cup',
      sorter: (a, b) => sorterFirstAlphabe(a.osImage, b.osImage),
      render: (_, record) => (
        <Space>
          <div className="table_single_box">
            <div className="table_single_box_number_format">
              <span>{record.osImage}</span>
            </div>
          </div>
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'node.setOversold' }),
      key: 'node_setting',
      render: (text, record) => (
        <Space>
          <Switch
            checked={record.isOversold}
            key={record.id}
            disabled={record.name === 'fuyao-master'}
            onChange={() => handleSwitchChangeToNoColocationList(record.uid, record.isOversold)}
          />
        </Space>
      ),
    },
  ];

  const handleShowChange = () => {
    setIsShow(false);
  };

  const handleSwitchChange = async (id, checked) => {
    const ary = JSON.parse(JSON.stringify(colocationNodeList));
    let dataObj = {};
    ary.forEach((item) => {
      if (item.uid === id) {
        dataObj = item;
        if (item.isOversold === true) {
          item['oversubscription-status'] = 'false';
        } else {
          item['oversubscription-status'] = 'true';
        }
      }
    });
    // 此处直接调接口修改混部节点列表isOversold的状态后刷新页面
    try {
      const res = await setOneNodeOversold(dataObj.name, dataObj['oversubscription-status']);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'node.setSuccess' }));
      } else {
        messageApi.error(intl.formatMessage({ id: 'node.setFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setColocationNodeList(ary);
    getColocationNodeList();
  };

  // 设置非混部table的switch开关
  const handleSwitchChangeToNoColocationList = (id, checked) => {
    const ary = JSON.parse(JSON.stringify(modelColocationNodelist));
    ary.forEach((item) => {
      if (item.uid === id) {
        item.isOversold = !item.isOversold;
      }
    });
    // 数组对比取交集改变选中行的开关状态
    const intersection = ary.filter((obj1) => selectedRows.some((obj2) => obj2.uid === obj1.uid));
    setModelColocationNodelist(ary);
    setSelectedRows(intersection);
  };

  // 获取colocationNodeList
  const getColocationNodeList = useCallback(async (isChange = true) => {
    setColocationNodeLoading(true);
    try {
      // 此处接口
      const res = await getNodeListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data?.data?.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        const sorterArr = arr.sort((a, b) => b.nodeStatus.localeCompare(a.nodeStatus));
        setOriginalList(sorterArr);
        handleSearchColocationNode(sorterArr, isChange); // 先搜索
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setColocationNodeList([]); // 数组为空
    }
    setColocationNodeLoading(false);
  }, []);

  // 获取非混部节点列表
  const getNonColocationNodeList = useCallback(async () => {
    try {
      // 此处接口
      const res = await getNonColocationNodeListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data.data.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        const sorterArr = arr.sort((a, b) => b.nodeStatus.localeCompare(a.nodeStatus));
        setModelColocationNodelist(sorterArr);
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setModelColocationNodelist([]); // model数组为空
    }
  }, []);

  // 外部检索
  const handleSearchColocationNode = (totalData = originalList, isChange = true) => {
    const colocationNodeFormName = colocationNodeForm.getFieldValue('colocationNode_name');
    let temporyList = totalData;
    if (colocationNodeFormName) {
      temporyList = temporyList.filter((item) =>
        item.name.toLowerCase().includes(colocationNodeFormName.toLowerCase()),
      );
    }
    setColocationNodeList([...temporyList]);
    isChange ? setColocationNodePage(1) : null;
  };

  const handleNodeTableChange = useCallback((_pagination, filter, _sorter, extra) => {
    if (extra.action === 'filter') {
      setFilterNodeValue(filter.node_status);
    }
  }, []);

  // 混部节点移除
  const handleDeleteNode = async (record) => {
    setDeleteModal(true);
    setDeleteName(record.name);
  };

  const refreshModelData = () => {
    setSelectedRowKeys([]);
    setSelectedRows([]);
  };

  // 打开model
  const openButton = () => {
    setColocationTableModal(true);
    getNonColocationNodeList();
  };

  // 混部选择
  const rowSelection = {
    selectedRowKeys,
    onChange: (newSelectedRowKeys, newselectedRows) => {
      setSelectedRowKeys(newSelectedRowKeys);
      setSelectedRows(newselectedRows);
    },
    getCheckboxProps: (record) => ({
      name: record.name,
      disabled: record.nodeStatus === 'failed',
    }),
  };

  // 混部model的确定按钮
  const handleNodeConfirmBack = async (data) => {
    // 代接口
    try {
      const res = await addColocationNodes(data);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'node.addSuccess' }));
      } else {
        messageApi.error(intl.formatMessage({ id: 'node.addFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setColocationTableModal(false);
    setIsColocationNodeDelCheck(false);
    refreshModelData();
    getColocationNodeList();
  };

  // 混部model的取消按钮
  const handleNodeCancel = () => {
    setColocationTableModal(false);
    setIsColocationNodeDelCheck(false);
    refreshModelData();
  };

  // 设置非混部的确定按钮
  const handleDeleteConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationNodesById(deleteName);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'message.nonColocationSetSuccess' }));
        getColocationNodeList();
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.nonColocationSetFailed' }));
      }
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
    }
    setDeleteModal(false);
    setIsNodeDelCheck(false);
  };

  // 设置非混部的取消按钮
  const handleDeleteCancel = () => {
    setDeleteModal(false);
    setIsNodeDelCheck(false);
    setColocationTableModal(false);
  };

  // 重置按钮
  const handleColocationNodeReset = () => {
    getColocationNodeList();
  };

  // 勾选回调
  const handleNodeCheckFn = (e) => {
    setIsNodeDelCheck(e.target.checked);
  };

  // 非混部列表勾选回调
  const handleColocationNodeCheckFn = (e) => {
    setIsColocationNodeDelCheck(e.target.checked);
  };

  useEffect(() => {
    getColocationNodeList();
  }, [getColocationNodeList]);

  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'colocation.title' }), path: `/ColocationNodeManagement` },
          { title: intl.formatMessage({ id: 'node.management' }), path: `/` },
        ]}
      />
      {contextHolder}
      <div className="hybrid_detail_content">
        <div className={`${isShow ? '' : 'cant_show'}`}>
          <ToolTipComponent>
            <div
              className="daemonsetDetail"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div className="promot-box">{intl.formatMessage({ id: 'node.tipMessage' })}</div>
              <CloseOutlined onClick={handleShowChange} />
            </div>
          </ToolTipComponent>
        </div>
        <div className="hybrid_detail_content_box">
          <div className={` ${isShow ? 'tooltip_container_height' : 'normal_container_height'}`}>
            <div className="hybrid_detail_form">
              <Form className="hybrid-searchForm form_padding_bottom" form={colocationNodeForm}>
                <Form.Item name="colocationNode_name" className="hybrid-search-input">
                  <Input.Search
                    placeholder={intl.formatMessage({ id: 'node.searchName' })}
                    onSearch={() => handleSearchColocationNode()}
                    autoComplete="off"
                    maxLength={53}
                  />
                </Form.Item>
                <Form.Item>
                  <Space>
                    <Button className="hybrid_detail_form_button" onClick={openButton}>
                      {intl.formatMessage({ id: 'node.addColocation' })}
                    </Button>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleColocationNodeReset}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                </Form.Item>
              </Form>
            </div>
            <div className="hybrid_detail_content_table">
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                loading={colocationNodeLoading}
                columns={workLoadColumns}
                dataSource={colocationNodeList}
                onChange={handleNodeTableChange}
                pagination={{
                  className: 'colocation-website-page',
                  current: colocationNodePage,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                  onChange: (page) => setColocationNodePage(page),
                }}
              ></Table>
            </div>
          </div>
        </div>

        <ColocationTableModal
          title={intl.formatMessage({ id: 'node.addColocation' })}
          open={colocationTableModal}
          cancelFn={handleNodeCancel}
          tableColumns={modelColumns}
          dataSource={modelColocationNodelist}
          rowSelection={rowSelection}
          confirmFn={handleNodeConfirmBack}
          selectedRows={selectedRows}
          refresh={refreshModelData}
          isCheck={isColocationNodeDelCheck}
          checkFn={handleColocationNodeCheckFn}
        />

        <ColocationDeleteModal
          title={intl.formatMessage({ id: 'node.setNonColocation' })}
          open={deleteModal}
          cancelFn={handleDeleteCancel}
          content={[intl.formatMessage({ id: 'node.confirmDelete' }, { name: deleteName })]}
          confirmFn={handleDeleteConfirmBack}
          isCheck={isNodeDelCheck}
          showCheck={true}
          checkFn={handleNodeCheckFn}
        />
      </div>
    </div>
  );
}
