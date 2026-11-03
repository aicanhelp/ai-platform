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
import { SyncOutlined, MinusCircleOutlined } from '@ant-design/icons';
import { Form, Table, Space, Input, Button, message } from 'antd';

import {
  getNodeListData,
  getNodesAndPodsCount,
  deleteColocationNodesById,
  addColocationNodes,
  getNonColocationNodeListData,
} from '../api/colocationApi';
import '../styles/colocationNodeManagementPage.less';
import OverviewEcharts from '../components/OverviewEcharts';
import { sorterFirstAlphabe } from '../tools/utils';

export default function ColocationNodeManagementPage() {
  const intl = useIntl();
  const param = useParams();
  const [colocationNodeForm] = Form.useForm();
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [messageApi, contextHolder] = message.useMessage();
  // top
  const [nodeDataOption, setNodeDataOption] = useState();
  const [allNodesCount, setAllNodesCount] = useState(0);
  const [colocationNodesCount, setColocationNodesCount] = useState(0);
  // table部分
  const [originalList, setOriginalList] = useState([]); // 原始数据
  const [colocationNodeLoading, setColocationNodeLoading] = useState(false);
  const [colocationNodeList, setColocationNodeList] = useState([]); // 工作负载table数据集
  const [colocationNodePage, setColocationNodePage] = useState(1);
  // 添加弹窗
  const [colocationTableModal, setColocationTableModal] = useState(false);
  const [modelColocationNodelist, setModelColocationNodelist] = useState([]);
  // 删除提醒弹窗
  const [deleteModal, setDeleteModal] = useState(false);
  const [deleteName, setDeleteName] = useState('');
  // 列表项
  const workLoadColumns = [
    {
      title: intl.formatMessage({ id: 'colocation.nodeName' }),
      key: 'workLoad_name',
      dataIndex: 'name',
      width: 300,
      sorter: (a, b) => sorterFirstAlphabe(a.name, b.name),
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'colocation.ipAddress' }),
      key: 'workLoad_tune',
      width: 200,
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
      key: 'workLoad_cup',
      width: 200,
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
      title: intl.formatMessage({ id: 'colocation.setNonColocation' }),
      key: 'handle',
      width: 100,
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
      title: intl.formatMessage({ id: 'colocation.nodeName' }),
      key: 'workLoad_name',
      dataIndex: 'name',
      sorter: (a, b) => sorterFirstAlphabe(a.name, b.name),
      render: (_, record) => <Space>{record.name}</Space>,
    },
    {
      title: intl.formatMessage({ id: 'colocation.ipAddress' }),
      key: 'workLoad_tune',
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
      key: 'workLoad_cup',
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
  ];

  // 获取colocationNodeList
  const getColocationNodeList = useCallback(async (isChange = true) => {
    setColocationNodeLoading(true);
    getColocationNodeManageInfo();
    try {
      // 此处接口
      const res = await getNodeListData();
      if (res.status === ResponseCode.OK) {
        const arr = res.data.data.map((item, index) => {
          return {
            ...item,
            key: item.uid,
          };
        });
        setModelColocationNodelist(arr);
        setOriginalList(arr);
        handleSearchColocationNode(arr, isChange); // 先搜索
      } else {
        messageApi.error('接口异常');
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
        setModelColocationNodelist(arr);
      } else {
        messageApi.error('接口异常');
      }
    } catch (e) {
      setModelColocationNodelist(arr); // model数组为空
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
  };

  // 混部model的确定按钮
  const handleNodeConfirmBack = async (data) => {
    // 代接口
    try {
      const res = await addColocationNodes(data);
      if (res.status === ResponseCode.OK) {
        messageApi.success('添加成功');
      } else {
        messageApi.error('添加失败');
      }
    } catch (e) {
      messageApi.error('接口异常');
    }
    setColocationTableModal(false);
    refreshModelData();
    getColocationNodeList();
  };

  // 混部model的取消按钮
  const handleNodeCancel = () => {
    setColocationTableModal(false);
    refreshModelData();
  };

  // 设置非混部的确定按钮
  const handleDeleteConfirmBack = async () => {
    // 此处接口
    try {
      const res = await deleteColocationNodesById(deleteName);
      if (res.status === ResponseCode.OK) {
        messageApi.success('非混部设置成功');
        getColocationNodeList();
      } else {
        messageApi.error('非混部设置失败');
      }
    } catch (e) {
      messageApi.error('接口异常');
    }
    setDeleteModal(false);
  };

  // 设置非混部的取消按钮
  const handleDeleteCancel = () => {
    setDeleteModal(false);
    setColocationTableModal(false);
  };

  // 重置按钮
  const handleColocationNodeReset = () => {
    getColocationNodeList();
  };

  // 获取顶部echarts以及节点个数数据
  const getColocationNodeManageInfo = useCallback(async () => {
    try {
      // 此处代接口
      const res = await getNodesAndPodsCount();
      if (res.status === ResponseCode.OK) {
        // 节点个数赋值
        setAllNodesCount(res.data.data.totalNodeCount);
        setColocationNodesCount(res.data.data.colocationNodeCount);
        //  处理echarts数据
        let _arr = [];
        let onlineCount = res.data.data.colocationOnlinePodCount;
        let offlineCount = res.data.data.colocationOfflinePodCount;
        let allCount = 0;
        allCount = res.data.data.totalPodCount;

        _arr.push({ name: '在线Pod', value: onlineCount, percent: onlineCount / allCount });
        _arr.push({ name: '离线Pod', value: offlineCount, percent: offlineCount / allCount });

        let appSeries = [
          {
            name: '',
            type: 'pie',
            radius: ['50%', '70%'],
            center: ['30%', '60%'],
            label: {
              show: false,
              useHTML: true,
              fontSize: 16,
              textAlign: 'center',
              position: 'center',
            },
            labelLine: {
              show: false,
            },
            data: _arr,
            itemStyle: {
              color: (params) => {
                let colors = ['#f4840c', '#77b0f7'];
                return colors[params.dataIndex % colors.length];
              },
            },
          },
        ];
        let appOption = {
          tooltip: {
            trigger: 'item',
            borderColor: '#fff',
            borderWidth: 1,
          },
          legend: {
            right: '5%',
            top: '40%',
            orient: '',
            borderWidth: 0,
            itemGap: 20,
            itemWidth: 8,
            itemHeight: 8,
            icon: 'circle',
            formatter: (name) => {
              let _name = '';
              appSeries[0].data.forEach((item) => {
                if (item.name === name) {
                  _name = `${item.name}` + `     ` + `${item.value}`;
                }
              });
              return _name;
            },
          },
          series: appSeries,
        };
        setNodeDataOption(appOption);
      } else {
        messageApi.error('接口异常');
      }
    } catch (e) {
      setNodeDataOption([]);
    }
  }, []);

  useEffect(() => {
    getColocationNodeList();
  }, [getColocationNodeList]);

  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: '在离线混部 / 混部节点管理', path: `/${containerRouterPrefix}/colocation/ColocationNodeManagement` },
        ]}
      />
      {contextHolder}
      <div className="hybrid_detail_content">
        <div className="hybrid_detail_content_top">
          <div className="hybrid_detail_top_node_manage_content">
            <div className="hybrid_detail_top_node_manage_content_image">
              <img src={nodeManagementIcon} />
            </div>
            <div className="hybrid_detail_top_node_manage_content_count">
              <div className="hybrid_detail_top_node_manage_content_count_top">
                <span className="first_span">{colocationNodesCount}</span>
                <span className="nomarl_span">/</span>
                <span className="nomarl_span">{allNodesCount}</span>
              </div>
              <div className="hybrid_detail_top_node_manage_content_count_bottom">
                <span className="nomarl_span">混部节点</span>
                <span className="nomarl_span">/</span>
                <span className="nomarl_span">总节点</span>
              </div>
            </div>
          </div>

          <div className="hybrid_detail_top_echart_content">
            <div className="hybrid_flex_node_progress_chart">
              <OverviewEcharts overOption={nodeDataOption} echartHeight={'180px'} echartWidth={'300px'} />
            </div>
          </div>
        </div>

        <div className="hybrid_detail_content_box">
          <div className="hybrid_detail_form">
            <Form className="hybrid-searchForm form_padding_bottom" form={colocationNodeForm}>
              <Form.Item name="colocationNode_name" className="hybrid-search-input">
                <Input.Search
                  placeholder="搜索节点名称"
                  onSearch={() => handleSearchColocationNode()}
                  autoComplete="off"
                  maxLength={53}
                />
              </Form.Item>
              <Form.Item>
                <Space>
                  <Button className="hybrid_detail_form_button" onClick={() => openButton()}>
                    添加混部节点
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
              pagination={{
                className: 'colocation-website-page',
                current: colocationNodePage,
                showTotal: (total) => `共${total}条`,
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
        title="添加混部节点"
        open={colocationTableModal}
        cancelFn={handleNodeCancel}
        tableColumns={modelColumns}
        dataSource={modelColocationNodelist}
        rowSelection={rowSelection}
        confirmFn={handleNodeConfirmBack}
        selectedRows={selectedRows}
        refresh={refreshModelData}
      />

      <ColocationDeleteModal
        title="设置为非混部节点"
        open={deleteModal}
        cancelFn={handleDeleteCancel}
        content={[`确定将混部节点 ${deleteName} 设置为非混部节点吗？`]}
        confirmFn={handleDeleteConfirmBack}
      />
    </div>
  );
}
