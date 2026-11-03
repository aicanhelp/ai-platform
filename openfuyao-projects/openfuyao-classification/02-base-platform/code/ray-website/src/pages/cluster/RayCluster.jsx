/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import {
  containerRouterPrefix,
  ResponseCode,
  DEFAULT_CURRENT_PAGE,
  getFilterClusterStatusList,
  getFilterTypeList,
} from '@/common/constants';
import { useHistory, Link } from 'inula-router';
import { useCallback, useEffect, useState } from 'openinula';
import { Input, Space, Button, Dropdown, Table, Popover, message, Form } from 'antd';
import { DownOutlined, UpOutlined, MoreOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import UploadConfigInfoModal from '@/components/UploadConfig';
import { sorterFirstAlphabet, filterClusterStatusValue, yamlTojson } from '@/tools/utils';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import Dayjs from 'dayjs';

import '@/styles/rayService.less';
import {
  getClusterList,
  deleteClusterList,
  addCluster,
  stopClusterList,
  restartClusterList,
  getClusterDetail,
} from '@/api/rayApi.js';
import { useIntl } from 'inula-intl';
export default function RayCluster() {
  const intl = useIntl();
  const filterClusterStatusList = getFilterClusterStatusList(intl);
  const filterTypeList = getFilterTypeList(intl);
  const [clusterForm] = Form.useForm();
  const history = useHistory();
  const creator = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).name : '';
  const [messageApi, contextHolder] = message.useMessage();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [isShowConfig, setIsShowConfig] = useState(false);
  const [popOpen, setPopOpen] = useState(''); // 悬浮框是否展示
  const items = [
    {
      key: '1',
      label: <div onClick={() => goAddConig()}>{intl.formatMessage({ id: 'ray.cluster.addConfig' })}</div>,
    },
    {
      key: '2',
      label: (
        <div onClick={() => setIsClusterUpload(true)}>{intl.formatMessage({ id: 'ray.cluster.uploadConfig' })}</div>
      ),
    },
  ];
  const [rayClusterPage, setRayClusterPage] = useState(DEFAULT_CURRENT_PAGE);
  const [isClusterUpload, setIsClusterUpload] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [filterVersion, setFilterVersion] = useState();
  const [filterVersionList, setFilterVersionList] = useState([]);
  const [filterCreator, setFilterCreator] = useState();
  const [filterType, setFilterType] = useState();
  const [filterCreatorList, setFilterCreatorList] = useState([]);
  const [filterStatus, setFilterStatus] = useState();
  const [originalList, setOriginalList] = useState([]); // 原始数据
  const [rayClusterTableData, setRayClusterTableData] = useState([]);
  const [isClusterDelCheck, setIsClusterDelCheck] = useState(false); // 是否选中
  const [clusterDelModalOpen, setClusterDelModalOpen] = useState(false);
  const [selectClusterNameList, setSelectClusterNameList] = useState([]);
  const [selectClusterNamespaceList, setSelectClusterNamespaceList] = useState([]);
  const [multipleDeleteList, setMultipleDeleteList] = useState([]);
  const handleRayClusterTableChange = useCallback((_pagination, filter, _sorter, extra) => {
    if (extra.action === 'filter') {
      setFilterVersion(filter.ray_version);
      setFilterCreator(filter.create_by);
      setFilterType(filter.template_status);
      setFilterStatus(filter.cluster_status_zhcn);
    }
  }, []);
  const goAddConig = () => {
    history.push('/RayCluster/addConfig');
  };

  const solveGcsData = (data) => {
    let configData = yamlTojson(data.config_detail);
    const headGroupSpec = configData.spec.headGroupSpec;
    const rayHead = headGroupSpec.template.spec.containers.filter((item) => item.name !== 'fluentbit');
    return rayHead[0].ports.filter((item) => item.name === 'gcs-server' || item.name === 'gcs')?.[0]?.containerPort;
  };
  const rayClusterColumns = [
    {
      title: intl.formatMessage({ id: 'ray.cluster.name' }),
      dataIndex: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <Space size="middle">
          <Link
            style={{ wordBreak: 'break-all' }}
            to={{
              pathname: `/RayCluster/detail`,
              state: { name: record.name, namespace: record.namespace, status: record.cluster_status_zhcn },
            }}
          >
            {record.name}
          </Link>
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'ray.cluster.rayVersion' }),
      dataIndex: 'ray_version',
      filters: filterVersionList,
      filterMultiple: false,
      filteredValue: filterVersion && filterVersion.length ? filterVersion : [],
      sorter: (a, b) => sorterFirstAlphabet(a.ray_version, b.ray_version),
      onFilter: (value, record) => record.ray_version.toLowerCase() === value.toLowerCase(),
      render: (_, record) => <p style={{ wordBreak: 'break-all' }}>{record.ray_version || '-'}</p>,
    },
    {
      title: intl.formatMessage({ id: 'common.type' }),
      dataIndex: 'ray_type',
      filters: filterTypeList,
      filterMultiple: false,
      filterSearch: true,
      filteredValue: filterType && filterType.length ? filterType : [],
      sorter: (a, b) => sorterFirstAlphabet(a.template_status, b.template_status),
      onFilter: (value, record) => record.template_status.toLowerCase() === value.toLowerCase(),
      render: (_, record) => record.template_status || '-',
      key: 'template_status',
    },
    {
      title: intl.formatMessage({ id: 'common.status' }),
      dataIndex: 'cluster_status_zhcn',
      filters: filterClusterStatusList,
      filterMultiple: false,
      filteredValue: filterStatus && filterStatus.length ? filterStatus : [],
      sorter: (a, b) => sorterFirstAlphabet(a.cluster_status_zhcn, b.cluster_status_zhcn),
      onFilter: (value, record) => record.cluster_status_zhcn.toLowerCase() === value.toLowerCase(),
      render: (_, record) => (
        <div className="status_content">
          <p className="status" style={{ background: filterClusterStatusValue(record.cluster_status_zhcn) }} />
          <p className="defaultRayCluster">{record.cluster_status_zhcn || '-'}</p>
        </div>
      ),
    },
    {
      title: intl.formatMessage({ id: 'ray.cluster.ip' }),
      sorter: (a, b) => sorterFirstAlphabet(a.cluster_ip, b.cluster_ip),
      dataIndex: 'cluster_ip',
      render: (_, record) => record.cluster_ip || '-',
    },
    {
      title: intl.formatMessage({ id: 'ray.cluster.gcsPort' }),
      dataIndex: 'gcs_port',
      sorter: (a, b) => sorterFirstAlphabet(a.gcs_port, b.gcs_port),
      render: (_, record) => solveGcsData(record) || record.gcs_port || '-',
    },
    {
      title: intl.formatMessage({ id: 'ray.cluster.dashboardPort' }),
      dataIndex: 'dashboard_port',
      sorter: (a, b) => sorterFirstAlphabet(a.dashboard_port, b.dashboard_port),
      render: (_, record) => record.dashboard_port || '-',
    },
    {
      title: intl.formatMessage({ id: 'common.creator' }),
      dataIndex: 'create_by',
      filters: filterCreatorList,
      filterMultiple: false,
      filterSearch: true,
      filteredValue: filterCreator && filterCreator.length ? filterCreator : [],
      sorter: (a, b) => sorterFirstAlphabet(a.create_by, b.create_by),
      onFilter: (value, record) => record.create_by.toLowerCase() === value.toLowerCase(),
      render: (_, record) => record.create_by || '-',
      key: 'create_by',
    },
    {
      title: intl.formatMessage({ id: 'common.createTime' }),
      dataIndex: 'create_time',
      key: 'defaultRayCluster',
      sorter: (a, b) => sorterFirstAlphabet(a.create_time, b.create_time),
      render: (_, record) => record.create_time || '-',
    },
    {
      title: intl.formatMessage({ id: 'common.updateTime' }),
      key: 'defaultRayCluster',
      dataIndex: 'update_time',
      sorter: (a, b) => sorterFirstAlphabet(a.update_time, b.update_time),
      render: (_, record) => record.update_time || '-',
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      key: 'handle',
      fixed: 'right',
      width: 120,
      render: (_, record) => (
        <Space>
          <Popover
            className="defaultRayCluster"
            placement="bottom"
            getPopupContainer={() => document.getElementById('rayServiceTable')}
            content={
              <div className="pop_modal">
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn !== '失败' && record.cluster_status_zhcn !== '未运行'}
                  onClick={() => handleStartCluster([record])}
                >
                  {intl.formatMessage({ id: 'common.start' })}
                </Button>
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn === '未运行'}
                  onClick={() => handleStopCluster([record])}
                >
                  {intl.formatMessage({ id: 'common.stop' })}
                </Button>
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn === '运行中'}
                  onClick={() => handleDeleteCluster([record])}
                >
                  {intl.formatMessage({ id: 'common.delete' })}
                </Button>
              </div>
            }
            trigger="click"
            open={popOpen === record.name}
            onOpenChange={(newOpen) => (newOpen ? setPopOpen(record.name) : setPopOpen(''))}
          >
            <MoreOutlined style={{ color: '#3f66f5' }} className={`common_antd_icon`} />
          </Popover>
        </Space>
      ),
    },
  ];

  const getClusterTable = async (isChange = true) => {
    try {
      const res = await getClusterList();
      if (res.data.code === ResponseCode.OK) {
        if (res.data.data.items) {
          let versionArr = [];
          let versionOption = {};
          let creatorArr = [];
          let creatorOption = {};
          let _data = res.data.data.items;
          for (let item of _data) {
            const extraResponse = await getClusterDetail(item.namespace, item.name);
            if (extraResponse.data.code === ResponseCode.OK) {
              item.config_detail = extraResponse.data.data.config_detail;
            }
          }
          _data.map((item, index) => {
            item.key = index;
            if (!Object.keys(versionOption).includes(item.ray_version)) {
              versionOption[item.ray_version] = item.ray_version;
            }
            if (!Object.keys(creatorOption).includes(item.create_by)) {
              creatorOption[item.create_by] = item.create_by;
            }
          });
          const versionKey = Object.keys(versionOption);
          versionKey.forEach((item) => {
            versionArr.push({ text: item, value: versionOption[item] });
          });
          setFilterVersionList([...versionArr]);
          const creatorKey = Object.keys(creatorOption);
          creatorKey.forEach((item) => {
            creatorArr.push({ text: item, value: creatorOption[item] });
          });
          setFilterCreatorList([...creatorArr]);
          setOriginalList(_data);
          handleSearchRayCluster(_data, isChange); // 先搜索
        } else {
          setRayClusterTableData([]); // 数组为空
        }
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(msg);
      }
    } catch (error) {
      setRayClusterTableData([]); // 数组为空
      if (error) {
        let msg = '';
        msg = error.response.data.data
          ? `${error.response.data.msg}:${error.response.data.data}`
          : `${error.response.data.msg}`;
        messageApi.error(msg);
      }
    }
  };

  // 检索
  const handleSearchRayCluster = (totalData = originalList, isChange = true) => {
    let temporyList = totalData;
    const clusterFormName = clusterForm.getFieldValue('cluster_name');
    if (clusterFormName) {
      temporyList = totalData.filter((item) => item.name.toLowerCase().includes(clusterFormName.toLowerCase()));
    } else {
      temporyList = totalData;
    }
    temporyList.map((item, index) => {
      item.key = index;
    });
    setRayClusterTableData([...temporyList]);
    if (isChange) {
      setRayClusterPage(DEFAULT_CURRENT_PAGE);
    } else {
      setRayClusterPage(rayClusterPage);
    }
  };
  //终止、删除、启动后重置页面
  const resetClusterPage = () => {
    setMultipleDeleteList([]);
    setTimeout(() => {
      getClusterTable();
    }, 1000);
    setSelectedRowKeys();
    setRayClusterPage(DEFAULT_CURRENT_PAGE);
    clusterForm.setFieldsValue('cluster_name', '');
  };

  //启动接口
  const handleStartCluster = (clusterData) => {
    let startClusterNameList = [];
    let startClusterNamespaceList = [];
    clusterData.map((item) => {
      startClusterNameList.push(item.name);
      startClusterNamespaceList.push(item.namespace);
    });
    setSelectClusterNameList(startClusterNameList);
    setSelectClusterNamespaceList(startClusterNamespaceList);
    const jobStartList = [];
    startClusterNameList.forEach((item, index) => {
      jobStartList.push(restartClusterList(startClusterNamespaceList[index], item));
      return item;
    });
    Promise.allSettled([jobStartList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(
                  intl.formatMessage({ id: 'ray.cluster.startSuccess' }, { name: startClusterNameList[index] }),
                );
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.startFail' })},${msg}`);
              }
            })
            .catch((error) => {
              if (error) {
                let msg = '';
                msg =
                  JSON.stringify(error.response.data.data) !== '{}'
                    ? `${error.response.data.msg + ':' + error.response.data.data}`
                    : error.response.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.startFail' })},${msg}`);
              }
            });
        });
      });
      resetClusterPage();
    });
  };

  // 终止接口
  const handleStopCluster = (clusterData) => {
    console.log(clusterData);
    let stopClusterNameList = [];
    let stopClusterNamespaceList = [];
    clusterData.map((item) => {
      stopClusterNameList.push(item.name);
      stopClusterNamespaceList.push(item.namespace);
    });
    setSelectClusterNameList(stopClusterNameList);
    setSelectClusterNamespaceList(stopClusterNamespaceList);
    const clusterStopList = [];
    stopClusterNameList.forEach((item, index) => {
      clusterStopList.push(stopClusterList(stopClusterNamespaceList[index], item));
      return item;
    });
    Promise.allSettled([clusterStopList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(
                  intl.formatMessage({ id: 'ray.cluster.stopSuccess' }, { name: stopClusterNameList[index] }),
                );
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.stopFail' })},${msg}`);
              }
            })
            .catch((error) => {
              let msg = '';
              msg =
                JSON.stringify(error.response.data.data) !== '{}'
                  ? `${error.response.data.msg + ':' + error.response.data.data}`
                  : error.response.data.msg;
              messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.stopFail' })},${msg}`);
            });
        });
      });
      resetClusterPage();
    });
  };
  // 取消删除
  const handleDelClusterCancel = () => {
    setClusterDelModalOpen(false);
    setIsClusterDelCheck(false);
  };
  // 确认删除按钮删除
  const handleClusterCheckFn = (e) => {
    setIsClusterDelCheck(e.target.checked);
  };
  // 打开删除弹窗
  const handleDeleteCluster = (clusterData) => {
    let _deleteNameList = [];
    let _deleteNamespaceList = [];
    clusterData.map((item) => {
      _deleteNameList.push(item.name);
      _deleteNamespaceList.push(item.namespace);
    });
    setSelectClusterNameList(_deleteNameList);
    setSelectClusterNamespaceList(_deleteNamespaceList);
    setClusterDelModalOpen(true);
  };
  // 删除接口
  const deleteCluster = async () => {
    const deleterList = [];
    selectClusterNameList.map((item, index) => {
      deleterList.push(deleteClusterList(selectClusterNamespaceList[index], item));
    });
    Promise.allSettled([deleterList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(
                  intl.formatMessage({ id: 'ray.cluster.deleteSuccess' }, { name: selectClusterNameList[index] }),
                );
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.deleteFail' })},${msg}`);
              }
            })
            .catch((error) => {
              let msg = '';
              msg =
                JSON.stringify(error.response.data.data) !== '{}'
                  ? `${error.response.data.msg + ':' + error.response.data.data}`
                  : error.response.data.msg;
              messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.deleteFail' })},${msg}`);
            });
        });
      });
      resetClusterPage();
      setTimeout(() => {
        setClusterDelModalOpen(false);
      }, 500);
      setIsClusterDelCheck(false);
    });
  };

  const rowSelection = {
    selectedRowKeys,
    onChange: (selectedRowKeys, selectedRows) => {
      setSelectedRowKeys(selectedRowKeys);
      setMultipleDeleteList(selectedRows);
    },
  };
  const handleChangeConfig = (open) => {
    // 展示创建下拉菜单
    setIsShowConfig(open);
  };
  const cancelFnUpload = () => {
    // 关闭上传配置弹窗
    setIsClusterUpload(false);
  };
  const getClusterUpload = async (data) => {
    // 获取上传配置文件
    const appFormData = new FormData();
    appFormData.append('config', data);
    appFormData.append('createBy', creator);
    appFormData.append('isStart', true);
    try {
      const res = await addCluster(appFormData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.cluster.createSuccess' }));
        setTimeout(() => {
          setIsClusterUpload(false);
        }, 500);
        getClusterTable();
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.createFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.createFail' })},${msg}`);
      }
    }
  };
  useEffect(() => {
    getClusterTable();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: 'openFuyao Ray', disabled: true },
          { title: 'RayCluster', path: `/RayCluster` },
        ]}
      />
      {contextHolder}
      <div className="service_content">
        <div className="service_table">
          <div className="service_table_top">
            <Form form={clusterForm}>
              <Form.Item name="cluster_name">
                <Input.Search
                  placeholder={intl.formatMessage({ id: 'ray.cluster.searchPlaceholder' })}
                  onSearch={() => handleSearchRayCluster()}
                  autoComplete="off"
                  style={{ width: 200 }}
                />
              </Form.Item>
            </Form>
            <div>
              <Dropdown menu={{ items }} open={isShowConfig} onOpenChange={handleChangeConfig} placement="bottom">
                <Button className="primary_btn">
                  {intl.formatMessage({ id: 'ray.cluster.createBtn' })}
                  {isShowConfig ? <UpOutlined /> : <DownOutlined />}{' '}
                </Button>
              </Dropdown>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleDeleteList.length}
                onClick={() => handleStartCluster(multipleDeleteList)}
              >
                {intl.formatMessage({ id: 'common.start' })}
              </Button>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleDeleteList.length}
                onClick={() => handleStopCluster(multipleDeleteList)}
              >
                {intl.formatMessage({ id: 'common.stop' })}
              </Button>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleDeleteList.length}
                onClick={() => handleDeleteCluster(multipleDeleteList)}
              >
                {intl.formatMessage({ id: 'common.delete' })}
              </Button>
            </div>
          </div>
          <Table
            locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
            id="rayServiceTable"
            rowSelection={rowSelection}
            columns={rayClusterColumns}
            dataSource={rayClusterTableData}
            onChange={handleRayClusterTableChange}
            style={{ overflow: 'scroll' }}
            pagination={{
              className: 'page',
              current: rayClusterPage,
              showTotal: (total) => intl.formatMessage({ id: 'common.total' }, { total }),
              showSizeChanger: true,
              showQuickJumper: true,
              pageSizeOptions: [10, 20, 50],
              onChange: (page) => setRayClusterPage(page),
            }}
          />
        </div>
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'ray.cluster.deleteTitle' })}
        open={clusterDelModalOpen}
        cancelFn={handleDelClusterCancel}
        content={[
          intl.formatMessage({ id: 'ray.cluster.deleteWarning' }),
          intl.formatMessage(
            { id: 'ray.cluster.deleteConfirm' },
            {
              name: selectClusterNameList[0],
              extra: selectClusterNameList.length > 1 ? intl.formatMessage({ id: 'delete.multipleItems' }) : '',
            },
          ),
        ]}
        showCheck={true}
        isCheck={isClusterDelCheck}
        checkFn={handleClusterCheckFn}
        confirmFn={deleteCluster}
      />
      <UploadConfigInfoModal
        open={isClusterUpload}
        downloadType={'raycluster'}
        title={intl.formatMessage({ id: 'ray.cluster.uploadConfig' })}
        cancelText={intl.formatMessage({ id: 'common.cancel' })}
        cancelFn={cancelFnUpload}
        confirmText={intl.formatMessage({ id: 'upload.confirmText' })}
        uploadText={intl.formatMessage({ id: 'upload.selectFile' })}
        downloadText={intl.formatMessage({ id: 'upload.downloadTemplate' }, { type: 'RayCluster' })}
        uploadtip={intl.formatMessage({ id: 'upload.sizeLimit' })}
        size={1}
        type={'yaml'}
        propsServiceUpload={getClusterUpload}
      />
    </div>
  );
}
