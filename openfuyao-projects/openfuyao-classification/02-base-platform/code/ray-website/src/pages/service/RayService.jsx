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
  getFilterServiceStatusList,
  getFilterTypeList,
} from '@/common/constants';
import { useHistory, Link } from 'inula-router';
import { useCallback, useEffect, useState } from 'openinula';
import { Input, Space, Button, Dropdown, Table, Popover, message, Form } from 'antd';
import { DownOutlined, UpOutlined, MoreOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import UploadConfigInfoModal from '@/components/UploadConfig';
import { sorterFirstAlphabet, filterServiceStatusValue } from '@/tools/utils';
import DeleteInfoModal from '@/components/DeleteInfoModal';

import '@/styles/rayService.less';
import {
  getServiceList,
  getClusterList,
  deleteServiceList,
  addService,
  stopServiceList,
  restartServiceList,
} from '@/api/rayApi.js';
import { useIntl } from 'inula-intl';
export default function RayService() {
  const intl = useIntl();
  const filterServiceStatusList = getFilterServiceStatusList(intl);
  const filterTypeList = getFilterTypeList(intl);
  const history = useHistory();
  const [serviceForm] = Form.useForm();
  const creator = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).name : '';
  const [messageApi, contextHolder] = message.useMessage();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [isShowServiceConfig, setIsShowServiceConfig] = useState(false);
  const [popOpenService, setPopOpenService] = useState(''); // 悬浮框是否展示
  const items = [
    {
      key: '1',
      label: <div onClick={() => goAddConig()}>{intl.formatMessage({ id: 'ray.service.addConfig' })}</div>,
    },
    {
      key: '2',
      label: (
        <div onClick={() => setIsServiceUpload(true)}>{intl.formatMessage({ id: 'ray.service.uploadConfig' })}</div>
      ),
    },
  ];
  const [filterType, setFilterType] = useState();
  const [rayServicePage, setRayServicePage] = useState(DEFAULT_CURRENT_PAGE);
  const [isServiceUpload, setIsServiceUpload] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [filterServiceCreator, setFilterServiceCreator] = useState();
  const [filterCreatorServiceList, setFilterCreatorServiceList] = useState([]);
  const [filterServiceStatus, setFilterServiceStatus] = useState();
  const [originalServiceList, setOriginalServiceList] = useState([]); // 原始数据
  const [rayServiceTableData, setRayServiceTableData] = useState([]);
  const [isServiceDelCheck, setIsServiceDelCheck] = useState(false); // 是否选中
  const [ServiceDelModalOpen, setServiceDelModalOpen] = useState(false);
  const [selectServiceNameList, setSelectServiceNameList] = useState([]);
  const [selectServiceNamespaceList, setSelectServiceNamespaceList] = useState([]);
  const [multipleServiceDeleteList, setMultipleServiceDeleteList] = useState([]);
  const handleRayServiceTableChange = useCallback((_pagination, filter, _sorter, extra) => {
    if (extra.action === 'filter') {
      setFilterServiceCreator(filter.create_by);
      setFilterType(filter.template_status);
      setFilterServiceStatus(filter.cluster_status_zhcn);
    }
  }, []);
  const goAddConig = () => {
    history.push('/RayService/addConfig');
  };
  const rayServiceColumns = [
    {
      title: intl.formatMessage({ id: 'ray.service.name' }),
      dataIndex: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <Space size="middle">
          <Link
            style={{ wordBreak: 'break-all' }}
            to={{
              pathname: `/RayService/detail`,
              state: { name: record.name, namespace: record.namespace, status: record.cluster_status_zhcn },
            }}
          >
            {record.name}
          </Link>
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'common.type' }),
      dataIndex: 'ray_type',
      filters: filterTypeList,
      filterMultiple: false,
      key: 'template_status',
      filterSearch: true,
      filteredValue: filterType && filterType.length ? filterType : [],
      sorter: (a, b) => sorterFirstAlphabet(a.template_status, b.template_status),
      onFilter: (value, record) => record.template_status.toLowerCase() === value.toLowerCase(),
      render: (_, record) => record.template_status || '-',
    },
    {
      title: intl.formatMessage({ id: 'common.status' }),
      dataIndex: 'cluster_status_zhcn',
      filters: filterServiceStatusList,
      filterMultiple: false,
      filteredValue: filterServiceStatus && filterServiceStatus.length ? filterServiceStatus : [],
      sorter: (a, b) => sorterFirstAlphabet(a.cluster_status_zhcn, b.cluster_status_zhcn),
      onFilter: (value, record) => record.cluster_status_zhcn.toLowerCase() === value.toLowerCase(),
      render: (_, record) => (
        <div className="status_content">
          <p className="status" style={{ background: filterServiceStatusValue(record.cluster_status_zhcn) }} />
          <p className="defaultRayService">{record.cluster_status_zhcn || '-'}</p>
        </div>
      ),
    },
    {
      dataIndex: 'cluster_ip',
      title: intl.formatMessage({ id: 'ray.job.clusterIpAddress' }),
      sorter: (a, b) => sorterFirstAlphabet(a.cluster_ip, b.cluster_ip),
      render: (_, record) => record.cluster_ip || '-',
    },
    {
      title: intl.formatMessage({ id: 'common.creator' }),
      dataIndex: 'create_by',
      filters: filterCreatorServiceList,
      filterMultiple: false,
      filterSearch: true,
      filteredValue: filterServiceCreator && filterServiceCreator.length ? filterServiceCreator : [],
      sorter: (a, b) => sorterFirstAlphabet(a.create_by, b.create_by),
      onFilter: (value, record) => record.create_by.toLowerCase() === value.toLowerCase(),
      render: (_, record) => record.create_by || '-',
      key: 'create_by',
    },
    {
      title: intl.formatMessage({ id: 'common.createTime' }),
      dataIndex: 'create_time',
      sorter: (a, b) => sorterFirstAlphabet(a.create_time, b.create_time),
      render: (_, record) => record.create_time || '-',
      key: 'defaultRayService',
    },
    {
      title: intl.formatMessage({ id: 'common.updateTime' }),
      dataIndex: 'update_time',
      sorter: (a, b) => sorterFirstAlphabet(a.update_time, b.update_time),
      render: (_, record) => record.update_time || '-',
      key: 'defaultRayService',
    },
    {
      width: 120,
      key: 'handle',
      title: intl.formatMessage({ id: 'common.action' }),
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Popover
            className="defaultRayService"
            placement="bottom"
            getPopupContainer={() => document.getElementById('rayServiceTable')}
            content={
              <div className="pop_modal">
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn !== '失败' && record.cluster_status_zhcn !== '未运行'}
                  onClick={() => handleStartService([record])}
                >
                  {intl.formatMessage({ id: 'common.start' })}
                </Button>
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn === '未运行'}
                  onClick={() => handleStopService([record])}
                >
                  {intl.formatMessage({ id: 'common.stop' })}
                </Button>
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn === '运行中'}
                  onClick={() => handleDeleteService([record])}
                >
                  {intl.formatMessage({ id: 'common.delete' })}
                </Button>
              </div>
            }
            trigger="click"
            open={popOpenService === record.key}
            onOpenChange={(newOpen) => (newOpen ? setPopOpenService(record.key) : setPopOpenService(''))}
          >
            <MoreOutlined style={{ color: '#3f66f5' }} className={`common_antd_icon`} />
          </Popover>
        </Space>
      ),
    },
  ];

  const getServiceTable = async (isChange = true) => {
    try {
      const res = await getServiceList();
      if (res.data.code === ResponseCode.OK) {
        if (res.data.data.items) {
          let creatorArr = [];
          let creatorOption = {};
          let _data = res.data.data.items;
          _data.map((item, index) => {
            item.key = index;
            if (!Object.keys(creatorOption).includes(item.create_by)) {
              creatorOption[item.create_by] = item.create_by;
            }
          });
          const creatorKey = Object.keys(creatorOption);
          creatorKey.forEach((item) => {
            creatorArr.push({ text: item, value: creatorOption[item] });
          });
          setFilterCreatorServiceList([...creatorArr]);
          setOriginalServiceList(_data);
          handleSearchRayService(_data, isChange); // 先搜索
        } else {
          setRayServiceTableData([]); // 数组为空
        }
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.listError' })},${msg}`);
      }
    } catch (error) {
      setRayServiceTableData([]); // 数组为空
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.listError' })},${msg}`);
      }
    }
  };

  // 检索
  const handleSearchRayService = (totalData = originalServiceList, isChange = true) => {
    let temporyList = totalData;
    const serviceFormName = serviceForm.getFieldValue('service_name');
    if (serviceFormName) {
      temporyList = totalData.filter((item) => item.name.toLowerCase().includes(serviceFormName.toLowerCase()));
    } else {
      temporyList = totalData;
    }
    temporyList.map((item, index) => {
      item.key = index;
    });
    setRayServiceTableData([...temporyList]);
    if (isChange) {
      setRayServicePage(DEFAULT_CURRENT_PAGE);
    } else {
      setRayServicePage(rayServicePage);
    }
  };
  //终止、删除、启动后重置页面
  const resetServicePage = () => {
    setMultipleServiceDeleteList([]);
    setTimeout(() => {
      getServiceTable();
    }, 1000);
    setSelectedRowKeys();
    setRayServicePage(DEFAULT_CURRENT_PAGE);
    serviceForm.setFieldsValue('service_name', '');
  };

  //启动接口
  const handleStartService = (serviceData) => {
    let startServiceNameList = [];
    let startServiceNamespaceList = [];
    serviceData.map((item) => {
      startServiceNameList.push(item.name);
      startServiceNamespaceList.push(item.namespace);
    });
    setSelectServiceNameList(startServiceNameList);
    setSelectServiceNamespaceList(startServiceNamespaceList);
    const serviceStartList = [];
    startServiceNameList.forEach((item, index) => {
      serviceStartList.push(restartServiceList(startServiceNamespaceList[index], item));
      return item;
    });
    Promise.allSettled([serviceStartList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(
                  intl.formatMessage({ id: 'ray.service.startSuccess' }, { name: startServiceNameList[index] }),
                );
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.service.startFail' })},${msg}`);
              }
            })
            .catch((error) => {
              if (error) {
                let msg = '';
                msg =
                  JSON.stringify(error.response.data.data) !== '{}'
                    ? `${error.response.data.msg + ':' + error.response.data.data}`
                    : error.response.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.service.startFail' })},${msg}`);
              }
            });
        });
      });
      resetServicePage();
    });
  };

  // 终止接口
  const handleStopService = (serviceData) => {
    let stopServiceNameList = [];
    let stopServiceNamespaceList = [];
    serviceData.map((item) => {
      stopServiceNameList.push(item.name);
      stopServiceNamespaceList.push(item.namespace);
    });
    setSelectServiceNameList(stopServiceNameList);
    setSelectServiceNamespaceList(stopServiceNamespaceList);
    const serviceStopList = [];
    stopServiceNameList.forEach((item, index) => {
      serviceStopList.push(stopServiceList(stopServiceNamespaceList[index], item));
      return item;
    });
    Promise.allSettled([serviceStopList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(
                  intl.formatMessage({ id: 'ray.service.stopSuccess' }, { name: stopServiceNameList[index] }),
                );
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.service.stopFail' })},${msg}`);
              }
            })
            .catch((error) => {
              if (error) {
                let msg = '';
                msg =
                  JSON.stringify(error.response.data.data) !== '{}'
                    ? `${error.response.data.msg + ':' + error.response.data.data}`
                    : error.response.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.service.stopFail' })},${msg}`);
              }
            });
        });
      });
      resetServicePage();
    });
  };
  // 取消删除
  const handleDelServiceCancel = () => {
    setServiceDelModalOpen(false);
    setIsServiceDelCheck(false);
  };
  // 确认删除按钮删除
  const handleServiceCheckFn = (e) => {
    setIsServiceDelCheck(e.target.checked);
  };
  // 打开删除弹窗
  const handleDeleteService = (jobData) => {
    let _deleteNameList = [];
    let _deleteNamespaceList = [];
    jobData.map((item) => {
      _deleteNameList.push(item.name);
      _deleteNamespaceList.push(item.namespace);
    });
    setSelectServiceNameList(_deleteNameList);
    setSelectServiceNamespaceList(_deleteNamespaceList);
    setServiceDelModalOpen(true);
  };
  // 删除接口
  const deleteService = async () => {
    const deleterList = [];
    selectServiceNameList.map((item, index) => {
      deleterList.push(deleteServiceList(selectServiceNamespaceList[index], item));
    });
    Promise.allSettled([deleterList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(
                  intl.formatMessage({ id: 'ray.service.deleteSuccess' }, { name: selectServiceNameList[index] }),
                );
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.service.deleteFail' })},${msg}`);
              }
            })
            .catch((error) => {
              if (error) {
                let msg = '';
                msg =
                  JSON.stringify(error.response.data.data) !== '{}'
                    ? `${error.response.data.msg + ':' + error.response.data.data}`
                    : error.response.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.service.deleteFail' })},${msg}！`);
              }
            });
        });
      });
      resetServicePage();
      setTimeout(() => {
        setServiceDelModalOpen(false);
      }, 500);
      setIsServiceDelCheck(false);
    });
  };

  const rowSelection = {
    selectedRowKeys,
    onChange: (selectedRowKeys, selectedRows) => {
      setSelectedRowKeys(selectedRowKeys);
      setMultipleServiceDeleteList(selectedRows);
    },
  };
  const handleChangeConfig = (open) => {
    // 展示创建下拉菜单
    setIsShowServiceConfig(open);
  };
  const cancelFnUpload = () => {
    // 关闭上传配置弹窗
    setIsServiceUpload(false);
  };
  const getServiceUpload = async (data) => {
    // 获取上传配置文件
    const appFormData = new FormData();
    appFormData.append('config', data);
    appFormData.append('createBy', creator);
    appFormData.append('isStart', true);
    try {
      const res = await addService(appFormData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.service.createSuccess' }));
        setTimeout(() => {
          setIsServiceUpload(false);
        }, 500);
        getServiceTable();
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.createFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.createFail' })},${msg}！`);
      }
    }
  };
  useEffect(() => {
    getServiceTable();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: 'openFuyao Ray', disabled: true },
          { title: 'RayService', path: `/RayService` },
        ]}
      />
      {contextHolder}
      <div className="service_content">
        <div className="service_table">
          <div className="service_table_top">
            <Form form={serviceForm}>
              <Form.Item name="service_name">
                <Input.Search
                  placeholder={intl.formatMessage({ id: 'ray.service.searchPlaceholder' })}
                  onSearch={() => handleSearchRayService()}
                  autoComplete="off"
                  style={{ width: 200 }}
                />
              </Form.Item>
            </Form>
            <div>
              <Dropdown
                menu={{ items }}
                open={isShowServiceConfig}
                onOpenChange={handleChangeConfig}
                placement="bottom"
              >
                <Button className="primary_btn">
                  {intl.formatMessage({ id: 'ray.service.createBtn' })}
                  {isShowServiceConfig ? <UpOutlined /> : <DownOutlined />}{' '}
                </Button>
              </Dropdown>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleServiceDeleteList.length}
                onClick={() => handleStartService(multipleServiceDeleteList)}
              >
                {intl.formatMessage({ id: 'common.start' })}
              </Button>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleServiceDeleteList.length}
                onClick={() => handleStopService(multipleServiceDeleteList)}
              >
                {intl.formatMessage({ id: 'common.stop' })}
              </Button>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleServiceDeleteList.length}
                onClick={() => handleDeleteService(multipleServiceDeleteList)}
              >
                {intl.formatMessage({ id: 'common.delete' })}
              </Button>
            </div>
          </div>
          <Table
            locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
            style={{ overflow: 'scroll' }}
            id="rayServiceTable"
            rowSelection={rowSelection}
            columns={rayServiceColumns}
            dataSource={rayServiceTableData}
            onChange={handleRayServiceTableChange}
            pagination={{
              className: 'page',
              current: rayServicePage,
              showTotal: (total) => intl.formatMessage({ id: 'common.total' }, { total }),
              showSizeChanger: true,
              showQuickJumper: true,
              pageSizeOptions: [10, 20, 50],
              onChange: (page) => setRayServicePage(page),
            }}
          />
        </div>
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'ray.service.deleteTitle' })}
        open={ServiceDelModalOpen}
        cancelFn={handleDelServiceCancel}
        content={[
          intl.formatMessage({ id: 'ray.service.deleteWarning' }),
          intl.formatMessage(
            { id: 'ray.service.deleteConfirm' },
            {
              name: selectServiceNameList[0],
              extra: selectServiceNameList.length > 1 ? intl.formatMessage({ id: 'delete.multipleItems' }) : '',
            },
          ),
        ]}
        showCheck={true}
        isCheck={isServiceDelCheck}
        checkFn={handleServiceCheckFn}
        confirmFn={deleteService}
      />
      <UploadConfigInfoModal
        open={isServiceUpload}
        downloadType={'rayservice'}
        title={intl.formatMessage({ id: 'ray.service.uploadConfig' })}
        cancelText={intl.formatMessage({ id: 'common.cancel' })}
        cancelFn={cancelFnUpload}
        confirmText={intl.formatMessage({ id: 'upload.confirmText' })}
        uploadText={intl.formatMessage({ id: 'upload.selectFile' })}
        downloadText={intl.formatMessage({ id: 'upload.downloadTemplate' }, { type: 'RayService' })}
        uploadtip={intl.formatMessage({ id: 'upload.sizeLimit' })}
        size={1}
        type={'yaml'}
        propsServiceUpload={getServiceUpload}
      />
    </div>
  );
}
