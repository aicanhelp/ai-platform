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
  getFilterJobStatusList,
  getFilterTypeList,
} from '@/common/constants';
import { useHistory, Link } from 'inula-router';
import { useCallback, useEffect, useState } from 'openinula';
import { Input, Space, Button, Dropdown, Table, Popover, message, Form } from 'antd';
import { DownOutlined, UpOutlined, MoreOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import UploadConfigInfoModal from '@/components/UploadConfig';
import { sorterFirstAlphabet, filterJobStatusValue } from '@/tools/utils';
import DeleteInfoModal from '@/components/DeleteInfoModal';

import '@/styles/rayJob.less';
import { getJobList, deleteJobList, addJob, stopJobList, restartJobList } from '@/api/rayApi.js';
import { useIntl } from 'inula-intl';
export default function RayJob() {
  const intl = useIntl();
  const filterJobStatusList = getFilterJobStatusList(intl);
  const filterTypeList = getFilterTypeList(intl);
  const history = useHistory();
  const [jobForm] = Form.useForm();
  const creator = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).name : '';
  const [messageApi, contextHolder] = message.useMessage();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [isShowJobConfig, setIsShowJobConfig] = useState(false);
  const [popOpenJob, setPopOpenJob] = useState(''); // 悬浮框是否展示
  const items = [
    {
      key: '1',
      label: <div onClick={() => goAddConig()}>{intl.formatMessage({ id: 'ray.job.addConfig' })}</div>,
    },
    {
      key: '2',
      label: <div onClick={() => setIsJobUpload(true)}>{intl.formatMessage({ id: 'ray.job.uploadConfig' })}</div>,
    },
  ];
  const [rayJobPage, setRayJobPage] = useState(DEFAULT_CURRENT_PAGE);
  const [isJobUpload, setIsJobUpload] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [filterJobCreator, setFilterJobCreator] = useState();
  const [filterCreatorJobList, setFilterCreatorJobList] = useState([]);
  const [filterJobStatus, setFilterJobStatus] = useState();
  const [filterType, setFilterType] = useState();
  const [originalJobList, setOriginalJobList] = useState([]); // 原始数据
  const [rayJobTableData, setRayJobTableData] = useState([]);
  const [isJobDelCheck, setIsJobDelCheck] = useState(false); // 是否选中
  const [JobDelModalOpen, setJobDelModalOpen] = useState(false);
  const [selectJobNameList, setSelectJobNameList] = useState([]);
  const [selectJobNamespaceList, setSelectJobNamespaceList] = useState([]);
  const [multipleJobDeleteList, setMultipleJobDeleteList] = useState([]);
  const handleRayJobTableChange = useCallback((_pagination, filter, _sorter, extra) => {
    if (extra.action === 'filter') {
      setFilterJobCreator(filter.create_by);
      setFilterType(filter.template_status);
      setFilterJobStatus(filter.cluster_status_zhcn);
    }
  }, []);
  const goAddConig = () => {
    history.push('/RayJob/addConfig');
  };
  const rayJobColumns = [
    {
      title: intl.formatMessage({ id: 'ray.job.name' }),
      dataIndex: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <Space size="middle">
          <Link
            style={{ wordBreak: 'break-all' }}
            to={{
              pathname: `/RayJob/detail`,
              state: { name: record.name, namespace: record.namespace, status: record.cluster_status_zhcn },
            }}
          >
            {record.name || '-'}
          </Link>
        </Space>
      ),
    },
    {
      title: 'Job ID',
      dataIndex: 'job_id',
      sorter: (a, b) => sorterFirstAlphabet(a.job_id, b.job_id),
      render: (_, record) => record.job_id || '-',
    },
    {
      title: intl.formatMessage({ id: 'ray.job.clusterName' }),
      dataIndex: 'raycluster_name',
      sorter: (a, b) => sorterFirstAlphabet(a.raycluster_name, b.name),
      render: (_, record) => record.raycluster_name || '-',
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
      filters: filterJobStatusList,
      filterMultiple: false,
      filteredValue: filterJobStatus && filterJobStatus.length ? filterJobStatus : [],
      sorter: (a, b) => sorterFirstAlphabet(a.cluster_status_zhcn, b.cluster_status_zhcn),
      onFilter: (value, record) => record.cluster_status_zhcn.toLowerCase() === value.toLowerCase(),
      render: (_, record) => (
        <div className="status_content">
          <p
            className="status defaultRayJob"
            style={{ background: filterJobStatusValue(record.cluster_status_zhcn) }}
          />
          <p>{record.cluster_status_zhcn || '-'}</p>
        </div>
      ),
    },
    {
      title: intl.formatMessage({ id: 'ray.job.clusterIpAddress' }),
      dataIndex: 'cluster_ip',
      sorter: (a, b) => sorterFirstAlphabet(a.dashboard_address, b.dashboard_address),
      render: (_, record) => record.dashboard_address || '-',
    },
    {
      title: intl.formatMessage({ id: 'common.creator' }),
      dataIndex: 'create_by',
      filters: filterCreatorJobList,
      filterMultiple: false,
      filterSearch: true,
      filteredValue: filterJobCreator && filterJobCreator.length ? filterJobCreator : [],
      sorter: (a, b) => sorterFirstAlphabet(a.create_by, b.create_by),
      onFilter: (value, record) => record.create_by.toLowerCase() === value.toLowerCase(),
      render: (_, record) => record.create_by || '-',
      key: 'create_by',
    },
    {
      title: intl.formatMessage({ id: 'common.createTime' }),
      dataIndex: 'create_time',
      sorter: (a, b) => sorterFirstAlphabet(a.create_time, b.create_time),
      key: 'defultRayJob',
      render: (_, record) => record.create_time || '-',
    },
    {
      title: intl.formatMessage({ id: 'common.updateTime' }),
      dataIndex: 'update_time',
      sorter: (a, b) => sorterFirstAlphabet(a.update_time, b.update_time),
      key: 'defultRayJob',
      render: (_, record) => record.update_time || '-',
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      key: 'handle',
      width: 120,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Popover
            className="defultRayJob"
            placement="bottom"
            getPopupContainer={() => document.getElementById('rayJobTable')}
            content={
              <div className="pop_modal">
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn !== '失败' && record.cluster_status_zhcn !== '未运行'}
                  onClick={() => handleStartJob([record])}
                >
                  {intl.formatMessage({ id: 'common.start' })}
                </Button>
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn === '未运行'}
                  onClick={() => handleStopJob([record])}
                >
                  {intl.formatMessage({ id: 'common.stop' })}
                </Button>
                <Button
                  type="link"
                  disabled={record.cluster_status_zhcn === '运行中'}
                  onClick={() => handleDeleteJob([record])}
                >
                  {intl.formatMessage({ id: 'common.delete' })}
                </Button>
              </div>
            }
            trigger="click"
            open={popOpenJob === record.key}
            onOpenChange={(newOpen) => (newOpen ? setPopOpenJob(record.key) : setPopOpenJob(''))}
          >
            <MoreOutlined style={{ color: '#3f66f5' }} className={`common_antd_icon`} />
          </Popover>
        </Space>
      ),
    },
  ];

  const getJobTable = async (isChange = true) => {
    try {
      const res = await getJobList();
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
          setFilterCreatorJobList([...creatorArr]);
          setOriginalJobList(_data);
          handleSearchRayJob(_data, isChange); // 先搜索
        } else {
          setRayJobTableData([]); // 数组为空
        }
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.listError' })},${msg}`);
      }
    } catch (error) {
      setRayJobTableData([]); // 数组为空
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.listError' })},${msg}`);
      }
    }
  };

  // 检索
  const handleSearchRayJob = (totalData = originalJobList, isChange = true) => {
    let temporyList = totalData;
    const jobFormName = jobForm.getFieldValue('job_name');
    if (jobFormName) {
      temporyList = totalData.filter((item) => item.name.toLowerCase().includes(jobFormName.toLowerCase()));
    } else {
      temporyList = totalData;
    }
    temporyList.map((item, index) => {
      item.key = index;
    });
    setRayJobTableData([...temporyList]);
    if (isChange) {
      setRayJobPage(DEFAULT_CURRENT_PAGE);
    } else {
      setRayJobPage(rayJobPage);
    }
  };
  //终止、删除、启动后重置页面
  const resetJobPage = () => {
    setMultipleJobDeleteList([]);
    setTimeout(() => {
      getJobTable();
    }, 1000);
    setSelectedRowKeys();
    setRayJobPage(DEFAULT_CURRENT_PAGE);
    jobForm.setFieldsValue('job_name', '');
  };

  //启动接口
  const handleStartJob = (jobData) => {
    let startJobNameList = [];
    let startJobNamespaceList = [];
    jobData.map((item) => {
      startJobNameList.push(item.name);
      startJobNamespaceList.push(item.namespace);
    });
    setSelectJobNameList(startJobNameList);
    setSelectJobNamespaceList(startJobNamespaceList);
    const jobStartList = [];
    startJobNameList.forEach((item, index) => {
      jobStartList.push(restartJobList(startJobNamespaceList[index], item));
      return item;
    });
    Promise.allSettled([jobStartList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(
                  intl.formatMessage({ id: 'ray.job.startSuccess' }, { name: startJobNameList[index] }),
                );
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.job.startFail' })},${msg}`);
              }
            })
            .catch((error) => {
              if (error) {
                let msg = '';
                msg =
                  JSON.stringify(error.response.data.data) !== '{}'
                    ? `${error.response.data.msg + ':' + error.response.data.data}`
                    : error.response.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.job.startFail' })},${msg}`);
              }
            });
        });
      });
      resetJobPage();
    });
  };

  // 终止接口
  const handleStopJob = (jobData) => {
    let stopJobNameList = [];
    let stopJobNamespaceList = [];
    jobData.map((item) => {
      stopJobNameList.push(item.name);
      stopJobNamespaceList.push(item.namespace);
    });
    setSelectJobNameList(stopJobNameList);
    setSelectJobNamespaceList(stopJobNamespaceList);
    const jobStopList = [];
    stopJobNameList.forEach((item, index) => {
      jobStopList.push(stopJobList(stopJobNamespaceList[index], item));
      return item;
    });
    Promise.allSettled([jobStopList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(intl.formatMessage({ id: 'ray.job.stopSuccess' }, { name: stopJobNameList[index] }));
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.job.stopFail' })},${msg}`);
              }
            })
            .catch((error) => {
              if (error) {
                let msg = '';
                msg =
                  JSON.stringify(error.response.data.data) !== '{}'
                    ? `${error.response.data.msg + ':' + error.response.data.data}`
                    : error.response.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.job.stopFail' })},${msg}`);
              }
            });
        });
      });
      resetJobPage();
    });
  };
  // 取消删除
  const handleDelJobCancel = () => {
    setJobDelModalOpen(false);
    setIsJobDelCheck(false);
  };
  // 确认删除按钮删除
  const handleJobCheckFn = (e) => {
    setIsJobDelCheck(e.target.checked);
  };
  // 打开删除弹窗
  const handleDeleteJob = (jobData) => {
    let _deleteNameList = [];
    let _deleteNamespaceList = [];
    jobData.map((item) => {
      _deleteNameList.push(item.name);
      _deleteNamespaceList.push(item.namespace);
    });
    setSelectJobNameList(_deleteNameList);
    setSelectJobNamespaceList(_deleteNamespaceList);
    setJobDelModalOpen(true);
  };
  // 删除接口
  const deleteJob = async () => {
    const deleterList = [];
    selectJobNameList.map((item, index) => {
      deleterList.push(deleteJobList(selectJobNamespaceList[index], item));
    });
    Promise.allSettled([deleterList]).then((res) => {
      res.forEach((res1) => {
        res1.value.forEach((item, index) => {
          item
            .then((data) => {
              if (data.data.code === ResponseCode.OK) {
                messageApi.success(
                  intl.formatMessage({ id: 'ray.job.deleteSuccess' }, { name: selectJobNameList[index] }),
                );
              } else {
                let msg = '';
                msg =
                  JSON.stringify(data.data.data) !== '{}' ? `${data.data.msg + ':' + data.data.data}` : data.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.job.deleteFail' })},${msg}`);
              }
            })
            .catch((error) => {
              if (error) {
                let msg = '';
                msg =
                  JSON.stringify(error.response.data.data) !== '{}'
                    ? `${error.response.data.msg + ':' + error.response.data.data}`
                    : error.response.data.msg;
                messageApi.error(`${intl.formatMessage({ id: 'ray.job.deleteFail' })},${msg}`);
              }
            });
        });
      });
      resetJobPage();
      setTimeout(() => {
        setJobDelModalOpen(false);
      }, 500);
      setIsJobDelCheck(false);
    });
  };

  const rowSelection = {
    selectedRowKeys,
    onChange: (selectedRowKeys, selectedRows) => {
      setSelectedRowKeys(selectedRowKeys);
      setMultipleJobDeleteList(selectedRows);
    },
  };
  const handleChangeConfig = (open) => {
    // 展示创建下拉菜单
    setIsShowJobConfig(open);
  };
  const cancelFnUpload = () => {
    // 关闭上传配置弹窗
    setIsJobUpload(false);
  };
  const getJobUpload = async (data) => {
    // 获取上传配置文件
    const appFormData = new FormData();
    appFormData.append('config', data);
    appFormData.append('createBy', creator);
    appFormData.append('isStart', true);
    try {
      const res = await addJob(appFormData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.job.createSuccess' }));
        setTimeout(() => {
          setIsJobUpload(false);
        }, 500);
        getJobTable();
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.createFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.createFail' })},${msg}`);
      }
    }
  };
  useEffect(() => {
    getJobTable();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: 'openFuyao Ray', disabled: true },
          { title: 'RayJob', path: `/RayJob` },
        ]}
      />
      {contextHolder}
      <div className="service_content">
        <div className="service_table">
          <div className="service_table_top">
            <Form form={jobForm}>
              <Form.Item name="job_name">
                <Input.Search
                  placeholder={intl.formatMessage({ id: 'ray.job.searchPlaceholder' })}
                  onSearch={() => handleSearchRayJob()}
                  autoComplete="off"
                  style={{ width: 200 }}
                />
              </Form.Item>
            </Form>
            <div>
              <Dropdown menu={{ items }} open={isShowJobConfig} onOpenChange={handleChangeConfig} placement="bottom">
                <Button className="primary_btn">
                  {intl.formatMessage({ id: 'ray.job.createBtn' })}
                  {isShowJobConfig ? <UpOutlined /> : <DownOutlined />}{' '}
                </Button>
              </Dropdown>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleJobDeleteList.length}
                onClick={() => handleStartJob(multipleJobDeleteList)}
              >
                {intl.formatMessage({ id: 'common.start' })}
              </Button>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleJobDeleteList.length}
                onClick={() => handleStopJob(multipleJobDeleteList)}
              >
                {intl.formatMessage({ id: 'common.stop' })}
              </Button>
              <Button
                className="cancel_btn ml16"
                disabled={!multipleJobDeleteList.length}
                onClick={() => handleDeleteJob(multipleJobDeleteList)}
              >
                {intl.formatMessage({ id: 'common.delete' })}
              </Button>
            </div>
          </div>
          <Table
            locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
            id="rayJobTable"
            style={{ overflow: 'scroll' }}
            rowSelection={rowSelection}
            columns={rayJobColumns}
            dataSource={rayJobTableData}
            onChange={handleRayJobTableChange}
            pagination={{
              className: 'page',
              current: rayJobPage,
              showTotal: (total) => intl.formatMessage({ id: 'common.total' }, { total }),
              showSizeChanger: true,
              showQuickJumper: true,
              pageSizeOptions: [10, 20, 50],
              onChange: (page) => setRayJobPage(page),
            }}
          />
        </div>
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'ray.job.deleteTitle' })}
        open={JobDelModalOpen}
        cancelFn={handleDelJobCancel}
        content={[
          intl.formatMessage({ id: 'ray.job.deleteWarning' }),
          intl.formatMessage(
            { id: 'ray.job.deleteConfirm' },
            {
              name: selectJobNameList[0],
              extra: selectJobNameList.length > 1 ? intl.formatMessage({ id: 'delete.multipleItems' }) : '',
            },
          ),
        ]}
        showCheck={true}
        isCheck={isJobDelCheck}
        checkFn={handleJobCheckFn}
        confirmFn={deleteJob}
      />
      <UploadConfigInfoModal
        open={isJobUpload}
        downloadType={'rayjob'}
        title={intl.formatMessage({ id: 'ray.job.uploadConfig' })}
        cancelText={intl.formatMessage({ id: 'common.cancel' })}
        cancelFn={cancelFnUpload}
        confirmText={intl.formatMessage({ id: 'upload.confirmText' })}
        uploadText={intl.formatMessage({ id: 'upload.selectFile' })}
        downloadText={intl.formatMessage({ id: 'upload.downloadTemplate' }, { type: 'RayJob' })}
        uploadtip={intl.formatMessage({ id: 'upload.sizeLimit' })}
        size={1}
        type={'yaml'}
        propsServiceUpload={getJobUpload}
      />
    </div>
  );
}
