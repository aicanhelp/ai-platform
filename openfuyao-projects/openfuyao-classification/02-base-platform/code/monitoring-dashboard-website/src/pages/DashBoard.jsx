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
import { containerRouterPrefix, ResponseCode, DEFAULT_CURRENT_PAGE, BusinessCode } from '@/common/constants';
import '@/styles/pages/dashboard.less';
import { Fragment, useCallback, useEffect, useState } from 'openinula';
import { requestResource, sortData, filterData } from '@/tools/utils';
import { getMonitorDashboardList, deleteDashboard, createEachDashboard, updateEachDashboard } from '@/api/monitorApi';
import { Radio, Button, Pagination, Popover, Table, message, Tooltip } from 'antd';
import { UnorderedListOutlined, AppstoreOutlined, MoreOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import EmptyHandle from '@/components/EmptyHandle';
import emptyDashboardImg from '@/assets/images/chart/emptyDashboard.png';
import CreateMonitorBoardModal from '@/components/CreateMonitorBoardModal';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { useHistory, Link } from 'inula-router';
import { useIntl } from 'inula-intl';

export default function DashBoard() {
  const intl = useIntl();
  const history = useHistory();
  const [dashBoardList, setDashBoardList] = useState([]);
  const [showDashBoardList, setShowDashBoardList] = useState([]);
  const [showListValue, setShowListValue] = useState('Card');
  const [dashBoardPage, setDashBoardPage] = useState(DEFAULT_CURRENT_PAGE);
  const [dashBoardPageSize, setDashBoardPageSize] = useState(20);
  const [loading, setLoading] = useState(true);
  const [monitorBoardModalOpen, setMonitorBoardModalOpen] = useState(false);
  const [modalType, setModalType] = useState('create');
  const [modalBoardName, setModalBoardName] = useState('');
  const [openUid, setOpenUid] = useState('');
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [isMonitorDelCheck, setIsMonitorDelCheck] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  const [handleDashboardID, setHandleDashboardID] = useState('');
  const [sortObj, setSortObj] = useState({}); // 排序
  const [originalData, setOriginalData] = useState([]); // 原始数据
  const [creatorFilter, setCreatorFilter] = useState([]);
  const [filterValue, setFilterValue] = useState(''); // 创建者筛选项

  const options = [
    {
      label: (
        <div>
          <AppstoreOutlined style={{ color: showListValue === 'Card' ? '#fff' : '#333333' }} />
        </div>
      ),
      value: 'Card',
    },
    {
      label: (
        <div>
          <UnorderedListOutlined style={{ color: showListValue === 'List' ? '#fff' : '#333333' }} />
        </div>
      ),
      value: 'List',
    },
  ];

  const handleChangeShowStyle = ({ target: { value } }) => {
    setShowListValue(value);
  };

  const getDashBoardData = useCallback(async () => {
    setLoading(true);
    const [error, res] = await requestResource(getMonitorDashboardList());
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'dashboard.fetchBoardFailed' }));
      setDashBoardList([]); // 异常
      setOriginalData([]);
      setLoading(false);
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        // 设置创建者列表
        let creatorList = [];
        let finallyList = [];
        if (res.data.data) {
          setDashBoardList([...res.data.data]);
          setOriginalData([...res.data.data]);
          res.data.data.map((item) => {
            creatorList.push(item.creator);
          });
          [...new Set(creatorList)].map((item) => {
            finallyList.push({ text: item, value: item });
          });
        } else {
          setDashBoardList([]);
          setOriginalData([]);
        }
        setCreatorFilter(finallyList);
        setLoading(false);
      }
    }
  }, []);

  const handleDashboardPageChange = (current, limit) => {
    setDashBoardPage(current);
    setDashBoardPageSize(limit);
  };

  const handleAddUUid = (record) => {
    setOpenUid(record.dashboardID);
    setHandleDashboardID(record.dashboardID);
  };

  const handleMonitorDashboardModify = (name) => {
    setOpenUid('');
    setModalBoardName(name);
    setModalType('edit');
    setMonitorBoardModalOpen(true);
  };

  const dashBoardColumns = [
    {
      title: intl.formatMessage({ id: 'common.name' }),
      key: 'name',
      sorter: true,
      sortOrder: sortObj.columnKey === 'name' ? sortObj.order : null,
      render: (_, record) => (
        <Link
          to={{
            pathname: `/dashboard/bigDataDashboard/detail/${record.dashboardID}/${encodeURIComponent(record.displayName)}`,
          }}
        >
          {record.displayName}
        </Link>
      ),
    },
    {
      title: intl.formatMessage({ id: 'dashboard.componentCount' }),
      key: 'monitorComponentNum',
      sorter: true,
      sortOrder: sortObj.columnKey === 'monitorComponentNum' ? sortObj.order : null,
      render: (_, record) => record.totalWidgets,
    },
    {
      title: intl.formatMessage({ id: 'dashboard.creator' }),
      key: 'creator',
      filters: creatorFilter,
      filterMultiple: false,
      sorter: true,
      sortOrder: sortObj.columnKey === 'creator' ? sortObj.order : null,
      render: (_, record) => record.creator,
    },
    {
      title: intl.formatMessage({ id: 'common.updateTime' }),
      key: 'lastUpdateTime',
      sorter: true,
      sortOrder: sortObj.columnKey === 'lastUpdateTime' ? sortObj.order : null,
      render: (_, record) => dayjs(record.lastUpdateTime).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: intl.formatMessage({ id: 'common.operations' }),
      key: 'handle',
      render: (_, record) => (
        <Popover
          placement="bottom"
          getPopupContainer={(triggerNode) => {
            if (triggerNode && triggerNode.parentNode) {
              return triggerNode.parentNode;
            }
            return document.body;
          }}
          content={
            <div className="pop_modal">
              <Button
                type="link"
                className="monitor_link_btn"
                onClick={() => handleMonitorDashboardModify(record.displayName)}
              >
                {intl.formatMessage({ id: 'common.edit' })}
              </Button>
              <Button
                type="link"
                className="monitor_link_btn"
                onClick={() => handleMonitorDashboardDel(record.displayName)}
              >
                {intl.formatMessage({ id: 'common.delete' })}
              </Button>
            </div>
          }
          trigger="click"
          arrow={false}
          open={openUid === record.dashboardID}
          onOpenChange={(newOpen) => (newOpen ? handleAddUUid(record) : setOpenUid(''))}
        >
          <MoreOutlined style={{ color: '#3f66f5' }} />
        </Popover>
      ),
    },
  ];

  const handleCreateMonitorBoard = () => {
    setOpenUid('');
    setModalType('create');
    setModalBoardName('');
    setMonitorBoardModalOpen(true);
  };

  const handleCancelMonitorBoard = () => {
    setMonitorBoardModalOpen(false);
  };

  const handleConfirmMonitorBoard = async (name) => {
    if (modalType === 'create') {
      const localUser = localStorage.getItem('user');
      const jsonData = {
        displayName: name,
        creator: localUser ? JSON.parse(localUser).name : 'admin',
      };
      const [error, res] = await requestResource(createEachDashboard(jsonData));
      if (error) {
        if (error.response.status === ResponseCode.Forbidden) {
          messageApi.error(intl.formatMessage({ id: 'monitor.operationFailedNoPermission' }));
        } else {
          if (error.response.data.code === BusinessCode.DuplicateName) {
            messageApi.error(intl.formatMessage({ id: 'monitor.nameDuplicate' }));
          } else {
            messageApi.error(intl.formatMessage({ id: 'monitor.createFailed' }));
          }
        }
      }
      if (res) {
        if (res.data.code === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'monitor.createSuccess' }));
          setMonitorBoardModalOpen(false);
          getDashBoardData();
        }
      }
    } else {
      const data = {
        displayName: name,
      };
      const [error, res] = await requestResource(updateEachDashboard(handleDashboardID, data));
      if (error) {
        if (error.response.status === ResponseCode.Forbidden) {
          messageApi.error(intl.formatMessage({ id: 'monitor.operationFailedNoPermission' }));
        } else {
          if (error.response.data.code === BusinessCode.DuplicateName) {
            messageApi.error(intl.formatMessage({ id: 'monitor.nameDuplicate' }));
          } else {
            messageApi.error(intl.formatMessage({ id: 'monitor.updateFailed' }));
          }
        }
      }
      if (res) {
        if (res.data.code === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'monitor.updateSuccess' }));
          setMonitorBoardModalOpen(false);
          getDashBoardData();
        }
      }
    }
  };

  const handleMonitorDashboardDel = (name) => {
    setModalBoardName(name);
    setOpenUid('');
    setDeleteModalOpen(true);
  };

  useEffect(() => {
    if (showListValue === 'Card') {
      setDashBoardPageSize(20);
      setDashBoardPage(DEFAULT_CURRENT_PAGE);
    } else {
      setDashBoardPageSize(10);
      setDashBoardPage(DEFAULT_CURRENT_PAGE);
    }
  }, [showListValue]);

  useEffect(() => {
    let list = [];
    list = dashBoardList.slice(
      (dashBoardPage - 1) * dashBoardPageSize,
      (dashBoardPage - 1) * dashBoardPageSize + dashBoardPageSize,
    );
    setShowDashBoardList([...list]);
  }, [dashBoardPage, dashBoardPageSize, JSON.stringify(dashBoardList)]);

  useEffect(() => {
    let newList = originalData;
    if (sortObj) {
      newList = sortData(originalData, sortObj);
    }
    if (filterValue) {
      newList = filterData(newList, filterValue);
    }
    setDashBoardList([...newList]);
  }, [sortObj, JSON.stringify(originalData), filterValue]);

  const handleDelModalCancel = () => {
    setDeleteModalOpen(false);
    setIsMonitorDelCheck(false);
  };

  const handleMonitorCheckFn = (e) => {
    setIsMonitorDelCheck(e.target.checked);
  };

  const handleDelMonitorConfirm = async (uuid) => {
    const [error, res] = await requestResource(deleteDashboard(uuid));
    if (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        messageApi.error(intl.formatMessage({ id: 'monitor.operationFailedNoPermission' }));
      } else {
        messageApi.error(intl.formatMessage({ id: 'monitor.deleteFailed' }));
      }
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'monitor.deleteSuccess' }));
        setIsMonitorDelCheck(false);
        setDeleteModalOpen(false);
        getDashBoardData();
      }
    }
  };

  const handleTableChange = (_pagination, filters, sorter, extra) => {
    if (extra.action === 'sort') {
      setSortObj({ columnKey: sorter.columnKey, order: sorter.order });
      setDashBoardPage(1);
    }
    if (extra.action === 'filter') {
      setFilterValue(filters.creator ? filters.creator[0] : '');
      setDashBoardPage(1);
    }
  };

  const handleSkip = (uuid, name) => {
    history.push({ pathname: `/dashboard/bigDataDashboard/detail/${uuid}/${encodeURIComponent(name)}` });
  };

  useEffect(() => {
    getDashBoardData();
  }, [getDashBoardData]);

  return (
    <div className="child_content">
      {contextHolder}
      <BreadCrumbCom
        className="create_bread"
        items={[{ title: intl.formatMessage({ id: 'dashboard.title' }), path: `/${containerRouterPrefix}/dashboard` }]}
      />
      <div className="dashboard_box">
        <div className="dashboard_header">
          <h2>
            {intl.formatMessage({ id: 'dashboard.monitorBoard' })}
            <span>({dashBoardList.length})</span>
          </h2>
          <div className="dashboard_tools">
            <Radio.Group
              className="toogle_btn"
              options={options}
              onChange={handleChangeShowStyle}
              value={showListValue}
              optionType="button"
              buttonStyle="solid"
            />
            <Button className="monitor_primary_btn" onClick={handleCreateMonitorBoard}>
              {intl.formatMessage({ id: 'dashboard.createBoard' })}
            </Button>
          </div>
        </div>
        {!loading ? (
          <div className="dashboard_content">
            {showDashBoardList.length ? (
              <Fragment>
                {showListValue === 'Card' && (
                  <Fragment>
                    <div className="dashboard_card">
                      {showDashBoardList.map((item) => (
                        <div className="card_item" onClick={() => handleSkip(item.dashboardID, item.displayName)}>
                          <div className="card_header">
                            <Tooltip title={item.displayName} placement="topLeft">
                              <h4>{item.displayName}</h4>
                            </Tooltip>
                            <Popover
                              placement="bottom"
                              content={
                                <div className="pop_modal">
                                  <Button
                                    type="link"
                                    className="monitor_link_btn"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleMonitorDashboardModify(item.displayName);
                                    }}
                                  >
                                    {intl.formatMessage({ id: 'common.edit' })}
                                  </Button>
                                  <Button
                                    type="link"
                                    className="monitor_link_btn"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleMonitorDashboardDel(item.displayName);
                                    }}
                                  >
                                    {intl.formatMessage({ id: 'common.delete' })}
                                  </Button>
                                </div>
                              }
                              trigger="click"
                              getPopupContainer={(triggerNode) => {
                                if (triggerNode && triggerNode.parentNode) {
                                  return triggerNode.parentNode;
                                }
                                return document.body;
                              }}
                              arrow={false}
                              open={openUid === item.dashboardID}
                              onOpenChange={(newOpen) => (newOpen ? handleAddUUid(item) : setOpenUid(''))}
                            >
                              <MoreOutlined style={{ color: '#3f66f5' }} onClick={(e) => e.stopPropagation()} />
                            </Popover>
                          </div>
                          <div className="card_body">
                            <div className="monitor_component">
                              <span>
                                {intl.formatMessage({ id: 'dashboard.componentCount' })}:
                                <span className="word">{item.totalWidgets}</span>
                              </span>
                            </div>
                            <div className="monitor_description">
                              <Tooltip
                                title={intl.formatMessage({ id: 'monitor.createdBy' }, { creator: item.creator })}
                              >
                                <div className="name">
                                  {intl.formatMessage({ id: 'monitor.createdBy' }, { creator: item.creator })}
                                </div>
                              </Tooltip>
                              <div className="time">
                                {intl.formatMessage(
                                  { id: 'monitor.updatedAt' },
                                  { time: dayjs(item.lastUpdateTime).format('YYYY-MM-DD HH:mm') },
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    <Pagination
                      className="page"
                      total={dashBoardList.length}
                      pageSize={dashBoardPageSize}
                      current={dashBoardPage}
                      showTotal={(total) => intl.formatMessage({ id: 'common.totalItems' }, { total })}
                      showSizeChanger
                      showQuickJumper
                      pageSizeOptions={[20, 50]}
                      onChange={handleDashboardPageChange}
                    />
                  </Fragment>
                )}
                {showListValue === 'List' && (
                  <div className="dashboard_table">
                    <Table
                      loading={loading}
                      dataSource={showDashBoardList}
                      columns={dashBoardColumns}
                      pagination={false}
                      onChange={handleTableChange}
                      locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                    />
                    <Pagination
                      className="page"
                      total={dashBoardList.length}
                      pageSize={dashBoardPageSize}
                      current={dashBoardPage}
                      showTotal={(total) => intl.formatMessage({ id: 'common.totalItems' }, { total })}
                      showSizeChanger
                      showQuickJumper
                      pageSizeOptions={[10, 20, 50]}
                      onChange={handleDashboardPageChange}
                    />
                  </div>
                )}
              </Fragment>
            ) : (
              <EmptyHandle imgSrc={emptyDashboardImg} description={intl.formatMessage({ id: 'dashboard.noBoard' })}>
                {' '}
                <Button onClick={handleCreateMonitorBoard}>
                  {intl.formatMessage({ id: 'dashboard.createBoard' })}
                </Button>
              </EmptyHandle>
            )}
          </div>
        ) : (
          <div className="dashboard_content" />
        )}
        <CreateMonitorBoardModal
          title={
            modalType === 'create'
              ? intl.formatMessage({ id: 'dashboard.createBoard' })
              : intl.formatMessage({ id: 'dashboard.editBoard' })
          }
          open={monitorBoardModalOpen}
          name={modalBoardName}
          cancelFn={handleCancelMonitorBoard}
          okFn={handleConfirmMonitorBoard}
        />
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'dashboard.deleteBoard' })}
        open={deleteModalOpen}
        cancelFn={handleDelModalCancel}
        content={[
          intl.formatMessage({ id: 'monitor.deleteBoardWarning' }),
          intl.formatMessage({ id: 'monitor.confirmDeleteBoard' }, { name: modalBoardName }),
        ]}
        isCheck={isMonitorDelCheck}
        showCheck={true}
        checkFn={handleMonitorCheckFn}
        confirmFn={() => handleDelMonitorConfirm(handleDashboardID)}
      />
    </div>
  );
}
